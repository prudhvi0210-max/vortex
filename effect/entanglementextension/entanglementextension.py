import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp
import importlib.util
from scipy.stats import circmean
from scipy.ndimage import map_coordinates  #for swirl remap

# Load utils
spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def polar_swirl(region, center, angle, image, strength=4.0, radius=20):
    """
    Swirl (warp) pixels around 'center' by 'angle', but write back only inside 'region'.
    Works on RGBA images.
    """
    if region is None or len(region) == 0:
        return image

    h, w, c = image.shape
    yy, xx = np.ogrid[:h, :w]

    # Polar coordinates relative to center
    r = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    theta = np.arctan2(yy - center[0], xx - center[1])

    local_max_r = max(1e-6, float(radius))

    # Swirl rotation grows with distance from center (scaled by strength)
    k = float(angle) * float(strength)
    swirl_theta = theta + k * (r / local_max_r)

    # Back to Cartesian sample locations
    new_y = r * np.sin(swirl_theta) + center[0]
    new_x = r * np.cos(swirl_theta) + center[1]

    # Remap with interpolation on the whole image, then paste only the region
    coords = np.stack([new_y, new_x])
    img_f = image.astype(np.float32)

    remapped = np.zeros_like(img_f)
    for ch in range(c):
        remapped[:, :, ch] = map_coordinates(
            img_f[:, :, ch],
            coords,
            order=1,           # bilinear
            mode="constant",   # outside -> 0
            cval=0.0
        )

    remapped = np.clip(remapped, 0, 255).astype(image.dtype)
    image[region[:, 0], region[:, 1]] = remapped[region[:, 0], region[:, 1]]
    return image


def entanglement(initial_angles, strength):
    """
    Entangle qubits representing strokes.

    Args:
        initial_angles: [(phi, theta), ...] per stroke segment
        strength: 0..1

    Returns:
        [(phi, theta), ...] after entanglement (same length as initial_angles, except
        odd count is duplicated internally but we only use the original count).
    """
    num_qubits = len(initial_angles)
    original_len = num_qubits

    # Ensure even number of qubits for entanglement (duplicate last)
    angles_work = list(initial_angles)
    if num_qubits % 2 != 0:
        angles_work.append(angles_work[-1])
        num_qubits += 1

    qc = QuantumCircuit(num_qubits)
    rotation = np.pi * strength

    # 1) Prepare qubits using (phi, theta)
    for i, (phi, theta) in enumerate(angles_work):
        qc.ry(theta, i)
        qc.rz(phi, i)

    # 2) Entangle first half with second half
    half = num_qubits // 2
    for i in range(half):
        qc.crx(rotation, control_qubit=i, target_qubit=i + half)

    # 3) Measure X, Y, Z expectations for each qubit
    ops = [
        SparsePauliOp(Pauli('I' * (num_qubits - i - 1) + p + 'I' * i))
        for p in ['X', 'Y', 'Z']
        for i in range(num_qubits)
    ]

    # 4) Run estimator
    obs = utils.run_estimator(qc, ops)
    x = obs[:num_qubits]
    y = obs[num_qubits:2 * num_qubits]
    z = obs[2 * num_qubits:]

    # 5) Convert Bloch components back into (phi, theta)
    phi_out = [np.arctan2(yy, xx) % (2 * np.pi) for xx, yy in zip(x, y)]
    theta_out = [np.arctan2(np.sqrt(xx ** 2 + yy ** 2), zz) for xx, yy, zz in zip(x, y, z)]

    final = list(zip(phi_out, theta_out))

    # Only return angles for the original (non-duplicated) segments
    return final[:original_len]


def run(params):
    """
    Entanglement + Visible Swirl + HLS (hue/lightness) shift.

    The key fix: we apply swirl first, then re-sample pixels from the *swirled* image
    before applying the HLS offsets, so the swirl doesn't get overwritten.
    """
    image = params["stroke_input"]["image_rgba"]
    assert image.shape[-1] == 4, "Image must be RGBA"

    height, width = image.shape[:2]
    path = params["stroke_input"]["path"]
    clicks = params["stroke_input"]["clicks"]
    assert len(clicks) < 20, "Max 20 clicks allowed"

    # Split path into subpaths starting at each click
    split_paths = []
    click_indices = []
    c = 0
    for i, p in enumerate(path):
        if c < len(clicks) and np.all(p == clicks[c]):
            click_indices.append(i)
            c += 1
            if c >= len(clicks):
                break

    for idx, start in enumerate(click_indices):
        end = click_indices[idx + 1] if idx + 1 < len(click_indices) else len(path)
        interp_path = utils.interpolate_pixels(path[start:end])
        split_paths.append(interp_path)

    radius = params["user_input"]["Radius"]
    assert radius > 0, "Radius must be positive"

    strength = params["user_input"]["Strength"]
    assert 0 <= strength <= 1, "Strength must be 0..1"

    # Gather per-segment regions and initial (phi, theta)
    initial_angles = []
    segments = []  # (lines, region)

    for lines in split_paths:
        region = utils.points_within_radius(lines, radius, border=(height, width))
        if region is None or len(region) == 0:
            continue

        selection = image[region[:, 0], region[:, 1]].astype(np.float32) / 255.0
        selection_hls = utils.rgb_to_hls(selection)

        # Hue is circular
        phi = circmean(2 * np.pi * selection_hls[..., 0])
        theta = np.pi * float(np.mean(selection_hls[..., 1]))

        initial_angles.append((phi, theta))
        segments.append((lines, region))

    if len(initial_angles) == 0:
        return image

    # Quantum entanglement -> new angles
    final_angles = entanglement(initial_angles, strength)

    # Debug print
    print("=== Qubits (based on segments) Analysis ===")
    print(f"Number of qubits (segments used): {len(initial_angles)}\n")
    for q_idx, (init, final) in enumerate(zip(initial_angles, final_angles)):
        phi_i, theta_i = init
        phi_f, theta_f = final
        print(
            f"Qubit {q_idx + 1}: Initial (phi,theta)=({phi_i:.3f}, {theta_i:.3f}), "
            f"Final (phi,theta)=({phi_f:.3f}, {theta_f:.3f})"
        )
    print("========================================")

    # Make swirl more visible
    swirl_gain = 8.0  # try 6..12 depending on your canvas size

    # Apply effect per segment
    for i, (lines, region) in enumerate(segments):
        new_phi, new_theta = final_angles[i]
        old_phi, old_theta = initial_angles[i]

        # 1) SWIRL FIRST (geometry) using phi shift
        swirl_angle = (new_phi - old_phi) * swirl_gain
        center = np.mean(lines, axis=0).astype(int)

        image = polar_swirl(
            region=region,
            center=center,
            angle=swirl_angle,
            image=image,
            strength=strength,
            radius=radius
        )

        # 2) Re-sample pixels AFTER swirl, then apply your HLS offsets
        selection = image[region[:, 0], region[:, 1]].astype(np.float32) / 255.0
        selection_hls = utils.rgb_to_hls(selection)

        offset_h = (new_phi - old_phi) / (2 * np.pi)
        offset_l = (new_theta - old_theta) / np.pi

        selection_hls[..., 0] = (selection_hls[..., 0] + offset_h) % 1.0
        selection_hls[..., 1] = selection_hls[..., 1] + offset_l
        selection_hls = np.clip(selection_hls, 0, 1)

        selection_rgb = utils.hls_to_rgb(selection_hls)
        selection_rgb = (selection_rgb * 255).astype(np.uint8)

        # Write RGB back; keep alpha as-is
        image[region[:, 0], region[:, 1], :3] = selection_rgb[:, :3]

    return image