import numpy as np
import colorsys
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp
import importlib.util
from scipy.stats import circmean

spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def entanglement(initial_angles, strength):
    """
    Entangle qubits representing strokes.
    
    Args:
        initial_angles (list of tuples): [(phi, theta), ...] for each stroke
        strength (float): entanglement strength (0 = low, 1 = max)
    
    Returns:
        final_angles: list of (phi, theta) after entanglement
    """
    num_qubits = len(initial_angles)

    # Ensure even number of qubits for entanglement
    if num_qubits % 2 != 0:
        initial_angles.append(initial_angles[-1])
        num_qubits += 1

    qc = QuantumCircuit(num_qubits)
    rotation = np.pi * strength  # entanglement rotation

    # 1️ Prepare qubits
    for i, (phi, theta) in enumerate(initial_angles):
        qc.ry(theta, i)
        qc.rz(phi, i)

    # 2️ Entangle first half with second half
    half = num_qubits // 2
    for i in range(half):
        qc.crx(rotation, control_qubit=i, target_qubit=i+half)

    # 3️ Define Pauli observables to measure X, Y, Z
    ops = [SparsePauliOp(Pauli('I'*(num_qubits-i-1) + p + 'I'*i))
           for p in ['X','Y','Z'] for i in range(num_qubits)]

    # 4️ Run estimator
    obs = utils.run_estimator(qc, ops)
    x_expectations = obs[:num_qubits]
    y_expectations = obs[num_qubits:2*num_qubits]
    z_expectations = obs[2*num_qubits:]

    # 5️ Convert back to (phi, theta)
    phi_expectations = [np.arctan2(y, x) % (2 * np.pi)
                        for x, y in zip(x_expectations, y_expectations)]
    theta_expectations = [np.arctan2(np.sqrt(x**2 + y**2), z)
                          for x, y, z in zip(x_expectations, y_expectations, z_expectations)]

    return list(zip(phi_expectations, theta_expectations))


def run(params):
    """
    Executes the entanglement effect pipeline.

    Args:
        params (dict): Contains image and stroke info.

    Returns:
        Image with color/brightness shifted by entangled strokes
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
        if np.all(p == clicks[c]):
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

    initial_angles = []
    pixels = []

    for lines in split_paths:
        region = utils.points_within_radius(lines, radius, border=(height, width))
        selection = image[region[:, 0], region[:, 1]]
        selection = selection.astype(np.float32) / 255.0
        selection_hls = utils.rgb_to_hls(selection)

        phi = circmean(2 * np.pi * selection_hls[..., 0])
        theta = np.pi * np.mean(selection_hls[..., 1], axis=0)

        initial_angles.append((phi, theta))
        pixels.append((region, selection_hls))

    strength = params["user_input"]["Strength"]
    assert 0 <= strength <= 1, "Strength must be 0..1"

    final_angles = entanglement(initial_angles, strength)  # no invert

    # After computing final_angles in your existing run() function
    
    print("=== Qubits (based on clicks) Analysis ===")
    print(f"Number of qubits (clicks): {len(initial_angles)}\n")

    for q_idx, (init, final) in enumerate(zip(initial_angles, final_angles)):
        phi_i, theta_i = init
        phi_f, theta_f = final
        print(f"Qubit {q_idx + 1}: Initial (phi,theta)=({phi_i:.3f}, {theta_i:.3f}), "
          f"Final (phi,theta)=({phi_f:.3f}, {theta_f:.3f})")

        print("========================================")
    # Apply the new angles to pixels
    for i, (region, selection_hls) in enumerate(pixels):
        new_phi, new_theta = final_angles[i]
        old_phi, old_theta = initial_angles[i]

        offset_h = (new_phi - old_phi) / (2 * np.pi)
        offset_l = (new_theta - old_theta) / np.pi

        selection_hls[..., 0] = (selection_hls[..., 0] + offset_h) % 1
        selection_hls[..., 1] += offset_l

        selection_hls = np.clip(selection_hls, 0, 1)
        selection_rgb = utils.hls_to_rgb(selection_hls)
        selection_rgb = (selection_rgb * 254).astype(np.uint8)

        image[region[:, 0], region[:, 1]] = selection_rgb

    return image
