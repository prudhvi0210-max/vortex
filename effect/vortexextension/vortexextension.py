import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp
import importlib.util
import os
from scipy.ndimage import map_coordinates

# Dynamic import utils (robust relative path)
utils_dir = os.path.dirname(__file__)
utils_path = os.path.join(utils_dir, '../utils.py')
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

def vortex_quantum(num_segments, strength):
    """
    Quantum: Generates randomized rotation angles via H + measure.
    Aux entangles for global coherence.
    Returns list of angles [0, 2π] per segment.
    """
    num_qubits = num_segments
    qc = QuantumCircuit(num_qubits + 1)
    anc = num_qubits

    # Prep superposition for random angles
    for i in range(num_qubits):
        qc.h(i)  # Superposition |+> for uniform random measure

    # Strength scales entanglement
    rotation = np.pi * strength
    for i in range(num_qubits):
        qc.cry(rotation, i, anc)  # If measured 1, rotate anc

    qc.cx(anc, 0)  # Entangle: Anc controls first for global twist

    # Measure X for angle (sin/cos → atan2)
    ops = [SparsePauliOp(Pauli("I" * (num_qubits - i) + "X" + "I" * i)) for i in range(num_qubits)]
    x_exps = utils.run_estimator(qc, ops)

    # Reconstruct angles: atan2(Y=0, X=exp) but randomized; scale to [0,2π]
    angles = [np.arccos(max(-1, min(1, x))) * 2 for x in x_exps]  # Approx uniform [0,π]*2
    print(f"Quantum angles: {angles}")
    return angles

def polar_swirl(region, center, angle, image, strength=1.0, radius=20):
    """
    Classical swirl: Remap pixels radially around center by angle.
    Fixed: Local max_r for stronger twist, full remap + region patch for smoothness.
    """
    if len(region) == 0:
        return image

    h, w, c = image.shape
    yy, xx = np.ogrid[:h, :w]

    # Polar coords relative to center
    r = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    theta = np.arctan2(yy - center[0], xx - center[1])

    # Local max_r: Use radius for stronger effect near stroke
    local_max_r = radius
    k = angle * strength * (10 / np.pi)  # Boost: ~3x stronger (pi-normalize)
    swirl_theta = theta + k * (r / local_max_r)

    # New positions
    new_y = r * np.sin(swirl_theta) + center[0]
    new_x = r * np.cos(swirl_theta) + center[1]

    # Remap full image per channel for smooth interp
    remapped = np.zeros_like(image)
    for ch in range(c):
        coords = np.stack([new_y, new_x])
        remapped[:, :, ch] = map_coordinates(image[:, :, ch], coords, order=1, mode='constant', cval=0)

    # Copy back only to region (localized)
    image[region[:, 0], region[:, 1]] = remapped[region[:, 0], region[:, 1]]

    return image

def run(params):
    """
    Quantum Vortex: Swirls colors around stroke path with quantum-random angles.
    Compatible with apply_effect.py (uses params dict).
    """
    image = params["stroke_input"]["image_rgba"]
    assert image.shape[-1] == 4, "Image must be RGBA"

    height, width = image.shape[0], image.shape[1]
    path = params["stroke_input"]["path"]
    clicks = params["stroke_input"]["clicks"]
    assert len(clicks) < 20, "Max 20 clicks"

    # Split into segments
    n_drops = len(clicks)
    split_paths = []
    click_indices = []
    c = 0
    for i, p in enumerate(path):
        if np.all(p == clicks[c]):
            click_indices.append(i)
            c += 1
            if c >= n_drops:
                break

    for idx, start in enumerate(click_indices):
        end = click_indices[idx + 1] if idx + 1 < len(click_indices) else len(path)
        interp_path = utils.interpolate_pixels(path[start:end])
        split_paths.append(interp_path)

    radius = params["user_input"]["Radius"]
    strength = params["user_input"]["Strength"]
    invert = params["user_input"]["Invert Luminosity"]  # Reuse: Invert swirl direction

    num_segments = len(split_paths)
    if num_segments == 0:
        return image

    # Quantum: Get random angles
    angles = vortex_quantum(num_segments, strength)
    if invert:
        angles = [-a for a in angles]  # Reverse swirl

    # Apply per segment: Center on path midpoint, swirl region
    for i, (lines, angle) in enumerate(zip(split_paths, angles)):
        region = utils.points_within_radius(lines, radius, border=(height, width))
        if len(region) == 0:
            continue

        center = np.mean(lines, axis=0).astype(int)  # Midpoint vortex eye
        image = polar_swirl(region, center, angle, image, strength, radius)  # Pass radius

    return image