# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:45:59 2025

@author: Hamish
"""

# --- Section A: Parse structure.cif and prepare crystal model ---

from ase.io import read
from ase.visualize import view
import numpy as np

# --- Load the CIF file
structure_file = 'silicon_structure.cif'  # Replace with your file path
atoms = read(structure_file)

# --- Extract unit cell and lattice info
a, b, c = atoms.cell.lengths()
alpha, beta, gamma = atoms.cell.angles()
cell_volume = atoms.get_volume()
chemical_formula = atoms.get_chemical_formula()

print("Chemical formula:", chemical_formula)
print("Number of atoms:", len(atoms))
print("Lattice parameters:")
print(f"  a = {a:.4f} Å, b = {b:.4f} Å, c = {c:.4f} Å")
print(f"  α = {alpha:.2f}°, β = {beta:.2f}°, γ = {gamma:.2f}°")
print(f"  Cell volume = {cell_volume:.3f} Å³")

# --- Extract fractional positions
frac_positions = atoms.get_scaled_positions()
symbols = atoms.get_chemical_symbols()

print("Atomic positions (fractional):")
for symbol, pos in zip(symbols, frac_positions):
    print(f"  {symbol}: ({pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f})")

# --- Optional visualisation
# view(atoms)  # Uncomment this line to open ASE viewer

# --- Output for later stages
# You will use:
# - `atoms.cell` for cell vectors
# - `atoms.get_scaled_positions()` for fractional atomic positions
# - `atoms.get_chemical_symbols()` for element types

# --- Section B: Parse PETS .dyn.cif (Experimental Data)

# -*- coding: utf-8 -*-
"""
Section B: Parse PETS .dyn.cif file
Extracts unit cell, wavelength, UB matrix, frame range, step size, and reflection profiles
"""

import gemmi
import numpy as np
from collections import defaultdict

def parse_pets_dyn_cif(filepath):
    doc = gemmi.cif.read_file(filepath)
    block = doc.sole_block()

    # --- 1. Unit cell parameters
    a = float(block.find_value('_cell_length_a'))
    b = float(block.find_value('_cell_length_b'))
    c = float(block.find_value('_cell_length_c'))
    alpha = float(block.find_value('_cell_angle_alpha'))
    beta = float(block.find_value('_cell_angle_beta'))
    gamma = float(block.find_value('_cell_angle_gamma'))
    cell_volume = float(block.find_value('_cell_volume'))

    # --- 2. Wavelength
    wavelength = float(block.find_value('_diffrn_radiation_wavelength'))

    # --- 3. UB matrix
    UB = np.array([
        [float(block.find_value('_diffrn_orient_matrix_UB_11')),
         float(block.find_value('_diffrn_orient_matrix_UB_12')),
         float(block.find_value('_diffrn_orient_matrix_UB_13'))],
        [float(block.find_value('_diffrn_orient_matrix_UB_21')),
         float(block.find_value('_diffrn_orient_matrix_UB_22')),
         float(block.find_value('_diffrn_orient_matrix_UB_23'))],
        [float(block.find_value('_diffrn_orient_matrix_UB_31')),
         float(block.find_value('_diffrn_orient_matrix_UB_32')),
         float(block.find_value('_diffrn_orient_matrix_UB_33'))],
    ])

    # --- 4. Frame geometry: step size and number of merged frames
    details = block.find_value('_diffrn_measurement_details')
    n_frames = None
    frame_step = None
    for line in details.splitlines():
        if 'number of merged frames' in line:
            n_frames = int(line.split(':')[-1].strip())
        elif 'step between frames' in line:
            frame_step = float(line.split(':')[-1].strip())
    if n_frames is None or frame_step is None:
        raise ValueError("Could not find 'number of merged frames' or 'step between frames' in metadata.")

    # --- 5. Reflection data: load full rocking curve for each reflection
    refl_loop = block.find_loop('_refln_index_h')
    h_index = refl_loop.tags.index('_refln_index_h')
    k_index = refl_loop.tags.index('_refln_index_k')
    l_index = refl_loop.tags.index('_refln_index_l')
    I_index = refl_loop.tags.index('_refln_intensity_meas')
    sigma_index = refl_loop.tags.index('_refln_intensity_sigma')
    frame_index = refl_loop.tags.index('_refln_zone_axis_id')  # used here for frame number

    # Dictionary: {(h, k, l): [(frame, intensity)]}
    reflection_profiles = defaultdict(list)

    for row in refl_loop:
        h = int(row[h_index])
        k = int(row[k_index])
        l = int(row[l_index])
        intensity = float(row[I_index])
        frame = int(row[frame_index])  # PETS stores frame as zone ID
        reflection_profiles[(h, k, l)].append((frame, intensity))

    # Sort frame profiles for each reflection
    for hkl in reflection_profiles:
        reflection_profiles[hkl] = sorted(reflection_profiles[hkl], key=lambda x: x[0])

    return {
        'unit_cell': (a, b, c, alpha, beta, gamma),
        'volume': cell_volume,
        'wavelength': wavelength,
        'UB_matrix': UB,
        'frame_step': frame_step,
        'n_frames': n_frames,
        'reflection_profiles': reflection_profiles  # dict: (h,k,l) → [(frame, intensity), ...]
    }

# --- Example usage ---
if __name__ == "__main__":
    file_path = 'Si_3_dyn.cif_pets'  # Replace with your path
    data = parse_pets_dyn_cif(file_path)

    print("\n--- Geometry ---")
    print("Unit cell (a, b, c, α, β, γ):", data['unit_cell'])
    print("Volume:", data['volume'])
    print("Wavelength:", data['wavelength'])
    print("UB matrix:\n", data['UB_matrix'])
    print("Frame step:", data['frame_step'])
    print("Number of merged frames:", data['n_frames'])

    print("\n--- Sample reflections ---")
    for i, (hkl, profile) in enumerate(data['reflection_profiles'].items()):
        print(f"{hkl}: {len(profile)} frames, peak I = {max(profile, key=lambda x: x[1])}")
        if i >= 4:
            break

# --- Section C ---

# -*- coding: utf-8 -*-
"""
Section C: Calculate centroid (sub-frame) positions of reflections
"""

def calculate_centroids(reflection_profiles):
    """
    Computes the intensity-weighted centroid frame for each reflection.

    Parameters
    ----------
    reflection_profiles : dict
        Dictionary of (h,k,l) → list of (frame, intensity) pairs

    Returns
    -------
    centroid_dict : dict
        Dictionary of (h,k,l) → centroid frame (float)
    """
    centroid_dict = {}

    for hkl, profile in reflection_profiles.items():
        # Filter out negative or zero intensities
        filtered = [(f, I) for f, I in profile if I > 0]
        if not filtered:
            continue  # skip if all intensities are zero or invalid

        frames, intensities = zip(*filtered)
        total_intensity = sum(intensities)
        if total_intensity == 0:
            continue  # avoid division by zero

        # Weighted average = Σ(frame_i * intensity_i) / Σ(intensity_i)
        centroid = sum(f * I for f, I in filtered) / total_intensity
        centroid_dict[hkl] = centroid

    return centroid_dict

# --- Example usage (assuming you've parsed the dyn.cif first with Section B) ---
if __name__ == "__main__":
    from pprint import pprint

    from pathlib import Path
    file_path = Path("Si_3_dyn.cif_pets")  # Your PETS file

    # Import Section B parser
    from section_b_parser import parse_pets_dyn_cif  # Adjust this line if needed

    parsed = parse_pets_dyn_cif(file_path)
    reflection_profiles = parsed["reflection_profiles"]

    centroid_dict = calculate_centroids(reflection_profiles)

    print("\n--- Sample Centroid Frames ---")
    for i, (hkl, centroid) in enumerate(centroid_dict.items()):
        print(f"{hkl}: centroid = {centroid:.3f}")
        if i >= 9:
            break


