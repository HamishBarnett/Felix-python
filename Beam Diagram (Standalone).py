# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:45:59 2025

@author: Hamish
"""

# Install "ase" in command prompt:
# pip install ase

# --- Section A ---

from ase.io import read
from ase.visualize import view

# --- Load the CIF file (replace with actual path to your structure file)
structure_file = 'silicon_structure.cif'  # <-- your structure.cif file path
atoms = read(structure_file)

# --- Print key extracted information
print("Chemical formula:", atoms.get_chemical_formula())
print("Number of atoms:", len(atoms))
print("Lattice cell parameters (a, b, c):", atoms.cell.lengths())
print("Lattice angles (alpha, beta, gamma):", atoms.cell.angles())
print("Unit cell volume:", atoms.get_volume())
print("Atomic positions (fractional):")
for atom in atoms:
    print(f"  {atom.symbol} at fractional position {atom.position}")

# --- Optional: visualise the structure
# view(atoms)  # Uncomment this line to launch ASE GUI viewer

# --- Output: atoms is now a usable structure object

# --- Section B ---

import gemmi
import numpy as np

def parse_pets_dyn_cif(filepath):
    doc = gemmi.cif.read_file(filepath)
    block = doc.sole_block()

    # --- 1. Extract unit cell
    a = float(block.find_value('_cell_length_a'))
    b = float(block.find_value('_cell_length_b'))
    c = float(block.find_value('_cell_length_c'))
    alpha = float(block.find_value('_cell_angle_alpha'))
    beta = float(block.find_value('_cell_angle_beta'))
    gamma = float(block.find_value('_cell_angle_gamma'))

    # --- 2. Extract wavelength
    wavelength = float(block.find_value('_diffrn_radiation_wavelength'))

    # --- 3. Extract UB matrix
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

    # --- 4. Extract frame geometry from header
    details = block.find_value('_diffrn_measurement_details')
    lines = details.split('\n')
    n_frames = None
    frame_step = None
    for line in lines:
        if 'number of merged frames' in line:
            n_frames = int(line.split(':')[1].strip())
        elif 'step between frames' in line:
            frame_step = float(line.split(':')[1].strip())
    if n_frames is None or frame_step is None:
        raise ValueError("Could not find frame info in _diffrn_measurement_details")

    # --- 5. Parse reflection loop
    refl_loop = block.find_loop('_refln_index_h')
    h_index = refl_loop.tags.index('_refln_index_h')
    k_index = refl_loop.tags.index('_refln_index_k')
    l_index = refl_loop.tags.index('_refln_index_l')
    I_index = refl_loop.tags.index('_refln_intensity_meas')
    sigma_index = refl_loop.tags.index('_refln_intensity_sigma')
    zone_index = refl_loop.tags.index('_refln_zone_axis_id')  # optional

    reflections = []
    for row in refl_loop:
        h = int(row[h_index])
        k = int(row[k_index])
        l = int(row[l_index])
        intensity = float(row[I_index])
        sigma = float(row[sigma_index])
        zone_id = row[zone_index] if zone_index < len(row) else None
        reflections.append({'hkl': (h, k, l), 'intensity': intensity, 'sigma': sigma, 'zone_id': zone_id})

    # --- 6. Group reflections and find peak frame (simplified assumption: one value per reflection)
    # For more complex data (e.g., multiple intensities per frame), further parsing needed
    peak_frames = {}  # {(h,k,l): peak_frame}
    for refl in reflections:
        hkl = refl['hkl']
        if hkl not in peak_frames:
            peak_frames[hkl] = {'intensity': refl['intensity'], 'frame': int(refl['zone_id']) if refl['zone_id'] else 0}
        elif refl['intensity'] > peak_frames[hkl]['intensity']:
            peak_frames[hkl] = {'intensity': refl['intensity'], 'frame': int(refl['zone_id']) if refl['zone_id'] else 0}

    return {
        'unit_cell': (a, b, c, alpha, beta, gamma),
        'wavelength': wavelength,
        'UB_matrix': UB,
        'frame_step': frame_step,
        'n_frames': n_frames,
        'reflections': reflections,
        'peak_frames': peak_frames  # dict: {(h,k,l): {'frame': int, 'intensity': float}}
    }

# Example usage
filepath = 'Si_3_dyn.cif_pets'
data = parse_pets_dyn_cif(filepath)

# Quick checks
print("Unit cell:", data['unit_cell'])
print("Wavelength:", data['wavelength'])
print("UB matrix:\n", data['UB_matrix'])
print("Frame step:", data['frame_step'])
print("Total reflections parsed:", len(data['reflections']))
print("Peak frame for first few reflections:")
for i, (hkl, entry) in enumerate(data['peak_frames'].items()):
    print(f"  {hkl} → frame {entry['frame']}, intensity {entry['intensity']}")
    if i >= 5:
        break
