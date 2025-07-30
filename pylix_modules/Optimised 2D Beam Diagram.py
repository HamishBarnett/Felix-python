# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:30:01 2025

@author: Hamish
"""

import numpy as np
import pylix  # Make sure pylix.py is in the same folder or installed as a module

# --- Step 1: Load .inp and .cif files ---

# Path to your input files
inp_file = "path/to/your/input_file.inp"  # Replace with actual path
cif_file = "path/to/your/structure.cif"   # Replace with actual path

# Load parameters from the .inp file
inp_data = pylix.read_inp_file(inp_file)

# Extract variables from the inp_data dictionary
debug = bool(inp_data['debug'])
cell_a = float(inp_data['cell_a'])
cell_b = float(inp_data['cell_b'])
cell_c = float(inp_data['cell_c'])
cell_alpha = float(inp_data['cell_alpha']) * np.pi / 180.0  # convert degrees to radians
cell_beta  = float(inp_data['cell_beta'])  * np.pi / 180.0
cell_gamma = float(inp_data['cell_gamma']) * np.pi / 180.0
space_group = str(inp_data['space_group'])
x_dir_c = np.array(inp_data['x_direction'])
z_dir_c = np.array(inp_data['incident_beam_direction'])
norm_dir_c = np.array(inp_data['normal_direction'])
n_frames = int(inp_data['n_frames'])
frame_angle = float(inp_data['frame_angle'])

# Read structure data from CIF file (atoms, symmetry, etc.)
cif_data = pylix.read_cif(cif_file)

# You now have all parameters needed for Step 2 (reference_frames)

# --- Step 2: Generate orientation matrices for each frame ---

# Call reference_frames to compute:
# t_m2o: microscope-to-orthogonal transformation for each frame
# t_c2o: crystal-to-orthogonal transformation matrix
# t_cr2or: reciprocal-crystal to reciprocal-orthogonal transformation matrix
t_m2o, t_c2o, t_cr2or = pylix.reference_frames(
    debug,
    cell_a, cell_b, cell_c,
    cell_alpha, cell_beta, cell_gamma,
    space_group,
    x_dir_c, z_dir_c, norm_dir_c,
    n_frames, frame_angle
)

# These will be used in Step 3 for generating reciprocal lattice vectors

# --- Step 3: Generate reciprocal lattice vectors (g_pool) ---

# You need to specify or infer the lattice type.
# Felix may include a lattice type assignment elsewhere.
# For Silicon (diamond cubic), the lattice type is typically 'F' (face-centered).
lattice_type = 'F'

# g_limit is read from the input file in Step 1
g_pool, g_pool_mag = pylix.hkl_make(t_cr2or, g_limit, lattice_type)

print(f"Generated {g_pool.shape[0]} reciprocal lattice vectors.")

# --- Step 4: Compute Bragg condition (excitation error) for each g and frame ---

# Required values from earlier steps:
# - convergence_angle (from input .inp)
# - image_radius (from input .inp)
# - t_m2o (orientation matrix from reference_frames)
# - g_pool and g_pool_mag (from Step 3)

# Compute magnitude of incident beam k0 (in 1/Angstroms)
h = 6.62607015e-34  # Planck's constant (J*s)
e = 1.602176634e-19  # elementary charge (C)
m0 = 9.10938356e-31  # electron mass (kg)
c = 2.99792458e8     # speed of light (m/s)
lambda_angstrom = 12.398 / np.sqrt((2 * accelerating_voltage_kv * 1e3 * e) / (m0 * c**2 * e) + accelerating_voltage_kv * 1e3 / 511000.0)
k_mag = 1.0 / lambda_angstrom  # inverse Angstroms

# Calculate excitation error (Bragg deviation) for all g vectors over all frames
s_g_frame = pylix.deviation_parameter(convergence_angle,
                                      image_radius,
                                      k_mag,
                                      g_pool,
                                      t_m2o)

print(f"Excitation error shape (frames × g vectors): {s_g_frame.shape}")

# --- Step 5: Select reflections that satisfy Bragg condition (strong beams) ---

# The ug_matrix is required but not used directly here (just a placeholder)
# The strong_beams function expects an "ug_matrix" — you can give np.zeros with proper shape
ug_matrix_dummy = np.zeros((g_pool.shape[0], 3, 3))  # Placeholder

# Define the threshold for how many frames a beam must satisfy Bragg to be considered "strong"
min_strong_beams = 20  # Can adjust as needed

# Use s_g_frame (from Step 4) and g_pool (from Step 3)
strong_beam_indices = pylix.strong_beams(s_g_frame,
                                         ug_matrix_dummy,
                                         min_strong_beams)

# Filter the g_pool to get strong beams
g_strong = g_pool[strong_beam_indices]
g_strong_mag = g_pool_mag[strong_beam_indices]

print(f"Selected {len(g_strong)} strong reflections that satisfy the Bragg condition.")

# --- Step 6: Plot the 2D beam diagram using selected reflections ---

# Use pylix’s built-in plotting utility
# This will show reciprocal vectors as yellow dots on a 2D plane

pylix.pool_plot(g_strong, g_strong_mag)
