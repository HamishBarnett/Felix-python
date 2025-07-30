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

