# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:30:01 2025

@author: Hamish
"""

from pylix_modules import pylix_dicts as fu
from pylix_modules.pylix_class import Var
from pylix_modules.pylix import read_cif
import numpy as np


# Helper function to read felix.inp manually
def read_inp_file(filename):
    """
    Reads felix.inp and returns a dictionary of values.
    """
    inp_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, val = line.split('=')
                    key = key.strip()
                    val = val.strip()
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    inp_dict[key] = val
    return inp_dict


def load_felix_input(inp_filename="felix.inp", cif_filename="felix.cif"):
    """
    Loads Felix input file and CIF structure file into a configuration object (v).
    """
    import numpy as np
    import pylix_modules.pylix_dicts as fu
    import pylix_modules.pylix as px

    # Use Var class and load the .inp file
    v = fu.Var()
    v.load_inp(inp_filename)

    # Read the CIF structure
    cif_data = px.read_cif(cif_filename)

    # Fill unit cell parameters from CIF
    v.cell_a = cif_data["cell_length_a"][0]
    v.cell_b = cif_data["cell_length_b"][0]
    v.cell_c = cif_data["cell_length_c"][0]
    v.cell_alpha = np.radians(cif_data["cell_angle_alpha"][0])
    v.cell_beta = np.radians(cif_data["cell_angle_beta"][0])
    v.cell_gamma = np.radians(cif_data["cell_angle_gamma"][0])

    # Add symmetry operations and atomic positions
    v.symmetry_xyz = cif_data["space_group_symop_operation_xyz"]
    v.basis_atom_label = cif_data["atom_site_label"]
    v.basis_atom_name = cif_data["atom_site_type_symbol"]

    # Convert fractional coordinates into array
    x = np.array([t[0] for t in cif_data["atom_site_fract_x"]])
    y = np.array([t[0] for t in cif_data["atom_site_fract_y"]])
    z = np.array([t[0] for t in cif_data["atom_site_fract_z"]])
    v.basis_atom_position = np.vstack((x, y, z)).T

    # Thermal factors and occupancy
    v.basis_B_iso = np.array([t[0] for t in cif_data["atom_site_b_iso_or_equiv"]])
    v.basis_occupancy = np.array([t[0] for t in cif_data["atom_site_occupancy"]])

    return v



# Example usage
v = load_felix_input("felix.inp", "felix.cif")
print("Felix input and CIF data loaded")

# === Load and print summary ===
v = load_felix_input("felix.inp")
print("Felix input loaded.")
print(f"Unit cell: a={v['a']}, b={v['b']}, c={v['c']}")
print(f"Space group: {v['space_group']} (#{v['space_group_number']})")
