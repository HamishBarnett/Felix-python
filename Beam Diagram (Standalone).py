# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:45:59 2025

@author: Hamish
"""

# Install "ase" in command prompt:
# pip install ase

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

# --- Optional: visualize the structure
# view(atoms)  # Uncomment this line to launch ASE GUI viewer

# --- Output: atoms is now a usable structure object
