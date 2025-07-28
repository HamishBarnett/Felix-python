# -*- coding: utf-8 -*-
# %% modules and subroutines
"""
Created August 2024

@author: R Beanland

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import matplotlib.colors as mcolors
import time
from scipy.constants import c, h, e, m_e, angstrom

import logging

logging.basicConfig(
    level=logging.INFO,  # Or logging.DEBUG for more detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# felix modules
from pylix_modules import pylix as px
from pylix_modules import simulate as sim
from pylix_modules import pylix_dicts as fu
from pylix_modules import pylix_class as pc

path = os.getcwd()
start = time.time()
latest_commit_id = px.get_git()
# outputs
print("-----------------------------------------------------------------")
print(f"felixrefine:  version {latest_commit_id[:8]}")
print("felixrefine:  https://github.com/WarwickMicroscopy/Felix-python")
print("-----------------------------------------------------------------")

# initialise class objects
v = pc.Var()  # working variables used in the simulation, see pylix_class
# initialise iteration count
v.iter_count = 0


# %% read felix.cif

# Read CIF file and extract dictionary
cif_dict = px.read_cif('felix.cif')
v.update_from_dict(cif_dict)

# Extract chemical formula
v.space_group = v.symmetry_space_group_name_h_m
if v.chemical_formula_structural:
    v.chemical_formula = v.chemical_formula_structural
elif v.chemical_formula_sum:
    v.chemical_formula = v.chemical_formula_sum
elif v.chemical_formula_iupac:
    v.chemical_formula = v.chemical_formula_iupac
else:
    v.chemical_formula = "Unknown"
print("Material:", v.chemical_formula)

# Retrieve space group and number
space_group_keys = [
    "space_group_symbol",
    "space_group_name_h_m_alt",
    "symmetry_space_group_name_h_m"
]
for key in space_group_keys:
    if key in cif_dict:
        v.space_group = cif_dict[key].replace(" ", "")
        break

# Ensure space group number
if "space_group_it_number" in cif_dict:
    v.space_group_number = int(cif_dict["space_group_it_number"][0])
elif v.space_group in fu.space_groups:
    v.space_group_number = fu.space_groups[v.space_group]
else:
    raise ValueError("Unable to determine space group number from .cif")

# Determine lattice type
v.lattice_type = v.space_group[0]

# === Cell parameters ===
v.cell_a = v.cell_length_a[0]
v.cell_b = v.cell_length_b[0]
v.cell_c = v.cell_length_c[0]
v.cell_alpha = v.cell_angle_alpha[0] * np.pi / 180.0  # convert to radians
v.cell_beta  = v.cell_angle_beta[0]  * np.pi / 180.0
v.cell_gamma = v.cell_angle_gamma[0] * np.pi / 180.0

n_basis = len(v.atom_site_label)

# === Symmetry operations ===
if "space_group_symop_operation_xyz" in cif_dict:
    sym_ops = cif_dict["space_group_symop_operation_xyz"]
elif "symmetry_equiv_pos_as_xyz" in cif_dict:
    sym_ops = cif_dict["symmetry_equiv_pos_as_xyz"]
else:
    raise ValueError("Symmetry operations not found in .cif")

v.symmetry_matrix, v.symmetry_vector = px.symop_convert(sym_ops)

# === Atom labels and names ===
v.basis_atom_label = v.atom_site_label

def clean_element_name(raw):
    name = ''.join(filter(str.isalpha, raw))
    return name.capitalize() if name else "X"

v.basis_atom_name = [clean_element_name(n) for n in v.atom_site_type_symbol]

# === Wyckoff symbols ===
v.basis_wyckoff = v.atom_site_wyckoff_symbol
# Optional: check they are valid characters
if v.basis_wyckoff is not None:
    for w in v.basis_wyckoff:
        if not w.isalpha():
            raise ValueError(f"Invalid Wyckoff symbol: {w}")

# === Basis positions (fractional coordinates) ===
v.basis_atom_position = np.column_stack([
    np.array([val[0] for val in v.atom_site_fract_x]),
    np.array([val[0] for val in v.atom_site_fract_y]),
    np.array([val[0] for val in v.atom_site_fract_z])
])

# === Debye-Waller factors (B_iso or U_iso) ===
if "atom_site_b_iso_or_equiv" in cif_dict:
    v.basis_B_iso = np.array([val[0] for val in v.atom_site_b_iso_or_equiv])
elif "atom_site_u_iso_or_equiv" in cif_dict:
    v.basis_B_iso = np.array([val[0] for val in cif_dict["atom_site_u_iso_or_equiv"]]) * 8 * np.pi**2
else:
    v.basis_B_iso = np.zeros(n_basis)

# === Occupancy ===
if v.atom_site_occupancy is not None:
    v.basis_occupancy = np.array([val[0] for val in v.atom_site_occupancy])
else:
    v.basis_occupancy = np.ones(n_basis)

# === Initialise basis_atom_delta for coordinate refinement ===
v.basis_atom_delta = np.zeros([n_basis, 3])


# %% read felix.inp and initialise

inp_dict = px.read_inp_file('felix.inp')
v.update_from_dict(inp_dict)

# --- Thickness array setup ---
if v.final_thickness > v.initial_thickness + v.delta_thickness:
    v.thickness = np.arange(v.initial_thickness, v.final_thickness, v.delta_thickness)
    v.n_thickness = len(v.thickness)
else:
    # redefine a single value to be a list [] so that it is iterable
    v.thickness = [v.initial_thickness]
    v.n_thickness = 1

# --- Convert important inputs to numpy arrays ---
v.incident_beam_direction = np.array(v.incident_beam_direction, dtype='float')
v.normal_direction = np.array(v.normal_direction, dtype='float')
v.x_direction = np.array(v.x_direction, dtype='float')
v.atomic_sites = np.array(v.atomic_sites, dtype='int')

# --- Absorption setup ---
if v.absorption_method != 1:
    v.absorption_per = 0.0

# --- Reciprocal space limits ---
v.frame_g_limit *= 2 * np.pi
v.g_limit *= 2 * np.pi

# --- Frame resolution: use more descriptive name and check validity ---
if v.frame_g_limit > 0:
    v.frame_resolution = (v.frame_size_x // 2) / v.frame_g_limit  # A^-1/pixel
else:
    raise ValueError("Invalid frame_g_limit; cannot calculate frame resolution.")

# === Output basic simulation info ===
print(f"Initial orientation: {v.incident_beam_direction.astype(int)}")
print(f"{v.n_frames} frames, each integrating over {v.frame_angle} degrees")
if v.frame_output == 1:
    print("Will output kinematic frame simulation")

if v.n_thickness == 1:
    print(f"Specimen thickness {v.initial_thickness / 10:.1f} nm")
else:
    thickness_str = ', '.join(map(lambda t: f"{t/10:.1f}", v.thickness))
    print(f"{v.n_thickness} thicknesses: {thickness_str} nm")

# --- Scattering factor method ---
scatter_method_names = {
    0: "Kirkland",
    1: "Lobato",
    2: "Peng",
    3: "Doyle & Turner"
}
method_name = scatter_method_names.get(v.scatter_factor_method)
if method_name:
    print(f"Using {method_name} scattering factors")
else:
    raise ValueError("No valid scattering factor method selected in felix.inp")

# --- Refinement mode setup ---
invalid_combination = False
if 'A' in v.refine_mode:
    print("Refining Structure Factors, A")
    # Structure factor refinement cannot be combined with others
    other_modes = [m for m in v.refine_mode if m not in ('A', 'S')]
    if other_modes:
        raise ValueError(f"Refinement mode 'A' cannot be combined with other modes: {other_modes}")
elif 'S' in v.refine_mode:
    print("Simulation only, S")
else:
    # All other refinement types
    if 'B' in v.refine_mode:
        print("Refining Atomic Coordinates, B")
        try:
            v.basis_atom_position = px.preferred_basis(
                v.space_group_number, v.basis_atom_position, v.basis_wyckoff)
        except Exception as err:
            logging.warning(f"Could not compute preferred basis: {err}")
            raise ValueError(f"Could not compute preferred basis: {err}")

    if 'C' in v.refine_mode:
        print("Refining Occupancies, C")
    if 'D' in v.refine_mode:
        print("Refining Isotropic Debye Waller Factors, D")
    if 'E' in v.refine_mode:
        raise ValueError("Refinement mode E (anisotropic Debye-Waller) not yet implemented")
    if len(v.atomic_sites) > n_basis:
        raise ValueError("Number of atomic sites to refine exceeds number of basis atoms")

# These refinement modes are not mutually exclusive
if 'F' in v.refine_mode:
    print("Refining Lattice Parameters, F")
if 'G' in v.refine_mode:
    print("Refining Lattice Angles, G")
if 'H' in v.refine_mode:
    print("Refining Convergence Angle, H")
if 'I' in v.refine_mode:
    print("Refining Accelerating Voltage, I")



# %% read felix.hkl
v.input_hkls, v.i_obs, v.sigma_obs = px.read_hkl_file("felix.hkl")

# Ensure 000 is not in the list — remove if present
zero_hkl = [0, 0, 0]
if any(np.array_equal(hkl, zero_hkl) for hkl in v.input_hkls):
    print("Warning: (0,0,0) reflection found in felix.hkl and will be ignored.")
    idx_to_keep = [not np.array_equal(hkl, zero_hkl) for hkl in v.input_hkls]
    v.input_hkls = [hkl for hkl, keep in zip(v.input_hkls, idx_to_keep) if keep]
    v.i_obs = [i for i, keep in zip(v.i_obs, idx_to_keep) if keep]
    v.sigma_obs = [s for s, keep in zip(v.sigma_obs, idx_to_keep) if keep]

v.n_out = len(v.input_hkls) + 1  # Account for simulated 000


# %% Setup kV and unit cell

# Electron velocity in metres per second
electron_velocity = (c * np.sqrt(1.0 - ((m_e * c**2) /
                     (e * v.accelerating_voltage_kv*1000.0 +
                      m_e * c**2))**2))
# Electron wavelength in Angstroms
electron_wavelength = h / (
    np.sqrt(2.0 * m_e * e * v.accelerating_voltage_kv*1000.0) *
    np.sqrt(1.0 + (e * v.accelerating_voltage_kv*1000.0) /
            (2.0 * m_e * c**2))) / angstrom
# Wavevector magnitude k
electron_wave_vector_magnitude = 2.0 * np.pi / electron_wavelength
# Relativistic correction
relativistic_correction = 1.0 / np.sqrt(1.0 - (electron_velocity / c)**2)
# Conversion from scattering factor to volts
cell_volume = v.cell_a*v.cell_b*v.cell_c*np.sqrt(1.0-np.cos(v.cell_alpha)**2
              - np.cos(v.cell_beta)**2 - np.cos(v.cell_gamma)**2
              +2.0*np.cos(v.cell_alpha)*np.cos(v.cell_beta)*np.cos(v.cell_gamma))
scatt_fac_to_volts = ((h**2) /
                      (2.0*np.pi * m_e * e * cell_volume * (angstrom**2)))

# ===============================================
# fill the unit cell and get mean inner potential
# when iterating we only do it if necessary?
# if v.iter_count == 0 or v.current_variable_type < 6:
    
if v.iter_count == 0 or v.current_variable_type < 6:
    atom_position, atom_label, atom_name, B_iso, occupancy = \
        px.unique_atom_positions(
            v.symmetry_matrix, v.symmetry_vector, v.basis_atom_label,
            v.basis_atom_name,
            v.basis_atom_position, v.basis_B_iso, v.basis_occupancy)


# Generate atomic numbers based on the elemental symbols
atomic_number = np.array([fu.atomic_number_map[na] for na in atom_name])

n_atoms = len(atom_label)
print("  There are "+str(n_atoms)+" atoms in the unit cell")
# plot
if v.plot:
    atom_cvals = mcolors.Normalize(vmin=1, vmax=103)
    atom_cmap = plt.cm.viridis
    atom_colours = atom_cmap(atom_cvals(atomic_number))
    border_cvals = mcolors.Normalize(vmin=0, vmax=1)
    border_cmap = plt.cm.plasma
    border_colours = border_cmap(border_cvals(atom_position[:, 2]))
    bb = 5
    fig, ax = plt.subplots(figsize=(bb, bb))
    plt.scatter(atom_position[:, 0], atom_position[:, 1],
                color=atom_colours, edgecolor=border_colours,
                linewidth=1, s=100)
    plt.xlim(left=0.0, right=1.0)
    plt.ylim(bottom=0.0, top=1.0)
    ax.set_axis_off
    plt.grid(True)

# mean inner potential as the sum of scattering factors at g=0
# multiplied by h^2/(2pi*m0*e*CellVolume)
mip = 0.0
for i in range(n_atoms):  # get the scattering factor
    if v.scatter_factor_method == 0:
        mip += px.f_kirkland(atomic_number[i], 0.0)
    elif v.scatter_factor_method == 1:
        mip += px.f_lobato(atomic_number[i], 0.0)
    elif v.scatter_factor_method == 2:
        mip += px.f_peng(atomic_number[i], 0.0)
    elif v.scatter_factor_method == 3:
        mip += px.f_doyle_turner(atomic_number[i], 0.0)
    else:
        raise ValueError("No scattering factors chosen in felix.inp")
mip = mip.item()*scatt_fac_to_volts  # NB convert array to float
print(f"  Mean inner potential = {mip:.1f} Volts")

# Wave vector magnitude in crystal, in reciprocal Angstroms
# high-energy approximation (not HOLZ compatible)
# K^2=k^2+U0
big_k_mag = np.sqrt(electron_wave_vector_magnitude**2+mip)

# %% set up reference frames
# We work in a fixed orthogonal crystal frame _o
# t_m2o = transformation microscope to orthogonal
# t_c2o = transformation crystal to orthogonal
# t_cr2o = transformation crystal to orthogonal, reciprocal space
t_m2o, t_c2o, t_cr2or = \
    px.reference_frames(v.debug, v.cell_a, v.cell_b, v.cell_c,
                        v.cell_alpha, v.cell_beta, v.cell_gamma,
                        v.space_group, v.x_direction,
                        v.incident_beam_direction, v.normal_direction,
                        v.n_frames, v.frame_angle)


# %% Observable reflexions, their structure factor and deviation parameter

# kinematic beam pool
print(f"Experimental resolution limit {0.5*v.frame_g_limit/np.pi:.3} reciprocal Angstroms")

# Observable reflections are found within frame_g_limit
# Adjust reciprocal space resolution to account for unit cell angle distortions
# This ensures that even in low-symmetry cells (e.g., triclinic), 
# all relevant g-vectors are still included
min_sine = np.min([np.sin(v.cell_alpha),
                   np.sin(v.cell_beta), np.sin(v.cell_gamma)])
# Prevent division by zero in pathological cases (e.g. 0-degree angles)
expand_factor = max(min_sine, 1e-4)
g_limit = int(v.frame_g_limit / expand_factor)

hkl_pool, g_pool, g_mag = px.hkl_make(t_cr2or, g_limit, v.lattice_type)
n_g = len(g_mag)

print(f"giving {len(g_mag)} reflexions")  # n_g
px.pool_plot(g_pool, g_mag)

# structure factor Fg for all reflexions in g_pool
F_g = px.Fg(g_pool, g_mag, atom_position, atomic_number, occupancy,
            v.scatter_factor_method, v.absorption_method,
            v.absorption_per, electron_velocity, B_iso)

I_kin = (F_g * np.conj(F_g)).real

# incident wave vector lies along Z in the microscope frame
# so we can get if for all frames from the last column of the
# transformation matrix t_m20. size [n_frames, 3]
big_k = big_k_mag * t_m2o[:, :, 2]

# Deviation parameter sg for all frames and g-vectors, size [n_frames, n_g]
sg = px.sg(big_k, g_pool)


# %% kinematic simulation

# we assume kinematic rocking curves are Gaussian in shape
# with FWHM rc_fwhm, in reciprocal angstroms, when plotted against sg

# === Rocking curve FWHM and deviation limit ===
# Allow user to specify via felix.inp, else use defaults
rc_fwhm = getattr(v, 'rc_fwhm', 0.04)  # reciprocal Ångstroms
ds = getattr(v, 'ds_limit', 0.15)
# The sg limit ds is used to determine whether a reflexion is in a frame
# note that sg of 0.1 is a long way from the Bragg condition at 200kV
# a value of 0.05 seems about right to match to experiment

# Ensure FWHM is less than the deviation limit (avoids unphysical RC)
if rc_fwhm >= ds:
    raise ValueError(f"rc_fwhm ({rc_fwhm}) must be smaller than ds_limit ({ds})")

# Gaussian coefficient for rocking curve
cc = (rc_fwhm / (2**1.5 * np.log(2)))**2

# find all reflexions in all frames in the sg limit
mask = np.abs(sg) < ds  # boolean, size [n_frames, n_g]

# Now we make lists of numpy arrays, using this mask
# The first list g_where, length n_frames, gives the indices of the reflexions
# in the numpy arrays sg, hkl_pool, g_pool, I_kin
g_where = [np.where(mask[i])[0] for i in range(mask.shape[0])]
sg_frame = [sg[j, i] for j, i in enumerate(g_where)]  # sg values
hkl_frame = [hkl_pool[i] for i in g_where]  # Miller indices
g_frame_o = [g_pool[i] for i in g_where]  # g-vectors (orthogonal frame)
direct = np.array([[0, 0, 0]])  # Shape (1,3) for proper stacking
gD_frame_o = [np.vstack((direct, g)) for g in g_frame_o]  # g-vectors and 000
# expanded g-pool for each frame that will make the F_g matrix
g_pool_dyn = [(g[:, np.newaxis, :] - g[np.newaxis, :, :]) for g in gD_frame_o]
# kinematic intensity is constant for each reflexion
I_kin_frame = [I_kin[i] for i in g_where]

# set intensity of the strongest reflexion to unity (100%)
I_100 = np.max(np.concatenate(I_kin_frame))
# calculated intensity applies the rocking curve profile
I_calc_frame = [np.array(I_k) *
                np.exp(-np.abs(np.array(sg_))*np.abs(np.array(sg_))/cc)/I_100
                for I_k, sg_ in zip(I_kin_frame, sg_frame)]

print("Kinematic simulation complete")

log_scale = True
if v.frame_output == 1:
    px.frame_plot(t_m2o, g_frame_o, I_calc_frame, v.n_frames, v.frame_size_x,
                  v.frame_size_y, v.frame_resolution, log_scale)


# %% dynamical simulation

I_dyn_frame = []
for i in range(v.n_frames):
    # g pool for this frame
    g_pool_f = g_pool_dyn[i].reshape(-1, 3)
    g_mag_f = np.linalg.norm(g_pool_f, axis=1) + 1.0e-12  # avoid division by 0
    sg_f = np.concatenate(([0], sg_frame[i]))  # sg's for first column of F_g matrix
    ng_f = len(sg_f)

    # Show the beam pool? – perhaps comment out
    if v.plot and v.debug > 1:
        px.pool_plot(g_pool_f, g_mag_f)

    # structure factor Fg_matrix for this frame's g_pool
    # diagonal values depend on sg
    Fg_matrix = px.Fg(
        g_pool_f, g_mag_f, atom_position, atomic_number,
        occupancy, v.scatter_factor_method, v.absorption_method,
        v.absorption_per, electron_velocity, B_iso
    ).reshape(ng_f, ng_f)

    # Conversion factor from F_g to U_g
    Fg_to_Ug = relativistic_correction / cell_volume
    ug_matrix = Fg_to_Ug * Fg_matrix
    # Spence's (1990) 'Structure matrix'
    # off-diagonal elements are Ug/2K, diagonal elements are Sg
    ug_sg_matrix = 2.0 * np.pi * ug_matrix / big_k_mag
    np.fill_diagonal(ug_sg_matrix, sg_f)

    # Print top-left of structure matrix? – perhaps comment out
    if v.debug:
        np.set_printoptions(precision=3, suppress=True)
        print(f"Structure matrix (frame {i+1}/{v.n_frames}):")
        print(ug_sg_matrix[:min(5, ng_f), :min(5, ng_f)])

    # Solve Bloch wave equation
    wave_functions = px.wave_functions(ug_sg_matrix, v.thickness, v.debug)

    # Dynamical intensities, discarding 000 so we have the same output length
    I_dyn_frame.append(np.squeeze(np.abs(wave_functions)**2)[1:])

print("Dynamic simulation complete")

# Plot dynamical frame output if requested
log_scale = True
if v.frame_output == 1:
    px.frame_plot(
        t_m2o, g_frame_o, I_dyn_frame,
        v.n_frames, v.frame_size_x, v.frame_size_y,
        v.frame_resolution, log_scale
    )


# %% Bragg position and rocking curves

bragg = np.zeros_like(g_mag)
for g in np.unique(np.concatenate(g_where)):
    # Extract intensity for this reflection g across frames
    I_kin_rc = np.squeeze([
        I_f[idx_list == g]
        for I_f, idx_list in zip(I_calc_frame, g_where)
        for idx in range(len(idx_list)) if idx_list[idx] == g
    ])
    I_dyn_rc = np.squeeze([
        I_f[idx_list == g]
        for I_f, idx_list in zip(I_dyn_frame, g_where)
        for idx in range(len(idx_list)) if idx_list[idx] == g
    ])
    sg_rc = np.squeeze([
        s_f[idx_list == g]
        for s_f, idx_list in zip(sg_frame, g_where)
        for idx in range(len(idx_list)) if idx_list[idx] == g
    ])
    f_rc = [i for i, indices in enumerate(g_where) if g in indices]

    # Estimate sub-frame Bragg position where sg crosses zero (Bragg condition)
    if sg_rc.size < 2 or f_rc is None:
        continue  # Not enough points to estimate
    if np.min(sg_rc) >= 0 or np.max(sg_rc) <= 0:
        continue  # sg does not cross zero (no Bragg peak)

    # Find last negative and first positive sg — use linear interpolation
    neg = np.where(sg_rc < 0)[0][-1]   # Last occurrence of a -ve value
    pos = np.where(sg_rc > 0)[0][0]    # First occurrence of a +ve value
    sg_neg = sg_rc[neg]
    sg_pos = sg_rc[pos]
    frame_neg = f_rc[neg]
    frame_pos = f_rc[pos]

    # Linear interpolation to find Bragg position (sg = 0)
    slope = sg_pos - sg_neg
    if abs(slope) > 1e-12:  # Avoid division by 0
        bragg[g] = frame_neg + abs(sg_neg) / abs(slope)
    else:
        # Compute ds/df
        # dsdf = sg_rc[pos] - sg_rc[neg]
        # Compute bragg position
        bragg[g] = 0.5 * (frame_neg + frame_pos)  # Fallback: midpoint

    # Plot rocking curves? – perhaps comment out
    if v.frame_output == 1:
        px.rock_plot(hkl_pool, g, sg_rc, f_rc, I_kin_rc, I_dyn_rc)


# %% Set up refinement

# n_variables calculated depending upon Ug and non-Ug refinement
# Ug refinement is a special case, cannot do any other refinement alongside
# We count the independent variables:
# v.refined_variable = variable to be refined
# v.refined_variable_type = what kind of variable, as follows

# 0 = Ug Amplitude
# 1 = Ug Phase
# 2 = Coordinate Refinement (*** PARTIALLY IMPLEMENTED *** not all space groups)
# 3 = Occupancy Refinement
# 4 = Isotropic Debye-Waller, B_iso
# 5 = Anisotropic Debye-Waller, B_aniso (*** NOT YET IMPLEMENTED ***)
# 61,62,63 = Lattice Parameters (*** PARTIALLY IMPLEMENTED *** not rhombohedral)
# 7 = Unit Cell Angles (*** NOT YET IMPLEMENTED ***)
# 8 = Convergence Angle
# 9 = Accelerating Voltage, accelerating_voltage_kv (*** NOT YET IMPLEMENTED ***)

v.refined_variable = []
v.refined_variable_type = []
v.atom_refine_flag = []

if 'S' not in v.refine_mode:
    v.n_variables = 0
    # count refinement variables

    # --- Coordinate Refinement (2) --- (*** PARTIALLY IMPLEMENTED *** not all space groups)
    if 'B' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            # the [3, 3] matrix 'moves' returned by atom_move gives the
            # allowed movements for an atom (depending on its Wyckoff
            # symbol and space group) as row vectors with magnitude 1.
            # ***NB NOT ALL SPACE GROUPS IMPLEMENTED ***
            moves = px.atom_move(v.space_group_number, v.basis_wyckoff[i])
            degrees_of_freedom = int(np.sum(moves**2))
            if degrees_of_freedom == 0:
                raise ValueError(f"Atom coordinate refinement not supported for space group {v.space_group_number}")
            for j in range(degrees_of_freedom):
                r_dot_v = np.dot(v.basis_atom_position[v.atomic_sites[i]], moves[j, :])
                v.refined_variable.append(r_dot_v)
                v.refined_variable_type.append(2)
                v.atom_refine_flag.append(v.atomic_sites[i])

    # --- Occupancy Refinement (3) ---
    if 'C' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            v.refined_variable.append(v.basis_occupancy[v.atomic_sites[i]])
            v.refined_variable_type.append(3)
            v.atom_refine_flag.append(v.atomic_sites[i])

    # --- Isotropic Debye-Waller (4) ---
    if 'D' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            v.refined_variable.append(v.basis_B_iso[v.atomic_sites[i]])
            v.refined_variable_type.append(4)
            v.atom_refine_flag.append(v.atomic_sites[i])

    # --- Anisotropic Debye-Waller (5) --- (*** NOT YET IMPLEMENTED ***)
    if 'E' in v.refine_mode:
        raise NotImplementedError("Refinement mode 'E' (Anisotropic Debye-Waller) is not supported in this version.")

    # --- Lattice Parameters (6) --- (*** PARTIALLY IMPLEMENTED *** not rhombohedral)
    if 'F' in v.refine_mode:
        # variable_type first digit=6 indicates lattice parameter
        # second digit=1,2,3 indicates a,b,c
        # This section needs work to include rhombohedral cells and
        # non-standard settings!!!
        v.refined_variable.append(v.cell_a) # is in all lattice types
        v.refined_variable_type.append(61)
        v.atom_refine_flag.append(-1) # -1 indicates not an atom

        if v.space_group_number < 75: # Triclinic, monoclinic, orthorhombic
            v.refined_variable += [v.cell_b, v.cell_c]
            v.refined_variable_type += [62, 63]
            v.atom_refine_flag += [-1, -1]

        elif 142 < v.space_group_number < 168: # Rhombohedral
            # Need to work out R- vs H- settings!!!
            raise NotImplementedError("Refinement of rhombohedral cell dimensions (R vs H setting) is not supported.")

        elif (74 < v.space_group_number < 143) or (v.space_group_number >= 168): # Hexagonal or Tetragonal
            v.refined_variable.append(v.cell_c)
            v.refined_variable_type.append(63)
            v.atom_refine_flag.append(-1)

    # --- Unit Cell Angles (7) --- (*** NOT YET IMPLEMENTED ***)
    if 'G' in v.refine_mode:
        raise NotImplementedError("Refinement of unit cell angles (G) is not implemented.")

    # --- Convergence Angle (8) ---
    if 'H' in v.refine_mode:
        v.refined_variable.append(v.convergence_angle)
        v.refined_variable_type.append(8)
        v.atom_refine_flag.append(-1)
        print(f"Starting convergence angle {v.convergence_angle} Å^-1")

    # --- Accelerating Voltage (9) --- (*** NOT YET IMPLEMENTED ***)
    if 'I' in v.refine_mode:
        v.refined_variable.append(v.accelerating_voltage_kv)
        v.refined_variable_type.append(9)
        v.atom_refine_flag.append(-1)

    # --- Final setup --- (Total number of independent variables)
    v.n_variables = len(v.refined_variable)
    if v.n_variables == 0:
        raise ValueError("No refinement variables specified. Check refine_mode in felix.inp.")
    elif v.n_variables == 1:
        print("Only one independent variable.")
    else:
        print(f"Number of independent variables = {v.n_variables}")

    v.refined_variable = np.array(v.refined_variable)
    independent_delta = np.zeros(v.n_variables)
    v.refined_variable_type = np.array(v.refined_variable_type)
    v.refined_variable_atom = np.array(v.atom_refine_flag[:v.n_variables])


# %% Set up Ug refinement
# if 'A' in refine_mode:  # Ug refinement
#     print("Refining Structure Factors, A")
#     # needs error check for any other refinement
#     # raise ValueError("Structure factor refinement incompatible
#     # with anything else")
#     # we refine magnitude and phase for each Ug.  However for space groups
#     # with a centre of symmetry phases are fixed at 0 or pi, so only
#     # amplitude is refined (1 independent variable per Ug)
#     # Identify the 92 centrosymmetric space groups
#     centrosymmetric = [2, 10, 11, 12, 13, 14, 15, 47, 48, 49, 50, 51, 52,
#                        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
#                        66, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85,
#                        86, 87, 88, 123, 124, 125, 126, 127, 128, 129, 130,
#                        131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
#                        141, 142, 147, 148, 162, 163, 164, 165, 166, 167,
#                        175, 176, 191, 192, 193, 194, 200, 201, 202,
#                        203, 204, 205, 206, 221, 222, 223, 224, 225, 226,
#                        227, 228, 229, 230]
#     if space_group_number in centrosymmetric:
#         vars_per_ug = 1
#     else:
#         vars_per_ug = 2

#     # set up Ug refinement
#     # equivalent g's identified by abs(h)+abs(k)+abs(l)+a*h^2+b*k^2+c*l^2
#     g_eqv = (10000*(np.sum(np.abs(g_matrix), axis=2) +
#                     g_magnitude**2)).astype(int)
#     # we keep track of individual Ug's in a matrix ug_eqv
#     ug_eqv = np.zeros([n_hkl, n_hkl], dtype=int)
#     # bit of a hack here - can skip Ug's in refinement using ug_offset, but
#     # should really be an input in felix.inp if it's going to be used
#     ug_offset = 0
#     # Ug number (position in ug_matrix)
#     i_ug = 1 + ug_offset
#     # the first column of the Ug matrix has g-vectors in ascending order
#     # we work through this list until we have identified the required no.
#     # of Ugs. The matrix ug_eqv identifies equivalent Ug's with an integer
#     # whose sign is that of the imaginary part of Ug. (may need to take
#     # care of floating point residuals where im(Ug) is nominally zero???)
#     j = 1  # number of Ug's processed
#     while j < no_of_ugs+1:
#         if ug_eqv[i_ug, 0] != 0:  # already in the list, skip it
#             i_ug += 1
#             continue
#         g_id = abs(g_eqv[i_ug, 0])
#         # update relevant locations in ug_eqv
#         ug_eqv[np.abs(g_eqv) == g_id] = j*np.sign(
#             np.imag(ug_matrix[i_ug, 0]))
#         # amplitude is type 1, always a variable
#         variable.append(ug_matrix[i_ug, 0])
#         variable_type.append(0)
#         if vars_per_ug == 2:  # we also adjust phase
#             variable.append(ug_matrix[i_ug, 0])
#             variable_type.append(1)
#         j += 1


# %% Baseline simulation
print("-------------------------------")
print("Baseline simulation:")

try:
    # Run simulation using full input configuration
    setup, bwc = sim.simulate(v)
    print("Baseline simulation completed successfully.")
except Exception as err:
    print(f"Error during baseline simulation: {err}")
    raise

# %% Read in experimental images
if 'S' not in v.refine_mode:
    v.lacbed_expt = np.zeros([2*v.image_radius, 2*v.image_radius, v.n_out])
    x_str = str(2*v.image_radius)
    
    # Optional override from felix.inp
    dm3_folder = getattr(v, 'dm3_path', None)

    if dm3_folder is None:
        for dirpath, dirnames, filenames in os.walk(path):
            for dirname in dirnames:
                if 'dm3' in dirname.lower() and x_str in dirname:
                    dm3_folder = os.path.join(dirpath, dirname)
                    break

    if dm3_folder is None or not os.path.exists(dm3_folder):
        raise FileNotFoundError("No suitable .dm3 folder found. "
                                "Make sure the folder name contains 'dm3' and "
                                f"{x_str}, or set 'dm3_path' in felix.inp.")
    
    dm3_files = [file for file in os.listdir(dm3_folder)
                 if file.lower().endswith('.dm3')]

    n_expt = v.n_out
    found_count = 0
    for i in range(v.n_out):
        g_string = px.hkl_string(v.hkl[v.g_output[i]])
        found = False
        for file_name in dm3_files:
            if g_string in file_name:
                file_path = os.path.join(dm3_folder, file_name)
                v.lacbed_expt[:, :, i] = px.read_dm3(file_path,
                                                     2*v.image_radius,
                                                     v.debug)
                found = True
                found_count += 1
                break
        if not found:
            print(f"Warning: {g_string} not found in .dm3 folder")

    print(f"Loaded {found_count} / {v.n_out} experimental images")

    # Plot experimental patterns
    w = int(np.ceil(np.sqrt(v.n_out)))
    h = int(np.ceil(v.n_out / w))
    fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
    text_effect = withStroke(linewidth=3, foreground='black')
    axes = axes.flatten()
    for i in range(v.n_out):
        axes[i].imshow(v.lacbed_expt[:, :, i], cmap='gist_earth')
        axes[i].axis('off')
        annotation = f"{v.hkl[v.g_output[i], 0]}{v.hkl[v.g_output[i], 1]}{v.hkl[v.g_output[i], 2]}"
        axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                         size=30, color='w', path_effects=[text_effect])
    for i in range(v.n_out, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    best_corr = np.ones(v.n_out)  # Initialise correlation

# %% output — apply blur to simulated LACBED patterns if requested
from scipy.ndimage import gaussian_filter

if v.image_processing == 1:
    print(f"  Applying Gaussian blur: radius = {v.blur_radius} pixels")
    if hasattr(v, 'lacbed_sim') and v.lacbed_sim is not None:
        for i in range(v.lacbed_sim.shape[2]):
            v.lacbed_sim[:, :, i] = gaussian_filter(v.lacbed_sim[:, :, i], sigma=v.blur_radius)
    else:
        print("Warning: No simulated LACBED data found to blur.")

if 'S' in v.refine_mode:
    sim.print_LACBED(v)
else:
    fom = sim.figure_of_merit(v)
    print(f"  Figure of merit {100*fom:.2f}%")
    print("-------------------------------")
    sim.print_LACBED(v)

# %% start refinement loop *** needs work
if 'S' not in v.refine_mode:
    # Initialise variables for refinement
    fit0 = fom*1.0
    best_fit = fom*1.0
    last_fit = fom*1.0
    # p is a vector along the gradient in n-dimensional space
    p = np.ones(v.n_variables)
    last_p = np.ones(v.n_variables)
    df = 1.0
    r3_var = np.zeros(3)  # for parabolic minimum
    r3_fom = np.zeros(3)
    
    # Initialise storage for uncertainty estimates (not yet implemented)
    independent_delta = np.zeros(v.n_variables)

    
    # for a plot
    var_pl = ([])
    fit_pl = ([])
    
    # Refinement loop
    while df >= v.exit_criteria:
        # v.refined_variable is the working set of variables going into a sim
        # best_var is the running best fit during this refinement cycle
        best_var = np.copy(v.refined_variable)
        # next_var is the predicted next (best) point
        next_var = np.copy(v.refined_variable)
        # if all parameters have been refined and we're still in the loop, reset
        if np.sum(np.abs(last_p)) < 1e-10:
            last_p = np.ones(v.n_variables)
    
        # ===========individual variable minimisation
        # we go through the variables - if there's an easy minimisation
        # we take it and remove it from the list of variables to refine in
        # vector descent.  If it's been fixed, last_p[i] = 0
        # at the end of this for loop we have a predicted best starting point
        # for gradient descent best_var
        for i in range(v.n_variables):
            # Skip variables already optimized in previous refinements
            if abs(last_p[i]) < 1e-10:
                p[i] = 0.0
                continue
            v.current_variable_type = v.refined_variable_type[i]
            # Check if Debye-Waller factor is zero, skip if so
            if v.current_variable_type == 4 and v.refined_variable[i] <= 1e-10:
                p[i] = 0.0
                continue
    
            # middle point is the previous best simulation
            r3_var[1] = v.refined_variable[i]*1.0
            r3_fom[1] = last_fit*1.0
            var_pl.append(v.refined_variable[i])
            fit_pl.append(last_fit)
    
            # Update iteration
            
    
            print(f"Finding gradient, variable {i+1} of {v.n_variables}")
    
            # Display messages based on variable type (last digit)
            if v.current_variable_type % 10 == 1:
                print("Changing Ug")
            elif v.current_variable_type % 10 == 2:
                print("Changing atomic coordinate")
            elif v.current_variable_type % 10 == 3:
                print("Changing occupancy")
            elif v.current_variable_type % 10 == 4:
                print("Changing isotropic Debye-Waller factor")
            elif v.current_variable_type % 10 == 5:
                print("Changing anisotropic Debye-Waller factor")
            elif v.current_variable_type % 10 == 6:
                print("Changing lattice parameter")
            elif v.current_variable_type % 10 == 8:
                print("Changing convergence angle")
    
            # dx is a small change in the current variable determined by RScale
            # which is either refinement_scale for atomic coordinates and
            # refinement_scale*variable for everything else
            dx = abs(v.refinement_scale * v.refined_variable[i])
            if v.refined_variable_type[i] == 2:
                dx = abs(v.refinement_scale)
    
            # Three-point gradient measurement, starting with plus
            v.refined_variable[i] += dx
            # update variables
            sim.update_variables(v)
            sim.print_current_var(v, v.refined_variable[i])
            # simulate
            setup, bwc = sim.simulate(v)
            v.iter_count += 1
            # figure of merit
            fom = sim.figure_of_merit(v)
            if (fom < best_fit):
                best_fit = fom*1.0
                best_var = np.copy(v.refined_variable)
            r3_var[2] = v.refined_variable[i]*1.0
            r3_fom[2] = fom*1.0
            print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
            print(f"-1-----------------------------")  # {r3_var},{r3_fom}")
            var_pl.append(v.refined_variable[i])
            fit_pl.append(fom)
    
            # keep going or turn round?
            if r3_fom[2] < r3_fom[1]:  # keep going
                v.refined_variable[i] += np.exp(0.5) * dx
            else:  # turn round
                dx = - dx
                v.refined_variable[i] += np.exp(0.75) * dx
    
            # update variables
            sim.update_variables(v)
            sim.print_current_var(v, v.refined_variable[i])
            # simulate
            setup, bwc = sim.simulate(v)
            v.iter_count += 1
            # figure of merit
            fom = sim.figure_of_merit(v)
            if (fom < best_fit):
                best_fit = fom*1.0
                best_var = np.copy(v.refined_variable)
            r3_var[0] = v.refined_variable[i]*1.0
            r3_fom[0] = fom*1.0
            print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
            print(f"-2-----------------------------")  # {r3_var},{r3_fom}")
            var_pl.append(v.refined_variable[i])
            fit_pl.append(fom)
    
            # predict the next point as a minimum or a step on
            next_var[i], exclude = px.convex(r3_var, r3_fom)
            if exclude:
                p[i] = 0.0  # this variable doesn't get included in vector downhill
            else:
                # we weight the variable by -df/dx
                p[i] = -(r3_fom[2] - r3_fom[0]) / (2 * dx)
                # error estimate goes here
                # independent_delta[i] = delta_x(r3_var, r3_fom, precision, err)
    
        # ===========vector descent
        # either: point 1 of 3, or final simulation using the prediction next_var
        v.refined_variable = np.copy(next_var)
        # simulation
        sim.update_variables(v)
        setup, bwc = sim.simulate(v)
        v.iter_count += 1
        fom = sim.figure_of_merit(v)
        if (fom < best_fit):
            best_fit = fom*1.0
            best_var = np.copy(v.refined_variable)
        print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
        fit_pl.append(fom)
        sim.print_LACBED(v)
    
        # Reset the gradient vector magnitude and initialize vector descent
        p_mag = np.linalg.norm(p)
        if np.isinf(p_mag) or np.isnan(p_mag):
            raise ValueError(f"Infinite or NaN gradient! Refinement vector = {p}")
        if abs(p_mag) > 1e-10:  # There are gradients, do the vector descent
            p = p / p_mag   # Normalized direction of max/min gradient
            j = np.where(np.abs(p) >= 1e-10)[0][0]
            v.current_variable_type = v.refined_variable_type[j]
            sim.print_current_var(v, v.refined_variable[i])
            print(f"Refining, refinement vector {p}")
            # Find index of the first non-zero element in the gradient vector
            # reset the refinement scale (last term reverses sign if we overshot)
            p_mag = -best_var[j] * v.refinement_scale  
            # First of three points for concavity test is the best simulation
            r3_var[0] = best_var[j]*1.0
            r3_fom[0] = fom*1.0
            print(f"-a-----------------------------")  # {r3_var},{r3_fom}")
    
            # Second point
            print("Refining, point 2 of 3")
            # NB vectors here, not individual variables
            v.refined_variable = v.refined_variable + p * p_mag
            # simulation
            sim.print_current_var(v, v.refined_variable[j])
            sim.update_variables(v)
            setup, bwc = sim.simulate(v)
            v.iter_count += 1
            fom = sim.figure_of_merit(v)
            if (fom < best_fit):
                best_fit = fom*1.0
                best_var = np.copy(v.refined_variable[j])
            r3_var[1] = v.refined_variable[j]*1.0
            r3_fom[1] = fom*1.0
            print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
            print(f"-b-----------------------------")  # {r3_var},{r3_fom}")
            var_pl.append(v.refined_variable[j])
            fit_pl.append(fom)
    
            # Third point
            print("Refining, point 3 of 3")
            if r3_fom[1] > r3_fom[0]:  # if second point is worse
                p_mag = -p_mag
                v.refined_variable += np.exp(0.6)*p*p_mag  # Go in the opposite direction
            else:  # keep going
                v.refined_variable += p*p_mag
    
            # simulation
            sim.print_current_var(v, v.refined_variable[j])
            sim.update_variables(v)
            setup, bwc = sim.simulate(v)
            v.iter_count += 1
            fom = sim.figure_of_merit(v)
            if (fom < best_fit):
                best_fit = fom*1.0
                best_var = np.copy(v.refined_variable)
            r3_var[2] = v.refined_variable[j]*1.0
            r3_fom[2] = fom*1.0
            print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
            print(f"-c-----------------------------")  # {r3_var},{r3_fom}")
            var_pl.append(v.refined_variable[j])
            fit_pl.append(fom)
    
            # we continue until we get a predicted minnymum
            minny = False
            while minny is False:
                last_x = v.refined_variable[j]*1.0
                # predict the next point as a minimum or a step on
                next_x, minny = px.convex(r3_var, r3_fom)
                v.refined_variable += p * (next_x-last_x) / p[j]
                # simulation
                sim.print_current_var(v, v.refined_variable[j])
                sim.update_variables(v)
                setup, bwc = sim.simulate(v)
                v.iter_count += 1
                fom = sim.figure_of_merit(v)
                if (fom < best_fit):
                    best_fit = fom*1.0
                    best_var = np.copy(v.refined_variable)
                    # replace worst point with this one
                    i = np.argmax(r3_fom)
                    r3_var[i] = v.refined_variable[j]
                    r3_fom[i] = fom*1.0
                else:
                    minny = True
                print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
                print(f"-.-----------------------------")  # {r3_var},{r3_fom}")
                var_pl.append(v.refined_variable[j])
                fit_pl.append(fom)
    
            # End of this cycle
            print("Refinement cycle complete")
            p[j] = 0.0
            sim.print_LACBED(v)
        # Update for next iteration
        last_p = p
        df = last_fit - best_fit
        last_fit = best_fit*1.0
        v.refined_variable = np.copy(best_var)
        v.refinement_scale *= (1 - 1 / (2 * v.n_variables))
        print(f"Improvement in fit {100*df:.2f}%, will stop at {100*v.exit_criteria:.2f}%")
        print("-------------------------------")
        plt.figure(figsize=(6, 4))
        plt.plot(fit_pl, marker='o', linestyle='-')
        plt.title("Figure of Merit over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Figure of Merit (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
    print(f"Refinement complete after {v.iter_count} simulations.  Refined values: {best_var}")

# %% final print
# sim.print_LACBED(v)
total_time = time.time() - start
print("-----------------------------------------------------------------")
print(f"Beam pool calculation took {setup:.3f} seconds")
print(f"Bloch wave calculation in {bwc:.1f} s ({1000*(bwc)/(4*v.image_radius**2):.2f} ms/pixel)")
print(f"Total time {total_time:.1f} s")
print("-----------------------------------------------------------------")
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")


###|||JUNK ZONE|||###

# attempt at an animated set of frames
# # Initialize figure
# fig, ax = plt.subplots(frameon=False)
# ax.set_xticks([])  # Hide ticks
# ax.set_yticks([])
# frame_img = ax.imshow(np.zeros((v.frame_size_x, v.frame_size_y)), cmap='cividis',
#                       vmin=0, vmax=I000)

# def update(frame_idx):
#     frame = np.zeros((v.frame_size_x, v.frame_size_y), dtype=float)
#     if hkl_indices[frame_idx].size > 0:
#         # Set max intensity in the center region
#         frame[x0 - dw:x0 + dw, y0 - dw:x0 + dw] = I000

#         # Place intensities at calculated positions
#         for j, xy in enumerate(x_y[frame_idx]):
#             frame[x0 + xy[0] - dw:x0 + xy[0] + dw,
#                   y0 + xy[1] - dw:y0 + xy[1] + dw] = I_kin_frame[frame_idx][j]
#     frame_img.set_array(frame)
#     return [frame_img]
    
# ani = animation.FuncAnimation(fig, update, frames=v.n_frames, interval=38.46, blit=False)
# plt.show()
# ani.save("animation.gif", writer="pillow", fps=26)


# %% [Work in progress]: Animation of simulated frames (replace "JUNK ZONE")
if hasattr(v, "frame_output") and v.frame_output == 1 and 'I_dyn_frame' in globals():
    print("Creating animation from simulated LACBED frames...")

    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    frame_img = ax.imshow(np.zeros((v.frame_size_y, v.frame_size_x)),
                          cmap='cividis', vmin=0, vmax=1.0)

    def update(frame_idx):
        frame = np.zeros((v.frame_size_y, v.frame_size_x))
        if frame_idx < len(I_dyn_frame):
            norm = np.max(I_dyn_frame[frame_idx])
            if norm > 0:
                frame = I_dyn_frame[frame_idx] / norm  # normalize per frame
        frame_img.set_array(frame)
        return [frame_img]

    ani = animation.FuncAnimation(fig, update, frames=v.n_frames, interval=40, blit=False)
    plt.show()
    ani.save("simulation_animation.gif", writer="pillow", fps=25)
    print("Animation saved as simulation_animation.gif")
