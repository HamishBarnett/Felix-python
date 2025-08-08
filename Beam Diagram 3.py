# ===============================================================
# STEP 1 — LOAD AND PARSE INPUT CIF FILES
# ===============================================================
# This step:
#   1) Reads the crystal structure CIF (silicon_structure.cif)
#      — contains lattice parameters and atomic positions.
#   2) Reads the PETS experimental dyn.cif file (Si_3_dyn.cif_pets)
#      — contains UB matrix, zone axis table, and reflection intensities.
# We store their contents in convenient Python data structures for later steps.
#
# Notes:
#   - No external crystallography library is required here unless
#     we want symmetry handling later (e.g., pymatgen or ASE).
#   - We keep it text-based parsing because PETS dyn.cif is not
#     a standard structural CIF — it has custom loops for experimental data.
#
# ===============================================================

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 1.1 — Define file paths
# ---------------------------------------------------------------
# Update these paths if files are located elsewhere.
# The CIF file: silicon_structure.cif  — crystal structure
# The PETS file: Si_3_dyn.cif_pets     — experimental data

structure_cif_path = "silicon_structure.cif"
pets_dyn_cif_path = "Si_3_dyn.cif_pets"

# ---------------------------------------------------------------
# 1.2 — Read the structure CIF (basic parsing)
# ---------------------------------------------------------------
# We will extract:
#   - Lattice parameters a, b, c, alpha, beta, gamma
#   - Wavelength (if present)
#   - Atomic coordinates (optional for later)
#
# Since the silicon_structure.cif is a standard structural CIF,
# we can use a simple text scan to get lattice parameters.
# For robust parsing in production, use `pymatgen` or `ase`.

def read_structure_cif(file_path):
    """Reads a simple CIF and returns lattice parameters in a dict."""
    lattice_params = {}
    atoms = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Helper to extract float value safely
            def extract_float(val):
                return float(val.split('(')[0])

            # Lattice constants (in Å)
            if line.lower().startswith("_cell_length_a"):
                lattice_params['a'] = extract_float(line.split()[1])
            elif line.lower().startswith("_cell_length_b"):
                lattice_params['b'] = extract_float(line.split()[1])
            elif line.lower().startswith("_cell_length_c"):
                lattice_params['c'] = extract_float(line.split()[1])

            # Lattice angles (in degrees)
            elif line.lower().startswith("_cell_angle_alpha"):
                lattice_params['alpha'] = extract_float(line.split()[1])
            elif line.lower().startswith("_cell_angle_beta"):
                lattice_params['beta'] = extract_float(line.split()[1])
            elif line.lower().startswith("_cell_angle_gamma"):
                lattice_params['gamma'] = extract_float(line.split()[1])

            # Wavelength (if defined)
            elif line.lower().startswith("_diffrn_radiation_wavelength"):
                lattice_params['wavelength'] = extract_float(line.split()[1])

            # Atomic coordinates (optional, ignored if not present)
            elif len(line.split()) >= 4:
                parts = line.split()
                try:
                    # Try to interpret as: label, x, y, z
                    label = parts[0]
                    x, y, z = map(float, parts[1:4])
                    atoms.append((label, x, y, z))
                except ValueError:
                    pass  # Ignore lines that don't match numeric coords

    return lattice_params, atoms

# ---------------------------------------------------------------
# 1.3 — Read the PETS dyn.cif file
# ---------------------------------------------------------------
# The PETS dyn.cif contains:
#   - UB matrix (orientation)
#   - Zone axis table
#   - Reflection list with intensities
#
# We'll store:
#   UB: 3x3 numpy array
#   wavelength: float
#   zone_axes: DataFrame with columns:
#       [id, u, v, w, precession_angle, alpha, beta, omega, scale]
#   reflections: DataFrame with columns:
#       [h, k, l, intensity, sigma, zone_axis_id]

def read_pets_dyn_cif(file_path):
    """Parses a PETS dyn.cif file into UB, wavelength, zone axes, reflections."""
    UB = np.zeros((3, 3))
    wavelength = None
    zone_axes_data = []
    reflections_data = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # --- Scan for UB matrix and wavelength ---
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            tag = parts[0].lower()
            if tag == "_diffrn_orient_matrix_ub_11":
                UB[0, 0] = float(parts[1])
                UB[0, 1] = float(lines[i+1].split()[1])
                UB[0, 2] = float(lines[i+2].split()[1])
                UB[1, 0] = float(lines[i+3].split()[1])
                UB[1, 1] = float(lines[i+4].split()[1])
                UB[1, 2] = float(lines[i+5].split()[1])
                UB[2, 0] = float(lines[i+6].split()[1])
                UB[2, 1] = float(lines[i+7].split()[1])
                UB[2, 2] = float(lines[i+8].split()[1])
            elif tag == "_diffrn_radiation_wavelength":
                wavelength = float(parts[1])

    # --- Find zone axis loop ---
    zone_axis_start = None
    for i, line in enumerate(lines):
        if "_diffrn_zone_axis_id" in line:
            zone_axis_start = i + 9  # Skip header lines (known offset)
            break
    if zone_axis_start is not None:
        for line in lines[zone_axis_start:]:
            parts = line.strip().split()
            if len(parts) == 0 or parts[0].startswith('_'):
                break  # End of table
            if len(parts) >= 9:
                zone_axes_data.append((
                    int(parts[0]), float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]), float(parts[5]),
                    float(parts[6]), float(parts[7]), float(parts[8])
                ))

    zone_axes_df = pd.DataFrame(zone_axes_data, columns=[
        "id", "u", "v", "w", "precession_angle", "alpha", "beta", "omega", "scale"
    ])

    # --- Find reflection list ---
    refl_start = None
    for i, line in enumerate(lines):
        if "_refln_index_h" in line:
            refl_start = i + 6  # Skip header lines (known offset)
            break
    if refl_start is not None:
        for line in lines[refl_start:]:
            parts = line.strip().split()
            if len(parts) == 0 or parts[0].startswith('_'):
                break
            if len(parts) >= 6:
                reflections_data.append((
                    int(parts[0]), int(parts[1]), int(parts[2]),
                    float(parts[3]), float(parts[4]), int(parts[5])
                ))

    reflections_df = pd.DataFrame(reflections_data, columns=[
        "h", "k", "l", "intensity", "sigma", "zone_axis_id"
    ])

    return UB, wavelength, zone_axes_df, reflections_df

# ---------------------------------------------------------------
# 1.4 — Execute the readers and show a summary
# ---------------------------------------------------------------
lattice_params, atoms = read_structure_cif(structure_cif_path)
UB, wavelength_exp, zone_axes_df, reflections_df = read_pets_dyn_cif(pets_dyn_cif_path)

print("=== Structure CIF Summary ===")
print(lattice_params)
print(f"Number of atoms read: {len(atoms)}")
print()
print("=== PETS dyn.cif Summary ===")
print("UB matrix:")
print(UB)
print(f"Wavelength (PETS file): {wavelength_exp}")
print(f"Zone axes count: {len(zone_axes_df)}")
print(f"Reflections count: {len(reflections_df)}")

# ===============================================================
# STEP 2 — CALCULATE RECIPROCAL SPACE G-VECTORS & BRAGG CURVES
# ===============================================================
# This step:
#   - Converts (h,k,l) from the PETS reflections into reciprocal
#     space vectors (G) in the laboratory frame.
#   - Simulates the Bragg condition curves by rotating the crystal
#     about the experimental axis (rocking curve simulation).
#   - Stores each curve for later plotting.
#
# References:
#   G = UB * hkl
#   |k_out| = |k_in| = 1 / wavelength
#   Bragg condition: k_out - k_in = G
#
# ===============================================================

# ---------------------------------------------------------------
# 2.1 — Helper: Convert hkl to reciprocal space vector
# ---------------------------------------------------------------
def hkl_to_G(UB_matrix, h, k, l):
    """
    Convert Miller indices (h,k,l) to a reciprocal space vector G.
    UB_matrix: 3x3 numpy array from PETS
    Returns: 3-element numpy array
    """
    hkl_vec = np.array([h, k, l], dtype=float)
    return UB_matrix @ hkl_vec  # matrix multiplication

# ---------------------------------------------------------------
# 2.2 — Simulate Bragg condition curves
# ---------------------------------------------------------------
# Assumptions:
#   - The red beam path will later be drawn as a horizontal line
#     in the reciprocal space plot (k_y vs frame/angle).
#   - Here we generate for each reflection the locus of points
#     (curve) where it would satisfy the Bragg condition as the
#     crystal rotates about the goniometer axis.
#
# Simplification:
#   - We assume single-axis rotation for now (e.g. omega axis).
#   - α, β tilt adjustments will be introduced in Step 5 for panel (b).

# Rotation matrix around y-axis (example for omega tilt)
def rotation_matrix_y(theta_rad):
    """Rotation about the y-axis by theta (in radians)."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

# Generate Bragg curves for all reflections
def generate_bragg_curves(UB_matrix, reflections_df, wavelength, omega_range_deg=(-5, 5), n_points=200):
    """
    For each reflection in reflections_df, simulate the Bragg condition curve
    over a range of omega rotation angles.
    
    Returns:
        curves: dict mapping reflection index -> array of shape (n_points, 2)
                columns: [rotation_angle_deg, scattering_parameter]
    """
    k0_mag = 1.0 / wavelength  # magnitude of wavevector in Å^-1
    curves = {}

    # Generate omega angles in radians
    omega_angles_deg = np.linspace(omega_range_deg[0], omega_range_deg[1], n_points)
    omega_angles_rad = np.deg2rad(omega_angles_deg)

    for idx, row in reflections_df.iterrows():
        # hkl vector
        G_vec = hkl_to_G(UB_matrix, row.h, row.k, row.l)

        curve_points = []
        for omega_deg, omega_rad in zip(omega_angles_deg, omega_angles_rad):
            # Rotate G according to omega
            R = rotation_matrix_y(omega_rad)
            G_rot = R @ G_vec

            # Bragg condition check:
            # Project G onto incident beam direction (assumed along z)
            # The deviation from exact Bragg is represented here as ky
            ky = G_rot[1]  # y-component in lab frame
            curve_points.append((omega_deg, ky))

        curves[idx] = np.array(curve_points)

    return curves

# ---------------------------------------------------------------
# 2.3 — Run the Bragg curve generation
# ---------------------------------------------------------------
bragg_curves = generate_bragg_curves(UB, reflections_df, wavelength_exp)

print(f"Generated Bragg curves for {len(bragg_curves)} reflections.")

# ===============================================================
# STEP 3 — BUILD SIMULATED ROCKING-CURVE INTENSITIES, EXTRACT
#           EXPERIMENTAL PEAK FRAMES, AND COMPUTE YELLOW DOTS
# ===============================================================
#
# This step:
#  - Uses the per-frame omega values from zone_axes_df to build a
#    simulation grid that exactly matches the experimental angles.
#  - For each observed reflection (h,k,l) we compute a simulated
#    rocking curve I_sim(omega) using the linearized Bragg dev -> dtheta.
#  - We compute an experimental peak position per reflection using an
#    intensity-weighted centroid across the frames where it was observed.
#  - We evaluate the simulated curve at that experimental peak (yellow dot).
#
# IMPORTANT:
#  - This code uses the same UB matrix and wavelength read earlier.
#  - The rocking-curve width (rc_width_deg) is set from PETS header or
#    a default; ensure units (degrees vs radians) consistency.
#
# ===============================================================

import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  # optional gaussian fit helper

# ---------------------------------------------------------------
# 3.0 — Parameters & helpers
# ---------------------------------------------------------------
# Rocking-curve width: use PETS RC width if known, otherwise default.
# The PETS header you provided earlier included "RC width: 0.00100".
# We'll treat that as degrees and convert to radians internally.
rc_width_deg_default = 0.0010  # degrees (tweakable)
rc_width_deg = rc_width_deg_default
rc_width_rad = np.deg2rad(rc_width_deg)  # gaussian sigma in radians

# Incident wavevector magnitude (1/Å)
k0 = 1.0 / wavelength_exp

# Small-angle conversion helper: dev -> delta_theta approximation
def dev_to_dtheta(dev, k0_val, G_norm):
    """
    Convert scalar 'dev' = (2 k0·G - |G|^2) to an angular offset DeltaTheta
    using the small-angle linearization:
        DeltaTheta ≈ dev / (2 * k0 * |G|)
    """
    # avoid division by zero
    if G_norm <= 0:
        return 0.0
    return dev / (2.0 * k0_val * G_norm)

# Gaussian rocking-curve model (intensity vs dtheta)
def rc_gaussian(dtheta, sigma_rad):
    """Normalized Gaussian rocking curve: peak=1 at dtheta=0."""
    return np.exp(-(dtheta / sigma_rad) ** 2)

# Optional: Gaussian fit to refine experimental peak (sub-frame)
def gaussian(x, A, x0, s, b):
    """A * exp(-(x-x0)^2/(2 s^2)) + b"""
    return A * np.exp(-0.5 * ((x - x0) / s) ** 2) + b

# ---------------------------------------------------------------
# 3.1 — Prepare omega grid (use experimental per-frame omegas)
# ---------------------------------------------------------------
# Ensure zone axis frames are sorted by id and extract omega values.
zone_axes_sorted = zone_axes_df.sort_values("id").reset_index(drop=True)
# omega values in degrees (as recorded in PETS); convert to numpy array
omega_by_frame = zone_axes_sorted["omega"].to_numpy(dtype=float)
frame_ids = zone_axes_sorted["id"].to_numpy(dtype=int)
n_frames = len(frame_ids)
print(f"[STEP 3] Number of frames (zone axes): {n_frames}")

# For simulation we will evaluate the model at these exact omegas:
omega_sim_deg = omega_by_frame.copy()
omega_sim_rad = np.deg2rad(omega_sim_deg)  # convert to radians for trig

# ---------------------------------------------------------------
# 3.2 — Vectorize per-frame rotation matrices (rotation about +y)
#         Note: we use exact rotations (no small-angle approx) here.
# ---------------------------------------------------------------
def rotation_matrix_y_vec(theta_rad_array):
    """
    Build (N,3,3) array of rotation matrices about y for each theta in theta_rad_array.
    Rotation follows right-hand rule; positive rotates +z towards +x.
    """
    thetas = np.asarray(theta_rad_array)
    c = np.cos(thetas)
    s = np.sin(thetas)
    # create stacked rotation matrices
    R = np.zeros((len(thetas), 3, 3))
    R[:, 0, 0] = c
    R[:, 0, 2] = s
    R[:, 1, 1] = 1.0
    R[:, 2, 0] = -s
    R[:, 2, 2] = c
    return R

R_omega_stack = rotation_matrix_y_vec(omega_sim_rad)  # shape (n_frames, 3, 3)

# ---------------------------------------------------------------
# 3.3 — Utility: compute simulated intensity curve for one reflection
# ---------------------------------------------------------------
def simulate_reflection_intensity(UB_matrix, h, k, l, R_stack, k0_val, rc_sigma_rad):
    """
    Simulate normalized rocking-curve intensity for (h,k,l) over the rotation
    matrices provided in R_stack (shape (n_frames,3,3)).
    Returns: I_sim array of length n_frames (values between 0 and 1).
    """
    # compute G_crystal (lab) using UB: UB @ [h,k,l]
    G_crystal = UB_matrix @ np.array([h, k, l], dtype=float)  # vector in 1/Å
    # We'll rotate G_crystal by each frame's rotation R to get lab G_rot
    # Vectorize: for each frame i, compute G_rot_i = R_stack[i] @ G_crystal
    G_rot = R_stack @ G_crystal  # shape (n_frames, 3)

    # compute norms and dot products with k0 vector (incident along +z)
    G_norms = np.linalg.norm(G_rot, axis=1)  # shape (n_frames,)
    # k0 vector is (0,0,k0)
    k0_vec = np.array([0.0, 0.0, k0_val])
    # compute 2*k0·G - |G|^2 for each frame
    two_k0_dot_G = 2.0 * (G_rot @ k0_vec)  # shape (n_frames,)
    G_sq = np.sum(G_rot ** 2, axis=1)
    dev = two_k0_dot_G - G_sq

    # convert dev -> dtheta (radians) using linearized formula
    # handle zero-norms gracefully
    with np.errstate(divide='ignore', invalid='ignore'):
        dtheta = np.where(G_norms > 0.0, dev / (2.0 * k0_val * G_norms), 0.0)

    # simulated intensity: normalized Gaussian with sigma = rc_sigma_rad
    I_sim = rc_gaussian(dtheta, rc_sigma_rad)

    return I_sim, dtheta, G_rot

# ---------------------------------------------------------------
# 3.4 — Build simulated curves for a selected subset of reflections
#        (to avoid plotting extreme clutter, default to top N by intensity)
# ---------------------------------------------------------------
max_reflections_to_plot = 50  # tweakable; set None to process all
# compute max intensity per reflection (across frames) to rank
refl_group = reflections_df.groupby(["h", "k", "l"])["intensity"].max().reset_index()
refl_group = refl_group.rename(columns={"intensity": "Imax"})
refl_group_sorted = refl_group.sort_values("Imax", ascending=False).reset_index(drop=True)

if max_reflections_to_plot is not None:
    refl_to_use = refl_group_sorted.head(max_reflections_to_plot)
else:
    refl_to_use = refl_group_sorted

print(f"[STEP 3] Selected {len(refl_to_use)} reflections for simulation/plotting (top by Imax).")

# Pre-allocate structures to store results
simulated_curves = {}      # key: idx (h,k,l tuple) -> dict with 'omega_deg','I_sim', 'dtheta'
experimental_peaks = []    # list of dicts with per-reflection experimental centroid info
yellow_dots = []           # list of dicts for plotting: omega_deg, I_sim_at_peak

# group the raw reflection rows by (h,k,l) for centroid calculation
refl_full_groups = reflections_df.groupby(["h", "k", "l"])

# Loop over selected reflections
for _, row in refl_to_use.iterrows():
    h, k, l = int(row.h), int(row.k), int(row.l)
    hkl_key = (h, k, l)

    # 3.4.1 — Simulate the rocking curve at the experimental omega grid
    I_sim, dtheta_vec, G_rot_stack = simulate_reflection_intensity(
        UB, h, k, l, R_omega_stack, k0, rc_width_rad
    )
    simulated_curves[hkl_key] = {
        "omega_deg": omega_sim_deg,
        "I_sim": I_sim,
        "dtheta": dtheta_vec,
        "G_rot": G_rot_stack
    }

    # 3.4.2 — Extract experimental intensities & frames for this reflection
    # Some reflections may not have entries at every frame; group returns only the observed rows
    try:
        obs = refl_full_groups.get_group((h, k, l)).copy()
    except KeyError:
        # No observed points for this (hkl) - skip
        # (shouldn't happen for selected reflections, but safe guard)
        continue

    # Frames referenced in PETS refl table are zone_axis_id – map those to omega using zone_axes_df
    obs_frames = obs["zone_axis_id"].to_numpy(dtype=int)
    # Map frame ids to omega (we assume frame ids correspond exactly to zone_axes_df id)
    # Build a mapping dict from frame id -> omega_deg
    frame_to_omega = dict(zip(frame_ids, omega_sim_deg))
    # Use only frames that exist in our mapping
    valid_mask = [f in frame_to_omega for f in obs_frames]
    if not any(valid_mask):
        # no valid frames -> skip
        continue
    obs = obs.loc[valid_mask].reset_index(drop=True)
    obs_frames = obs["zone_axis_id"].to_numpy(dtype=int)
    obs_omega = np.array([frame_to_omega[f] for f in obs_frames])
    obs_intensity = obs["intensity"].to_numpy(dtype=float)

    # 3.4.3 — Compute experimental centroid (sub-frame weighted mean)
    # centroid in frame-id space (but we will convert to omega for plotting)
    centroid_frame = np.sum(obs_frames * obs_intensity) / np.sum(obs_intensity)
    centroid_omega = np.sum(obs_omega * obs_intensity) / np.sum(obs_intensity)

    # Optional: refine centroid by Gaussian fit in omega space (uncomment if desired)
    do_gauss_fit = False
    if do_gauss_fit and len(obs_omega) >= 5:
        # initial guesses: A, x0, s, b
        A0 = obs_intensity.max() - obs_intensity.min()
        x0_0 = centroid_omega
        s0 = max(1.0, (obs_omega.max() - obs_omega.min()) / 6.0)
        b0 = obs_intensity.min()
        try:
            popt, pcov = curve_fit(gaussian, obs_omega, obs_intensity, p0=(A0, x0_0, s0, b0))
            fitted_x0 = popt[1]
            centroid_omega = float(fitted_x0)
        except Exception as e:
            # if fit fails, keep centroid
            pass

    # 3.4.4 — Evaluate simulated intensity at experimental centroid omega (yellow dot)
    # Interpolate I_sim (which is defined on omega_sim_deg) at centroid_omega
    I_interp = np.interp(centroid_omega, simulated_curves[hkl_key]["omega_deg"], simulated_curves[hkl_key]["I_sim"])

    # Store experimental peak info and yellow dot
    experimental_peaks.append({
        "h": h, "k": k, "l": l,
        "centroid_frame": centroid_frame,
        "centroid_omega_deg": centroid_omega,
        "I_peak_obs": obs_intensity.max(),
        "I_sim_at_peak": I_interp
    })
    yellow_dots.append({
        "h": h, "k": k, "l": l,
        "omega_deg": centroid_omega,
        "I_sim": I_interp
    })

print(f"[STEP 3] Computed experimental centroids and yellow dots for {len(yellow_dots)} reflections.")

# ---------------------------------------------------------------
# 3.5 — Quick diagnostic plot (Panel Fig. 8a style)
# ---------------------------------------------------------------
# Plot simulated curves (white), experimental peak verticals (blue),
# yellow dots (where experimental peak meets simulated curve), and red horizontal line.

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_title("Panel (a) — Simulated rocking curves vs experimental peak frames (omega)")
ax.set_xlabel("Omega (degrees)")
ax.set_ylabel("Simulated normalized intensity (arb units)")

# Plot each simulated curve (light/grey to mimic white curves on a light bg)
for hkl_key, data in simulated_curves.items():
    ax.plot(data["omega_deg"], data["I_sim"], color='0.85', linewidth=0.9, alpha=0.9)

# Plot blue verticals at each experimental centroid omega
for ep in experimental_peaks:
    ax.axvline(ep["centroid_omega_deg"], color='blue', linewidth=0.7, alpha=0.8)

# Plot yellow dots (I_sim evaluated at the experimental centroid omega)
ys = [d["I_sim"] for d in yellow_dots]
xs = [d["omega_deg"] for d in yellow_dots]
ax.scatter(xs, ys, s=30, facecolors='yellow', edgecolors='black', zorder=5, label='Exp peak on sim curve')

# Plot red horizontal "beam path" at I0 = 1 (simulated peak intensity)
ax.axhline(1.0, color='red', linewidth=1.2, label='Beam path (I0=1)')

ax.set_ylim(-0.05, 1.05)
ax.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 3.6 — Save the experimental peaks table for later steps
# ---------------------------------------------------------------
peaks_df = pd.DataFrame(experimental_peaks)
peaks_df.to_csv("results_experimental_peaks.csv", index=False)
print(f"[STEP 3] Wrote experimental peaks summary to results_experimental_peaks.csv")
