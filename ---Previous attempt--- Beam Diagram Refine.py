# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 15:04:34 2025

@author: Hamish
"""

# === Step 1: Parse CIF files ===

from CifFile import ReadCif
import pandas as pd

def parse_structure_cif(file_path):
    """Parse silicon_structure.cif to extract unit cell and symmetry info."""
    cf = ReadCif(file_path)
    block = list(cf.keys())[0]
    data = cf[block]

    unit_cell = {
        'a': float(data['_cell_length_a'].split('(')[0]),
        'b': float(data['_cell_length_b'].split('(')[0]),
        'c': float(data['_cell_length_c'].split('(')[0]),
        'alpha': float(data['_cell_angle_alpha']),
        'beta': float(data['_cell_angle_beta']),
        'gamma': float(data['_cell_angle_gamma']),
    }

    space_group = data.get('_space_group_name_H-M_alt', None)
    symmetry_ops = data.get('_space_group_symop_operation_xyz', [])

    return unit_cell, space_group, symmetry_ops


def parse_dyn_cif(file_path):
    """Parse Si_3_dyn.cif_pets to extract zone axes and reflection data."""
    cf = ReadCif(file_path)
    block = list(cf.keys())[0]
    data = cf[block]

    # Parse zone axis orientations
    zone_axis_df = pd.DataFrame({
        'id': data['_diffrn_zone_axis_id'],
        'u': list(map(float, data['_diffrn_zone_axis_u'])),
        'v': list(map(float, data['_diffrn_zone_axis_v'])),
        'w': list(map(float, data['_diffrn_zone_axis_w'])),
        'alpha': list(map(float, data['_diffrn_zone_axis_alpha'])),
        'beta': list(map(float, data['_diffrn_zone_axis_beta'])),
        'omega': list(map(float, data['_diffrn_zone_axis_alpha'])),  # override omega with alpha (made error previously)
        'scale': list(map(float, data['_diffrn_zone_axis_scale'])),
    })

    # Parse reflections
    reflections_df = pd.DataFrame({
        'h': list(map(int, data['_refln_index_h'])),
        'k': list(map(int, data['_refln_index_k'])),
        'l': list(map(int, data['_refln_index_l'])),
        'intensity': list(map(float, data['_refln_intensity_meas'])),
        'sigma': list(map(float, data['_refln_intensity_sigma'])),
        'zone_axis_id': data['_refln_zone_axis_id']
    })

    return zone_axis_df, reflections_df


# === Load data ===

structure_path = 'silicon_structure.cif'  # Update if your path is different
dyn_cif_path = 'Si_3_dyn.cif_pets'  # Update if needed

# Step 1a: Parse structure info
unit_cell, space_group, symmetry_ops = parse_structure_cif(structure_path)
print("Unit cell parameters:")
print(unit_cell)

# Step 1b: Parse experimental beam and reflection data
zone_axes, reflections = parse_dyn_cif(dyn_cif_path)
print("\nFirst few zone axis entries:")
print(zone_axes.head())

print("\nFirst few reflections:")
print(reflections.head())


# === Step 2: Generate Allowed Reflections ===

import numpy as np

def generate_reflections(unit_cell, d_min=0.5, hkl_max=10):
    """
    Generate all (h, k, l) reflections with d >= d_min.
    Assumes pseudo-cubic cell. Can be extended to general triclinic.
    
    Parameters:
        unit_cell: dict with a, b, c, alpha, beta, gamma
        d_min: minimum d-spacing to include [Å]
        hkl_max: max index to consider in each direction

    Returns:
        List of tuples: (h, k, l, d_spacing, g_vector)
    """
    a, b, c = unit_cell['a'], unit_cell['b'], unit_cell['c']
    alpha = np.radians(unit_cell['alpha'])
    beta = np.radians(unit_cell['beta'])
    gamma = np.radians(unit_cell['gamma'])

    # Volume of unit cell
    volume = a * b * c * np.sqrt(
        1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
        + 2 * np.cos(alpha)*np.cos(beta)*np.cos(gamma)
    )

    # Reciprocal lattice parameters (a*, b*, c*)
    astar = (b * c * np.sin(alpha)) / volume
    bstar = (a * c * np.sin(beta)) / volume
    cstar = (a * b * np.sin(gamma)) / volume

    allowed_reflections = []

    for h in range(-hkl_max, hkl_max + 1):
        for k in range(-hkl_max, hkl_max + 1):
            for l in range(-hkl_max, hkl_max + 1):
                if h == k == l == 0:
                    continue

                # Calculate |g|^2
                g_squared = (h * astar)**2 + (k * bstar)**2 + (l * cstar)**2
                d_spacing = 1 / np.sqrt(g_squared)

                if d_spacing >= d_min:
                    # g-vector in reciprocal space
                    g_vec = np.array([h * astar, k * bstar, l * cstar])
                    allowed_reflections.append((h, k, l, d_spacing, g_vec))

    return allowed_reflections


# === Run Step 2 ===

generated_reflections = generate_reflections(unit_cell, d_min=0.5, hkl_max=8)
print(f"\nGenerated {len(generated_reflections)} allowed reflections with d ≥ 0.5 Å")

# Preview some reflections
for refl in generated_reflections[:5]:
    h, k, l, d, g = refl
    print(f"hkl=({h},{k},{l})  d={d:.3f} Å  g=({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})")


# === Step 3: Predict Bragg condition satisfaction ===

def rotate_vector_y(vec, angle_deg):
    """Rotate a 3D vector around the y-axis by angle in degrees."""
    theta = np.radians(angle_deg)
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return R @ vec


def predict_bragg_angles(reflections, k0_dir=np.array([0, 0, 1]), omega_range=(-70, 70), omega_step=0.1, tol=0.01):
    """
    Predict the omega angle where each reflection satisfies the Bragg condition.
    
    Parameters:
        reflections: list of (h, k, l, d, g_vector)
        k0_dir: incident beam vector (unit vector along z)
        omega_range: min and max omega values in degrees
        omega_step: angular step in degrees
        tol: tolerance for Bragg condition satisfaction (in reciprocal length^2)

    Returns:
        List of (h, k, l, predicted_omega) for each reflection that satisfies Bragg
    """
    predictions = []
    k0 = k0_dir / np.linalg.norm(k0_dir)

    omega_values = np.arange(omega_range[0], omega_range[1], omega_step)

    for h, k, l, d, g in reflections:
        for omega in omega_values:
            g_rot = rotate_vector_y(g, omega)
            lhs = 2 * np.dot(k0, g_rot)
            rhs = np.dot(g_rot, g_rot)
            if abs(lhs - rhs) < tol:
                predictions.append((h, k, l, omega))
                break  # assume only one omega per reflection is needed

    return predictions


# === Run Step 3 ===

# You must have 'reflections' from Step 2
predicted_bragg = predict_bragg_angles(generated_reflections)

print(f"\nPredicted Bragg angles for {len(predicted_bragg)} reflections:")
for h, k, l, omega in predicted_bragg[:10]:
    print(f"hkl=({h},{k},{l})  ω = {omega:.2f}°")

# === Step 4: Compare predicted vs observed ω angles ===

def match_reflections_to_observed(predicted, reflections_df, zone_axes_df):
    """
    Match predicted ω angles to observed ones for the same (h, k, l),
    and compute delta ω = ω_obs - ω_pred.

    Parameters:
        predicted: list of (h, k, l, omega_pred)
        reflections_df: DataFrame with 'h', 'k', 'l', 'zone_axis_id'
        zone_axes_df: DataFrame with 'id', 'omega'

    Returns:
        List of dicts with keys:
        {
            'h', 'k', 'l', 'omega_pred', 'omega_obs', 'delta_omega'
        }
    """
    results = []

    # Create quick lookup from zone ID to omega
    zone_to_omega = dict(zip(zone_axes_df['id'], zone_axes_df['omega']))

    for h_pred, k_pred, l_pred, omega_pred in predicted:
        # Filter experimental reflections for this hkl
        match = reflections_df[
            (reflections_df['h'] == h_pred) &
            (reflections_df['k'] == k_pred) &
            (reflections_df['l'] == l_pred)
        ]

        if match.empty:
            continue  # no match in experimental data

        # Use the first match (if multiple, could average)
        row = match.iloc[0]
        zone_id = row['zone_axis_id']
        omega_obs = zone_to_omega.get(zone_id, None)

        if omega_obs is None:
            continue  # missing omega

        delta = omega_obs - omega_pred
        results.append({
            'h': h_pred,
            'k': k_pred,
            'l': l_pred,
            'omega_pred': omega_pred,
            'omega_obs': omega_obs,
            'delta_omega': delta
        })

    return results


# === Run Step 4 ===

# You must have 'predicted_bragg' from Step 3
# and 'reflections', 'zone_axes' from Step 1

matched_results = match_reflections_to_observed(predicted_bragg, reflections, zone_axes)

print(f"\nMatched {len(matched_results)} reflections with observed ω values")
for item in matched_results[:5]:
    print(f"hkl=({item['h']},{item['k']},{item['l']}) | ω_pred={item['omega_pred']:.2f}°, ω_obs={item['omega_obs']:.2f}°, Δω={item['delta_omega']:.2f}°")

# === Step 5: Refine α and β tilt corrections per window ===

from scipy.optimize import minimize

def rotate_vector(vec, alpha_deg, beta_deg):
    """Rotate vector by alpha (x-axis) then beta (y-axis)."""
    a = np.radians(alpha_deg)
    b = np.radians(beta_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    Ry = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    return Ry @ Rx @ vec


def apply_tilt_and_predict(g, alpha, beta, k0=np.array([0, 0, 1]), omega_range=(-70, 70), omega_step=0.1, tol=0.01):
    """Apply tilt (α, β) to g and find new Bragg ω."""
    g_tilted = rotate_vector(g, alpha, beta)
    k0 = k0 / np.linalg.norm(k0)
    for omega in np.arange(*omega_range, omega_step):
        g_rot = rotate_vector_y(g_tilted, omega)
        lhs = 2 * np.dot(k0, g_rot)
        rhs = np.dot(g_rot, g_rot)
        if abs(lhs - rhs) < tol:
            return omega
    return None  # No solution found


def refine_tilts_per_window(matched_results, g_vectors_dict, window_width=50, step=10):
    """
    Divide data into overlapping ω_obs windows, and fit α/β in each.

    Parameters:
        matched_results: output from Step 4
        g_vectors_dict: {(h,k,l): g-vector}
        window_width: number of degrees in ω per window
        step: step size between windows

    Returns:
        List of {'center_omega', 'alpha', 'beta'}
    """
    tilts = []
    omega_vals = np.array([d['omega_obs'] for d in matched_results])
    omega_min, omega_max = min(omega_vals), max(omega_vals)

    for center in np.arange(omega_min, omega_max, step):
        omega_start = center - window_width / 2
        omega_end = center + window_width / 2

        # Select reflections in this window
        window_refls = [
            d for d in matched_results
            if omega_start <= d['omega_obs'] <= omega_end and (d['h'], d['k'], d['l']) in g_vectors_dict
        ]

        if len(window_refls) < 5:
            continue  # Not enough data

        def cost(params):
            alpha, beta = params
            error = 0
            for d in window_refls:
                g = g_vectors_dict[(d['h'], d['k'], d['l'])]
                omega_pred = apply_tilt_and_predict(g, alpha, beta)
                if omega_pred is not None:
                    error += (d['omega_obs'] - omega_pred) ** 2
            return error

        result = minimize(cost, x0=[0.0, 0.0], bounds=[(-2, 2), (-2, 2)])
        alpha_opt, beta_opt = result.x

        tilts.append({
            'center_omega': center,
            'alpha': alpha_opt,
            'beta': beta_opt
        })

    return tilts


# === Run Step 5 ===

# Build a lookup from hkl to g-vector
g_vectors = {
    (h, k, l): g
    for h, k, l, d, g in generated_reflections
}


# You must have matched_results from Step 4
tilt_fit = refine_tilts_per_window(matched_results, g_vectors)

print(f"\nComputed tilt corrections for {len(tilt_fit)} windows.")
for t in tilt_fit[:5]:
    print(f"ω = {t['center_omega']:.1f}°, α = {t['alpha']:.3f}°, β = {t['beta']:.3f}°")

# === Step 6: Plot beam path diagram (Figure 8-style) ===

import matplotlib.pyplot as plt

#def beam_path_diagram(reflections, omega_range=(-70, 70), omega_step=0.5):
    """
    Plot 2D beam path diagram using gnomonic projection of Bragg conditions.
    
    Parameters:
        reflections: list of (h, k, l, d, g_vector)
        omega_range: rotation angles to scan
        omega_step: step size in degrees
    """
    k0 = np.array([0, 0, 1])  # Incident beam along z

    fig, ax = plt.subplots(figsize=(8, 6))

    for h, k, l, d, g in reflections:
        x_list, omega_list = [], []
        for omega in np.arange(*omega_range, omega_step):
            g_rot = rotate_vector_y(g, omega)
            k_out = k0 + g_rot
            if k_out[2] == 0:
                continue  # avoid divide-by-zero
            x_proj = k_out[0] / k_out[2]  # Gnomonic x
            y_proj = k_out[1] / k_out[2]  # Gnomonic y

            # Check Bragg condition (approx)
            lhs = 2 * np.dot(k0, g_rot)
            rhs = np.dot(g_rot, g_rot)
            if abs(lhs - rhs) < 0.01:
                x_list.append(x_proj)
                omega_list.append(omega)

        if len(x_list) > 1:
            ax.plot(x_list, omega_list, lw=0.6, alpha=0.7)

    ax.set_title("Beam Path Diagram (Gnomonic Projection)")
    ax.set_xlabel("Gnomonic x (k_x / k_z)")
    ax.set_ylabel("Rotation angle ω (degrees)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# === Run Step 6 ===

beam_path_diagram(generated_reflections, omega_range=(-70, 70), omega_step=0.5)


#Next GPT request: make sure define what should be what in the beam diagram of Figure 8 or ensure that it has correctly
#interpreted what figure 8 is. perhaps can reuse much of GPT messages for BDF
#make sure for next GPT request ask to add comprehensive comments so that one not familiar with this code
#but knows the subject area and specific problem well. 