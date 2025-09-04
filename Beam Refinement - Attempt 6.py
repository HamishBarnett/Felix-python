# -*- coding: utf-8 -*-
"""
cRED rotation refinement: alpha/beta deviation vs rotation and sliding-window
estimation of (z0, x0) components from Bragg conditions.

Requirements: numpy, matplotlib, gemmi
Install gemmi if needed: pip install gemmi
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
try:
    import gemmi
except Exception as e:
    raise RuntimeError(
        "This script requires the 'gemmi' package to read CIF files.\n"
        "Install it with: pip install gemmi\n"
        f"Import error was: {e}"
    )


# ----------------------------- Utility functions -----------------------------

def deg2rad(deg: np.ndarray | float) -> np.ndarray | float:
    return np.deg2rad(deg)

def rad2deg(rad: np.ndarray | float) -> np.ndarray | float:
    return np.rad2deg(rad)

def normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def wrap_degrees(delta_deg: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180]."""
    out = (delta_deg + 180.0) % 360.0 - 180.0
    return out

def safe_arcsin(x: np.ndarray | float) -> np.ndarray | float:
    return np.arcsin(np.clip(x, -1.0, 1.0))

def gram_schmidt_frame(x0: np.ndarray, z0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormalize initial guesses x0, z0 into a right-handed frame (x0, y, z0).
    Returns (x0_hat, y_hat, z0_hat) with y = x0 × z0.
    """
    x = normalize(x0.reshape(3))
    z = z0.reshape(3) - np.dot(z0, x) * x
    z = normalize(z)
    y = np.cross(x, z)
    y = normalize(y)
    # Rebuild x as y × z to guarantee right-handedness and orthogonality
    x = np.cross(y, z)
    x = normalize(x)
    return x, y, z

def rotation_model_vectors(alpha_rad: np.ndarray | float,
                           x0: np.ndarray, y: np.ndarray, z0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given alpha angle(s) and orthonormal (x0, y, z0), return z(alpha) and x(alpha)
    from z = z0 cos(a) + x0 sin(a), x = x0 cos(a) - z0 sin(a).
    """
    a = np.array(alpha_rad, dtype=float)
    ca, sa = np.cos(a), np.sin(a)
    # broadcasting-friendly
    z = np.outer(ca, z0) + np.outer(sa, x0) if a.ndim else ca * z0 + sa * x0
    x = np.outer(ca, x0) - np.outer(sa, z0) if a.ndim else ca * x0 - sa * z0
    return z, x

def solve_z0_x0_by_lstsq(alpha_rad: np.ndarray, z_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve z_meas ≈ z0 cos(a) + x0 sin(a) for z0, x0 (each 3,) via linear least squares.
    Returns (x0, z0, rms_residual)
    """
    n = len(alpha_rad)
    I3 = np.eye(3)
    blocks = []
    rhs = []
    for i in range(n):
        ca = math.cos(alpha_rad[i])
        sa = math.sin(alpha_rad[i])
        A_i = np.hstack([ca * I3, sa * I3])  # 3x6
        blocks.append(A_i)
        rhs.append(z_meas[i])
    A = np.vstack(blocks)       # (3n x 6)
    b = np.vstack(rhs).reshape(3*n)  # (3n,)
    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    z0 = sol[:3]
    x0 = sol[3:]
    # Compute residual RMS
    pred = (A @ sol).reshape(n, 3)
    err = z_meas - pred
    rms = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    return x0, z0, rms

def svd_null_vector(A: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Solve A v ≈ 0 (non-trivial v) using SVD. Optionally apply weights per row.
    Returns the right singular vector corresponding to smallest singular value.
    """
    if weights is not None:
        w = np.sqrt(np.clip(weights, 1e-12, None))
        A = (A.T * w).T
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    v = VT[-1, :]
    # scale normalization (optional)
    return v / np.linalg.norm(v)

def plane_normal_and_theta(hkl: Tuple[int, int, int],
                           UB: np.ndarray,
                           unit_cell: gemmi.UnitCell,
                           wavelength_A: float) -> Tuple[np.ndarray, float, float]:
    """
    Compute plane normal vector (lab frame) and Bragg angle theta for an hkl.
    n_hat is along UB @ hkl (reciprocal direction) normalized.
    theta from d-spacing via unit cell and wavelength.
    Returns (n_hat, theta_rad, d_spacing_A).
    """
    h, k, l = hkl
    g = UB @ np.array([h, k, l], dtype=float)  # reciprocal direction in lab frame
    n_hat = normalize(g)
    # d-spacing from unit cell
    d = unit_cell.calculate_d([h, k, l])
    # Guard against invalid d (e.g., forbidden/zero)
    if not np.isfinite(d) or d <= 0.0:
        return n_hat, np.nan, np.nan
    arg = wavelength_A / (2.0 * d)
    if arg > 1.0:
        # beyond Ewald sphere (no Bragg)
        return n_hat, np.nan, d
    theta = math.asin(np.clip(arg, 0.0, 1.0))
    return n_hat, theta, d

def atan2_from_frame(z_vec: np.ndarray, x0: np.ndarray, z0: np.ndarray) -> float:
    """Compute modeled alpha_hat = atan2(z·x0, z·z0) (radians)."""
    return math.atan2(float(np.dot(z_vec, x0)), float(np.dot(z_vec, z0)))

def weighted_centroid_angle(alphas_deg: np.ndarray,
                            weights: np.ndarray) -> float:
    """
    Intensity-weighted centroid of alpha (degrees) robust to small gaps.
    Assumes alphas lie within a limited range (< 180° span), so no wrap issues.
    """
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if np.sum(w) <= 0.0:
        return float(np.mean(alphas_deg))
    return float(np.sum(alphas_deg * w) / np.sum(w))

def robust_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float('nan')

def robust_int(x: str) -> int:
    try:
        return int(x)
    except Exception:
        return 0


# -------------------------- CIF parsing (with gemmi) --------------------------

def read_cif_block(path: str) -> gemmi.cif.Block:
    doc = gemmi.cif.read_file(path)
    if len(doc) == 0:
        raise RuntimeError(f"No CIF blocks found in file: {path}")
    return doc[0]

def extract_unit_cell_and_wavelength(block: gemmi.cif.Block) -> Tuple[gemmi.UnitCell, Optional[float]]:
    # Unit cell
    cell = block.get_mmcif_cell()
    if cell is None:
        # Fallback manual
        a = robust_float(block.find_value('_cell_length_a'))
        b = robust_float(block.find_value('_cell_length_b'))
        c = robust_float(block.find_value('_cell_length_c'))
        alpha = robust_float(block.find_value('_cell_angle_alpha'))
        beta = robust_float(block.find_value('_cell_angle_beta'))
        gamma = robust_float(block.find_value('_cell_angle_gamma'))
        if any(map(lambda v: not np.isfinite(v) or v <= 0.0, [a, b, c])) or any(
                map(lambda v: not np.isfinite(v), [alpha, beta, gamma])):
            raise RuntimeError("Could not read unit cell from CIF.")
        unit_cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
    else:
        unit_cell = cell

    # Wavelength (Å)
    wv_tags = [
        '_diffrn_radiation_wavelength',
        '_diffrn_radiation_wavelength_1',
        '_beam_wavelength',
        '_diffrn_source_wavelength',
        '_diffrn_radiation_wavelength_nm'  # if nm, will convert below
    ]
    wav = None
    for tag in wv_tags:
        val = block.find_value(tag)
        if val:
            try:
                wav = float(val)
                if tag.endswith('_nm'):
                    wav *= 10.0  # nm -> Å
                break
            except Exception:
                continue
    return unit_cell, wav

def extract_UB_matrix(block: gemmi.cif.Block) -> Optional[np.ndarray]:
    """
    Try to read UB matrix from common tag patterns.
    Returns 3x3 matrix or None if missing.
    """
    # Common PETS tags use _diffrn_orient_matrix_UB_ij
    candidates = []
    for i in range(1, 4):
        row = []
        for j in range(1, 4):
            for key in [
                f'_diffrn_orient_matrix_UB_{i}{j}',
                f'_diffrn_orient_matrix_ub_{i}{j}',
                f'_diffrn_orient_matrix_UB_{i}{j:1d}',
                f'_diffrn_orient_matrix_ub_{i}{j:1d}',
                f'_diffrn_orient_UB_{i}{j}',
                f'_diffrn_orient_ub_{i}{j}',
            ]:
                val = block.find_value(key)
                if val:
                    row.append(robust_float(val))
                    break
            else:
                row.append(np.nan)
        candidates.append(row)
    M = np.array(candidates, dtype=float)
    if np.all(np.isfinite(M)):
        return M
    # Try a looped UB (rare)
    try:
        loop = block.find_loop('_diffrn_orient_matrix_UB_11')
        if loop:
            vals = [robust_float(v) for v in loop.get_values('_diffrn_orient_matrix_UB_11',
                                                             '_diffrn_orient_matrix_UB_12',
                                                             '_diffrn_orient_matrix_UB_13',
                                                             '_diffrn_orient_matrix_UB_21',
                                                             '_diffrn_orient_matrix_UB_22',
                                                             '_diffrn_orient_matrix_UB_23',
                                                             '_diffrn_orient_matrix_UB_31',
                                                             '_diffrn_orient_matrix_UB_32',
                                                             '_diffrn_orient_matrix_UB_33')]
            M = np.array(vals, dtype=float).reshape(3, 3)
            if np.all(np.isfinite(M)):
                return M
    except Exception:
        pass
    return None

def extract_zone_axis_series(block: gemmi.cif.Block) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read zone-axis loop: ids, alpha_deg, z_vectors(u,v,w).
    Returns (ids:int, alphas_deg:float, z_hat:float[N,3])
    """
    # Likely loop tags
    tag_sets = [
        ('_diffrn_zone_axis_id',
         '_diffrn_zone_axis_u',
         '_diffrn_zone_axis_v',
         '_diffrn_zone_axis_w',
         '_diffrn_zone_axis_alpha'),
        # Alternate names (rare)
        ('_zone_axis_id',
         '_zone_axis_u', '_zone_axis_v', '_zone_axis_w', '_zone_axis_alpha'),
    ]
    loop = None
    for id_tag, u_tag, v_tag, w_tag, a_tag in tag_sets:
        try:
            loop = block.find_loop(id_tag)
            if loop and all(loop.has_tag(t) for t in [id_tag, u_tag, v_tag, w_tag, a_tag]):
                ids = [robust_int(x) for x in loop.get_values(id_tag)]
                u = [robust_float(x) for x in loop.get_values(u_tag)]
                v = [robust_float(x) for x in loop.get_values(v_tag)]
                w = [robust_float(x) for x in loop.get_values(w_tag)]
                a = [robust_float(x) for x in loop.get_values(a_tag)]
                ids = np.array(ids, dtype=int)
                alphas_deg = np.array(a, dtype=float)
                z = np.vstack([u, v, w]).T
                z = normalize(z)
                return ids, alphas_deg, z
        except Exception:
            continue
    raise RuntimeError("Could not find a zone-axis loop with (id,u,v,w,alpha) tags in the PETS CIF.")

def extract_reflections(block: gemmi.cif.Block) -> Dict[str, np.ndarray]:
    """
    Read reflections with h,k,l, intensity (or F^2), sigma (if present), and zone_axis_id (frame id).
    Returns dict with numpy arrays.
    """
    # Common tag variants
    tag_variants = [
        ('_refln_index_h', '_refln_index_k', '_refln_index_l',
         '_refln_intensity', '_refln_intensity_sigma', '_refln_zone_axis_id'),
        ('_refln_index_h', '_refln_index_k', '_refln_index_l',
         '_refln_F_squared_meas', '_refln_F_squared_sigma', '_refln_zone_axis_id'),
        ('_refln_index_h', '_refln_index_k', '_refln_index_l',
         '_refln_intensity_net', '_refln_intensity_sigma', '_refln_zone_axis_id'),
    ]
    for tags in tag_variants:
        try:
            loop = block.find_loop(tags[0])
            if not loop:
                continue
            if not all(loop.has_tag(t) for t in tags):
                continue
            h = np.array([robust_int(x) for x in loop.get_values(tags[0])], dtype=int)
            k = np.array([robust_int(x) for x in loop.get_values(tags[1])], dtype=int)
            l = np.array([robust_int(x) for x in loop.get_values(tags[2])], dtype=int)
            I = np.array([robust_float(x) for x in loop.get_values(tags[3])], dtype=float)
            sig = np.array([robust_float(x) for x in loop.get_values(tags[4])], dtype=float)
            zid = np.array([robust_int(x) for x in loop.get_values(tags[5])], dtype=int)
            return {'h': h, 'k': k, 'l': l, 'I': I, 'sigma': sig, 'zone_id': zid}
        except Exception:
            continue
    raise RuntimeError("Could not find a reflections loop with (h,k,l,intensity,sigma,zone_id).")


# -------------------------- Core analysis functions --------------------------

def fit_initial_frame_from_zone_axis(alphas_deg: np.ndarray,
                                     z_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit z(alpha) ≈ z0 cos a + x0 sin a, then orthonormalize to get (x0, y, z0).
    Returns (x0, y, z0, rms_residual)
    """
    a_rad = deg2rad(alphas_deg)
    x0_guess, z0_guess, rms = solve_z0_x0_by_lstsq(a_rad, z_meas)
    x0, y, z0 = gram_schmidt_frame(x0_guess, z0_guess)
    return x0, y, z0, rms

def compute_deviation_series(alphas_deg: np.ndarray,
                             z_meas: np.ndarray,
                             x0: np.ndarray, y: np.ndarray, z0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each frame i: beta_i = asin(z_i · y), and Δalpha_i = wrap(atan2(z_i·x0, z_i·z0) - alpha_i).
    Returns (delta_alpha_deg, beta_deg)
    """
    # beta (radians)
    beta_rad = safe_arcsin(np.einsum('ij,j->i', z_meas, y))
    beta_deg = rad2deg(beta_rad)
    # alpha hat from measured z and fitted (x0,z0)
    a_hat_rad = np.array([atan2_from_frame(z_meas[i], x0, z0) for i in range(len(z_meas))])
    delta_deg = wrap_degrees(rad2deg(a_hat_rad) - alphas_deg)
    return delta_deg, beta_deg

def build_hkl_alpha_records(reflections: Dict[str, np.ndarray],
                            frame_ids: np.ndarray,
                            alphas_deg: np.ndarray,
                            min_snr: float = 0.0,
                            use_centroid: bool = True) -> List[Dict]:
    """
    For each unique (h,k,l), compute representative alpha_hkl.
    If use_centroid: intensity-weighted centroid across its frames; otherwise: max-intensity frame.
    Returns list of dicts with keys: h,k,l, alpha_deg, I_sum, snr_mean
    """
    # Map frame id -> alpha
    id_to_alpha = {int(fid): float(a) for fid, a in zip(frame_ids, alphas_deg)}
    # Build keyed groups
    h, k, l = reflections['h'], reflections['k'], reflections['l']
    I = reflections['I']
    sig = reflections['sigma']
    zid = reflections['zone_id']
    # optional SNR filter per reflection
    valid = np.ones_like(I, dtype=bool)
    if min_snr > 0.0:
        snr = np.divide(I, sig, out=np.zeros_like(I), where=(sig > 0))
        valid &= (snr >= min_snr)
    h, k, l, I, sig, zid = h[valid], k[valid], l[valid], I[valid], sig[valid], zid[valid]

    # group by hkl
    from collections import defaultdict
    groups = defaultdict(list)
    for hh, kk, ll, ii, ss, zz in zip(h, k, l, I, sig, zid):
        if zz in id_to_alpha:
            groups[(hh, kk, ll)].append((id_to_alpha[zz], ii, ss))

    records = []
    for hkl, rows in groups.items():
        if not rows:
            continue
        alphas = np.array([r[0] for r in rows], dtype=float)
        intens = np.array([r[1] for r in rows], dtype=float)
        sigmas = np.array([r[2] for r in rows], dtype=float)
        if use_centroid:
            a_hkl = weighted_centroid_angle(alphas, intens)
        else:
            idx = int(np.argmax(intens))
            a_hkl = float(alphas[idx])
        I_sum = float(np.sum(np.clip(intens, 0.0, None)))
        snr_mean = float(np.mean(np.divide(intens, sigmas, out=np.zeros_like(intens), where=(sigmas > 0))))
        records.append({'h': hkl[0], 'k': hkl[1], 'l': hkl[2],
                        'alpha_deg': a_hkl, 'I_sum': I_sum, 'snr_mean': snr_mean})
    # sort by alpha
    records.sort(key=lambda r: r['alpha_deg'])
    return records

def compute_phi_list(records: List[Dict],
                     UB: np.ndarray,
                     unit_cell: gemmi.UnitCell,
                     wavelength_A: float,
                     x0: np.ndarray, y: np.ndarray, z0: np.ndarray) -> List[Dict]:
    """
    For each hkl record with alpha_hkl, compute (n_hat, theta, phi) where
    phi = theta / (n·x(alpha_hkl)), and x(alpha) from the fitted frame (x0,y,z0).
    Skips ill-conditioned cases where |n·x| < tol or theta is NaN.
    Returns list of dicts with keys: h,k,l, alpha_deg, phi_rad, n_hat(3,), theta_rad, I_sum, snr_mean
    """
    out = []
    for rec in records:
        hkl = (rec['h'], rec['k'], rec['l'])
        n_hat, theta, dA = plane_normal_and_theta(hkl, UB, unit_cell, wavelength_A)
        if not np.isfinite(theta):
            continue
        a_rad = deg2rad(rec['alpha_deg'])
        # model x(alpha)
        _, x_alpha = rotation_model_vectors(a_rad, x0, y, z0)
        x_alpha = x_alpha.reshape(3)
        denom = float(np.dot(n_hat, x_alpha))
        if abs(denom) < 1e-6:
            continue
        phi = theta / denom
        # Limit absurdly large tilt magnitudes to avoid numerical silliness (optional)
        if abs(phi) > np.deg2rad(15.0):  # 15 degrees cap
            continue
        out.append({
            'h': rec['h'], 'k': rec['k'], 'l': rec['l'],
            'alpha_deg': rec['alpha_deg'],
            'phi_rad': float(phi),
            'n_hat': n_hat.reshape(3),
            'theta_rad': float(theta),
            'I_sum': rec['I_sum'],
            'snr_mean': rec['snr_mean']
        })
    return out

def sliding_window_svd_six_components(phi_list: List[Dict],
                                      window_size: int = 24,
                                      step: int = 6,
                                      weight_by_intensity: bool = True) -> List[Dict]:
    """
    Build the linear system for consecutive subsets of reflections:
    [ n cos(alpha-phi) | n sin(alpha-phi) ] · [z0; x0] = 0
    Solve by SVD nullspace. Return list of dicts with alpha_center_deg, z0_vec, x0_vec, y_vec, cond, n_rows.
    """
    if len(phi_list) < 6:
        return []

    # Sort by alpha
    phi_list_sorted = sorted(phi_list, key=lambda r: r['alpha_deg'])
    alphas_deg = np.array([r['alpha_deg'] for r in phi_list_sorted], dtype=float)
    nvecs = np.vstack([r['n_hat'] for r in phi_list_sorted])  # (M,3)
    phis = np.array([r['phi_rad'] for r in phi_list_sorted], dtype=float)
    intens = np.array([r['I_sum'] for r in phi_list_sorted], dtype=float)

    out = []
    M = len(phi_list_sorted)
    idx = 0
    while idx < M:
        j0 = idx
        j1 = min(M, j0 + window_size)
        mask = slice(j0, j1)
        n_sub = nvecs[mask]        # (m,3)
        a_sub = alphas_deg[mask]   # (m,)
        p_sub = phis[mask]         # (m,)
        m = len(a_sub)
        if m >= 6:
            ca = np.cos(deg2rad(a_sub) - p_sub)
            sa = np.sin(deg2rad(a_sub) - p_sub)
            # Build A: (m x 6); row i: [n_i * ca_i | n_i * sa_i]
            A = np.hstack([n_sub * ca[:, None], n_sub * sa[:, None]])
            w = intens[mask] if weight_by_intensity else None

            v = svd_null_vector(A, w)
            z0_raw = v[:3]
            x0_raw = v[3:]
            # Orthonormalize to get right-handed frame
            x0, y, z0 = gram_schmidt_frame(x0_raw, z0_raw)
            # condition indicator = ratio of smallest to second-smallest singular values
            # (smaller means better-defined nullspace); compute quickly
            if w is not None:
                Aw = (A.T * np.sqrt(np.clip(w, 1e-12, None))).T
                S = np.linalg.svd(Aw, compute_uv=False)
            else:
                S = np.linalg.svd(A, compute_uv=False)
            cond = float(S[-1] / S[-2]) if len(S) >= 2 and S[-2] > 0 else float('nan')

            out.append({
                'alpha_center_deg': float(np.mean(a_sub)),
                'z0x': float(z0[0]), 'z0y': float(z0[1]), 'z0z': float(z0[2]),
                'x0x': float(x0[0]), 'x0y': float(x0[1]), 'x0z': float(x0[2]),
                'y0x': float(y[0]),  'y0y': float(y[1]),  'y0z': float(y[2]),
                'n_rows': int(m),
                'cond': cond
            })
        idx += step

    return out

def save_six_components_csv(series: List[Dict], path: str) -> None:
    headers = ['alpha_center_deg',
               'z0x','z0y','z0z',
               'x0x','x0y','x0z',
               'y0x','y0y','y0z',
               'n_rows','cond']
    with open(path, 'w', newline='') as f:
        f.write(','.join(headers) + '\n')
        for rec in series:
            row = [rec.get(h, float('nan')) for h in headers]
            f.write(','.join(str(x) for x in row) + '\n')

def plot_deviation_series(alphas_deg: np.ndarray,
                          delta_alpha_deg: np.ndarray,
                          beta_deg: np.ndarray,
                          out_png: Optional[str] = None) -> None:
    plt.figure(figsize=(9.5, 5.5))
    plt.plot(alphas_deg, delta_alpha_deg, label=r'$\Delta \alpha$ (deg)')
    plt.plot(alphas_deg, beta_deg, label=r'$\beta$ (deg)')
    plt.xlabel('Nominal rotation α (deg)')
    plt.ylabel('Deviation (deg)')
    plt.title('Rotation deviations vs α')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    if out_png:
        plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.show()


# ------------------------------- Main pipeline --------------------------------

def run_pipeline(pets_cif_path: str,
                 ref_cif_path: Optional[str],
                 min_snr_reflections: float = 0.0,
                 window_size: int = 24,
                 window_step: int = 6,
                 weight_by_intensity: bool = True,
                 out_csv_path: str = 'z0_x0_six_components_series.csv',
                 out_png_path: str = 'alpha_beta_deviation.png') -> Dict:
    """
    End-to-end execution. Returns a dict with key results.
    """
    # Read PETS CIF
    pets_block = read_cif_block(pets_cif_path)
    unit_cell_pets, wavelength = extract_unit_cell_and_wavelength(pets_block)
    if wavelength is None or not np.isfinite(wavelength):
        raise RuntimeError("Could not read wavelength from PETS CIF; please ensure it has a wavelength in Å.")
    UB = extract_UB_matrix(pets_block)
    if UB is None:
        raise RuntimeError("Could not find UB matrix in PETS CIF; required to compute plane normals.")

    # Zone-axis series
    frame_ids, alphas_deg, z_meas = extract_zone_axis_series(pets_block)

    # Reflections
    refl = extract_reflections(pets_block)

    # (Optional) Reference CIF unit cell for cross-check; we'll keep PETS cell for θ
    ref_cell = None
    if ref_cif_path:
        try:
            ref_block = read_cif_block(ref_cif_path)
            ref_cell, _ = extract_unit_cell_and_wavelength(ref_block)
        except Exception:
            ref_cell = None

    # Fit initial frame from measured z(alpha)
    x0, y, z0, rms = fit_initial_frame_from_zone_axis(alphas_deg, z_meas)

    # Deviation series
    delta_alpha_deg, beta_deg = compute_deviation_series(alphas_deg, z_meas, x0, y, z0)
    plot_deviation_series(alphas_deg, delta_alpha_deg, beta_deg, out_png=out_png_path)

    # Build alpha_hkl for each plane
    records = build_hkl_alpha_records(refl, frame_ids, alphas_deg,
                                      min_snr=min_snr_reflections,
                                      use_centroid=True)

    # Compute phi for these planes
    phi_list = compute_phi_list(records, UB, unit_cell_pets, wavelength, x0, y, z0)

    # Sliding-window SVD to get six components along rotation series
    six_series = sliding_window_svd_six_components(phi_list,
                                                   window_size=window_size,
                                                   step=window_step,
                                                   weight_by_intensity=weight_by_intensity)
    # Save CSV
    save_six_components_csv(six_series, out_csv_path)

    # Return a compact result bundle
    return {
        'unit_cell_pets': unit_cell_pets,   # gemmi.UnitCell
        'wavelength_A': wavelength,
        'UB': UB,
        'x0': x0, 'y': y, 'z0': z0,
        'fit_rms': rms,
        'alphas_deg': alphas_deg,
        'delta_alpha_deg': delta_alpha_deg,
        'beta_deg': beta_deg,
        'phi_count': len(phi_list),
        'six_series_csv': out_csv_path
    }


# --------------------------------- Execution ----------------------------------

if __name__ == '__main__':
    # Update these paths as needed in your environment (defaults to your uploaded filenames)
    PETS_CIF_PATH = r'/mnt/data/Si_3_dyn.cif_pets'
    REF_CIF_PATH = r'/mnt/data/silicon_structure.cif'  # optional but useful

    # Tunable parameters
    MIN_SNR_REFLECTIONS = 0.0     # e.g., 2.0 to filter weak reflections
    WINDOW_SIZE = 24              # number of reflections per SVD window (>=6)
    WINDOW_STEP = 6               # stride between windows
    WEIGHT_BY_INTENSITY = True    # weight equations by summed intensity
    OUT_CSV = 'z0_x0_six_components_series.csv'
    OUT_PNG = 'alpha_beta_deviation.png'

    results = run_pipeline(
        pets_cif_path=PETS_CIF_PATH,
        ref_cif_path=REF_CIF_PATH,
        min_snr_reflections=MIN_SNR_REFLECTIONS,
        window_size=WINDOW_SIZE,
        window_step=WINDOW_STEP,
        weight_by_intensity=WEIGHT_BY_INTENSITY,
        out_csv_path=OUT_CSV,
        out_png_path=OUT_PNG
    )

    # Minimal console summary
    print("\n=== Summary ===")
    print(f"Cell (PETS): a,b,c,α,β,γ = {results['unit_cell_pets']}")
    print(f"Wavelength (Å): {results['wavelength_A']:.6f}")
    print(f"Initial fit RMS error on z(alpha): {results['fit_rms']:.3e}")
    print(f"Frames: {len(results['alphas_deg'])}")
    print(f"φ entries kept: {results['phi_count']}")
    print(f"Six-components CSV: {results['six_series_csv']}")
