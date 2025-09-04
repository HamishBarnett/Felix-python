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
import re
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

def _parse_loop_from_text(cif_path: str, required_tags: List[str]) -> Optional[Dict[str, List[str]]]:
    """
    Very simple CIF loop parser for numeric tables.
    Scans the file for a loop_ that contains all required_tags (any order),
    then returns a dict[tag] -> list of string values. Handles multi-row lines.
    """
    with open(cif_path, 'r', encoding='utf-8', errors='ignore') as fh:
        lines = fh.readlines()

    i, n = 0, len(lines)
    while i < n:
        if lines[i].lstrip().startswith('loop_'):
            # collect tag lines
            j = i + 1
            tags = []
            while j < n and lines[j].lstrip().startswith('_'):
                tags.append(lines[j].split()[0].strip())
                j += 1
            if not tags:
                i = j
                continue

            # if this loop has all required tags, parse its data block
            if all(t in tags for t in required_tags):
                tag_to_idx = {t: k for k, t in enumerate(tags)}
                # collect data lines until next loop_/data_/tag
                data_tokens = []
                k = j
                while k < n:
                    s = lines[k].strip()
                    if (not s) or s.startswith('#'):
                        k += 1
                        continue
                    if s.startswith('loop_') or s.startswith('data_') or s.startswith('_'):
                        break
                    data_tokens.extend(s.split())
                    k += 1
                # split tokens into rows of len(tags)
                rows = []
                m = len(tags)
                p = 0
                while p + m <= len(data_tokens):
                    rows.append(data_tokens[p:p+m])
                    p += m
                if not rows:
                    return None
                # build result dict for just the required tags
                result = {t: [] for t in required_tags}
                for row in rows:
                    for t in required_tags:
                        result[t].append(row[tag_to_idx[t]])
                return result
            i = j
        else:
            i += 1
    return None

def detect_alpha_sense(alphas_deg: np.ndarray,
                       z_meas: np.ndarray,
                       x0: np.ndarray,
                       z0: np.ndarray) -> Tuple[int, float, float]:
    """
    Compare modeled alpha_hat from measured z against nominal alphas.
    Returns (sense, slope, intercept) where sense is +1 (ok) or -1 (flip).
    """
    a_hat_deg = np.rad2deg(np.array(
        [atan2_from_frame(z_meas[i], x0, z0) for i in range(len(z_meas))]
    ))
    A = np.vstack([alphas_deg, np.ones_like(alphas_deg)]).T
    slope, intercept = np.linalg.lstsq(A, a_hat_deg, rcond=None)[0]
    sense = 1 if slope >= 0 else -1
    return sense, float(slope), float(intercept)

def orthonormalize_keep_signs(x0: np.ndarray, z0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormalize while preserving the signs of x0 and z0 that the caller chose.
    Projects x0 to be orthogonal to z0, normalizes both, and builds y = x0 × z0.
    """
    z = normalize(z0.reshape(3))
    # project x onto plane perpendicular to z
    x_tmp = x0.reshape(3) - np.dot(x0, z) * z
    if np.linalg.norm(x_tmp) < 1e-12:
        # degenerate case: pick any vector not parallel to z
        t = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(t, z)) > 0.9:
            t = np.array([0.0, 1.0, 0.0])
        x_tmp = t - np.dot(t, z) * z
    x = normalize(x_tmp)
    y = normalize(np.cross(x, z))
    return x, y, z


# -------------------------- CIF parsing (with gemmi) --------------------------

def read_cif_block(path: str) -> gemmi.cif.Block:
    doc = gemmi.cif.read_file(path)
    if len(doc) == 0:
        raise RuntimeError(f"No CIF blocks found in file: {path}")
    return doc[0]

def extract_unit_cell_and_wavelength(block: gemmi.cif.Block) -> Tuple[gemmi.UnitCell, Optional[float]]:
    """
    Read unit cell from standard CIF tags (_cell_length_* and _cell_angle_*)
    and wavelength from common diffraction tags. This avoids using
    block.get_mmcif_cell(), which some gemmi builds don’t have.
    Also strips crystallographic uncertainty notation like '5.431(2)'.
    """
    def get_float_from_tag(tag: str) -> float:
        s = block.find_value(tag)
        if not s:
            return float('nan')
        # strip e.s.d., e.g. '5.431(2)' -> '5.431'
        s = re.sub(r'\(.*\)$', '', s.strip())
        try:
            return float(s)
        except Exception:
            return float('nan')

    a = get_float_from_tag('_cell_length_a')
    b = get_float_from_tag('_cell_length_b')
    c = get_float_from_tag('_cell_length_c')
    alpha = get_float_from_tag('_cell_angle_alpha')
    beta  = get_float_from_tag('_cell_angle_beta')
    gamma = get_float_from_tag('_cell_angle_gamma')

    if not all(np.isfinite([a, b, c, alpha, beta, gamma])):
        raise RuntimeError("Could not read unit cell from CIF (_cell_length_* and _cell_angle_* tags).")

    unit_cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)

    # Wavelength (Å), try several common tags
    wav = None
    for tag in [
        '_diffrn_radiation_wavelength',
        '_diffrn_radiation_wavelength_1',
        '_beam_wavelength',
        '_diffrn_source_wavelength',
        '_diffrn_radiation_wavelength_nm'  # will convert nm->Å
    ]:
        val = block.find_value(tag)
        if not val:
            continue
        s = re.sub(r'\(.*\)$', '', val.strip())
        try:
            wav = float(s)
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

def extract_zone_axis_series(block: gemmi.cif.Block,
                             cif_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read zone-axis loop: ids, alpha_deg, and z_hat from (u,v,w).
    Tries gemmi APIs first; falls back to plain-text parsing if needed.
    """
    required = ['_diffrn_zone_axis_id',
                '_diffrn_zone_axis_u',
                '_diffrn_zone_axis_v',
                '_diffrn_zone_axis_w',
                '_diffrn_zone_axis_alpha']

    # --- Try gemmi (find_loop + get_values) ---
    try:
        lp = block.find_loop('_diffrn_zone_axis_id')
        if lp is not None:
            tags = [str(t) for t in getattr(lp, 'tags', [])]
            # If 'tags' isn’t exposed, we’ll still try get_values directly.
            def vals(tag):
                try:
                    return lp.get_values(tag)
                except Exception:
                    # Fallback if get_values not present
                    if tags and tag in tags:
                        j = tags.index(tag)
                        return [row[j] for row in lp]
                    raise

            ids = [robust_int(x) for x in vals('_diffrn_zone_axis_id')]
            u   = [robust_float(x) for x in vals('_diffrn_zone_axis_u')]
            v   = [robust_float(x) for x in vals('_diffrn_zone_axis_v')]
            w   = [robust_float(x) for x in vals('_diffrn_zone_axis_w')]
            a   = [robust_float(x) for x in vals('_diffrn_zone_axis_alpha')]

            ids = np.array(ids, dtype=int)
            alphas_deg = np.array(a, dtype=float)
            z = np.vstack([u, v, w]).T
            z = normalize(z)
            return ids, alphas_deg, z
    except Exception:
        pass

    # --- Fallback: parse the CIF text directly ---
    table = _parse_loop_from_text(cif_path, required)
    if table is None:
        raise RuntimeError("Could not find a zone-axis loop with tags (id,u,v,w,alpha) in the PETS CIF.")

    ids = np.array([robust_int(x) for x in table['_diffrn_zone_axis_id']], dtype=int)
    alphas_deg = np.array([robust_float(x) for x in table['_diffrn_zone_axis_alpha']], dtype=float)
    u = np.array([robust_float(x) for x in table['_diffrn_zone_axis_u']], dtype=float)
    v = np.array([robust_float(x) for x in table['_diffrn_zone_axis_v']], dtype=float)
    w = np.array([robust_float(x) for x in table['_diffrn_zone_axis_w']], dtype=float)
    z = normalize(np.vstack([u, v, w]).T)
    return ids, alphas_deg, z

def extract_reflections(block: gemmi.cif.Block,
                        cif_path: str) -> Dict[str, np.ndarray]:
    """
    Read reflections: h,k,l, intensity (or F^2), sigma, and zone_axis_id.
    Tries gemmi first; falls back to plain-text parsing. Supports PETS tags.
    """
    base_tags = ['_refln_index_h', '_refln_index_k', '_refln_index_l', '_refln_zone_axis_id']
    variants = [
        ('_refln_intensity_meas', '_refln_intensity_sigma'),
        ('_refln_intensity',      '_refln_intensity_sigma'),
        ('_refln_F_squared_meas', '_refln_F_squared_sigma'),
        ('_refln_intensity_net',  '_refln_intensity_sigma'),
    ]

    # --- Try gemmi ---
    try:
        # Find a loop that has h,k,l,zone plus any of the intensity/sigma pairs
        for I_tag, S_tag in variants:
            lp = block.find_loop('_refln_index_h')
            if lp is None:
                continue
            tags = [str(t) for t in getattr(lp, 'tags', [])]
            if not all(t in tags for t in base_tags + [I_tag, S_tag]):
                # Some PETS exports split reflections across loops; try text fallback later
                continue

            def vals(tag):
                try:
                    return lp.get_values(tag)
                except Exception:
                    if tags and tag in tags:
                        j = tags.index(tag)
                        return [row[j] for row in lp]
                    raise

            H = np.array([robust_int(x) for x in vals('_refln_index_h')], dtype=int)
            K = np.array([robust_int(x) for x in vals('_refln_index_k')], dtype=int)
            L = np.array([robust_int(x) for x in vals('_refln_index_l')], dtype=int)
            Z = np.array([robust_int(x) for x in vals('_refln_zone_axis_id')], dtype=int)
            I = np.array([robust_float(x) for x in vals(I_tag)], dtype=float)
            S = np.array([robust_float(x) for x in vals(S_tag)], dtype=float)

            return {'h': H, 'k': K, 'l': L, 'I': I, 'sigma': S, 'zone_id': Z}
    except Exception:
        pass

    # --- Fallback: parse the CIF text directly ---
    # Find the first loop that contains base_tags plus one of the intensity/sigma pairs
    for I_tag, S_tag in variants:
        req = base_tags + [I_tag, S_tag]
        table = _parse_loop_from_text(cif_path, req)
        if table is not None:
            H = np.array([robust_int(x) for x in table['_refln_index_h']], dtype=int)
            K = np.array([robust_int(x) for x in table['_refln_index_k']], dtype=int)
            L = np.array([robust_int(x) for x in table['_refln_index_l']], dtype=int)
            Z = np.array([robust_int(x) for x in table['_refln_zone_axis_id']], dtype=int)
            I = np.array([robust_float(x) for x in table[I_tag]], dtype=float)
            S = np.array([robust_float(x) for x in table[S_tag]], dtype=float)
            return {'h': H, 'k': K, 'l': L, 'I': I, 'sigma': S, 'zone_id': Z}

    raise RuntimeError("Could not find a reflections loop with h,k,l,intensity,sigma,zone_id tags.")


# -------------------------- Core analysis functions --------------------------

def fit_initial_frame_from_zone_axis(alphas_deg: np.ndarray,
                                     z_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit z(alpha) ≈ z0 cos a + x0 sin a, then orthonormalize to get (x0, y, z0).
    Disambiguate signs so that z·x0 ~ +sin(a) and z·z0 ~ +cos(a).
    Crucially, preserve the chosen signs during orthonormalization.
    """
    a_rad = deg2rad(alphas_deg)

    # Linear LSQ to get raw x0,z0
    x0_raw, z0_raw, rms = solve_z0_x0_by_lstsq(a_rad, z_meas)

    # Initial orthonormalization (no sign enforcement yet)
    x0_hat, y_hat, z0_hat = orthonormalize_keep_signs(x0_raw, z0_raw)

    # --- Sign disambiguation using the data ---
    sin_a = np.sin(a_rad)
    cos_a = np.cos(a_rad)
    proj_x = np.einsum('ij,j->i', z_meas, x0_hat)  # z·x0 across frames
    proj_z = np.einsum('ij,j->i', z_meas, z0_hat)  # z·z0 across frames

    s_x = 1.0 if np.sum(proj_x * sin_a) >= 0.0 else -1.0
    s_z = 1.0 if np.sum(proj_z * cos_a) >= 0.0 else -1.0

    # Apply chosen signs and RE-orthonormalize while KEEPING those signs
    x0, y, z0 = orthonormalize_keep_signs(s_x * x0_hat, s_z * z0_hat)

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
                                      weight_by_intensity: bool = True,
                                      ref_frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                                      cond_max: Optional[float] = None) -> List[Dict]:
    """
    Build the linear system for consecutive subsets of reflections:
    [ n cos(alpha-phi) | n sin(alpha-phi) ] · [z0; x0] = 0
    Solve by SVD nullspace.

    NEW:
      - ref_frame=(x0_ref, y_ref, z0_ref): align each window’s (x0,z0,y) to the global frame
        to remove arbitrary ± sign flips.
      - cond_max: if set, skip windows whose conditioning indicator exceeds this value.
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
        w_sub = intens[mask] if weight_by_intensity else None
        m = len(a_sub)

        if m >= 6:
            ca = np.cos(deg2rad(a_sub) - p_sub)
            sa = np.sin(deg2rad(a_sub) - p_sub)
            # Build A: (m x 6); row i: [n_i * ca_i | n_i * sa_i]
            A = np.hstack([n_sub * ca[:, None], n_sub * sa[:, None]])

            v = svd_null_vector(A, w_sub)
            z0_raw = v[:3]
            x0_raw = v[3:]

            # Orthonormalize
            x0, y, z0 = gram_schmidt_frame(x0_raw, z0_raw)

            # Compute conditioning indicator
            if w_sub is not None:
                Aw = (A.T * np.sqrt(np.clip(w_sub, 1e-12, None))).T
                S = np.linalg.svd(Aw, compute_uv=False)
            else:
                S = np.linalg.svd(A, compute_uv=False)
            cond = float(S[-1] / S[-2]) if len(S) >= 2 and S[-2] > 0 else float('nan')

            # Optionally skip poorly-conditioned windows
            if cond_max is not None and np.isfinite(cond) and cond > cond_max:
                idx += step
                continue

            # ---- ALIGN TO GLOBAL FRAME (removes ± sign flips) ----
            if ref_frame is not None:
                x_ref, y_ref, z_ref = ref_frame

                # 1) Global ± ambiguity: flip both if z is opposite to reference
                if np.dot(z0, z_ref) < 0:
                    z0 = -z0
                    x0 = -x0
                # 2) Ensure y has consistent sign with reference
                y = normalize(np.cross(x0, z0))
                if np.dot(y, y_ref) < 0:
                    x0 = -x0
                    y = -y
                # (Optional) bring x closer to its reference if still opposite
                if np.dot(x0, x_ref) < 0:
                    x0 = -x0
                    y = -y  # keep right-handedness

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
    frame_ids, alphas_deg, z_meas = extract_zone_axis_series(pets_block, pets_cif_path)

    # Reflections
    refl = extract_reflections(pets_block, pets_cif_path)

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

    # Detect if nominal alpha runs in the opposite sense; if so, flip and refit
    sense, slope, intercept = detect_alpha_sense(alphas_deg, z_meas, x0, z0)
    if sense < 0:
        print(f"Note: detected opposite alpha sense (slope≈{slope:.3f}). Flipping α and refitting.")
        alphas_deg = -alphas_deg
        x0, y, z0, rms = fit_initial_frame_from_zone_axis(alphas_deg, z_meas)

    # Deviation series
    delta_alpha_deg, beta_deg = compute_deviation_series(alphas_deg, z_meas, x0, y, z0)
    plot_deviation_series(alphas_deg, delta_alpha_deg, beta_deg, out_png=out_png_path)

    print("Δα stats: mean = {:.3e} deg, std = {:.3e} deg, 95% = [{:.3e}, {:.3e}]".format(
        np.mean(delta_alpha_deg), np.std(delta_alpha_deg),
        *np.percentile(delta_alpha_deg, [2.5, 97.5])
    ))
    print("β stats:  mean = {:.3e} deg, std = {:.3e} deg, 95% = [{:.3e}, {:.3e}]".format(
        np.mean(beta_deg), np.std(beta_deg),
        *np.percentile(beta_deg, [2.5, 97.5])
    ))

    # Build alpha_hkl for each plane
    records = build_hkl_alpha_records(refl, frame_ids, alphas_deg,
                                      min_snr=min_snr_reflections,
                                      use_centroid=True)

    # Compute phi for these planes
    phi_list = compute_phi_list(records, UB, unit_cell_pets, wavelength, x0, y, z0)

    # Sliding-window SVD to get six components along rotation series
    six_series = sliding_window_svd_six_components(
        phi_list,
        window_size=window_size,
        step=window_step,
        weight_by_intensity=weight_by_intensity,
        ref_frame=(x0, y, z0),       # align per-window frames to global
        cond_max=0.80                # skip poorly-conditioned windows (tune as desired)
    )

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
    PETS_CIF_PATH = r'C:\Users\Hamis\Documents\GitHub\Felix-python/Si_3_dyn.cif_pets'
    REF_CIF_PATH = r'C:\Users\Hamis\Documents\GitHub\Felix-python/silicon_structure.cif'  # optional but useful

    # Tunable parameters
    MIN_SNR_REFLECTIONS = 0.0     # e.g., 2.0 to filter weak reflections
    WINDOW_SIZE = 100              # number of reflections per SVD window (>=6)
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
