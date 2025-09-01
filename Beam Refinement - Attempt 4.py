"""
wobble_refiner_cifpets.py

Per-frame alpha/beta wobble refinement for Si_3_dyn.cif_pets.
Paste into Spyder and run locally.

Outputs:
 - CSV of per-frame delta_alpha/delta_beta (radians)
 - PNG figure with Δα and Δβ vs frame and vs nominal angle
"""

import numpy as np
from math import cos, sin
from typing import Tuple, Dict, List
from dataclasses import dataclass
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import os

# -------------------------------
# Data containers
# -------------------------------
@dataclass
class ZoneAxis:
    idx: int
    u: float
    v: float
    w: float
    precession_angle: float
    alpha: float         # nominal alpha (radians)
    beta: float          # nominal beta (radians)
    omega: float
    scale: float

@dataclass
class Reflection:
    h: int
    k: int
    l: int
    intensity: float
    sigma: float
    zone_id: int

@dataclass
class UnitCell:
    a: float; b: float; c: float
    alpha: float; beta: float; gamma: float  # degrees

# -------------------------------
# Rotation matrices (explicit)
# -------------------------------
def R_y(alpha: float) -> np.ndarray:
    """Rotation matrix about y. Uses convention in supervisor note."""
    ca = cos(alpha); sa = sin(alpha)
    return np.array([[ ca, 0.0,  sa],
                     [ 0.0, 1.0, 0.0],
                     [-sa, 0.0,  ca]])

def R_x(beta: float) -> np.ndarray:
    """Rotation matrix about x."""
    cb = cos(beta); sb = sin(beta)
    return np.array([[1.0, 0.0,  0.0],
                     [0.0,  cb, -sb],
                     [0.0,  sb,  cb]])

# -------------------------------
# Unit-cell geometry helpers (fallback)
# -------------------------------
def build_direct_lattice_vectors(cell: UnitCell) -> np.ndarray:
    a, b, c = cell.a, cell.b, cell.c
    alpha_r = np.deg2rad(cell.alpha)
    beta_r  = np.deg2rad(cell.beta)
    gamma_r = np.deg2rad(cell.gamma)
    ca = np.cos(alpha_r); cb = np.cos(beta_r); cg = np.cos(gamma_r)
    sg = np.sin(gamma_r)
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * cg, b * sg, 0.0])
    c_x = c * cb
    # compute c_y carefully
    denom = sg if abs(sg) > 1e-12 else 1e-12
    c_y = c * (ca - cb * cg) / denom
    c_z_sq = max(1.0 - cb**2 - ((ca - cb * cg) / denom)**2, 0.0)
    c_z = c * np.sqrt(c_z_sq)
    c_vec = np.array([c_x, c_y, c_z])
    M = np.column_stack([a_vec, b_vec, c_vec])
    return M

def reciprocal_basis_from_direct(M: np.ndarray) -> np.ndarray:
    a_vec = M[:,0]; b_vec = M[:,1]; c_vec = M[:,2]
    V = np.dot(a_vec, np.cross(b_vec, c_vec))
    if abs(V) < 1e-12:
        raise ValueError("Unit cell volume near zero.")
    a_star = np.cross(b_vec, c_vec) / V
    b_star = np.cross(c_vec, a_vec) / V
    c_star = np.cross(a_vec, b_vec) / V
    return np.column_stack([a_star, b_star, c_star])

# -------------------------------
# File parser for .cif_pets
# -------------------------------
def parse_cif_pets(filepath: str) -> Tuple[UnitCell, np.ndarray, float, List[ZoneAxis], List[Reflection]]:
    """
    Parse unit cell, UB (if present), wavelength, zone axes and reflections from .cif_pets file.
    Returns: (unitcell, UB_matrix_or_None, wavelength, zoneaxis_list, reflection_list)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    ub = None
    wavelength = None
    cell = {}
    zoneaxes: List[ZoneAxis] = []
    reflections: List[Reflection] = []
    i = 0
    n = len(lines)

    # first, pick off single-line keys we expect
    for ln in lines:
        s = ln.strip()
        if s.startswith('_cell_length_a'):
            cell['a'] = float(s.split()[1])
        elif s.startswith('_cell_length_b'):
            cell['b'] = float(s.split()[1])
        elif s.startswith('_cell_length_c'):
            cell['c'] = float(s.split()[1])
        elif s.startswith('_cell_angle_alpha'):
            cell['alpha'] = float(s.split()[1])
        elif s.startswith('_cell_angle_beta'):
            cell['beta'] = float(s.split()[1])
        elif s.startswith('_cell_angle_gamma'):
            cell['gamma'] = float(s.split()[1])
        elif s.startswith('_diffrn_radiation_wavelength'):
            wavelength = float(s.split()[1])

    # parse UB if present
    ub_keys = ['_diffrn_orient_matrix_UB_11','_diffrn_orient_matrix_UB_12','_diffrn_orient_matrix_UB_13',
               '_diffrn_orient_matrix_UB_21','_diffrn_orient_matrix_UB_22','_diffrn_orient_matrix_UB_23',
               '_diffrn_orient_matrix_UB_31','_diffrn_orient_matrix_UB_32','_diffrn_orient_matrix_UB_33']
    ubvals = {}
    for ln in lines:
        s = ln.strip()
        for k in ub_keys:
            if s.startswith(k):
                try:
                    ubvals[k] = float(s.split()[1])
                except:
                    ubvals[k] = None
    if len(ubvals) == 9:
        ubmat = np.zeros((3,3))
        ubmat[0,0] = ubvals.get('_diffrn_orient_matrix_UB_11', 0.0)
        ubmat[0,1] = ubvals.get('_diffrn_orient_matrix_UB_12', 0.0)
        ubmat[0,2] = ubvals.get('_diffrn_orient_matrix_UB_13', 0.0)
        ubmat[1,0] = ubvals.get('_diffrn_orient_matrix_UB_21', 0.0)
        ubmat[1,1] = ubvals.get('_diffrn_orient_matrix_UB_22', 0.0)
        ubmat[1,2] = ubvals.get('_diffrn_orient_matrix_UB_23', 0.0)
        ubmat[2,0] = ubvals.get('_diffrn_orient_matrix_UB_31', 0.0)
        ubmat[2,1] = ubvals.get('_diffrn_orient_matrix_UB_32', 0.0)
        ubmat[2,2] = ubvals.get('_diffrn_orient_matrix_UB_33', 0.0)
        ub = ubmat
    else:
        ub = None

    # Now find loop_ blocks and parse zone axis and reflection loops
    i = 0
    while i < n:
        l = lines[i].strip()
        if l.startswith('loop_'):
            # peek ahead to identify loop type
            # collect consecutive header lines starting with _
            headers = []
            j = i + 1
            while j < n and lines[j].strip().startswith('_'):
                headers.append(lines[j].strip())
                j += 1
            header_block = ' '.join(headers)
            # zone axis loop
            if '_diffrn_zone_axis_id' in header_block:
                # read data lines until blank or next loop_ or header
                k = j
                while k < n:
                    s = lines[k].strip()
                    if s == '' or s.startswith('loop_') or s.startswith('_'):
                        break
                    parts = s.split()
                    if len(parts) >= 9:
                        zid = int(parts[0])
                        u = float(parts[1]); v = float(parts[2]); w = float(parts[3])
                        pa = float(parts[4])
                        alpha = float(parts[5])
                        beta = float(parts[6])
                        omega = float(parts[7])
                        scale = float(parts[8])
                        zoneaxes.append(ZoneAxis(zid,u,v,w,pa,np.deg2rad(alpha),np.deg2rad(beta),omega,scale))
                    k += 1
                i = k
                continue
            # reflection loop
            elif '_refln_index_h' in header_block:
                k = j
                while k < n:
                    s = lines[k].strip()
                    if s == '' or s.startswith('loop_') or s.startswith('_'):
                        break
                    parts = s.split()
                    # expect at least 6 columns: h k l intensity sigma zone_id
                    if len(parts) >= 6:
                        try:
                            h = int(parts[0]); k_ = int(parts[1]); l = int(parts[2])
                            intensity = float(parts[3]); sigma = float(parts[4]); zid = int(parts[5])
                            reflections.append(Reflection(h, k_, l, intensity, sigma, zid))
                        except:
                            pass
                    k += 1
                i = k
                continue
        i += 1

    if not all(k in cell for k in ('a','b','c','alpha','beta','gamma')):
        raise ValueError("Unit cell parameters not found in file.")
    if wavelength is None:
        raise ValueError("Wavelength not found in file.")
    unitcell = UnitCell(cell['a'], cell['b'], cell['c'], cell['alpha'], cell['beta'], cell['gamma'])
    return unitcell, ub, wavelength, zoneaxes, reflections

# -------------------------------
# Compute g_lab from hkl using UB (preferred) or unit cell fallback
# -------------------------------
def g_lab_from_hkl(hkl: Tuple[int,int,int], unitcell: UnitCell, UB: np.ndarray=None) -> np.ndarray:
    h,k,l = hkl
    if UB is not None:
        return UB @ np.array([h, k, l], dtype=float)
    # fallback: compute reciprocal basis from unit cell
    M = build_direct_lattice_vectors(unitcell)
    rec_basis = reciprocal_basis_from_direct(M)  # columns a*,b*,c*
    g_crystal = rec_basis @ np.array([h, k, l], dtype=float)
    # assume crystal frame == lab if we have no UB (best-effort)
    return g_crystal

# -------------------------------
# Residuals and per-frame fit
# -------------------------------
def per_frame_residuals(deltas: np.ndarray, alpha_nom: float, beta_nom: float, g_list: np.ndarray) -> np.ndarray:
    """
    Residuals for one frame: dot(z_frame, n_i) for each g_i in g_list.
    deltas = [delta_alpha, delta_beta] in radians
    """
    da, db = float(deltas[0]), float(deltas[1])
    alpha = alpha_nom + da
    beta  = beta_nom  + db
    z0 = np.array([0.0, 0.0, 1.0])
    z_rot = R_y(alpha) @ z0
    z_frame = R_x(beta) @ z_rot
    norms = np.linalg.norm(g_list, axis=1)
    norms[norms == 0] = 1e-12
    n_list = g_list / norms[:, None]
    res = n_list @ z_frame
    return res

def refine_frame_deltas(alpha_nom: float, beta_nom: float, g_list: np.ndarray,
                        initial_guess: Tuple[float,float]=(0.0,0.0),
                        min_refs: int = 3) -> Tuple[float,float,Dict]:
    """
    Solve for delta_alpha, delta_beta for a single frame.
    Returns (delta_alpha, delta_beta, info).
    """
    info = {}
    if len(g_list) < min_refs:
        info['status'] = 'too_few_refs'
        return 0.0, 0.0, info

    x0 = np.array([initial_guess[0], initial_guess[1]])
    # robust least squares with Huber loss
    try:
        res = least_squares(per_frame_residuals, x0,
                            args=(alpha_nom, beta_nom, g_list),
                            method='trf',
                            loss='huber',
                            ftol=1e-10, xtol=1e-10, gtol=1e-10,
                            max_nfev=5000, verbose=0)
    except Exception:
        # fallback
        res = least_squares(per_frame_residuals, x0,
                            args=(alpha_nom, beta_nom, g_list),
                            method='lm',
                            ftol=1e-10, xtol=1e-10, gtol=1e-10,
                            max_nfev=5000, verbose=0)

    da, db = float(res.x[0]), float(res.x[1])
    info = {'status':'ok', 'success': bool(res.success), 'cost': float(res.cost), 'nfev': int(res.nfev)}
    return da, db, info

# -------------------------------
# Top-level driver
# -------------------------------
def refine_wobble_cifpets(filepath: str,
                          out_csv: str = None,
                          out_png: str = None,
                          min_refs_per_frame: int = 3,
                          max_frames: int = None):
    """
    Main driver. Reads filepath (Si_3_dyn.cif_pets) and returns results list.
    Saves CSV and PNG if out paths provided.
    """
    # parse file
    unitcell, UB, wavelength, zoneaxes, reflections = parse_cif_pets(filepath)
    UB_mat = None if UB is None else np.array(UB, dtype=float)

    # map zone axes by id
    zone_map = {z.idx: z for z in zoneaxes}

    # group reflections by zone id
    ref_groups = defaultdict(list)
    for r in reflections:
        ref_groups[r.zone_id].append(r)

    zone_ids = sorted(ref_groups.keys())
    if max_frames is not None:
        zone_ids = zone_ids[:max_frames]

    results = []
    for zid in zone_ids:
        zaxis = zone_map.get(zid, None)
        if zaxis is None:
            continue
        alpha_nom = zaxis.alpha
        beta_nom = zaxis.beta
        refs = ref_groups[zid]
        # build g_list
        g_list = []
        for r in refs:
            g = g_lab_from_hkl((r.h, r.k, r.l), unitcell, UB=UB_mat)
            g_list.append(g)
        g_list = np.array(g_list) if len(g_list) > 0 else np.zeros((0,3))
        if len(g_list) < min_refs_per_frame:
            res = {'zone_id': zid, 'alpha_nom': alpha_nom, 'beta_nom': beta_nom,
                   'delta_alpha': 0.0, 'delta_beta': 0.0, 'n_reflections': len(g_list),
                   'info': {'status':'too_few'}}
            results.append(res)
            continue
        da, db, info = refine_frame_deltas(alpha_nom, beta_nom, g_list, initial_guess=(0.0,0.0), min_refs=min_refs_per_frame)
        res = {'zone_id': zid, 'alpha_nom': alpha_nom, 'beta_nom': beta_nom,
               'delta_alpha': da, 'delta_beta': db, 'n_reflections': len(g_list), 'info': info}
        results.append(res)

    # Save CSV
    if out_csv is None:
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_csv = f"{base}_wobble_results.csv"
    _save_results_csv(results, out_csv)

    # Make plot
    if out_png is None:
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_png = f"{base}_wobble_plot.png"
    _plot_results(results, out_png)

    return results

# -------------------------------
# Utilities: save CSV and plot
# -------------------------------
def _save_results_csv(results: List[Dict], out_path: str):
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['zone_id', 'alpha_nom_deg', 'beta_nom_deg', 'delta_alpha_deg', 'delta_beta_deg', 'n_reflections', 'info'])
        for r in results:
            writer.writerow([r['zone_id'],
                             np.rad2deg(r['alpha_nom']),
                             np.rad2deg(r['beta_nom']),
                             np.rad2deg(r['delta_alpha']),
                             np.rad2deg(r['delta_beta']),
                             r['n_reflections'],
                             r['info']])
    print(f"Wobble results written to: {out_path}")

def _plot_results(results: List[Dict], out_png: str):
    zone_ids = [r['zone_id'] for r in results]
    alpha_nom_deg = [np.rad2deg(r['alpha_nom']) for r in results]
    da_deg = [np.rad2deg(r['delta_alpha']) for r in results]
    db_deg = [np.rad2deg(r['delta_beta']) for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(10,8), sharex=True)
    ax0, ax1 = axes

    ax0.plot(zone_ids, da_deg, marker='o', linestyle='-', label=r'$\Delta \alpha$ (deg)')
    ax0.set_ylabel(r'$\Delta \alpha$ (deg)')
    ax0.grid(True)
    ax0.legend()

    ax1.plot(zone_ids, db_deg, marker='o', linestyle='-', label=r'$\Delta \beta$ (deg)', color='tab:orange')
    ax1.set_ylabel(r'$\Delta \beta$ (deg)')
    ax1.set_xlabel('Zone axis / frame index')
    ax1.grid(True)
    ax1.legend()

    plt.tight_layout()
    fig.suptitle('Per-frame wobble: Δα and Δβ', y=1.02)
    plt.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.show()
    print(f"Wobble plot saved to: {out_png}")

# -------------------------------
# Example entrypoint (edit path as required)
# -------------------------------
if __name__ == "__main__":
    # Edit this to the path of your .cif_pets file
    INPUT_FILE = "Si_3_dyn.cif_pets"   # change to full path if not in working dir
    # Optionally adjust these
    OUT_CSV = None
    OUT_PNG = None
    MIN_REFS = 3
    MAX_FRAMES = None  # set to an int for quicker debugging

    print("Starting wobble refinement for:", INPUT_FILE)
    results = refine_wobble_cifpets(INPUT_FILE, out_csv=OUT_CSV, out_png=OUT_PNG,
                                    min_refs_per_frame=MIN_REFS, max_frames=MAX_FRAMES)
    print("Done. Processed frames:", len(results))
