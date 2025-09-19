# -*- coding: utf-8 -*-
"""
Solve for orthonormal u, v in R^3 from equations:
    (u*cos(Phi_i) + v*sin(Phi_i)) · n_i = 0
where n_i is the unit vector along (h,k,l),
      Phi_i is computed from alpha_i, theta_i, phi_i,
      theta_i = arcsin(lambda / (2 * d_hkl)), d_hkl = a / sqrt(h^2 + k^2 + l^2),
      cos(phi_i) = n_i · xhat  (with xhat = (1,0,0)).

The script evaluates two plausible Phi models:
    'divide':    Phi = alpha - theta / cos(phi)
    'multiply':  Phi = alpha - theta * cos(phi)
and automatically picks the one with lower global residual.

It also computes sliding-window estimates vs frame.

Paste into Spyder and run.
"""

import numpy as np

# ----------------------------
# Input constants and dataset
# ----------------------------
LAMBDA = 0.02508      # wavelength (same units as 'a') 
A_LATTICE = 5.42196   # lattice parameter 'a'

# (h, k, l, frame_at_peak, alpha_at_peak_deg)
REFLECTIONS = [
  (-14, -2,  -2,  4, -69.487),
  (-11, -1,  -3,  4, -69.487),
  (  8,  4,  -4,  9, -68.974),
  (  3, -3,   9, 12, -68.666),
  ( -1,  3,  -7, 13, -68.563),
  ( -8,  0,  -4, 17, -68.153),
  ( -5,  3,  -9, 17, -68.153),
  (  3,  3,  -5, 18, -68.050),
  (  1, -1,   3, 23, -67.537),
  (  7, -1,   7, 24, -67.434),
  ( -2,  4, -10, 25, -67.332),
  ( -1, -3,   7, 26, -67.229),
  ( -5,  1,  -5, 27, -67.126),
  ( -4, -4,   8, 28, -67.024),
  ( -1,  1,  -3, 31, -66.716),
  ( -6,  2,  -8, 35, -66.305),
  ( -2, -2,   4, 35, -66.305),
  ( -5, -3,   5, 37, -66.100),
  (  6,  4,  -6, 37, -66.100),
  (  7,  3,  -3, 38, -65.997),
]

# ----------------------------
# Helper functions
# ----------------------------
def unit_n(h, k, l):
    v = np.array([h, k, l], dtype=float)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        raise ValueError("Invalid (h,k,l) = (0,0,0)")
    return v / nrm

def d_hkl(h, k, l, a):
    return a / np.sqrt(h*h + k*k + l*l)

def theta_from_hkl(h, k, l, a, lam):
    d = d_hkl(h, k, l, a)
    x = lam / (2.0 * d)
    x = np.clip(x, -1.0, 1.0)   # numerical safety
    return np.arcsin(x)

def phi_from_n(n):
    # angle between n and x-axis (xhat = (1,0,0))
    cosphi = np.clip(n[0], -1.0, 1.0)
    return np.arccos(cosphi)

def build_rows(model='divide'):
    """
    Build data rows: for each reflection create tuple
    (n_i, cos(Phi_i), sin(Phi_i), frame, alpha_deg, Phi_i_rad)
    """
    rows = []
    skipped = 0
    for h,k,l,frame,alpha_deg in REFLECTIONS:
        n = unit_n(h, k, l)
        phi = phi_from_n(n)
        theta = theta_from_hkl(h, k, l, A_LATTICE, LAMBDA)
        alpha = np.deg2rad(alpha_deg)

        if model == 'divide':
            cphi = np.cos(phi)
            if np.abs(cphi) < 1e-9:
                skipped += 1
                continue  # avoid blow-up
            Phi = alpha - theta / cphi
        elif model == 'multiply':
            Phi = alpha - theta * np.cos(phi)
        else:
            raise ValueError("model must be 'divide' or 'multiply'")

        rows.append((n, np.cos(Phi), np.sin(Phi), frame, alpha_deg, Phi))
    return rows, skipped

def build_M(rows):
    """ Construct the M matrix with rows [cos(Phi)*n^T,  sin(Phi)*n^T]. """
    m = len(rows)
    M = np.zeros((m, 6), dtype=float)
    for i, (n, cP, sP, *_rest) in enumerate(rows):
        M[i, :3] = cP * n
        M[i, 3:] = sP * n
    return M

def solve_uv_from_rows(rows):
    """ Solve M [u; v] = 0 via SVD; return orthonormal u, v, residuals, singular values. """
    M = build_M(rows)
    U, S, VT = np.linalg.svd(M, full_matrices=True)
    a = VT[-1, :]                  # right singular vector for smallest singular value
    u_raw, v_raw = a[:3], a[3:]

    # Orthonormalize (Gram–Schmidt)
    if np.linalg.norm(u_raw) == 0 or np.linalg.norm(v_raw) == 0:
        raise RuntimeError("Degenerate solution vector encountered.")
    u = u_raw / np.linalg.norm(u_raw)
    v = v_raw - (u @ v_raw) * u
    v = v / np.linalg.norm(v)

    # Residuals
    res = np.array([(cP * u + sP * v) @ n for (n, cP, sP, *_rest) in rows])
    rmse = float(np.sqrt(np.mean(res**2)))

    return u, v, rmse, S, res

def pick_best_model():
    """Try both Phi models and pick the one with smaller global RMSE."""
    best = None
    for model in ('divide', 'multiply'):
        rows, skipped = build_rows(model)
        u, v, rmse, S, res = solve_uv_from_rows(rows)
        info = dict(model=model, rows=len(rows), skipped=skipped,
                    u=u, v=v, rmse=rmse, S=S, res=res)
        if (best is None) or (rmse < best['rmse']):
            best = info
    return best

def sliding_window_estimates(model, window_size=6, step=1):
    """Compute u, v over sliding windows (by row order)."""
    rows, skipped = build_rows(model)
    out = []
    for start in range(0, len(rows) - window_size + 1, step):
        win = rows[start:start + window_size]
        u, v, rmse, S, res = solve_uv_from_rows(win)
        frames = [w[3] for w in win]
        alpha_deg = [w[4] for w in win]
        out.append({
            'start_idx': start,
            'end_idx': start + window_size - 1,
            'frame_mid': float(np.mean(frames)),
            'alpha_deg_mid': float(np.mean(alpha_deg)),
            'u': u, 'v': v, 'rmse': rmse,
            'sigma_min': float(S[-1]),
        })
    return rows, skipped, out

# ----------------------------
# Run: global fit + sliding windows
# ----------------------------
if __name__ == "__main__":
    best = pick_best_model()

    print("\n=== GLOBAL FIT ===")
    print(f"Chosen Phi model       : {best['model']}  (rows used: {best['rows']}, skipped: {best['skipped']})")
    print(f"Singular values (M)    : {np.array2string(best['S'], precision=6)}")
    print(f"Global RMSE            : {best['rmse']:.6f}")

    u, v = best['u'], best['v']
    print("\nEstimated u, v (unit & orthogonal):")
    print(f"u = [{u[0]: .6f}, {u[1]: .6f}, {u[2]: .6f}]")
    print(f"v = [{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}]")
    print(f"‖u‖={np.linalg.norm(u):.6f}, ‖v‖={np.linalg.norm(v):.6f}, u·v={u@v:.6f}")

    # Sliding window settings
    WINDOW_SIZE = 6    # use 6 reflections per estimate (adjust as you like)
    STEP = 1           # slide by 1 row each time

    print("\n=== SLIDING WINDOW ESTIMATES ===")
    rows, skipped, sw = sliding_window_estimates(best['model'], WINDOW_SIZE, STEP)
    print(f"(model='{best['model']}', rows used: {len(rows)}, skipped: {skipped}, window={WINDOW_SIZE}, step={STEP})\n")

    for r in sw:
        u_w, v_w = r['u'], r['v']
        print(f"[{r['start_idx']:02d}-{r['end_idx']:02d}] frame≈{r['frame_mid']:5.2f}, "
              f"alpha≈{r['alpha_deg_mid']:8.3f}° | RMSE={r['rmse']:.6f}, σ_min={r['sigma_min']:.6f}")
        print(f"    u = [{u_w[0]: .6f}, {u_w[1]: .6f}, {u_w[2]: .6f}]")
        print(f"    v = [{v_w[0]: .6f}, {v_w[1]: .6f}, {v_w[2]: .6f}]")
