#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: Initial orientation (UB -> U -> R0) and sanity checks.

- Decompose UB with the B matrix to obtain an initial U.
- Project U to the nearest proper rotation R0 via SVD (polar projection).
- Sanity checks:
  * Orthonormality and det(R0).
  * Zone-axis consistency: compare predicted beam direction in crystal frame
    vs. the (u,v,w) in the PETS zone-axis table; report angular misfits.
  * Bragg residuals: Z · [R_y(α) R_x(β) R0 n̂(hkl)] - sin θ(hkl | λ) for a subset
    of reflections; report summary stats.

All functions are defined at the top and use explicit arguments only.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# Stage 1 and Stage 2 modules should be in the same folder.
try:
    from stage1_parse_inputs import parse_pets_cif_pets, parse_reference_cif
    from stage2_reciprocal_tools import (
        build_B_matrix,
        ub_b_consistency_report,
        infer_ub_2pi_convention,
        plane_normal_unit_from_B,
        d_spacing_from_B,
        bragg_angle_from_d,
    )
except Exception as e:
    raise ImportError(
        "Could not import Stage 1/2 modules. Ensure 'stage1_parse_inputs.py' and "
        "'stage2_reciprocal_tools.py' are in the same directory."
    ) from e


# --------------------------- Linear algebra helpers --------------------------

def project_to_SO3(M: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Project a 3x3 matrix to the nearest proper rotation (SO(3)) using SVD.

    Args:
        M: 3x3 matrix (float)

    Returns:
        R: nearest rotation (det=+1)
        detM: determinant of input M
        ortho_err: ||R^T R - I||_F (Frobenius norm) as an orthonormality check
    """
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Enforce det +1 (handle reflections)
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    detM = float(np.linalg.det(M))
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    return R, detM, ortho_err


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about X by angle (radians)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]], dtype=float)


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about Y by angle (radians)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[  c, 0.0,   s],
                     [0.0, 1.0, 0.0],
                     [ -s, 0.0,   c]], dtype=float)


def unit_vector(v: np.ndarray) -> np.ndarray:
    """Return v / ||v|| (handles zero safely)."""
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v.copy()
    return v / n


def angle_between_vectors_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two 3D vectors."""
    a_u = unit_vector(a)
    b_u = unit_vector(b)
    dot = float(np.dot(a_u, b_u))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


# ------------------------- Orientation & residuals ---------------------------

def ub_to_u(UB: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute U ≈ UB * B^{-1}.
    For a perfect UB = U * B, this returns U exactly.
    """
    B_inv = np.linalg.inv(B)
    return UB @ B_inv


def predicted_beam_dir_crystal(R0: np.ndarray, alpha_deg: float, beta_deg: float) -> np.ndarray:
    """
    Predicted incident beam direction expressed in the crystal frame at a given (alpha, beta),
    according to the nominal model R(α,β) = R_y(α) R_x(β) R0.

    We want z_cryst such that: z_lab = Z = R_y(α) R_x(β) R0 z_cryst  =>  z_cryst = R0^T R_x(-β) R_y(-α) Z
    """
    Z = np.array([0.0, 0.0, 1.0], dtype=float)
    a = math.radians(alpha_deg)
    b = math.radians(beta_deg)
    z_cryst = R0.T @ rotation_matrix_x(-b) @ rotation_matrix_y(-a) @ Z
    return unit_vector(z_cryst)


def bragg_residual_one(
    R0: np.ndarray,
    alpha_deg: float,
    beta_deg: float,
    n_hat_cryst: np.ndarray,
    theta_rad: float
) -> float:
    """
    Bragg residual for one reflection (no precession modeled here):
      r = Z · [ R_y(α) R_x(β) R0 n̂ ] - sin θ
    """
    Z = np.array([0.0, 0.0, 1.0], dtype=float)
    a = math.radians(alpha_deg)
    b = math.radians(beta_deg)
    n_lab = rotation_matrix_y(a) @ rotation_matrix_x(b) @ R0 @ n_hat_cryst
    return float(np.dot(Z, n_lab) - math.sin(theta_rad))


def link_reflections_with_zone_alpha_beta(
    reflections_df: pd.DataFrame,
    zone_axes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge alpha/beta onto reflections via zone_axis_id.
    Keeps only rows where alpha/beta are present.
    """
    z = zone_axes_df[["zone_axis_id", "alpha_deg", "beta_deg"]].dropna()
    z = z.rename(columns={"alpha_deg": "alpha_deg_zone", "beta_deg": "beta_deg_zone"})
    merged = reflections_df.merge(z, how="left", on="zone_axis_id")
    merged = merged.dropna(subset=["alpha_deg_zone", "beta_deg_zone"])
    return merged


# --------------------------- Stage 3 driver ----------------------------------

def main(experiment_path: str, cif_path: str, n_bragg_preview: int = 2000) -> None:
    """
    Stage 3 driver:
      - Parse inputs (Stage 1),
      - Determine B convention via UB vs. B norms,
      - Compute U_raw = UB @ B^{-1}, then project to R0 ∈ SO(3),
      - Sanity: report det(U_raw), orthonormality of R0, R0 matrix,
      - Zone-axis misfit stats,
      - Bragg residual stats on a subset of reflections.
    """
    # ---------- Parse inputs ----------
    pets_header, UB, wavelength, zone_axes_df, reflections_df = parse_pets_cif_pets(experiment_path)
    # CIF is not used here, but could be used to cross-check cell; we prefer PETS cell to match UB
    _, cif_cell, cif_space_group, cif_atoms = parse_reference_cif(cif_path)

    # ---------- Build B consistent with UB ----------
    cell = {
        "a": float(pets_header.get("_cell_length_a", "nan")),
        "b": float(pets_header.get("_cell_length_b", "nan")),
        "c": float(pets_header.get("_cell_length_c", "nan")),
        "alpha": float(pets_header.get("_cell_angle_alpha", "90")),
        "beta": float(pets_header.get("_cell_angle_beta", "90")),
        "gamma": float(pets_header.get("_cell_angle_gamma", "90")),
    }

    report = ub_b_consistency_report(
        UB, cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"]
    )
    include_2pi = infer_ub_2pi_convention(report["scale_ratio_vs_no2pi"], report["scale_ratio_vs_2pi"])

    print("\n=== Stage 3: Initial Orientation ===")
    print("Cell (PETS):", cell)
    print(f"Wavelength λ (Å): {wavelength}")
    print(f"Inferred UB convention → build B with include_2pi = {include_2pi}")

    B = build_B_matrix(cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"], include_2pi)

    # ---------- U from UB and B; project to R0 ----------
    U_raw = ub_to_u(UB, B)
    R0, detUraw, ortho_err_R0 = project_to_SO3(U_raw)

    # Deviation of U_raw from rotation (Frobenius distance) for info
    # Using R_diff = R0^T U_raw, angle from trace
    R_diff = R0.T @ U_raw
    trace_val = float(np.trace(R_diff))
    # Clamp trace-based angle calc
    tr_clamped = max(-1.0, min(3.0, trace_val))
    angle_err_rad = math.acos((tr_clamped - 1.0) / 2.0) if -1.0 <= (tr_clamped - 1.0) / 2.0 <= 1.0 else float("nan")
    angle_err_deg = math.degrees(angle_err_rad) if not math.isnan(angle_err_rad) else float("nan")

    print("\nU_raw = UB @ B^{-1} (not forced to be a rotation):")
    with np.printoptions(precision=6, suppress=True):
        print(U_raw)
    print(f"det(U_raw) = {detUraw:.6f}")
    print(f"Nearest rotation R0 (proj to SO(3)), orthonormality ||R0^T R0 - I||_F = {ortho_err_R0:.3e}")
    with np.printoptions(precision=6, suppress=True):
        print("R0 =")
        print(R0)
    print(f"Angle between U_raw and R0 (via R0^T U_raw) ≈ {angle_err_deg:.6f} deg")

    # ---------- Zone-axis sanity check ----------
    if zone_axes_df.empty:
        print("\nZone-axis table is empty; skipping zone-axis checks.")
    else:
        print("\nZone-axis misfit (predicted z_cryst vs. PETS (u,v,w)):")
        # Compute angular misfit per zone-axis row
        angles: List[float] = []
        preview_rows = min(10, len(zone_axes_df))
        print(f"Preview of first {preview_rows} frames:")
        for idx in range(preview_rows):
            row = zone_axes_df.iloc[idx]
            z_pred = predicted_beam_dir_crystal(R0, float(row["alpha_deg"]), float(row["beta_deg"] or 0.0))
            z_meas = np.array([float(row["u"]), float(row["v"]), float(row["w"])], dtype=float)
            ang_deg = angle_between_vectors_deg(z_pred, z_meas)
            angles.append(ang_deg)
            print(f"  id={int(row['zone_axis_id']):4d}  α={row['alpha_deg']:8.3f}  β={row['beta_deg']:6.3f}  "
                  f"angle(z_pred,uvw)={ang_deg:7.4f}°")

        # Full stats (all frames)
        angles_all: List[float] = []
        for _, row in zone_axes_df.iterrows():
            z_pred = predicted_beam_dir_crystal(R0, float(row["alpha_deg"]), float(row["beta_deg"] or 0.0))
            z_meas = np.array([float(row["u"]), float(row["v"]), float(row["w"])], dtype=float)
            angles_all.append(angle_between_vectors_deg(z_pred, z_meas))

        angles_arr = np.array(angles_all, dtype=float)
        print("\nZone-axis misfit statistics (degrees):")
        print(f"  count={angles_arr.size}")
        print(f"  mean = {angles_arr.mean():.4f}")
        print(f"  median = {np.median(angles_arr):.4f}")
        print(f"  p90 = {np.percentile(angles_arr, 90):.4f}")
        print(f"  max  = {angles_arr.max():.4f}")

    # ---------- Bragg residual preview ----------
    if reflections_df.empty or zone_axes_df.empty:
        print("\nReflection table or zone-axis table is empty; skipping Bragg residual preview.")
        return

    merged = link_reflections_with_zone_alpha_beta(reflections_df, zone_axes_df)
    if merged.empty:
        print("\nNo reflections could be linked to zone-axis α/β; skipping Bragg residual preview.")
        return

    # Compute residuals on a subset for speed
    n_use = min(n_bragg_preview, len(merged))
    sub = merged.iloc[:n_use].copy()

    # Precompute: for each (h,k,l) -> n_hat and θ
    # NOTE: n̂ direction is independent of the 2π convention; θ uses d-spacing which *does* depend on it,
    # but we already built B consistently with UB.
    n_list: List[np.ndarray] = []
    theta_list: List[float] = []

    for h, k, l in zip(sub["h"], sub["k"], sub["l"]):
        n_hat = plane_normal_unit_from_B(B, int(h), int(k), int(l))
        dA = d_spacing_from_B(B, int(h), int(k), int(l), include_2pi=include_2pi)
        theta_rad, _theta_deg = bragg_angle_from_d(wavelength, dA)
        n_list.append(n_hat)
        theta_list.append(theta_rad)

    sub["_theta_rad"] = theta_list

    # Residuals
    resids: List[float] = []
    for (alpha_deg, beta_deg, n_hat, theta_rad) in zip(
        sub["alpha_deg_zone"], sub["beta_deg_zone"], n_list, sub["_theta_rad"]
    ):
        r = bragg_residual_one(R0, float(alpha_deg), float(beta_deg or 0.0), n_hat, float(theta_rad))
        resids.append(r)

    resids = np.array(resids, dtype=float)
    abs_res = np.abs(resids)
    print(f"\nBragg residual preview on {len(resids)} reflections:")
    print("  r = Z·[R_y(α) R_x(β) R0 n̂] - sinθ   (unitless)")
    print(f"  mean(r) = {resids.mean(): .6e}")
    print(f"  std(r)  = {resids.std(): .6e}")
    print(f"  mean(|r|) = {abs_res.mean(): .6e}")
    print(f"  p90(|r|)  = {np.percentile(abs_res, 90): .6e}")
    print(f"  max(|r|)  = {abs_res.max(): .6e}")

    # Show a tiny sample
    print("\nSample rows (hkl, α, β, sinθ, Z·n_lab, residual):")
    for i in range(min(8, len(sub))):
        h, k, l = int(sub.iloc[i]["h"]), int(sub.iloc[i]["k"]), int(sub.iloc[i]["l"])
        a_deg = float(sub.iloc[i]["alpha_deg_zone"])
        b_deg = float(sub.iloc[i]["beta_deg_zone"] or 0.0)
        n_hat = n_list[i]
        theta = float(sub.iloc[i]["_theta_rad"])
        z_nlab = float(np.dot(
            np.array([0.0, 0.0, 1.0], dtype=float),
            rotation_matrix_y(math.radians(a_deg)) @ rotation_matrix_x(math.radians(b_deg)) @ R0 @ n_hat
        ))
        print(f"  ({h:3d},{k:3d},{l:3d})  α={a_deg:8.3f}  β={b_deg:7.3f}  sinθ={math.sin(theta): .6f}  "
              f"Z·n_lab={z_nlab: .6f}  r={z_nlab - math.sin(theta): .6e}")


# ------------------------------ Entrypoint -----------------------------------

if __name__ == "__main__":
    # Adjust these paths as needed
    EXPERIMENT_PATH = "Si_3_dyn.cif_pets"
    CIF_PATH = "silicon_structure.cif"
    # Number of reflections to preview in the residual computation
    N_BRAGG_PREVIEW = 2000
    main(EXPERIMENT_PATH, CIF_PATH, n_bragg_preview=N_BRAGG_PREVIEW)
