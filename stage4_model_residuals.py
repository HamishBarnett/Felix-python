#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4: Orientation model and residual functions for cRED refinement.

This module provides:
- Rotation utilities (Rx, Ry), unit/angle helpers.
- Forward model: R_actual(α) = R_y(α + δα(α)) · R_x(β(α)) · R0.
- Bragg residual:
    r = Z · [ R_actual(α) · n̂_hkl ] - sin(θ_hkl)
  and an optional precession-aware version (min over azimuth samples).
- Zone-axis residual: angular misfit between predicted and measured beam
  direction in the crystal frame.

All functions accept explicit arguments, and there are no inner functions that
capture outer variables. Designed to be called by later optimisation code.

A small preview driver at the bottom computes baseline residuals with δα=β=0.
"""

from __future__ import annotations
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Bring in parsers and reciprocal-lattice tools from earlier stages
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
        "Ensure 'stage1_parse_inputs.py' and 'stage2_reciprocal_tools.py' are in the same directory."
    ) from e


# ----------------------------- Linear algebra --------------------------------

def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about X (radians)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]], dtype=float)


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about Y (radians)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[  c, 0.0,   s],
                     [0.0, 1.0, 0.0],
                     [ -s, 0.0,   c]], dtype=float)


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about Z (radians)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def unit_vector(v: np.ndarray) -> np.ndarray:
    """Return v normalized; leaves zero vector unchanged."""
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


# ------------------------------ Model utilities ------------------------------

def r_actual(R0: np.ndarray, alpha_deg: float, delta_alpha_deg: float, beta_deg: float) -> np.ndarray:
    """
    Build R_actual(α) = R_y(α + δα) · R_x(β) · R0.

    Args:
        R0: 3x3 initial orientation (rotation matrix)
        alpha_deg: nominal rotation angle for the frame (deg)
        delta_alpha_deg: along-axis phase correction at this α (deg)
        beta_deg: small tilt about X at this α (deg)

    Returns:
        3x3 rotation matrix mapping crystal vectors to lab at this α.
    """
    a = math.radians(alpha_deg + delta_alpha_deg)
    b = math.radians(beta_deg)
    return rotation_matrix_y(a) @ rotation_matrix_x(b) @ R0


def predicted_beam_dir_crystal(R0: np.ndarray, alpha_deg: float, delta_alpha_deg: float, beta_deg: float) -> np.ndarray:
    """
    Predicted incident beam direction expressed in the *crystal* frame at a given (α, δα, β).
    From Z_lab = R_actual(α) · z_cryst  ⇒  z_cryst = R0^T R_x(-β) R_y(-(α+δα)) Z.
    """
    Z = np.array([0.0, 0.0, 1.0], dtype=float)
    a = math.radians(alpha_deg + delta_alpha_deg)
    b = math.radians(beta_deg)
    z_cryst = R0.T @ rotation_matrix_x(-b) @ rotation_matrix_y(-a) @ Z
    return unit_vector(z_cryst)


# ------------------------------ Bragg residuals ------------------------------

def bragg_residual_scalar(
    R0: np.ndarray,
    alpha_deg: float,
    delta_alpha_deg: float,
    beta_deg: float,
    n_hat_cryst: np.ndarray,
    theta_rad: float
) -> float:
    """
    Basic Bragg residual for one reflection (no precession):
      r = Z · [ R_actual(α) · n̂_hkl ] - sin(θ_hkl)
    Positive if the plane normal has too large a Z component in lab.
    """
    Z = np.array([0.0, 0.0, 1.0], dtype=float)
    n_lab = r_actual(R0, alpha_deg, delta_alpha_deg, beta_deg) @ n_hat_cryst
    return float(np.dot(Z, n_lab) - math.sin(theta_rad))


def bragg_residual_scalar_with_precession(
    R0: np.ndarray,
    alpha_deg: float,
    delta_alpha_deg: float,
    beta_deg: float,
    n_hat_cryst: np.ndarray,
    theta_rad: float,
    precession_deg: float,
    n_azim_samples: int = 12
) -> float:
    """
    Precession-aware Bragg residual:
      r(ψ) = Z_ψ · [ R_actual(α) · n̂ ] - sin(θ)
    where Z_ψ = R_z(ψ) R_x(γ) Z and γ = precession_deg.

    We return the *minimum absolute* residual over ψ samples, with the sign of
    the residual at the minimising ψ (so it's still signed, but near-zero if
    the cone intersects the Bragg condition).

    This is more expensive; use only if the simple residual shows a systematic bias.
    """
    if precession_deg <= 0.0 or n_azim_samples < 1:
        return bragg_residual_scalar(R0, alpha_deg, delta_alpha_deg, beta_deg, n_hat_cryst, theta_rad)

    Z = np.array([0.0, 0.0, 1.0], dtype=float)
    gamma = math.radians(precession_deg)
    # Rotate plane normal to lab once; vary only the beam direction on its cone
    n_lab = r_actual(R0, alpha_deg, delta_alpha_deg, beta_deg) @ n_hat_cryst

    best_val = None
    best_signed = None
    for k in range(n_azim_samples):
        psi = 2.0 * math.pi * (k / n_azim_samples)
        Z_psi = rotation_matrix_z(psi) @ rotation_matrix_x(gamma) @ Z
        r = float(np.dot(Z_psi, n_lab) - math.sin(theta_rad))
        val = abs(r)
        if (best_val is None) or (val < best_val):
            best_val = val
            best_signed = r
    return float(best_signed if best_signed is not None else 0.0)


# ------------------------------ Zone-axis residual ---------------------------

def zone_axis_residual_scalar(
    R0: np.ndarray,
    alpha_deg: float,
    delta_alpha_deg: float,
    beta_deg: float,
    measured_u_v_w: Sequence[float]
) -> float:
    """
    Angular misfit (radians) between predicted beam direction in the crystal frame
    and the measured (u,v,w) from the PETS zone-axis table.
    """
    z_meas = unit_vector(np.asarray(measured_u_v_w, dtype=float))
    z_pred = predicted_beam_dir_crystal(R0, alpha_deg, delta_alpha_deg, beta_deg)
    # Use angle from dot product
    dot = float(np.dot(z_meas, z_pred))
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)  # radians


# ------------------------------ Batch evaluators -----------------------------

def evaluate_bragg_residuals_batch(
    R0: np.ndarray,
    alphas_deg: np.ndarray,
    delta_alpha_deg_vals: np.ndarray,
    betas_deg_vals: np.ndarray,
    n_hats_cryst: np.ndarray,
    thetas_rad: np.ndarray,
    use_precession: bool = False,
    precession_deg: Optional[float] = None,
    n_azim_samples: int = 12
) -> np.ndarray:
    """
    Vectorised computation of Bragg residuals for many reflections.

    Args:
        R0: 3x3 rotation
        alphas_deg: shape (N,) nominal α per reflection
        delta_alpha_deg_vals: shape (N,) δα(α) per reflection
        betas_deg_vals: shape (N,) β(α) per reflection
        n_hats_cryst: shape (N,3) unit normals in crystal frame
        thetas_rad: shape (N,) Bragg angles
        use_precession: switch to precession-aware residual
        precession_deg: half-cone angle (deg); if None, taken as 0
        n_azim_samples: ψ samples on the precession cone

    Returns:
        residuals: shape (N,) floats
    """
    N = int(len(alphas_deg))
    res = np.empty(N, dtype=float)
    if not use_precession or (precession_deg is None) or precession_deg <= 0.0:
        for i in range(N):
            res[i] = bragg_residual_scalar(
                R0,
                float(alphas_deg[i]),
                float(delta_alpha_deg_vals[i]),
                float(betas_deg_vals[i]),
                n_hats_cryst[i, :],
                float(thetas_rad[i]),
            )
        return res

    for i in range(N):
        res[i] = bragg_residual_scalar_with_precession(
            R0,
            float(alphas_deg[i]),
            float(delta_alpha_deg_vals[i]),
            float(betas_deg_vals[i]),
            n_hats_cryst[i, :],
            float(thetas_rad[i]),
            precession_deg=float(precession_deg),
            n_azim_samples=int(n_azim_samples),
        )
    return res


def evaluate_zone_axis_residuals_batch(
    R0: np.ndarray,
    alphas_deg: np.ndarray,
    delta_alpha_deg_vals: np.ndarray,
    betas_deg_vals: np.ndarray,
    measured_uvws: np.ndarray
) -> np.ndarray:
    """
    Vectorised computation of zone-axis residuals (radians) for many frames.

    Args:
        measured_uvws: shape (M,3) (u,v,w) vectors from PETS zone-axis table.

    Returns:
        residuals_rad: shape (M,)
    """
    M = int(len(alphas_deg))
    out = np.empty(M, dtype=float)
    for i in range(M):
        out[i] = zone_axis_residual_scalar(
            R0,
            float(alphas_deg[i]),
            float(delta_alpha_deg_vals[i]),
            float(betas_deg_vals[i]),
            measured_uvws[i, :],
        )
    return out


# ------------------------- Data preparation helpers --------------------------

def link_reflections_with_zone_alpha_beta(
    reflections_df: pd.DataFrame,
    zone_axes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge α and β onto reflections via zone_axis_id.
    Keeps rows where both α and β are present.
    """
    z = zone_axes_df[["zone_axis_id", "alpha_deg", "beta_deg", "precession_deg"]].dropna()
    z = z.rename(columns={"alpha_deg": "alpha_deg_zone", "beta_deg": "beta_deg_zone", "precession_deg": "precession_deg_zone"})
    merged = reflections_df.merge(z, how="left", on="zone_axis_id")
    merged = merged.dropna(subset=["alpha_deg_zone", "beta_deg_zone"])
    return merged


# ------------------------------- Preview driver ------------------------------

def main_preview(experiment_path: str, cif_path: str, n_reflections_preview: int = 2000) -> None:
    """
    Non-optimising preview to sanity-check residual scales with δα=β=0.

    Steps:
      - Parse files.
      - Build B consistent with UB convention.
      - Compute plane normals n̂ and θ for a subset of reflections.
      - Evaluate Bragg residuals (simple and precession-aware) with δα=β=0.
      - Evaluate zone-axis residuals (δα=β=0).
    """
    # Parse
    pets_header, UB, wavelength, zone_axes_df, reflections_df = parse_pets_cif_pets(experiment_path)
    _cif_header, cif_cell, _sg, _atoms = parse_reference_cif(cif_path)

    # Use PETS cell for consistency with UB
    cell = {
        "a": float(pets_header.get("_cell_length_a", "nan")),
        "b": float(pets_header.get("_cell_length_b", "nan")),
        "c": float(pets_header.get("_cell_length_c", "nan")),
        "alpha": float(pets_header.get("_cell_angle_alpha", "90")),
        "beta": float(pets_header.get("_cell_angle_beta", "90")),
        "gamma": float(pets_header.get("_cell_angle_gamma", "90")),
    }

    # UB vs B to infer 2π convention
    report = ub_b_consistency_report(
        UB, cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"]
    )
    include_2pi = infer_ub_2pi_convention(report["scale_ratio_vs_no2pi"], report["scale_ratio_vs_2pi"])
    B = build_B_matrix(cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"], include_2pi)

    # U ~ UB B^{-1} and project to R0
    U_raw = UB @ np.linalg.inv(B)
    # Project to nearest rotation
    U, S, Vt = np.linalg.svd(U_raw)
    R0 = U @ Vt
    if np.linalg.det(R0) < 0.0:
        U[:, -1] *= -1.0
        R0 = U @ Vt

    print("\n=== Stage 4 preview: residuals with δα=β=0 ===")
    print(f"Wavelength λ (Å): {wavelength:.5f}")
    print(f"Cell (PETS): {cell}")
    print(f"Using B include_2pi = {include_2pi}")

    # Link reflections with zone α,β
    merged = link_reflections_with_zone_alpha_beta(reflections_df, zone_axes_df)
    if merged.empty:
        print("No reflections linked to zone-axis α/β; cannot compute Bragg residuals.")
    else:
        n_use = min(int(n_reflections_preview), len(merged))
        sub = merged.iloc[:n_use].copy()

        # Build n̂ and θ
        n_hats = np.zeros((n_use, 3), dtype=float)
        thetas = np.zeros(n_use, dtype=float)
        for i in range(n_use):
            h, k, l = int(sub.iloc[i]["h"]), int(sub.iloc[i]["k"]), int(sub.iloc[i]["l"])
            n_hat = plane_normal_unit_from_B(B, h, k, l)
            dA = d_spacing_from_B(B, h, k, l, include_2pi=include_2pi)
            theta_rad, _ = bragg_angle_from_d(wavelength, dA)
            n_hats[i, :] = n_hat
            thetas[i] = theta_rad

        alphas = sub["alpha_deg_zone"].to_numpy(dtype=float)
        betas = sub["beta_deg_zone"].to_numpy(dtype=float)
        delta_alphas = np.zeros_like(alphas)

        # Bragg residuals (simple)
        r_simple = evaluate_bragg_residuals_batch(
            R0, alphas, delta_alphas, betas, n_hats, thetas,
            use_precession=False
        )
        print(f"\nBragg residuals (no precession) on N={len(r_simple)} spots:")
        print(f"  mean = {r_simple.mean(): .6e}")
        print(f"  std  = {r_simple.std(): .6e}")
        print(f"  p90(|r|) = {np.percentile(np.abs(r_simple), 90): .6e}")

        # Bragg residuals (precession-aware), if precession info present
        if "precession_deg" in zone_axes_df.columns and zone_axes_df["precession_deg"].notna().any():
            gamma = float(zone_axes_df["precession_deg"].dropna().iloc[0])
        else:
            gamma = float(pets_header.get("_diffrn_zone_axis_precession_angle", "0") or 0.0)
        if gamma > 0.0:
            r_ped = evaluate_bragg_residuals_batch(
                R0, alphas, delta_alphas, betas, n_hats, thetas,
                use_precession=True, precession_deg=gamma, n_azim_samples=12
            )
            print(f"\nBragg residuals (precession-aware, γ={gamma:.3f}°):")
            print(f"  mean = {r_ped.mean(): .6e}")
            print(f"  std  = {r_ped.std(): .6e}")
            print(f"  p90(|r|) = {np.percentile(np.abs(r_ped), 90): .6e}")
        else:
            print("\nPrecession angle not provided (>0); skipped precession-aware residuals.")

    # Zone-axis residuals across frames with δα=β=0
    if zone_axes_df.empty:
        print("\nZone-axis table empty; skipping zone-axis residuals.")
        return

    alphas_frames = zone_axes_df["alpha_deg"].to_numpy(dtype=float)
    betas_frames = zone_axes_df["beta_deg"].fillna(0.0).to_numpy(dtype=float)
    delta_frames = np.zeros_like(alphas_frames)
    uvw = zone_axes_df[["u", "v", "w"]].to_numpy(dtype=float)
    rz = evaluate_zone_axis_residuals_batch(R0, alphas_frames, delta_frames, betas_frames, uvw)
    print(f"\nZone-axis angular misfit (δα=β=0) on M={len(rz)} frames:")
    print(f"  mean = {np.degrees(rz).mean(): .6f} deg")
    print(f"  p90  = {np.degrees(np.percentile(rz, 90)): .6f} deg")
    print(f"  max  = {np.degrees(rz).max(): .6f} deg")


# -------------------------------- Entrypoint ---------------------------------

if __name__ == "__main__":
    # Adjust paths as needed
    EXPERIMENT_PATH = "Si_3_dyn.cif_pets"
    CIF_PATH = "silicon_structure.cif"
    main_preview(EXPERIMENT_PATH, CIF_PATH, n_reflections_preview=2000)
