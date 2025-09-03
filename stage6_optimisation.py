#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 6: Joint optimisation of R0, δα(α), and β(α) with robust loss and smoothness.

This module:
- Builds the full optimisation problem from PETS + CIF.
- Parameterises δα(α) and β(α) using either Fourier (default) or clamped cubic B-splines.
- Uses robust nonlinear least squares to fit:
    params = [ rvec(3) ⊕ c_delta_alpha (M1) ⊕ c_beta (M2) ]
  where R0 = Exp(rvec) @ R0_init.

Residual blocks (stacked):
  1) Reflections (Bragg):     w_j * [ Z · ( R_y(α+δα) R_x(β) R0 n̂_hkl ) - sin θ_hkl ].
  2) Zone-axis (angular):     sqrt(μ) * angle( z_pred_cryst(α,δα,β), z_meas_uvw ), in radians.
  3) Smoothness penalties:    sqrt(λ_*) * ( L_* @ c_* ), (spline) or sqrt(λ_*) * ( W_* @ c_* ) (Fourier).
  4) Mean-zero δα gauge:      sqrt(κ) * ( mean_α Φ_δα @ c_δα ).

All functions accept explicit arguments; there are no closures or implicit globals.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# --- External dependencies ---
from scipy.optimize import least_squares  # robust nonlinear least squares

# Stages 1–5
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
    from stage3_initial_orientation import project_to_SO3, rotation_matrix_y, rotation_matrix_x
    from stage4_model_residuals import (
        evaluate_bragg_residuals_batch,
        evaluate_zone_axis_residuals_batch,
        link_reflections_with_zone_alpha_beta,
    )
    from stage5_parametrisations import (
        Fourier1D, fourier_design_matrix, fourier_harmonic_weights,
        Spline1D, bspline_design_matrix, bspline_second_diff_penalty_matrix, _HAVE_SCIPY
    )
except Exception as e:
    raise ImportError(
        "Please ensure Stages 1–5 are available in the same directory:\n"
        "  stage1_parse_inputs.py, stage2_reciprocal_tools.py, stage3_initial_orientation.py,\n"
        "  stage4_model_residuals.py, stage5_parametrisations.py"
    ) from e


# ============================ SO(3) utilities ================================

def exp_so3(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map on SO(3) via Rodrigues' formula.
    Args:
        omega: shape (3,), rotation vector (radians)
    Returns:
        R = Exp([ω]) ∈ SO(3)
    """
    wx, wy, wz = float(omega[0]), float(omega[1]), float(omega[2])
    theta = math.sqrt(wx*wx + wy*wy + wz*wz)
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    kx, ky, kz = wx/theta, wy/theta, wz/theta
    K = np.array([[ 0.0, -kz,  ky],
                  [ kz,  0.0, -kx],
                  [-ky,  kx,  0.0]], dtype=float)
    c = math.cos(theta)
    s = math.sin(theta)
    return np.eye(3) * c + (1.0 - c) * np.outer([kx,ky,kz], [kx,ky,kz]) + s * K


# ============================ Data containers ================================

@dataclass
class ProblemData:
    """
    All numeric arrays needed by the residual function (explicitly passed).
    """
    # Orientation seed
    R0_init: np.ndarray          # 3x3
    # Reflection data (size N)
    alphas_ref: np.ndarray       # (N,)
    betas_ref: np.ndarray        # (N,)
    n_hats_ref: np.ndarray       # (N,3)
    thetas_ref: np.ndarray       # (N,) in radians
    weights_ref: np.ndarray      # (N,) weight per reflection residual
    # Zone-axis data (size M)
    alphas_zone: np.ndarray      # (M,)
    betas_zone: np.ndarray       # (M,)
    uvw_zone: np.ndarray         # (M,3) measured (u,v,w) in crystal frame
    # Precession handling
    use_precession: bool
    precession_deg: float
    n_azim_samples: int
    # Design matrices for δα and β (reflections + zone)
    Phi_da_ref: np.ndarray       # (N,M1)
    Phi_beta_ref: np.ndarray     # (N,M2)
    Phi_da_zone: np.ndarray      # (M,M1)
    Phi_beta_zone: np.ndarray    # (M,M2)
    # Smoothness operators / weights
    L_da: Optional[np.ndarray]   # (K1,M1) or None
    L_beta: Optional[np.ndarray] # (K2,M2) or None
    W_da: Optional[np.ndarray]   # (M1,M1) diag matrix (Fourier); may be None
    W_beta: Optional[np.ndarray] # (M2,M2) diag matrix (Fourier); may be None


@dataclass
class HyperParams:
    """
    Scalar hyperparameters for weighting residual blocks.
    """
    mu_zone: float = 1.0               # relative weight for zone-axis angular residuals
    lambda_da_smooth: float = 1e-2     # smoothness strength for δα
    lambda_beta_smooth: float = 1e-2   # smoothness strength for β
    lambda_da_mean0: float = 1e-3      # mean-zero gauge for δα
    lambda_beta_mag: float = 1e-4      # small Tikhonov on β coefficients (keeps it small)
    robust_loss: str = "soft_l1"       # 'linear','soft_l1','huber','cauchy','arctan'
    robust_f_scale: float = 1.0        # scale for robust loss functions


# =========================== Problem building ================================

def build_problem_from_files(
    experiment_path: str,
    cif_path: str,
    basis_type: str = "fourier",  # 'fourier' or 'spline'
    # Fourier options
    fourier_harmonics_da: int = 3,
    fourier_harmonics_beta: int = 3,
    # Spline options
    spline_spacing_deg_da: float = 3.0,
    spline_spacing_deg_beta: float = 3.0,
    # Reflection selection / sampling
    strength_metric: str = "snr",             # 'snr' or 'intensity'
    min_strength: Optional[float] = None,     # absolute threshold on chosen metric
    top_strength_percent: Optional[float] = None,  # keep top X% by metric
    max_reflections_fit: Optional[int] = None,
    random_state: int = 42,
    # Precession residual option
    use_precession: bool = False,
    n_azim_samples: int = 12
) -> Tuple[ProblemData, Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Parse inputs, construct B and R0_init, assemble all arrays for optimisation.

    Returns:
        problem: ProblemData
        meta: dict with wavelength, include_2pi, cell values
        B, R0_init, reflections_df (subset), zone_axes_df
    """
    # ------------- Parse inputs -------------
    pets_header, UB, wavelength, zone_axes_df, reflections_df = parse_pets_cif_pets(experiment_path)
    _cif_header, cif_cell, _space_group, _atoms = parse_reference_cif(cif_path)

    # Use PETS cell for consistency with UB
    cell = {
        "a": float(pets_header.get("_cell_length_a", "nan")),
        "b": float(pets_header.get("_cell_length_b", "nan")),
        "c": float(pets_header.get("_cell_length_c", "nan")),
        "alpha": float(pets_header.get("_cell_angle_alpha", "90")),
        "beta": float(pets_header.get("_cell_angle_beta", "90")),
        "gamma": float(pets_header.get("_cell_angle_gamma", "90")),
    }

    # UB vs B convention → include_2pi
    report = ub_b_consistency_report(
        UB, cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"]
    )
    include_2pi = infer_ub_2pi_convention(report["scale_ratio_vs_no2pi"], report["scale_ratio_vs_2pi"])
    B = build_B_matrix(cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"], include_2pi)

    # Initial R0 from UB and B, projected to SO(3)
    U_raw = UB @ np.linalg.inv(B)
    R0_init, detU, ortho_err = project_to_SO3(U_raw)

    # Link reflections to zone α/β
    merged = link_reflections_with_zone_alpha_beta(reflections_df, zone_axes_df)
    if merged.empty:
        raise RuntimeError("No reflections could be linked to zone-axis α/β.")

    # -------- Strength-based reflection selection (speed-up) --------
    # Build the chosen strength metric
    eps = 1e-6
    if strength_metric.lower() == "snr":
        strength_vals = merged["I"].to_numpy(dtype=float) / np.maximum(merged["sigma"].to_numpy(dtype=float), eps)
        metric_name = "SNR (I/sigma)"
    elif strength_metric.lower() == "intensity":
        strength_vals = merged["I"].to_numpy(dtype=float)
        metric_name = "intensity (I)"
    else:
        raise ValueError("strength_metric must be 'snr' or 'intensity'.")

    # Absolute threshold (keep reflections with metric >= min_strength)
    if min_strength is not None:
        mask = strength_vals >= float(min_strength)
        merged = merged.loc[mask].copy()

    # Top X% by metric
    if (top_strength_percent is not None) and (len(merged) > 0):
        pct = float(top_strength_percent)
        if not (0.0 < pct <= 100.0):
            raise ValueError("top_strength_percent must be in (0, 100].")
        # Recompute strength on the (maybe) thresholded subset
        if strength_metric.lower() == "snr":
            strength_vals = merged["I"].to_numpy(dtype=float) / np.maximum(merged["sigma"].to_numpy(dtype=float), eps)
        else:
            strength_vals = merged["I"].to_numpy(dtype=float)
        cutoff = np.percentile(strength_vals, 100.0 - pct)
        mask = strength_vals >= cutoff
        merged = merged.loc[mask].copy()

    if merged.empty:
        raise RuntimeError("After strength filtering, no reflections remain. Loosen thresholds.")


    # Optional sampling to limit problem size
    if (max_reflections_fit is not None) and (len(merged) > int(max_reflections_fit)):
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(len(merged), size=int(max_reflections_fit), replace=False)
        merged = merged.iloc[np.sort(idx)].copy()

    # Build reflection arrays
    N = len(merged)
    alphas_ref = merged["alpha_deg_zone"].to_numpy(dtype=float)
    betas_ref  = merged["beta_deg_zone"].fillna(0.0).to_numpy(dtype=float)

    n_hats_ref = np.zeros((N, 3), dtype=float)
    thetas_ref = np.zeros(N, dtype=float)
    for i in range(N):
        h, k, l = int(merged.iloc[i]["h"]), int(merged.iloc[i]["k"]), int(merged.iloc[i]["l"])
        n_hat = plane_normal_unit_from_B(B, h, k, l)
        dA = d_spacing_from_B(B, h, k, l, include_2pi=include_2pi)
        theta_rad, _theta_deg = bragg_angle_from_d(wavelength, dA)
        n_hats_ref[i, :] = n_hat
        thetas_ref[i] = theta_rad

    # Reflection weights: use 1 / (sigma + eps), capped
    sig = merged["sigma"].to_numpy(dtype=float)
    w_ref = 1.0 / np.maximum(sig, 1e-6)
    w_ref = np.clip(w_ref, 0.0, np.percentile(w_ref, 95))  # cap extremes
    # Normalise weight scale near 1
    if np.mean(w_ref) > 0:
        w_ref /= float(np.mean(w_ref))

    # Zone-axis arrays
    M = len(zone_axes_df)
    alphas_zone = zone_axes_df["alpha_deg"].to_numpy(dtype=float)
    betas_zone  = zone_axes_df["beta_deg"].fillna(0.0).to_numpy(dtype=float)
    uvw_zone    = zone_axes_df[["u", "v", "w"]].to_numpy(dtype=float)

    # Precession angle (take first non-NaN)
    if "precession_deg" in zone_axes_df.columns and zone_axes_df["precession_deg"].notna().any():
        precession_deg = float(zone_axes_df["precession_deg"].dropna().iloc[0])
    else:
        precession_deg = float(pets_header.get("_diffrn_zone_axis_precession_angle", "0") or 0.0)

    # Design matrices for δα and β
    alpha_min = float(np.min(alphas_zone))
    alpha_max = float(np.max(alphas_zone))

    if basis_type.lower() == "fourier":
        # δα
        f_da = Fourier1D.create(alpha_min, alpha_max, n_harmonics=fourier_harmonics_da, include_bias=True, include_linear=True)
        Phi_da_zone = f_da.design_matrix(alphas_zone)
        Phi_da_ref  = f_da.design_matrix(alphas_ref)
        W_da = np.diag(fourier_harmonic_weights(f_da.n_harmonics, include_bias=f_da.include_bias, include_linear=f_da.include_linear, weight_scale=1.0))
        L_da = None
        # β
        f_be = Fourier1D.create(alpha_min, alpha_max, n_harmonics=fourier_harmonics_beta, include_bias=True, include_linear=True)
        Phi_beta_zone = f_be.design_matrix(alphas_zone)
        Phi_beta_ref  = f_be.design_matrix(alphas_ref)
        W_beta = np.diag(fourier_harmonic_weights(f_be.n_harmonics, include_bias=f_be.include_bias, include_linear=f_be.include_linear, weight_scale=1.0))
        L_beta = None
        M1 = Phi_da_zone.shape[1]
        M2 = Phi_beta_zone.shape[1]
    elif basis_type.lower() == "spline":
        if not _HAVE_SCIPY:
            raise ImportError("Spline basis requested but SciPy was not found. Install 'scipy' or use basis_type='fourier'.")
        # δα
        s_da = Spline1D.create_uniform(alpha_min, alpha_max, target_spacing_deg=float(spline_spacing_deg_da), degree=3)
        Phi_da_zone = s_da.design_matrix(alphas_zone)
        Phi_da_ref  = s_da.design_matrix(alphas_ref)
        L_da = bspline_second_diff_penalty_matrix(len(s_da.coeffs), scale=1.0)
        W_da = None
        # β
        s_be = Spline1D.create_uniform(alpha_min, alpha_max, target_spacing_deg=float(spline_spacing_deg_beta), degree=3)
        Phi_beta_zone = s_be.design_matrix(alphas_zone)
        Phi_beta_ref  = s_be.design_matrix(alphas_ref)
        L_beta = bspline_second_diff_penalty_matrix(len(s_be.coeffs), scale=1.0)
        W_beta = None
        M1 = Phi_da_zone.shape[1]
        M2 = Phi_beta_zone.shape[1]
    else:
        raise ValueError("basis_type must be 'fourier' or 'spline'.")

    problem = ProblemData(
        R0_init=R0_init,
        alphas_ref=alphas_ref,
        betas_ref=betas_ref,
        n_hats_ref=n_hats_ref,
        thetas_ref=thetas_ref,
        weights_ref=w_ref,
        alphas_zone=alphas_zone,
        betas_zone=betas_zone,
        uvw_zone=uvw_zone,
        use_precession=bool(use_precession),
        precession_deg=float(precession_deg),
        n_azim_samples=int(n_azim_samples),
        Phi_da_ref=Phi_da_ref,
        Phi_beta_ref=Phi_beta_ref,
        Phi_da_zone=Phi_da_zone,
        Phi_beta_zone=Phi_beta_zone,
        L_da=L_da,
        L_beta=L_beta,
        W_da=W_da,
        W_beta=W_beta
    )

    meta = {
        "wavelength": float(wavelength),
        "include_2pi": float(include_2pi),
        "cell_a": cell["a"], "cell_b": cell["b"], "cell_c": cell["c"],
        "cell_alpha": cell["alpha"], "cell_beta": cell["beta"], "cell_gamma": cell["gamma"]
    }

    return problem, meta, B, R0_init, merged, zone_axes_df


# ============================== Param packing ================================

def pack_params(rvec: np.ndarray, c_da: np.ndarray, c_be: np.ndarray) -> np.ndarray:
    """Pack parameter vector."""
    return np.concatenate([rvec.reshape(3), c_da.reshape(-1), c_be.reshape(-1)]).astype(float)

def unpack_params(x: np.ndarray, M1: int, M2: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack parameter vector into (rvec, c_da, c_beta)."""
    rvec = x[:3]
    c_da = x[3:3+M1]
    c_be = x[3+M1:3+M1+M2]
    return rvec, c_da, c_be


# ============================== Residual stack ===============================

def build_residuals(
    x: np.ndarray,
    problem: ProblemData,
    hparams: HyperParams
) -> np.ndarray:
    """
    Assemble the stacked residual vector for least_squares.
    """
    # Unpack
    M1 = problem.Phi_da_zone.shape[1]
    M2 = problem.Phi_beta_zone.shape[1]
    rvec, c_da, c_be = unpack_params(x, M1, M2)

    # Current R0
    R0 = exp_so3(rvec) @ problem.R0_init

    # Evaluate δα and β at ref and zone α
    da_ref  = problem.Phi_da_ref  @ c_da
    be_ref  = problem.Phi_beta_ref @ c_be
    da_zone = problem.Phi_da_zone @ c_da
    be_zone = problem.Phi_beta_zone @ c_be

    # 1) Reflections: Bragg residuals (N,)
    r_bragg = evaluate_bragg_residuals_batch(
        R0,
        problem.alphas_ref,
        da_ref,
        problem.betas_ref + be_ref,    # additive perturbation to reported β
        problem.n_hats_ref,
        problem.thetas_ref,
        use_precession=problem.use_precession,
        precession_deg=problem.precession_deg,
        n_azim_samples=problem.n_azim_samples
    )
    # Apply per-reflection weights (scale residuals)
    r1 = problem.weights_ref * r_bragg

    # 2) Zone-axis angular residuals in radians (M,)
    rz = evaluate_zone_axis_residuals_batch(
        R0,
        problem.alphas_zone,
        da_zone,
        problem.betas_zone + be_zone,
        problem.uvw_zone
    )
    r2 = math.sqrt(max(hparams.mu_zone, 0.0)) * rz

    # 3) Smoothness penalties
    smooth_list: List[float] = []
    if problem.L_da is not None and hparams.lambda_da_smooth > 0.0:
        smooth_da = math.sqrt(hparams.lambda_da_smooth) * (problem.L_da @ c_da)
        smooth_list.append(smooth_da)
    if problem.W_da is not None and hparams.lambda_da_smooth > 0.0:
        Wsqrt_da = np.sqrt(np.maximum(np.diag(problem.W_da), 0.0))
        smooth_da = math.sqrt(hparams.lambda_da_smooth) * (Wsqrt_da * c_da)
        smooth_list.append(smooth_da)
    if problem.L_beta is not None and hparams.lambda_beta_smooth > 0.0:
        smooth_be = math.sqrt(hparams.lambda_beta_smooth) * (problem.L_beta @ c_be)
        smooth_list.append(smooth_be)
    if problem.W_beta is not None and hparams.lambda_beta_smooth > 0.0:
        Wsqrt_be = np.sqrt(np.maximum(np.diag(problem.W_beta), 0.0))
        smooth_be = math.sqrt(hparams.lambda_beta_smooth) * (Wsqrt_be * c_be)
        smooth_list.append(smooth_be)

    # 4) Mean-zero gauge for δα (so average δα over frames is ~0)
    r_mean_da = np.array([0.0], dtype=float)
    if hparams.lambda_da_mean0 > 0.0:
        # Use zone grid (uniform-ish) to define the mean
        mean_da = float(np.mean(da_zone))
        r_mean_da = math.sqrt(hparams.lambda_da_mean0) * np.array([mean_da], dtype=float)

    # 5) Small magnitude prior on β coefficients (keeps β small)
    r_beta_mag = np.array([], dtype=float)
    if hparams.lambda_beta_mag > 0.0:
        r_beta_mag = math.sqrt(hparams.lambda_beta_mag) * c_be

    # Stack all blocks
    stack = [r1, r2]
    if len(smooth_list) > 0:
        stack.extend(smooth_list)
    stack.append(r_mean_da)
    if r_beta_mag.size > 0:
        stack.append(r_beta_mag)
    return np.concatenate(stack).astype(float)


# ================================ Optimiser ==================================

def run_optimisation(
    problem: ProblemData,
    hparams: HyperParams,
    max_nfev: int = 100,
    xtol: float = 1e-10,
    ftol: float = 1e-10,
    gtol: float = 1e-10
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Solve min_x ρ(||r(x)||^2) with robust loss using scipy.optimize.least_squares.
    Returns:
        x_opt: optimal parameter vector
        info: dict of status, nfev, cost, etc.
    """
    # Initial params: rvec=0, c_da=0, c_beta=0
    M1 = problem.Phi_da_zone.shape[1]
    M2 = problem.Phi_beta_zone.shape[1]
    x0 = pack_params(np.zeros(3), np.zeros(M1), np.zeros(M2))

    # Objective wrapper (explicit)
    def fun_for_solver(x: np.ndarray) -> np.ndarray:
        return build_residuals(x, problem, hparams)

    res = least_squares(
        fun_for_solver, x0,
        loss=hparams.robust_loss, f_scale=hparams.robust_f_scale,
        max_nfev=int(max_nfev),
        xtol=float(xtol), ftol=float(ftol), gtol=float(gtol),
        method="trf", jac="2-point"  # robust and stable
    )

    info = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "cost": float(res.cost),               # 0.5 * sum(residuals**2)
        "optimality": float(res.optimality),
        "active_mask_sum": int(np.sum(res.active_mask != 0))
    }
    return res.x.astype(float), info


# ================================ Main driver ================================

def main(
    experiment_path: str = "Si_3_dyn.cif_pets",
    cif_path: str = "silicon_structure.cif",
    basis_type: str = "fourier",      # 'fourier' or 'spline'
    # Basis settings:
    fourier_harmonics_da: int = 3,
    fourier_harmonics_beta: int = 3,
    spline_spacing_deg_da: float = 3.0,
    spline_spacing_deg_beta: float = 3.0,
    # Selection / speed:
    strength_metric: str = "snr",             # 'snr' or 'intensity'
    min_strength: Optional[float] = None,     # e.g. 5.0 for SNR, or 5000 for intensity
    top_strength_percent: Optional[float] = 10.0,  # keep top 10% strongest by default
    max_reflections_fit: Optional[int] = None,     # optional cap after strength filter
    random_state: int = 42,
    use_precession: bool = False,     # set True if you see bias without it
    n_azim_samples: int = 12,
    # Hyperparameters:
    mu_zone: float = 1.0,
    lambda_da_smooth: float = 1e-2,
    lambda_beta_smooth: float = 1e-2,
    lambda_da_mean0: float = 1e-3,
    lambda_beta_mag: float = 1e-4,
    robust_loss: str = "soft_l1",
    robust_f_scale: float = 1.0,
    # Solver stopping:
    max_nfev: int = 100,
    xtol: float = 1e-10, ftol: float = 1e-10, gtol: float = 1e-10,
    # Output:
    results_npz_path: str = "stage6_fit_results.npz"
) -> None:
    """
    Stage 6 optimisation end-to-end. Produces an NPZ you can load in Stage 7 for plotting.
    """
    # ---- Build problem ----
    problem, meta, B, R0_init, reflections_df, zone_axes_df = build_problem_from_files(
        experiment_path=experiment_path,
        cif_path=cif_path,
        basis_type=basis_type,
        fourier_harmonics_da=fourier_harmonics_da,
        fourier_harmonics_beta=fourier_harmonics_beta,
        spline_spacing_deg_da=spline_spacing_deg_da,
        spline_spacing_deg_beta=spline_spacing_deg_beta,
        strength_metric=strength_metric,
        min_strength=min_strength,
        top_strength_percent=top_strength_percent,
        max_reflections_fit=max_reflections_fit,
        random_state=random_state,
        use_precession=use_precession,
        n_azim_samples=n_azim_samples
    )


    # ---- Hyperparameters ----
    hparams = HyperParams(
        mu_zone=float(mu_zone),
        lambda_da_smooth=float(lambda_da_smooth),
        lambda_beta_smooth=float(lambda_beta_smooth),
        lambda_da_mean0=float(lambda_da_mean0),
        lambda_beta_mag=float(lambda_beta_mag),
        robust_loss=str(robust_loss),
        robust_f_scale=float(robust_f_scale)
    )

    print("\n=== Stage 6: Optimisation ===")
    print(f"Basis: {basis_type}")
    print(f"Reflections used: {len(reflections_df)}   Zone frames: {len(zone_axes_df)}")
    print(f"Precession-aware residuals: {use_precession} (γ={problem.precession_deg:.3f}°, samples={problem.n_azim_samples})")
    print(f"Hyperparams: {hparams}")

    # ---- Solve ----
    x_opt, info = run_optimisation(problem, hparams, max_nfev=max_nfev, xtol=xtol, ftol=ftol, gtol=gtol)
    print("\nSolver info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # ---- Unpack solution & evaluate curves on zone α grid ----
    M1 = problem.Phi_da_zone.shape[1]
    M2 = problem.Phi_beta_zone.shape[1]
    rvec_opt, c_da_opt, c_be_opt = unpack_params(x_opt, M1, M2)
    R0_opt = exp_so3(rvec_opt) @ problem.R0_init

    da_zone = problem.Phi_da_zone @ c_da_opt      # degrees (since basis models were built in degrees)
    be_zone = problem.Phi_beta_zone @ c_be_opt

    # ---- Save NPZ for Stage 7 ----
    np.savez_compressed(
        results_npz_path,
        R0_init=problem.R0_init,
        R0_opt=R0_opt,
        rvec_opt=rvec_opt,
        c_da_opt=c_da_opt,
        c_be_opt=c_be_opt,
        alphas_zone=problem.alphas_zone,
        delta_alpha_zone_deg=da_zone,
        beta_zone_deg=be_zone,
        meta=np.array(list(meta.items()), dtype=object),
        basis_type=basis_type,
        Phi_da_zone=problem.Phi_da_zone,
        Phi_beta_zone=problem.Phi_beta_zone
    )
    print(f"\nSaved optimisation results to: {results_npz_path}")

    # ---- Quick textual summary ----
    print("\nFitted curves on zone α-grid (preview):")
    print(f"  δα: mean={np.mean(da_zone): .6f}°, std={np.std(da_zone): .6f}°, min={np.min(da_zone): .6f}°, max={np.max(da_zone): .6f}°")
    print(f"  β : mean={np.mean(be_zone): .6f}°, std={np.std(be_zone): .6f}°, min={np.min(be_zone): .6f}°, max={np.max(be_zone): .6f}°")


# -------------------------------- Entrypoint ---------------------------------

if __name__ == "__main__":
    main(
        experiment_path="Si_3_dyn.cif_pets",
        cif_path="silicon_structure.cif",
        basis_type="fourier",
        fourier_harmonics_da=8,
        fourier_harmonics_beta=8,
        # Keep only strong reflections to speed up:
        strength_metric="snr",          # 'snr' or 'intensity'
        min_strength=5,              # keeps SNR ≥ [x] or min intensity of x (choose strength_metric using line above)
        top_strength_percent=20,      # then keep top [x%] of those limited by min_strength
        max_reflections_fit=None,       # optionally also cap, e.g., 50000
        use_precession=None,
        n_azim_samples=12,
        # Hyperparameters:
        mu_zone=1.0,                   # Zone-axis weight - higher value gives graeter weight to reflection data
        lambda_da_smooth=1e-4,
        lambda_beta_smooth=1e-4,
        lambda_da_mean0=1e-3,
        lambda_beta_mag=1e-4,
        robust_loss="soft_l1",
        robust_f_scale=1.0,
        max_nfev=120,
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        results_npz_path="stage6_fit_results.npz"
    )
