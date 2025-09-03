#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 7: Plots, QC, and CSV export for cRED refinement results.

Reads the NPZ saved by Stage 6 and:
  - Plots δα(α) and β(α) vs α (separate figures).
  - Recomputes Bragg residuals and zone-axis misfits BEFORE (δα=β=0) and AFTER
    the fit by interpolating the fitted curves to reflection α values.
  - Plots residual diagnostics (histograms, residual vs α).
  - Exports a CSV with columns: alpha_deg, delta_alpha_deg, beta_deg.

Design notes:
  - All functions are top-level and take explicit arguments.
  - No reliance on Stage 6's internal basis; we use saved α-grid and curves,
    and interpolate to reflection α values for residuals.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse parsers and geometry utilities
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
    from stage3_initial_orientation import project_to_SO3
    from stage4_model_residuals import (
        evaluate_bragg_residuals_batch,
        evaluate_zone_axis_residuals_batch,
        link_reflections_with_zone_alpha_beta,
    )
except Exception as e:
    raise ImportError(
        "Please ensure Stages 1–4 are available in the same directory:\n"
        "  stage1_parse_inputs.py, stage2_reciprocal_tools.py,\n"
        "  stage3_initial_orientation.py, stage4_model_residuals.py"
    ) from e


# ----------------------------- NPZ utilities --------------------------------

def load_stage6_npz(npz_path: str) -> Dict[str, object]:
    """
    Load NPZ saved by Stage 6 and return a dict of arrays/values.
    """
    data = np.load(npz_path, allow_pickle=True)
    out: Dict[str, object] = {}
    for k in data.files:
        out[k] = data[k]
    return out


def meta_array_to_dict(meta_arr: np.ndarray) -> Dict[str, float]:
    """
    Convert the 'meta' array saved by Stage 6 back into a dict.
    """
    meta: Dict[str, float] = {}
    for k, v in meta_arr.tolist():
        try:
            meta[str(k)] = float(v)
        except Exception:
            try:
                meta[str(k)] = v
            except Exception:
                pass
    return meta


# ----------------------------- Interpolation ---------------------------------

def interp_to_alphas(alpha_query: np.ndarray,
                     alpha_grid: np.ndarray,
                     values_on_grid: np.ndarray) -> np.ndarray:
    """
    Interpolate values defined on (alpha_grid) to (alpha_query).
    Extrapolates as constants beyond the grid endpoints.
    """
    alpha_q = np.asarray(alpha_query, dtype=float).reshape(-1)
    grid = np.asarray(alpha_grid, dtype=float).reshape(-1)
    vals = np.asarray(values_on_grid, dtype=float).reshape(-1)
    # Clamp/extrapolate with edge values
    left_val = float(vals[0])
    right_val = float(vals[-1])
    y = np.interp(alpha_q, grid, vals, left=left_val, right=right_val)
    return y


# ----------------------------- Plot helpers ----------------------------------

def plot_curve(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str,
               out_png: Optional[str] = None) -> None:
    """
    Create a single simple line plot (no seaborn, no styles).
    """
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.5)
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")


def plot_histogram(values: np.ndarray, bins: int, xlabel: str, title: str,
                   out_png: Optional[str] = None) -> None:
    """
    Plot a histogram of values.
    """
    plt.figure()
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.5)
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")


def plot_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str,
                 out_png: Optional[str] = None) -> None:
    """
    Scatter plot (e.g., residual vs α).
    """
    plt.figure()
    plt.scatter(x, y, s=6, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.5)
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")


# ---------------------------- Residual recompute -----------------------------

def build_reflection_arrays_for_residuals(
    experiment_path: str,
    cif_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, bool, np.ndarray]:
    """
    Recompute arrays needed for residuals from files (independent of Stage 6 basis).
    Returns:
        alphas_ref: (N,) reflection α (deg) from zone table
        betas_ref:  (N,) reflection β (deg) from zone table
        n_hats_ref: (N,3) unit normals in crystal frame
        thetas_ref: (N,) Bragg angles (rad)
        wavelength: float (Å)
        alphas_zone_all: (M,) zone α (deg) full grid
        include_2pi: bool
        uvw_zone: (M,3) measured (u,v,w) for zone residuals
    """
    pets_header, UB, wavelength, zone_axes_df, reflections_df = parse_pets_cif_pets(experiment_path)
    # Build B consistent with UB
    cell = {
        "a": float(pets_header.get("_cell_length_a", "nan")),
        "b": float(pets_header.get("_cell_length_b", "nan")),
        "c": float(pets_header.get("_cell_length_c", "nan")),
        "alpha": float(pets_header.get("_cell_angle_alpha", "90")),
        "beta": float(pets_header.get("_cell_angle_beta", "90")),
        "gamma": float(pets_header.get("_cell_angle_gamma", "90")),
    }
    report = ub_b_consistency_report(UB, cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"])
    include_2pi = infer_ub_2pi_convention(report["scale_ratio_vs_no2pi"], report["scale_ratio_vs_2pi"])
    B = build_B_matrix(cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"], include_2pi)

    merged = link_reflections_with_zone_alpha_beta(reflections_df, zone_axes_df)
    if merged.empty:
        raise RuntimeError("No reflections linked to zone-axis α/β; cannot compute residuals.")

    N = len(merged)
    alphas_ref = merged["alpha_deg_zone"].to_numpy(dtype=float)
    betas_ref  = merged["beta_deg_zone"].fillna(0.0).to_numpy(dtype=float)

    n_hats_ref = np.zeros((N, 3), dtype=float)
    thetas_ref = np.zeros(N, dtype=float)
    for i in range(N):
        h, k, l = int(merged.iloc[i]["h"]), int(merged.iloc[i]["k"]), int(merged.iloc[i]["l"])
        n_hat = plane_normal_unit_from_B(B, h, k, l)
        dA = d_spacing_from_B(B, h, k, l, include_2pi=include_2pi)
        theta_rad, _ = bragg_angle_from_d(wavelength, dA)
        n_hats_ref[i, :] = n_hat
        thetas_ref[i] = theta_rad

    alphas_zone_all = zone_axes_df["alpha_deg"].to_numpy(dtype=float)
    uvw_zone = zone_axes_df[["u", "v", "w"]].to_numpy(dtype=float)

    return alphas_ref, betas_ref, n_hats_ref, thetas_ref, float(wavelength), alphas_zone_all, bool(include_2pi), uvw_zone


def compute_residuals_before_after(
    R0_opt: np.ndarray,
    alphas_ref: np.ndarray,
    betas_ref: np.ndarray,
    n_hats_ref: np.ndarray,
    thetas_ref: np.ndarray,
    alphas_zone_all: np.ndarray,
    uvw_zone: np.ndarray,
    alpha_grid: np.ndarray,
    delta_alpha_grid_deg: np.ndarray,
    beta_grid_deg: np.ndarray,
    use_precession: bool,
    precession_deg: float,
    n_azim_samples: int
) -> Dict[str, np.ndarray]:
    """
    Compute reflection and zone-axis residuals BEFORE (δα=β=0) and AFTER (fitted curves).
    """
    # BEFORE (δα=0, β_add=0)
    da_ref_before = np.zeros_like(alphas_ref)
    be_ref_before = np.zeros_like(alphas_ref)
    r_bragg_before = evaluate_bragg_residuals_batch(
        R0_opt, alphas_ref, da_ref_before, betas_ref + be_ref_before,
        n_hats_ref, thetas_ref, use_precession=use_precession,
        precession_deg=precession_deg, n_azim_samples=n_azim_samples
    )

    da_zone_before = np.zeros_like(alphas_zone_all)
    be_zone_before = np.zeros_like(alphas_zone_all)
    r_zone_before = evaluate_zone_axis_residuals_batch(
        R0_opt, alphas_zone_all, da_zone_before, be_zone_before, uvw_zone
    )

    # AFTER (interpolate δα, β from fitted zone grid to reflection αs)
    da_ref_after = interp_to_alphas(alphas_ref, alpha_grid, delta_alpha_grid_deg)
    be_ref_after = interp_to_alphas(alphas_ref, alpha_grid, beta_grid_deg)
    r_bragg_after = evaluate_bragg_residuals_batch(
        R0_opt, alphas_ref, da_ref_after, betas_ref + be_ref_after,
        n_hats_ref, thetas_ref, use_precession=use_precession,
        precession_deg=precession_deg, n_azim_samples=n_azim_samples
    )

    # Zone-axis AFTER: directly on the zone grid
    da_zone_after = delta_alpha_grid_deg.copy()
    be_zone_after = beta_grid_deg.copy()
    r_zone_after = evaluate_zone_axis_residuals_batch(
        R0_opt, alphas_zone_all, da_zone_after, be_zone_after, uvw_zone
    )

    return {
        "r_bragg_before": r_bragg_before,
        "r_bragg_after": r_bragg_after,
        "r_zone_before_rad": r_zone_before,
        "r_zone_after_rad": r_zone_after,
        "da_ref_after": da_ref_after,
        "be_ref_after": be_ref_after
    }


# ------------------------------- CSV export ----------------------------------

def export_alpha_delta_beta_csv(alpha_grid: np.ndarray,
                                delta_alpha_grid_deg: np.ndarray,
                                beta_grid_deg: np.ndarray,
                                csv_path: str) -> None:
    """
    Write α, δα, β to CSV.
    """
    df = pd.DataFrame({
        "alpha_deg": alpha_grid.astype(float),
        "delta_alpha_deg": delta_alpha_grid_deg.astype(float),
        "beta_deg": beta_grid_deg.astype(float),
    })
    df.to_csv(csv_path, index=False)


# ---------------------------------- Main -------------------------------------

def main(
    experiment_path: str = "Si_3_dyn.cif_pets",
    cif_path: str = "silicon_structure.cif",
    npz_path: str = "stage6_fit_results.npz",
    # Precession handling for residuals:
    use_precession: bool = False,
    n_azim_samples: int = 12,
    # Plots & export:
    csv_out: str = "delta_alpha_beta_vs_alpha.csv",
    fig_prefix: str = "stage7_",
    show_plots: bool = True
) -> None:
    """
    End-to-end Stage 7:
      - Load Stage 6 NPZ
      - Plot δα(α), β(α)
      - Recompute residuals pre/post fit, plot diagnostics
      - Export CSV
    """
    # ----- Load Stage 6 results -----
    bundle = load_stage6_npz(npz_path)
    R0_opt = np.asarray(bundle["R0_opt"], dtype=float)
    alpha_grid = np.asarray(bundle["alphas_zone"], dtype=float)
    delta_alpha_grid_deg = np.asarray(bundle["delta_alpha_zone_deg"], dtype=float)
    beta_grid_deg = np.asarray(bundle["beta_zone_deg"], dtype=float)
    meta = meta_array_to_dict(np.asarray(bundle["meta"], dtype=object))
    basis_type_saved = str(bundle["basis_type"]) if "basis_type" in bundle else "unknown"

    # If precession degree is known, try to read from the PETS zone table
    pets_header, UB, wavelength, zone_axes_df, reflections_df = parse_pets_cif_pets(experiment_path)
    if "precession_deg" in zone_axes_df.columns and zone_axes_df["precession_deg"].notna().any():
        precession_deg = float(zone_axes_df["precession_deg"].dropna().iloc[0])
    else:
        precession_deg = float(pets_header.get("_diffrn_zone_axis_precession_angle", "0") or 0.0)

    print("\n=== Stage 7: Plots & Export ===")
    print(f"Loaded: {npz_path}")
    print(f"Basis in Stage 6: {basis_type_saved}")
    print(f"α-grid length: {len(alpha_grid)}")
    print(f"Precession for residual eval: use_precession={use_precession}, γ={precession_deg:.3f}°, ψ-samples={n_azim_samples}")

    # ----- Plots: δα(α) and β(α) -----
    plot_curve(alpha_grid, delta_alpha_grid_deg,
               xlabel="α (deg)", ylabel="δα (deg)",
               title="Fitted δα(α)",
               out_png=f"{fig_prefix}delta_alpha_vs_alpha.png")

    plot_curve(alpha_grid, beta_grid_deg,
               xlabel="α (deg)", ylabel="β (deg)",
               title="Fitted β(α)",
               out_png=f"{fig_prefix}beta_vs_alpha.png")

    # ----- Prepare arrays for residual recomputation -----
    alphas_ref, betas_ref, n_hats_ref, thetas_ref, wavelength2, alphas_zone_all, include_2pi, uvw_zone = \
        build_reflection_arrays_for_residuals(experiment_path, cif_path)

    # ----- Residuals before vs after -----
    res = compute_residuals_before_after(
        R0_opt=R0_opt,
        alphas_ref=alphas_ref,
        betas_ref=betas_ref,
        n_hats_ref=n_hats_ref,
        thetas_ref=thetas_ref,
        alphas_zone_all=alphas_zone_all,
        uvw_zone=uvw_zone,
        alpha_grid=alpha_grid,
        delta_alpha_grid_deg=delta_alpha_grid_deg,
        beta_grid_deg=beta_grid_deg,
        use_precession=use_precession,
        precession_deg=precession_deg,
        n_azim_samples=int(n_azim_samples)
    )

    r_bragg_before = np.asarray(res["r_bragg_before"], dtype=float)
    r_bragg_after  = np.asarray(res["r_bragg_after"], dtype=float)
    r_zone_before_deg = np.degrees(np.asarray(res["r_zone_before_rad"], dtype=float))
    r_zone_after_deg  = np.degrees(np.asarray(res["r_zone_after_rad"], dtype=float))

    # ----- Residual diagnostics -----
    # Bragg residuals
    plot_histogram(r_bragg_before, bins=80,
                   xlabel="Bragg residual r (unitless)",
                   title="Bragg residuals BEFORE fit",
                   out_png=f"{fig_prefix}bragg_residuals_before_hist.png")

    plot_histogram(r_bragg_after, bins=80,
                   xlabel="Bragg residual r (unitless)",
                   title="Bragg residuals AFTER fit",
                   out_png=f"{fig_prefix}bragg_residuals_after_hist.png")

    plot_scatter(alphas_ref, r_bragg_before,
                 xlabel="α (deg)", ylabel="r",
                 title="Bragg residual vs α (BEFORE)",
                 out_png=f"{fig_prefix}bragg_residual_vs_alpha_before.png")

    plot_scatter(alphas_ref, r_bragg_after,
                 xlabel="α (deg)", ylabel="r",
                 title="Bragg residual vs α (AFTER)",
                 out_png=f"{fig_prefix}bragg_residual_vs_alpha_after.png")

    # Zone-axis angular misfit (degrees)
    plot_histogram(r_zone_before_deg, bins=80,
                   xlabel="Zone-axis misfit (deg)",
                   title="Zone-axis misfit BEFORE fit",
                   out_png=f"{fig_prefix}zone_misfit_before_hist.png")

    plot_histogram(r_zone_after_deg, bins=80,
                   xlabel="Zone-axis misfit (deg)",
                   title="Zone-axis misfit AFTER fit",
                   out_png=f"{fig_prefix}zone_misfit_after_hist.png")

    plot_scatter(alphas_zone_all, r_zone_before_deg,
                 xlabel="α (deg)", ylabel="misfit (deg)",
                 title="Zone-axis misfit vs α (BEFORE)",
                 out_png=f"{fig_prefix}zone_misfit_vs_alpha_before.png")

    plot_scatter(alphas_zone_all, r_zone_after_deg,
                 xlabel="α (deg)", ylabel="misfit (deg)",
                 title="Zone-axis misfit vs α (AFTER)",
                 out_png=f"{fig_prefix}zone_misfit_vs_alpha_after.png")

    # ----- Console summaries -----
    def _summ(vals: np.ndarray, label: str) -> None:
        v = np.asarray(vals, dtype=float).reshape(-1)
        print(f"{label:30s}  mean={np.mean(v): .6e}  std={np.std(v): .6e}  p90(|·|)={np.percentile(np.abs(v),90): .6e}  max(|·|)={np.max(np.abs(v)): .6e}")

    print("\nResidual summaries:")
    _summ(r_bragg_before, "Bragg residual BEFORE")
    _summ(r_bragg_after,  "Bragg residual AFTER ")
    print(f"Zone misfit BEFORE (deg): mean={np.mean(r_zone_before_deg): .6f}, p90={np.percentile(r_zone_before_deg,90): .6f}, max={np.max(r_zone_before_deg): .6f}")
    print(f"Zone misfit AFTER  (deg): mean={np.mean(r_zone_after_deg):.6f}, p90={np.percentile(r_zone_after_deg,90):.6f}, max={np.max(r_zone_after_deg):.6f}")

    # ----- CSV export -----
    export_alpha_delta_beta_csv(alpha_grid, delta_alpha_grid_deg, beta_grid_deg, csv_out)
    print(f"\nExported α–δα–β table to: {csv_out}")

    # ----- Show figures -----
    if show_plots:
        plt.show()
    else:
        plt.close('all')


# -------------------------------- Entrypoint ---------------------------------

if __name__ == "__main__":
    # Typical usage after Stage 6 has run.
    main(
        experiment_path="Si_3_dyn.cif_pets",
        cif_path="silicon_structure.cif",
        npz_path="stage6_fit_results.npz",
        use_precession=False,       # set True if you used precession in Stage 6 diagnostics
        n_azim_samples=12,
        csv_out="delta_alpha_beta_vs_alpha.csv",
        fig_prefix="stage7_",
        show_plots=True
    )
