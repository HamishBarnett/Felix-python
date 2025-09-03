#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2: Reciprocal lattice tools for cRED refinement.

This module:
- Builds a triclinic-safe B matrix (Å^-1), with optional 2π factor.
- Computes reciprocal vectors g(hkl), plane normals n̂(hkl), d-spacings, and Bragg angles.
- Checks consistency between PETS UB and B to determine the 2π convention.
- Prints QC summaries (d(111), d(220), d(311), θ for given λ; and previews for first few reflections).

All functions are defined at the top and accept explicit arguments only.
"""

from __future__ import annotations
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# You should have Stage 1 saved as stage1_parse_inputs.py in the same folder.
try:
    from stage1_parse_inputs import parse_pets_cif_pets, parse_reference_cif
except Exception as e:
    raise ImportError(
        "Could not import 'stage1_parse_inputs'. "
        "Please ensure Stage 1 is saved as 'stage1_parse_inputs.py' in the same directory."
    ) from e


# ------------------------- Linear algebra helpers ----------------------------

def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about Z by angle (radians)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


# ------------------ Build A (direct) and B (reciprocal) matrices --------------

def build_direct_cell_matrix(
    a: float, b: float, c: float,
    alpha_deg: float, beta_deg: float, gamma_deg: float
) -> np.ndarray:
    """
    Construct the 3x3 direct lattice matrix A (Å), whose columns are the
    Cartesian vectors a⃗, b⃗, c⃗ for a general triclinic cell.

    Convention:
      a⃗ is along X,
      b⃗ lies in the XY-plane,
      c⃗ has components in XYZ determined by cell angles.

    Returns:
        A: shape (3,3), columns = [a⃗, b⃗, c⃗] in Å.
    """
    alpha = math.radians(alpha_deg)
    beta  = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sg = math.sin(gamma)

    # Volume factor for the parallelepiped (dimensionless inside the sqrt)
    v_sq = 1.0 - ca*ca - cb*cb - cg*cg + 2.0*ca*cb*cg
    if v_sq <= 0.0:
        raise ValueError("Invalid cell geometry: negative/zero volume factor.")
    v = math.sqrt(v_sq)

    A = np.zeros((3, 3), dtype=float)
    # a along x
    A[0, 0] = a
    # b in xy-plane
    A[0, 1] = b * cg
    A[1, 1] = b * sg
    # c with components in x,y,z
    A[0, 2] = c * cb
    A[1, 2] = c * (ca - cb * cg) / sg
    A[2, 2] = c * (v / sg)
    return A


def build_B_matrix(
    a: float, b: float, c: float,
    alpha_deg: float, beta_deg: float, gamma_deg: float,
    include_2pi: bool = False
) -> np.ndarray:
    """
    Build the reciprocal lattice matrix B (Å^-1) that maps integer (h,k,l)
    to a Cartesian reciprocal vector g = B @ [h,k,l]^T.

    Implementation:
      - Build the direct matrix A (Å) with columns [a⃗,b⃗,c⃗].
      - The reciprocal basis matrix A* (Å^-1) is (A^{-1})^T.
      - B = A* (optionally multiplied by 2π).

    Args:
        include_2pi: if True, B includes a factor of 2π (i.e., g in rad·Å^-1).
    """
    A = build_direct_cell_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg)
    A_inv_T = np.linalg.inv(A).T
    if include_2pi:
        return 2.0 * math.pi * A_inv_T
    return A_inv_T


# ----------------------- g-vectors, d-spacings, normals ----------------------

def g_vector(B: np.ndarray, h: int, k: int, l: int) -> np.ndarray:
    """Compute reciprocal vector g = B @ (h,k,l)."""
    hkl = np.array([h, k, l], dtype=float)
    return B @ hkl


def d_spacing_from_B(B: np.ndarray, h: int, k: int, l: int, include_2pi: bool = False) -> float:
    """
    Compute d-spacing (Å) using g = B @ (h,k,l).

    If B excludes 2π (Å^-1), then ||g|| = 1/d.
    If B includes 2π (rad·Å^-1), then ||g|| = 2π/d.

    Args:
        include_2pi: pass the same flag used when building B.
    """
    g = g_vector(B, h, k, l)
    gnorm = float(np.linalg.norm(g))
    if gnorm <= 0.0:
        return float("inf")
    if include_2pi:
        return (2.0 * math.pi) / gnorm
    else:
        return 1.0 / gnorm


def plane_normal_unit_from_B(B: np.ndarray, h: int, k: int, l: int) -> np.ndarray:
    """
    Unit plane normal direction n̂ (dimensionless) in Cartesian reciprocal space,
    obtained by normalising g = B @ (h,k,l). Independent of 2π convention.
    """
    g = g_vector(B, h, k, l)
    nrm = float(np.linalg.norm(g))
    if nrm == 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return g / nrm


def bragg_angle_from_d(lambda_angstrom: float, d_angstrom: float) -> Tuple[float, float]:
    """
    Compute Bragg angle θ given wavelength λ and d-spacing (Å).

    Returns:
        (theta_rad, theta_deg)

    Uses: sin θ = λ / (2d); clamps argument to [-1, 1] for safety.
    """
    if d_angstrom <= 0.0:
        return (float("nan"), float("nan"))
    arg = lambda_angstrom / (2.0 * d_angstrom)
    arg = max(-1.0, min(1.0, arg))
    theta_rad = math.asin(arg)
    return theta_rad, math.degrees(theta_rad)


# --------------------------- UB vs B consistency -----------------------------

def ub_b_consistency_report(
    UB: np.ndarray,
    a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float
) -> Dict[str, float]:
    """
    Compare column norms of UB against those of B (with and without 2π) to
    infer which convention UB likely uses. For any rotation U, ||UB[:,j]|| = ||B[:,j]||.

    Returns a dict with column norms and scale ratios.
    """
    B_no2pi = build_B_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg, include_2pi=False)
    B_2pi   = build_B_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg, include_2pi=True)

    def col_norms(M: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(M[:, j]) for j in range(3)], dtype=float)

    ub_norms = col_norms(UB)
    b_norms_no2pi = col_norms(B_no2pi)
    b_norms_2pi   = col_norms(B_2pi)

    # Scale factors (mean over columns)
    # If UB ≈ U * B_no2pi, scale_no2pi ≈ 1; if UB ≈ U * (2π B), scale_2pi ≈ 1.
    ratio_no2pi = float(np.mean(ub_norms / b_norms_no2pi))
    ratio_2pi   = float(np.mean(ub_norms / b_norms_2pi))

    return {
        "ub_col_norm_1": ub_norms[0],
        "ub_col_norm_2": ub_norms[1],
        "ub_col_norm_3": ub_norms[2],
        "b_no2pi_col_norm_1": b_norms_no2pi[0],
        "b_no2pi_col_norm_2": b_norms_no2pi[1],
        "b_no2pi_col_norm_3": b_norms_no2pi[2],
        "b_2pi_col_norm_1": b_norms_2pi[0],
        "b_2pi_col_norm_2": b_norms_2pi[1],
        "b_2pi_col_norm_3": b_norms_2pi[2],
        "scale_ratio_vs_no2pi": ratio_no2pi,
        "scale_ratio_vs_2pi": ratio_2pi,
    }


def infer_ub_2pi_convention(scale_ratio_vs_no2pi: float, scale_ratio_vs_2pi: float) -> bool:
    """
    Decide whether UB likely includes 2π based on scale ratios.
    Returns:
        include_2pi_for_B (bool)
    """
    # Pick the hypothesis whose scale ratio is closer to 1.
    if abs(scale_ratio_vs_no2pi - 1.0) <= abs(scale_ratio_vs_2pi - 1.0):
        return False
    return True


# ----------------------------- QC / Preview ----------------------------------

def preview_common_silicon_reflections(
    B: np.ndarray, wavelength: float, include_2pi: bool
) -> pd.DataFrame:
    """
    Compute d and Bragg θ for a few common Si reflections using the given B and λ.
    """
    hkls = [(1,1,1), (2,2,0), (3,1,1), (4,0,0)]
    rows: List[Dict[str, float]] = []
    for (h,k,l) in hkls:
        d = d_spacing_from_B(B, h, k, l, include_2pi=include_2pi)
        theta_rad, theta_deg = bragg_angle_from_d(wavelength, d)
        rows.append({
            "h": h, "k": k, "l": l,
            "d_A": d,
            "theta_deg": theta_deg
        })
    return pd.DataFrame(rows)


def attach_d_theta_to_reflections(
    refl_df: pd.DataFrame, B: np.ndarray, wavelength: float, include_2pi: bool, max_rows: int = 8
) -> pd.DataFrame:
    """
    For a small preview subset of reflections, compute d and θ and return a new DataFrame.
    """
    if refl_df.empty:
        return pd.DataFrame(columns=["h","k","l","I","sigma","zone_axis_id","d_A","theta_deg"])

    subset = refl_df.head(max_rows).copy()
    d_list = []
    th_list = []
    for h, k, l in zip(subset["h"], subset["k"], subset["l"]):
        d = d_spacing_from_B(B, int(h), int(k), int(l), include_2pi=include_2pi)
        theta_rad, theta_deg = bragg_angle_from_d(wavelength, d)
        d_list.append(d)
        th_list.append(theta_deg)
    subset["d_A"] = d_list
    subset["theta_deg"] = th_list
    return subset


# --------------------------------- Main --------------------------------------

def main(experiment_path: str, cif_path: str) -> None:
    """
    Stage 2 driver:
      - Parse inputs (Stage 1),
      - Build B from PETS cell (to be consistent with UB),
      - Check UB↔B consistency (2π or not),
      - Print QC: common Si d-spacings & θ, plus a small reflection preview.
    """
    pets_header, UB, wavelength, zone_axes_df, reflections_df = parse_pets_cif_pets(experiment_path)
    _, cif_cell, cif_space_group, cif_atoms = parse_reference_cif(cif_path)  # not strictly needed here

    # Choose cell from PETS header for consistency with UB
    cell_pets = {
        "a": float(pets_header.get("_cell_length_a", "nan")),
        "b": float(pets_header.get("_cell_length_b", "nan")),
        "c": float(pets_header.get("_cell_length_c", "nan")),
        "alpha": float(pets_header.get("_cell_angle_alpha", "90")),
        "beta": float(pets_header.get("_cell_angle_beta", "90")),
        "gamma": float(pets_header.get("_cell_angle_gamma", "90")),
    }

    print("\n=== Cell (from PETS header) used to build B ===")
    print(cell_pets)
    print(f"Wavelength λ (Å): {wavelength}")

    # UB vs B consistency check
    report = ub_b_consistency_report(
        UB,
        cell_pets["a"], cell_pets["b"], cell_pets["c"],
        cell_pets["alpha"], cell_pets["beta"], cell_pets["gamma"]
    )
    print("\n=== UB vs B consistency report ===")
    for k in [
        "ub_col_norm_1","ub_col_norm_2","ub_col_norm_3",
        "b_no2pi_col_norm_1","b_no2pi_col_norm_2","b_no2pi_col_norm_3",
        "b_2pi_col_norm_1","b_2pi_col_norm_2","b_2pi_col_norm_3",
        "scale_ratio_vs_no2pi","scale_ratio_vs_2pi"
    ]:
        print(f"{k:24s}: {report[k]:.6f}")

    include_2pi = infer_ub_2pi_convention(report["scale_ratio_vs_no2pi"], report["scale_ratio_vs_2pi"])
    print(f"\nInferred UB convention → B include_2pi = {include_2pi}")

    # Build B with the selected convention
    B = build_B_matrix(
        cell_pets["a"], cell_pets["b"], cell_pets["c"],
        cell_pets["alpha"], cell_pets["beta"], cell_pets["gamma"],
        include_2pi=include_2pi
    )

    print("\nB matrix (Å^-1{}):".format(" with 2π" if include_2pi else ""))
    with np.printoptions(precision=6, suppress=True):
        print(B)

    # QC: common Si reflections
    df_si = preview_common_silicon_reflections(B, wavelength, include_2pi)
    print("\n=== Common Si reflections (using PETS cell & inferred convention) ===")
    print(df_si.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    # Small preview on your reflections
    df_preview = attach_d_theta_to_reflections(reflections_df, B, wavelength, include_2pi, max_rows=8)
    print("\n=== First few reflections with computed d and θ ===")
    print(df_preview.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    # Optional: quick check for cubic case — compare d(hkl) to a/√(h^2+k^2+l^2)
    if abs(cell_pets["a"] - cell_pets["b"]) < 1e-6 and abs(cell_pets["a"] - cell_pets["c"]) < 1e-6:
        a = cell_pets["a"]
        for (h,k,l) in [(1,1,1),(2,2,0),(3,1,1),(4,0,0)]:
            d_theory = a / math.sqrt(h*h + k*k + l*l) if (h,k,l)!=(4,0,0) else a/4.0
            d_calc = d_spacing_from_B(B, h, k, l, include_2pi=include_2pi)
            print(f"d({h}{k}{l}) calc = {d_calc:.6f} Å, cubic theory ≈ {d_theory:.6f} Å")



if __name__ == "__main__":
    # Adjust these paths if needed
    EXPERIMENT_PATH = "Si_3_dyn.cif_pets"
    CIF_PATH = "silicon_structure.cif"
    main(EXPERIMENT_PATH, CIF_PATH)
