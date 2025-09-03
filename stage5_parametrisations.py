#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5: Parameterisations for δα(α) and β(α).

This module provides two smooth function families over rotation angle α (in degrees):

1) Clamped cubic B-splines (uniform internal knots)
   - Good for general smooth, non-periodic drift/wobble.
   - Includes utilities to build knots, make a design matrix, evaluate, and
     build a smoothness (second-difference) penalty matrix.

2) Fourier series on the α-range
   - Ideal for periodic spindle wobble.
   - Includes utilities to build a design matrix, evaluate, and a harmonic-
     weighted Tikhonov penalty (penalise higher harmonics more strongly).

All functions/classes take explicit arguments; there are no closures or implicit
globals. A small preview driver at the bottom shows shapes and zero-valued
evaluations on your dataset (δα=β=0) so you can sanity check before Stage 6.

Dependencies:
- numpy, pandas
- scipy (for BSpline). If SciPy is not available, spline-based utilities will
  raise a clear ImportError (Fourier still works).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    # SciPy is required for B-splines
    from scipy.interpolate import BSpline
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------- Simple helpers ----------

def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))

def ensure_1d_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr

# ============================================================================
#                           B-SPLINE PARAMETRISATION
# ============================================================================

def build_uniform_clamped_knots(alpha_min_deg: float,
                                alpha_max_deg: float,
                                n_internal_knots: int,
                                degree: int = 3) -> np.ndarray:
    """
    Build a clamped (open) knot vector for cubic B-splines on [alpha_min, alpha_max]
    with uniform internal knots.

    Knot vector structure:
      [α_min] repeated (degree+1) times,
      then n_internal_knots uniformly spaced nodes in (α_min, α_max),
      then [α_max] repeated (degree+1) times.

    Returns:
        knots: shape (n_knots,)
    """
    if degree < 1:
        raise ValueError("B-spline degree must be >= 1.")
    if n_internal_knots < 0:
        raise ValueError("n_internal_knots must be >= 0.")
    if alpha_max_deg <= alpha_min_deg:
        raise ValueError("alpha_max_deg must be > alpha_min_deg.")
    if n_internal_knots == 0:
        internal = np.array([], dtype=float)
    else:
        internal = np.linspace(alpha_min_deg, alpha_max_deg, n_internal_knots + 2)[1:-1]
    knots = np.concatenate([
        np.full(degree + 1, alpha_min_deg, dtype=float),
        internal,
        np.full(degree + 1, alpha_max_deg, dtype=float),
    ])
    return knots


def bspline_num_basis(knots: np.ndarray, degree: int) -> int:
    """
    Number of basis functions for a B-spline with given knot vector and degree.
    n_basis = len(knots) - degree - 1
    """
    return int(len(knots) - degree - 1)


def bspline_design_matrix(x_deg: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Build the B-spline design matrix Φ for inputs x (degrees).

    For N inputs and M basis functions, returns Φ with shape (N, M) such that
        f(x) = Φ @ c
    for coefficient vector c (length M).

    Requires SciPy. Extrapolation is disabled; outside the clamped support the
    basis is zero (clamped endpoints keep typical α within support).
    """
    if not _HAVE_SCIPY:
        raise ImportError("SciPy is required for B-splines. Install 'scipy' or use Fourier basis instead.")
    x = ensure_1d_float_array(x_deg)
    M = bspline_num_basis(knots, degree)
    Phi = np.zeros((len(x), M), dtype=float)
    # Efficient enough for modest M (e.g., M ~ 10–40)
    for j in range(M):
        coeffs = np.zeros(M, dtype=float)
        coeffs[j] = 1.0
        bs = BSpline(knots, coeffs, degree, extrapolate=False)
        y = bs(x)
        # Replace NaNs (outside support) with 0.0
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        Phi[:, j] = y
    return Phi


def bspline_second_diff_penalty_matrix(n_basis: int, scale: float = 1.0) -> np.ndarray:
    """
    Build a simple curvature penalty L such that L @ c are the discrete second
    differences of the coefficient vector c. This approximates ∫ (f''(α))^2 dα.

    Args:
        n_basis: number of coefficients
        scale: optional scalar multiplier (e.g., to reflect average knot spacing)

    Returns:
        L: shape (n_basis - 2, n_basis)
    """
    if n_basis < 3:
        # No second-difference penalty possible
        return np.zeros((0, n_basis), dtype=float)
    L = np.zeros((n_basis - 2, n_basis), dtype=float)
    for i in range(n_basis - 2):
        L[i, i]     =  1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] =  1.0
    if scale != 1.0:
        L *= float(scale)
    return L


@dataclass
class Spline1D:
    """
    Clamped cubic B-spline parameterisation over α ∈ [alpha_min_deg, alpha_max_deg].

    Fields:
        knots: clamped knot vector (built via build_uniform_clamped_knots or similar)
        degree: usually 3 (cubic)
        coeffs: shape (M,) coefficient vector (initialise to zeros)
    """
    knots: np.ndarray
    degree: int
    coeffs: np.ndarray

    @staticmethod
    def create_uniform(alpha_min_deg: float,
                       alpha_max_deg: float,
                       target_spacing_deg: float = 3.0,
                       degree: int = 3) -> "Spline1D":
        """
        Construct a Spline1D with uniform internal knots roughly every target_spacing_deg.
        """
        span = float(alpha_max_deg - alpha_min_deg)
        n_internal = max(0, int(round(span / target_spacing_deg)) - 1)
        knots = build_uniform_clamped_knots(alpha_min_deg, alpha_max_deg, n_internal, degree=degree)
        M = bspline_num_basis(knots, degree)
        coeffs = np.zeros(M, dtype=float)
        return Spline1D(knots=knots, degree=degree, coeffs=coeffs)

    def design_matrix(self, alpha_deg: np.ndarray) -> np.ndarray:
        """Design matrix Φ for given α (degrees)."""
        return bspline_design_matrix(alpha_deg, self.knots, self.degree)

    def evaluate(self, alpha_deg: np.ndarray) -> np.ndarray:
        """Evaluate f(α) = Φ @ coeffs for α (degrees)."""
        Phi = self.design_matrix(alpha_deg)
        return Phi @ self.coeffs

    def smoothness_matrix(self, scale: float = 1.0) -> np.ndarray:
        """Second-difference penalty matrix L (so smoothness penalty is ||L @ coeffs||^2)."""
        return bspline_second_diff_penalty_matrix(len(self.coeffs), scale=scale)

# ============================================================================
#                           FOURIER PARAMETRISATION
# ============================================================================

def fourier_design_matrix(alpha_deg: np.ndarray,
                          alpha_min_deg: float,
                          alpha_max_deg: float,
                          n_harmonics: int,
                          include_bias: bool = True,
                          include_linear: bool = False) -> np.ndarray:
    """
    Build a Fourier design matrix over α ∈ [alpha_min, alpha_max].

    We map α to t = 2π * (α - α_min) / (α_max - α_min), so one fundamental period
    spans the full α-range. Basis columns are:
      [1] (optional),
      [(α - α_mid)] (optional linear drift),
      sin(m t), cos(m t) for m = 1..n_harmonics.

    Returns:
        Φ: shape (N, M)
    """
    alpha = ensure_1d_float_array(alpha_deg)
    L = float(alpha_max_deg - alpha_min_deg)
    if L <= 0.0:
        raise ValueError("alpha_max_deg must be > alpha_min_deg.")
    t = 2.0 * math.pi * (alpha - alpha_min_deg) / L
    cols: List[np.ndarray] = []
    if include_bias:
        cols.append(np.ones_like(t))
    if include_linear:
        alpha_mid = 0.5 * (alpha_min_deg + alpha_max_deg)
        cols.append(alpha - alpha_mid)
    for m in range(1, int(n_harmonics) + 1):
        cols.append(np.sin(m * t))
        cols.append(np.cos(m * t))
    if len(cols) == 0:
        return np.zeros((len(alpha), 0), dtype=float)
    return np.column_stack(cols)


def fourier_harmonic_weights(n_harmonics: int,
                             include_bias: bool = True,
                             include_linear: bool = False,
                             base_weight_bias: float = 0.0,
                             base_weight_linear: float = 0.0,
                             weight_scale: float = 1.0) -> np.ndarray:
    """
    Construct a diagonal weight vector w for Tikhonov penalty ||W c||^2 where
    higher harmonics are penalised more strongly (e.g., weight ∝ m^2).

    Returns:
        w: shape (M,), to be used as diag(W)
    """
    weights: List[float] = []
    if include_bias:
        weights.append(float(base_weight_bias))
    if include_linear:
        weights.append(float(base_weight_linear))
    for m in range(1, int(n_harmonics) + 1):
        w_m = (float(m) ** 2) * float(weight_scale)
        weights.append(w_m)  # sin m
        weights.append(w_m)  # cos m
    return np.asarray(weights, dtype=float)


@dataclass
class Fourier1D:
    """
    Fourier parameterisation over α ∈ [alpha_min_deg, alpha_max_deg].

    Fields:
        alpha_min_deg, alpha_max_deg: domain (degrees)
        n_harmonics: number of sine/cosine pairs
        include_bias: include constant term
        include_linear: include linear (drift) term
        coeffs: shape (M,) coefficient vector (initialise to zeros)
    """
    alpha_min_deg: float
    alpha_max_deg: float
    n_harmonics: int
    include_bias: bool
    include_linear: bool
    coeffs: np.ndarray

    @staticmethod
    def create(alpha_min_deg: float,
               alpha_max_deg: float,
               n_harmonics: int = 3,
               include_bias: bool = True,
               include_linear: bool = False) -> "Fourier1D":
        """Construct a Fourier1D with zero initial coefficients."""
        Phi_dummy = fourier_design_matrix(
            np.array([alpha_min_deg, alpha_max_deg], dtype=float),
            alpha_min_deg, alpha_max_deg, n_harmonics,
            include_bias=include_bias, include_linear=include_linear
        )
        M = Phi_dummy.shape[1]
        coeffs = np.zeros(M, dtype=float)
        return Fourier1D(alpha_min_deg, alpha_max_deg, n_harmonics, include_bias, include_linear, coeffs)

    def design_matrix(self, alpha_deg: np.ndarray) -> np.ndarray:
        """Design matrix Φ for given α (degrees)."""
        return fourier_design_matrix(
            alpha_deg,
            self.alpha_min_deg, self.alpha_max_deg,
            self.n_harmonics,
            include_bias=self.include_bias,
            include_linear=self.include_linear
        )

    def evaluate(self, alpha_deg: np.ndarray) -> np.ndarray:
        """Evaluate f(α) = Φ @ coeffs for α (degrees)."""
        Phi = self.design_matrix(alpha_deg)
        return Phi @ self.coeffs

    def smoothness_weights(self,
                           base_weight_bias: float = 0.0,
                           base_weight_linear: float = 0.0,
                           weight_scale: float = 1.0) -> np.ndarray:
        """
        Return a diagonal weight vector w for Tikhonov penalty ||W c||^2 (use np.diag(w)).
        Typically set base weights small (or zero) and penalise harmonics with m^2 scaling.
        """
        return fourier_harmonic_weights(
            self.n_harmonics,
            include_bias=self.include_bias,
            include_linear=self.include_linear,
            base_weight_bias=base_weight_bias,
            base_weight_linear=base_weight_linear,
            weight_scale=weight_scale
        )

# ============================================================================
#                           PREVIEW / SANITY DRIVER
# ============================================================================

# We’ll reuse Stage 1 to get α-range for sensible defaults.
try:
    from stage1_parse_inputs import parse_pets_cif_pets
except Exception:
    parse_pets_cif_pets = None  # handled in main_preview


def infer_alpha_range_from_zone_axes(zone_axes_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Return (alpha_min_deg, alpha_max_deg) from the zone-axis table.
    """
    if zone_axes_df.empty or "alpha_deg" not in zone_axes_df.columns:
        raise ValueError("zone_axes_df must contain 'alpha_deg'.")
    alpha_vals = zone_axes_df["alpha_deg"].dropna().to_numpy(dtype=float)
    return float(np.min(alpha_vals)), float(np.max(alpha_vals))


def main_preview(experiment_path: str,
                 use_spline: bool = True,
                 target_spacing_deg: float = 3.0,
                 fourier_harmonics: int = 3) -> None:
    """
    Preview builder: reads the PETS file, infers α-range, constructs both models
    with zero coefficients, and prints basic shapes & a few evaluations to confirm.

    Args:
        experiment_path: path to the PETS .cif_pets file
        use_spline: if True, build a spline model as well (requires SciPy)
        target_spacing_deg: desired knot spacing for spline
        fourier_harmonics: number of sin/cos pairs for Fourier
    """
    if parse_pets_cif_pets is None:
        print("Stage 1 parser not found. Please place 'stage1_parse_inputs.py' in the same directory.")
        return

    pets_header, UB, wavelength, zone_axes_df, reflections_df = parse_pets_cif_pets(experiment_path)
    alpha_min, alpha_max = infer_alpha_range_from_zone_axes(zone_axes_df)

    print("\n=== Stage 5 Preview ===")
    print(f"α-range inferred from zone-axis table: [{alpha_min:.3f}, {alpha_max:.3f}] deg")
    sample_alphas = np.linspace(alpha_min, alpha_max, 7)

    # Fourier model (always available)
    f_model = Fourier1D.create(alpha_min, alpha_max, n_harmonics=fourier_harmonics, include_bias=True, include_linear=False)
    Phi_f = f_model.design_matrix(sample_alphas)
    vals_f = f_model.evaluate(sample_alphas)
    print(f"\nFourier basis: n_harmonics={fourier_harmonics}, include_bias=True, include_linear=False")
    print(f"  Design matrix Φ_f shape: {Phi_f.shape} (rows=Nα, cols=M)")
    print(f"  Coeffs length M: {len(f_model.coeffs)}")
    print(f"  Zero-eval on 7 α samples: {np.array2string(vals_f, precision=6)}")
    w = f_model.smoothness_weights(weight_scale=1.0)
    print(f"  Smoothness weights length: {len(w)} (first few: {w[:min(6,len(w))]})")

    # Spline model (if SciPy available)
    if use_spline:
        if not _HAVE_SCIPY:
            print("\nSciPy not available → skipping spline preview. Install 'scipy' to enable B-splines.")
        else:
            s_model = Spline1D.create_uniform(alpha_min, alpha_max, target_spacing_deg=target_spacing_deg, degree=3)
            Phi_s = s_model.design_matrix(sample_alphas)
            vals_s = s_model.evaluate(sample_alphas)
            L = s_model.smoothness_matrix(scale=1.0)
            print(f"\nSpline basis: cubic, target_spacing≈{target_spacing_deg}°, internal knots={len(s_model.knots) - 2*4}")
            print(f"  Knot vector length: {len(s_model.knots)}  → n_basis M = {len(s_model.coeffs)}")
            print(f"  Design matrix Φ_s shape: {Phi_s.shape}")
            print(f"  Zero-eval on 7 α samples: {np.array2string(vals_s, precision=6)}")
            print(f"  Smoothness matrix L shape: {L.shape} (rows ~ M-2, cols=M)")

    # Tip: For Stage 6, you will instantiate *two* models (one for δα, one for β),
    # build Φ on all α needed (frames or reflections), and fit their coefficients
    # jointly with R0 via nonlinear least squares + robust/smooth penalties.


if __name__ == "__main__":
    # Adjust path if needed
    EXPERIMENT_PATH = "Si_3_dyn.cif_pets"
    # Preview both Fourier (always) and Spline (if SciPy present)
    main_preview(EXPERIMENT_PATH, use_spline=True, target_spacing_deg=3.0, fourier_harmonics=3)
