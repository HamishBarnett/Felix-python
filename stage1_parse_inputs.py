#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1: File parsing and data extraction for cRED refinement.
- Parses PETS .cif_pets experimental file:
  * header values (cell, wavelength, UB, measurement details)
  * zone-axis loop (u,v,w, precession, alpha, beta, omega, scale)
  * reflection loop (h,k,l, I, sigma, zone_axis_id)
- Parses reference CIF for cell, space group, atom sites (optional downstream use).

All functions are defined at the top and use only explicit arguments/returns,
as requested. No reliance on implicit globals.
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# ----------------------------- Utilities ------------------------------------

def _safe_float(s: str) -> Optional[float]:
    """Convert string to float if possible, else return None."""
    try:
        return float(s)
    except Exception:
        # Handle CIF uncertainty format like 5.43053(7)
        m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)(?:\(\d+\))?\s*$", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None


def _safe_int(s: str) -> Optional[int]:
    """Convert string to int if possible, else return None."""
    try:
        return int(s)
    except Exception:
        return None


def _strip_quotes(s: str) -> str:
    """Strip CIF-like single quotes around a token if present."""
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s


def _read_text_block(lines: List[str], start_idx: int) -> Tuple[str, int]:
    """
    Read a semicolon-delimited CIF text block starting at lines[start_idx] which should be a line with only ';'.
    Returns (text, next_index_after_block).
    """
    assert lines[start_idx].strip().startswith(";"), "Expected ';' to start a CIF text block."
    start_idx += 1
    buf = []
    while start_idx < len(lines):
        line = lines[start_idx]
        if line.strip().startswith(";"):
            # end of text block
            return ("\n".join(buf).rstrip("\n"), start_idx + 1)
        buf.append(line.rstrip("\n"))
        start_idx += 1
    # If we get here, block wasn't closed; return what we collected.
    return ("\n".join(buf).rstrip("\n"), start_idx)


# ---------------------- PETS .cif_pets parsing -------------------------------

def parse_pets_cif_pets(path: str) -> Tuple[Dict[str, str], np.ndarray, float, pd.DataFrame, pd.DataFrame]:
    """
    Parse a PETS .cif_pets file.

    Returns:
        header: dict of header key -> string value (raw, untyped; use _safe_float/_safe_int as needed)
        UB: 3x3 numpy array from _diffrn_orient_matrix_UB_ij (row-major, i,j in {1,2,3})
        wavelength: float (Angstrom)
        zone_axes_df: DataFrame with columns:
            ['zone_axis_id','u','v','w','precession_deg','alpha_deg','beta_deg','omega','scale']
        reflections_df: DataFrame with columns:
            ['h','k','l','I','sigma','zone_axis_id']
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header: Dict[str, str] = {}
    UB_vals: Dict[Tuple[int, int], float] = {}
    wavelength: Optional[float] = None

    zone_axes_cols: List[str] = []
    zone_axes_rows: List[List[str]] = []

    refl_cols: List[str] = []
    refl_rows: List[List[str]] = []

    i = 0
    current_loop_cols: List[str] = []
    current_loop_target: Optional[str] = None  # 'zone' or 'refl' or None

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # Skip blank
        if not line:
            i += 1
            continue

        # CIF-style text block (e.g., _diffrn_measurement_details ; ... ;)
        if line.startswith("_") and (" " not in line):
            # single token key with value on following line? Rare; skip to next.
            # We'll handle key-value pairs with spaces below.
            i += 1
            continue

        # Key / value line: starts with _
        if line.startswith("_"):
            # Handle multi-token values and potential multi-line text blocks starting with ';'
            # Split at first whitespace
            parts = raw.split(None, 1)
            key = parts[0].strip()
            val = ""
            if len(parts) > 1:
                val = parts[1].rstrip("\n")
                if val.strip() == ";":
                    # multi-line text block
                    text, new_i = _read_text_block(lines, i + 1)
                    header[key] = text
                    i = new_i
                    continue
            header[key] = val.strip()
            # Track wavelength and UB here if present
            if key == "_diffrn_radiation_wavelength":
                wl = _safe_float(header[key])
                if wl is not None:
                    wavelength = wl
            m = re.match(r"^_diffrn_orient_matrix_UB_(\d)(\d)\b", key)
            if m:
                r = int(m.group(1))
                c = int(m.group(2))
                v = _safe_float(header[key])
                if v is not None:
                    UB_vals[(r, c)] = v
            i += 1
            continue

        # Loop block
        if line.startswith("loop_"):
            # Collect column headers (lines starting with '_') until first data row
            current_loop_cols = []
            current_loop_target = None
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("_"):
                col = lines[i].strip()
                current_loop_cols.append(col)
                i += 1

            # Decide which table this loop is (zone-axis or reflections) from columns
            if all(c.startswith("_diffrn_zone_axis_") for c in current_loop_cols):
                current_loop_target = "zone"
                zone_axes_cols = current_loop_cols.copy()
            elif all(c.startswith("_refln_") for c in current_loop_cols):
                current_loop_target = "refl"
                refl_cols = current_loop_cols.copy()
            else:
                # Unknown loop: skip rows until next loop_/data_/EOF
                while i < len(lines):
                    s = lines[i].strip()
                    if s.startswith("loop_") or s.startswith("data_") or s.startswith("_"):
                        break
                    i += 1
                continue

            # Now read data rows for this loop until next 'loop_' or 'data_' or '_' header
            while i < len(lines):
                s = lines[i].strip()
                if (not s) or s.startswith("#"):
                    i += 1
                    continue
                if s.startswith("loop_") or s.startswith("data_") or s.startswith("_"):
                    # next block begins; do not consume this line
                    break
                # Tokenize the row respecting quoted fields
                row_tokens = _tokenize_cif_row(lines[i].rstrip("\n"))
                # Some loops might have rows spanning multiple lines; attempt to read more lines
                # until token count matches the number of columns.
                while len(row_tokens) < len(current_loop_cols) and (i + 1) < len(lines):
                    i += 1
                    more = _tokenize_cif_row(lines[i].rstrip("\n"))
                    row_tokens += more
                if len(row_tokens) >= len(current_loop_cols):
                    if current_loop_target == "zone":
                        zone_axes_rows.append(row_tokens[:len(current_loop_cols)])
                    elif current_loop_target == "refl":
                        refl_rows.append(row_tokens[:len(current_loop_cols)])
                i += 1
            continue

        # Other lines (comments, data_ labels, etc.)
        i += 1

    # Build UB matrix (default zeros if missing entries)
    UB = np.zeros((3, 3), dtype=float)
    for (r, c), v in UB_vals.items():
        if 1 <= r <= 3 and 1 <= c <= 3:
            UB[r - 1, c - 1] = v

    if wavelength is None:
        raise ValueError("Wavelength not found in PETS file (_diffrn_radiation_wavelength).")

    # Build zone-axes DataFrame with clean column names/types
    zone_axes_df = _build_zone_axes_df(zone_axes_cols, zone_axes_rows)

    # Build reflections DataFrame with clean column names/types
    reflections_df = _build_reflections_df(refl_cols, refl_rows)

    return header, UB, wavelength, zone_axes_df, reflections_df


def _tokenize_cif_row(s: str) -> List[str]:
    """
    Tokenize a CIF/PETS row: split on whitespace but keep quoted substrings intact.
    Handles single or double quotes.
    """
    tokens: List[str] = []
    buf = []
    quote_char = None
    i = 0
    while i < len(s):
        ch = s[i]
        if quote_char:
            if ch == quote_char:
                # end quote
                tokens.append("".join(buf))
                buf = []
                quote_char = None
                # consume any immediate trailing token boundary
                # (next loop iteration will handle whitespace)
            else:
                buf.append(ch)
            i += 1
            continue
        # not in quotes
        if ch in ("'", '"'):
            # start quote; flush buf if any (as separate token)
            if buf:
                tokens.append("".join(buf))
                buf = []
            quote_char = ch
            i += 1
            continue
        if ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        tokens.append("".join(buf))
    return tokens


def _build_zone_axes_df(cols: List[str], rows: List[List[str]]) -> pd.DataFrame:
    """
    Construct a tidy DataFrame for the zone-axis loop.
    Expects columns like:
      _diffrn_zone_axis_id
      _diffrn_zone_axis_u
      _diffrn_zone_axis_v
      _diffrn_zone_axis_w
      _diffrn_zone_axis_precession_angle
      _diffrn_zone_axis_alpha
      _diffrn_zone_axis_beta
      _diffrn_zone_axis_omega
      _diffrn_zone_axis_scale
    """
    if not cols or not rows:
        return pd.DataFrame(columns=[
            "zone_axis_id", "u", "v", "w", "precession_deg",
            "alpha_deg", "beta_deg", "omega", "scale"
        ])

    # Map CIF column names to output names
    mapping = {
        "_diffrn_zone_axis_id": "zone_axis_id",
        "_diffrn_zone_axis_u": "u",
        "_diffrn_zone_axis_v": "v",
        "_diffrn_zone_axis_w": "w",
        "_diffrn_zone_axis_precession_angle": "precession_deg",
        "_diffrn_zone_axis_alpha": "alpha_deg",
        "_diffrn_zone_axis_beta": "beta_deg",
        "_diffrn_zone_axis_omega": "omega",
        "_diffrn_zone_axis_scale": "scale",
    }

    # Build DataFrame
    df_raw = pd.DataFrame(rows, columns=cols)
    df = pd.DataFrame()
    for c_old, c_new in mapping.items():
        if c_old in df_raw.columns:
            df[c_new] = df_raw[c_old].map(_strip_quotes)
        else:
            df[c_new] = np.nan

    # Convert types
    df["zone_axis_id"] = df["zone_axis_id"].map(_safe_int)
    for c in ["u", "v", "w", "precession_deg", "alpha_deg", "beta_deg", "omega", "scale"]:
        df[c] = df[c].map(_safe_float)

    return df


def _build_reflections_df(cols: List[str], rows: List[List[str]]) -> pd.DataFrame:
    """
    Construct a tidy DataFrame for the reflection loop.
    Expects columns like:
      _refln_index_h
      _refln_index_k
      _refln_index_l
      _refln_intensity_meas
      _refln_intensity_sigma
      _refln_zone_axis_id
    """
    if not cols or not rows:
        return pd.DataFrame(columns=["h", "k", "l", "I", "sigma", "zone_axis_id"])

    mapping = {
        "_refln_index_h": "h",
        "_refln_index_k": "k",
        "_refln_index_l": "l",
        "_refln_intensity_meas": "I",
        "_refln_intensity_sigma": "sigma",
        "_refln_zone_axis_id": "zone_axis_id",
    }

    df_raw = pd.DataFrame(rows, columns=cols)
    df = pd.DataFrame()
    for c_old, c_new in mapping.items():
        if c_old in df_raw.columns:
            df[c_new] = df_raw[c_old].map(_strip_quotes)
        else:
            df[c_new] = np.nan

    # Convert types
    for c in ["h", "k", "l", "zone_axis_id"]:
        df[c] = df[c].map(_safe_int)
    for c in ["I", "sigma"]:
        df[c] = df[c].map(_safe_float)

    return df


# ---------------------------- Reference CIF parsing --------------------------

def parse_reference_cif(path: str) -> Tuple[Dict[str, str], Dict[str, float], Optional[str], pd.DataFrame]:
    """
    Minimal parser for a standard CIF file to extract:
      - raw header dict (key -> value)
      - cell parameters dict: {'a','b','c','alpha','beta','gamma','volume'} as floats where available
      - space group (string) if present
      - atom sites DataFrame (if _atom_site_* loop exists)

    Returns:
        raw_header: dict of all _key -> raw string value (untyped)
        cell: dict with float values where parsed
        space_group: str or None
        atom_sites_df: DataFrame with atom-site columns if found, else empty DataFrame
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    raw_header: Dict[str, str] = {}
    i = 0
    # Collect header key-values (outside loops)
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            i += 1
            continue

        if line.startswith("loop_"):
            break

        if line.startswith("_"):
            parts = raw.split(None, 1)
            key = parts[0].strip()
            val = ""
            if len(parts) > 1:
                val = parts[1].rstrip("\n")
                if val.strip() == ";":
                    text, new_i = _read_text_block(lines, i + 1)
                    raw_header[key] = text
                    i = new_i
                    continue
            raw_header[key] = val.strip()
        i += 1

    # Find atom-site loop (if present)
    atom_cols: List[str] = []
    atom_rows: List[List[str]] = []

    # Scan the rest for loops
    j = i
    current_cols: List[str] = []
    while j < len(lines):
        line = lines[j].strip()
        if line.startswith("loop_"):
            current_cols = []
            j += 1
            while j < len(lines) and lines[j].lstrip().startswith("_"):
                current_cols.append(lines[j].strip())
                j += 1
            # Atom site loop?
            if any(c.startswith("_atom_site_") for c in current_cols):
                atom_cols = current_cols.copy()
                # read rows
                while j < len(lines):
                    s = lines[j].strip()
                    if (not s) or s.startswith("#"):
                        j += 1
                        continue
                    if s.startswith("loop_") or s.startswith("_") or s.startswith("data_"):
                        break
                    row_tokens = _tokenize_cif_row(lines[j].rstrip("\n"))
                    # Allow multi-line rows if needed
                    while len(row_tokens) < len(atom_cols) and (j + 1) < len(lines):
                        j += 1
                        row_tokens += _tokenize_cif_row(lines[j].rstrip("\n"))
                    if len(row_tokens) >= len(atom_cols):
                        atom_rows.append(row_tokens[:len(atom_cols)])
                    j += 1
            else:
                # Skip loop rows
                while j < len(lines):
                    s = lines[j].strip()
                    if s.startswith("loop_") or s.startswith("_") or s.startswith("data_"):
                        break
                    j += 1
                continue
        else:
            j += 1

    # Cell and space group
    cell_keys = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
        "_cell_volume": "volume",
    }
    cell: Dict[str, float] = {}
    for k_src, k_dst in cell_keys.items():
        if k_src in raw_header:
            v = _safe_float(raw_header[k_src])
            if v is not None:
                cell[k_dst] = v

    space_group = None
    for sg_key in ("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M", "_space_group_name_Hall"):
        if sg_key in raw_header:
            space_group = _strip_quotes(raw_header[sg_key])
            break

    # Build atom sites DataFrame
    if atom_cols and atom_rows:
        atom_df = pd.DataFrame(atom_rows, columns=atom_cols)
        # Try to normalise common columns
        rename_map = {
            "_atom_site_label": "label",
            "_atom_site_type_symbol": "type_symbol",
            "_atom_site_fract_x": "fract_x",
            "_atom_site_fract_y": "fract_y",
            "_atom_site_fract_z": "fract_z",
            "_atom_site_B_iso_or_equiv": "B_iso",
            "_atom_site_occupancy": "occupancy",
            "_atom_site_Wyckoff_symbol": "Wyckoff",
            "_atom_site_symmetry_multiplicity": "multiplicity",
        }
        out = pd.DataFrame()
        for c_old, c_new in rename_map.items():
            if c_old in atom_df.columns:
                out[c_new] = atom_df[c_old].map(_strip_quotes)
        # Types
        for c in ("fract_x", "fract_y", "fract_z", "B_iso", "occupancy"):
            if c in out.columns:
                out[c] = out[c].map(_safe_float)
        if "multiplicity" in out.columns:
            out["multiplicity"] = out["multiplicity"].map(_safe_int)
        atom_sites_df = out
    else:
        atom_sites_df = pd.DataFrame(columns=[
            "label", "type_symbol", "fract_x", "fract_y", "fract_z", "B_iso", "occupancy", "Wyckoff", "multiplicity"
        ])

    return raw_header, cell, space_group, atom_sites_df


# ------------------------------ Pretty printing ------------------------------

def preview_parsed_data(
    pets_header: Dict[str, str],
    UB: np.ndarray,
    wavelength: float,
    zone_axes_df: pd.DataFrame,
    reflections_df: pd.DataFrame,
    cif_cell: Dict[str, float],
    cif_space_group: Optional[str],
    cif_atoms: pd.DataFrame,
    max_rows: int = 7,
) -> None:
    """Print a concise preview of all parsed pieces."""
    print("\n=== PETS header (selected) ===")
    for k in [
        "_cell_length_a", "_cell_length_b", "_cell_length_c",
        "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
        "_cell_volume", "_diffrn_radiation_wavelength",
        "_diffrn_pets_omega", "_diffrn_measurement_details"
    ]:
        if k in pets_header:
            val = pets_header[k]
            if k == "_diffrn_measurement_details" and len(val) > 120:
                val = val[:117] + "..."
            print(f"{k:35s} {val}")

    print("\nUB matrix (from _diffrn_orient_matrix_UB_ij):")
    with np.printoptions(precision=5, suppress=True):
        print(UB)

    print(f"\nWavelength (Å): {wavelength}")

    print("\n=== Zone axes table ===")
    print(f"Rows: {len(zone_axes_df)}   Columns: {list(zone_axes_df.columns)}")
    if len(zone_axes_df) > 0:
        print(zone_axes_df.head(max_rows).to_string(index=False))

    print("\n=== Reflections table ===")
    print(f"Rows: {len(reflections_df)}   Columns: {list(reflections_df.columns)}")
    if len(reflections_df) > 0:
        print(reflections_df.head(max_rows).to_string(index=False))

    print("\n=== Reference CIF: cell & space group ===")
    print(f"Cell (Å/deg): {cif_cell}")
    print(f"Space group:  {cif_space_group}")

    print("\n=== Reference CIF: atom sites (first rows) ===")
    print(f"Rows: {len(cif_atoms)}   Columns: {list(cif_atoms.columns)}")
    if len(cif_atoms) > 0:
        print(cif_atoms.head(max_rows).to_string(index=False))


# ------------------------------ Main script ----------------------------------

def main(experiment_path: str, cif_path: str) -> None:
    """Run Stage 1 parsing and print previews."""
    pets_header, UB, wavelength, zone_axes_df, reflections_df = parse_pets_cif_pets(experiment_path)
    cif_header, cif_cell, cif_space_group, cif_atoms = parse_reference_cif(cif_path)

    preview_parsed_data(
        pets_header=pets_header,
        UB=UB,
        wavelength=wavelength,
        zone_axes_df=zone_axes_df,
        reflections_df=reflections_df,
        cif_cell=cif_cell,
        cif_space_group=cif_space_group,
        cif_atoms=cif_atoms,
    )


if __name__ == "__main__":
    # Adjust these paths as needed
    EXPERIMENT_PATH = "Si_3_dyn.cif_pets"
    CIF_PATH = "silicon_structure.cif"
    main(EXPERIMENT_PATH, CIF_PATH)
