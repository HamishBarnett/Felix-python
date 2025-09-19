# Paste into Spyder and run.
# Adjust paths if needed:
FILEPATH_DAT  = "reflprofiles_strong.dat"   # profiles file
FILEPATH_PETS = "Si_3_dyn.cif_pets"         # PETS CIF with alpha mapping
OUTPUT_PATH   = "hkl_peak_frame_alpha.txt"  # output text file

import os
import re
from collections import defaultdict

# --- Regex to catch the "concatenated float + frame" at the line end ---
# Captures: <float with exactly 6 decimals><frame digits> <final float>
# Example: 300.8695981337 -245.488  ->  float=300.869598, frame=1337
_CONCAT_END = re.compile(r'([+-]?\d+\.\d{6})(\d{3,6})\s+([+-]?\d+\.\d+)\s*$')

# -----------------------------
# Robust PETS CIF loop parser
# -----------------------------
def _is_loop_terminator(s: str) -> bool:
    s = s.strip()
    if not s:
        # blank line: treat as loop terminator in this file (safe for numeric loops)
        return True
    return s.startswith("_") or s.lower().startswith(("loop_", "data_", "save_", ";"))

def _tokenize_loop_values(lines, start_idx):
    tokens = []
    i, n = start_idx, len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        if _is_loop_terminator(line):
            break
        # strip inline comments
        if "#" in line:
            line = line.split("#", 1)[0]
        parts = line.split()
        tokens.extend(parts)
        i += 1
    return tokens, i

def load_alpha_map_from_pets(filepath):
    """
    Build dict: frame -> alpha (degrees) from ANY loop that includes both
    _diffrn_zone_axis_id and _diffrn_zone_axis_alpha (column order agnostic).
    Works whether rows are one-per-line or wrapped.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PETS file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    alpha_map = {}
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip().lower().startswith("loop_"):
            # gather headers
            j = i + 1
            headers = []
            while j < n and lines[j].strip().startswith("_"):
                headers.append(lines[j].split()[0].strip())
                j += 1
            hl = [h.lower() for h in headers]

            if "_diffrn_zone_axis_id" in hl and "_diffrn_zone_axis_alpha" in hl:
                id_idx    = hl.index("_diffrn_zone_axis_id")
                alpha_idx = hl.index("_diffrn_zone_axis_alpha")
                tokens, end_idx = _tokenize_loop_values(lines, j)

                hcount = len(headers)
                # group tokens into rows of length hcount
                rows = [tokens[k:k+hcount] for k in range(0, len(tokens), hcount)]
                if rows and len(rows[-1]) != hcount:
                    rows.pop()

                for row in rows:
                    try:
                        frame_id = int(float(row[id_idx]))
                        alpha    = float(row[alpha_idx])
                        alpha_map[frame_id] = alpha
                    except Exception:
                        pass
                i = end_idx
                continue
            else:
                i = j
                continue
        i += 1

    if not alpha_map:
        raise ValueError("No _diffrn_zone_axis_id/_diffrn_zone_axis_alpha found in PETS.")
    return alpha_map

# -------------------------------------------
# Profiles reader: find peak frame per (h,k,l)
# -------------------------------------------
def extract_frame_from_line(line: str):
    """
    Return frame number from a profiles line.
    Prefer the normal '... <frame> <angle>' layout.
    If absent, split a '... <float(6dp)><frame> <angle>' concatenation at the end.
    """
    s = line.rstrip("\n")
    parts = s.split()
    # Normal case: last two tokens are <frame> <angle>
    if len(parts) >= 2:
        try:
            return int(parts[-2])
        except Exception:
            pass
    # Concatenated case at end of line
    m = _CONCAT_END.search(s)
    if m:
        return int(m.group(2))
    return None

def get_peak_frame_by_hkl(dat_path):
    """
    Read the profiles file and determine, for each (h,k,l),
    the frame where its intensity (7th numeric column) is maximal.
    """
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"Profiles file not found: {dat_path}")

    per_hkl = defaultdict(list)  # (h,k,l) -> list of (frame, intensity)

    with open(dat_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Need at least 7 tokens to get intensity at index 6
            if len(parts) < 7:
                continue
            try:
                h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                intensity = float(parts[6])  # correct per-frame intensity
            except Exception:
                continue

            frame = extract_frame_from_line(line)
            if frame is None:
                continue

            per_hkl[(h, k, l)].append((frame, intensity))

    # Collapse duplicates per frame by max intensity, and sort by frame
    per_hkl_sorted = {}
    for hkl, rows in per_hkl.items():
        by_frame = {}
        for fr, I in rows:
            if fr not in by_frame or I > by_frame[fr]:
                by_frame[fr] = I
        per_hkl_sorted[hkl] = sorted(by_frame.items(), key=lambda t: t[0])
    return per_hkl_sorted

def frame_of_peak(seq):
    """
    seq: list[(frame, I)] sorted by frame.
    Returns the lowest frame among those that share the global max intensity.
    """
    if not seq:
        return None
    maxI = max(I for _, I in seq)
    for fr, I in seq:
        if I == maxI:
            return fr
    return None

# --------------------
# Writing the text file
# --------------------
def write_results_txt(rows, out_path):
    """
    rows: iterable of (h, k, l, frame, alpha)
    """
    W_HKL   = 6
    W_FRAME = 14
    W_ALPHA = 20
    GAP     = " " * 6

    header = (
        f"{'h':>{W_HKL}} {'k':>{W_HKL}} {'l':>{W_HKL}}{GAP}"
        f"{'frame_at_peak':>{W_FRAME}}{GAP}"
        f"{'alpha_at_peak (deg)':>{W_ALPHA}}"
    )
    line_sep = "-" * len(header)

    with open(out_path, "w", encoding="utf-8") as w:
        w.write(header + "\n")
        w.write(line_sep + "\n")
        for h, k, l, frame, alpha in rows:
            alpha_str = f"{alpha:.3f}" if alpha is not None else "NA"
            w.write(
                f"{h:>{W_HKL}} {k:>{W_HKL}} {l:>{W_HKL}}{GAP}"
                f"{frame:>{W_FRAME}}{GAP}"
                f"{alpha_str:>{W_ALPHA}}\n"
            )

# ------------
# Main routine
# ------------
if __name__ == "__main__":
    # 1) Read PETS alpha map
    alpha_map = load_alpha_map_from_pets(FILEPATH_PETS)
    min_alpha_f, max_alpha_f = min(alpha_map), max(alpha_map)

    # 2) Parse profiles and find peak frame per hkl
    hkl_profiles = get_peak_frame_by_hkl(FILEPATH_DAT)

    results = []
    missing_alpha = 0
    max_frame_seen = 0
    min_frame_seen = 10**9

    for (h, k, l), seq in hkl_profiles.items():
        peak_fr = frame_of_peak(seq)
        if peak_fr is None:
            continue
        if peak_fr > max_frame_seen: max_frame_seen = peak_fr
        if peak_fr < min_frame_seen: min_frame_seen = peak_fr
        alpha = alpha_map.get(peak_fr)  # None if outside 1..1389
        if alpha is None and not (min_alpha_f <= peak_fr <= max_alpha_f):
            missing_alpha += 1
        results.append((h, k, l, peak_fr, alpha))

    # 3) Sort by frame ascending, then h,k,l
    results.sort(key=lambda r: (r[3], r[0], r[1], r[2]))

    # 4) Write everything to the output text file
    write_results_txt(results, OUTPUT_PATH)

    # 5) Console validation / audit
    print(f"Wrote {len(results)} hkls to '{OUTPUT_PATH}'.")
    print(f"Peak frames observed: min={min_frame_seen}, max={max_frame_seen}")
    print(f"PETS alpha coverage: min={min_alpha_f}, max={max_alpha_f}")
    if missing_alpha:
        print(f"Note: {missing_alpha} reflection(s) peak outside the PETS alpha map (alpha printed as 'NA').")
