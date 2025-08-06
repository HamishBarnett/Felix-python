from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

def parse_structure_cif(cif_path):
    """
    Parses a crystallographic CIF file and extracts lattice parameters,
    atomic positions, and a pymatgen Structure object.

    Parameters:
        cif_path (str): Path to the CIF file.

    Returns:
        dict: {
            'structure': pymatgen Structure object,
            'lattice_parameters': (a, b, c, alpha, beta, gamma),
            'atom_positions': [ {'element': str, 'frac_coords': list}, ... ]
        }
    """
    parser = CifParser(cif_path)
    structure = parser.get_structures()[0]  # Get first (or only) structure
    lattice = structure.lattice

    # Extract lattice parameters
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    lattice_params = (a, b, c, alpha, beta, gamma)

    # Extract atom positions (fractional coordinates)
    atom_positions = []
    for site in structure.sites:
        atom_positions.append({
            'element': site.species_string,
            'frac_coords': [float(x) for x in site.frac_coords]
        })

    # Print summary
    print("   Structure parsed successfully.")
    print(f"  Lattice: a={a:.4f}, b={b:.4f}, c={c:.4f}")
    print(f"  Angles: alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}")
    print(f"  Total atoms: {len(atom_positions)}")
    for atom in atom_positions:
        print(f"    {atom['element']} at {atom['frac_coords']}")

    return {
        'structure': structure,
        'lattice_parameters': lattice_params,
        'atom_positions': atom_positions
    }

# Example usage:
if __name__ == "__main__":
    cif_file = "silicon_structure.cif"  # Replace with path
    data = parse_structure_cif(cif_file)

    # Access output
    structure_obj = data['structure']
    lattice_params = data['lattice_parameters']
    atom_sites = data['atom_positions']


import re
from collections import defaultdict
import numpy as np

def parse_dyn_cif_manual(filepath):
    """
    Manually parse PETS dyn.cif without using gemmi.
    Extracts unit cell, wavelength, UB matrix, frame step, and reflection intensities.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    unit_cell = {}
    UB = np.zeros((3, 3))
    wavelength = None
    frame_step = None
    n_frames = None
    reflections = []

    in_loop = False
    loop_tags = []
    data_rows = []

    for i, line in enumerate(lines):
        line = line.strip()

        # --- Unit cell
        if line.startswith('_cell_length_a'):
            unit_cell['a'] = float(line.split()[1])
        elif line.startswith('_cell_length_b'):
            unit_cell['b'] = float(line.split()[1])
        elif line.startswith('_cell_length_c'):
            unit_cell['c'] = float(line.split()[1])
        elif line.startswith('_cell_angle_alpha'):
            unit_cell['alpha'] = float(line.split()[1])
        elif line.startswith('_cell_angle_beta'):
            unit_cell['beta'] = float(line.split()[1])
        elif line.startswith('_cell_angle_gamma'):
            unit_cell['gamma'] = float(line.split()[1])

        # --- Wavelength
        elif line.startswith('_diffrn_radiation_wavelength'):
            wavelength = float(line.split()[1])

        # --- UB matrix
        elif '_diffrn_orient_matrix_UB_' in line:
            parts = line.split()
            idx = int(re.search(r'UB_(\d)(\d)', line).group(1)) - 1
            jdx = int(re.search(r'UB_(\d)(\d)', line).group(2)) - 1
            UB[idx][jdx] = float(parts[1])

        # --- Measurement details
        elif '_diffrn_measurement_details' in line:
            j = i + 1
            while j < len(lines) and not lines[j].startswith('_'):
                if 'step between frames' in lines[j]:
                    frame_step = float(lines[j].split(':')[-1].strip())
                elif 'number of merged frames' in lines[j]:
                    n_frames = int(lines[j].split(':')[-1].strip())
                j += 1

        # --- Start of reflection loop
        elif line.startswith('loop_') and '_refln_index_h' in lines[i+1]:
            in_loop = True
            loop_tags = []
            data_rows = []

        elif in_loop:
            if line.startswith('_'):
                loop_tags.append(line)
            elif line.strip() == '':
                continue
            elif line.startswith('data_'):
                in_loop = False
            else:
                data_rows.append(line.strip().split())

    # --- Parse reflection data
    tag_map = {tag: idx for idx, tag in enumerate(loop_tags)}
    reflection_profiles = defaultdict(list)
    for row in data_rows:
        try:
            h = int(row[tag_map['_refln_index_h']])
            k = int(row[tag_map['_refln_index_k']])
            l = int(row[tag_map['_refln_index_l']])
            I = float(row[tag_map['_refln_intensity_meas']])
            frame = int(row[tag_map['_refln_zone_axis_id']])
            reflection_profiles[(h, k, l)].append((frame, I))
        except Exception:
            continue  # skip malformed lines

    # --- Compute centroid frames
    centroid_frames = {}
    for hkl, values in reflection_profiles.items():
        frames = np.array([f for f, _ in values])
        intensities = np.array([i for _, i in values])
        if intensities.sum() == 0:
            centroid = frames.mean()
        else:
            centroid = np.sum(frames * intensities) / np.sum(intensities)
        centroid_frames[hkl] = centroid

    return {
        'unit_cell': (
            unit_cell['a'], unit_cell['b'], unit_cell['c'],
            unit_cell['alpha'], unit_cell['beta'], unit_cell['gamma']
        ),
        'wavelength': wavelength,
        'UB_matrix': UB,
        'frame_step': frame_step,
        'n_frames': n_frames,
        'reflection_profiles': dict(reflection_profiles),
        'centroid_frames': centroid_frames
    }

dyn_data = parse_dyn_cif_manual("Si_3_dyn.cif_pets")

print("Unit cell:", dyn_data['unit_cell'])
print("Wavelength:", dyn_data['wavelength'])
print("UB matrix:\n", dyn_data['UB_matrix'])
print("Frame step:", dyn_data['frame_step'])
print("Reflections parsed:", len(dyn_data['reflection_profiles']))


import numpy as np
from pymatgen.core import Structure
from itertools import product

def generate_allowed_reflections(structure, d_min=0.5, hkl_limit=10):
    """
    Generate allowed reciprocal lattice vectors g = h·a* + k·b* + l·c*
    within a given d-spacing threshold.

    Parameters:
        structure (pymatgen.Structure): The crystal structure object.
        d_min (float): Minimum allowed d-spacing in Å (default: 0.5 Å).
        hkl_limit (int): Maximum |h|, |k|, |l| to search over (default: 10).

    Returns:
        List[dict]: Each entry is {
            'hkl': (h, k, l),
            'g_cart': np.array([gx, gy, gz]),
            'g_len': float (|g|),
            'd_spacing': float (1/|g|)
        }
    """
    rec_lattice = structure.lattice.reciprocal_lattice
    reflections = []

    for h, k, l in product(range(-hkl_limit, hkl_limit + 1), repeat=3):
        if (h, k, l) == (0, 0, 0):
            continue
        g_cart = rec_lattice.get_cartesian_coords([h, k, l])
        g_len = np.linalg.norm(g_cart)
        d_spacing = 1 / g_len if g_len != 0 else np.inf
        if d_spacing >= d_min:
            reflections.append({
                'hkl': (h, k, l),
                'g_cart': g_cart,
                'g_len': g_len,
                'd_spacing': d_spacing
            })

    print(f" Generated {len(reflections)} allowed reflections with d ≥ {d_min} Å")
    return reflections

# Example usage
if __name__ == "__main__":
    from pymatgen.io.cif import CifParser

    # Load structure from CIF (as in Section A)
    structure = CifParser("silicon_structure.cif").get_structures()[0]

    allowed_reflections = generate_allowed_reflections(structure, d_min=0.5, hkl_limit=8)

    # Show sample output
    print("Sample allowed reflections:")
    for r in allowed_reflections[:5]:
        print(f"  hkl = {r['hkl']}, |g| = {r['g_len']:.3f} Å⁻¹, d = {r['d_spacing']:.3f} Å")
