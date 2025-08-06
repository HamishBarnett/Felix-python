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
            'frac_coords': list(site.frac_coords)
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
    cif_file = "silicon_structure.cif"  # Replace with actual path
    data = parse_structure_cif(cif_file)

    # Access output
    structure_obj = data['structure']
    lattice_params = data['lattice_parameters']
    atom_sites = data['atom_positions']


import gemmi
from collections import defaultdict
import numpy as np

def parse_pets_dyn_cif(cif_path):
    """
    Parse a PETS .dyn.cif file and extract geometry and reflection data.

    Parameters:
        cif_path (str): Path to the .dyn.cif file.

    Returns:
        dict: {
            'unit_cell': (a, b, c, alpha, beta, gamma),
            'wavelength': float,
            'UB_matrix': 3x3 numpy array,
            'frame_step': float (deg/frame),
            'n_frames': int,
            'reflection_profiles': { (h,k,l): [ (frame, intensity), ... ] },
            'centroid_frames': { (h,k,l): centroid_frame (float) }
        }
    """
    doc = gemmi.cif.read_file(cif_path)
    block = doc.sole_block()

    # --- Unit cell
    a = float(block.find_value('_cell_length_a'))
    b = float(block.find_value('_cell_length_b'))
    c = float(block.find_value('_cell_length_c'))
    alpha = float(block.find_value('_cell_angle_alpha'))
    beta = float(block.find_value('_cell_angle_beta'))
    gamma = float(block.find_value('_cell_angle_gamma'))
    unit_cell = (a, b, c, alpha, beta, gamma)

    # --- Wavelength
    wavelength = float(block.find_value('_diffrn_radiation_wavelength'))

    # --- UB matrix
    UB = np.array([
        [float(block.find_value('_diffrn_orient_matrix_UB_11')),
         float(block.find_value('_diffrn_orient_matrix_UB_12')),
         float(block.find_value('_diffrn_orient_matrix_UB_13'))],
        [float(block.find_value('_diffrn_orient_matrix_UB_21')),
         float(block.find_value('_diffrn_orient_matrix_UB_22')),
         float(block.find_value('_diffrn_orient_matrix_UB_23'))],
        [float(block.find_value('_diffrn_orient_matrix_UB_31')),
         float(block.find_value('_diffrn_orient_matrix_UB_32')),
         float(block.find_value('_diffrn_orient_matrix_UB_33'))]
    ])

    # --- Frame step and frame count from _diffrn_measurement_details
    details = block.find_value('_diffrn_measurement_details')
    frame_step = None
    n_frames = None
    for line in details.split('\n'):
        if 'step between frames' in line:
            frame_step = float(line.split(':')[-1].strip())
        elif 'number of merged frames' in line:
            n_frames = int(line.split(':')[-1].strip())
    if frame_step is None or n_frames is None:
        raise ValueError("Could not extract frame step or frame count.")

    # --- Parse reflection data loop
    loop = block.find_loop('_refln_index_h')
    tags = loop.tags
    idx = {tag: i for i, tag in enumerate(tags)}

    h_idx = idx['_refln_index_h']
    k_idx = idx['_refln_index_k']
    l_idx = idx['_refln_index_l']
    I_idx = idx['_refln_intensity_meas']
    frame_idx = idx.get('_refln_zone_axis_id')  # typically holds frame index

    reflection_profiles = defaultdict(list)
    for row in loop:
        try:
            h = int(row[h_idx])
            k = int(row[k_idx])
            l = int(row[l_idx])
            intensity = float(row[I_idx])
            frame = int(row[frame_idx]) if frame_idx is not None else 0
            if intensity > 0:
                reflection_profiles[(h, k, l)].append((frame, intensity))
        except Exception:
            continue  # skip malformed rows

    # --- Compute centroid frame (sub-frame precision)
    centroid_frames = {}
    for hkl, data in reflection_profiles.items():
        frames = np.array([f for f, _ in data])
        intensities = np.array([I for _, I in data])
        if intensities.sum() == 0:
            centroid = np.mean(frames)
        else:
            centroid = np.sum(frames * intensities) / np.sum(intensities)
        centroid_frames[hkl] = centroid

    # --- Final return
    return {
        'unit_cell': unit_cell,
        'wavelength': wavelength,
        'UB_matrix': UB,
        'frame_step': frame_step,
        'n_frames': n_frames,
        'reflection_profiles': dict(reflection_profiles),
        'centroid_frames': centroid_frames
    }

# Example usage
if __name__ == "__main__":
    cif_file = "Si_3_dyn.cif_pets"  # Replace with your actual file
    data = parse_pets_dyn_cif(cif_file)

    print("  Parsed PETS .dyn.cif successfully")
    print("  Unit cell:", data['unit_cell'])
    print("  Wavelength:", data['wavelength'])
    print("  Frame step (deg):", data['frame_step'])
    print("  UB matrix:\n", data['UB_matrix'])
    print("  Total reflections:", len(data['reflection_profiles']))
    print("  Sample centroid frames:")
    for hkl, cf in list(data['centroid_frames'].items())[:5]:
        print(f"    {hkl} → centroid frame = {cf:.2f}")


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
