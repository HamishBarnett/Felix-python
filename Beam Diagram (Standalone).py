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

def generate_allowed_reflections(structure, d_min=0.01, hkl_limit=10):
    """
    Generate allowed reciprocal lattice vectors g = h·a* + k·b* + l·c*
    within a given d-spacing threshold.

    Parameters:
        structure (pymatgen.Structure): The crystal structure object.
        d_min (float): Minimum allowed d-spacing in Å (default: 0.01 Å).
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

    allowed_reflections = generate_allowed_reflections(structure, d_min=0.01, hkl_limit=8)

    # Show sample output
    print("Sample allowed reflections:")
    for r in allowed_reflections[:5]:
        print(f"  hkl = {r['hkl']}, |g| = {r['g_len']:.3f} Å⁻¹, d = {r['d_spacing']:.3f} Å")

import numpy as np

def identify_bragg_reflections(reflections, wavelength, beam_direction=[0, 0, 1], tolerance=0.5):
    """
    Identify which reflections satisfy the Bragg condition:
        2 * k0 · g ≈ |g|^2

    Parameters:
        reflections (list): Output from Section C (each with hkl and g_cart)
        wavelength (float): Electron wavelength in Å
        beam_direction (list): Unit vector of incident beam (default [0, 0, 1])
        tolerance (float): Max deviation from Bragg condition to accept (Å⁻²)

    Returns:
        list of dict: Each entry is {
            'hkl': (h, k, l),
            'g_cart': np.array,
            'bragg_error': float (excitation error),
        }
    """
    beam_unit = np.array(beam_direction) / np.linalg.norm(beam_direction)
    k0 = beam_unit * (1 / wavelength)  # |k0| = 1 / λ

    bragg_reflections = []

    for refl in reflections:
        g = refl['g_cart']
        g_len_sq = np.dot(g, g)
        two_k0_dot_g = 2 * np.dot(k0, g)
        bragg_diff = abs(two_k0_dot_g - g_len_sq)
        if bragg_diff < tolerance:
            bragg_reflections.append({
                'hkl': refl['hkl'],
                'g_cart': g,
                'bragg_error': two_k0_dot_g - g_len_sq
            })

    print(f" Found {len(bragg_reflections)} Bragg-allowed reflections within tolerance ±{tolerance}")
    return bragg_reflections

# Example usage
if __name__ == "__main__":
    from pymatgen.io.cif import CifParser

    structure = CifParser("silicon_structure.cif").get_structures()[0]
    reflections = generate_allowed_reflections(structure, d_min=0.01, hkl_limit=8)

    # Wavelength of 200 keV electrons ≈ 0.02508 Å (as from your dyn.cif)
    wavelength = 0.02508
    bragg_refls = identify_bragg_reflections(reflections, wavelength)

    print("Sample Bragg-allowed reflections:")
    for b in bragg_refls[:5]:
        print(f"  hkl = {b['hkl']}, Bragg error = {b['bragg_error']:.3e}")


import numpy as np

def compute_exit_wavevectors(bragg_reflections, wavelength, beam_direction=[0, 0, 1]):
    """
    For each Bragg-allowed reflection, compute the exit wavevector:
        k_exit = k0 + g

    Parameters:
        bragg_reflections (list): Output from Section D, each with g_cart and hkl.
        wavelength (float): Electron wavelength in Å.
        beam_direction (list): Unit vector of incident beam direction (default [0, 0, 1]).

    Returns:
        list of dict: Each entry is {
            'hkl': (h, k, l),
            'g_cart': np.array,
            'k_exit': np.array,
            'k_dir': np.array (unit vector of k_exit)
        }
    """
    k0_unit = np.array(beam_direction) / np.linalg.norm(beam_direction)
    k0 = k0_unit * (1 / wavelength)

    results = []
    for refl in bragg_reflections:
        g = refl['g_cart']
        k_exit = k0 + g
        k_dir = k_exit / np.linalg.norm(k_exit)

        results.append({
            'hkl': refl['hkl'],
            'g_cart': g,
            'k_exit': k_exit,
            'k_dir': k_dir
        })

    print(f" Computed exit wavevectors for {len(results)} reflections.")
    return results

# Example usage
if __name__ == "__main__":
    # Assumes you've already run sections A–D and have `bragg_refls`
    wavelength = 0.02508
    beam_direction = [0, 0, 1]

    from pymatgen.io.cif import CifParser
    
    structure = CifParser("silicon_structure.cif").parse_structures(primitive=True)[0]
    allowed = generate_allowed_reflections(structure, d_min=0.01, hkl_limit=8)
    
    # Increased tolerance to allow reasonable Bragg matches
    bragg_refls = identify_bragg_reflections(allowed, wavelength=0.02508, beam_direction=[0,0,1], tolerance=0.5)
    
    exit_waves = compute_exit_wavevectors(bragg_refls, wavelength=0.02508)
    
    for ew in exit_waves[:5]:
        print(f"hkl = {ew['hkl']} → k_dir = {ew['k_dir']}")


import numpy as np

def gnomonic_projection(exit_wave_data, proj_plane_normal=[0, 0, 1]):
    """
    Apply gnomonic projection to exit wavevectors to convert them into 2D coordinates.

    Parameters:
        exit_wave_data (list): List of dicts from Section E, each with 'hkl' and 'k_dir'.
        proj_plane_normal (list): Normal to projection plane (default: [0, 0, 1] → x–y plane).

    Returns:
        list of dicts: Each with {
            'hkl': (h, k, l),
            'projected_2d': (x_proj, y_proj),
            'k_dir': np.array
        }
    """
    normal = np.array(proj_plane_normal)
    normal = normal / np.linalg.norm(normal)  # ensure unit vector

    # Define two orthogonal vectors in the projection plane
    if np.allclose(normal, [0, 0, 1]):
        # If projecting onto x–y plane, use x and y
        u = np.array([1, 0, 0])  # x-direction
        v = np.array([0, 1, 0])  # y-direction
    else:
        # Generate orthonormal basis for arbitrary plane
        u = np.cross([0, 1, 0], normal)
        if np.linalg.norm(u) < 1e-6:
            u = np.cross([1, 0, 0], normal)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)

    projected_points = []

    for refl in exit_wave_data:
        k = refl['k_dir']
        denom = np.dot(k, normal)
        if denom <= 0:
            continue  # Skip beams pointing away from the projection plane

        scale = 1.0 / denom
        intersect = scale * k  # projected intersection point on the plane

        x_proj = np.dot(intersect, u)
        y_proj = np.dot(intersect, v)

        projected_points.append({
            'hkl': refl['hkl'],
            'projected_2d': (x_proj, y_proj),
            'k_dir': k
        })

    print(f" Projected {len(projected_points)} reflections onto 2D plane.")
    return projected_points

# Example usage
if __name__ == "__main__":
    from pymatgen.io.cif import CifParser

    # Load structure
    structure = CifParser("silicon_structure.cif").get_structures()[0]
    allowed = generate_allowed_reflections(structure, d_min=0.01, hkl_limit=8)
    bragg = identify_bragg_reflections(allowed, wavelength=0.02508)
    exit_waves = compute_exit_wavevectors(bragg, wavelength=0.02508)

    projected = gnomonic_projection(exit_waves, proj_plane_normal=[0, 0, 1])

    # Print a few projected points
    for p in projected[:5]:
        hkl = p['hkl']
        x, y = p['projected_2d']
        print(f"hkl = {hkl} → x' = {x:.3f}, y' = {y:.3f}")

import matplotlib.pyplot as plt

def plot_beam_diagram(
    projected_reflections,
    centroid_frames,
    frame_step,
    beam_path_y=0,
    title="2D Beam Diagram"
):
    """
    Plot the 2D beam diagram.

    Parameters:
        projected_reflections (list): Output from gnomonic_projection.
        centroid_frames (dict): {hkl: frame number} from PETS dyn.cif.
        frame_step (float): Degrees per frame (e.g., 0.001).
        beam_path_y (float): y-position of direct beam path (default: 0).
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('black')

    # Plot red line: beam path
    x_vals = [cf * frame_step for cf in centroid_frames.values()]
    ax.plot(
        [min(x_vals), max(x_vals)],
        [beam_path_y, beam_path_y],
        color='red', linewidth=1.5, label='Beam Path (red)'
    )

    # White lines: Bragg condition traces (here plotted as points)
    for refl in projected_reflections:
        hkl = refl['hkl']
        x = centroid_frames.get(hkl)
        if x is not None:
            x_proj = x * frame_step
            y_proj = refl['projected_2d'][1]
            ax.plot(x_proj, y_proj, 'w.', alpha=0.6)

    # Blue vertical lines: frame positions
    for hkl, frame in centroid_frames.items():
        x_pos = frame * frame_step
        ax.axvline(x_pos, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)

    # Yellow dots: intersection points (on beam path at centroid frame)
    for refl in projected_reflections:
        hkl = refl['hkl']
        frame = centroid_frames.get(hkl)
        if frame is not None:
            x_pos = frame * frame_step
            ax.plot(x_pos, beam_path_y, 'o', color='yellow', markersize=3)

    ax.set_xlabel("Beam Tilt / Rotation (degrees)")
    ax.set_ylabel("Reciprocal Space Projection (arb. units)")
    ax.set_title(title)
    ax.legend(loc='upper right', facecolor='black', framealpha=0.2, fontsize=8)
    ax.set_xlim(min(x_vals), max(x_vals))
    ax.grid(False)
    plt.tight_layout()
    plt.show()


    # Assumes previous sections have provided these:
    # projected = output from Section F
    # centroid_frames = from Section B
    # frame_step = from Section B (e.g., 0.001)
    
    plot_beam_diagram(
        projected_reflections=projected,
        centroid_frames=centroid_frames,
        frame_step=0.001,
        beam_path_y=0,
        title="Simulated Beam Diagram (Initial Orientation)"
    )

