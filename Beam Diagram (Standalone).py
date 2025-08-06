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
