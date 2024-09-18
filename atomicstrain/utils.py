import numpy as np
from MDAnalysis.analysis.distances import distance_array

def create_selections(ref, defm, residue_numbers, min_neighbors=3, use_all_heavy=False):
    """
    Create atom selections for strain analysis.

    This function generates paired selections of atoms from the reference and deformed
    structures for each specified residue.

    Args:
        ref (MDAnalysis.Universe): Reference structure Universe.
        defm (MDAnalysis.Universe): Deformed structure Universe.
        residue_numbers (list): List of residue numbers to analyze.
        min_neighbors (int): Minimum number of neighbors for atom selection.
        use_all_heavy (bool): If True, use all heavy atoms. If False, use only CA atoms.

    Returns:
        list: A list of tuples, each containing:
            ((ref_selection, ref_center), (defm_selection, defm_center))
            where each selection is an MDAnalysis.AtomGroup.
    """
    selections = []
    
    # Select atoms based on the use_all_heavy flag
    if use_all_heavy:
        atom_selection = f"not name H* and resid {' '.join(map(str, residue_numbers))}"
    else:
        atom_selection = f"name CA and resid {' '.join(map(str, residue_numbers))}"

    ref_atoms = ref.select_atoms(atom_selection)
    defm_atoms = defm.select_atoms(atom_selection)
    
    print(f"Debug: Number of atoms selected: {len(ref_atoms)}")
    
    # Calculate distances between all selected atoms in the reference structure
    distances = distance_array(ref_atoms.positions, ref_atoms.positions)
    
    print(f"Debug: Shape of distances array: {distances.shape}")
    
    # For each atom, find the distance to its (min_neighbors+1)th nearest neighbor
    sorted_distances = np.sort(distances, axis=1)
    neighbor_distances = sorted_distances[:, min_neighbors+1]
    
    # Use the maximum of these distances as our new radius
    R = np.max(neighbor_distances)
    print(f"Debug: Calculated radius R = {R}")
    
    for resid in residue_numbers:
        if use_all_heavy:
            ref_residue_atoms = ref_atoms.select_atoms(f"resid {resid} and not name H*")
            defm_residue_atoms = defm_atoms.select_atoms(f"resid {resid} and not name H*")
        else:
            ref_residue_atoms = ref_atoms.select_atoms(f"name CA and resid {resid}")
            defm_residue_atoms = defm_atoms.select_atoms(f"name CA and resid {resid}")

        for i, ref_atom in enumerate(ref_residue_atoms):
            # Select atoms within radius R of the current atom in the reference structure
            ref_selection = ref_atoms[distances[ref_atoms.indices.tolist().index(ref_atom.index)] <= R]
            ref_center = ref_atom
            
            # Use the same atom indices for the deformed structure
            defm_selection = defm_atoms[distances[ref_atoms.indices.tolist().index(ref_atom.index)] <= R]
            defm_center = defm_residue_atoms[i]
            
            print(f"Resid {resid}, Atom {ref_atom.name}: Ref atoms: {len(ref_selection)}, Defm atoms: {len(defm_selection)}")
            
            selections.append(((ref_selection, ref_center), (defm_selection, defm_center)))
    
    return selections