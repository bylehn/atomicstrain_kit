import numpy as np
from MDAnalysis.analysis.distances import distance_array

def create_selections(ref, defm, residue_numbers, min_neighbors=3):
    """
    Create atom selections for strain analysis.

    This function generates paired selections of atoms from the reference and deformed
    structures for each specified residue.

    Args:
        ref (MDAnalysis.Universe): Reference structure Universe.
        defm (MDAnalysis.Universe): Deformed structure Universe.
        residue_numbers (list): List of residue numbers to analyze.
        R (float): Radius for atom selection.

    Returns:
        list: A list of tuples, each containing:
            ((ref_selection, ref_center), (defm_selection, defm_center))
            where each selection is an MDAnalysis.AtomGroup.
    """
    selections = []
    
    # Select all CA atoms from the specified residues
    ref_cas = ref.select_atoms(f"name CA and resid {' '.join(map(str, residue_numbers))}")
    defm_cas = defm.select_atoms(f"name CA and resid {' '.join(map(str, residue_numbers))}")
    
    print(f"Debug: Number of CA atoms selected: {len(ref_cas)}")
    
    # Calculate distances between all CA atoms in the reference structure
    distances = distance_array(ref_cas.positions, ref_cas.positions)
    
    print(f"Debug: Shape of distances array: {distances.shape}")
    
    # For each atom, find the distance to its (min_neighbors+1)th nearest neighbor
    sorted_distances = np.sort(distances, axis=1)
    neighbor_distances = sorted_distances[:, min_neighbors+1]
    
    # Use the maximum of these distances as our new radius
    R = np.max(neighbor_distances)
    print(f"Debug: Calculated radius R = {R}")
    
    for i, resid in enumerate(residue_numbers):
        # Select CA atoms within radius R of the current CA in the reference structure
        ref_selection = ref_cas[distances[i] <= R]
        ref_center = ref.select_atoms(f"name CA and resid {resid}")
        
        # Use the same atom indices for the deformed structure
        defm_selection = defm_cas[distances[i] <= R]
        defm_center = defm.select_atoms(f"name CA and resid {resid}")
        
        print(f"Resid {resid}: Ref atoms: {len(ref_selection)}, Defm atoms: {len(defm_selection)}")
        
        selections.append(((ref_selection, ref_center), (defm_selection, defm_center)))
    
    return selections