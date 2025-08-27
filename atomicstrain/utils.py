import numpy as np
from MDAnalysis.analysis.distances import distance_array

def create_selections(ref, residue_numbers, strategy='weighted', min_neighbors=3, 
                     radius=8, inner_radius=6, outer_radius=10, use_all_heavy=False):
    """
    Create atom selections for strain analysis.

    This function generates selections of atoms from the reference structure for each 
    specified residue using different neighbor selection strategies. The same selections
    will be used for both reference and deformed structures.

    Args:
        ref (MDAnalysis.Universe): Reference structure Universe.
        residue_numbers (list): List of residue numbers to analyze.
        strategy (str): Strategy for neighbor selection:
            - 'adaptive': Adaptive radius based on min_neighbors (current behavior)
            - 'fixed': Fixed radius with minimum neighbor check
            - 'weighted': Inner/outer radius with differential weighting
        min_neighbors (int): Minimum number of neighbors for atom selection.
        radius (float): Fixed radius for 'fixed' strategy.
        inner_radius (float): Inner radius for 'weighted' strategy.
        outer_radius (float): Outer radius for 'weighted' strategy.
        use_all_heavy (bool): If True, use all heavy atoms. If False, use only CA atoms.

    Returns:
        list: A list of tuples, each containing:
            (selection, center_atom, weights)
            where selection is an MDAnalysis.AtomGroup, center_atom is the central atom,
            and weights is an array of weights for each atom in the selection.
    """
    selections = []
    
    # Validate strategy parameters
    if strategy == 'fixed' and radius is None:
        raise ValueError("Fixed strategy requires 'radius' parameter")
    if strategy == 'weighted' and (inner_radius is None or outer_radius is None):
        raise ValueError("Weighted strategy requires both 'inner_radius' and 'outer_radius' parameters")
    if strategy == 'weighted' and inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius")
    
    # Select atoms based on the use_all_heavy flag
    if use_all_heavy:
        atom_selection = f"not name H* and resid {' '.join(map(str, residue_numbers))}"
    else:
        atom_selection = f"name CA and resid {' '.join(map(str, residue_numbers))}"

    ref_atoms = ref.select_atoms(atom_selection)
    
    print(f"Debug: Number of atoms selected: {len(ref_atoms)}")
    
    # Calculate distances between all selected atoms in the reference structure
    distances = distance_array(ref_atoms.positions, ref_atoms.positions)
    print(f"Debug: Shape of distances array: {distances.shape}")
    
    # Determine radius/radii based on strategy
    if strategy == 'adaptive':
        # For each atom, find the distance to its (min_neighbors+1)th nearest neighbor
        sorted_distances = np.sort(distances, axis=1)
        neighbor_distances = sorted_distances[:, min_neighbors+1]
        R = np.max(neighbor_distances)
        print(f"Debug: Calculated adaptive radius R = {R}")
    elif strategy == 'fixed':
        R = radius
        print(f"Debug: Using fixed radius R = {R}")
    elif strategy == 'weighted':
        print(f"Debug: Using weighted strategy with inner_radius = {inner_radius}, outer_radius = {outer_radius}")
    
    for resid in residue_numbers:
        if use_all_heavy:
            ref_residue_atoms = ref_atoms.select_atoms(f"resid {resid} and not name H*")
        else:
            ref_residue_atoms = ref_atoms.select_atoms(f"name CA and resid {resid}")

        for i, ref_atom in enumerate(ref_residue_atoms):
            ref_atom_idx = ref_atoms.indices.tolist().index(ref_atom.index)
            atom_distances = distances[ref_atom_idx]
            
            if strategy == 'adaptive' or strategy == 'fixed':
                # Select atoms within radius R
                neighbor_mask = atom_distances <= R
                n_neighbors = np.sum(neighbor_mask) - 1  # -1 to exclude self
                
                if strategy == 'fixed' and n_neighbors < 3:
                    raise ValueError(f"Resid {resid}, atom {ref_atom.name}: Only {n_neighbors} neighbors "
                                   f"found within radius {R:.2f}, minimum is 3")
                
                ref_selection = ref_atoms[neighbor_mask]
                weights = np.ones(len(ref_selection))  # Equal weights
                
            elif strategy == 'weighted':
                # Select atoms within outer radius and assign weights
                outer_mask = atom_distances <= outer_radius
                n_neighbors = np.sum(outer_mask) - 1  # -1 to exclude self
                
                if n_neighbors < 3:
                    raise ValueError(f"Resid {resid}, atom {ref_atom.name}: Only {n_neighbors} neighbors "
                                   f"found within outer radius {outer_radius:.2f}, minimum is 3")
                
                ref_selection = ref_atoms[outer_mask]
                
                # Assign weights: higher for inner radius, lower for between inner and outer
                selected_distances = atom_distances[outer_mask]
                weights = np.where(selected_distances <= inner_radius, 1.0, 0.5)
                
                print(f"Resid {resid}, Atom {ref_atom.name}: "
                      f"Inner neighbors: {np.sum(selected_distances <= inner_radius)}, "
                      f"Outer neighbors: {np.sum((selected_distances > inner_radius) & (selected_distances <= outer_radius))}")
            
            print(f"Resid {resid}, Atom {ref_atom.name}: Selected atoms: {len(ref_selection)}")
            
            if strategy != 'weighted':
                if n_neighbors < 4:
                    print(f"Warning: Resid {resid} has only {n_neighbors} neighbors within radius")
            
            selections.append((ref_selection, ref_atom, weights))
    
    return selections