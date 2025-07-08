# compute.py - Highly optimized CPU version
import numpy as np
from scipy.linalg import inv
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from numba import njit, prange
import warnings

# Suppress warnings from potential numerical issues
warnings.filterwarnings('ignore')

@njit(fastmath=True, cache=True)
def _compute_single_strain_numba(A, B):
    """
    Numba-optimized version of the strain computation.
    Uses Numba's @njit decorator to compile to machine code.
    """
    try:
        # Compute A.T @ A
        ATA = A.T @ A
        # Compute inverse (Numba's implementation will be fast)
        D = np.linalg.inv(ATA)
        # Compute C = B@B.T - A@A.T
        C = B @ B.T - A @ A.T
        # Compute Q matrix
        temp1 = D @ A.T
        temp2 = temp1 @ C
        Q = 0.5 * (temp2 @ A @ D)
        Q = 0.5 * (Q + Q.T)  # Ensure symmetry

        # Compute eigenvalues and sort in descending order
        eigenvalues = np.linalg.eigh(Q)[0]
        # Manual sort for eigenvalues since Numba doesn't support numpy's sort well
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]

        # Compute shear strain
        trace_Q = np.trace(Q)
        trace_Q2 = np.trace(Q @ Q)
        shear = trace_Q2 - (1/3) * (trace_Q ** 2)

        return shear, sorted_eigenvalues
    except Exception:
        # Return NaNs if computation fails
        return np.nan, np.array([np.nan, np.nan, np.nan])

def process_frame_data(ref_positions_list, ref_centers_list, def_positions_list, def_centers_list, parallel=False, batch_size=100):
    """
    Highly optimized version with multiple techniques for speed improvement.

    Args:
        ref_positions_list: List of reference positions for each atom
        ref_centers_list: List of reference centers for each atom
        def_positions_list: List of deformed positions for each atom
        def_centers_list: List of deformed centers for each atom
        parallel: If True, use multiprocessing (default: False)
        batch_size: Number of atoms to process in each batch (for parallel processing)
    """
    n_atoms = len(ref_positions_list)
    shear_strains = np.zeros(n_atoms, dtype=np.float32)
    principal_strains = np.zeros((n_atoms, 3), dtype=np.float32)

    # Pre-process: center positions and count neighbors
    centered_ref_pos = []
    centered_def_pos = []
    neighbor_counts = []
    valid_atoms = []

    for i in range(n_atoms):
        ref_pos = ref_positions_list[i]
        def_pos = def_positions_list[i]
        ref_center = ref_centers_list[i]
        def_center = def_centers_list[i]

        count = len(ref_pos)
        neighbor_counts.append(count)

        if count >= 4:
            valid_atoms.append(i)
            centered_ref_pos.append(ref_pos - ref_center)
            centered_def_pos.append(def_pos - def_center)
        else:
            shear_strains[i] = np.nan
            principal_strains[i] = np.nan

    if not valid_atoms:
        return shear_strains, principal_strains

    # Group by neighbor count for batch processing
    count_to_indices = defaultdict(list)
    for i, atom_idx in enumerate(valid_atoms):
        count = len(centered_ref_pos[i])
        count_to_indices[count].append((atom_idx, i))  # Store original atom index and position in valid_atoms

    if parallel:
        # Parallel processing version with optimized batching
        def process_task(task):
            count, indices = task
            results = []
            for atom_idx, valid_idx in indices:
                A = centered_ref_pos[valid_idx]
                B = centered_def_pos[valid_idx]
                shear, principal = _compute_single_strain_numba(A, B)
                results.append((atom_idx, shear, principal))
            return results

        # Create tasks grouped by neighbor count
        tasks = []
        for count, indices in count_to_indices.items():
            # Split into batches if there are many atoms with this count
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                tasks.append((count, batch_indices))

        # Process in parallel
        with Pool(processes=min(cpu_count(), len(tasks))) as pool:
            results = pool.map(process_task, tasks)

        # Flatten and store results
        for batch_results in results:
            for atom_idx, shear, principal in batch_results:
                shear_strains[atom_idx] = float(shear)
                principal_strains[atom_idx] = principal

    else:
        # Sequential processing version with Numba optimization
        for count, indices in count_to_indices.items():
            for atom_idx, valid_idx in indices:
                A = centered_ref_pos[valid_idx]
                B = centered_def_pos[valid_idx]
                shear, principal = _compute_single_strain_numba(A, B)
                shear_strains[atom_idx] = float(shear)
                principal_strains[atom_idx] = principal

    return shear_strains, principal_strains
