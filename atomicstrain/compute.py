# compute.py - Final optimized CPU version
import numpy as np
from scipy.linalg import inv
from collections import defaultdict
from multiprocessing import Pool, cpu_count

def _compute_single_strain(A, B):
    """Core computation for a single strain tensor using numpy."""
    ATA = np.dot(A.T, A)
    D = inv(ATA)
    C = np.dot(B, B.T) - np.dot(A, A.T)
    Q = 0.5 * np.dot(np.dot(np.dot(D, A.T), C), np.dot(A, D))
    Q = 0.5 * (Q + Q.T)
    eigenvalues, _ = np.linalg.eigh(Q)
    eigenvalues = np.sort(eigenvalues)[::-1]
    shear = np.trace(np.dot(Q, Q)) - (1/3) * np.square(np.trace(Q))
    return shear, eigenvalues

def process_batch_parallel(task_data):
    """Parallel worker function for processing batches."""
    batch_indices, ref_pos_batch, def_pos_batch, count = task_data
    results = []

    for j in range(len(batch_indices)):
        idx = batch_indices[j]
        # Extract only the valid portion (non-padded part)
        A = ref_pos_batch[j, :count]
        B = def_pos_batch[j, :count]

        try:
            shear, principal = _compute_single_strain(A, B)
            results.append((idx, shear, principal))
        except Exception as e:
            print(f"Error computing strain: {str(e)}")
            results.append((idx, np.nan, np.array([np.nan, np.nan, np.nan])))

    return results

def process_frame_data(ref_positions_list, ref_centers_list, def_positions_list, def_centers_list, parallel=False):
    """
    Optimized CPU version with optional parallel processing.

    Args:
        ref_positions_list: List of reference positions for each atom
        ref_centers_list: List of reference centers for each atom
        def_positions_list: List of deformed positions for each atom
        def_centers_list: List of deformed centers for each atom
        parallel: If True, use multiprocessing (default: False)

    Returns:
        Tuple of shear strains and principal strains arrays
    """
    n_atoms = len(ref_positions_list)
    shear_strains = np.zeros(n_atoms, dtype=np.float32)
    principal_strains = np.zeros((n_atoms, 3), dtype=np.float32)

    # First process atoms with sufficient neighbors in batches by count
    count_to_indices = defaultdict(list)
    neighbor_counts = []

    # Collect neighbor counts and group indices
    for i in range(n_atoms):
        count = len(ref_positions_list[i])
        neighbor_counts.append(count)
        if count >= 4:
            count_to_indices[count].append(i)

    # For atoms with insufficient neighbors
    for i in range(n_atoms):
        if neighbor_counts[i] < 4:
            shear_strains[i] = np.nan
            principal_strains[i] = np.nan

    if not count_to_indices:
        return shear_strains, principal_strains

    if parallel:
        # Parallel processing version
        tasks = []
        max_count = max(count_to_indices.keys())

        # Prepare tasks
        for count, indices in count_to_indices.items():
            if count < 4 or len(indices) == 0:
                continue

            batch_size = len(indices)
            ref_pos_batch = np.zeros((batch_size, max_count, 3), dtype=np.float32)
            def_pos_batch = np.zeros((batch_size, max_count, 3), dtype=np.float32)

            for j, idx in enumerate(indices):
                ref_pos = ref_positions_list[idx]
                def_pos = def_positions_list[idx]
                ref_center = ref_centers_list[idx]
                def_center = def_centers_list[idx]

                # Center positions
                ref_pos_centered = ref_pos - ref_center
                def_pos_centered = def_pos - def_center

                # Store in batch array (padding with zeros if needed)
                ref_pos_batch[j, :len(ref_pos_centered)] = ref_pos_centered
                def_pos_batch[j, :len(def_pos_centered)] = def_pos_centered

            tasks.append((indices, ref_pos_batch, def_pos_batch, count))

        # Process in parallel
        with Pool(processes=min(cpu_count(), len(tasks))) as pool:
            batch_results = pool.map(process_batch_parallel, tasks)

        # Store results
        for batch_result in batch_results:
            for idx, shear, principal in batch_result:
                shear_strains[idx] = float(shear)
                principal_strains[idx] = principal

    else:
        # Sequential processing version
        for count, indices in count_to_indices.items():
            if count < 4 or len(indices) == 0:
                continue

            batch_size = len(indices)
            # For sequential processing, we don't need to create full batch arrays
            # since we're processing one at a time anyway
            for idx in indices:
                ref_pos = ref_positions_list[idx]
                def_pos = def_positions_list[idx]
                ref_center = ref_centers_list[idx]
                def_center = def_centers_list[idx]

                # Center positions
                A = ref_pos - ref_center
                B = def_pos - def_center

                try:
                    shear, principal = _compute_single_strain(A, B)
                    shear_strains[idx] = float(shear)
                    principal_strains[idx] = principal
                except Exception as e:
                    print(f"Error computing strain for atom {idx}: {str(e)}")
                    shear_strains[idx] = np.nan
                    principal_strains[idx] = np.nan

    return shear_strains, principal_strains
