import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import time
from collections import defaultdict

# Force CPU usage for JAX (GPU can be slower for small problems)
jax.config.update('jax_platform_name', 'cpu')
# Disable JAX JIT overhead tracking for production
jax.config.update('jax_disable_jit', False)

# Global profiling stats
compute_timings = defaultdict(float)
profiling_enabled = False

def enable_profiling():
    global profiling_enabled
    profiling_enabled = True

def disable_profiling():
    global profiling_enabled
    profiling_enabled = False

@jit
def compute_F_jax(ref_positions, ref_center, def_positions, def_center):
    """Compute deformation gradient using JAX."""
    # Compute displacement vectors
    ref_vectors = ref_positions - ref_center
    def_vectors = def_positions - def_center
    
    # Build moment matrix A = sum(w_i * u_i * u_i^T)
    A = jnp.sum(ref_vectors[:, :, None] * ref_vectors[:, None, :], axis=0)
    
    # Compute A_inv
    A_inv = jnp.linalg.inv(A)
    
    # Compute B = sum(w_i * v_i * u_i^T)
    B = jnp.sum(def_vectors[:, :, None] * ref_vectors[:, None, :], axis=0)
    
    # F = B @ A_inv
    F = B @ A_inv
    
    return F

@jit
def compute_strain_from_F_jax(F):
    """Compute strain metrics from deformation gradient using JAX."""
    # Right Cauchy-Green deformation tensor
    C = F.T @ F
    
    # Green-Lagrange strain tensor
    I = jnp.eye(3)
    E = 0.5 * (C - I)
    
    # Von Mises (shear) strain
    E_dev = E - (1/3) * jnp.trace(E) * I
    shear_strain = jnp.sqrt(2 * jnp.sum(E_dev * E_dev))
    
    # Principal strains (eigenvalues of E)
    eigenvalues = jnp.linalg.eigvalsh(E)
    principal_strains = jnp.sort(eigenvalues)[::-1]
    
    return shear_strain, principal_strains

# Vectorized versions for batch processing
compute_F_batch = vmap(compute_F_jax, in_axes=(0, 0, 0, 0))
compute_strain_batch = vmap(compute_strain_from_F_jax, in_axes=(0,))

# Create a combined function that processes everything in one JAX call
@jit
def compute_strains_batch_jax(ref_positions_batch, ref_centers_batch, 
                              def_positions_batch, def_centers_batch):
    """Compute all strains in a single JAX call."""
    # Compute all F matrices
    F_batch = compute_F_batch(ref_positions_batch, ref_centers_batch,
                              def_positions_batch, def_centers_batch)
    
    # Compute all strains
    shear_strains, principal_strains = compute_strain_batch(F_batch)
    
    return shear_strains, principal_strains

def process_frame_data(ref_positions_list, ref_centers_list,
                      def_positions_list, def_centers_list):
    """
    Process strain data for all atoms in a frame with optimized JAX usage.
    
    This version minimizes JAX conversion overhead by processing all atoms
    in a single batch operation.
    """
    n_atoms = len(ref_positions_list)
    
    if profiling_enabled:
        t0 = time.time()
    
    # Find maximum number of neighbors across all atoms
    max_neighbors = max(len(positions) for positions in ref_positions_list)
    
    # Prepare padded arrays for batch processing
    ref_positions_padded = np.zeros((n_atoms, max_neighbors, 3), dtype=np.float32)
    ref_centers_array = np.array(ref_centers_list, dtype=np.float32)
    def_positions_padded = np.zeros((n_atoms, max_neighbors, 3), dtype=np.float32)
    def_centers_array = np.array(def_centers_list, dtype=np.float32)
    
    # Track which atoms have valid data
    valid_atoms = []
    
    # Fill padded arrays
    for i in range(n_atoms):
        n_neighbors = len(ref_positions_list[i])
        if n_neighbors >= 4:  # Need at least 4 neighbors for 3D strain
            ref_positions_padded[i, :n_neighbors] = ref_positions_list[i]
            def_positions_padded[i, :n_neighbors] = def_positions_list[i]
            valid_atoms.append(i)
    
    if profiling_enabled:
        compute_timings['batch_prep'] += time.time() - t0
        t1 = time.time()
    
    # Process all valid atoms in a single JAX call
    if valid_atoms:
        # Convert to JAX arrays only for valid atoms
        valid_indices = np.array(valid_atoms)
        ref_pos_jax = jnp.array(ref_positions_padded[valid_indices])
        ref_cen_jax = jnp.array(ref_centers_array[valid_indices])
        def_pos_jax = jnp.array(def_positions_padded[valid_indices])
        def_cen_jax = jnp.array(def_centers_array[valid_indices])
        
        if profiling_enabled:
            compute_timings['jax_convert'] += time.time() - t1
            t2 = time.time()
        
        # Compute all strains in one go
        shear_strains_jax, principal_strains_jax = compute_strains_batch_jax(
            ref_pos_jax, ref_cen_jax, def_pos_jax, def_cen_jax
        )
        
        if profiling_enabled:
            compute_timings['jax_compute'] += time.time() - t2
            t3 = time.time()
        
        # Convert back to numpy
        shear_strains_valid = np.array(shear_strains_jax)
        principal_strains_valid = np.array(principal_strains_jax)
        
        if profiling_enabled:
            compute_timings['result_convert'] += time.time() - t3
    
    # Prepare final results
    shear_strains = np.zeros(n_atoms, dtype=np.float32)
    principal_strains = np.zeros((n_atoms, 3), dtype=np.float32)
    
    # Fill in valid results
    if valid_atoms:
        shear_strains[valid_indices] = shear_strains_valid
        principal_strains[valid_indices] = principal_strains_valid
    
    return shear_strains, principal_strains

# Alternative: Pure NumPy implementation for comparison
def compute_F_numpy(ref_positions, ref_center, def_positions, def_center):
    """Compute deformation gradient using NumPy (no JAX)."""
    # Compute displacement vectors
    ref_vectors = ref_positions - ref_center
    def_vectors = def_positions - def_center
    
    # Build moment matrix A = sum(w_i * u_i * u_i^T)
    A = np.sum(ref_vectors[:, :, None] * ref_vectors[:, None, :], axis=0)
    
    # Compute A_inv
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    
    # Compute B = sum(w_i * v_i * u_i^T)
    B = np.sum(def_vectors[:, :, None] * ref_vectors[:, None, :], axis=0)
    
    # F = B @ A_inv
    F = B @ A_inv
    
    return F

def compute_strain_from_F_numpy(F):
    """Compute strain metrics from deformation gradient using NumPy."""
    # Right Cauchy-Green deformation tensor
    C = F.T @ F
    
    # Green-Lagrange strain tensor
    I = np.eye(3)
    E = 0.5 * (C - I)
    
    # Von Mises (shear) strain
    E_dev = E - (1/3) * np.trace(E) * I
    shear_strain = np.sqrt(2 * np.sum(E_dev * E_dev))
    
    # Principal strains (eigenvalues of E)
    eigenvalues = np.linalg.eigvalsh(E)
    principal_strains = np.sort(eigenvalues)[::-1]
    
    return shear_strain, principal_strains

def process_frame_data_numpy(ref_positions_list, ref_centers_list,
                            def_positions_list, def_centers_list):
    """
    Pure NumPy implementation without JAX overhead.
    Use this for small batches where JAX overhead dominates.
    """
    n_atoms = len(ref_positions_list)
    shear_strains = np.zeros(n_atoms, dtype=np.float32)
    principal_strains = np.zeros((n_atoms, 3), dtype=np.float32)
    
    for i in range(n_atoms):
        if len(ref_positions_list[i]) < 4:
            continue
            
        F = compute_F_numpy(
            ref_positions_list[i],
            ref_centers_list[i],
            def_positions_list[i],
            def_centers_list[i]
        )
        
        if F is not None:
            shear, principal = compute_strain_from_F_numpy(F)
            shear_strains[i] = shear
            principal_strains[i] = principal
    
    return shear_strains, principal_strains

# Profiling utilities
def print_compute_profile():
    """Print profiling statistics for compute functions."""
    if not profiling_enabled or not compute_timings:
        return
        
    total = sum(compute_timings.values())
    if total == 0:
        return
        
    print("\n=== Compute Function Profiling ===")
    for key, value in sorted(compute_timings.items()):
        pct = (value / total) * 100
        print(f"{key:15s}: {pct:5.1f}%")

def reset_compute_profile():
    """Reset profiling statistics."""
    global compute_timings
    compute_timings = defaultdict(float)