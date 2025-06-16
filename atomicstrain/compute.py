# compute.py - Optimized version for maximum CPU performance
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

print("Using JAX on CPU with vectorized operations")
print(f"JAX devices: {jax.devices()}")

# Vectorized strain computation for multiple atoms at once
@jit
def compute_strain_batch(A_batch, B_batch):
    """
    Compute strain for multiple atoms in parallel.
    A_batch, B_batch: (n_atoms, max_neighbors, 3)
    Returns: shear_strains (n_atoms,), principal_strains (n_atoms, 3)
    """
    # Compute F = B^T @ pinv(A^T) for each atom
    def compute_single(A, B):
        F = B.T @ jnp.linalg.pinv(A.T, rcond=1e-6)
        C = F.T @ F
        E = 0.5 * (C - jnp.eye(3))
        E = 0.5 * (E + E.T)
        
        eigenvalues = jnp.linalg.eigvalsh(E)
        shear = jnp.trace(E @ E) - (1/3) * jnp.square(jnp.trace(E))
        
        return shear, jnp.sort(eigenvalues)[::-1]
    
    # Vectorize over the batch dimension
    compute_vectorized = vmap(compute_single)
    shears, principals = compute_vectorized(A_batch, B_batch)
    
    return shears, principals

# Pre-compile for common neighbor counts
print("Pre-compiling JAX functions...")
for n_neighbors in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
    dummy_A = jnp.ones((10, n_neighbors, 3), dtype=jnp.float32)
    dummy_B = jnp.ones((10, n_neighbors, 3), dtype=jnp.float32)
    try:
        _ = compute_strain_batch(dummy_A, dummy_B)
    except:
        pass

def process_frame_data(ref_positions_list, ref_centers_list, def_positions_list, def_centers_list):
    """Process all positions for a frame with maximum efficiency."""
    n_atoms = len(ref_positions_list)
    shear_strains = np.zeros(n_atoms, dtype=np.float32)
    principal_strains = np.zeros((n_atoms, 3), dtype=np.float32)
    
    # Group atoms by number of neighbors for batch processing
    atoms_by_neighbors = {}
    for i in range(n_atoms):
        n_neighbors = len(ref_positions_list[i])
        if n_neighbors < 4:
            shear_strains[i] = np.nan
            principal_strains[i, :] = np.nan
            continue
        
        if n_neighbors not in atoms_by_neighbors:
            atoms_by_neighbors[n_neighbors] = []
        atoms_by_neighbors[n_neighbors].append(i)
    
    # Process each group in batches
    for n_neighbors, atom_indices in atoms_by_neighbors.items():
        if not atom_indices:
            continue
        
        n_atoms_group = len(atom_indices)
        
        # Prepare batch arrays
        A_batch = np.zeros((n_atoms_group, n_neighbors, 3), dtype=np.float32)
        B_batch = np.zeros((n_atoms_group, n_neighbors, 3), dtype=np.float32)
        
        # Fill batch arrays
        for j, i in enumerate(atom_indices):
            A_batch[j] = ref_positions_list[i] - ref_centers_list[i]
            B_batch[j] = def_positions_list[i] - def_centers_list[i]
        
        try:
            # Convert to JAX and compute all at once
            A_jax = jnp.asarray(A_batch)
            B_jax = jnp.asarray(B_batch)
            
            # Compute strain for all atoms in this group
            shears, principals = compute_strain_batch(A_jax, B_jax)
            
            # Store results
            shears_np = np.asarray(shears)
            principals_np = np.asarray(principals)
            
            for j, i in enumerate(atom_indices):
                shear_strains[i] = shears_np[j]
                principal_strains[i] = principals_np[j]
                
        except Exception as e:
            # Fallback for any errors
            for i in atom_indices:
                shear_strains[i] = np.nan
                principal_strains[i, :] = np.nan
    
    return shear_strains, principal_strains