# compute.py
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.scipy.linalg import eigh
from jax import config
import numpy as np

# Force JAX to use GPU
config.update('jax_platform_name', 'gpu')
# Use 32-bit precision for better performance
config.update("jax_enable_x64", False)

print("Available devices:", jax.devices())

# JIT compile core computations
@jit
def _compute_single_strain(A, B):
    """Core computation for a single strain tensor."""
    D = jnp.linalg.inv(A.T @ A)
    C = B @ B.T - A @ A.T
    Q = 0.5 * (D @ A.T @ C @ A @ D)
    Q = 0.5 * (Q + Q.T)  # Ensure symmetry
    
    # Compute strains in the same function to reduce overhead
    eigenvalues = jnp.linalg.eigvalsh(Q)
    shear = jnp.trace(Q @ Q) - (1/3) * jnp.square(jnp.trace(Q))
    
    return shear, jnp.sort(eigenvalues)[::-1]

@jit
def _compute_batch_strain(ref_pos_batch, def_pos_batch):
    """Vectorized computation for a batch of strain tensors."""
    D = jnp.linalg.inv(ref_pos_batch.transpose((0, 2, 1)) @ ref_pos_batch)
    C = def_pos_batch @ def_pos_batch.transpose((0, 2, 1)) - ref_pos_batch @ ref_pos_batch.transpose((0, 2, 1))
    Q = 0.5 * (D @ ref_pos_batch.transpose((0, 2, 1)) @ C @ ref_pos_batch @ D)
    Q = 0.5 * (Q + Q.transpose((0, 2, 1)))
    
    eigenvalues = vmap(jnp.linalg.eigvalsh)(Q)
    shear = jnp.trace(Q @ Q, axis1=1, axis2=2) - (1/3) * jnp.square(jnp.trace(Q, axis1=1, axis2=2))
    
    return shear, jnp.sort(eigenvalues, axis=1)[:, ::-1]

def process_frame_data(ref_positions, ref_centers, def_positions, def_centers):
    """Process all positions for a frame efficiently with batching."""
    n_atoms = len(ref_positions)
    shear_strains = np.zeros(n_atoms, dtype=np.float32)
    principal_strains = np.zeros((n_atoms, 3), dtype=np.float32)

    # Filter valid positions (length >= 3)
    valid_indices = [i for i in range(n_atoms) if len(ref_positions[i]) >= 3]
    
    if not valid_indices:
        return shear_strains, principal_strains

    # Prepare batch data
    ref_batch = jnp.array([ref_positions[i] - ref_centers[i] for i in valid_indices], dtype=jnp.float32)
    def_batch = jnp.array([def_positions[i] - def_centers[i] for i in valid_indices], dtype=jnp.float32)
    
    # Single GPU computation for entire batch
    shear_batch, principal_batch = _compute_batch_strain(ref_batch, def_batch)
    
    # Back to CPU and assign results
    shear_batch = np.asarray(shear_batch)
    principal_batch = np.asarray(principal_batch)
    
    for idx, orig_idx in enumerate(valid_indices):
        shear_strains[orig_idx] = shear_batch[idx]
        principal_strains[orig_idx] = principal_batch[idx]

    return shear_strains, principal_strains