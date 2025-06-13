# compute.py
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.scipy.linalg import eigh
from jax import config
import numpy as np

# Check available devices and select appropriate platform
def get_available_platform():
    backends = [backend.platform for backend in jax.devices()]
    if 'gpu' in backends:
        return 'gpu'
    elif 'tpu' in backends:
        return 'tpu'
    else:
        return 'cpu'

# Configure JAX to use the available platform
platform = get_available_platform()
config.update('jax_platform_name', platform)
print(f"Using {platform.upper()} for computations")

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

def process_frame_data(ref_positions_list, ref_centers_list, def_positions_list, def_centers_list):
    """Process all positions for a frame with variable neighbor counts."""
    n_atoms = len(ref_positions_list)
    shear_strains = np.zeros(n_atoms, dtype=np.float32)
    principal_strains = np.zeros((n_atoms, 3), dtype=np.float32)

    # Process each atom individually
    for i in range(n_atoms):
        ref_pos = ref_positions_list[i]
        def_pos = def_positions_list[i]
        ref_center = ref_centers_list[i]
        def_center = def_centers_list[i]
        
        # Check if we have enough neighbors (at least 4 for 3D stability)
        if len(ref_pos) < 4:
            print(f"Warning: Atom {i} has only {len(ref_pos)} neighbors (need at least 4)")
            # Set to NaN to indicate problematic calculation
            shear_strains[i] = np.nan
            principal_strains[i] = np.nan
            continue
            
        # Center the positions
        A = ref_pos - ref_center
        B = def_pos - def_center
        
        try:
            # Convert to JAX arrays
            A_jax = jnp.array(A, dtype=jnp.float32)
            B_jax = jnp.array(B, dtype=jnp.float32)
            
            # Check condition number before proceeding
            AtA = A_jax.T @ A_jax
            cond_num = np.linalg.cond(AtA)
            
            if cond_num > 1e10:
                print(f"Warning: Atom {i} has poorly conditioned matrix (condition number: {cond_num:.2e})")
                shear_strains[i] = np.nan
                principal_strains[i] = np.nan
                continue
            
            # Compute strain
            shear, principal = _compute_single_strain(A_jax, B_jax)
            
            # Convert back to numpy and store
            shear_strains[i] = float(shear)
            principal_strains[i] = np.array(principal)
            
        except Exception as e:
            print(f"Error computing strain for atom {i}: {str(e)}")
            shear_strains[i] = np.nan
            principal_strains[i] = np.nan

    return shear_strains, principal_strains