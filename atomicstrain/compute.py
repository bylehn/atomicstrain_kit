# compute.py
import jax.numpy as jnp
from jax import jit, vmap, device_put
from jax.scipy.linalg import eigh
from jax import config
import numpy as np
import jax

# Configure JAX
DEVICES = jax.devices()
print(f"JAX devices available: {DEVICES}")

# Try to use GPU if available
try:
    if any(d.platform == 'gpu' for d in DEVICES):
        config.update('jax_platform_name', 'gpu')
        print("Using GPU for computations")
    else:
        config.update('jax_platform_name', 'cpu')
        print("No GPU found, using CPU")
except:
    config.update('jax_platform_name', 'cpu')
    print("Failed to set GPU, using CPU")

@jit
def compute_strain_tensor(Am, Bm):
    """Compute strain tensor using JAX."""
    D = jnp.linalg.inv(Am.T @ Am)
    C = Bm @ Bm.T - Am @ Am.T
    Q = 0.5 * (D @ Am.T @ C @ Am @ D)
    return 0.5 * (Q + Q.T)

@jit
def compute_principal_strains_and_shear(Q):
    """Compute principal strains and shear using JAX."""
    eigenvalues, _ = eigh(Q)
    shear = jnp.trace(Q @ Q) - (1/3) * (jnp.trace(Q))**2
    return shear, jnp.sort(eigenvalues)[::-1]

def pad_positions(positions, max_length):
    """Pad position array to fixed length with zeros."""
    padded = np.zeros((max_length, 3), dtype=np.float32)
    actual_length = min(len(positions), max_length)
    padded[:actual_length] = positions[:actual_length]
    return padded

def process_frame_data(ref_positions, ref_centers, def_positions, def_centers):
    """
    Process all positions for a frame efficiently using JAX vectorization.
    """
    n_atoms = len(ref_positions)
    
    # Find maximum number of neighbors
    max_neighbors = max(len(pos) for pos in ref_positions if len(pos) >= 3)
    
    # Initialize output arrays
    shear_strains = np.zeros(n_atoms, dtype=np.float32)
    principal_strains = np.zeros((n_atoms, 3), dtype=np.float32)
    
    # Process valid atoms (those with enough neighbors)
    valid_indices = []
    valid_ref_data = []
    valid_def_data = []
    
    for i in range(n_atoms):
        if len(ref_positions[i]) >= 3:
            # Center and pad the positions
            ref_centered = ref_positions[i] - ref_centers[i]
            def_centered = def_positions[i] - def_centers[i]
            
            ref_padded = pad_positions(ref_centered, max_neighbors)
            def_padded = pad_positions(def_centered, max_neighbors)
            
            valid_indices.append(i)
            valid_ref_data.append(ref_padded)
            valid_def_data.append(def_padded)
    
    if not valid_indices:
        return shear_strains, principal_strains
    
    # Convert to JAX arrays
    ref_array = device_put(jnp.array(valid_ref_data))
    def_array = device_put(jnp.array(valid_def_data))
    
    try:
        # Vectorized computation for all valid atoms
        Q_tensors = vmap(compute_strain_tensor)(ref_array, def_array)
        shear_vals, principal_vals = vmap(compute_principal_strains_and_shear)(Q_tensors)
        
        # Move results back to CPU and store in output arrays
        shear_strains[valid_indices] = np.array(shear_vals)
        principal_strains[valid_indices] = np.array(principal_vals)
        
    except Exception as e:
        print(f"Error in strain computation: {str(e)}")
        # Return zeros if computation fails
        return shear_strains, principal_strains
    
    return shear_strains, principal_strains