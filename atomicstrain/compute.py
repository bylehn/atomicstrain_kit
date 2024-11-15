import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import eigh

@jit
def compute_strain_tensor(Am, Bm):
    """
    Compute the strain tensor for given reference and deformed configurations using masks.
    """
    Am = jnp.asarray(Am)
    Bm = jnp.asarray(Bm)
    
    # Compute tensors directly without boolean indexing
    D = jnp.linalg.pinv(Am.T @ Am)  # using pinv for better numerical stability
    C = Bm @ Bm.T - Am @ Am.T
    Q = 0.5 * (D @ Am.T @ C @ Am @ D)
    # Explicitly symmetrize the tensor
    Q = 0.5 * (Q + Q.T)
    return Q

@jit
def compute_principal_strains_and_shear(Q):
    """
    Compute principal strains and shear from the strain tensor.

    Args:
        Q (jnp.ndarray): Strain tensor.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - shear (jnp.ndarray): Computed shear strain.
            - eigenvalues (jnp.ndarray): Principal strains (eigenvalues of the strain tensor).
    """
    eigenvalues, _ = eigh(Q)
    shear = jnp.trace(Q @ Q) - (1/3) * (jnp.trace(Q))**2
    sorted_eigenvalues = jnp.sort(eigenvalues)[::-1]  # Sort in descending order
    return shear, sorted_eigenvalues