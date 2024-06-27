import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import eigh

@jit
def compute_strain_tensor(Am, Bm):
    D = jnp.linalg.inv(Am.T @ Am)
    C = Bm @ Bm.T - Am @ Am.T
    Q = 0.5 * (D @ Am.T @ C @ Am @ D)
    return Q

@jit
def compute_principal_strains_and_shear(Q):
    eigenvalues, _ = eigh(Q)
    shear = jnp.trace(Q @ Q) - (1/3) * (jnp.trace(Q))**2
    return shear, eigenvalues