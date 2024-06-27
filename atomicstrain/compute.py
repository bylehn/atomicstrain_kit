from typing import List, Tuple
import MDAnalysis as mda
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import eigh

@jit
def compute_strain_tensor(Am: jnp.ndarray, Bm: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the strain tensor for given reference and deformed configurations.

    Args:
        Am (jnp.ndarray): Reference configuration matrix.
        Bm (jnp.ndarray): Deformed configuration matrix.

    Returns:
        jnp.ndarray: Computed strain tensor.
    """
    D = jnp.linalg.inv(Am.T @ Am)
    C = Bm @ Bm.T - Am @ Am.T
    Q = 0.5 * (D @ Am.T @ C @ Am @ D)
    return Q

@jit
def compute_principal_strains_and_shear(Q: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    return shear, eigenvalues

def process_frame(ref: mda.Universe, defm: mda.Universe, selections: List, frame: int) -> Tuple[List[float], List[List[float]]]:
    """
    Process a single frame to compute shear and principal strains.

    Args:
        ref (mda.Universe): MDAnalysis Universe object for the reference structure.
        defm (mda.Universe): MDAnalysis Universe object for the deformed structure.
        selections (List): List of atom selections for strain calculations.
        frame (int): Frame number to process.

    Returns:
        Tuple[List[float], List[List[float]]]: A tuple containing:
            - frame_shear (List[float]): List of shear strain values for each selection.
            - frame_principal (List[List[float]]): List of principal strain values for each selection.
    """
    ref.trajectory[frame]
    defm.trajectory[frame]

    frame_shear = []
    frame_principal = []

    for ((ref_sel, ref_center), (defm_sel, defm_center)) in selections:
        A = ref_sel.positions - ref_center.positions[0]
        B = defm_sel.positions - defm_center.positions[0]

        A = np.array(A)
        B = np.array(B)

        Q = compute_strain_tensor(A, B)
        shear, principal = compute_principal_strains_and_shear(Q)
        frame_shear.append(float(shear))  # Convert to Python float
        frame_principal.append(principal.tolist())  # Convert to Python list

    return frame_shear, frame_principal