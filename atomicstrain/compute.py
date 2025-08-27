# compute.py - Highly optimized CPU version
import numpy as np
import numpy.linalg as npla
from scipy.linalg import inv
import scipy.linalg as spla
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from numba import njit, prange, jit
import warnings
import MDAnalysis as mda

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

def process_frame_data(ref_positions_list, ref_centers_list, def_positions_list, def_centers_list, 
                      weights_list=None, parallel=False, batch_size=100, 
                      compute_deformation_gradient=False, compute_strain_metrics=None):
    """
    Highly optimized version with multiple techniques for speed improvement.

    Args:
        ref_positions_list: List of reference positions for each atom
        ref_centers_list: List of reference centers for each atom
        def_positions_list: List of deformed positions for each atom
        def_centers_list: List of deformed centers for each atom
        weights_list: List of weight arrays for each atom (optional)
        parallel: If True, use multiprocessing (default: False)
        batch_size: Number of atoms to process in each batch (for parallel processing)
        compute_deformation_gradient: If True, compute deformation gradients
        compute_strain_metrics: Dict of strain metrics to compute 
                               {'euler': bool, 'lagrange': bool, 'invariants': bool, 
                                'stretches': bool, 'rotations': bool, 'energy': bool}
    """
    n_atoms = len(ref_positions_list)
    shear_strains = np.zeros(n_atoms, dtype=np.float32)
    principal_strains = np.zeros((n_atoms, 3), dtype=np.float32)
    
    # Initialize result containers for new metrics
    results = {
        'shear_strains': shear_strains,
        'principal_strains': principal_strains
    }
    
    if compute_deformation_gradient or compute_strain_metrics:
        results['deformation_gradients'] = np.zeros((n_atoms, 3, 3), dtype=np.float32)
        
    if compute_strain_metrics:
        if compute_strain_metrics.get('euler', False):
            results['euler_linear'] = np.zeros((n_atoms, 3, 3), dtype=np.float32)
            results['euler_nonlinear'] = np.zeros((n_atoms, 3, 3), dtype=np.float32)
        
        if compute_strain_metrics.get('lagrange', False):
            results['lagrange_linear'] = np.zeros((n_atoms, 3, 3), dtype=np.float32)
            results['lagrange_nonlinear'] = np.zeros((n_atoms, 3, 3), dtype=np.float32)
            
        if compute_strain_metrics.get('invariants', False):
            results['invariants'] = np.zeros((n_atoms, 3), dtype=np.float32)  # I1, I2, I3
            
        if compute_strain_metrics.get('stretches', False):
            results['principal_stretches'] = np.zeros((n_atoms, 3), dtype=np.float32)
            results['principal_axes'] = np.zeros((n_atoms, 3, 3), dtype=np.float32)
            
        if compute_strain_metrics.get('rotations', False):
            results['rotation_angles'] = np.zeros(n_atoms, dtype=np.float32)
            results['rotation_axes'] = np.zeros((n_atoms, 3), dtype=np.float32)
            
        if compute_strain_metrics.get('energy', False):
            results['elastic_energy'] = np.zeros(n_atoms, dtype=np.float32)

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
                
                # Compute standard strain metrics
                shear, principal = _compute_single_strain_numba(A, B)
                results['shear_strains'][atom_idx] = float(shear)
                results['principal_strains'][atom_idx] = principal
                
                # Compute deformation gradient and additional metrics if requested
                if compute_deformation_gradient or compute_strain_metrics:
                    # Get weights for this atom if provided
                    weights = weights_list[valid_idx] if weights_list is not None else np.ones(len(A))
                    
                    # Compute deformation gradient using simplified approach
                    try:
                        # Create weight dictionary for single atom
                        n_neighbors = len(A)
                        weight_dict = {}
                        for i in range(n_neighbors):
                            for j in range(n_neighbors):
                                if i != j:
                                    weight_dict[(i, j)] = weights[j] if len(weights) > j else 1.0
                        
                        # Compute deformation gradient for single atom
                        F_atom = compute_single_deformation_gradient(A, B, weight_dict)
                        results['deformation_gradients'][atom_idx] = F_atom
                        
                        # Compute additional strain metrics if requested
                        if compute_strain_metrics:
                            if compute_strain_metrics.get('euler', False):
                                euler_lin, euler_nlin = euler_strain([F_atom])
                                results['euler_linear'][atom_idx] = euler_lin[0]
                                results['euler_nonlinear'][atom_idx] = euler_nlin[0]
                                
                            if compute_strain_metrics.get('lagrange', False):
                                lag_lin, lag_nlin = lagrange_strain([F_atom])
                                results['lagrange_linear'][atom_idx] = lag_lin[0]
                                results['lagrange_nonlinear'][atom_idx] = lag_nlin[0]
                                
                                # Compute other metrics that depend on Lagrange strain
                                if compute_strain_metrics.get('invariants', False):
                                    I1, I2, I3 = invariants_from_g([lag_lin[0]])
                                    results['invariants'][atom_idx] = [I1[0], I2[0], I3[0]]
                                    
                                if compute_strain_metrics.get('stretches', False):
                                    stretches, axes = principal_stretches_from_g([lag_lin[0]])
                                    results['principal_stretches'][atom_idx] = stretches[0]
                                    results['principal_axes'][atom_idx] = axes[0]
                                    
                                if compute_strain_metrics.get('energy', False):
                                    energy = saint_venant_energy_density([lag_lin[0]])
                                    results['elastic_energy'][atom_idx] = energy[0]
                            
                            if compute_strain_metrics.get('rotations', False):
                                angles, axes = rotations([F_atom])
                                results['rotation_angles'][atom_idx] = angles[0]
                                results['rotation_axes'][atom_idx] = axes[0]
                                
                    except Exception as e:
                        print(f"Warning: Deformation gradient computation failed for atom {atom_idx}: {e}")
                        if 'deformation_gradients' in results:
                            results['deformation_gradients'][atom_idx] = np.nan

    # Return based on what was computed
    if compute_deformation_gradient or compute_strain_metrics:
        return results
    else:
        return results['shear_strains'], results['principal_strains']


def compute_single_deformation_gradient(A, B, weight_dict):
    """
    Compute deformation gradient for a single atom using centered positions.
    
    Args:
        A: Reference positions (N x 3) - already centered
        B: Deformed positions (N x 3) - already centered  
        weight_dict: Dictionary of weights for atom pairs
        
    Returns:
        F: 3x3 deformation gradient tensor
    """
    try:
        # Compute weighted D and A matrices
        D = np.zeros((3, 3))
        A_matrix = np.zeros((3, 3))
        
        n_atoms = len(A)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    # Get weight, default to 1.0 if not found
                    weight = weight_dict.get((i, j), 1.0)
                    
                    # Relative positions
                    dX = A[j] - A[i]  # Reference relative position
                    dx = B[j] - B[i]  # Deformed relative position
                    
                    # Accumulate matrices
                    D += weight * np.outer(dX, dX)
                    A_matrix += weight * np.outer(dx, dX)
        
        # Compute deformation gradient F = A * D^(-1)
        if np.linalg.det(D) > 1e-12:
            F = A_matrix @ np.linalg.inv(D)
        else:
            # Fallback to identity if D is singular
            F = np.eye(3)
            
        return F
        
    except Exception as e:
        print(f"Warning: Deformation gradient computation failed: {e}")
        return np.eye(3)


# Deformation gradient and strain computation functions

@jit(nopython=True)
def intermediate_matrixes(weights, xyz_rel, xyz_def):
    """
    Calculate the matrixes D and A, intermediate steps in the calculation of F in 
    [Gullet et al,] (Eq. X), or Eq. 17 in [Zimmerman et al, 2009].
    """
    num_res = xyz_rel.shape[0]
    D = np.zeros((num_res, 3, 3))
    A = np.zeros((num_res, 3, 3))

    dX = np.zeros((3, 3), dtype=np.float64)
    dx = np.zeros((3, 3), dtype=np.float64)

    for pos in weights:
        i, j = pos
        dX = xyz_rel[j, :] - xyz_rel[i, :]
        dx = xyz_def[j, :] - xyz_def[i, :]

        # Generate intermediate matrices
        D[i, :, :] += np.dot(dX.reshape((-1, 1)), dX.reshape((1, -1))) * weights[pos]
        A[i, :, :] += np.dot(dx.reshape((-1, 1)), dX.reshape((1, -1))) * weights[pos]

    return D, A


def deformation_gradient_fast(weights, xyz_rel, xyz_def):
    """
    Calculate the deformation gradient tensor for each (F=dx/dX) using the
    approach from [Gullet et al,] (Eq. X), or Eq. 17 in [Zimmerman et al, 2009]
    In solving, note that D.T*U=A.T -> U.T*D=A -> F=U.T.
    
    This implementation requires weights in the form of a dictionary, which
    is also generated by the fast versions of weight calculation.
    """
    D, A = intermediate_matrixes(weights, xyz_rel, xyz_def)

    num_res = D.shape[0]
    F = np.zeros((num_res, 3, 3))

    for i in range(num_res):
        F[i, :, :] = spla.solve(D[i, :, :].T, A[i, :, :].T).T

    return F


def rotations(F):
    """
    Calculate the rotation angles and axis from the deformation gradients. Note
    that, due to large deformations, we can have det F<0. This implies an im-
    proper rotation matrix, http://scipp.ucsc.edu/~haber/ph116A/rotation_11.pdf
    """
    rotation_angle = []
    rotation_axis = []

    for alpha in range(len(F)):
        # Polar decomposition and diagonalize
        Falpha = F[alpha]
        u, p = spla.polar(Falpha)

        # Obtain axis and angle, being mindful of whether u is improper
        if spla.det(u) > 0:
            axis = np.array([u[2, 1] - u[1, 2], u[2, 0] - u[0, 2], u[0, 1] - u[1, 0]])
            axis /= npla.norm(axis)
            angle = np.arccos((np.trace(u) - 1) / 2.0)
        else:
            axis = -np.array([u[2, 1] - u[1, 2], u[2, 0] - u[0, 2], u[0, 1] - u[1, 0]])
            axis /= npla.norm(axis)
            angle = np.arccos((np.trace(u) + 1) / 2.0)

        # Store
        rotation_axis.append(axis)
        rotation_angle.append(angle)

    return rotation_angle, rotation_axis


def euler_strain(F):
    """
    Calculate linear and non-linear euler strain from deformation gradients.
    """
    strain_linear = []
    strain_non_linear = []

    for alpha in range(len(F)):
        # Calculate displacement gradient
        Falpha = F[alpha]
        Finv = spla.inv(Falpha)
        dU = np.eye(3) - Finv

        # Calculate linear and non-linear strain
        eps_lin = (dU + dU.T) / 2.0
        eps_n_lin = (dU + dU.T - np.dot(dU.T, dU)) / 2.0

        # Append results
        strain_linear.append(eps_lin)
        strain_non_linear.append(eps_n_lin)

    return strain_linear, strain_non_linear


def lagrange_strain(F):
    """
    Calculate linear and non-linear lagrange strain from deformation gradients.
    """
    strain_linear = []
    strain_non_linear = []

    for alpha in range(len(F)):
        # Calculate displacement gradient
        Falpha = F[alpha]
        du = Falpha - np.eye(3)

        # Calculate linear and non-linear strain
        gam_lin = (du + du.T) / 2.0
        gam_n_lin = (du + du.T + np.dot(du.T, du)) / 2.0

        # Append results
        strain_linear.append(gam_lin)
        strain_non_linear.append(gam_n_lin)

    return strain_linear, strain_non_linear


def invariants_from_g(lagrange_strain):
    """
    Calculate the three usual invariants (I1, I2, I3). These are functions of
    the right/left Cauchy tensors. The right one is related to the lagrangian
    strain as C = 2 gam + I.
    """
    # Right Cauchy tensor
    C = 2 * np.array(lagrange_strain) + np.eye(3)

    # First invariant
    I1 = np.trace(C, axis1=1, axis2=2)

    # Second invariant
    I2_list = []
    for Ca in C:
        I2_list.append(-(np.trace(Ca ** 2) - np.trace(Ca) ** 2) / 2.0)
    I2 = np.array(I2_list)

    # Third invariant
    I3_list = []
    for Ca in C:
        I3_list.append(np.linalg.det(Ca))
    I3 = np.array(I3_list)

    return I1, I2, I3


def principal_stretches_from_g(lagrange_strain):
    """
    Calculate principal stretches and axis. These are the eigenvalues/vectors
    of the tensor U from F=RU. Since C = U^2 = 2 gam + I, we can calculate the
    stretches and axis from the lagrangian strain
    """
    # Right Cauchy tensor
    C = 2 * np.array(lagrange_strain) + np.eye(3)

    # Calculate for each atom
    stretches = []
    axis_1, axis_2, axis_3 = [], [], []
    for Ca in C:
        ls, vs = spla.eig(Ca)
        # Sort eigen-values
        stretches.append(np.sort(ls))

        # Sort eigen-vectors
        axis_1.append(vs[:, np.argsort(ls)[0]])
        axis_2.append(vs[:, np.argsort(ls)[1]])
        axis_3.append(vs[:, np.argsort(ls)[2]])

    # Convert to array
    stretches = np.array(stretches)
    axis_1 = np.array(axis_1)
    axis_2 = np.array(axis_2)
    axis_3 = np.array(axis_3)

    # Use a single hemisphere, as this carries no information
    neg_z_1 = np.argwhere(axis_1[:, 2] < 0)
    neg_z_2 = np.argwhere(axis_2[:, 2] < 0)
    neg_z_3 = np.argwhere(axis_3[:, 2] < 0)
    axis_1[neg_z_1, :] = -axis_1[neg_z_1, :]
    axis_2[neg_z_2, :] = -axis_2[neg_z_2, :]
    axis_3[neg_z_3, :] = -axis_3[neg_z_3, :]

    # Randomly distribute half in the other hemisphere, for consistency
    axis_size = np.shape(axis_3)[0]
    flip_1 = np.random.choice(range(axis_size), axis_size // 2, replace=False)
    flip_2 = np.random.choice(range(axis_size), axis_size // 2, replace=False)
    flip_3 = np.random.choice(range(axis_size), axis_size // 2, replace=False)
    axis_1[flip_1, :] = -axis_1[flip_1, :]
    axis_2[flip_2, :] = -axis_2[flip_2, :]
    axis_3[flip_3, :] = -axis_3[flip_3, :]

    return np.array(stretches), [axis_1, axis_2, axis_3]


def new_principal_stretches_from_g(lagrange_strain):
    """
    Calculate principal stretches and axis. These are the eigenvalues/vectors
    of the tensor U from F=RU. Since C = U^2 = 2 gam + I, we can calculate the
    stretches and axis from the lagrangian strain
    """
    # Right Cauchy tensor
    C = 2 * np.array(lagrange_strain) + np.eye(3)

    # Calculate for each atom
    stretches = []
    axis_1, axis_2, axis_3 = [], [], []
    for Ca in C:
        ls, vs = spla.eig(Ca)
        # Sort eigen-values
        stretches.append(np.sort(ls))

        # Sort eigen-vectors
        axis_1.append(vs[:, np.argsort(ls)[0]])
        axis_2.append(vs[:, np.argsort(ls)[1]])
        axis_3.append(vs[:, np.argsort(ls)[2]])

    axes = np.transpose(
        np.array([np.array(axis_1), np.array(axis_2), np.array(axis_3)]), (1, 0, 2)
    )

    return np.array(stretches), axes


# Energy functions

def saint_venant_energy_density(lagrange_strain, first_lame=7.30e7, second_lame=3.76e7):
    """
    Calculate the energy density for a hyperelastic material considering the
    Saint Venant-Kirchhoff model, psi = lambda/2 * tr(gam)^2 + mu * tr(gam^2)
    [Holzapfel, 2000], where lambda and mu are the first and the second Lamé's
    coefficients. The default values were calculated considering the Young's
    moduls Y = 100MPa and the Poisson's ratio nu = 1/3.
    """
    # Calculate stretches squared
    stretches2, _ = principal_stretches_from_g(lagrange_strain)
    stretches2 = np.abs(stretches2)
    
    # Calculate trace elements
    tr2_gam = np.sum((stretches2-1)/2, axis=1)**2
    tr_gam2 = np.sum(((stretches2-1)/2)**2, axis=1)
    
    psi = first_lame / 2 * tr2_gam + second_lame * tr_gam2
    
    return psi


def saint_venant_energy(energy_density, protein_volume, labels, 
                        units="kT", verbose=False):
    """
    Calculate the energy a hyperelastic material considering the
    Saint Venant-Kirchhoff model. It integrates the energy density over the
    protein volume considering the effective volume occupied by each atom.
    
    The output is returned in kT units as default, other options are
    'kJ/mol', 'kcal/mol', and 'J'.
    
    Note: This function requires the spatial module for effective_volume calculation.
    Import spatial module separately or implement effective_volume function.
    """
    try:
        # This would need to be imported from your spatial module
        # vf = spatial.effective_volume(labels, protein_volume, verbose)
        # For now, using a placeholder - you'll need to implement or import this
        vf = np.ones_like(energy_density)  # Placeholder
        print("Warning: Using placeholder for effective_volume. Import spatial module or implement this function.")
    except:
        vf = np.ones_like(energy_density)
        print("Warning: effective_volume not available. Using unit volumes.")
    
    E = energy_density * vf
    
    if units == "kT":
        kT = 1.380649e-23*300
        E = E/kT
    elif units == "kJ/mol":
        mol = 6.02214076e23
        E = E*mol/1e3
    elif units == "kcal/mol":
        kcal = 4.184e3
        mol = 6.02214076e23
        E = E*mol/kcal
    elif units != "J":
        print("Units of energy not recognized, returning energy in Joules (J)")
    
    return E
    

def elastic_energy(lagrange_strain, labels, protein_volume, 
                   units="kT", verbose=False,
                   first_lame=7.30e7, second_lame=3.76e7):
    """
    Condensated function to calculate the elastic energy, calling
    'saint_venant_energy_density' and 'saint_venant_energy'.
    """
    psi = saint_venant_energy_density(lagrange_strain, first_lame, second_lame)
    E = saint_venant_energy(psi, protein_volume, labels, units, verbose)
    
    return E

