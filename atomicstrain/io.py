import os
import MDAnalysis as mda
import numpy as np

def write_strain_files(output_dir, shear_strains, principal_strains, avg_shear_strains, avg_principal_strains):
    """
    Write strain data to files.

    Args:
        output_dir (str): Directory to write the output files.
        shear_strains (np.ndarray): Array of shear strains.
        principal_strains (np.ndarray): Array of principal strains.
        avg_shear_strains (np.ndarray): Array of average shear strains.
        avg_principal_strains (np.ndarray): Array of average principal strains.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write average shear strains to file
    avg_shear_strains_file = os.path.join(output_dir, 'avg_shear_strains.txt')
    with open(avg_shear_strains_file, 'w') as f_shear:
        for i, avg_shear in enumerate(avg_shear_strains):
            f_shear.write(f'Residue {i+1}: {avg_shear:.4f}\n')

    # Write average principal strains for each component
    for component in range(3):  # Assuming 3 principal components
        avg_principal_file = os.path.join(output_dir, f'avg_principal_{component+1}.txt')
        with open(avg_principal_file, 'w') as f_principal:
            for i, principal in enumerate(avg_principal_strains):
                f_principal.write(f'Residue {i+1}: {principal[component]:.4f}\n')

    # Write raw shear strains
    raw_shear_strains_file = os.path.join(output_dir, 'raw_shear_strains.txt')
    with open(raw_shear_strains_file, 'w') as f_raw_shear:
        for frame in shear_strains:
            f_raw_shear.write(' '.join(map(lambda x: f'{x:.4f}', frame)) + '\n')

    # Write raw principal strains for each component
    for component in range(3):
        raw_principal_file = os.path.join(output_dir, f'raw_principal_{component+1}.txt')
        with open(raw_principal_file, 'w') as f_raw_principal:
            for frame in principal_strains:
                f_raw_principal.write(' '.join(map(lambda x: f'{x[component]:.4f}', frame)) + '\n')

def write_pdb_with_strains(deformed_pdb, output_dir, residue_numbers, avg_shear_strains, avg_principal_strains):
    """
    Write a PDB file with strain data in the B-factor column.

    Args:
        deformed_pdb (str): Path to the deformed PDB file.
        output_dir (str): Directory to write the output PDB file.
        residue_numbers (list): List of residue numbers to process.
        avg_shear_strains (np.ndarray): Array of average shear strains.
        avg_principal_strains (np.ndarray): Array of average principal strains.
    """
    # Load the deformed PDB structure
    u = mda.Universe(deformed_pdb)
    residue_selection_string = "resid " + " ".join(map(str, residue_numbers))
    selected_residues = u.select_atoms(residue_selection_string)
    pdb_filename = os.path.join(output_dir, 'strains.pdb')

    # Initialize a PDB writer for multiple frames
    with mda.Writer(pdb_filename, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
        # Frame 1: Average Shear Strains
        for residue in u.residues:
            if residue.resid in selected_residues.resids:
                ca_atom = residue.atoms.select_atoms('name CA')
                if ca_atom:  # Check if CA atom exists
                    ca_atom.tempfactors = 100 * avg_shear_strains[residue.resid - residue_numbers[0]]
        PDB.write(u.atoms)  # Write first frame

        # Next 3 Frames: Principal Strains
        for component in range(3):  # Assuming 3 principal components
            for residue in u.residues:
                if residue.resid in selected_residues.resids:
                    ca_atom = residue.atoms.select_atoms('name CA')
                    if ca_atom:  # Check if CA atom exists
                        # Assign principal strain component to B-factor
                        ca_atom.tempfactors = 100 * avg_principal_strains[residue.resid - residue_numbers[0], component]
            PDB.write(u.atoms)  # Write each principal component frame