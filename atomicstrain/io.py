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

    # Create a mapping of residue numbers to their index in the residue_numbers list
    residue_index_map = {resid: i for i, resid in enumerate(residue_numbers)}

    # Check if element information is available
    has_elements = hasattr(u.atoms[0], 'element')

    # Function to write a single frame
    def write_frame(strain_values):
        with open(pdb_filename, 'a') as f:
            f.write("MODEL\n")
            for atom in u.atoms:
                if atom.resid in residue_index_map and atom.name == 'CA':
                    strain_value = strain_values[residue_index_map[atom.resid]]
                else:
                    strain_value = 0.0
                
                # Format the PDB line manually
                line = f"ATOM  {atom.id:5d} {atom.name:<4s} {atom.resname:<3s} {atom.segment.segid:1s}{atom.resid:4d}    "
                line += f"{atom.position[0]:8.3f}{atom.position[1]:8.3f}{atom.position[2]:8.3f}"
                line += f"{1.00:6.2f}{strain_value:6.2f}"
                
                # Add element if available, otherwise pad with spaces
                if has_elements:
                    line += f"          {atom.element:>2s}"
                else:
                    line += "              "
                
                line += "\n"
                f.write(line)
            f.write("ENDMDL\n")

    # Write the file
    with open(pdb_filename, 'w') as f:
        f.write("TITLE     Strain Analysis Results\n")
        f.write("REMARK    Frame 1: Average Shear Strains\n")
        f.write("REMARK    Frames 2-4: Principal Strains\n")

    # Frame 1: Average Shear Strains
    write_frame(avg_shear_strains * 100)

    # Next 3 Frames: Principal Strains
    for component in range(3):
        write_frame(avg_principal_strains[:, component] * 100)

    print(f"PDB file with strain data has been written to {pdb_filename}")