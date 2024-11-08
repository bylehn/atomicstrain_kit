import os
import MDAnalysis as mda
import numpy as np

def write_strain_files(output_dir, shear_strains, principal_strains, avg_shear_strains, avg_principal_strains, atom_info, use_all_heavy):
    """
    Write strain data to files, handling memory-mapped arrays efficiently and including all strain files.
    
    Args:
        output_dir (str): Directory to write the output files.
        shear_strains (np.ndarray): Array/memmap of shear strains.
        principal_strains (np.ndarray): Array/memmap of principal strains.
        avg_shear_strains (np.ndarray): Array of average shear strains.
        avg_principal_strains (np.ndarray): Array of average principal strains.
        atom_info (list): List of tuples containing residue number and atom name.
        use_all_heavy (bool): Whether to use all heavy atoms or only CA atoms.
    """
    # Create data subdirectory if it doesn't exist
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 1. Write average shear strains
    print("Writing average shear strains...")
    avg_shear_strains_file = os.path.join(data_dir, 'avg_shear_strains.txt')
    with open(avg_shear_strains_file, 'w') as f_shear:
        f_shear.write("# Average shear strains for each residue\n")
        f_shear.write("# Format: Residue number, Atom name, Average shear strain\n")
        for i, (avg_shear, (resid, atom_name)) in enumerate(zip(avg_shear_strains, atom_info)):
            f_shear.write(f'Residue {resid}, Atom {atom_name}: {avg_shear:.4f}\n')
    print(f"Wrote {avg_shear_strains_file}")

    # 2. Write average principal strains for each component
    print("Writing average principal strains...")
    for component in range(3):  
        avg_principal_file = os.path.join(data_dir, f'avg_principal_{component+1}.txt')
        with open(avg_principal_file, 'w') as f_principal:
            f_principal.write(f"# Average principal strains (component {component+1}) for each residue\n")
            f_principal.write("# Format: Residue number, Atom name, Average principal strain\n")
            for i, (principal, (resid, atom_name)) in enumerate(zip(avg_principal_strains, atom_info)):
                f_principal.write(f'Residue {resid}, Atom {atom_name}: {principal[component]:.4f}\n')
        print(f"Wrote {avg_principal_file}")

    # 3. Write raw strain data in chunks
    chunk_size = 1000  # Adjust based on memory constraints
    n_chunks = (len(shear_strains) + chunk_size - 1) // chunk_size

    # Write raw shear strains
    print("Writing raw shear strains...")
    raw_shear_strains_file = os.path.join(data_dir, 'raw_shear_strains.txt')
    with open(raw_shear_strains_file, 'w') as f_raw_shear:
        # Write header
        header = ['Frame'] + [f'Res{resid}_{atom_name}' for resid, atom_name in atom_info]
        f_raw_shear.write("# Raw shear strains for each frame and residue\n")
        f_raw_shear.write("# Columns: Frame number, followed by shear strain for each residue\n")
        f_raw_shear.write('\t'.join(header) + '\n')
        
        # Write data in chunks
        for chunk in range(n_chunks):
            start_idx = chunk * chunk_size
            end_idx = min((chunk + 1) * chunk_size, len(shear_strains))
            
            for frame_idx in range(start_idx, end_idx):
                frame_data = shear_strains[frame_idx]
                line = [str(frame_idx)] + [f'{value:.4f}' for value in frame_data]
                f_raw_shear.write('\t'.join(line) + '\n')
    print(f"Wrote {raw_shear_strains_file}")

    # Write raw principal strains
    print("Writing raw principal strains...")
    for component in range(3):
        raw_principal_file = os.path.join(data_dir, f'raw_principal_{component+1}.txt')
        with open(raw_principal_file, 'w') as f_raw_principal:
            # Write header
            header = ['Frame'] + [f'Res{resid}_{atom_name}' for resid, atom_name in atom_info]
            f_raw_principal.write(f"# Raw principal strains (component {component+1}) for each frame and residue\n")
            f_raw_principal.write("# Columns: Frame number, followed by principal strain for each residue\n")
            f_raw_principal.write('\t'.join(header) + '\n')
            
            # Write data in chunks
            for chunk in range(n_chunks):
                start_idx = chunk * chunk_size
                end_idx = min((chunk + 1) * chunk_size, len(principal_strains))
                
                for frame_idx in range(start_idx, end_idx):
                    frame_data = principal_strains[frame_idx]
                    line = [str(frame_idx)] + [f'{value[component]:.4f}' for value in frame_data]
                    f_raw_principal.write('\t'.join(line) + '\n')
        print(f"Wrote {raw_principal_file}")

    print(f"\nAll strain files written to {data_dir}")
    print("\nFiles created:")
    print(f"  - avg_shear_strains.txt")
    print(f"  - avg_principal_1.txt")
    print(f"  - avg_principal_2.txt")
    print(f"  - avg_principal_3.txt")
    print(f"  - raw_shear_strains.txt")
    print(f"  - raw_principal_1.txt")
    print(f"  - raw_principal_2.txt")
    print(f"  - raw_principal_3.txt")

def write_pdb_with_strains(deformed_pdb, output_dir, residue_numbers, avg_shear_strains, avg_principal_strains, atom_info, use_all_heavy):
    """
    Write PDB files with strain data in organized subdirectories.
    """
    # Create structures subdirectory
    structures_dir = os.path.join(output_dir, 'structures')
    os.makedirs(structures_dir, exist_ok=True)
    
    u = mda.Universe(deformed_pdb)
    pdb_filename = os.path.join(structures_dir, 'strains.pdb')

    with mda.Writer(pdb_filename, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
        # Frame 1: Average Shear Strains
        for atom, (resid, atom_name), shear_strain in zip(u.atoms, atom_info, avg_shear_strains):
            if atom.resid in residue_numbers and (not use_all_heavy or atom.name != 'H'):
                atom.tempfactor = 100 * shear_strain
        PDB.write(u.atoms)

        # Next 3 Frames: Principal Strains
        for component in range(3):
            for atom, (resid, atom_name), principal_strain in zip(u.atoms, atom_info, avg_principal_strains):
                if atom.resid in residue_numbers and (not use_all_heavy or atom.name != 'H'):
                    atom.tempfactor = 100 * principal_strain[component]
            PDB.write(u.atoms)