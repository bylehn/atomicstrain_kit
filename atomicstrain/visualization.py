import os
import matplotlib.pyplot as plt
import numpy as np

def plot_strain_histograms(shear_strains, principal_strains, output_dir):
    """
    Plot histograms for shear strain and principal strains, including log histograms.

    Args:
        shear_strains (np.ndarray): Array of shear strains.
        principal_strains (np.ndarray): Array of principal strains.
        output_dir (str): Directory to save the output figures.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot histograms for shear strain
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.hist(shear_strains.flatten(), bins=30, edgecolor='black')
    ax1.set_title('Histogram of Shear Strain')
    ax1.set_xlabel('Shear Strain')
    ax1.set_ylabel('Frequency')
    
    # Log histogram for shear strain
    log_shear = np.log10(np.abs(shear_strains.flatten()) + 1e-10)  # Add small value to avoid log(0)
    ax2.hist(log_shear, bins=30, edgecolor='black')
    ax2.set_title('Histogram of Log10 Shear Strain')
    ax2.set_xlabel('Log10 Shear Strain')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shear_strain_histograms.png'))
    plt.close()

    # Plot histograms for principal strains
    for i in range(3):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        ax1.hist(principal_strains[:,:,i].flatten(), bins=30, edgecolor='black')
        ax1.set_title(f'Histogram of Principal Strain {i+1}')
        ax1.set_xlabel(f'Principal Strain {i+1}')
        ax1.set_ylabel('Frequency')
        
        # Log histogram for principal strain
        log_principal = np.log10(np.abs(principal_strains[:,:,i].flatten()) + 1e-10)  # Add small value to avoid log(0)
        ax2.hist(log_principal, bins=30, edgecolor='black')
        ax2.set_title(f'Histogram of Log10 Principal Strain {i+1}')
        ax2.set_xlabel(f'Log10 Principal Strain {i+1}')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'principal_strain_{i+1}_histograms.png'))
        plt.close()

def create_strain_plot(x, x_labels, avg_strains, std_strains, title, output_path, zoomed=False):
    """
    Create a strain plot with or without zooming.

    Args:
        x (list): X-axis values.
        x_labels (list): Labels for x-axis ticks.
        avg_strains (dict): Dictionary of average strains for each strain type.
        std_strains (dict): Dictionary of standard deviations for each strain type.
        title (str): Plot title.
        output_path (str): Path to save the plot.
        zoomed (bool): Whether to create a zoomed plot.
    """
    plt.figure(figsize=(20, 10))
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73']
    line_styles = ['-', '--', '-.', ':']

    if zoomed:
        all_data = np.concatenate([avg for avg in avg_strains.values()])
        Q1, Q3 = np.percentile(all_data, [25, 75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    for i, (strain_type, avg) in enumerate(avg_strains.items()):
        if zoomed:
            avg = np.clip(avg, lower_bound, upper_bound)
            std = np.clip(std_strains[strain_type], 0, (upper_bound - lower_bound) / 2)
        else:
            std = std_strains[strain_type]

        plt.plot(x, avg, label=strain_type, color=colors[i], linestyle=line_styles[i], linewidth=2)
        plt.fill_between(x, avg - std, avg + std, color=colors[i], alpha=0.2)

    plt.title(title)
    plt.xlabel('Residue Number_Atom Name')
    plt.ylabel('Strain')
    plt.legend()

    tick_spacing = max(1, len(x) // 20)
    plt.xticks(x[::tick_spacing], x_labels[::tick_spacing], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_strain_line(atom_info, avg_shear_strains, avg_principal_strains, output_dir):
    """
    Plot a line graph of average strains vs residue number and atom name,
    using a color-blind friendly palette.

    Args:
        atom_info (list): List of tuples containing (residue_number, atom_name) for each atom.
        avg_shear_strains (np.ndarray): Array of average shear strains.
        avg_principal_strains (np.ndarray): Array of average principal strains.
        output_dir (str): Directory to save the output figure.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x = range(len(atom_info))
    x_labels = [f"{info[0]}_{info[1]}" for info in atom_info]

    avg_strains = {
        'Shear Strain': avg_shear_strains,
        'Principal Strain 1': avg_principal_strains[:, 0],
        'Principal Strain 2': avg_principal_strains[:, 1],
        'Principal Strain 3': avg_principal_strains[:, 2]
    }
    std_strains = {key: np.zeros_like(value) for key, value in avg_strains.items()}

    # Create regular plot
    create_strain_plot(
        x, x_labels, avg_strains, std_strains,
        'Average Strains vs Residue Number_Atom Name',
        os.path.join(output_dir, 'average_strains_line_plot.png')
    )

    # Create zoomed plot
    create_strain_plot(
        x, x_labels, avg_strains, std_strains,
        'Average Strains vs Residue Number_Atom Name (Zoomed)',
        os.path.join(output_dir, 'average_strains_line_plot_zoomed.png'),
        zoomed=True
    )

def plot_strain_line_std(atom_info, shear_strains, principal_strains, output_dir):
    """
    Plot line graphs of average strains vs residue number and atom name
    with standard deviation bands, using a color-blind friendly palette.

    Args:
        atom_info (list): List of tuples containing (residue_number, atom_name) for each atom.
        shear_strains (np.ndarray): Array of shear strains for all frames.
        principal_strains (np.ndarray): Array of principal strains for all frames.
        output_dir (str): Directory to save the output figures.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x = range(len(atom_info))
    x_labels = [f"{info[0]}_{info[1]}" for info in atom_info]

    avg_shear = np.mean(shear_strains, axis=0)
    std_shear = np.std(shear_strains, axis=0)
    avg_principal = np.mean(principal_strains, axis=0)
    std_principal = np.std(principal_strains, axis=0)

    avg_strains = {
        'Shear Strain': avg_shear,
        'Principal Strain 1': avg_principal[:, 0],
        'Principal Strain 2': avg_principal[:, 1],
        'Principal Strain 3': avg_principal[:, 2]
    }
    std_strains = {
        'Shear Strain': std_shear,
        'Principal Strain 1': std_principal[:, 0],
        'Principal Strain 2': std_principal[:, 1],
        'Principal Strain 3': std_principal[:, 2]
    }

    # Create regular plot
    create_strain_plot(
        x, x_labels, avg_strains, std_strains,
        'Average Strains vs Residue Number_Atom Name (with Standard Deviation)',
        os.path.join(output_dir, 'average_strains_line_plot_std.png')
    )

    # Create zoomed plot
    create_strain_plot(
        x, x_labels, avg_strains, std_strains,
        'Average Strains vs Residue Number_Atom Name (Zoomed, with Standard Deviation)',
        os.path.join(output_dir, 'average_strains_line_plot_std_zoomed.png'),
        zoomed=True
    )

    print("Finished creating standard deviation plots")

def visualize_strains(atom_info, shear_strains, principal_strains, output_dir):
    """
    Create and save all strain visualizations.

    Args:
        atom_info (list): List of tuples containing (residue_number, atom_name) for each atom.
        shear_strains (np.ndarray): Array of shear strains.
        principal_strains (np.ndarray): Array of principal strains.
        output_dir (str): Directory to save the output figures.
    """
    print("Starting visualization process...")

    # Plot histograms
    print("Plotting histograms...")
    plot_strain_histograms(shear_strains, principal_strains, output_dir)

    # Calculate average strains
    print("Calculating average strains...")
    avg_shear_strains = np.mean(shear_strains, axis=0)
    avg_principal_strains = np.mean(principal_strains, axis=0)

    # Plot line graphs
    print("Plotting average strain lines...")
    plot_strain_line(atom_info, avg_shear_strains, avg_principal_strains, output_dir)
    
    print("Plotting strain lines with standard deviation...")
    print(f"Shape of shear_strains: {shear_strains.shape}")
    print(f"Shape of principal_strains: {principal_strains.shape}")
    print(f"Number of atoms: {len(atom_info)}")
    plot_strain_line_std(atom_info, shear_strains, principal_strains, output_dir)

    print(f"Visualization figures have been saved in {output_dir}")

# Add more visualization functions as needed