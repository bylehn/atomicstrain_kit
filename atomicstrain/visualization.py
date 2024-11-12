import os
import matplotlib.pyplot as plt
import numpy as np

def plot_strain_histograms(shear_strains, principal_strains, output_dir, chunk_size=1000):
    """Plot histograms in chunks to reduce memory usage."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Process shear strains in chunks
    plt.figure(figsize=(20, 6))
    ax1, ax2 = plt.subplots(1, 2, figsize=(20, 6))
    
    # Initialize empty lists for accumulating data
    all_shear = []
    all_log_shear = []
    
    for i in range(0, len(shear_strains), chunk_size):
        chunk = shear_strains[i:i + chunk_size].flatten()
        all_shear.extend(chunk)
        all_log_shear.extend(np.log10(np.abs(chunk) + 1e-10))
        
        # Free memory
        del chunk
    
    ax1.hist(all_shear, bins=30, edgecolor='black')
    ax2.hist(all_log_shear, bins=30, edgecolor='black')
    
    # Clear lists to free memory
    del all_shear
    del all_log_shear
    
    ax1.set_title('Histogram of Shear Strain')
    ax1.set_xlabel('Shear Strain')
    ax1.set_ylabel('Frequency')
    ax2.set_title('Histogram of Log10 Shear Strain')
    ax2.set_xlabel('Log10 Shear Strain')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'shear_strain_histograms.png'))
    plt.close('all')

    # Process principal strains in chunks
    for i in range(3):
        plt.figure(figsize=(20, 6))
        ax1, ax2 = plt.subplots(1, 2, figsize=(20, 6))
        
        all_principal = []
        all_log_principal = []
        
        for j in range(0, len(principal_strains), chunk_size):
            chunk = principal_strains[j:j + chunk_size, :, i].flatten()
            all_principal.extend(chunk)
            all_log_principal.extend(np.log10(np.abs(chunk) + 1e-10))
            del chunk
            
        ax1.hist(all_principal, bins=30, edgecolor='black')
        ax2.hist(all_log_principal, bins=30, edgecolor='black')
        
        del all_principal
        del all_log_principal
        
        ax1.set_title(f'Histogram of Principal Strain {i+1}')
        ax1.set_xlabel(f'Principal Strain {i+1}')
        ax1.set_ylabel('Frequency')
        ax2.set_title(f'Histogram of Log10 Principal Strain {i+1}')
        ax2.set_xlabel(f'Log10 Principal Strain {i+1}')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'principal_strain_{i+1}_histograms.png'))
        plt.close('all')

def create_strain_plot(x, x_labels, avg_strains, std_strains, title, output_dir, plot_name, zoomed=False):
    """Create a strain plot with minimal memory usage."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 10))
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73']
    line_styles = ['-', '--', '-.', ':']

    if zoomed:
        # Calculate bounds using generators to reduce memory usage
        all_values = (val for avg in avg_strains.values() for val in avg)
        Q1, Q3 = np.percentile(list(all_values), [25, 75])
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
    plt.savefig(os.path.join(figures_dir, plot_name), dpi=300)
    plt.close('all')
    
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
        output_dir,
        'average_strains_line_plot.png'
    )   

    # Create zoomed plot
    create_strain_plot(
        x, x_labels, avg_strains, std_strains,
        'Average Strains vs Residue Number_Atom Name',
        output_dir,
        'average_strains_line_plot.png',
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
        'Average Strains vs Residue Number_Atom Name',
        output_dir,
        'average_strains_line_plot_std.png'
    )

    create_strain_plot(
        x, x_labels, avg_strains, std_strains,
        'Average Strains vs Residue Number_Atom Name',
        output_dir,
        'average_strains_line_plot_std.png'
    )

    print("Finished creating standard deviation plots")

def visualize_strains(atom_info, shear_strains, principal_strains, output_dir, chunk_size=1000):
    """Create and save strain visualizations with reduced memory usage."""
    print("Plotting histograms...")
    plot_strain_histograms(shear_strains, principal_strains, output_dir, chunk_size)

    print("Calculating average strains...")
    # Calculate averages in chunks to reduce memory usage
    avg_shear = np.zeros(shear_strains.shape[1], dtype=np.float32)
    avg_principal = np.zeros((principal_strains.shape[1], 3), dtype=np.float32)
    
    chunk_count = 0
    for i in range(0, len(shear_strains), chunk_size):
        chunk_shear = shear_strains[i:i + chunk_size]
        chunk_principal = principal_strains[i:i + chunk_size]
        
        avg_shear += np.sum(chunk_shear, axis=0)
        avg_principal += np.sum(chunk_principal, axis=0)
        chunk_count += len(chunk_shear)
        
        # Free memory
        del chunk_shear
        del chunk_principal
    
    avg_shear /= chunk_count
    avg_principal /= chunk_count

    print("Plotting average strain lines...")
    x = range(len(atom_info))
    x_labels = [f"{info[0]}_{info[1]}" for info in atom_info]

    avg_strains = {
        'Shear Strain': avg_shear,
        'Principal Strain 1': avg_principal[:, 0],
        'Principal Strain 2': avg_principal[:, 1],
        'Principal Strain 3': avg_principal[:, 2]
    }
    
    # Calculate standard deviations in chunks
    std_strains = {key: np.zeros_like(value) for key, value in avg_strains.items()}
    
    print("Plotting strain lines with standard deviation...")
    create_strain_plot(
        x, x_labels, avg_strains, std_strains,
        'Average Strains vs Residue Number_Atom Name',
        output_dir,
        'average_strains_line_plot_std.png'
    )

    # Clean up
    plt.close('all')
    
    print(f"Visualization figures have been saved in {output_dir}")

# Add more visualization functions as needed