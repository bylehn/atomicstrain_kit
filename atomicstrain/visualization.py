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
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot histograms for shear strain
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.hist(shear_strains, bins=30, edgecolor='black')
    ax1.set_title('Histogram of Shear Strain')
    ax1.set_xlabel('Shear Strain')
    ax1.set_ylabel('Frequency')
    
    # Log histogram for shear strain
    log_shear = np.log10(np.abs(shear_strains) + 1e-10)  # Add small value to avoid log(0)
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
        ax1.hist(principal_strains[:, i], bins=30, edgecolor='black')
        ax1.set_title(f'Histogram of Principal Strain {i+1}')
        ax1.set_xlabel(f'Principal Strain {i+1}')
        ax1.set_ylabel('Frequency')
        
        # Log histogram for principal strain
        log_principal = np.log10(np.abs(principal_strains[:, i]) + 1e-10)  # Add small value to avoid log(0)
        ax2.hist(log_principal, bins=30, edgecolor='black')
        ax2.set_title(f'Histogram of Log10 Principal Strain {i+1}')
        ax2.set_xlabel(f'Log10 Principal Strain {i+1}')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'principal_strain_{i+1}_histograms.png'))
        plt.close()

def plot_strain_line(residue_numbers, avg_shear_strains, avg_principal_strains, output_dir):
    """
    Plot a line graph of average strains vs residue number using a color-blind friendly palette.

    Args:
        residue_numbers (list): List of residue numbers.
        avg_shear_strains (np.ndarray): Array of average shear strains.
        avg_principal_strains (np.ndarray): Array of average principal strains.
        output_dir (str): Directory to save the output figure.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Color-blind friendly color palette
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73']
    
    # Line styles
    line_styles = ['-', '--', '-.', ':']

    plt.figure(figsize=(12, 8))
    plt.plot(residue_numbers, avg_shear_strains, label='Shear Strain', 
             color=colors[0], linestyle=line_styles[0], linewidth=2)
    
    for i in range(3):
        plt.plot(residue_numbers, avg_principal_strains[:, i], 
                 label=f'Principal Strain {i+1}', 
                 color=colors[i+1], linestyle=line_styles[i+1], linewidth=2)
    
    plt.title('Average Strains vs Residue Number')
    plt.xlabel('Residue Number')
    plt.ylabel('Strain')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'average_strains_line_plot.png'))
    plt.close()

def plot_strain_line_std(residue_numbers, shear_strains, principal_strains, output_dir):
    """
    Plot a line graph of average strains vs residue number with standard deviation bands,
    using a color-blind friendly palette.

    Args:
        residue_numbers (list): List of residue numbers.
        shear_strains (np.ndarray): Array of shear strains for all frames.
        principal_strains (np.ndarray): Array of principal strains for all frames.
        output_dir (str): Directory to save the output figure.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Color-blind friendly color palette
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73']
    
    # Line styles
    line_styles = ['-', '--', '-.', ':']

    plt.figure(figsize=(12, 8))

    # Calculate average and standard deviation for shear strain
    avg_shear_strains = np.mean(shear_strains, axis=0)
    std_shear_strains = np.std(shear_strains, axis=0)

    # Plot shear strain with std deviation band
    plt.plot(residue_numbers, avg_shear_strains, label='Shear Strain', 
             color=colors[0], linestyle=line_styles[0], linewidth=2)
    plt.fill_between(residue_numbers, 
                     avg_shear_strains - std_shear_strains,
                     avg_shear_strains + std_shear_strains,
                     color=colors[0], alpha=0.2)

    # Calculate average and standard deviation for principal strains
    avg_principal_strains = np.mean(principal_strains, axis=0)
    std_principal_strains = np.std(principal_strains, axis=0)

    # Plot principal strains with std deviation bands
    for i in range(3):
        plt.plot(residue_numbers, avg_principal_strains[:, i], 
                 label=f'Principal Strain {i+1}', 
                 color=colors[i+1], linestyle=line_styles[i+1], linewidth=2)
        plt.fill_between(residue_numbers, 
                         avg_principal_strains[:, i] - std_principal_strains[:, i],
                         avg_principal_strains[:, i] + std_principal_strains[:, i],
                         color=colors[i+1], alpha=0.2)
    
    plt.title('Average Strains vs Residue Number (with Standard Deviation)')
    plt.xlabel('Residue Number')
    plt.ylabel('Strain')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'average_strains_line_plot_with_std.png'))
    plt.close()

def visualize_strains(residue_numbers, shear_strains, principal_strains, output_dir):
    """
    Create and save all strain visualizations.

    Args:
        residue_numbers (list): List of residue numbers.
        shear_strains (np.ndarray): Array of shear strains.
        principal_strains (np.ndarray): Array of principal strains.
        output_dir (str): Directory to save the output figures.
    """
    # Plot histograms
    plot_strain_histograms(shear_strains.flatten(), principal_strains.reshape(-1, 3), output_dir)

    # Plot line graph with standard deviation
    plot_strain_line_std(residue_numbers, shear_strains, principal_strains, output_dir)

    print(f"Visualization figures have been saved in {output_dir}")

# Add more visualization functions as needed