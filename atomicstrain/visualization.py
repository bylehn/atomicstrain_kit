import os
import matplotlib.pyplot as plt
import numpy as np

def plot_strain_histograms(shear_strains, principal_strains, output_dir):
    """
    Plot histograms for shear strain and principal strains.

    Args:
        shear_strains (np.ndarray): Array of shear strains.
        principal_strains (np.ndarray): Array of principal strains.
        output_dir (str): Directory to save the output figures.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot histogram for shear strain
    plt.figure(figsize=(10, 6))
    plt.hist(shear_strains, bins=30, edgecolor='black')
    plt.title('Histogram of Shear Strain')
    plt.xlabel('Shear Strain')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'shear_strain_histogram.png'))
    plt.close()

    # Plot histograms for principal strains
    for i in range(3):
        plt.figure(figsize=(10, 6))
        plt.hist(principal_strains[:, i], bins=30, edgecolor='black')
        plt.title(f'Histogram of Principal Strain {i+1}')
        plt.xlabel(f'Principal Strain {i+1}')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'principal_strain_{i+1}_histogram.png'))
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