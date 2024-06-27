import matplotlib.pyplot as plt

def plot_shear_strains(shear_strains, residue_numbers):
    """
    Plot shear strains over time for each residue.

    Args:
        shear_strains (np.ndarray): Array of shear strains with shape (n_frames, n_residues).
        residue_numbers (list): List of residue numbers corresponding to the shear strains.

    Returns:
        None: This function displays the plot directly.
    """
    plt.figure(figsize=(10, 6))
    for i in range(len(residue_numbers)):
        plt.plot(shear_strains[:, i], label=f'Residue {residue_numbers[i]}')
    plt.xlabel('Frame')
    plt.ylabel('Shear Strain')
    plt.title('Shear Strain over Time')
    plt.legend()

# Add more visualization functions as needed