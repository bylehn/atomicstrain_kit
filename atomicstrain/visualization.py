import matplotlib.pyplot as plt

def plot_shear_strains(shear_strains, residue_numbers):
    plt.figure(figsize=(10, 6))
    for i in range(len(residue_numbers)):
        plt.plot(shear_strains[:, i], label=f'Residue {residue_numbers[i]}')
    plt.xlabel('Frame')
    plt.ylabel('Shear Strain')
    plt.title('Shear Strain over Time')
    plt.legend()
    plt.show()

# Add more visualization functions as needed