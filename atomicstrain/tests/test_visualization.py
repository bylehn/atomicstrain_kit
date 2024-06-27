import pytest
import numpy as np
import matplotlib.pyplot as plt
from atomicstrain.visualization import plot_shear_strains

def test_plot_shear_strains():
    """Test the plot_shear_strains function."""
    shear_strains = np.random.rand(10, 5)  # 10 frames, 5 residues
    residue_numbers = list(range(1, 6))
    
    # Call the function
    plot_shear_strains(shear_strains, residue_numbers)
    
    # Check if a figure was created
    assert plt.gcf().number > 0
    
    # Check if the correct number of lines were plotted
    assert len(plt.gca().lines) == 5
    
    # Check if labels are correct
    assert plt.gca().get_xlabel() == 'Frame'
    assert plt.gca().get_ylabel() == 'Shear Strain'
    assert plt.gca().get_title() == 'Shear Strain over Time'
    
    # Clear the current figure
    plt.clf()

if __name__ == "__main__":
    pytest.main()