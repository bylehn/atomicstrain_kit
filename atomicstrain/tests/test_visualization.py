import pytest
import numpy as np
import matplotlib.pyplot as plt
from atomicstrain.visualization import plot_shear_strains

def test_plot_shear_strains():
    """Test the plot_shear_strains function."""
    shear_strains = np.random.rand(10, 5)  # 10 frames, 5 residues
    residue_numbers = list(range(1, 6))
    
    plot_shear_strains(shear_strains, residue_numbers)
    
    fig = plt.gcf()
    fig.canvas.draw()
    
    assert len(plt.gca().lines) == 5
    plt.close(fig)  # Close the figure to free up memory