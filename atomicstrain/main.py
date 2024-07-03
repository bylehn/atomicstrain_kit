from atomicstrain import StrainAnalysis
from atomicstrain.visualization import plot_shear_strains
from atomicstrain.data.files import REFERENCE_PDB, DEFORMED_PDB
import MDAnalysis as mda

# Set up your analysis
ref = mda.Universe(REFERENCE_PDB)
defm = mda.Universe(DEFORMED_PDB)  # Assuming you don't have a separate trajectory file
residue_numbers = list(range(307, 398))
output_dir = 'results'
min_neighbors = 3  # Using min_neighbors instead of R as per the updated StrainAnalysis

# Run the analysis
strain_analysis = StrainAnalysis(ref, defm, residue_numbers, output_dir, min_neighbors)
strain_analysis.run()

# Visualize results
plot_shear_strains(strain_analysis.results.shear_strains, residue_numbers)