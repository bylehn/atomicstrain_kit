from atomicstrain import StrainAnalysis
from atomicstrain.visualization import plot_shear_strains
import MDAnalysis as mda

# Set up your analysis
ref = mda.Universe("reference.pdb")
defm = mda.Universe("deformed.pdb", "trajectory.xtc")
residue_numbers = list(range(307, 398))
R = 10

# Run the analysis
strain_analysis = StrainAnalysis(ref, defm, residue_numbers, R)
strain_analysis.run()

# Visualize results
plot_shear_strains(strain_analysis.results.shear_strains, residue_numbers)