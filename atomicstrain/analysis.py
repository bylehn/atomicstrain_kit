import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from .compute import compute_strain_tensor, compute_principal_strains_and_shear
from .utils import create_selections
from .io import write_strain_files, write_pdb_with_strains

class StrainAnalysis(AnalysisBase):
    """
    Analyze strain in molecular dynamics trajectories.

    This class computes shear and principal strains for specified residues
    over the course of a molecular dynamics trajectory.

    Attributes:
        ref (MDAnalysis.Universe): Reference structure Universe.
        defm (MDAnalysis.Universe): Deformed structure Universe.
        residue_numbers (list): List of residue numbers to analyze.
        R (float): Radius for atom selection.
        selections (list): List of atom selections for analysis.
        output_dir (str): Directory to write output files.

    """

    def __init__(self, reference, deformed, residue_numbers, output_dir, min_neighbors=3, **kwargs):
        """
        Initialize the StrainAnalysis.

        Args:
            reference (MDAnalysis.Universe): Reference structure Universe.
            deformed (MDAnalysis.Universe): Deformed structure Universe.
            residue_numbers (list): List of residue numbers to analyze.
            R (float): Radius for atom selection.
            output_dir (str): Directory to write output files.
            **kwargs: Additional keyword arguments for AnalysisBase.
        """
        self.ref = reference
        self.defm = deformed
        self.residue_numbers = residue_numbers
        self.min_neighbors = min_neighbors
        self.selections = create_selections(self.ref, self.defm, residue_numbers, min_neighbors)
        self.output_dir = output_dir
        super().__init__(self.defm.trajectory, **kwargs)

    def _prepare(self):
        """
        Prepare for analysis by initializing results containers.

        This method is called before iteration on the trajectory begins.
        """
        self.results.shear_strains = []
        self.results.principal_strains = []

    def _single_frame(self):
        """
        Analyze a single frame of the trajectory.

        This method is called for each frame in the trajectory.
        It computes shear and principal strains for the current frame.
        """
        frame_shear = []
        frame_principal = []

        for ((ref_sel, ref_center), (defm_sel, defm_center)) in self.selections:
            A = ref_sel.positions - ref_center.positions[0]
            B = defm_sel.positions - defm_center.positions[0]
            
            # Ensure A and B have the same shape
            if A.shape != B.shape:
                print(f"Warning: Shapes don't match for resid {ref_center.resids[0]}. Skipping.")
                continue

            Q = compute_strain_tensor(A, B)
            shear, principal = compute_principal_strains_and_shear(Q)
            frame_shear.append(float(shear))
            frame_principal.append(principal.tolist())

        self.results.shear_strains.append(frame_shear)
        self.results.principal_strains.append(frame_principal)

    def _conclude(self):
        """
        Conclude the analysis by processing the collected data.

        This method is called after iteration on the trajectory is finished.
        It computes average strains, converts results to numpy arrays,
        and writes the results to files.
        """
        self.results.shear_strains = np.array(self.results.shear_strains)
        self.results.principal_strains = np.array(self.results.principal_strains)
        self.results.avg_shear_strains = np.mean(self.results.shear_strains, axis=0)
        self.results.avg_principal_strains = np.mean(self.results.principal_strains, axis=0)

        # Write strain data to files
        write_strain_files(
            self.output_dir,
            self.results.shear_strains,
            self.results.principal_strains,
            self.results.avg_shear_strains,
            self.results.avg_principal_strains
        )

        # Write PDB file with strains
        write_pdb_with_strains(
            self.defm.filename,
            self.output_dir,
            self.residue_numbers,
            self.results.avg_shear_strains,
            self.results.avg_principal_strains
        )