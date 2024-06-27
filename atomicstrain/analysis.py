import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from .compute import compute_strain_tensor, compute_principal_strains_and_shear
from .utils import create_selections

class StrainAnalysis(AnalysisBase):
    def __init__(self, reference, deformed, residue_numbers, protein_ca, R, **kwargs):
        self.ref = reference
        self.defm = deformed
        self.residue_numbers = residue_numbers
        self.protein_ca = protein_ca
        self.R = R
        self.selections = create_selections(self.ref, self.defm, residue_numbers, protein_ca, R)
        super().__init__(self.defm.trajectory, **kwargs)

    def _prepare(self):
        self.results.shear_strains = []
        self.results.principal_strains = []

    def _single_frame(self):
        frame_shear = []
        frame_principal = []

        for ((ref_sel, ref_center), (defm_sel, defm_center)) in self.selections:
            A = ref_sel.positions - ref_center.positions[0]
            B = defm_sel.positions - defm_center.positions[0]

            Q = compute_strain_tensor(A, B)
            shear, principal = compute_principal_strains_and_shear(Q)
            frame_shear.append(float(shear))
            frame_principal.append(principal.tolist())

        self.results.shear_strains.append(frame_shear)
        self.results.principal_strains.append(frame_principal)

    def _conclude(self):
        self.results.shear_strains = np.array(self.results.shear_strains)
        self.results.principal_strains = np.array(self.results.principal_strains)
        self.results.avg_shear_strains = np.mean(self.results.shear_strains, axis=0)
        self.results.avg_principal_strains = np.mean(self.results.principal_strains, axis=0)