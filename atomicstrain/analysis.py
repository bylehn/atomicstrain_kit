import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from .compute import compute_strain_tensor, compute_principal_strains_and_shear
from .utils import create_selections
from .io import write_strain_files, write_pdb_with_strains
from tqdm import tqdm

class StrainAnalysis(AnalysisBase):
    def __init__(self, reference, deformed, residue_numbers, output_dir, min_neighbors=3, n_frames=None, use_all_heavy=False, **kwargs):
        self.ref = reference
        self.defm = deformed
        self.residue_numbers = residue_numbers
        self.min_neighbors = min_neighbors
        self.use_all_heavy = use_all_heavy
        self.selections = create_selections(self.ref, self.defm, residue_numbers, min_neighbors, use_all_heavy)
        self.output_dir = output_dir
        self.n_frames = n_frames
        self.has_ref_trajectory = hasattr(self.ref, 'trajectory') and len(self.ref.trajectory) > 1
        super().__init__(self.defm.trajectory, n_frames=n_frames, **kwargs)

    def _prepare(self):
        # Determine the number of atoms we're analyzing
        n_atoms = len(self.selections)
        
        # Create memory-mapped arrays for results
        self.results.shear_strains = np.memmap(f"{self.output_dir}/shear_strains.npy", dtype='float32', mode='w+', shape=(self.n_frames, n_atoms))
        self.results.principal_strains = np.memmap(f"{self.output_dir}/principal_strains.npy", dtype='float32', mode='w+', shape=(self.n_frames, n_atoms, 3))
        
        # Store atom info
        self.results.atom_info = [(ref_center.resid, ref_center.name) for (_, ref_center), _ in self.selections]


    def _single_frame(self):
        frame_shear = np.zeros(len(self.selections), dtype='float32')
        frame_principal = np.zeros((len(self.selections), 3), dtype='float32')

        # Update reference frame only if it has a trajectory
        if self.has_ref_trajectory:
            self.ref.trajectory[self._frame_index]

        for i, ((ref_sel, ref_center), (defm_sel, defm_center)) in enumerate(self.selections):
            A = ref_sel.positions - ref_center.position
            B = defm_sel.positions - defm_center.position
            
            if A.shape != B.shape:
                print(f"Warning: Shapes don't match for atom {ref_center.index}. Skipping.")
                continue

            Q = compute_strain_tensor(A, B)
            shear, principal = compute_principal_strains_and_shear(Q)
            frame_shear[i] = float(shear)
            frame_principal[i] = principal

        self.results.shear_strains[self._frame_index] = frame_shear
        self.results.principal_strains[self._frame_index] = frame_principal

    def run(self, start=None, stop=None, stride=None, verbose=True):
        self._prepare()
        
        if self.n_frames is not None:
            stop = min(self.n_frames * (stride or 1) + (start or 0), len(self.defm.trajectory))
        
        # Determine the frames to analyze
        frames = range(start or 0, stop or len(self.defm.trajectory), stride or 1)
        
        # Use tqdm for a progress bar if verbose
        iterator = tqdm(frames, desc="Analyzing frames", disable=not verbose)
        
        for frame in iterator:
            self._frame_index = frame
            self.defm.trajectory[frame]
            self._single_frame()
        
        self._conclude()
        return self

    def _conclude(self):
        # Compute average strains
        self.results.avg_shear_strains = np.mean(self.results.shear_strains, axis=0)
        self.results.avg_principal_strains = np.mean(self.results.principal_strains, axis=0)

        write_strain_files(
            self.output_dir,
            self.results.shear_strains,
            self.results.principal_strains,
            self.results.avg_shear_strains,
            self.results.avg_principal_strains,
            self.results.atom_info,
            self.use_all_heavy
        )

        write_pdb_with_strains(
            self.defm.filename,
            self.output_dir,
            self.residue_numbers,
            self.results.avg_shear_strains,
            self.results.avg_principal_strains,
            self.results.atom_info,
            self.use_all_heavy
        )

        # Clean up memory-mapped arrays
        del self.results.shear_strains
        del self.results.principal_strains