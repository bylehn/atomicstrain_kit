import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from .compute import compute_strain_tensor, compute_principal_strains_and_shear
from .utils import create_selections
from .io import write_strain_files, write_pdb_with_strains
from tqdm import tqdm
import os

# analysis.py
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
import os
from .compute import compute_strain_tensor, compute_principal_strains_and_shear
from .utils import create_selections
from .io import write_strain_files, write_pdb_with_strains

class StrainAnalysis(AnalysisBase):
    def __init__(self, reference, deformed, residue_numbers, output_dir, min_neighbors=3, n_frames=None, use_all_heavy=False, **kwargs):
        self.ref = reference
        self.defm = deformed
        self.residue_numbers = residue_numbers
        self.min_neighbors = min_neighbors
        self.use_all_heavy = use_all_heavy
        self.selections = create_selections(self.ref, self.defm, residue_numbers, min_neighbors, use_all_heavy)
        self.output_dir = output_dir
        self.has_ref_trajectory = hasattr(self.ref, 'trajectory') and len(self.ref.trajectory) > 1
        
        # Store n_frames but don't use it directly for array initialization
        self.requested_n_frames = n_frames
        
        super().__init__(self.defm.trajectory, **kwargs)

    def _prepare(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Determine the number of atoms we're analyzing
        n_atoms = len(self.selections)
        
        # Calculate actual number of frames that will be analyzed
        start = self.start if self.start is not None else 0
        stop = self.stop if self.stop is not None else len(self.defm.trajectory)
        step = self.step if self.step is not None else 1
        
        # Calculate actual number of frames based on start, stop, and stride
        actual_n_frames = len(range(start, stop, step))
        print(f"Preparing analysis for {actual_n_frames} frames")
        
        # Use the data subdirectory for memory-mapped files
        data_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create memory-mapped arrays with correct size
        self.results.shear_strains = np.memmap(
            f"{data_dir}/shear_strains.npy",
            dtype='float32',
            mode='w+',
            shape=(actual_n_frames, n_atoms)
        )
        
        self.results.principal_strains = np.memmap(
            f"{data_dir}/principal_strains.npy",
            dtype='float32',
            mode='w+',
            shape=(actual_n_frames, n_atoms, 3)
        )
        
        # Store atom info
        self.results.atom_info = [(ref_center.resid, ref_center.name) 
                                 for (_, ref_center), _ in self.selections]
        
        # Initialize frame counter
        self._frame_counter = 0

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

        # Use frame counter instead of frame index for array indexing
        self.results.shear_strains[self._frame_counter] = frame_shear
        self.results.principal_strains[self._frame_counter] = frame_principal
        self._frame_counter += 1

    def run(self, start=None, stop=None, stride=None, verbose=True):
        self.start = start
        self.stop = stop
        self.step = stride
        return super().run(start=start, stop=stop, step=stride, verbose=verbose)

    def _conclude(self):
        # Compute average strains using the actual number of frames analyzed
        self.results.avg_shear_strains = np.mean(self.results.shear_strains[:self._frame_counter], axis=0)
        self.results.avg_principal_strains = np.mean(self.results.principal_strains[:self._frame_counter], axis=0)

        # Save a copy of the strains before writing files
        self.results.final_shear_strains = np.array(self.results.shear_strains[:self._frame_counter])
        self.results.final_principal_strains = np.array(self.results.principal_strains[:self._frame_counter])

        write_strain_files(
            self.output_dir,
            self.results.shear_strains[:self._frame_counter],
            self.results.principal_strains[:self._frame_counter],
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