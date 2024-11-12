import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from .compute import compute_strain_tensor, compute_principal_strains_and_shear
from .utils import create_selections
from .io import write_strain_files, write_pdb_with_strains
from tqdm import tqdm
import os

class StrainAnalysis(AnalysisBase):
    def __init__(self, reference, deformed, residue_numbers, output_dir, min_neighbors=3, n_frames=None, use_all_heavy=False, **kwargs):
        # Initialize base class first
        super().__init__(deformed.trajectory, n_frames=n_frames, **kwargs)
        
        # Initialize results namespace early
        self._results = {'shear_strains': None,
                        'principal_strains': None,
                        'atom_info': None,
                        'avg_shear_strains': None,
                        'avg_principal_strains': None}
        
        self.ref = reference
        self.defm = deformed
        self.residue_numbers = residue_numbers
        self.min_neighbors = min_neighbors
        self.use_all_heavy = use_all_heavy
        self.selections = create_selections(self.ref, self.defm, residue_numbers, min_neighbors, use_all_heavy)
        self.output_dir = output_dir
        self.n_frames = n_frames
        self.has_ref_trajectory = hasattr(self.ref, 'trajectory') and len(self.ref.trajectory) > 1
        
        # Initialize atom_info early
        self._results['atom_info'] = [(ref_center.resid, ref_center.name) 
                                     for (_, ref_center), _ in self.selections]
        
        # Track memmap files for cleanup
        self._memmap_files = []

    def _prepare(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        n_atoms = len(self.selections)
        
        if self.n_frames is not None:
            self.actual_n_frames = self.n_frames
        else:
            self.actual_n_frames = len(range(0, len(self.defm.trajectory), self.stride or 1))
            
        # Create memmap files with proper paths
        shear_path = os.path.join(self.output_dir, 'shear_strains.npy')
        principal_path = os.path.join(self.output_dir, 'principal_strains.npy')
        
        try:
            self._results['shear_strains'] = np.memmap(
                shear_path,
                dtype='float32',
                mode='w+',
                shape=(self.actual_n_frames, n_atoms)
            )
            self._memmap_files.append(self._results['shear_strains'])
            
            self._results['principal_strains'] = np.memmap(
                principal_path,
                dtype='float32',
                mode='w+',  
                shape=(self.actual_n_frames, n_atoms, 3)
            )
            self._memmap_files.append(self._results['principal_strains'])
            
        except Exception as e:
            self._cleanup_memmaps()
            raise RuntimeError(f"Failed to create memmap arrays: {str(e)}")
            
    def _cleanup_memmaps(self):
        """Clean up memmap files"""
        for mmap in self._memmap_files:
            if mmap is not None:
                mmap._mmap.close()
        self._memmap_files = []

    def __del__(self):
        """Ensure memmap cleanup on deletion"""
        self._cleanup_memmaps()

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

        self._results['shear_strains'][self._frame_index] = frame_shear
        self._results['principal_strains'][self._frame_index] = frame_principal

    def run(self, start=None, stop=None, stride=None, verbose=True):
        self.stride = stride
        self._prepare()
        
        if self.n_frames is not None:
            stop = min(self.n_frames * (stride or 1) + (start or 0), len(self.defm.trajectory))
        
        # Determine the frames to analyze
        frames = range(start or 0, stop or len(self.defm.trajectory), stride or 1)
        
        # Use tqdm for a progress bar if verbose
        iterator = tqdm(frames, desc="Analyzing frames", disable=not verbose)
        
        for frame_idx, frame in enumerate(iterator):
            self._frame_index = frame_idx  # Use frame_idx instead of frame
            self.defm.trajectory[frame]
            self._single_frame()
        
        self._conclude()
        return self

    def _conclude(self):
        # Compute average strains
        self._results['avg_shear_strains'] = np.mean(self._results['shear_strains'], axis=0)
        self._results['avg_principal_strains'] = np.mean(self._results['principal_strains'], axis=0)

        write_strain_files(
            self.output_dir,
            self._results['shear_strains'],
            self._results['principal_strains'],
            self._results['avg_shear_strains'],
            self._results['avg_principal_strains'],
            self._results['atom_info'],
            self.use_all_heavy
        )

        write_pdb_with_strains(
            self.defm.filename,
            self.output_dir,
            self.residue_numbers,
            self._results['avg_shear_strains'],
            self._results['avg_principal_strains'],
            self._results['atom_info'],
            self.use_all_heavy
        )

        # Clean up memory-mapped arrays
        del self._results['shear_strains']
        del self._results['principal_strains']