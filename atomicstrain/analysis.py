import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from .compute import compute_strain_tensor, compute_principal_strains_and_shear
from .utils import create_selections
from .io import write_strain_files, write_pdb_with_strains
from tqdm import tqdm
import os
from functools import partial
from jax import jit, vmap, device_put, devices, local_device_count, pmap, device_put_sharded
import jax.numpy as jnp

@jit
def process_frame_batch(ref_sel_positions, ref_center_positions, defm_sel_positions, defm_center_positions, masks):
    """Single JIT-compiled function for entire batch processing"""
    def process_atom(ref_sel, ref_center, defm_sel, defm_center, mask):
        A = (ref_sel - ref_center) * mask[:, None]
        B = (defm_sel - defm_center) * mask[:, None]
        Q = compute_strain_tensor(A, B)
        return compute_principal_strains_and_shear(Q)
    
    # Process all atoms in all frames at once
    return vmap(vmap(process_atom))(
        ref_sel_positions,
        ref_center_positions,
        defm_sel_positions,
        defm_center_positions,
        masks
    )

class StrainAnalysis(AnalysisBase):
    def __init__(self, ref, defm, residue_numbers, output_dir, min_neighbors=3, 
                 n_frames=None, use_all_heavy=False, batch_size=512):
        """Initialize strain analysis.
        
        Args:
            ref: Reference Universe
            defm: Deformed Universe
            residue_numbers: List of residue numbers to analyze
            output_dir: Directory for output files
            min_neighbors: Minimum number of neighbors required
            n_frames: Number of frames to analyze (default: all)
            use_all_heavy: Whether to use all heavy atoms (default: False)
            batch_size: Size of batches for GPU processing (default: 512)
        """
        self.n_frames = len(defm.universe.trajectory) if n_frames is None else n_frames
        super().__init__(defm.universe, verbose=True)
        
        # Get available devices
        self.devices = devices("gpu") if devices("gpu") else devices("cpu")
        self.num_devices = len(self.devices)
        self.batch_size = batch_size
        
        # Store other instance variables
        self.ref = ref
        self.defm = defm
        self.output_dir = output_dir
        self.min_neighbors = min_neighbors
        self.use_all_heavy = use_all_heavy
        
        self.selections = create_selections(ref, defm, residue_numbers, 
                                         min_neighbors=min_neighbors, 
                                         use_all_heavy=use_all_heavy)

        # Pre-compile function with sample data
        self._warmup_jit()

    def _warmup_jit(self):
        """Pre-compile JIT functions with dummy data"""
        sample_batch = self._prepare_sample_batch()
        _ = process_frame_batch(*sample_batch)

    def _prepare_sample_batch(self):
        """Create small sample batch for JIT warmup"""
        shapes = self._get_batch_shapes(1)
        return tuple(jnp.zeros(shape) for shape in shapes)

    def _get_batch_shapes(self, batch_size):
        """Get shapes for batch arrays"""
        n_atoms = len(self.selections)
        max_neighbors = max(len(ref_sel) for (ref_sel, _), _ in self.selections)
        return (
            (batch_size, n_atoms, max_neighbors, 3),  # ref_sel
            (batch_size, n_atoms, 3),                 # ref_center
            (batch_size, n_atoms, max_neighbors, 3),  # defm_sel
            (batch_size, n_atoms, 3),                 # defm_center
            (batch_size, n_atoms, max_neighbors),     # masks
        )

    def _init_device_buffers(self):
        """Pre-allocate device buffers for batch processing"""
        shapes = self._get_batch_shapes(self.batch_size // self.num_devices)
        self.device_buffers = tuple(
            device_put_sharded(jnp.zeros(shape), self.devices)
            for shape in shapes
        )

    def _prepare(self):
        #Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Determine the number of atoms we're analyzing
        n_atoms = len(self.selections)
        
        # Create memory-mapped arrays for results
        self.results.shear_strains = np.memmap(f"{self.output_dir}/shear_strains.npy", dtype='float32', mode='w+', shape=(self.n_frames, n_atoms))
        self.results.principal_strains = np.memmap(f"{self.output_dir}/principal_strains.npy", dtype='float32', mode='w+', shape=(self.n_frames, n_atoms, 3))
        
        # Store atom info
        self.results.atom_info = [(ref_center.resid, ref_center.name) for (_, ref_center), _ in self.selections]

        # Pre-allocate arrays for batched processing
        self.current_batch = {
            'ref_sel': [],
            'ref_center': [],
            'defm_sel': [],
            'defm_center': [],
            'masks': []  # Add masks to batch storage
        }

    def _single_frame(self):
        # Convert selections to JAX arrays with padding
        ref_sel_positions = []
        ref_center_positions = []
        defm_sel_positions = []
        defm_center_positions = []
        masks = []
        
        # Find maximum number of neighbors
        max_neighbors = max(len(ref_sel) for (ref_sel, _), _ in self.selections)
        
        for (ref_sel, ref_center), (defm_sel, defm_center) in self.selections:
            # Create padded arrays and masks
            n_atoms = len(ref_sel)
            mask = jnp.zeros(max_neighbors)
            mask = mask.at[:n_atoms].set(1.0)
            
            # Pad positions with zeros
            padded_ref_pos = jnp.zeros((max_neighbors, 3))
            padded_ref_pos = padded_ref_pos.at[:n_atoms].set(ref_sel.positions)
            
            padded_defm_pos = jnp.zeros((max_neighbors, 3))
            padded_defm_pos = padded_defm_pos.at[:n_atoms].set(defm_sel.positions)
            
            ref_sel_positions.append(padded_ref_pos)
            ref_center_positions.append(jnp.array(ref_center.position))
            defm_sel_positions.append(padded_defm_pos)
            defm_center_positions.append(jnp.array(defm_center.position))
            masks.append(mask)
        
        # Stack all arrays separately
        ref_sel_positions = jnp.stack(ref_sel_positions)
        ref_center_positions = jnp.stack(ref_center_positions)
        defm_sel_positions = jnp.stack(defm_sel_positions)
        defm_center_positions = jnp.stack(defm_center_positions)
        masks = jnp.stack(masks)
        
        # Add current frame data to batch
        self.current_batch['ref_sel'].append(jnp.stack(ref_sel_positions))
        self.current_batch['ref_center'].append(jnp.stack(ref_center_positions))
        self.current_batch['defm_sel'].append(jnp.stack(defm_sel_positions))
        self.current_batch['defm_center'].append(jnp.stack(defm_center_positions))
        self.current_batch['masks'].append(jnp.stack(masks))
        
        # Process batch when it's full
        if len(self.current_batch['ref_sel']) >= self.batch_size:
            self._process_batch()

    @partial(pmap, static_broadcasted_argnums=(3,))
    def _parallel_process_batch(ref_sel, ref_center, defm_sel, defm_center, masks):
        """Process batch using multiple GPUs"""
        return process_frame_batch(ref_sel, ref_center, defm_sel, defm_center, masks)

    def _process_batch(self):
        # Reshape batch for parallel processing
        batch_size = len(self.current_batch['ref_sel'])
        devices_size = batch_size // self.num_devices
        
        # Prepare data for parallel processing
        batch_data = [
            jnp.reshape(jnp.stack(self.current_batch[key]), 
                       (self.num_devices, devices_size, -1))
            for key in ['ref_sel', 'ref_center', 'defm_sel', 'defm_center', 'masks']
        ]
        
        # Process in parallel across devices
        shears, principals = self._parallel_process_batch(*batch_data)
        
        # Combine results
        shears = jnp.reshape(shears, (batch_size, -1))
        principals = jnp.reshape(principals, (batch_size, -1, 3))
        
        # Store results efficiently
        frame_indices = range(self._frame_index - batch_size, self._frame_index)
        self.results.shear_strains[frame_indices] = np.array(shears)
        self.results.principal_strains[frame_indices] = np.array(principals)
        
        # Clear batch
        self.current_batch = {key: [] for key in self.current_batch}

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