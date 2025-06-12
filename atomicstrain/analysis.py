import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from .compute import process_frame_data
from .utils import create_selections
from .io import write_strain_files, write_pdb_with_strains
from tqdm import tqdm
import os

class StrainAnalysis(AnalysisBase):
    def __init__(self, reference, deformed, residue_numbers, output_dir, min_neighbors=3, n_frames=None, use_all_heavy=False,
                 calculate_rmsf=True, **kwargs):
        self.ref = reference
        self.defm = deformed
        self.residue_numbers = residue_numbers
        self.min_neighbors = min_neighbors
        self.use_all_heavy = use_all_heavy
        self.calculate_rmsf = calculate_rmsf
        self.selections = create_selections(self.ref, self.defm, residue_numbers, min_neighbors, use_all_heavy)
        self.output_dir = output_dir
        self.has_ref_trajectory = hasattr(self.ref, 'trajectory') and len(self.ref.trajectory) > 1
        
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        
        super().__init__(self.defm.trajectory, **kwargs)

    def _prepare(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calculate actual number of frames
        start = self.start if self.start is not None else 0
        stop = self.stop if self.stop is not None else len(self.defm.trajectory)
        step = self.step if self.step is not None else 1
        actual_n_frames = len(range(start, stop, step))
        n_atoms = len(self.selections)
        
        print(f"Preparing analysis for {actual_n_frames} frames")
        
        # Create data subdirectory
        data_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create memory-mapped arrays
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
        
        self.results.atom_info = [(ref_center.resid, ref_center.name) 
                                 for (_, ref_center), _ in self.selections]
        
        # Initialize RMSF-related attributes only if RMSF calculation is enabled
        if self.calculate_rmsf:
            # For RMSF, we need to store absolute positions, not displacements
            self.results._positions_sum = np.zeros((n_atoms, 3), dtype=np.float32)
            self.results._positions_sq_sum = np.zeros((n_atoms, 3), dtype=np.float32)
        
        self._frame_counter = 0

    def _single_frame(self):
        if self.has_ref_trajectory:
            self.ref.trajectory[self._frame_index]

        # Pre-allocate numpy arrays instead of lists
        n_selections = len(self.selections)
        ref_positions = np.zeros((n_selections, self.min_neighbors, 3), dtype=np.float32)
        ref_centers = np.zeros((n_selections, 3), dtype=np.float32)
        def_positions = np.zeros((n_selections, self.min_neighbors, 3), dtype=np.float32)
        def_centers = np.zeros((n_selections, 3), dtype=np.float32)

        # Fill arrays directly
        for i, ((ref_sel, ref_center), (defm_sel, defm_center)) in enumerate(self.selections):
            ref_positions[i, :len(ref_sel.positions)] = ref_sel.positions[:self.min_neighbors]
            ref_centers[i] = ref_center.position
            def_positions[i, :len(defm_sel.positions)] = defm_sel.positions[:self.min_neighbors]
            def_centers[i] = defm_center.position

        # Process frame with pre-allocated arrays
        frame_shear, frame_principal = process_frame_data(
            ref_positions,
            ref_centers,
            def_positions,
            def_centers
        )

        # Store results
        self.results.shear_strains[self._frame_counter] = frame_shear
        self.results.principal_strains[self._frame_counter] = frame_principal

        # Calculate RMSF if enabled
        if self.calculate_rmsf:
            # Get current positions of deformed centers
            current_positions = def_centers
            
            # Add to running sums for RMSF calculation
            self.results._positions_sum += current_positions
            self.results._positions_sq_sum += current_positions ** 2
        
        # Flush less frequently
        if self._frame_counter % 1000 == 0:
            self.results.shear_strains.flush()
            self.results.principal_strains.flush()
        
        self._frame_counter += 1
        
    def run(self, start=None, stop=None, stride=None, verbose=True):
        """
        Run the analysis with enhanced progress tracking and error handling.

        Parameters
        ----------
        start : int, optional
            First frame of trajectory to analyze, default: None (start at beginning)
        stop : int, optional
            Last frame of trajectory to analyze, default: None (end of trajectory)
        stride : int, optional
            Number of frames to skip between each analyzed frame, default: None (use every frame)
        verbose : bool, optional
            Show detailed progress, default: True

        Returns
        -------
        self : StrainAnalysis
            Return self to allow for method chaining
        """
        # Store parameters
        self.start = start
        self.stop = stop
        self.step = stride

        # Calculate total frames to be analyzed
        trajectory_length = len(self.defm.trajectory)
        start_frame = start if start is not None else 0
        end_frame = stop if stop is not None else trajectory_length
        step_size = stride if stride is not None else 1
        
        total_frames = len(range(start_frame, end_frame, step_size))
        
        if verbose:
            print("\nAnalysis Setup:")
            print(f"Total trajectory frames: {trajectory_length}")
            print(f"Analyzing frames {start_frame} to {end_frame} with stride {step_size}")
            print(f"Total frames to analyze: {total_frames}")
            if self.has_ref_trajectory:
                print("Using reference trajectory")
            print(f"Number of atoms to analyze: {len(self.selections)}")
            print(f"RMSF calculation: {'enabled' if self.calculate_rmsf else 'disabled'}")
            
            # Memory usage estimate
            mem_per_frame = (len(self.selections) * 4 * 4)  # 4 bytes per float32, 4 values per atom
            total_mem_estimate = (mem_per_frame * total_frames) / (1024 * 1024)  # Convert to MB
            print(f"Estimated memory usage: {total_mem_estimate:.2f} MB")
            
            print("\nStarting analysis...")

        try:
            import time
            start_time = time.time()
            
            # Run the analysis
            result = super().run(start=start, stop=stop, step=stride, verbose=verbose)
            
            if verbose:
                end_time = time.time()
                duration = end_time - start_time
                frames_per_second = total_frames / duration
                print(f"\nAnalysis completed in {duration:.2f} seconds")
                print(f"Average processing speed: {frames_per_second:.2f} frames/second")
                
                # Memory usage report
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
                print(f"Final memory usage: {memory_usage:.2f} MB")
            
            return result
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            
            # Additional error context
            if self._frame_counter > 0:
                print(f"Error occurred after processing {self._frame_counter} frames")
                print("Partial results may be available")
            
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            
            # Try to clean up memory-mapped files
            try:
                del self.results.shear_strains
                del self.results.principal_strains
            except:
                pass
            
            raise

    def _conclude(self):
        # Ensure all data is written
        self.results.shear_strains.flush()
        self.results.principal_strains.flush()
        
        # Compute averages
        self.results.avg_shear_strains = np.mean(self.results.shear_strains[:self._frame_counter], axis=0)
        self.results.avg_principal_strains = np.mean(self.results.principal_strains[:self._frame_counter], axis=0)
        
        # Calculate RMSF if enabled
        if self.calculate_rmsf:
            n_frames = self._frame_counter
            
            # Calculate mean positions
            mean_positions = self.results._positions_sum / n_frames
            
            # Calculate mean of squared positions
            mean_sq_positions = self.results._positions_sq_sum / n_frames
            
            # Calculate variance: Var(X) = E[X^2] - E[X]^2
            variance = mean_sq_positions - mean_positions ** 2
            
            # Handle numerical errors (variance should never be negative)
            variance = np.maximum(variance, 0)
            
            # RMSF is the square root of the sum of variances for x, y, z
            self.results.rmsf = np.sqrt(np.sum(variance, axis=1))
            
            # Calculate normalized strains
            epsilon = 1e-10  # Small value to prevent division by zero
            self.results.norm_avg_shear_strains = (
                self.results.avg_shear_strains / (self.results.rmsf + epsilon)
            )
            self.results.norm_avg_principal_strains = (
                self.results.avg_principal_strains / (self.results.rmsf[:, np.newaxis] + epsilon)
            )
            
            print(f"\nRMSF statistics:")
            print(f"  Mean RMSF: {np.mean(self.results.rmsf):.4f} Å")
            print(f"  Min RMSF: {np.min(self.results.rmsf):.4f} Å")
            print(f"  Max RMSF: {np.max(self.results.rmsf):.4f} Å")
        else:
            # Set to None if not calculated
            self.results.rmsf = None
            self.results.norm_avg_shear_strains = None
            self.results.norm_avg_principal_strains = None

        # Save copies for visualization
        self.results.final_shear_strains = np.array(
            self.results.shear_strains[:self._frame_counter],
            dtype=np.float32
        )
        self.results.final_principal_strains = np.array(
            self.results.principal_strains[:self._frame_counter],
            dtype=np.float32
        )

        write_strain_files(
            self.output_dir,
            self.results.shear_strains[:self._frame_counter],
            self.results.principal_strains[:self._frame_counter],
            self.results.avg_shear_strains,
            self.results.avg_principal_strains,
            self.results.atom_info,
            self.use_all_heavy,
            rmsf=self.results.rmsf,
            norm_avg_shear_strains=self.results.norm_avg_shear_strains,
            norm_avg_principal_strains=self.results.norm_avg_principal_strains
        )

        write_pdb_with_strains(
            self.defm.filename,
            self.output_dir,
            self.residue_numbers,
            self.results.avg_shear_strains,
            self.results.avg_principal_strains,
            self.results.atom_info,
            self.use_all_heavy,
            rmsf=self.results.rmsf,
            norm_avg_shear_strains=self.results.norm_avg_shear_strains,
            norm_avg_principal_strains=self.results.norm_avg_principal_strains
        )

        # Clean up
        del self.results.shear_strains
        del self.results.principal_strains