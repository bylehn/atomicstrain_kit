import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from .compute import process_frame_data
from .utils import create_selections
from .io import write_strain_files, write_pdb_with_strains
from tqdm import tqdm
import os

class StrainAnalysis(AnalysisBase):
    def __init__(self, reference, deformed, residue_numbers, output_dir, min_neighbors=3, 
                 n_frames=None, use_all_heavy=False, compute_deformation_gradient=False,
                 compute_strain_metrics=None, **kwargs):
        self.ref = reference
        self.defm = deformed
        self.residue_numbers = residue_numbers
        self.min_neighbors = min_neighbors
        self.use_all_heavy = use_all_heavy
        self.compute_deformation_gradient = compute_deformation_gradient
        self.compute_strain_metrics = compute_strain_metrics or {}
        self.selections = create_selections(self.ref, residue_numbers, min_neighbors=min_neighbors, use_all_heavy=use_all_heavy)
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
        
        # Create memory-mapped arrays for additional metrics if requested
        if self.compute_deformation_gradient:
            self.results.deformation_gradients = np.memmap(
                f"{data_dir}/deformation_gradients.npy",
                dtype='float32',
                mode='w+',
                shape=(actual_n_frames, n_atoms, 3, 3)
            )
            
        if self.compute_strain_metrics:
            if self.compute_strain_metrics.get('euler', False):
                self.results.euler_linear = np.memmap(
                    f"{data_dir}/euler_linear.npy",
                    dtype='float32',
                    mode='w+',
                    shape=(actual_n_frames, n_atoms, 3, 3)
                )
                self.results.euler_nonlinear = np.memmap(
                    f"{data_dir}/euler_nonlinear.npy",
                    dtype='float32',
                    mode='w+',
                    shape=(actual_n_frames, n_atoms, 3, 3)
                )
                
            if self.compute_strain_metrics.get('lagrange', False):
                self.results.lagrange_linear = np.memmap(
                    f"{data_dir}/lagrange_linear.npy",
                    dtype='float32',
                    mode='w+',
                    shape=(actual_n_frames, n_atoms, 3, 3)
                )
                self.results.lagrange_nonlinear = np.memmap(
                    f"{data_dir}/lagrange_nonlinear.npy",
                    dtype='float32',
                    mode='w+',
                    shape=(actual_n_frames, n_atoms, 3, 3)
                )
                
            if self.compute_strain_metrics.get('invariants', False):
                self.results.invariants = np.memmap(
                    f"{data_dir}/invariants.npy",
                    dtype='float32',
                    mode='w+',
                    shape=(actual_n_frames, n_atoms, 3)
                )
                
            if self.compute_strain_metrics.get('stretches', False):
                self.results.principal_stretches = np.memmap(
                    f"{data_dir}/principal_stretches.npy",
                    dtype='float32',
                    mode='w+',
                    shape=(actual_n_frames, n_atoms, 3)
                )
                self.results.principal_axes = np.memmap(
                    f"{data_dir}/principal_axes.npy",
                    dtype='float32',
                    mode='w+',
                    shape=(actual_n_frames, n_atoms, 3, 3)
                )
                
            if self.compute_strain_metrics.get('rotations', False):
                self.results.rotation_angles = np.memmap(
                    f"{data_dir}/rotation_angles.npy",
                    dtype='float32',
                    mode='w+',
                    shape=(actual_n_frames, n_atoms)
                )
                self.results.rotation_axes = np.memmap(
                    f"{data_dir}/rotation_axes.npy",
                    dtype='float32',
                    mode='w+',
                    shape=(actual_n_frames, n_atoms, 3)
                )
                
            if self.compute_strain_metrics.get('energy', False):
                self.results.elastic_energy = np.memmap(
                    f"{data_dir}/elastic_energy.npy",
                    dtype='float32',
                    mode='w+',
                    shape=(actual_n_frames, n_atoms)
                )
        
        self.results.atom_info = [(ref_center.resid, ref_center.name) 
                                 for _, ref_center, _ in self.selections]
        
        self._frame_counter = 0


    def _single_frame(self):
        if self.has_ref_trajectory:
            self.ref.trajectory[self._frame_index]

        # Remove the pre-allocated arrays and use lists instead
        ref_positions_list = []
        ref_centers_list = []
        def_positions_list = []
        def_centers_list = []

        # Collect all positions for each atom
        weights_list = []
        for i, (ref_sel, ref_center, weights) in enumerate(self.selections):
            # Use ALL atoms in the selection, not just min_neighbors
            ref_positions_list.append(ref_sel.positions)
            ref_centers_list.append(ref_center.position)
            weights_list.append(weights)
            # For deformed structure, use the same atom indices as reference
            defm_sel = self.defm.atoms[ref_sel.indices]
            def_positions_list.append(defm_sel.positions)
            # Find the corresponding center atom in deformed structure
            defm_center = self.defm.atoms[ref_center.index]
            def_centers_list.append(defm_center.position)

        # Process frame with lists of variable-sized arrays
        if self.compute_deformation_gradient or self.compute_strain_metrics:
            results = process_frame_data(
                ref_positions_list,
                ref_centers_list,
                def_positions_list,
                def_centers_list,
                weights_list=weights_list,
                compute_deformation_gradient=self.compute_deformation_gradient,
                compute_strain_metrics=self.compute_strain_metrics
            )
            
            # Store standard results
            self.results.shear_strains[self._frame_counter] = results['shear_strains']
            self.results.principal_strains[self._frame_counter] = results['principal_strains']
            
            # Store additional results if computed
            if self.compute_deformation_gradient and 'deformation_gradients' in results:
                self.results.deformation_gradients[self._frame_counter] = results['deformation_gradients']
                
            if self.compute_strain_metrics:
                for metric, enabled in self.compute_strain_metrics.items():
                    if enabled:
                        if metric == 'euler' and 'euler_linear' in results:
                            self.results.euler_linear[self._frame_counter] = results['euler_linear']
                            self.results.euler_nonlinear[self._frame_counter] = results['euler_nonlinear']
                        elif metric == 'lagrange' and 'lagrange_linear' in results:
                            self.results.lagrange_linear[self._frame_counter] = results['lagrange_linear']
                            self.results.lagrange_nonlinear[self._frame_counter] = results['lagrange_nonlinear']
                        elif metric == 'invariants' and 'invariants' in results:
                            self.results.invariants[self._frame_counter] = results['invariants']
                        elif metric == 'stretches' and 'principal_stretches' in results:
                            self.results.principal_stretches[self._frame_counter] = results['principal_stretches']
                            self.results.principal_axes[self._frame_counter] = results['principal_axes']
                        elif metric == 'rotations' and 'rotation_angles' in results:
                            self.results.rotation_angles[self._frame_counter] = results['rotation_angles']
                            self.results.rotation_axes[self._frame_counter] = results['rotation_axes']
                        elif metric == 'energy' and 'elastic_energy' in results:
                            self.results.elastic_energy[self._frame_counter] = results['elastic_energy']
        else:
            # Standard computation
            frame_shear, frame_principal = process_frame_data(
                ref_positions_list,
                ref_centers_list,
                def_positions_list,
                def_centers_list
            )
            
            # Store results
            self.results.shear_strains[self._frame_counter] = frame_shear
            self.results.principal_strains[self._frame_counter] = frame_principal
        
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

        # Clean up
        del self.results.shear_strains
        del self.results.principal_strains