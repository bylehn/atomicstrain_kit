#!/usr/bin/env python
"""
Comprehensive plotting script for all strain metrics including deformation gradients,
Euler strains, Lagrange strains, invariants, principal stretches, rotations, and elastic energy.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os
import argparse
from pathlib import Path

# Set up plotting style
plt.style.use('default')

class StrainMetricsPlotter:
    """
    A comprehensive plotter for all strain metrics computed by atomicstrain_kit.
    """
    
    def __init__(self, data_dir, atom_info=None):
        """
        Initialize the plotter.
        
        Args:
            data_dir (str): Path to the data directory containing .npy files
            atom_info (list): List of (resid, atom_name) tuples for labeling
        """
        self.data_dir = Path(data_dir)
        self.atom_info = atom_info
        self.figures_dir = self.data_dir.parent / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load atom info if not provided
        if self.atom_info is None:
            self._load_atom_info()
        
        # Available metrics
        self.available_metrics = self._check_available_metrics()
        print(f"Found {len(self.available_metrics)} strain metric files")
        for metric in self.available_metrics:
            print(f"  - {metric}")
    
    def _load_atom_info(self):
        """Load atom information from a text file if available."""
        info_files = [
            self.data_dir / 'avg_shear_strains.txt',
            self.data_dir / 'atom_info.txt'
        ]
        
        for info_file in info_files:
            if info_file.exists():
                try:
                    self.atom_info = []
                    with open(info_file, 'r') as f:
                        for line in f:
                            if line.startswith('#') or not line.strip():
                                continue
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    # Try to parse the first part as residue number
                                    resid = int(parts[0])
                                    atom_name = parts[1]
                                    self.atom_info.append((resid, atom_name))
                                except ValueError:
                                    # Skip lines that don't start with a number (like headers)
                                    continue
                    print(f"Loaded atom info from {info_file}")
                    break
                except Exception as e:
                    print(f"Warning: Could not load atom info from {info_file}: {e}")
                    continue
        
        # If still no atom info, create generic labels
        if self.atom_info is None:
            print("No atom info found, using generic labels")
    
    def _check_available_metrics(self):
        """Check which strain metric files are available."""
        possible_metrics = [
            'shear_strains.npy',
            'principal_strains.npy', 
            'deformation_gradients.npy',
            'euler_linear.npy',
            'euler_nonlinear.npy',
            'lagrange_linear.npy',
            'lagrange_nonlinear.npy',
            'invariants.npy',
            'principal_stretches.npy',
            'principal_axes.npy',
            'rotation_angles.npy',
            'rotation_axes.npy',
            'elastic_energy.npy'
        ]
        
        available = []
        for metric in possible_metrics:
            if (self.data_dir / metric).exists():
                available.append(metric)
        
        return available
    
    def _load_metric(self, metric_name):
        """Load a strain metric from file."""
        file_path = str(self.data_dir / metric_name)
        try:
            # First try standard numpy loading
            return np.load(file_path, allow_pickle=True)
        except Exception as e:
            if "pickle" in str(e).lower():
                try:
                    # If it's a pickle error, try loading as raw binary data
                    raw_data = np.fromfile(file_path, dtype=np.float32)
                    return self._reshape_raw_data(raw_data, metric_name)
                except Exception as e2:
                    try:
                        # Try float64 if float32 fails
                        raw_data = np.fromfile(file_path, dtype=np.float64)
                        return self._reshape_raw_data(raw_data, metric_name)
                    except Exception as e3:
                        print(f"Warning: Could not load {metric_name} as binary data: {e3}")
                        return None
            else:
                try:
                    # Try regular loading without pickle
                    return np.load(file_path)
                except Exception as e2:
                    try:
                        # Finally try as raw binary
                        raw_data = np.fromfile(file_path, dtype=np.float32)
                        return self._reshape_raw_data(raw_data, metric_name)
                    except Exception as e3:
                        print(f"Warning: Could not load {metric_name}: {e3}")
                        return None
    
    def _reshape_raw_data(self, raw_data, metric_name):
        """Reshape raw binary data based on expected dimensions."""
        # We know we have 91 atoms from the atom info
        n_atoms = len(self.atom_info) if self.atom_info else 91
        
        # Try to infer the shape based on the metric type and data size
        total_elements = len(raw_data)
        
        # For scalar metrics (shear_strains, elastic_energy, rotation_angles)
        if metric_name in ['shear_strains.npy', 'elastic_energy.npy', 'rotation_angles.npy']:
            if total_elements % n_atoms == 0:
                n_frames = total_elements // n_atoms
                return raw_data.reshape(n_frames, n_atoms)
            else:
                # If it doesn't divide evenly, return as 1D array with correct length
                return raw_data[:n_atoms]
        
        # For 3-component vector metrics (principal_strains, principal_stretches, invariants, rotation_axes, principal_axes)
        elif metric_name in ['principal_strains.npy', 'principal_stretches.npy', 'invariants.npy', 'rotation_axes.npy', 'principal_axes.npy']:
            expected_size = n_atoms * 3
            if total_elements % expected_size == 0:
                n_frames = total_elements // expected_size
                return raw_data.reshape(n_frames, n_atoms, 3)
            else:
                # Fallback: assume single frame
                return raw_data[:expected_size].reshape(n_atoms, 3)
        
        # For tensor metrics (euler, lagrange strains) - 3x3 matrices
        elif metric_name in ['euler_linear.npy', 'euler_nonlinear.npy', 'lagrange_linear.npy', 'lagrange_nonlinear.npy']:
            expected_size = n_atoms * 9  # 3x3 = 9 components per atom
            if total_elements % expected_size == 0:
                n_frames = total_elements // expected_size
                return raw_data.reshape(n_frames, n_atoms, 3, 3)
            else:
                # Fallback: assume single frame
                return raw_data[:expected_size].reshape(n_atoms, 3, 3)
        
        # For deformation gradient (3x3 matrices)
        elif metric_name == 'deformation_gradients.npy':
            expected_size = n_atoms * 9
            if total_elements % expected_size == 0:
                n_frames = total_elements // expected_size
                return raw_data.reshape(n_frames, n_atoms, 3, 3)
            else:
                return raw_data[:expected_size].reshape(n_atoms, 3, 3)
        
        # Default: try to reshape as (n_frames, n_atoms)
        else:
            if total_elements % n_atoms == 0:
                n_frames = total_elements // n_atoms
                return raw_data.reshape(n_frames, n_atoms)
            else:
                return raw_data
    
    def _create_atom_labels(self, n_atoms):
        """Create atom labels for x-axis."""
        if self.atom_info and len(self.atom_info) >= n_atoms:
            return [f"{resid}_{name}" for resid, name in self.atom_info[:n_atoms]]
        else:
            return [f"Atom_{i+1}" for i in range(n_atoms)]
    
    def plot_scalar_metrics_overview(self):
        """Plot overview of all scalar strain metrics."""
        scalar_metrics = []
        data_arrays = []
        labels = []
        
        # Check for scalar metrics
        if 'shear_strains.npy' in self.available_metrics:
            data = self._load_metric('shear_strains.npy')
            if data is not None:
                scalar_metrics.append('Shear Strain')
                data_arrays.append(np.mean(data, axis=0))
                
        if 'elastic_energy.npy' in self.available_metrics:
            data = self._load_metric('elastic_energy.npy')
            if data is not None:
                scalar_metrics.append('Elastic Energy')
                data_arrays.append(np.mean(data, axis=0))
                
        if 'rotation_angles.npy' in self.available_metrics:
            data = self._load_metric('rotation_angles.npy')
            if data is not None:
                scalar_metrics.append('Rotation Angle')
                data_arrays.append(np.mean(data, axis=0))
        
        if not data_arrays:
            print("No scalar metrics found to plot")
            return
            
        n_atoms = len(data_arrays[0])
        atom_labels = self._create_atom_labels(n_atoms)
        
        # Create subplot figure
        n_metrics = len(scalar_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(20, 6*n_metrics), sharex=True)
        if n_metrics == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_metrics))
        
        for i, (metric, data, color) in enumerate(zip(scalar_metrics, data_arrays, colors)):
            axes[i].plot(range(n_atoms), data, 'o-', color=color, linewidth=2, markersize=4)
            axes[i].set_ylabel(f'Average {metric}')
            axes[i].set_title(f'Average {metric} per Atom')
            axes[i].grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = np.nanmean(data)
            axes[i].axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                           label=f'Mean = {mean_val:.3f}')
            axes[i].legend()
        
        # Set x-axis labels only on bottom plot
        axes[-1].set_xlabel('Atom')
        tick_spacing = max(1, n_atoms // 20)
        axes[-1].set_xticks(range(0, n_atoms, tick_spacing))
        axes[-1].set_xticklabels([atom_labels[i] for i in range(0, n_atoms, tick_spacing)], 
                                rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'scalar_metrics_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved scalar metrics overview to {self.figures_dir / 'scalar_metrics_overview.png'}")
    
    def plot_principal_strains_and_stretches(self):
        """Plot principal strains and stretches comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot principal strains if available
        if 'principal_strains.npy' in self.available_metrics:
            data = self._load_metric('principal_strains.npy')
            if data is not None:
                avg_data = np.mean(data, axis=0)  # Shape: (n_atoms, 3)
                n_atoms = avg_data.shape[0]
                atom_labels = self._create_atom_labels(n_atoms)
                
                for i in range(3):
                    axes[0,0].plot(range(n_atoms), avg_data[:, i], 'o-', 
                                  label=f'Principal Strain {i+1}', markersize=3)
                
                axes[0,0].set_title('Principal Strains (from Q-tensor)')
                axes[0,0].set_xlabel('Atom')
                axes[0,0].set_ylabel('Principal Strain')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)
        
        # Plot principal stretches if available
        if 'principal_stretches.npy' in self.available_metrics:
            data = self._load_metric('principal_stretches.npy')
            if data is not None:
                avg_data = np.mean(data, axis=0)  # Shape: (n_atoms, 3)
                n_atoms = avg_data.shape[0]
                
                for i in range(3):
                    axes[0,1].plot(range(n_atoms), avg_data[:, i], 'o-', 
                                  label=f'Principal Stretch {i+1}', markersize=3)
                
                axes[0,1].set_title('Principal Stretches (from Lagrange strain)')
                axes[0,1].set_xlabel('Atom')
                axes[0,1].set_ylabel('Principal Stretch')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
        
        # Plot strain invariants if available
        if 'invariants.npy' in self.available_metrics:
            data = self._load_metric('invariants.npy')
            if data is not None:
                avg_data = np.mean(data, axis=0)  # Shape: (n_atoms, 3)
                n_atoms = avg_data.shape[0]
                
                invariant_names = ['I₁', 'I₂', 'I₃']
                for i in range(3):
                    axes[1,0].plot(range(n_atoms), avg_data[:, i], 'o-', 
                                  label=f'Invariant {invariant_names[i]}', markersize=3)
                
                axes[1,0].set_title('Strain Invariants')
                axes[1,0].set_xlabel('Atom')
                axes[1,0].set_ylabel('Invariant Value')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
        
        # Plot correlation between different strain measures
        if ('principal_strains.npy' in self.available_metrics and 
            'principal_stretches.npy' in self.available_metrics):
            
            ps_data = self._load_metric('principal_strains.npy')
            stretch_data = self._load_metric('principal_stretches.npy')
            
            if ps_data is not None and stretch_data is not None:
                # Use first principal component for correlation
                ps_avg = np.mean(ps_data[:, :, 0], axis=0)
                stretch_avg = np.mean(stretch_data[:, :, 0], axis=0)
                
                axes[1,1].scatter(ps_avg, stretch_avg, alpha=0.6, s=30)
                axes[1,1].set_xlabel('Principal Strain 1 (Q-tensor)')
                axes[1,1].set_ylabel('Principal Stretch 1 (Lagrange)')
                axes[1,1].set_title('Correlation: Principal Strain vs Stretch')
                axes[1,1].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = np.corrcoef(ps_avg, stretch_avg)[0,1]
                axes[1,1].text(0.05, 0.95, f'r = {corr:.3f}', 
                              transform=axes[1,1].transAxes, 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'principal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved principal analysis to {self.figures_dir / 'principal_analysis.png'}")
    
    def plot_strain_tensors_heatmaps(self):
        """Plot heatmaps of strain tensor components."""
        tensor_metrics = [
            ('euler_linear.npy', 'Euler Strain (Linear)'),
            ('euler_nonlinear.npy', 'Euler Strain (Non-linear)'),
            ('lagrange_linear.npy', 'Lagrange Strain (Linear)'),
            ('lagrange_nonlinear.npy', 'Lagrange Strain (Non-linear)')
        ]
        
        available_tensors = [(f, name) for f, name in tensor_metrics if f in self.available_metrics]
        
        if not available_tensors:
            print("No strain tensor data found for heatmaps")
            return
        
        n_tensors = len(available_tensors)
        fig, axes = plt.subplots(n_tensors, 3, figsize=(15, 5*n_tensors))
        if n_tensors == 1:
            axes = axes.reshape(1, -1)
        
        for row, (filename, name) in enumerate(available_tensors):
            data = self._load_metric(filename)
            if data is None:
                continue
                
            # Average over time
            avg_tensor = np.mean(data, axis=0)  # Shape: (n_atoms, 3, 3)
            
            # Plot diagonal components as heatmap
            for col in range(3):
                component_data = avg_tensor[:, col, col]  # Diagonal component
                n_atoms = len(component_data)
                
                # Reshape for heatmap (try to make roughly square)
                n_rows = int(np.sqrt(n_atoms))
                n_cols = int(np.ceil(n_atoms / n_rows))
                
                # Pad if necessary
                padded_data = np.zeros(n_rows * n_cols)
                padded_data[:n_atoms] = component_data
                heatmap_data = padded_data.reshape(n_rows, n_cols)
                
                im = axes[row, col].imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
                axes[row, col].set_title(f'{name}\nComponent ({col+1},{col+1})')
                plt.colorbar(im, ax=axes[row, col])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'strain_tensor_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved strain tensor heatmaps to {self.figures_dir / 'strain_tensor_heatmaps.png'}")
    
    def plot_deformation_gradient_analysis(self):
        """Plot deformation gradient tensor analysis."""
        if 'deformation_gradients.npy' not in self.available_metrics:
            print("No deformation gradient data found")
            return
        
        data = self._load_metric('deformation_gradients.npy')
        if data is None:
            return
        
        # Average over time
        avg_F = np.mean(data, axis=0)  # Shape: (n_atoms, 3, 3)
        n_atoms = avg_F.shape[0]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot determinant of F (volume change)
        det_F = np.array([np.linalg.det(avg_F[i]) for i in range(n_atoms)])
        axes[0,0].plot(range(n_atoms), det_F, 'o-', markersize=3)
        axes[0,0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No volume change')
        axes[0,0].set_title('Deformation Gradient Determinant\n(Volume Change)')
        axes[0,0].set_ylabel('det(F)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot Frobenius norm of F
        frobenius_norm = np.array([np.linalg.norm(avg_F[i], 'fro') for i in range(n_atoms)])
        axes[0,1].plot(range(n_atoms), frobenius_norm, 'o-', color='green', markersize=3)
        axes[0,1].set_title('Deformation Gradient Magnitude\n(Frobenius Norm)')
        axes[0,1].set_ylabel('||F||_F')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot trace of F
        trace_F = np.array([np.trace(avg_F[i]) for i in range(n_atoms)])
        axes[0,2].plot(range(n_atoms), trace_F, 'o-', color='purple', markersize=3)
        axes[0,2].set_title('Deformation Gradient Trace')
        axes[0,2].set_ylabel('tr(F)')
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot individual components as heatmaps
        component_names = ['F₁₁', 'F₁₂', 'F₁₃']
        for col in range(3):
            component_data = avg_F[:, 0, col]  # First row of F
            
            # Create heatmap
            n_rows = int(np.sqrt(n_atoms))
            n_cols = int(np.ceil(n_atoms / n_rows))
            padded_data = np.zeros(n_rows * n_cols)
            padded_data[:n_atoms] = component_data
            heatmap_data = padded_data.reshape(n_rows, n_cols)
            
            im = axes[1, col].imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
            axes[1, col].set_title(f'Deformation Gradient\nComponent {component_names[col]}')
            plt.colorbar(im, ax=axes[1, col])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'deformation_gradient_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved deformation gradient analysis to {self.figures_dir / 'deformation_gradient_analysis.png'}")
    
    def plot_rotation_analysis(self):
        """Plot rotation analysis if available."""
        if ('rotation_angles.npy' not in self.available_metrics or 
            'rotation_axes.npy' not in self.available_metrics):
            print("No rotation data found")
            return
        
        angles_data = self._load_metric('rotation_angles.npy')
        axes_data = self._load_metric('rotation_axes.npy')
        
        if angles_data is None or axes_data is None:
            return
        
        # Average over time
        avg_angles = np.mean(angles_data, axis=0)
        avg_axes = np.mean(axes_data, axis=0)  # Shape: (n_atoms, 3)
        n_atoms = len(avg_angles)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot rotation angles
        axes[0,0].plot(range(n_atoms), avg_angles * 180/np.pi, 'o-', markersize=3)
        axes[0,0].set_title('Rotation Angles')
        axes[0,0].set_ylabel('Rotation Angle (degrees)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot rotation axis components
        axis_labels = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']
        for i in range(3):
            axes[0,1].plot(range(n_atoms), avg_axes[:, i], 'o-', 
                          label=f'{axis_labels[i]} component', 
                          color=colors[i], markersize=3)
        axes[0,1].set_title('Rotation Axis Components')
        axes[0,1].set_ylabel('Axis Component')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3D scatter plot of rotation axes (projected to 2D)
        axes[1,0].scatter(avg_axes[:, 0], avg_axes[:, 1], 
                         c=avg_angles * 180/np.pi, cmap='viridis', s=30)
        axes[1,0].set_xlabel('X component')
        axes[1,0].set_ylabel('Y component')
        axes[1,0].set_title('Rotation Axes (X-Y projection)')
        cbar = plt.colorbar(axes[1,0].collections[0], ax=axes[1,0])
        cbar.set_label('Rotation Angle (degrees)')
        
        # Histogram of rotation angles
        axes[1,1].hist(avg_angles * 180/np.pi, bins=20, alpha=0.7, edgecolor='black')
        axes[1,1].set_xlabel('Rotation Angle (degrees)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of Rotation Angles')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'rotation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved rotation analysis to {self.figures_dir / 'rotation_analysis.png'}")
    
    def plot_energy_analysis(self):
        """Plot elastic energy analysis."""
        if 'elastic_energy.npy' not in self.available_metrics:
            print("No elastic energy data found")
            return
        
        data = self._load_metric('elastic_energy.npy')
        if data is None:
            return
        
        # Time evolution and statistics
        avg_energy = np.mean(data, axis=0)
        std_energy = np.std(data, axis=0)
        n_atoms = len(avg_energy)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot average energy per atom
        axes[0,0].plot(range(n_atoms), avg_energy, 'o-', markersize=3, color='red')
        axes[0,0].fill_between(range(n_atoms), 
                              avg_energy - std_energy, 
                              avg_energy + std_energy, 
                              alpha=0.3, color='red')
        axes[0,0].set_title('Elastic Energy per Atom')
        axes[0,0].set_ylabel('Energy (J)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot energy distribution
        axes[0,1].hist(avg_energy, bins=20, alpha=0.7, edgecolor='black', color='blue')
        axes[0,1].set_xlabel('Energy (J)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Elastic Energy')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot time evolution of total energy
        total_energy_vs_time = np.sum(data, axis=1)
        axes[1,0].plot(range(len(total_energy_vs_time)), total_energy_vs_time, linewidth=2)
        axes[1,0].set_title('Total Elastic Energy vs Time')
        axes[1,0].set_xlabel('Frame')
        axes[1,0].set_ylabel('Total Energy (J)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot energy heatmap
        n_rows = int(np.sqrt(n_atoms))
        n_cols = int(np.ceil(n_atoms / n_rows))
        padded_energy = np.zeros(n_rows * n_cols)
        padded_energy[:n_atoms] = avg_energy
        energy_heatmap = padded_energy.reshape(n_rows, n_cols)
        
        im = axes[1,1].imshow(energy_heatmap, cmap='hot', aspect='auto')
        axes[1,1].set_title('Elastic Energy Heatmap')
        plt.colorbar(im, ax=axes[1,1], label='Energy (J)')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'energy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved energy analysis to {self.figures_dir / 'energy_analysis.png'}")
    
    def plot_comprehensive_overview(self):
        """Create a comprehensive overview of all metrics."""
        # Create a large summary figure
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3)
        
        # Load all available scalar data
        scalar_data = {}
        
        if 'shear_strains.npy' in self.available_metrics:
            data = self._load_metric('shear_strains.npy')
            if data is not None:
                scalar_data['Shear'] = np.mean(data, axis=0)
        
        if 'elastic_energy.npy' in self.available_metrics:
            data = self._load_metric('elastic_energy.npy')
            if data is not None:
                scalar_data['Energy'] = np.mean(data, axis=0)
        
        if 'rotation_angles.npy' in self.available_metrics:
            data = self._load_metric('rotation_angles.npy')
            if data is not None:
                scalar_data['Rotation'] = np.mean(data, axis=0) * 180/np.pi
        
        # Plot scalar metrics
        if scalar_data:
            n_atoms = len(list(scalar_data.values())[0])
            for i, (name, values) in enumerate(scalar_data.items()):
                ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
                ax.plot(range(n_atoms), values, 'o-', markersize=2)
                ax.set_title(f'Average {name}')
                ax.grid(True, alpha=0.3)
        
        # Plot principal strains/stretches comparison
        if ('principal_strains.npy' in self.available_metrics and 
            'principal_stretches.npy' in self.available_metrics):
            
            ps_data = self._load_metric('principal_strains.npy')
            stretch_data = self._load_metric('principal_stretches.npy')
            
            if ps_data is not None and stretch_data is not None:
                ax = fig.add_subplot(gs[1, :3])
                avg_ps = np.mean(ps_data, axis=0)
                for i in range(3):
                    ax.plot(range(avg_ps.shape[0]), avg_ps[:, i], 
                           label=f'Principal {i+1}', marker='o', markersize=2)
                ax.set_title('Principal Strains')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                ax = fig.add_subplot(gs[1, 3:])
                avg_stretch = np.mean(stretch_data, axis=0)
                for i in range(3):
                    ax.plot(range(avg_stretch.shape[0]), avg_stretch[:, i], 
                           label=f'Stretch {i+1}', marker='o', markersize=2)
                ax.set_title('Principal Stretches')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Plot invariants
        if 'invariants.npy' in self.available_metrics:
            data = self._load_metric('invariants.npy')
            if data is not None:
                avg_inv = np.mean(data, axis=0)
                for i in range(3):
                    ax = fig.add_subplot(gs[2, i*2:(i+1)*2])
                    ax.plot(range(avg_inv.shape[0]), avg_inv[:, i], 'o-', markersize=2)
                    ax.set_title(f'Invariant I{i+1}')
                    ax.grid(True, alpha=0.3)
        
        # Plot deformation gradient metrics
        if 'deformation_gradients.npy' in self.available_metrics:
            data = self._load_metric('deformation_gradients.npy')
            if data is not None:
                avg_F = np.mean(data, axis=0)
                n_atoms = avg_F.shape[0]
                
                # Determinant
                ax = fig.add_subplot(gs[3, :2])
                det_F = np.array([np.linalg.det(avg_F[i]) for i in range(n_atoms)])
                ax.plot(range(n_atoms), det_F, 'o-', markersize=2)
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
                ax.set_title('det(F) - Volume Change')
                ax.grid(True, alpha=0.3)
                
                # Frobenius norm
                ax = fig.add_subplot(gs[3, 2:4])
                frob_norm = np.array([np.linalg.norm(avg_F[i], 'fro') for i in range(n_atoms)])
                ax.plot(range(n_atoms), frob_norm, 'o-', markersize=2, color='green')
                ax.set_title('||F||_F - Deformation Magnitude')
                ax.grid(True, alpha=0.3)
                
                # Trace
                ax = fig.add_subplot(gs[3, 4:])
                trace_F = np.array([np.trace(avg_F[i]) for i in range(n_atoms)])
                ax.plot(range(n_atoms), trace_F, 'o-', markersize=2, color='purple')
                ax.set_title('tr(F) - Trace')
                ax.grid(True, alpha=0.3)
        
        fig.suptitle('Comprehensive Strain Metrics Overview', fontsize=20, fontweight='bold')
        plt.savefig(self.figures_dir / 'comprehensive_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comprehensive overview to {self.figures_dir / 'comprehensive_overview.png'}")
    
    def generate_all_plots(self):
        """Generate all available plots."""
        print("\n=== Generating Strain Metrics Plots ===")
        
        self.plot_scalar_metrics_overview()
        self.plot_principal_strains_and_stretches()
        self.plot_strain_tensors_heatmaps()
        self.plot_deformation_gradient_analysis()
        self.plot_rotation_analysis()
        self.plot_energy_analysis()
        self.plot_comprehensive_overview()
        
        print(f"\nAll plots saved to: {self.figures_dir}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Plot strain metrics from atomicstrain_kit analysis")
    parser.add_argument("data_dir", help="Path to data directory containing .npy files")
    parser.add_argument("--plots", choices=['all', 'scalar', 'principal', 'tensors', 'deformation', 'rotation', 'energy', 'overview'], 
                       default='all', help="Which plots to generate")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    plotter = StrainMetricsPlotter(args.data_dir)
    
    if args.plots == 'all':
        plotter.generate_all_plots()
    elif args.plots == 'scalar':
        plotter.plot_scalar_metrics_overview()
    elif args.plots == 'principal':
        plotter.plot_principal_strains_and_stretches()
    elif args.plots == 'tensors':
        plotter.plot_strain_tensors_heatmaps()
    elif args.plots == 'deformation':
        plotter.plot_deformation_gradient_analysis()
    elif args.plots == 'rotation':
        plotter.plot_rotation_analysis()
    elif args.plots == 'energy':
        plotter.plot_energy_analysis()
    elif args.plots == 'overview':
        plotter.plot_comprehensive_overview()


if __name__ == "__main__":
    main()
