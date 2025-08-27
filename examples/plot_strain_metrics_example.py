#!/usr/bin/env python
"""
Example usage of the strain metrics plotting functionality.
This script shows how to generate comprehensive plots of all strain metrics.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from atomicstrain.data.results.plot_strain_metrics import StrainMetricsPlotter

def plot_example_analysis():
    """Example of how to use the plotting functionality."""
    
    # Path to your analysis results
    results_dir = "path/to/your/results"  # Change this to your actual results directory
    data_dir = os.path.join(results_dir, "data")
    
    print("=== Strain Metrics Plotting Example ===")
    print(f"Looking for data in: {data_dir}")
    
    if not os.path.exists(data_dir):
        print("Error: Data directory not found!")
        print("Please run an analysis first with the new strain metrics enabled.")
        print("\nExample analysis command:")
        print("python main.py -r reference.pdb -d deformed.pdb --compute-all-metrics")
        return
    
    # Create plotter instance
    print("\nInitializing plotter...")
    plotter = StrainMetricsPlotter(data_dir)
    
    # Generate specific plots
    print("\nGenerating individual plots...")
    
    try:
        # Plot scalar metrics (shear strain, energy, rotation angles)
        plotter.plot_scalar_metrics_overview()
        
        # Plot principal strains vs principal stretches
        plotter.plot_principal_strains_and_stretches()
        
        # Plot strain tensor heatmaps
        plotter.plot_strain_tensors_heatmaps()
        
        # Plot deformation gradient analysis
        plotter.plot_deformation_gradient_analysis()
        
        # Plot rotation analysis
        plotter.plot_rotation_analysis()
        
        # Plot elastic energy analysis
        plotter.plot_energy_analysis()
        
        # Generate comprehensive overview
        plotter.plot_comprehensive_overview()
        
        print(f"\n✅ All plots generated successfully!")
        print(f"📁 Check the figures directory: {plotter.figures_dir}")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")


def demonstrate_command_line_usage():
    """Show command line usage examples."""
    
    print("\n=== Command Line Usage Examples ===")
    
    examples = [
        {
            'desc': 'Generate all plots',
            'cmd': 'python atomicstrain/data/plot_strain_metrics.py results/data'
        },
        {
            'desc': 'Generate only scalar metrics plots',
            'cmd': 'python atomicstrain/data/plot_strain_metrics.py results/data --plots scalar'
        },
        {
            'desc': 'Generate only deformation gradient plots',
            'cmd': 'python atomicstrain/data/plot_strain_metrics.py results/data --plots deformation'
        },
        {
            'desc': 'Generate comprehensive overview only',
            'cmd': 'python atomicstrain/data/plot_strain_metrics.py results/data --plots overview'
        }
    ]
    
    for example in examples:
        print(f"\n{example['desc']}:")
        print(f"  {example['cmd']}")
    
    print("\nAvailable plot types:")
    plot_types = ['all', 'scalar', 'principal', 'tensors', 'deformation', 'rotation', 'energy', 'overview']
    for plot_type in plot_types:
        print(f"  - {plot_type}")


def show_expected_outputs():
    """Show what output files will be generated."""
    
    print("\n=== Expected Output Files ===")
    
    output_files = [
        {
            'file': 'scalar_metrics_overview.png',
            'desc': 'Line plots of shear strain, elastic energy, and rotation angles'
        },
        {
            'file': 'principal_analysis.png', 
            'desc': 'Principal strains vs stretches, invariants, and correlations'
        },
        {
            'file': 'strain_tensor_heatmaps.png',
            'desc': 'Heatmap visualizations of Euler and Lagrange strain tensors'
        },
        {
            'file': 'deformation_gradient_analysis.png',
            'desc': 'Analysis of deformation gradient determinant, norm, and components'
        },
        {
            'file': 'rotation_analysis.png',
            'desc': 'Rotation angles, axes, and their distributions'
        },
        {
            'file': 'energy_analysis.png',
            'desc': 'Elastic energy per atom, distributions, and time evolution'
        },
        {
            'file': 'comprehensive_overview.png',
            'desc': 'Single-page overview of all computed strain metrics'
        }
    ]
    
    print("The following files will be created in results/figures/:")
    for output in output_files:
        print(f"\n📊 {output['file']}")
        print(f"   {output['desc']}")


if __name__ == "__main__":
    print("🔬 Atomicstrain Kit - Strain Metrics Plotting")
    print("=" * 50)
    
    plot_example_analysis()
    demonstrate_command_line_usage()
    show_expected_outputs()
    
    print("\n💡 Quick Start:")
    print("1. Run analysis with new metrics: python main.py -r ref.pdb -d def.pdb --compute-all-metrics")
    print("2. Generate plots: python atomicstrain/data/plot_strain_metrics.py results/data")
    print("3. Check results/figures/ for all generated plots")
