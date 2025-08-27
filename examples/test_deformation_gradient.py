#!/usr/bin/env python
"""
Example script demonstrating the new deformation gradient and strain metrics functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import MDAnalysis as mda
from atomicstrain import StrainAnalysis

def test_new_metrics():
    """Test the new deformation gradient and strain metrics functionality."""
    
    # Setup paths (adjust these to your actual files)
    reference_pdb = "path/to/reference.pdb"
    deformed_pdb = "path/to/deformed.pdb" 
    output_dir = "test_results"
    
    # Define residue range
    residue_numbers = list(range(6, 98))  # residues 6-97
    
    # Create strain metrics configuration
    strain_metrics = {
        'euler': True,           # Compute Euler strain (linear and non-linear)
        'lagrange': True,        # Compute Lagrange strain (linear and non-linear) 
        'invariants': True,      # Compute strain invariants (I1, I2, I3)
        'stretches': True,       # Compute principal stretches and axes
        'rotations': True,       # Compute rotation angles and axes
        'energy': True           # Compute elastic energy
    }
    
    print("=== Testing New Strain Metrics ===")
    print("Strain metrics to compute:")
    for metric, enabled in strain_metrics.items():
        if enabled:
            print(f"  - {metric}")
    
    try:
        # Load structures (replace with actual file loading)
        # ref = mda.Universe(reference_pdb)
        # defm = mda.Universe(deformed_pdb)
        
        # For testing, create simple demonstration
        print("\nNote: This is a demonstration of the new API.")
        print("Replace the file paths above with your actual PDB files to run.")
        
        # Show how to use the new StrainAnalysis class
        print("\nUsage example:")
        print("""
# Initialize with new parameters
strain_analysis = StrainAnalysis(
    reference=ref_universe,
    deformed=defm_universe, 
    residue_numbers=residue_numbers,
    output_dir=output_dir,
    min_neighbors=3,
    use_all_heavy=False,
    compute_deformation_gradient=True,  # Enable deformation gradients
    compute_strain_metrics=strain_metrics  # Enable strain metrics
)

# Run analysis
strain_analysis.run()

# Results will be saved as memory-mapped arrays in output_dir/data/
# Available result arrays:
# - shear_strains.npy (standard)
# - principal_strains.npy (standard)
# - deformation_gradients.npy (new)
# - euler_linear.npy, euler_nonlinear.npy (new)
# - lagrange_linear.npy, lagrange_nonlinear.npy (new)
# - invariants.npy (new)
# - principal_stretches.npy, principal_axes.npy (new)
# - rotation_angles.npy, rotation_axes.npy (new)
# - elastic_energy.npy (new)
        """)
        
    except Exception as e:
        print(f"Error: {e}")

def test_command_line():
    """Show command line usage examples."""
    
    print("\n=== Command Line Usage Examples ===")
    
    examples = [
        {
            'desc': 'Compute deformation gradients only',
            'cmd': 'python main.py -r reference.pdb -d deformed.pdb --compute-deformation-gradient'
        },
        {
            'desc': 'Compute Euler and Lagrange strains',
            'cmd': 'python main.py -r reference.pdb -d deformed.pdb --compute-euler-strain --compute-lagrange-strain'
        },
        {
            'desc': 'Compute all available strain metrics',
            'cmd': 'python main.py -r reference.pdb -d deformed.pdb --compute-all-metrics'
        },
        {
            'desc': 'Compute specific metrics with trajectory',
            'cmd': 'python main.py -r reference.pdb -d deformed.pdb -dtraj trajectory.xtc --compute-stretches --compute-energy'
        }
    ]
    
    for example in examples:
        print(f"\n{example['desc']}:")
        print(f"  {example['cmd']}")

if __name__ == "__main__":
    test_new_metrics()
    test_command_line()
