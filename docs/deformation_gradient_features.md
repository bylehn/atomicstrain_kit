# Deformation Gradient and Strain Metrics

This document describes the new deformation gradient and strain metrics functionality added to the atomicstrain_kit package.

## New Features

### 1. Deformation Gradient Computation
- Computes the 3x3 deformation gradient tensor F for each atom
- Uses the existing weight-based neighbor selection
- Based on the approach from Zimmerman et al. (2009)

### 2. Advanced Strain Metrics

#### Euler Strain
- Linear and non-linear Euler strain tensors
- Based on displacement gradient calculations

#### Lagrange Strain  
- Linear and non-linear Lagrange strain tensors
- Foundation for other derived metrics

#### Strain Invariants
- First invariant (I1): trace of right Cauchy tensor
- Second invariant (I2): measure of deviatoric strain
- Third invariant (I3): determinant of right Cauchy tensor

#### Principal Stretches
- Principal stretch values (eigenvalues)
- Principal stretch directions (eigenvectors) 
- Consistent hemisphere assignment for axes

#### Rotations
- Rotation angles from polar decomposition
- Rotation axes with proper/improper handling

#### Elastic Energy
- Saint Venant-Kirchhoff energy density model
- Configurable Lamé parameters
- Default values for biological systems

## Usage

### Python API

```python
from atomicstrain import StrainAnalysis

# Define strain metrics to compute
strain_metrics = {
    'euler': True,      # Euler strain (linear/nonlinear)
    'lagrange': True,   # Lagrange strain (linear/nonlinear)
    'invariants': True, # Strain invariants (I1, I2, I3)
    'stretches': True,  # Principal stretches and axes
    'rotations': True,  # Rotation angles and axes
    'energy': True      # Elastic energy density
}

# Initialize analysis with new parameters
analysis = StrainAnalysis(
    reference=ref_universe,
    deformed=defm_universe,
    residue_numbers=residue_numbers,
    output_dir="results",
    compute_deformation_gradient=True,
    compute_strain_metrics=strain_metrics
)

# Run analysis
analysis.run()
```

### Command Line Interface

```bash
# Compute all metrics
python main.py -r ref.pdb -d def.pdb --compute-all-metrics

# Compute specific metrics
python main.py -r ref.pdb -d def.pdb --compute-euler-strain --compute-energy

# Compute deformation gradients only
python main.py -r ref.pdb -d def.pdb --compute-deformation-gradient
```

## Available Command Line Options

- `--compute-deformation-gradient`: Compute deformation gradient tensors
- `--compute-euler-strain`: Compute Euler strain (linear and non-linear)
- `--compute-lagrange-strain`: Compute Lagrange strain (linear and non-linear)
- `--compute-invariants`: Compute strain invariants (I1, I2, I3)
- `--compute-stretches`: Compute principal stretches and axes
- `--compute-rotations`: Compute rotation angles and axes
- `--compute-energy`: Compute elastic energy using Saint Venant-Kirchhoff model
- `--compute-all-metrics`: Compute all available strain metrics

## Output Files

When advanced metrics are enabled, additional memory-mapped arrays are created in `output_dir/data/`:

### Deformation Gradients
- `deformation_gradients.npy`: (n_frames, n_atoms, 3, 3) - Deformation gradient tensors

### Strain Tensors
- `euler_linear.npy`: (n_frames, n_atoms, 3, 3) - Linear Euler strain tensors
- `euler_nonlinear.npy`: (n_frames, n_atoms, 3, 3) - Non-linear Euler strain tensors
- `lagrange_linear.npy`: (n_frames, n_atoms, 3, 3) - Linear Lagrange strain tensors  
- `lagrange_nonlinear.npy`: (n_frames, n_atoms, 3, 3) - Non-linear Lagrange strain tensors

### Scalar Metrics
- `invariants.npy`: (n_frames, n_atoms, 3) - Strain invariants [I1, I2, I3]
- `principal_stretches.npy`: (n_frames, n_atoms, 3) - Principal stretch values
- `rotation_angles.npy`: (n_frames, n_atoms) - Rotation angles
- `elastic_energy.npy`: (n_frames, n_atoms) - Elastic energy density

### Vector/Tensor Metrics  
- `principal_axes.npy`: (n_frames, n_atoms, 3, 3) - Principal stretch directions
- `rotation_axes.npy`: (n_frames, n_atoms, 3) - Rotation axes

## Memory Usage

The new metrics increase memory usage significantly:
- Each 3x3 tensor metric: ~36 bytes per atom per frame
- Each 3-component vector: ~12 bytes per atom per frame
- Each scalar: ~4 bytes per atom per frame

For large trajectories, consider computing only needed metrics.

## Integration with Existing Code

The new functionality is fully backward compatible:
- Standard shear and principal strains still computed by default
- New metrics are optional and disabled by default
- Existing analysis scripts work unchanged

## Mathematical Background

The deformation gradient F relates reference (X) and deformed (x) coordinates:
```
dx = F · dX
```

From F, various strain measures can be derived:
- Right Cauchy-Green tensor: C = F^T · F
- Lagrange strain: E = (C - I)/2
- Euler strain: e = (I - (F^T · F)^(-1))/2

## Performance Notes

- Deformation gradient computation adds ~20-30% overhead
- Full strain metrics can double computation time
- Memory usage scales with number of metrics enabled
- Consider using parallel processing for large systems

## Error Handling

- Singular deformation gradients default to identity matrix
- Failed computations result in NaN values
- Warnings issued for numerical issues
- Analysis continues even if some metrics fail
