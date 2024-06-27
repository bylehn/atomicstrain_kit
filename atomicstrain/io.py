import os
from typing import List
import numpy as np
from .params import AnalysisParams

def write_results(params: AnalysisParams, shear_strains: List[List[float]], principal_strains: List[List[List[float]]]):
    """
    Write the analysis results to output files.

    Args:
        params (AnalysisParams): Analysis parameters.
        shear_strains (List[List[float]]): Shear strain values for all frames.
        principal_strains (List[List[List[float]]]): Principal strain values for all frames.
    """
    avg_shear_strains = np.mean(shear_strains, axis=0)
    avg_principal_strains = np.mean(principal_strains, axis=0)

    # Write average shear strains
    with open(os.path.join(params.output_dir, 'avg_shear_strains.txt'), 'w') as f:
        for i, avg_shear in enumerate(avg_shear_strains):
            f.write(f'Residue {i+1}: {avg_shear:.4f}\n')

    # Write average principal strains for each component
    for component in range(3):
        with open(os.path.join(params.output_dir, f'avg_principal_{component+1}.txt'), 'w') as f:
            for i, principal in enumerate(avg_principal_strains):
                f.write(f'Residue {i+1}: {principal[component]:.4f}\n')

    # Write raw shear strains
    with open(os.path.join(params.output_dir, 'raw_shear_strains.txt'), 'w') as f:
        for frame in shear_strains:
            f.write(' '.join(map(lambda x: f'{x:.4f}', frame)) + '\n')

    # Write raw principal strains for each component
    for component in range(3):
        with open(os.path.join(params.output_dir, f'raw_principal_{component+1}.txt'), 'w') as f:
            for frame in principal_strains:
                f.write(' '.join(map(lambda x: f'{x[component]:.4f}', frame)) + '\n')