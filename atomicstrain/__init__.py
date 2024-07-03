"""
atomicstrain
An analysis module for calculating local atomistic strain tensors.
"""

# Add imports here
from importlib.metadata import version

__version__ = version("atomicstrain")

from .analysis import StrainAnalysis
from .io import write_strain_files, write_pdb_with_strains

__all__ = ['StrainAnalysis', 'write_strain_files', 'write_pdb_with_strains']