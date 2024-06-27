"""
atomicstrain
An analysis module for calculating local atomistic strain tensors.
"""

# Add imports here
from importlib.metadata import version

__version__ = version("atomicstrain")

from .analysis import StrainAnalysis

__all__ = ['StrainAnalysis']