"""
atomicstrain
An analysis module for calculating local atomistic strain tensors.
"""

# Add imports here
from importlib.metadata import version

__version__ = version("atomicstrain")

# Import main components from submodules
from .params import AnalysisParams
from .core import run_analysis
from .io import write_results

__all__ = ['AnalysisParams', 'run_analysis', 'write_results']