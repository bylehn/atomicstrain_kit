"""
Location of data files
======================

Use as ::

    from atomicstrain.data.files import *

"""

__all__ = [
    "MDANALYSIS_LOGO",  # example file of MDAnalysis logo
    "REFERENCE_PDB",    # reference structure PDB file
    "DEFORMED_PDB",     # deformed structure PDB file
]

import importlib.resources

data_directory = importlib.resources.files("atomicstrain") / "data"

MDANALYSIS_LOGO = data_directory / "mda.txt"
REFERENCE_PDB = data_directory / "cript_wt.pdb"
DEFORMED_PDB = data_directory / "cript_g330t.pdb"
