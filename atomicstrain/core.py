from typing import List, Tuple
import MDAnalysis as mda
import numpy as np
from .params import AnalysisParams
from .compute import process_frame

def initialize(params: AnalysisParams) -> Tuple[mda.Universe, mda.Universe, List]:
    """
    Initialize MDAnalysis Universe objects and create atom selections.

    Args:
        params (AnalysisParams): Analysis parameters.

    Returns:
        Tuple[mda.Universe, mda.Universe, List]: A tuple containing:
            - ref (mda.Universe): MDAnalysis Universe for the reference structure.
            - defm (mda.Universe): MDAnalysis Universe for the deformed structure.
            - selections (List): List of atom selections for strain calculations.
    """
    if params.process_trajectory:
        ref = mda.Universe(params.reference, params.traj_ref)
        defm = mda.Universe(params.deformed, params.traj_deformed)
    else:
        ref = mda.Universe(params.reference)
        defm = mda.Universe(params.deformed)
    
    selections = []
    for resid in params.residue_numbers:
        selection_str = f"({params.protein_ca} and around {params.R} (resid {resid} and name CA))"
        center_str = f"resid {resid} and name CA"

        ref_selection = ref.select_atoms(selection_str)
        ref_center = ref.select_atoms(center_str)

        ref_resids = ref_selection.resids
        defm_selection_str = f"(name CA and resid {' '.join(map(str, ref_resids))})"
        defm_center_str = f"resid {resid} and name CA" 

        defm_selection = defm.select_atoms(defm_selection_str)
        defm_center = defm.select_atoms(defm_center_str)

        selections.append(((ref_selection, ref_center), (defm_selection, defm_center)))
    
    return ref, defm, selections

def run_analysis(params: AnalysisParams) -> Tuple[List[List[float]], List[List[List[float]]]]:
    """
    Run the strain analysis for all frames.

    Args:
        params (AnalysisParams): Analysis parameters.

    Returns:
        Tuple[List[List[float]], List[List[List[float]]]]: A tuple containing:
            - shear_strains (List[List[float]]): Shear strain values for all frames.
            - principal_strains (List[List[List[float]]]): Principal strain values for all frames.
    """
    ref, defm, selections = initialize(params)
    
    shear_strains = []
    principal_strains = []

    if params.process_trajectory:
        num_frames = len(ref.trajectory[::params.stride])
        for frame_idx in range(num_frames):
            actual_frame_idx = frame_idx * params.stride
            frame_shear, frame_principal = process_frame(ref, defm, selections, actual_frame_idx)
            shear_strains.append(frame_shear)
            principal_strains.append(frame_principal)
    else:
        frame_shear, frame_principal = process_frame(ref, defm, selections, 0)
        shear_strains.append(frame_shear)
        principal_strains.append(frame_principal)

    return shear_strains, principal_strains