from typing import NamedTuple, List

class AnalysisParams(NamedTuple):
    reference: str
    deformed: str
    traj_ref: str
    traj_deformed: str
    residue_numbers: List[int]
    protein_ca: str
    R: float
    stride: int
    process_trajectory: bool
    output_dir: str