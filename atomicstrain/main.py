import os
from atomicstrain.params import AnalysisParams
from atomicstrain.core import run_analysis
from atomicstrain.io import write_results

def main():
    """
    Main function to run the strain analysis.

    This function sets up the analysis parameters, runs the analysis,
    and writes the results to output files.
    """
    params = AnalysisParams(
        reference='../examples/cript_wt_b1us.pdb',
        deformed='../examples/cript_g330t_b1us.pdb',
        traj_ref='../examples/cript_wt_b1us.xtc',
        traj_deformed='../examples/cript_g330t_b1us.xtc',
        residue_numbers=list(range(307, 398)),
        protein_ca='(name CA and resid 307-398)',
        R=10,
        stride=10,
        process_trajectory=False,
        output_dir='../examples/exp/results'
    )

    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    shear_strains, principal_strains = run_analysis(params)
    write_results(params, shear_strains, principal_strains)

if __name__ == "__main__":
    main()