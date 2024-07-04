from atomicstrain import StrainAnalysis, visualize_strains
from atomicstrain.data.files import REFERENCE_PDB, DEFORMED_PDB
import MDAnalysis as mda
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run atomic strain analysis with optional stride.")
    parser.add_argument("-r", "--reference", default=REFERENCE_PDB, help="Path to reference PDB file")
    parser.add_argument("-d", "--deformed", default=DEFORMED_PDB, help="Path to deformed PDB file")
    parser.add_argument("-rtraj", "--ref-trajectory", default=None, help="Path to reference trajectory file (optional)")
    parser.add_argument("-dtraj", "--def-trajectory", default=None, help="Path to deformed trajectory file (optional)")
    parser.add_argument("-s", "--stride", type=int, default=1, help="Stride for trajectory analysis (default: 1)")
    parser.add_argument("-o", "--output", default="results", help="Output directory for results")
    parser.add_argument("-m", "--min-neighbors", type=int, default=3, help="Minimum number of neighbors for analysis")
    args = parser.parse_args()

    # Set up your analysis
    if args.ref_trajectory:
        ref = mda.Universe(args.reference, args.ref_trajectory)
        print(f"Using reference trajectory: {args.ref_trajectory}")
    else:
        ref = mda.Universe(args.reference)
        print(f"Using reference PDB: {args.reference}")

    if args.def_trajectory:
        defm = mda.Universe(args.deformed, args.def_trajectory)
        print(f"Using deformed trajectory: {args.def_trajectory}")
    else:
        defm = mda.Universe(args.deformed)
        print(f"Using deformed PDB: {args.deformed}")

    # Determine the number of frames to analyze
    if args.def_trajectory:
        n_frames = len(defm.trajectory)
        if args.ref_trajectory:
            n_frames = min(n_frames, len(ref.trajectory))
        print(f"Analyzing {n_frames} frames with stride {args.stride}")
    else:
        n_frames = 1
        args.stride = None
        print("Analyzing single frame (no trajectories provided)")

    residue_numbers = list(range(307, 398))  # You might want to make this configurable as well

    # Run the analysis
    strain_analysis = StrainAnalysis(ref, defm, residue_numbers, args.output, args.min_neighbors, n_frames)
    strain_analysis.run(stride=args.stride)

    # Create visualizations
    visualize_strains(
        residue_numbers,
        strain_analysis.results.shear_strains,
        strain_analysis.results.principal_strains,
        args.output
    )

if __name__ == "__main__":
    main()