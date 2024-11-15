import argparse
import math
import sys
from atomicstrain import StrainAnalysis, visualize_strains
from atomicstrain.data.files import REFERENCE_PDB, DEFORMED_PDB
import MDAnalysis as mda
import numpy as np

def ns_to_frame(time_ns, dt_ps):
    """Convert time in nanoseconds to the nearest frame number."""
    return round(time_ns * 1000 / dt_ps)

def frame_to_ns(frame, dt_ps):
    """Convert frame number to time in nanoseconds."""
    return frame * dt_ps / 1000

def main():
    parser = argparse.ArgumentParser(description="Run atomic strain analysis with optional stride and time range.")
    parser.add_argument("-r", "--reference", default=REFERENCE_PDB, help="Path to reference PDB file")
    parser.add_argument("-d", "--deformed", default=DEFORMED_PDB, help="Path to deformed PDB file")
    parser.add_argument("-rtraj", "--ref-trajectory", default=None, help="Path to reference trajectory file (optional)")
    parser.add_argument("-dtraj", "--def-trajectory", default=None, help="Path to deformed trajectory file (optional)")
    parser.add_argument("-s", "--stride", type=int, default=1, help="Stride for trajectory analysis (default: 1)")
    parser.add_argument("-o", "--output", default="results", help="Output directory for results")
    parser.add_argument("-m", "--min-neighbors", type=int, default=3, help="Minimum number of neighbors for analysis")
    parser.add_argument("-b", "--begin", type=float, default=None, help="Start time for analysis (in ns)")
    parser.add_argument("-e", "--end", type=float, default=None, help="End time for analysis (in ns)")
    parser.add_argument("-dt", "--time-step", type=float, default=None, help="Time step between frames (in ps)")
    parser.add_argument("--use-all-heavy", action="store_true", help="Use all heavy atoms instead of only CA atoms")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for GPU processing (default: 512)")
    args = parser.parse_args()

    # Check if dt is provided when begin or end is specified
    if (args.begin is not None or args.end is not None) and args.time_step is None:
        print("Error: Time step (-dt) must be provided when specifying begin (-b) or end (-e) times.")
        sys.exit(1)

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

    # Determine the total number of frames
    total_frames = len(defm.trajectory)
    if args.ref_trajectory:
        total_frames = min(total_frames, len(ref.trajectory))

    # Handle begin and end times if provided
    if args.begin is not None or args.end is not None:
        total_time_ns = frame_to_ns(total_frames - 1, args.time_step)

        if args.begin is None:
            start_frame = 0
        else:
            if args.begin < 0:
                print(f"Error: Begin time ({args.begin} ns) cannot be negative.")
                sys.exit(1)
            if args.begin > total_time_ns:
                print(f"Error: Begin time ({args.begin} ns) is beyond the trajectory length ({total_time_ns:.2f} ns).")
                sys.exit(1)
            start_frame = ns_to_frame(args.begin, args.time_step)

        if args.end is None:
            end_frame = total_frames
        else:
            if args.end <= args.begin:
                print(f"Error: End time ({args.end} ns) must be greater than begin time ({args.begin} ns).")
                sys.exit(1)
            if args.end > total_time_ns:
                print(f"Warning: End time ({args.end} ns) is beyond the trajectory length ({total_time_ns:.2f} ns). Using the last frame.")
                end_frame = total_frames
            else:
                end_frame = min(ns_to_frame(args.end, args.time_step), total_frames)

        print(f"Time range: {frame_to_ns(start_frame, args.time_step):.2f} to {frame_to_ns(end_frame - 1, args.time_step):.2f} ns")
    else:
        start_frame = 0
        end_frame = total_frames

    n_frames = math.ceil((end_frame - start_frame) / args.stride)

    if n_frames <= 0:
        print("Error: No frames to analyze with the given parameters.")
        sys.exit(1)

    print(f"Total frames in trajectory: {total_frames}")
    print(f"Analyzing frames from {start_frame} to {end_frame} with stride {args.stride}")
    n_frames = math.ceil((end_frame - start_frame) / args.stride)
    print(f"Total frames to be analyzed: {n_frames}")

    residue_numbers = list(range(6, 97))  # You might want to make this configurable as well

    # Run the analysis
    strain_analysis = StrainAnalysis(
        ref, defm, residue_numbers, args.output, 
        min_neighbors=args.min_neighbors,
        n_frames=n_frames,
        use_all_heavy=args.use_all_heavy,
        batch_size=args.batch_size  # Add batch_size parameter
    )
    strain_analysis.run(start=start_frame, stop=end_frame, stride=args.stride)

    # Create visualizations
    visualize_strains(
        strain_analysis.results.atom_info,
        np.memmap(f"{args.output}/shear_strains.npy", dtype='float32', mode='r', shape=(n_frames, len(strain_analysis.results.atom_info))),
        np.memmap(f"{args.output}/principal_strains.npy", dtype='float32', mode='r', shape=(n_frames, len(strain_analysis.results.atom_info), 3)),
        args.output
    )

if __name__ == "__main__":
    main()