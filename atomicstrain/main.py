from atomicstrain import StrainAnalysis, visualize_strains
from atomicstrain.data.files import REFERENCE_PDB, DEFORMED_PDB
import MDAnalysis as mda
import argparse
from MDAnalysis import units

def convert_time(time_value, from_unit, to_unit='ps'):
    """Convert time from one unit to another."""
    return units.convert(time_value, from_unit, to_unit)

def main():
    parser = argparse.ArgumentParser(description="Run atomic strain analysis with optional stride and time range.")
    parser.add_argument("-r", "--reference", default=REFERENCE_PDB, help="Path to reference PDB file")
    parser.add_argument("-d", "--deformed", default=DEFORMED_PDB, help="Path to deformed PDB file")
    parser.add_argument("-rt", "--ref-trajectory", default=None, help="Path to reference trajectory file (optional)")
    parser.add_argument("-dt", "--def-trajectory", default=None, help="Path to deformed trajectory file (optional)")
    parser.add_argument("-s", "--stride", type=int, default=1, help="Stride for trajectory analysis (default: 1)")
    parser.add_argument("-o", "--output", default="results", help="Output directory for results")
    parser.add_argument("-m", "--min-neighbors", type=int, default=3, help="Minimum number of neighbors for analysis")
    parser.add_argument("-b", "--begin", type=float, default=None, help="Start time for analysis")
    parser.add_argument("-e", "--end", type=float, default=None, help="End time for analysis")
    parser.add_argument("-u", "--time-unit", default="ps", choices=['ps', 'ns', 'us', 'ms', 's'],
                        help="Time unit for begin and end times (default: ps)")
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

    # Determine the time range and number of frames to analyze
    start_frame, end_frame = None, None
    if args.def_trajectory:
        times = defm.trajectory.time
        time_unit = defm.trajectory.dt_object.unit
        print(f"Trajectory time unit: {time_unit}")
        
        if args.begin is not None:
            begin_ps = convert_time(args.begin, args.time_unit)
            start_frame = defm.trajectory.check_slice_indices([times[times >= begin_ps].min()])[0]
        if args.end is not None:
            end_ps = convert_time(args.end, args.time_unit)
            end_frame = defm.trajectory.check_slice_indices([times[times <= end_ps].max()])[0] + 1

        n_frames = len(defm.trajectory[start_frame:end_frame])
        if args.ref_trajectory:
            ref_times = ref.trajectory.time
            ref_start = ref.trajectory.check_slice_indices([ref_times[ref_times >= times[start_frame]].min()])[0] if start_frame else None
            ref_end = ref.trajectory.check_slice_indices([ref_times[ref_times <= times[end_frame-1]].max()])[0] + 1 if end_frame else None
            n_frames = min(n_frames, len(ref.trajectory[ref_start:ref_end]))

        print(f"Analyzing {n_frames} frames with stride {args.stride}")
        start_time = convert_time(times[start_frame if start_frame else 0], 'ps', args.time_unit)
        end_time = convert_time(times[end_frame-1 if end_frame else -1], 'ps', args.time_unit)
        print(f"Time range: {start_time:.2f} to {end_time:.2f} {args.time_unit}")
    else:
        n_frames = 1
        args.stride = None
        print("Analyzing single frame (no trajectories provided)")

    residue_numbers = list(range(307, 398))  # You might want to make this configurable as well

    # Run the analysis
    strain_analysis = StrainAnalysis(ref, defm, residue_numbers, args.output, args.min_neighbors, n_frames)
    strain_analysis.run(start=start_frame, stop=end_frame, stride=args.stride)

    # Create visualizations
    visualize_strains(
        residue_numbers,
        strain_analysis.results.shear_strains,
        strain_analysis.results.principal_strains,
        args.output
    )

if __name__ == "__main__":
    main()