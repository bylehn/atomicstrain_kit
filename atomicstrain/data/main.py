import argparse
import math
import numpy as np
import sys
import os
import datetime
import json
from typing import List, Dict, Any
from atomicstrain import StrainAnalysis, visualize_strains
from atomicstrain.data.files import REFERENCE_PDB, DEFORMED_PDB
import MDAnalysis as mda

def ensure_output_dirs(output_dir):
    """Create all necessary output directories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'structures'), exist_ok=True)

def ns_to_frame(time_ns: float, dt_ps: float) -> int:
    """Convert time in nanoseconds to the nearest frame number."""
    return round(time_ns * 1000 / dt_ps)

def frame_to_ns(frame: int, dt_ps: float) -> float:
    """Convert frame number to time in nanoseconds."""
    return frame * dt_ps / 1000

def load_universe_with_trajectories(topology: str, trajectories: str) -> mda.Universe:
    """
    Create an MDAnalysis Universe from a topology file and one or more trajectory files.
    
    Args:
        topology: Path to the topology file (PDB)
        trajectories: Comma-separated list of trajectory files or single trajectory file
    
    Returns:
        MDAnalysis Universe object with all trajectories
    """
    if trajectories is None:
        return mda.Universe(topology)
        
    traj_list = [t.strip() for t in trajectories.split(',')]
    print(f"Loading trajectory files: {traj_list}")
    
    return mda.Universe(topology, traj_list)

def get_total_frames(universe: mda.Universe) -> int:
    """
    Get the total number of frames from a Universe object.
    Takes into account that the trajectory might be concatenated.
    
    Args:
        universe: MDAnalysis Universe object
    
    Returns:
        Total number of frames across all trajectories
    """
    return len(universe.trajectory)

def save_run_info(args: argparse.Namespace, output_dir: str) -> None:
    """
    Save the command-line arguments and run information to a file.
    
    Args:
        args: Parsed command-line arguments
        output_dir: Directory to save the run info
    """
    # Create run info directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Convert args to dictionary, handling default values
    args_dict = {
        'reference': args.reference,
        'deformed': args.deformed,
        'ref_trajectory': args.ref_trajectory,
        'def_trajectory': args.def_trajectory,
        'stride': args.stride,
        'output': args.output,
        'min_neighbors': args.min_neighbors,
        'begin': args.begin,
        'end': args.end,
        'time_step': args.time_step,
        'use_all_heavy': args.use_all_heavy,
        'residue_range': args.residue_range,
        'calculate_rmsf': args.calculate_rmsf  # Add RMSF flag
    }
    
    # Create run info dictionary
    run_info = {
        'timestamp': timestamp,
        'arguments': args_dict,
        'command': reconstructed_command(args_dict)
    }
    
    # Save run info to JSON file
    run_info_file = os.path.join(output_dir, 'run_info.json')
    with open(run_info_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    # Also save as a plain text file for easy reading
    run_info_txt = os.path.join(output_dir, 'run_command.txt')
    with open(run_info_txt, 'w') as f:
        f.write(f"# Run performed on: {timestamp}\n\n")
        f.write("# Command to reproduce this analysis:\n")
        f.write(run_info['command'] + "\n\n")
        f.write("# Full arguments:\n")
        for key, value in args_dict.items():
            f.write(f"# {key}: {value}\n")

def reconstructed_command(args_dict: Dict[str, Any]) -> str:
    """
    Reconstruct the command-line command from the arguments.
    
    Args:
        args_dict: Dictionary of command-line arguments
    
    Returns:
        String containing the reconstructed command
    """
    cmd_parts = ["python", "main.py"]
    
    for key, value in args_dict.items():
        if value is None:
            continue
            
        # Convert argument name from python style to command line style
        arg_name = "--" + key.replace("_", "-")
        
        # Handle boolean flags
        if isinstance(value, bool):
            if key == 'calculate_rmsf':
                # Special handling for RMSF flag
                if value:
                    cmd_parts.append("--rmsf")
                else:
                    cmd_parts.append("--no-rmsf")
            elif value:
                cmd_parts.append(arg_name)
        # Handle all other arguments
        else:
            cmd_parts.append(arg_name)
            # Quote strings containing spaces
            if isinstance(value, str) and (" " in value or "," in value):
                cmd_parts.append(f'"{value}"')
            else:
                cmd_parts.append(str(value))
    
    return " ".join(cmd_parts)

def main():
    parser = argparse.ArgumentParser(description="Run atomic strain analysis with optional stride and time range.")
    parser.add_argument("-r", "--reference", default=REFERENCE_PDB, help="Path to reference PDB file")
    parser.add_argument("-d", "--deformed", default=DEFORMED_PDB, help="Path to deformed PDB file")
    parser.add_argument("-rtraj", "--ref-trajectory", default=None, 
                       help="Path to reference trajectory file(s). Multiple files should be comma-separated.")
    parser.add_argument("-dtraj", "--def-trajectory", default=None, 
                       help="Path to deformed trajectory file(s). Multiple files should be comma-separated.")
    parser.add_argument("-s", "--stride", type=int, default=1, help="Stride for trajectory analysis (default: 1)")
    parser.add_argument("-o", "--output", default="results", help="Output directory for results")
    parser.add_argument("-m", "--min-neighbors", type=int, default=3, help="Minimum number of neighbors for analysis")
    parser.add_argument("-b", "--begin", type=float, default=None, help="Start time for analysis (in ns)")
    parser.add_argument("-e", "--end", type=float, default=None, help="End time for analysis (in ns)")
    parser.add_argument("-dt", "--time-step", type=float, default=None, help="Time step between frames (in ps)")
    parser.add_argument("--use-all-heavy", action="store_true", help="Use all heavy atoms instead of only CA atoms")
    parser.add_argument("--residue-range", type=str, default="6-97",
                       help="Range of residues to analyze in format 'start-end' (default: '6-97')")
    
    # RMSF calculation options
    parser.add_argument('--rmsf', dest='calculate_rmsf', action='store_true', default=True,
                       help='Calculate RMSF and normalized strains (default: enabled)')
    parser.add_argument('--no-rmsf', dest='calculate_rmsf', action='store_false',
                       help='Disable RMSF calculation')
    
    args = parser.parse_args()

    # Create output directories
    ensure_output_dirs(args.output)
    
    # Save run information
    save_run_info(args, args.output)
    
    # Parse residue range
    try:
        start_res, end_res = map(int, args.residue_range.split('-'))
        residue_numbers = list(range(start_res, end_res + 1))
    except ValueError:
        print(f"Error: Invalid residue range format. Expected 'start-end', got '{args.residue_range}'")
        sys.exit(1)

    # Check if dt is provided when begin or end is specified
    if (args.begin is not None or args.end is not None) and args.time_step is None:
        print("Error: Time step (-dt) must be provided when specifying begin (-b) or end (-e) times.")
        sys.exit(1)

    print("\n=== Starting Atomic Strain Analysis ===")
    print(f"Run started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {os.path.abspath(args.output)}")
    print(f"RMSF calculation: {'enabled' if args.calculate_rmsf else 'disabled'}")
    print("\n=== Loading Structures and Trajectories ===")

    # Load universes with trajectories
    try:
        ref = load_universe_with_trajectories(args.reference, args.ref_trajectory)
        print(f"Loaded reference structure: {args.reference}")
        if args.ref_trajectory:
            print(f"with trajectory files: {args.ref_trajectory}")
    except Exception as e:
        print(f"Error loading reference files: {str(e)}")
        sys.exit(1)

    try:
        defm = load_universe_with_trajectories(args.deformed, args.def_trajectory)
        print(f"Loaded deformed structure: {args.deformed}")
        if args.def_trajectory:
            print(f"with trajectory files: {args.def_trajectory}")
    except Exception as e:
        print(f"Error loading deformed files: {str(e)}")
        sys.exit(1)

    print("\n=== Analysis Parameters ===")
    
    # Get total frames from both universes
    total_frames = get_total_frames(defm)
    if args.ref_trajectory:
        ref_frames = get_total_frames(ref)
        if ref_frames != total_frames:
            print(f"Warning: Reference ({ref_frames} frames) and deformed ({total_frames} frames) "
                  "trajectories have different lengths. Using shorter trajectory.")
            total_frames = min(total_frames, ref_frames)

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
                print(f"Warning: End time ({args.end} ns) is beyond the trajectory length ({total_time_ns:.2f} ns). "
                      "Using the last frame.")
                end_frame = total_frames
            else:
                end_frame = min(ns_to_frame(args.end, args.time_step), total_frames)

        print(f"Time range: {frame_to_ns(start_frame, args.time_step):.2f} to "
              f"{frame_to_ns(end_frame - 1, args.time_step):.2f} ns")
    else:
        start_frame = 0
        end_frame = total_frames

    n_frames = math.ceil((end_frame - start_frame) / args.stride)

    if n_frames <= 0:
        print("Error: No frames to analyze with the given parameters.")
        sys.exit(1)

    print(f"Total frames in trajectory: {total_frames}")
    print(f"Analyzing frames from {start_frame} to {end_frame} with stride {args.stride}")
    print(f"Total frames to be analyzed: {n_frames}")
    print(f"Analyzing residues {start_res} to {end_res}")
    print(f"Using {'all heavy atoms' if args.use_all_heavy else 'only CA atoms'}")

    print("\n=== Running Analysis ===")
    # Run the analysis with RMSF calculation parameter
    strain_analysis = StrainAnalysis(
        ref, defm, residue_numbers, args.output, args.min_neighbors, 
        n_frames, use_all_heavy=args.use_all_heavy,
        calculate_rmsf=args.calculate_rmsf  # Pass RMSF flag
    )
    strain_analysis.run(start=start_frame, stop=end_frame, stride=args.stride)

    print("\n=== Creating Visualizations ===")
    try:
        # Use the saved final arrays instead of the deleted memory-mapped arrays
        visualize_strains(
            strain_analysis.results.atom_info,
            strain_analysis.results.final_shear_strains,
            strain_analysis.results.final_principal_strains,
            args.output,
            rmsf=strain_analysis.results.rmsf,
            norm_avg_shear_strains=strain_analysis.results.norm_avg_shear_strains,
            norm_avg_principal_strains=strain_analysis.results.norm_avg_principal_strains
        )
    except Exception as e:
        print(f"Warning: Error during visualization: {str(e)}")
        print("Analysis results were saved, but visualization failed.")

    print("\n=== Analysis Complete ===")
    print(f"Run completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved in: {os.path.abspath(args.output)}")
    print(f"Run information saved in: {os.path.join(os.path.abspath(args.output), 'run_info.json')}")
    print("Command to reproduce this analysis saved in: "
          f"{os.path.join(os.path.abspath(args.output), 'run_command.txt')}")
    
    if args.calculate_rmsf:
        print("\nRMSF-normalized strains have been calculated.")
        print("These account for the natural flexibility of each atom.")

if __name__ == "__main__":
    main()