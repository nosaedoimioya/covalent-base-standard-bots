# src/identification/processCalibrationData.py
# This script reads calibration data from system identification of a robot arm
# and processes it for frequency response function (FRF) evaluation.

import os
import argparse
import math
from src.archive_for_safety.SineSweepReader import SineSweepReader
from src.archive_for_safety.MapFitter import MapFitter

# Constants
M_PI                         = math.pi
DEFAULT_ROBOT_NAME           = "test"
DEFAULT_FILE_FORMAT          = "csv"
DEFAULT_NUM_JOINTS           = 6
DEFAULT_MINFREQ              = 1       # [Hz]
DEFAULT_MAXFREQ              = 60      # [Hz]
DEFAULT_FREQSPACE            = 0.5     # [Hz]
DEFAULT_MDISP                = M_PI/36 # [rad]
DEFAULT_JOINT_MAX_VEL        = 18.0    # [rad/s]
DEFAULT_DWELL                = 0.0     # [s]
DEFAULT_TS                   = 0.004   # [s]
DEFAULT_SYSID_TYPE           = "sine"  # 'bcb' or 'sine'
DEFAULT_CONFIG               = "joint"
DEFAULT_JOINT_MAX_ACC        = 2.0     # [rad/s^2]
DEFAULT_SENSOR               = "ToolAcc"  # 'ToolAcc' or 'JointPos'
DEFAULT_SINE_CYCLES          = 5
DEFAULT_START_POSE           = 0
DEFAULT_MAX_MAP_POSES        = 12


def print_description():
    print(
        "This program loads sine sweep data from system identification\n"
        "of a robot arm and stores it for FRF evaluation.\n"
    )

def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load sysID sine-sweep data and produce calibration maps."
    )

    # required positionals
    p.add_argument('data_path',
        help="Path to data folder with sysID files (e.g. ../calibration/data/)")
    p.add_argument('poses', type=int,
        help="Number of poses robot visits (indexed from 0)")
    p.add_argument('axes', type=int,
        help="Number of commanded axes per pose (indexed from 0)")

    # optionals
    p.add_argument('--name',      dest='robot_name',
        default=DEFAULT_ROBOT_NAME,
        help=f"Robot name (default: {DEFAULT_ROBOT_NAME})")
    p.add_argument('--format',    dest='file_format',
        choices=['csv','npz','npy'],
        default=DEFAULT_FILE_FORMAT,
        help=f"Data file format (default: {DEFAULT_FILE_FORMAT})")
    p.add_argument('--numjoints', dest='num_joints', type=int,
        default=DEFAULT_NUM_JOINTS,
        help=f"Number of robot joints (default: {DEFAULT_NUM_JOINTS})")
    p.add_argument('--minfreq',   dest='min_freq', type=float,
        default=DEFAULT_MINFREQ,
        help=f"Min frequency [Hz] (default: {DEFAULT_MINFREQ})")
    p.add_argument('--maxfreq',   dest='max_freq', type=float,
        default=DEFAULT_MAXFREQ,
        help=f"Max frequency [Hz] (default: {DEFAULT_MAXFREQ})")
    p.add_argument('--freqspace', dest='freq_space', type=float,
        default=DEFAULT_FREQSPACE,
        help=f"Frequency spacing [Hz] (default: {DEFAULT_FREQSPACE})")
    p.add_argument('--mdisp',     dest='max_disp', type=float,
        default=DEFAULT_MDISP,
        help=f"Max sweep stroke [rad] (default: {DEFAULT_MDISP:.5f})")
    p.add_argument('--dwell',     dest='dwell', type=float,
        default=DEFAULT_DWELL,
        help=f"Post-sweep dwell [s] (default: {DEFAULT_DWELL})")
    p.add_argument('--timestep',  dest='Ts', type=float,
        default=DEFAULT_TS,
        help=f"Sampling time [s] (default: {DEFAULT_TS})")
    p.add_argument('--type',      dest='sysid_type',
        choices=['bcb','sine'],
        default=DEFAULT_SYSID_TYPE,
        help=f"SysID type (bcb|sine, default: {DEFAULT_SYSID_TYPE})")
    p.add_argument('--ctrl',      dest='ctrl_config',
        choices=['joint','task'],
        default=DEFAULT_CONFIG,
        help=f"Control mode (joint|task, default: {DEFAULT_CONFIG})")
    p.add_argument('--macc',      dest='max_acc', type=float,
        default=DEFAULT_JOINT_MAX_ACC,
        help=f"Max acceleration [rad/s^2] (default: {DEFAULT_JOINT_MAX_ACC})")
    p.add_argument('--mvel',     dest='max_vel', type=float,
        default=DEFAULT_JOINT_MAX_VEL,
        help=f"Max velocity [rad/s] (default: {DEFAULT_JOINT_MAX_VEL})")
    p.add_argument('--sine-cycles', dest='sine_cycles', type=int,
        default=DEFAULT_SINE_CYCLES,
        help=f"Number of sine cycles per pose (default: {DEFAULT_SINE_CYCLES})")
    p.add_argument('--sensor',    dest='sensor',
        choices=['ToolAcc','JointPos'],
        default=DEFAULT_SENSOR,
        help=f"Sensor type (ToolAcc|JointPos, default: {DEFAULT_SENSOR})")
    p.add_argument('--first-pose',dest='start_pose', type=int,
        default=DEFAULT_START_POSE,
        help=f"Starting pose index (default: {DEFAULT_START_POSE})")
    p.add_argument('--max-map-size', dest='max_map_size', type=int,
        default=DEFAULT_MAX_MAP_POSES,
        help=f"Max poses per map (default: {DEFAULT_MAX_MAP_POSES})")
    p.add_argument('--saved-maps', dest='saved_maps', action='store_true', 
                default=False,
                help="If set, will load existing calibration maps instead of generating new ones.")

    return p

def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.poses <= 0:
        parser.error("‘poses’ must be > 0")
    if args.axes <= 0:
        parser.error("‘axes’ must be > 0")
    if args.min_freq < 0 or args.max_freq < args.min_freq:
        parser.error("frequency range invalid (0 ≤ minfreq ≤ maxfreq)")
    if args.max_map_size <= 0:
        parser.error("‘max-map-size’ must be > 0")
    if args.max_map_size > args.poses:
        args.max_map_size = args.poses

def generate_calibration_maps(args: argparse.Namespace):
    stored_maps = []
    if args.sysid_type == 'bcb':
        print("Using BCB system identification type.")
        raise NotImplementedError("BCB system identification type is not implemented yet.")
    else:
        print("Using Sine system identification type.")
        # Create SineSweepReader instance
        data_reader = SineSweepReader(
                        data_folder=args.data_path,
                        num_poses=args.poses,
                        num_axes=args.axes,
                        robot_name=args.robot_name,
                        data_format=args.file_format,
                        num_joints=args.num_joints,
                        min_freq=args.min_freq,
                        max_freq=args.max_freq,
                        freq_space=args.freq_space,
                        max_disp=args.max_disp,
                        dwell=args.dwell,
                        Ts=args.Ts,
                        ctrl_config=args.ctrl_config,
                        max_acc=args.max_acc,
                        max_vel=args.max_vel,
                        sine_cycles=args.sine_cycles,
                        max_map_size=args.max_map_size
                    )
        # Get calibration maps
        stored_maps = data_reader.get_calibration_maps()

    return stored_maps

def load_calibration_maps(args: argparse.Namespace):
    '''
        Load existing calibration maps from the specified path.
        The maps are stored in self.map.
    '''
    start_pose = args.start_pose
    # Process calibration maps in batches
    # ===========================================
    # List to store the names of the saved calibration maps
    stored_calibration_maps = []
    
    # Break up the poses into storable map sizes
    num_runs = args.poses / args.max_map_size
    precision_tolerance = 1e-6
    if (num_runs % 1) > precision_tolerance:
        num_runs = int(num_runs) + 1
    else:
        num_runs = int(num_runs)

    for turns in range(num_runs):
        last_pose = (turns+1) * args.max_map_size if args.max_map_size > 0 else args.poses

        # Raise an error if the start pose is larger than the last pose
        if start_pose >= last_pose:
            if start_pose < args.poses:
                continue
            ValueError(f"Start pose, {start_pose}, cannot be larger than the last pose, {last_pose}")

        pickle_file = args.data_path + f'/{args.robot_name}_robot_calibration_map_lastPose{last_pose}_numAxes{args.axes}_startPose{start_pose}.pkl'

        # Check if the file exists
        if not os.path.exists(pickle_file):
            raise Exception(f"Calibration map file {pickle_file} doesn't exist. \
                            Remove the --saved-maps flag to generate new maps.")
        else:
            stored_calibration_maps.append(pickle_file)

            # Update the start pose
            start_pose = int(last_pose)

    return stored_calibration_maps

def get_model_directory(args: argparse.Namespace) -> str:
    
    model_file_prefix = f"../calibration/models/" + args.data_path.split('/')[-1]

    # Location to save data
    directory_name = model_file_prefix + f'/{args.robot_name}_robot_predictor_{args.ctrl_config}_numAxes{args.axes}'

    return directory_name        

def main():
    parser = create_parser()
    args   = parser.parse_args()
    validate_args(args, parser)

    print_description()
    if args.saved_maps:
        print("Loading existing calibration maps...")
        stored_map_names = load_calibration_maps(args)
        print(f"Loaded {len(stored_map_names)} calibration map(s).")
    else:
        print("Generating new calibration maps...")
        stored_map_names = generate_calibration_maps(args)

    # Generate neural network fits
    # ====================================
    map_fitter = MapFitter(map_names=stored_map_names,
                           num_positions=args.poses,
                           axes_commanded=args.axes,
                           num_joints=args.num_joints)
    
    # map_fitter.visualize_data()
    map_fitter.fit_shaper_neural_network_twohead()

    # Save fitted models
    directory = get_model_directory(args)
    map_fitter.save_models(directory=directory)
    

if __name__ == "__main__":
    main()

