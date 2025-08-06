import argparse
import math
import sys

import traceback

from util.RobotInterface import RobotInterface
from util.Utility import TrajParams, SystemIdParams

sys.path.append('..')

# Default constants
M_PI = math.pi
DEFAULT_TASK_MAX_DISP       = 0.10   # [m]
DEFAULT_TASK_MAX_VEL        = 0.900  # [m/s]
DEFAULT_TASK_MAX_ACC        = 1000.0 # [m/s^2]
DEFAULT_JOINT_MAX_DISP      = M_PI/18.0  # [rad]
DEFAULT_JOINT_MAX_VEL       = 18.0       # [rad/s]
DEFAULT_JOINT_MAX_ACC       = 2.0        # [rad/s^2]
DEFAULT_RUN_TIME            = 4.0        # [s]
DEFAULT_SYSID_ANGLES        = 8
DEFAULT_SYSID_RADII         = 4
DEFAULT_TASK_NUM_AXES       = 3
DEFAULT_HOME_SIGN           = 1
DEFAULT_SAMPLING_FREQUENCY  = 250  # Hz
DEFAULT_ROBOT_NAME          = "test"
DEFAULT_SYSID_TYPE          = "sine"  # 'bcb' or 'sine'
DEFAULT_CONFIG              = "joint"
DEFAULT_FIRST_POSE          = 0

def create_parser():
    parser = argparse.ArgumentParser(
        description="Run real-time joint position control for system identification."
    )
    # positional args
    parser.add_argument('robot_ip',
        help="Robot IP address")
    parser.add_argument('local_ip',
        help="Local IP address for control")

    # identification parameters
    parser.add_argument('--name', dest='robot_name', default=DEFAULT_ROBOT_NAME,
        help="Robot name (default: %(default)s)")
    parser.add_argument('--id', dest='robot_id', default='',
        help="Robot ID if required (default: blank)")
    parser.add_argument('--type', dest='sysid_type',
        choices=['bcb','sine'], default=DEFAULT_SYSID_TYPE,
        help="System ID type: 'bcb' or 'sine' (default: %(default)s)")
    parser.add_argument('--freq', dest='samp_freq', type=int,
        default=DEFAULT_SAMPLING_FREQUENCY,
        help="Sampling frequency (20–250 Hz, default: %(default)s)")
    parser.add_argument('--ctrl', dest='ctrl_config',
        choices=['joint','task'], default=DEFAULT_CONFIG,
        help="Control mode: 'joint' or 'task' (default: %(default)s)")

    # trajectory limits (None means use dynamic default later)
    parser.add_argument('--mdisp', dest='max_disp', type=float,
        help="Max displacement for trajectory (m or rad)")
    parser.add_argument('--mvel', dest='max_vel', type=float,
        help="Max velocity for trajectory (m/s or rad/s)")
    parser.add_argument('--macc', dest='max_acc', type=float,
        help="Max acceleration for trajectory (m/s^2 or rad/s^2)")

    # scan parameters
    parser.add_argument('--runtime', dest='runtime', type=float,
        default=DEFAULT_RUN_TIME,
        help="Runtime per pose in seconds (default: %(default)s)")
    parser.add_argument('--nv', dest='num_angles', type=int,
        default=DEFAULT_SYSID_ANGLES,
        help="Number of angles to sweep (default: %(default)s)")
    parser.add_argument('--nr', dest='num_radii', type=int,
        default=DEFAULT_SYSID_RADII,
        help="Number of radii to sweep (default: %(default)s)")
    parser.add_argument('--axes', dest='num_axes', type=int,
        help="Number of axes to command")
    parser.add_argument('--home', dest='home_sign', type=int, choices=[-1,1],
        default=DEFAULT_HOME_SIGN,
        help="Shoulder home direction -1 or 1 (default: %(default)s)")
    parser.add_argument('--first-pose', dest='start_pose', type=int,
        default=DEFAULT_FIRST_POSE,
        help="Index of the first pose (default: %(default)s)")

    return parser

def validate_args(args, parser):
    # sampling frequency must be in range
    if not 20 <= args.samp_freq <= 250:
        parser.error("--freq must be between 20 and 250 Hz")

    # if user specified axes, it must be ≥1
    if args.num_axes is not None and args.num_axes < 1:
        parser.error("--axes must be a positive integer")

def set_default_limits(args):
    # set default limits based on control mode
    if args.ctrl_config == 'task':
        args.max_disp = args.max_disp or DEFAULT_TASK_MAX_DISP
        args.max_vel = args.max_vel or DEFAULT_TASK_MAX_VEL
        args.max_acc = args.max_acc or DEFAULT_TASK_MAX_ACC
    else:  # joint control
        args.max_disp = args.max_disp or DEFAULT_JOINT_MAX_DISP
        args.max_vel = args.max_vel or DEFAULT_JOINT_MAX_VEL
        args.max_acc = args.max_acc or DEFAULT_JOINT_MAX_ACC

def main():
    parser = create_parser()
    args = parser.parse_args() # dictionary of arguments

    # validate arguments
    validate_args(args, parser)

    # set default limits if not specified
    set_default_limits(args)

    try:
        # Robot initialization and command execution
        robot_interface = RobotInterface(robot_name=args.robot_name, 
                                         robot_ip=args.robot_ip, 
                                         local_ip=args.local_ip,
                                         robot_id=args.robot_id)
        
        tParams = TrajParams(configuration=args.ctrl_config, 
                             max_displacement=args.max_disp, 
                             max_velocity=args.max_vel, 
                             max_acceleration=args.max_acc, 
                             sysid_type=args.sysid_type, 
                             single_pt_run_time=args.runtime)
        
        sysIdParams = SystemIdParams(nV=args.num_angles, 
                                     nR=args.num_radii)
        
        Ts = 1.0 / args.samp_freq  # sampling time in seconds
        
        num_joints = robot_interface.robot.num_joints

        # Decide how many axes to command
        if args.num_axes is not None:
            num_axes = args.num_axes
        else:
            # default task vs joint
            num_axes = (DEFAULT_TASK_NUM_AXES
                        if args.ctrl_config == 'task'
                        else num_joints)
            
        robot_interface.robot.move_home(home_sign=args.home_sign)

        # Command trajectory
        robot_interface.robot.command_robot(tParams, sysIdParams, Ts, num_axes, start_pose=args.start_pose)
        
    except Exception as e:
        # Print exception error message
        traceback.print_exc()

if __name__ == "__main__":
    main()