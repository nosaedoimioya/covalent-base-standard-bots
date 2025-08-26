# src/robots/Standard.py
# Author: Nosa Edoimioya
# Description: Specific code to run system identification on the R01 Robot.
# Version: 0.1

import os, time
import torch
import numpy as np
from standardbots import StandardBotsRobot, models
from typing import List, Tuple, Dict, Deque

from util.Utility import DataRecorder, get_polar_coordinates, polar_to_cartesian, store_recorder_data_in_csv, normalized_interpolation, transpose_list, cartesian_to_polar
from util.Robot import Robot, TrajParams, SystemIdParams, Trajectory
from util.RobotDynamics import Dynamics

from build.src.cpp.identification.MapFitter import ModelLoader

from control.python.BaseShaper import BaseShaper

from typing import TYPE_CHECKING

# ROS specific imports are loaded lazily so that the file can be used on
# systems without ROS installed.  The classes are only required when the ROS
# functionality is used.
if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    from robots.StandardRosManager import (
        Node,
    )

BOT_ID = "bot_0sapi_D9lfDQKWSUUaXogOBejf" # Change to your robot's ID
URDF_PATH = "urdf/modelone.urdf"
ROBOT_MAX_FREQ = 200 # Hz

# Home position of the robot
HOME_SHOULDER_ANGLE = np.pi/2 # [rad/s]
HOME_XYZ = [1.28989, 0.36866, 0.171]
HOME_QUAT = [0.499, 0.499, 0.499, 0.499]
HOME_JOINTS = [0.0, np.pi/2, 0.0, 0.0, 0.0, 0.0]

HTTP_HEADER = 'http://'

class StandardRobot(Robot):
    def __init__(self, robot_id: str, api_token: str):
        super().__init__("Standard Robot")  # Example: Standard robot
        # Initialize Robot attributes
        self.models = models
        self.recorder = DataRecorder()

        # Initialize URDF location
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        self.rel_path = os.path.join(self.module_dir, URDF_PATH)
        self._urdf_path = os.path.abspath(self.rel_path)
        print(f"URDF Path: {self._urdf_path}")

        # Dynamic model loaded flag
        self.model_is_loaded = False

        # Check if robot is in simulation mode
        self._in_sim_mode = (robot_id == "sim")

        # Load robot model from URDF
        if not self.model_is_loaded:
            self.initialize_model_from_urdf()
        
        try:
            if robot_id != "sim":
                # Instantiate robot interface
                self.robot = StandardBotsRobot(
                    url=HTTP_HEADER + robot_id,
                    token=api_token,
                    robot_kind=StandardBotsRobot.RobotKind.Live,
                )
                # Enable ROS control
                with self.robot.connection():
                    ## Set teleoperation/ROS control state
                    self.robot.ros.control.update_ros_control_state(
                        models.ROSControlUpdateRequest(
                            action=models.ROSControlStateEnum.Enabled,
                            # to disable: action=models.ROSControlStateEnum.Disabled,
                        )
                    )

                    # Get teleoperation state
                    self.state = self.robot.ros.status.get_ros_control_state().ok()
                    # Enable the robot, make sure the E-stop is released before enabling
                    print("Enabling live robot...")

                # Unbrake the robot if not operational
                self.robot.movement.brakes.unbrake().ok()

                # Set ID for ROS robot
                self.id = BOT_ID
            else:
                # Instantiate robot interface
                self.robot = StandardBotsRobot(
                    url=HTTP_HEADER + robot_id,
                    token=api_token,
                    robot_kind=StandardBotsRobot.RobotKind.Simulated,
                )
                self.state = None
                print("Enabling simulated robot...")
                    

            self.position = self.robot.movement.position.get_arm_position().ok()

        except Exception as e:
            # Print exception error message
            raise RuntimeError(f"Error getting {robot_id} operational: {str(e)}")
        
        self.num_joints = len(self.__get_joint_positions())
        self.pose_length = len(self.__get_tcp_pose())

    @property
    def in_sim_mode(self) -> bool:
        return self._in_sim_mode
    
    @property
    def urdf_path(self) -> str:
        return self._urdf_path

    def spin_thread(self, node: "Node"):
        """Spin a ROS node in a background thread."""
        from robots.StandardRosManager import rclpy
        rclpy.spin(node=node)
     
    def initialize_model_from_urdf(self):
        print(f"Loading robot model from URDF path: {self.urdf_path}")

        # Load the robot model from the URDF file
        self.model = Dynamics(robot_directory=self.module_dir,
                              urdf_file=self.urdf_path, 
                              initialize=True) 
        self.model_is_loaded = True

        print("Robot model loaded successfully.")

    def move_home(self, home_sign: int = 1, joint_move: bool = True) -> None:
        print('Moving to home position...')
        if joint_move:
            self.move_home_joint(home_sign=home_sign)
        else:
            self.move_home_pose()
        print("Home position reached.")

        # Store the home x-, y-, and z-axis (and r,p,y) parameters of the robot
        self.robot_home = self.__get_tcp_pose()
        self.home_xyz = np.array(self.robot_home, dtype=np.float32)[0:3]

        self.max_reach = max([abs(self.robot_home[0]),
                              abs(self.robot_home[1]),
                              abs(self.robot_home[2])])
        self.initial_height = self.home_xyz[2]

        depth_axis = np.where(np.logical_and(np.not_equal(np.abs(self.home_xyz), self.initial_height),
                                              np.not_equal(np.abs(self.home_xyz), self.max_reach)))[0][0]
        self.side_length = abs(self.robot_home[depth_axis])

    def move_home_pose(self):
        # R01 Robot home
        self.__move_to_pose(HOME_QUAT, HOME_XYZ)

    def move_home_joint(self, home_sign: int = 1):
        home_joints = [j for j in HOME_JOINTS]
        home_joints[1] = home_sign*home_joints[1]
        self.__move_to_joint(target_joint=home_joints)

    def command_robot(self, t_params: TrajParams, s_params: SystemIdParams, Ts: float, axes_to_command: int,
                      start_pose: int = 0):
        # check to make sure the sampling frequency is not larger than the max sampling frequency
        if (ROBOT_MAX_FREQ < (1/Ts)):
            raise RuntimeError(f"The sample time exceeds the maximum Standard Bot Robot\
                                sampling frequency of {ROBOT_MAX_FREQ} Hz.")
        
        # Set the mode and initial position of the robot based on the configuration given
        in_task_space = (t_params.configuration == "task")

        # Set the system id type
        sine_sweep_mode = (t_params.sysid_type == "sine")
        
        # Initialize parameters for robot trajectory generation and experiments
        # ==============================================================================================
        # Get the robot's home xyz position

        # Outstretched length of the robot
        # Expected: x-axis length/position of TCP. Depends on the home position of the robot.
        reach_axis = np.where(np.abs(self.home_xyz) == self.max_reach)[0][0]
        reach_sign = np.sign(self.robot_home[reach_axis])
        
        # Initial height above base frame of the base joint
        height_axis = np.where(np.abs(self.home_xyz) == self.initial_height)[0][0]
        height_sign = np.sign(self.robot_home[height_axis])

        # Get the depth in the axis location opposit of the max_reach of the robot
        depth_axis = np.where(np.logical_and(np.not_equal(np.abs(self.home_xyz), self.initial_height), 
                                             np.not_equal(np.abs(self.home_xyz), self.max_reach)))[0][0]
        depth_sign = np.sign(self.robot_home[depth_axis])

        # Generate V (angles) and R (radius) positions in base frame
        V, R = get_polar_coordinates(s_params.nV, s_params.nR, self.max_reach)

        # Get starting joint positions in case we don't have them yet - need base joint angle
        # for polar to cartesian computation
        initial_q = self.__get_joint_positions()
        base_angle = initial_q[0]

        # Initialize system ID variables for experiment
        # ==============================================================================================
        # Current x-, y-, and z-axis (and r,p,y) location of the robot
        current_pose = self.__get_tcp_pose()

        # Create a trajectory with the trajectory and system parameters
        trajectory = Trajectory(t_params, s_params)

        # Keep track of the number of runs (positions visited) and use to store values
        run_index = 0

        # Keep track of the inertia diagonals to store on each run
        current_mass_diagonals = np.zeros((self.num_joints))

        # Run system identification on robot through all positions
        # ==============================================================================================
        # Loop through angles and radii
        for i in range(s_params.nV):
            for j in range(s_params.nR):
                # 1) Skip until we reach start_pose
                if run_index < start_pose:
                    run_index += 1
                    continue
                
                # Determine length, width, and height positions from polar coordinates
                v = V[i]
                r = R[j]
                l, d, h = polar_to_cartesian(v,r,self.side_length,self.initial_height,base_angle)
                
                new_xyz = np.zeros(3)
                new_xyz[reach_axis] = l * reach_sign
                new_xyz[depth_axis] = d * depth_sign
                new_xyz[height_axis] = h * height_sign

                for move_axis in range(axes_to_command):
                    # Clear the recorder for starting a new position (run index)
                    self.__reset_recorder()

                    # Move to the initial position with point-to-point inverse kinematics
                    self.move_point_to_point_xyz(current_pose, new_xyz.tolist())

                    # Find the current joint positions after the robot moves
                    current_q = self.__get_joint_positions()

                    # Find the current mass matrix and update current mass diagonals
                    # self.model.update_model(current_q)
                    current_mass_matrix = self.model.get_mass_matrix(joint_angles=current_q) # numpy matrix

                    for joint_num in range(self.num_joints):
                        current_mass_diagonals[joint_num] = current_mass_matrix[joint_num][joint_num]
                    
                    # Generate system ID trajectory
                    if in_task_space:
                        if sine_sweep_mode:
                            trajectory.generate_sine_sweep_trajectory(new_xyz, move_axis, Ts)
                        else:
                            print("Bang-coast-bang system id is not implemented in this version")
                    else:
                        if sine_sweep_mode:
                            trajectory.generate_sine_sweep_trajectory(current_q, move_axis, Ts)
                        else:
                            print("Bang-coast-bang system id is not implemented in this version")


                    # Run trajectory on robot
                    # =============================================================================
                    # Start real-time joint control with ROS
                    self.rt_periodic_task(Ts, trajectory)

                    # Record static data
                    # ================================================================================
                    # Mass data
                    self.recorder.outputMassDiagonals = current_mass_diagonals # list

                    # End indices data
                    self.recorder.endIndices = trajectory.endIndices.copy()

                    # R and V
                    self.recorder.inputV.append(v)
                    self.recorder.inputR.append(r)

                    # Save recorder data in CSVs
                    store_recorder_data_in_csv(self.recorder, run_index, move_axis)

                # Increment run index
                run_index += 1

    def move_point_to_point_xyz(self, current_pose: List[float], target_xyz: List[float]) -> None:
        # Check if the orientation is included in the current pose
        if len(current_pose) < self.pose_length:
            raise RuntimeError("Not enough elements in currentPose to get the orientation \
                                of the current pose.")
        
        # Get quaternion from current pose
        target_quat = current_pose[3:]
        
        # Move to pose
        self.__move_to_pose(target_quat, target_xyz)

    def rt_periodic_task(self, Ts: float, trajectory: Trajectory) -> None:
        
        # Collect trajectory data
        position_data = trajectory.positionTrajectory # list
        velocity_data = trajectory.velocityTrajectory # list
        acceleration_data = trajectory.accelerationTrajectory # list
        time_data = trajectory.trajectoryTime.tolist() # np array -> list

        # Publish joint positions to the robot and record data
        data_log = self.__ros_publish_joint_positions(time_data=time_data,
                                                      position_stream=position_data,
                                                      velocity_stream=velocity_data,
                                                      acceleration_stream=acceleration_data,
                                                      Ts=Ts)
        # Reset recorder for storing data
        self.__reset_recorder()
       
        # Data post processing
        while data_log: # {'cmd_time','input_positions','output_positions','velocities','efforts','imu_time','linear_acceleration','angular_velocity','orientation'}
            log_entry = data_log.popleft()
            self.process_motion_data(entry=log_entry)

    def __ros_publish_joint_positions(self, time_data: List, position_stream: List, 
                                      velocity_stream: List = [], acceleration_stream: List = [],
                                      Ts: float = None) -> Deque[Tuple]:
        if Ts is None: Ts = self.Ts

        with self.robot.connection():
            ## Set teleoperation/ROS control state
            self.robot.ros.control.update_ros_control_state(
                models.ROSControlUpdateRequest(
                    action=models.ROSControlStateEnum.Enabled,
                )
            )
        
        from robots.StandardRosManager import (
            JointTrajectoryController,
            rclpy,
            threading,
        )

        rclpy.init()

        publish_complete_event = threading.Event()
        
        joint_controller = JointTrajectoryController(
            self.id,
            Ts,
            time_data,
            position_stream,
            velocity_stream,
            acceleration_stream,
            publish_complete_event
        )
        
        thread = threading.Thread(target=self.spin_thread, args=(joint_controller, ), daemon=True)
        thread.start()

        # wait until trajectory publishing finishes
        publish_complete_event.wait()
        rclpy.shutdown()
        thread.join()

        # Store data for return
        data_log = joint_controller.aligned_log
        joint_controller.destroy_node()

        return data_log # {'cmd_time','input_positions','output_positions','velocities',
                         # 'efforts','imu_time','linear_acceleration','angular_velocity','orientation'}

    def process_motion_data(self, entry: Dict) -> None:
        # entry: {'cmd_time','input_positions','output_positions',
        #       'velocities','efforts','imu_time','linear_acceleration',
        #       'angular_velocity','orientation'}
        self.recorder.servoTime.append(entry["cmd_time"])
        self.recorder.inputJointPositions.append(list(entry["input_positions"]))
        self.recorder.outputJointPositions.append(entry["output_positions"].tolist())
        self.recorder.outputCurrents.append(entry["efforts"].tolist())
        self.recorder.imuTime.append(entry["imu_time"])
        self.recorder.outputTcpAccelerations.append(entry["linear_acceleration"].tolist() + \
                                                        entry["angular_velocity"].tolist())
        self.recorder.quaternionTime.append(entry["imu_time"])
        self.recorder.quaternion.append(entry["orientation"].tolist())

    def execute_covalent_base_ptp_test(self, configs: List[List[float]], model_filename: str, 
                                        dwell: float = 1.0, num_axes: int = 3, Ts: float = 1/ROBOT_MAX_FREQ) -> None:
        '''
        Create and execute point-to-point tests for uncompensated and compensated trajectories.
        Parameters:
            - configs: List of joint configurations to move between. Each configuration is a list of joint angles.
            - model_filename: Filename of the trained NN model to use for compensation.
            - dwell: Time to wait at each configuration before starting the motion (in seconds).
            - num_axes: Number of axes to compensate (default is 3).

        '''
        # For each point in configs, run a PTP motion - uncompensated & compensated
        for i, (start, end) in enumerate(configs):

            # Generate the trajectories from point to point
            position, velocity, acceleration, time_list = self.__plan_joints_path(
                                                                    start_angles=start,
                                                                    goal_angles=end, 
                                                                    Ts=Ts,
                                                                    dwell=dwell)
            
            # Move to the start point
            self.__move_to_joint(target_joint=tuple(start))

            # wait for dynamics to settle
            time.sleep(dwell)

            # Run the uncompensated path and store data
            print(f"Running uncompensated path for config {i}")
            data_log = self.__ros_publish_joint_positions(
                time_data=time_list,
                position_stream=position,
                velocity_stream=velocity,
                acceleration_stream=acceleration,
                Ts=Ts
            )

            # Reset recorder for storing data
            self.__reset_recorder()
            
            # Get v, r, and inertias at the end point
            v, r, inertias = self.__compute_nn_inputs()

            # print(f"v: {v} rad, r: {r} m")

            # Data post processing
            while data_log: # {'cmd_time','input_positions','output_positions','velocities','efforts','imu_time','linear_acceleration','angular_velocity','orientation'}
                log_entry = data_log.popleft()
                self.process_motion_data(entry=log_entry)

            # Record static data
            # ===============================================
            # Mass data
            self.recorder.outputMassDiagonals = inertias # list

            # End indices data
            self.recorder.endIndices = [len(position)]

            # R and V
            self.recorder.inputV.append(v)
            self.recorder.inputR.append(r)

            # TO-DO (nosed): split up recording function for ptp motions vs sine sweep
            # Record/Store data
            filename = f"ptp_config{i}_uncompensated"
            store_recorder_data_in_csv(self.recorder, run_index=0, move_axis=0, filename=filename)

            
            frf_params = self.__compute_frf_params(joint_positions=np.array(position), model_filename=model_filename, 
                                                   num_axes=num_axes)

            # Compute shaped trajectory to send to robot
            shaped_position, shaped_velocity, \
                shaped_acceleration, shaped_time_list = self.__shape_joints_path(frf_params=frf_params,
                                                                            position_unshaped=np.array(position),
                                                                            velocity_unshaped=np.array(velocity),
                                                                            acceleration_unshaped=np.array(acceleration),
                                                                            time_unshaped=np.array(time_list),
                                                                            num_axes=num_axes, Ts=Ts)
            
            # Prep to run the compensated trajectory
            # ============================================
            # Move back to the start point
            self.__move_to_joint(target_joint=tuple(start))

            # wait for dynamics to settle
            time.sleep(dwell)

            # Run the compensated path and store data
            data_log = self.__ros_publish_joint_positions(
                time_data=shaped_time_list,
                position_stream=shaped_position,
                velocity_stream=shaped_velocity,
                acceleration_stream=shaped_acceleration,
                Ts=self.Ts
            )

            # Reset recorder for storing data
            self.__reset_recorder()

            # Data post processing
            while data_log: # {'cmd_time','input_positions','output_positions','velocities','efforts','imu_time','linear_acceleration','angular_velocity','orientation'}
                log_entry = data_log.popleft()
                self.process_motion_data(entry=log_entry)

            # Static data
            # ===============================================
            # Mass data
            self.recorder.outputMassDiagonals = inertias # list

            # End indices data
            self.recorder.endIndices = [len(position)]

            # R and V
            self.recorder.inputV.append(v)
            self.recorder.inputR.append(r)

            # Record/store shaped data
            filename = f"ptp_config{i}_compensated"
            store_recorder_data_in_csv(self.recorder, run_index=0, move_axis=0, filename=filename)

        # Return home
        self.move_home(joint_move=True)
    
    def __plan_joints_path(self, start_angles: List, goal_angles: List,
                           t_params: TrajParams = None,
                           s_params: SystemIdParams = None,
                           Ts: float = 1/ROBOT_MAX_FREQ, dwell: float = 0.0) -> Tuple[list, list, list, list]:
        # Initialize trajectory generator
        trajectoryGen = Trajectory(t_params=t_params, s_params=s_params)

        joint_trajectories = []
        joint_velocities = []
        joint_accelerations = []
        time_trajectories = []
        trajectoryGen = Trajectory(t_params=t_params, s_params=s_params)
        max_traj_pts = 0
        for (current_angle, target_angle) in zip(start_angles, goal_angles):
            direction = np.sign(target_angle - current_angle)
            distance = abs(target_angle - current_angle)
            qj, qj_dot, qj_ddot, tj = trajectoryGen.point_to_point_motion_jerk_limit(feedrate=t_params.max_velocity, 
                                                                                    acc_limit=t_params.max_acceleration,
                                                                                    dec_limit=-1*t_params.max_acceleration,
                                                                                    displacement=distance,
                                                                                    Ts=Ts, dwell=dwell)
            
            qj = current_angle + direction * qj
            joint_trajectories.append(qj)
            joint_velocities.append(direction*qj_dot)
            joint_accelerations.append(direction*qj_ddot)
            time_trajectories.append(tj)

            # Update the maximum trajectory points
            if len(qj) > max_traj_pts: max_traj_pts = len(qj)

        # Extend trajectories to the longest trajectory length by normalized interpolation
        # ===============================================================
        original_lists = [joint_trajectories, joint_velocities, joint_accelerations]
        interpolated_lists, path_time = normalized_interpolation(time_list=time_trajectories, 
                                                                 to_interp_lists=original_lists)
        joint_trajectories, joint_velocities, joint_accelerations = interpolated_lists

        # Transpose joint trajectories and time to match format
        return transpose_list(joint_trajectories), transpose_list(joint_velocities), transpose_list(joint_accelerations), path_time.tolist()
    
    
    def __move_to_pose(self, target_quat: List[float], target_xyz: List[float]) -> None:
        quatx, quaty, quatz, quatw = target_quat
        move_quat = models.Orientation(
                        kind=models.OrientationKindEnum.Quaternion,
                        quaternion=models.Quaternion(x=quatx, 
                                                     y=quaty, 
                                                     z=quatz, 
                                                     w=quatw
                                                     ),
                    )
        x, y, z = target_xyz
        move_xyz = models.Position(
                        unit_kind=models.LinearUnitKind.Meters,
                        x=x, 
                        y=y, 
                        z=z
                    )
        
        update_request = models.ArmPositionUpdateRequest(
            kind=models.ArmPositionUpdateRequestKindEnum.TooltipPosition,
            tooltip_position=models.PositionAndOrientation(
                position=move_xyz,
                orientation=move_quat)
        )
        
        response = self.robot.movement.position.set_arm_position(body=update_request).ok()
        return response
    
    def __move_to_joint(self, target_joint: Tuple[float]) -> None:
        update_request = models.ArmPositionUpdateRequest(
            kind=models.ArmPositionUpdateRequestKindEnum.JointRotation,
            joint_rotation=models.ArmJointRotations(
                joints=target_joint)
        )

        response = self.robot.movement.position.set_arm_position(body=update_request).ok()
        return response

    def __get_tcp_pose(self) -> List: 
        # Get positions
        position = self.robot.movement.position.get_arm_position().ok()
        pose = [position.tooltip_position.position.x,
                position.tooltip_position.position.y,
                position.tooltip_position.position.z,
                position.tooltip_position.orientation.quaternion.x,
                position.tooltip_position.orientation.quaternion.y,
                position.tooltip_position.orientation.quaternion.z,
                position.tooltip_position.orientation.quaternion.w]
        
        # Return tooltip pose as a list
        return pose

    def __get_joint_positions(self) -> List:
        # Get positions
        position = self.robot.movement.position.get_arm_position().ok()
        # Return joint positions as a list
        return list(position.joint_rotations)

    def __reset_recorder(self):
        self.recorder.inputJointPositions = []
        self.recorder.outputJointPositions = []
        self.recorder.outputCurrents = []
        self.recorder.outputTcpAccelerations = []
        self.recorder.servoTime = []
        self.recorder.imuTime = []
        self.recorder.quaternionTime = []
        self.recorder.quaternion = []

        self.recorder.outputMassDiagonals = []
        self.recorder.endIndices = []
        self.recorder.inputV = []
        self.recorder.inputR = []

    def __compute_frf_params(self, joint_positions: List[List[float]], model_filename: str, 
                             num_axes: int = 3, prob_thresh: float = 0.5) -> List[float]:
        # Create a [len(joint_positions)] array of wn, zeta for each axis
        frf_params = [] # list of np.arrays

        # Get the neural network model
        nn_models = ModelLoader.load(directory=model_filename, axes=num_axes, input_features=3, hidden=[64, 64])

        # For each configuration, compute the wn, zeta
        for i, joint_position in enumerate(joint_positions):
            # Use transformation matrix to convert joint positions to cartesian positions
            cartesian_position = self.model.get_forward_kinematics(joint_angles=joint_position)

            # Get the furthest axis from the base frame
            max_reach = max([abs(cartesian_position[0]),abs(cartesian_position[1]),abs(cartesian_position[2])])
            reach_axis = np.where(np.abs(cartesian_position) == max_reach)[0][0]

            # height above base frame of the base joint
            height = cartesian_position[2] # always the z-axis

            # Get the depth in the axis location orthogonal to the max_reach of the robot
            depth_axis = np.where(np.logical_and(np.not_equal(np.abs(cartesian_position), height), np.not_equal(np.abs(cartesian_position), max_reach)))[0][0]

            # Compute v and r
            v, r = cartesian_to_polar(long=cartesian_position[reach_axis], 
                                      width=cartesian_position[depth_axis], 
                                      height=height,
                                      side_arm=self.side_length, 
                                      base_height=self.initial_height)

            # Use current joint position to compute inertias
            mass_matrix = self.model.get_mass_matrix(joint_angles=joint_position) # numpy array

            # Compute the inertia matrix
            mass_diagonals = np.zeros((self.num_joints,))
            for joint_num in range(self.num_joints):
                mass_diagonals[joint_num] = mass_matrix[joint_num][joint_num]

            # Get the wn and zeta from the neural network model
            frf_params_axis = []
            for k in range(num_axes):
                # -------- features (match what C++ trained on) --------------------------
                X_raw = np.array([[                     # shape [1, 3]
                    v * 180.0 / np.pi,                  # V in degrees
                    r * 1000.0,                         # R in mm
                    mass_diagonals[k],                        # kg·m²
                ]], dtype=np.float32)

                X = torch.from_numpy(X_raw)

                # -------- inference ------------------------------------------------------
                with torch.no_grad():
                    p_cls, p_reg = nn_models.infer(k, X)   # (N×1, N×4) here N=1

                prob_second = float(p_cls.squeeze().item())
                wn1, z1_log, wn2, z2_log = p_reg.squeeze(0).tolist()
                z1, z2 = np.exp(z1_log), np.exp(z2_log)
                wn1_rad, wn2_rad = wn1 * 2*np.pi, wn2 * 2*np.pi

                # -------- decide shaper order -------------------------------------------
                two_mode = bool(prob_second > prob_thresh)
                if two_mode:
                    # Append both modes to the frf_params
                    frf_params_axis.append([wn1_rad, z1])
                    frf_params_axis.append([wn2_rad, z2])
                else:
                    # Append only the first mode to the frf_params
                    frf_params_axis.append([wn1_rad, z1])
                
            frf_params.append(np.array(frf_params_axis))

        return frf_params

    def __shape_joints_path(self,
                            frf_params: List[np.ndarray],
                            position_unshaped: np.ndarray,
                            velocity_unshaped: np.ndarray,
                            acceleration_unshaped: np.ndarray,
                            time_unshaped: np.ndarray,
                            num_axes: int, Ts: float,
                            ) -> Tuple[list, list, list, list]:
        """
        For each compensated axis:
        1.   Build the shaped trajectory with BaseShaper.
        """
        pos_shaped, vel_shaped, acc_shaped, time_lists = [], [], [], []

        for k in range(num_axes):
            shaper = BaseShaper(Ts=Ts)
            shaped_pos = shaper.shape_trajectory(x=position_unshaped[:, k], varying_params=frf_params[k])
            t_mod = np.linspace(0, len(shaped_pos)-1)*Ts
            shaped_vel = np.gradient(shaped_pos, t_mod, axis=0, edge_order=2)
            shaped_acc = np.gradient(shaped_vel, t_mod, axis=0, edge_order=2)

            pos_shaped.append(shaped_pos)
            vel_shaped.append(shaped_vel)
            acc_shaped.append(shaped_acc)
            time_lists.append(t_mod)

        # ------- pass-through for uncompensated joints ------------------------
        for k in range(num_axes, self.num_joints):
            pos_shaped.append(position_unshaped[:, k])
            vel_shaped.append(velocity_unshaped[:, k])
            acc_shaped.append(acceleration_unshaped[:, k])
            time_lists.append(time_unshaped)

        # ------- reconcile potentially different time vectors ----------------------------
        interp_lists, t_sync = normalized_interpolation(time_list=time_lists,
                                                        to_interp_lists=[pos_shaped, vel_shaped, acc_shaped])

        P, V, A = interp_lists
        return transpose_list(P), transpose_list(V), transpose_list(A), t_sync.tolist()