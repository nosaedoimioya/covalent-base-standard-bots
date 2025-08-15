# src/identification/SineSweepReader.py

import os
from typing import Tuple, List
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

from util.Utility import TrajParams, StructuredCalibrationData, load_data_file
from util.RobotInterface import RobotInterface
from src.identification.MapGenerationDelete import CalibrationMap

# Constants
GRAVITY_ACC                  = 9.81    # [m/s^2]
NUM_CARTESIAN_AXES           = 3
NUM_QUATERNION_AXES          = 4

# Default data locations in the data file
DEFAULT_TIMESTAMP_LOCATION   = 2
DEFAULT_STATIC_ANGLE_LOCATION= 0
DEFAULT_STATIC_RADIUS_LOCATION=1


class SineSweepReader:
    def __init__(self, data_folder: str, num_poses: int, num_axes: int, 
                 robot_name: str, data_format: str, num_joints: int, 
                 min_freq: float, max_freq: float, freq_space: float, 
                 max_disp: float, dwell: float, Ts: float, ctrl_config: str, 
                 max_acc: float, max_vel: float, sine_cycles: int,
                 max_map_size: int):
        self.data_folder = data_folder
        self.num_poses = num_poses
        self.num_axes = num_axes
        self.robot_name = robot_name
        self.data_format = data_format
        self.num_joints = num_joints
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_space = freq_space
        self.max_disp = max_disp
        self.dwell = dwell
        self.Ts = Ts
        self.ctrl_config = ctrl_config
        self.max_acc = max_acc
        self.max_vel = max_vel
        self.sine_cycles = sine_cycles
        self.max_map_size = max_map_size


        self.robot_ip = "sim" # Always use the robot simulator for model generation

        # List to hold calibration maps
        self.calibration_maps = []

        # Create the RobotInterface instance to handle kinematics and dynamics
        self.robot_interface = RobotInterface(robot_name=self.robot_name,
                                             robot_ip=self.robot_ip)

    def get_calibration_maps(self):
        # This method processes the data and create calibration maps
        # based on the sine sweep data loaded.

        # Create Trajectory parameters
        traj_params = TrajParams(
            configuration=self.ctrl_config,
            max_displacement=self.max_disp,
            max_velocity=self.max_vel,
            max_acceleration=self.max_acc,
            sysid_type='sine',
            single_pt_run_time=self.dwell
        )

        num_runs = self.compute_num_maps()

        for passes in range(num_runs):
            last_pose = (passes+1) * self.max_map_size if self.max_map_size > 0 else self.num_poses
            start_pose = passes * self.max_map_size

            # Raise an error if the start pose is larger than the last pose
            if start_pose >= last_pose:
                if start_pose < self.num_poses:
                    continue
                ValueError(f"Start pose, {start_pose}, cannot be larger than the last pose, {last_pose}")

            # Create CalibrationMap object
            calibration_map = CalibrationMap(numPositions=(last_pose-start_pose), 
                                             axesCommanded=self.num_axes, 
                                             numJoints=self.num_joints)
            # Pickle file name for storing the map
            pickle_file = self.data_folder + \
                f'/{self.robot_name}_robot_calibration_map_lastPose{last_pose}_numAxes{self.num_axes}_startPose{start_pose}.pkl'
            
            # Loop through each pose and axis
            # ==========================================================
            for run_index in range(start_pose,last_pose):
                # Execute calibration map generation
                robot_data = self._structure_robot_data(pose_index=run_index, show_plots=False)

                # update CalibrationMap with the structured data
                calibration_map.generateCalibrationMap(robot_data=robot_data,traj_params=traj_params,
                                                       robot_model=self.robot_interface.robot.model,
                                                       input_Ts=self.Ts,output_Ts=self.Ts, 
                                                       max_freq_fit=self.max_freq,
                                                       gravity_comp=True,
                                                       shift_store_position=run_index-start_pose)
            # Store the calibration map to a file
            calibration_map.save_map(filename=pickle_file)

            # Delete map to free RAM
            del calibration_map

            # Store the map file's name in the list
            self.calibration_maps.append(pickle_file)

            # Update the start pose for the next run
            start_pose = int(last_pose)

        return self.calibration_maps
    
    def compute_num_maps(self) -> int:
        # Compute number of maps we need to store
        num_runs = self.num_poses / self.max_map_size
        precision_tolerance = 1e-6
        if (num_runs % 1) > precision_tolerance:
            num_runs = int(num_runs) + 1
        else:
            num_runs = int(num_runs)

        return num_runs
    
    def reset_calibration_maps(self):
        # Reset the list of calibration maps
        self.calibration_maps = []
    
    def _structure_robot_data(self, pose_index, show_plots: bool = False) -> StructuredCalibrationData:
        # This method will handle the generation & storage of the calibration map
        # to a file or database.

        # Set system identification parameters
        freq_range = np.arange(self.min_freq, self.max_freq + self.freq_space, self.freq_space).tolist()


        begin_end_points = [[] for _ in range(self.num_axes)] # Store the beginning and end points for each axis (at each frequency)
        frequency_list = [[] for _ in range(self.num_axes)] # Store the frequency data for sine waves
        time_in = [] # Store the time data
        x_in = [] # Store the x-axis position data
        y_in = [] # Store the y-axis position data
        z_in = [] # Store the z-axis position data
        joints = [] # Store the joint position data
        ax_out = [] # Store the x-axis acceleration data
        ay_out = [] # Store the y-axis acceleration data
        az_out = [] # Store the z-axis acceleration data 
        max_N_pts = 0 # Store the maximum number of points for each axis

        for commanded_axis in range(self.num_axes):
            time, command_position, _, _, _, \
                tcp_acceleration, _, _, _ = self.extract_dynamic_variables(pose=pose_index,
                                                                           axis=commanded_axis,)
            
            N_pts = time.shape[0]
            if N_pts > max_N_pts:
                max_N_pts = N_pts

            # Get the starting point
            initial_joint_position = command_position[0,:] # Assume radians

            # Store position and acceleration data for this axis
            x = np.zeros((N_pts,1))
            y = np.zeros((N_pts,1))
            z = np.zeros((N_pts,1))
            for i in range(N_pts):
                position, _ = self.robot_interface.robot.model.get_forward_kinematics(joint_angles=command_position[i,:].tolist())
                x[i] = position[0]
                y[i] = position[1]
                z[i] = position[2]
            x_in.append(x)
            y_in.append(y)
            z_in.append(z)
            joints.append(command_position) 
            ax_out.append(tcp_acceleration[:,0])
            ay_out.append(tcp_acceleration[:,1])
            az_out.append(tcp_acceleration[:,2])
            time_in.append(time)

            v, r, _, _ = self.extract_static_variables(pose=pose_index,
                                                       axis=commanded_axis,
                                                       num_joints=self.num_joints)
            
            # Correlation with the input and the sine waves to identify the beginning 
            # and end-points for each frequency
            INPUT_temp = np.copy(command_position[:,commanded_axis] - initial_joint_position[commanded_axis]) # temporary storage for the input
            f_start_init = 0 # f_start initial guess
            for freq in freq_range:
                sin_frequency = freq

                f_start, f_end = self.correlate_sine_wave(input_signal=INPUT_temp, 
                                                          frequency=sin_frequency,
                                                          plotting=show_plots)
                
                # Check if the start and end points are valid
                if f_start < 0 or f_end < 0 or f_start >= N_pts or f_end >= N_pts:
                    Warning(f"Invalid start or end point for frequency {freq} on axis {commanded_axis}. Check data.")
                    Warning(f"f_start: {f_start}, f_end: {f_end}, N_pts: {N_pts}")
                    if f_end > N_pts:
                        f_end = N_pts - 1
                    if f_start < 0:
                        f_start = f_start_init
                    continue
                
                f_start_init = f_start
                INPUT_temp[1:f_start_init] = 0 # zero out the used values (helps with the future correlation)

                # Store points for this axis to use for the data processing
                begin_end_points[commanded_axis].append([f_start,f_end])
                frequency_list[commanded_axis].append(freq)

        # Store structured robot data for this pose
        # =================================
        # Initialize data record
        robot_data = StructuredCalibrationData(num_input_points=max_N_pts,
                                               num_output_points=max_N_pts,
                                               num_input_vars=NUM_CARTESIAN_AXES+self.num_joints,
                                               num_output_vars=NUM_CARTESIAN_AXES,
                                               num_axes=self.num_axes,
                                               input_begin_end_indices=begin_end_points,
                                               output_begin_end_indices=begin_end_points,
                                               frequency_list=frequency_list,
                                            )   
        
        # Store the data
        robot_data = self.store_robot_data_in_vectors(robot_data=robot_data,
                                                      time_in=time_in,
                                                      x_in=x_in,
                                                      y_in=y_in,
                                                      z_in=z_in,
                                                      joints=joints,
                                                      ax_out=ax_out,
                                                      ay_out=ay_out,
                                                      az_out=az_out
                                                    )
        
        # Store the angle and radius from base as the key for generating the calibration map
        robot_data.inputAngleFromHorizontal[0] = v
        robot_data.inputRadiusFromBase[0] = r
        
        # Return the structured robot data
        return robot_data

    def extract_dynamic_variables(self, pose: int, axis: int, joint_is_degrees: bool = False, 
                                  acc_is_mps2: bool = True) -> Tuple[np.ndarray, 
                                      np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray, np.ndarray, 
                                      np.ndarray, np.ndarray]:
        """
        Extracts dynamic variables from the data.
        Args:
            data (np.ndarray): The data array containing the robot's dynamic variables.
            num_joints (int): The number of joints in the robot.
            joint_is_degrees (bool): If True, convert joint positions from degrees to radians.
            acc_is_mps2 (bool): If True, acceleration is in m/s^2
        Returns:
            time, command_position, input_position, joint_positions, time_acceleration, tcp_acceleration, joint_velocities, joint_currents, joint_torques
        """
        dynamic_file  = self.data_folder + f'/robotData_motion_pose{pose}_axis{axis}.{self.data_format}'

        if not os.path.exists(dynamic_file):
            raise Exception(f"Robot dynamic data file ({dynamic_file}) does not exist.")
        
        # Load the data from the file
        data, _ = load_data_file(fileformat=self.data_format, filename=dynamic_file)

        time = data[:,DEFAULT_TIMESTAMP_LOCATION] - data[0,DEFAULT_TIMESTAMP_LOCATION] # Convert to zero-based time

        # Commanded joint position
        commanded_start = DEFAULT_TIMESTAMP_LOCATION + 1
        commanded_end = commanded_start + self.num_joints
        # Convert from degrees to radians
        commanded_joint_position = np.deg2rad(data[:,commanded_start:commanded_end]) if joint_is_degrees else data[:,commanded_start:commanded_end]

        # Encoder joint position
        encoder_start = commanded_end
        encoder_end = encoder_start + self.num_joints
        # Convert from degrees to radians
        encoder_joint_position = np.deg2rad(data[:,encoder_start:encoder_end]) if joint_is_degrees else data[:,encoder_start:encoder_end]

        # Joint currents
        joint_currents_start = encoder_end
        joint_currents_end = joint_currents_start + self.num_joints
        joint_currents = data[:,joint_currents_start:joint_currents_end]

        # TCP acceleration
        tcp_acceleration_time_index = joint_currents_end
        tcp_acceleration_time = data[:,tcp_acceleration_time_index] - data[0,tcp_acceleration_time_index]

        tcp_acceleration_start = tcp_acceleration_time_index + 1
        tcp_acceleration_end = tcp_acceleration_start + NUM_CARTESIAN_AXES
        # Convert from G's to m/s^2
        tcp_acceleration = data[:,tcp_acceleration_start:tcp_acceleration_end] if acc_is_mps2 else data[:,tcp_acceleration_start:tcp_acceleration_end] * GRAVITY_ACC
        
        # TCP gyro
        tcp_gyro_start = tcp_acceleration_end
        tcp_gyro_end = tcp_gyro_start + NUM_CARTESIAN_AXES

        tcp_gyro = data[:,tcp_gyro_start:tcp_gyro_end]

        # TCP orientation (quaternion)
        tcp_orientation_time_index = tcp_gyro_end
        tcp_orientation_time = data[:,tcp_orientation_time_index] - data[0,tcp_orientation_time_index]

        tcp_orientation_start = tcp_orientation_time_index + 1
        tcp_orientation_end = tcp_orientation_start + NUM_QUATERNION_AXES
        tcp_orientation = data[:,tcp_orientation_start:tcp_orientation_end]

        return time, commanded_joint_position, encoder_joint_position, joint_currents, tcp_acceleration_time,\
            tcp_acceleration, tcp_gyro, tcp_orientation_time, tcp_orientation
    
    def extract_static_variables(self, pose: int, axis: int, 
                                 num_joints: int) -> Tuple[float, 
                                                           float, 
                                                           np.ndarray,
                                                           np.ndarray]:
        # Check if the static file exists
        static_file = self.data_folder + f"/robotData_static_pose{pose}_axis{axis}.csv"
        if not os.path.exists(static_file):
            print(f"Static file {static_file} does not exist.")
            return None, None, None, None
    
        # Load the static variable from the CSV file
        data, _ = load_data_file(fileformat=self.data_format, filename=static_file)

        # Extract the static variable from the data
        v = data[0, DEFAULT_STATIC_ANGLE_LOCATION] # angle from horizontal
        r = data[0, DEFAULT_STATIC_RADIUS_LOCATION] # distance from base

        # Extract the joint inertias from the data
        joint_inertias_start = DEFAULT_STATIC_RADIUS_LOCATION + 1
        joint_inertias_end = joint_inertias_start + num_joints
        joint_inertias = data[0, joint_inertias_start:joint_inertias_end] # joint inertias

        # Extract the end indices from the data
        end_indices_start = joint_inertias_end
        end_indices = data[0, end_indices_start:] # end indices

        return v, r, joint_inertias, end_indices
    
    def correlate_sine_wave(self, input_signal: np.ndarray, frequency: float,
                            plotting: bool = False) -> Tuple[int, int]:
        # find pure sine index
        temp = []
        for j in range(int(np.ceil(((1/frequency)*self.sine_cycles)/self.Ts))):
            temp.append(j*self.Ts)
        
        time_temp = np.array(temp)
        # Generate the sine wave for the given frequency
        sin_amplitude = min(2*np.pi*frequency*np.sqrt(self.max_disp), self.max_acc)
        q_temp = -(sin_amplitude/(2*np.pi*frequency)**2)*np.sin(2*np.pi*frequency*time_temp) # displacement [rad]
        
        # Compute the full cross-correlation (equivalent to Matlab's xcorr)
        # and find the index of the maximum correlation
        corr = correlate(input_signal, q_temp, mode='full')
        lags = np.arange(-len(q_temp) + 1, len(input_signal))
        sin_index = np.where(corr == np.max(corr))[0]
        sin_index = lags[sin_index]+1
        f_start = sin_index[0] # start index number for this frequency
        f_end   = sin_index[0] + len(q_temp) - 1 # end index number for this frequency

        # Plot the comparison between the input signal and the sine wave
        if plotting:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(time_temp, q_temp)
            plt.title(f"Frequency: {frequency} Hz")
            plt.xlabel("Time [s]")
            plt.ylabel("Displacement [rad]")
            plt.subplot(2, 1, 2)
            plt.plot(np.arange(len(input_signal[f_start:f_end]))*self.Ts, input_signal[f_start:f_end])
            plt.xlabel("Time [s]")
            plt.ylabel("Displacement [rad]")
            # Non-blocking show
            plt.show(block=False)
            # Pause briefly
            plt.pause(0.1)
            # Close the figure and continue execution
            plt.close()
            
        # Return the start and end indices
        return f_start, f_end
    
    def store_robot_data_in_vectors(self, robot_data: StructuredCalibrationData,
                                    time_in: List[np.ndarray],x_in: List[np.ndarray],
                                    y_in: List[np.ndarray],z_in: List[np.ndarray],
                                    joints: List[np.ndarray],ax_out: List[np.ndarray],
                                    ay_out: List[np.ndarray], az_out: List[np.ndarray]):
        max_N_pts = robot_data.inputData.shape[0]
        for i in range(self.num_axes):
            pad = time_in[i].shape[0] != max_N_pts
            if pad:
                # Pad the time vector with additional time points
                dt = time_in[i][1] - time_in[i][0]
                time_in[i] = np.pad(time_in[i], (0, max_N_pts - time_in[i].shape[0]), 'linear_ramp',
                                    end_values=(time_in[i][-1], time_in[i][-1] + dt*(max_N_pts - time_in[i].shape[0])))
                # Pad the x, y, z vectors with their end points
                x_in[i] = np.pad(x_in[i], ((0, max_N_pts - x_in[i].shape[0]),(0,0)), 'edge')
                y_in[i] = np.pad(y_in[i], ((0, max_N_pts - y_in[i].shape[0]),(0,0)), 'edge')
                z_in[i] = np.pad(z_in[i], ((0, max_N_pts - z_in[i].shape[0]),(0,0)), 'edge')
                # Pad the joint positions with their end points
                joints[i] = np.pad(joints[i], ((0, max_N_pts - joints[i].shape[0]), (0, 0)), 'edge')
                # Pad the acceleration vectors with zeros
                ax_out[i] = np.pad(ax_out[i], (0, max_N_pts - ax_out[i].shape[0]), 'constant', constant_values=(0,))
                ay_out[i] = np.pad(ay_out[i], (0, max_N_pts - ay_out[i].shape[0]), 'constant', constant_values=(0,))
                az_out[i] = np.pad(az_out[i], (0, max_N_pts - az_out[i].shape[0]), 'constant', constant_values=(0,))

            robot_data.time[:,i,:] = time_in[i][:,np.newaxis]
            # Cartesian positions
            robot_data.inputData[:,0,i,:] = x_in[i]
            robot_data.inputData[:,1,i,:] = y_in[i]
            robot_data.inputData[:,2,i,:] = z_in[i]
            # Joint positions
            for j in range(self.num_joints):
                robot_data.inputData[:,3+j,i,:] = joints[i][:,j][:,np.newaxis]
            # Cartesian acceleration
            robot_data.outputData[:,0,i,:] = ax_out[i][:,np.newaxis] - ax_out[i][0]
            robot_data.outputData[:,1,i,:] = ay_out[i][:,np.newaxis] - ay_out[i][0]
            robot_data.outputData[:,2,i,:] = az_out[i][:,np.newaxis] - az_out[i][0]

        return robot_data

    def _read_file(self):
        with open(self.file_path, 'r') as file:
            data = file.readlines()
        return [line.strip() for line in data if line.strip()]

    def get_data(self):
        return self.data

    def parse_data(self):
        parsed_data = []
        for line in self.data:
            values = line.split(',')
            parsed_data.append([float(value) for value in values])
        return parsed_data