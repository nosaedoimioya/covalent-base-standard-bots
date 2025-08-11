# src/identification/MapGenerationDelete.py
# This module is responsible for generating calibration maps.

# DELETE THIS FILE and use the C++ binding instead.
import os
import pickle
import numpy as np
from typing import List
from scipy.signal import sosfiltfilt, butter
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from control import matlab
from typing import List

from util.Utility import StructuredCalibrationData, TrajParams, \
                         create_2D_plot, invfreqs
from util.RobotDynamics import Dynamics

from matplotlib import pyplot as plt

# Constants
g = 9.81 # gravity constant [m/s^2]
GRAVITY_VECTOR = np.array([0,0,g]).T # gravity vector
BUTTER_ORDER = 4 # Order of Butterworth filter
BUTTER_CUTOFF = 25 # Low-pass Butterworth filter cutoff

class CalibrationMap:
    def __init__(self, numPositions: int, axesCommanded: int, numJoints: int) -> None:
        # For storing calibration map raw values
        # =========================================================
        # List of all the numerators and denominators transfer functions for sine sweeps
        self.sineSweepNumerators = [[] for _ in range(numPositions)]
        self.sineSweepDenominators = [[] for _ in range(numPositions)]

        # Matrix of all the computed (median) natural frequencies at each position commanded
        self.allWn = np.zeros((numPositions, axesCommanded))

        # Matrix of all the computed (median) damping ratios at each position commanded
        self.allZeta = np.zeros((numPositions, axesCommanded))

        # Matrix of all the computed inertia matrices at each position commanded
        self.allInertia = np.zeros((numPositions, axesCommanded))

        # Vector of the computed (med.) natural frequencies that should be used for each of 
        # the joints when using an inverse dynamics controller
        self.jointWn = np.zeros((axesCommanded,))

        # Vector of the computed (med.) damping ratios that should be used for each of the
        # joints when using an inverse dynamics controller
        self.jointZeta = np.zeros((axesCommanded,))

        # Matrix of angular position at each position commanded (for neural network fitting)
        self.allV = np.zeros((numPositions, axesCommanded))

        # Matrix of radial position at each position commanded (for NN fitting)
        self.allR = np.zeros((numPositions, axesCommanded))

        # Matrix for storing initial joint positions
        self.initialPositions = np.zeros((numPositions, axesCommanded, numJoints))

        # For storing calibration map models (median natural frequency, damping ratio, and 
        # trained neural nets)
        # ===================================================================================
        # Vector of the median joint natural frequencies
        self.medianJointWn = np.zeros((axesCommanded,))

        # Vector of median joint damping ratios
        self.medianJointZeta = np.zeros((axesCommanded,))

        # Trained neural network parameters for natural frequency
        self.nnFunctionWn = [[]] * axesCommanded

        # Trained neural network parameters for damping ratio
        self.nnFunctionZeta = [[]] * axesCommanded

    def generateCalibrationMap(self, robot_data: StructuredCalibrationData, 
                               traj_params: TrajParams, robot_model: Dynamics,
                               input_Ts: float, output_Ts: float, 
                               max_freq_fit: float = 15.0, 
                               gravity_comp: bool = False, 
                               shift_store_position: int = 0) -> None:
        '''
        generateCalibrationMap takes in system ID data from the robot, the trajectory parameters, and other 
        robot parameters and sets the robot calibration map object containing a function to generate natural
        frequencies and damping ratios as a function of the robot position, represented by polar coordinates
        V (angle from the horizontal) and R (radius from the base frame).

        Args:
            robot_data (StructuredCalibrationData): Structured data from the robot.
            traj_params (TrajParams): Trajectory parameters for the system ID.
            robot_model (Dynamics): Robot dynamics model.
            input_Ts (float): Input sampling time.
            output_Ts (float): Output sampling time.
            max_freq_fit (float, optional): Maximum frequency for fitting. Defaults to 15.0.
            gravity_comp (bool, optional): Whether to apply gravity compensation. Defaults to False.
            shift_store_position (int, optional): Position index to shift stored data. Defaults to 0.

        ----------
        Sets the CalibrationMap values when complete.
        '''
        # Create model generation string based on trajectory parameters
        model_generation_str = self._generate_model_string(traj_params)

        num_axes_commanded = robot_data.outputData.shape[2]
        num_poses = robot_data.outputData.shape[3]

        # Loop through positions and axes
        for i in range(num_poses):
            # For each position, loop through the axes commanded
            for j in range(num_axes_commanded):
                output_data = robot_data.outputData[:, :, j, i]
                input_data = robot_data.inputData[:, :, j, i]
                input_begin_end_indices = robot_data.inputBeginEndIndices[j]
                output_begin_end_indices = robot_data.outputBeginEndIndices[j]
                time = robot_data.time[:, j, i]
                freq_list = robot_data.freq_list[j]
                initial_position = input_data[0, 3:]  # Contains [x, y, z, q0, q1, ... ] - we just want the [1 x numJoints] array

                if model_generation_str == "jspAndToolAccFFT":
                    h, f_fft = self.__jspAndToolAccFFT(input_position=input_data[:,3:],
                                                       output_acceleration=output_data,
                                                       input_Ts=input_Ts, output_Ts=output_Ts,
                                                       robot_model=robot_model, axis_commanded=j+1,
                                                       init_position=initial_position,
                                                       input_begin_end_indices=input_begin_end_indices,
                                                       output_begin_end_indices=output_begin_end_indices,
                                                       gravity_comp=gravity_comp)
                else:
                    ValueError(f"Covalent base does not support \
                               {model_generation_str} model generation in this version.")

                # FRF Fitting
                # ===========================
                if traj_params.sysid_type == "bcb":
                    ValueError("Bang coast bang system identification type is not supported in this version.")
                elif traj_params.sysid_type == "sine":
                    wnJoint, zetaJoint, b_fit, a_fit = self.__sine_sweep_fit_auto(h, f_fft, freq_list, max_freq_fit)
                    self.sineSweepNumerators[i+shift_store_position].append(b_fit)
                    self.sineSweepDenominators[i+shift_store_position].append(a_fit)

                # Store natural frequencies and damping ratios
                self.allWn[i+shift_store_position,j] = wnJoint
                self.allZeta[i+shift_store_position,j] = zetaJoint

                # Store initial position values
                self.initialPositions[i+shift_store_position,j,:] = initial_position

                # Store inertial values
                inertia_mat = robot_model.get_mass_matrix(joint_angles=initial_position)
                self.allInertia[i+shift_store_position,j] = inertia_mat[j,j]

                # Store V (angle) and R (radii) values
                self.allV[i+shift_store_position,j] = robot_data.inputAngleFromHorizontal[i]
                self.allR[i+shift_store_position,j] = robot_data.inputRadiusFromBase[i]

    def _generate_model_string(self, traj_params: TrajParams) -> str:
        '''
        Generate a string representation of the model based on trajectory parameters.
        Args:
            traj_params (TrajParams): Trajectory parameters for the system ID.
        '''
        if traj_params.configuration == "task":
            model_generation_str = "tspAnd"
        else:
            model_generation_str = "jspAnd"
        model_generation_str += traj_params.output_sensor + "FFT"
        return model_generation_str
    
    def save_map(self, filename: str):
        """
        Save the class instance to a file using pickle.

        Parameters:
        filename (str): The name of the file to save the class instance to.
        """
        # Ensure the directory exists
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Create directories if they don't exist

        print(f"Saving {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load_map(self, filename: str) -> 'CalibrationMap':
        """
        Load the class instance from a file using pickle.

        Parameters:
        filename (str): The name of the file to load the class instance from.
        """
        print(f"Loading {filename}...")
        with open(filename, 'rb') as f:
            calibration_map_new = pickle.load(f)

        return calibration_map_new
    
    def __jspAndToolAccFFT(self, input_position: np.ndarray, 
                           output_acceleration: np.ndarray,
                           input_Ts: float, output_Ts: float, 
                           robot_model: Dynamics, axis_commanded: int, 
                           init_position: List, input_begin_end_indices: List = [], 
                           output_begin_end_indices: List = [], 
                           input_frame: str = 'world', output_frame: str = 'world', 
                           gravity_comp: bool = True):
        '''
        jspAndToolAccFFT takes in structured data from input joint positions and tcp acceleration
        data from the robot and combines them to create a transfer function from input acceleration to
        output acceleration, H_q
        
        Parameters
        ----------
            input_position : [N x numJoints] numpy array
                Matrix of N time-stamped input (desired) Cartesian positions
            output_acceleration : [N x 3] numpy array
                Matrix of N time-stamped output tool center point accelerations
                in the end-effector or world frame (default)
            input_Ts : float
                Sample time of the input positions
            output_Ts : float
                Sample time of the output accelerations
            robot_model : Dynamics object
                Dynamic model of the robot
            axis_commanded : int
                Number representing the joint number that is the current focus of the FFT (e.g., 1, 2, ..., 6)
            init_position : [numJoints x 1] List
                Initial configuration of the robot at the start of the trajectory
            input_begin_end_indices : [m x 1] List(Tuple) (optional arg)
                List of beginning and end indices of each bang-coast-bang trajectory. 'm' is not a fixed value and can
                vary from list to list of indices. Leave empty if using the full trajectory (default).
            output_begin_end_indices : [m x 1] List(Tuple) (optional arg)
                List of beginning and end indices of each acceleration signal. Leave empty if using the full 
                trajectory (default).
            input_frame : string (optional arg)
                Frame of reference where input Cartesian position is measured from. Entered as a string
                'world' or 'tool' referring to the world and tool frames, respectively. 'world' is default.
            output_frame : string (optional arg)
                Frame of reference where TCP acceleration is measured. 'world' is default.
            gravity_comp : boolean (optional arg)
                Whether or not to include gravity compensation. If gravity_comp is True (or not specified),
                gravity is compensated from the accelerometer readings. If it is False, gravity is not
                compensated.

        Returns
        --------
            H_q : [m x 1] List
                Complex-valued numbers representing the transfer function of the commanded to output positions
                (one for each 'm' segments of a bang-coast-bang trajectory).
            F_fft : [m x 1] List
                Frequency values for h_q  

        '''
        # Number of joints
        num_joints = len(init_position)

        # Check if commanded axis is valid
        if (axis_commanded < 1) or (axis_commanded > num_joints):
            raise ValueError(f"Commanded axis must be an integer between 1 and {num_joints} for joint \
                             space control. You entered: {axis_commanded}")
        
        # Check if the reference frames are given
        predefined_frames = ['world', 'tool']

        if input_frame not in predefined_frames:
            raise ValueError(f"{input_frame} is not a valid frame. Enter one from: {', '.join(predefined_frames)}")
        
        if output_frame not in predefined_frames:
            raise ValueError(f"{output_frame} is not a valid frame. Enter one from: {', '.join(predefined_frames)}")
       
        joint_output_acc = self.__compute_joint_acceleration_from_end_effector_acceleration(end_effector_acceleration=output_acceleration,
                                                                                          robot_model=robot_model,
                                                                                          init_joint_angles=init_position,
                                                                                          num_joints=num_joints,
                                                                                          axis_commanded=axis_commanded,
                                                                                          output_frame=output_frame,
                                                                                          gravity_comp=gravity_comp)

        # Set output variable defaults
        H_q = [] * len(input_begin_end_indices)
        F_fft = [] * len(input_begin_end_indices)

        # Compute input acceleration of the current joint (and filter it)
        joint_input_acc = np.gradient(
                            np.gradient(input_position[:,axis_commanded-1],input_Ts,axis=0),
                            input_Ts, axis=0)
        # Filter accelerations from encoder measurements
        sos_out = butter(BUTTER_ORDER, BUTTER_CUTOFF, output='sos', fs=1/input_Ts)
        joint_input_acc_filt = sosfiltfilt(sos_out, joint_input_acc.flatten())
        
        for k in range(len(input_begin_end_indices)):
            if len(input_begin_end_indices) == 0:
                input_start_idx = 0
                output_start_idx = 0
                input_end_idx = len(joint_input_acc)
                output_end_idx = len(joint_output_acc)
            else:
                input_start_idx = input_begin_end_indices[k][0]
                output_start_idx = output_begin_end_indices[k][0]
                input_end_idx = input_begin_end_indices[k][1]
                output_end_idx = output_begin_end_indices[k][1]


            # Input joint acceleration to output acceleration FRF computation
            # ===========================================================
            h_q, f_fft = self.__compute_fft(input_data=joint_input_acc_filt, 
                                            output_data=joint_output_acc,
                                            input_start_idx=input_start_idx, 
                                            input_end_idx=input_end_idx, input_Ts=input_Ts, 
                                            output_start_idx=output_start_idx,
                                            output_end_idx=output_end_idx, output_Ts=output_Ts)

            # Store data
            H_q.append(h_q)
            F_fft.append(f_fft)

        # Return at the end of function
        return H_q, F_fft
    
    def __compute_fft(self, input_data: np.ndarray, output_data: np.ndarray, 
                      input_start_idx: int, input_end_idx: int, input_Ts: float,
                      output_start_idx: int, output_end_idx: int, output_Ts: float,
                      plotting: bool = False):
        if plotting:
            # Plot input and output data for comparison
            fig, ax = create_2D_plot(data=[np.arange(0,len(input_data[input_start_idx:input_end_idx]),1)*input_Ts,input_data[input_start_idx:input_end_idx],
                                           np.arange(0,len(output_data[output_start_idx:output_end_idx]),1)*output_Ts, output_data[output_start_idx:output_end_idx]],
                                     title="Input and output data", xlabel="Time[s]",
                                     ylabel="Acceleration [rad/s^2]")
            
            # Non-blocking show, pause briefly, and close the plot
            plt.show(block=False); plt.pause(0.02); plt.close()
            
        
        # Compute FFT of input
        in_NFFT = len(input_data[input_start_idx:input_end_idx])
        in_fft = fft(input_data[input_start_idx:input_end_idx])[:in_NFFT//2] # Single side FFT
        in_f_fft = fftfreq(in_NFFT, input_Ts)[:in_NFFT//2]

        # Compute FFT of output
        out_NFFT = len(output_data[output_start_idx:output_end_idx])
        out_fft = fft(output_data[output_start_idx:output_end_idx])[:out_NFFT//2] # Single side FFT
        out_f_fft = fftfreq(out_NFFT, output_Ts)[:out_NFFT//2]

        # Interpolate output FFT to match input FFT if necessary
        if len(in_fft) > len(out_fft):
            f_interpolator = interp1d(out_f_fft, out_fft, kind='linear', bounds_error=False, fill_value=0)
            out_fft_interpolated = f_interpolator(in_f_fft)
        else:
            out_fft_interpolated = out_fft

        h_q = out_fft_interpolated / in_fft
        f_fft = in_f_fft

        return h_q, f_fft
    
    def __sine_sweep_fit_auto(self, h_list, f_list, freq_list,
                        max_freq_fit, dc_gain_weight=1e6,
                        plotting=False):
        # Step 0: prepare measured FRF
        h_q, freq = [], []
        for h, f, cmd in zip(h_list, f_list, freq_list):
            idx = np.argmin(np.abs(f - cmd))
            h_q.append(h[idx])
            freq.append(cmd)
        h_q = np.array(h_q, complex)
        freq = np.array(freq)
        
        # include DC point
        H_meas = np.concatenate(([1], h_q / h_q[0]))
        freq_fit = np.concatenate(([0.001], freq))
        idx = np.where(freq_fit <= max_freq_fit)[0]
        w = 2*np.pi*freq_fit[idx]
        
        # weight vector
        weights = np.ones_like(w)
        weights[0] = dc_gain_weight
        
        # search over pole/zero counts
        best = {"error": np.inf}
        for nA in range(2, 5):               # poles = 2,3,4
            for nB in range(0, nA):         # zeros < poles
                try:
                    b, a = invfreqs(
                        g=H_meas[idx],
                        worN=w,
                        nB=nB,
                        nA=nA,
                        wf=weights
                    )
                except Exception:
                    print(f"Warning: could not fit a transfer function with {nB} poles and {nA} zeros.")
                    continue
                
                # build TF and compute error
                sys = matlab.tf(b, a)
                H_fit = np.array([
                    matlab.evalfr(sys, 1j*w).item()
                    for w in 2*np.pi*freq_fit[idx]
                ])
                err = np.sum(weights * np.abs(H_meas[idx] - H_fit)**2)
                
                if err < best["error"]:
                    best.update({
                        "error": err,
                        "nA": nA, "nB": nB,
                        "b": b, "a": a,
                        "sys": sys,
                        "H_fit": H_fit
                    })
        
        # extract best
        poles = matlab.pole(best["sys"])
        wn = abs(poles[0])
        zeta = -np.real(poles[0]) / wn
        
        # optional plotting
        if plotting:
            mag_meas = 20*np.log10(np.abs(H_meas))
            phase_meas = np.angle(H_meas, deg=True)
            mag_fit = 20*np.log10(np.abs(best["H_fit"]))
            phase_fit = np.angle(best["H_fit"], deg=True)
            
            fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
            ax1.semilogx(freq_fit, mag_meas, label="meas")
            ax1.semilogx(freq_fit, mag_fit, '--', label="fit")
            ax2.semilogx(freq_fit, phase_meas, label="meas")
            ax2.semilogx(freq_fit, phase_fit, '--', label="fit")
            for ax in (ax1, ax2): ax.grid(True, which='both', ls='--')
            ax1.set_ylabel("Magnitude (dB)")
            ax2.set_ylabel("Phase (°)"); ax2.set_xlabel("Freq (Hz)")
            plt.tight_layout(); plt.show(block=False); plt.pause(0.05); plt.close()
        
        return wn, zeta, best["b"], best["a"]
    
    def __compute_joint_acceleration_from_end_effector_acceleration(self,
            end_effector_acceleration: np.ndarray, robot_model: Dynamics,
            init_joint_angles: List, num_joints: int, axis_commanded: int,
            output_frame: str, gravity_comp: bool) -> np.ndarray:
        """
        Compute joint acceleration from end-effector acceleration using the robot model.
        Parameters
        ----------
        end_effector_acceleration : np.ndarray
            The end-effector acceleration vector.
        robot_model : Dynamics
            The robot model object.
        init_joint_angles : List
            The initial joint angles of the robot.
        num_joints : int
            The number of joints in the robot.
        axis_commanded : int
            The number of the commanded joint.
        output_frame : str
            The frame of reference for the output acceleration ('world' or 'tool').
        gravity_comp : bool
            Whether to compensate for gravity in the acceleration calculation.
        Returns
        -------
        np.ndarray
            The computed joint acceleration vector.
        """

        # Vector used to project x, y, z vectors onto the xy-plane for tangent vector calculation
        xy_projection = np.array([[1],[1],[0]])

        # Pre-process data
        # ================================
        # (1) Rotate acceleration and position data using the rotation matrix of the starting robot
        #     position (if it is in the tool frame) to get the accelerations in the universal (base) frame
        # (2) Subtract the gravity vector from the resulting rotated acceleration vector. Gravity 
        #     is assumed to be in the negative z-direction in the universal frame.

        # Update model to get all the kinematics and dynamics updated
        robot_model.update_model(joint_angles=init_joint_angles)

        rotation_matrix = robot_model.R_total
        # Position of vector of end-effector in the commanded joint frame
        T_Ej = np.eye(4)
        for k in range(axis_commanded,num_joints):
            T_Ej = T_Ej @ robot_model.T[k]
        commanded_frame_xyz = T_Ej[0:3,3]

        # Project the commanded_frame_xyz vector onto the xy-plane of the rotating joint
        # to get the distance from the tangential acceleration vector to the axis of rotation.
        # Dot product gives (signed) distance of vector's projection
        d = np.dot(commanded_frame_xyz, xy_projection)

        # Compute the rotation matrix of the world frame to the commanded to the commanded joint
        R_j0 = np.eye(3)
        for i in range(axis_commanded): # 0 to the commanded axis (commanded axis always >= 1)
            R_j0 = R_j0 @ robot_model.R[i]
        
        # Axis of joint rotation (relative to world frame - assumes z-axis is defined as rotation axis)
        joint_rotation_axis = R_j0[:,2]
        
        # Tangent direction of acceleration relative to the joint's rotation (normalized)
        axis = np.squeeze(joint_rotation_axis).astype(float)
        frame = np.squeeze(commanded_frame_xyz).astype(float)
        
        tangent_dir = np.cross(axis,frame).reshape(3,1)
        tangent_dir /= np.linalg.norm(tangent_dir)

        # Compute commanded joint acceleration from tangential acceleration
        joint_output_acc = np.zeros((end_effector_acceleration.shape[0],))
        for k in range(end_effector_acceleration.shape[0]):
            # Reference frame compensation
            if output_frame == 'tool':
                world_acc_data = (np.linalg.inv(rotation_matrix) @ end_effector_acceleration[k,:].T)
            else:
                world_acc_data = end_effector_acceleration[k,:].T
            # Gravity compensation
            if gravity_comp == True:
                world_acc_data -= GRAVITY_VECTOR
            
            # ** Assume angular acceleration of the end-effector is zero **
            # Tangential acceleration magnitude
            joint_tang_acc = np.dot(world_acc_data,tangent_dir) # [m/s^2]

            # Divide by distance to get the angular acceleration
            joint_output_acc[k] = joint_tang_acc / np.linalg.norm(d)

        return joint_output_acc