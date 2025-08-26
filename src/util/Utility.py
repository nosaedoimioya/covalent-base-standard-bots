# src/util/Utility.py
# This module provides utility objects and functions for other scripts.

from typing import List, Tuple
import os, csv, datetime
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve

# Data Structures
class MotionProfile:
    def __init__(self, t, q, endIndex):
        self.t = t
        self.q = q
        self.endIndex = endIndex

class TrajParams:
    def __init__(self, configuration, max_displacement, 
                 max_velocity, max_acceleration, sysid_type, 
                 single_pt_run_time, output_sensor = "ToolAcc"):
        ''' Initialize trajectory parameters.
        Args:
            configuration (str): Control configuration type ('task' or 'joint').
            max_displacement (float): Maximum displacement in meters.
            max_velocity (float): Maximum velocity in m/s.
            max_acceleration (float): Maximum acceleration in m/s^2.
            sysid_type (str): System identification type ('bcb' or 'sine').
            single_pt_run_time (float): Run time for single point in seconds.
            output_sensor (str): Sensor to output data from, default is "ToolAcc".
        '''
        self.configuration = configuration
        self.max_displacement = max_displacement
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.sysid_type = sysid_type
        self.single_pt_run_time = single_pt_run_time
        self.output_sensor = output_sensor

class SystemIdParams:
    def __init__(self, nV, nR):
        self.nV = nV
        self.nR = nR

class DataRecorder:
    def __init__(self):
        self.inputJointPositions = []
        self.outputJointPositions = []
        self.outputCurrents = []
        self.outputTcpAccelerations = []
        self.servoTime = []
        self.imuTime = []
        self.quaternionTime = []
        self.quaternion = []

        # Static parameters
        self.outputMassDiagonals = []
        self.endIndices = []
        self.inputV = []
        self.inputR = []

class StructuredCalibrationData:
    def __init__(self, num_input_points: int, num_output_points: int, 
                 num_output_vars: int, num_input_vars: int, 
                 num_axes: int, input_begin_end_indices: List[List[int]], 
                 output_begin_end_indices: List[List[int]],
                 frequency_list: List[float] = [],
                 numV: int = 1, numR: int = 1) -> None:
        self.outputData = np.zeros((num_output_points, num_output_vars, num_axes, numV*numR))
        self.inputData = np.zeros(((num_input_points, num_input_vars, num_axes, numV*numR)))
        self.inputAngleFromHorizontal = np.zeros((numV*numR,))
        self.inputRadiusFromBase = np.zeros((numV*numR,))
        self.inputBeginEndIndices = input_begin_end_indices
        self.outputBeginEndIndices = output_begin_end_indices
        self.time = np.zeros((num_input_points, num_axes, numV*numR))
        self.freq_list = frequency_list

# Functions
# ============

# Load data from a file based on the specified format
def load_data_file(fileformat: str, filename: str) -> Tuple[np.ndarray, list[str]]:
    if fileformat == "csv" or fileformat == ".csv":
        return load_vector_data_from_csv(filename=filename)
    elif (fileformat == "npz" or fileformat == "npy"\
          or fileformat == ".npz" or fileformat == ".npy"):
        return load_vector_data_from_npz(filename=filename)
    else:
        raise ValueError(f"Unsupported file format '{fileformat}'. Supported formats are 'csv', 'npz', and 'npy'.")
    
# Load from csv
def load_vector_data_from_csv(filename: str, delimiter: str = ',') -> Tuple[np.ndarray, List[str]]:
    """
    Load a CSV file where the first row is headers and the remaining rows are numeric.
    Returns:
      data    - a 2D NumPy array of shape (n_rows, n_cols)
      headers - a list of column names (strings)
    """
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)                           # read the first line
        rows = [list(map(float, row)) for row in reader] # convert each cell to float

    data = np.array(rows)                                # shape (n_rows, n_cols)
    return data, headers

# Load npz/npy file
def load_vector_data_from_npz(filename: str) -> Tuple[np.ndarray, list[str]]: 
    """
    Load a NumPy .npz file containing arrays.
    Returns:
      data    - a 2D NumPy array of shape (n_rows, n_cols)
      headers - a list of column names (strings)
    """
    npz = np.load(file=filename)
    data = npz["data"]
    headers = npz["headers"]
    npz.close()
    return data, headers

def create_2D_plot(data: List, title: str, ylabel: str, xlabel: str) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a 2D plot from the provided data.
    Args:
        data (List): List of 2D data points to plot.
        title (str): Title of the plot.
        ylabel (str): Label for the y-axis.
        xlabel (str): Label for the x-axis.
    Returns:
        fig (plt.Figure): The created figure.
        ax (plt.Axes): The axes of the plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(1,1)

    # Plot multiple lines
    for j in range(0,len(data),2):
        ax.plot(data[j], data[j+1])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    
    return fig, ax

def invfreqs(g, worN, nB, nA, wf=None, nk=0):
    """
    Compute frequency response of a digital filter.

    Description:

       Computes the numerator (b) and denominator (a) of a digital filter compute
       its frequency response, given a frequency response at frequencies given in worN.

                             nB      nB-1                             nk
               B(s)    (b[0]s + b[1]s   + .... + b[nB-1]s + b[nB])s
        H(s) = ---- = -----------------------------------------------
                            nA      nA-1
               A(s)    a[0]s + a[1]s + .... + a[nA-1]s+a[nA]

        with a[0]=1.

       Coefficients are determined by minimizing sum(wf |B-HA|**2).
       If opt is not None, minimization of sum(wf |H-B/A|**2) is done in at most
       MaxIter iterations until norm of gradient being less than Tol,
       with A constrained to be stable.

    Inputs:

       worN -- The frequencies at which h was computed.
       h -- The frequency response.

    Outputs: (w,h)

       b, a --- the numerator and denominator of a linear filter.
    """
    g = np.atleast_1d(g)
    worN = np.atleast_1d(worN)
    if wf is None:
        wf = np.ones_like(worN)
    if len(g)!=len(worN) or len(worN)!=len(wf):
        raise ValueError("The lengths of g, worN and wf must coincide.")
    if np.any(worN<0):
        raise ValueError("worN has negative values.")
    s = 1j*worN

    # Constraining B(s) with nk trailing zeros
    nm = np.maximum(nA, nB+nk)
    mD = np.vander(1j*worN, nm+1)
    mH = np.asmatrix(np.diag(g))
    mM = np.asmatrix(np.hstack(( mH*np.asmatrix(mD[:,-nA:]),\
            -np.asmatrix(mD[:,-nk-nB-1:][:,:nB+1]))))
    mW = np.asmatrix(np.diag(wf))
    Y = solve(np.real(mM.H*mW*mM), -np.real(mM.H*mW*mH*np.asmatrix(mD)[:,-nA-1]))
    a = np.ones(nA+1)
    a[1:] = Y[:nA].flatten()
    b = np.zeros(nB+nk+1)
    b[:nB+1] = Y[nA:].flatten()

    return b,a

def deg2rad(degrees: float) -> float:
    """
    Convert degrees to radians.
    
    Args:
        degrees (float): Angle in degrees.
    
    Returns:
        float: Angle in radians.
    """
    return np.deg2rad(degrees)

def rad2deg(radians: float) -> float:
    """
    Convert radians to degrees.
    
    Args:
        radians (float): Angle in radians.
    
    Returns:
        float: Angle in degrees.
    """
    return np.rad2deg(radians)

def rotation_matrix_to_quaternion(rotation_mat: np.ndarray) -> np.ndarray:
    # Check if the input is a 3x3 matrix
    if rotation_mat.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3")
    
    # Compute the trace of the matrix
    trace = np.trace(rotation_mat)

    # Compute the quaternion
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (rotation_mat[2, 1] - rotation_mat[1, 2]) / S
        qy = (rotation_mat[0, 2] - rotation_mat[2, 0]) / S
        qz = (rotation_mat[1, 0] - rotation_mat[0, 1]) / S
    elif (rotation_mat[0, 0] > rotation_mat[1, 1]) and (rotation_mat[0, 0] > rotation_mat[2, 2]):
        S = np.sqrt(1.0 + rotation_mat[0, 0] - rotation_mat[1, 1] - rotation_mat[2, 2]) * 2  # S = 4 * qx
        qw = (rotation_mat[2, 1] - rotation_mat[1, 2]) / S
        qx = 0.25 * S
        qy = (rotation_mat[0, 1] + rotation_mat[1, 0]) / S
        qz = (rotation_mat[0, 2] + rotation_mat[2, 0]) / S
    elif rotation_mat[1, 1] > rotation_mat[2, 2]:
        S = np.sqrt(1.0 + rotation_mat[1, 1] - rotation_mat[0, 0] - rotation_mat[2, 2]) * 2  # S = 4 * qy
        qw = (rotation_mat[0, 2] - rotation_mat[2, 0]) / S
        qx = (rotation_mat[0, 1] + rotation_mat[1, 0]) / S
        qy = 0.25 * S
        qz = (rotation_mat[1, 2] + rotation_mat[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rotation_mat[2, 2] - rotation_mat[0, 0] - rotation_mat[1, 1]) * 2  # S = 4 * qz
        qw = (rotation_mat[1, 0] - rotation_mat[0, 1]) / S
        qx = (rotation_mat[0, 2] + rotation_mat[2, 0]) / S
        qy = (rotation_mat[1, 2] + rotation_mat[2, 1]) / S
        qz = 0.25 * S

    return np.array([qx, qy, qz, qw])

def get_polar_coordinates(num_angles: int, num_radii: int, max_reach: float) -> Tuple[np.ndarray, np.ndarray]:
    V = np.linspace(0, np.pi / 2, num_angles).tolist()
    R = np.linspace(0.4 * max_reach, 0.8 * max_reach, num_radii).tolist()
    return V, R

def polar_to_cartesian(v: float, r: float, side_arm: float, 
                       base_height: float, base_angle: float) -> Tuple[float, float, float]:
    long = r * np.cos(v) * np.cos(base_angle)
    width = side_arm - r * np.cos(v) * np.sin(base_angle)
    height = base_height + r * np.sin(v)
    return long, width, height

def cartesian_to_polar(long: float, width: float, height: float,
                       side_arm: float, base_height: float) -> Tuple[float, float]:
    """
    Inverse of polar_to_cartesian.  
    Returns (v, r) such that:
      long  = r*cos(v)*cos(base_angle)
      width = side_arm - r*cos(v)*sin(base_angle)
      height= base_height + r*sin(v)
    """
    # 1) horizontal projection
    dx = np.sqrt(long**2 + (side_arm - width)**2)
    # 2) radial distance
    dz = height - base_height
    r = np.hypot(dx, dz)
    # 3) elevation angle
    v = np.arctan2(dz, dx)

    return v, r

def linspace_step_size(start: float, end: float, step: float) -> np.ndarray:
    num_steps = round((end - start) / step) + 1  # Ensure exact number of points
    return np.linspace(start, end, num_steps).tolist()

def store_recorder_data_in_csv(recorder: DataRecorder, run_index: int, move_axis: int, filename: str = None) -> None:
    # Motion parameters
    motion_headers = [
        'index',
        'servo_period',
        'servo_timestamp',
        *[f'j{i}_pos_cmd' for i in range(len(recorder.inputJointPositions[0]))],
        *[f'j{i}_pos_enc' for i in range(len(recorder.inputJointPositions[0]))],
        *[f'j{i}_current' for i in range(len(recorder.inputJointPositions[0]))],
        'imu_timestamp',
        'acc_x','acc_y','acc_z',
        'gyro_x','gyro_y','gyro_z',
        'quat_timestamp',
        'w','x','y','z'
    ]

    motion_data_rows = []
    for i in range(len(recorder.servoTime)):
        row = [
            i,
            recorder.servoTime[i] - recorder.servoTime[i-1] if i > 0 else 0.0,
            recorder.servoTime[i],
            *recorder.inputJointPositions[i],
            *recorder.outputJointPositions[i],
            *recorder.outputCurrents[i],
            recorder.imuTime[i],
            *recorder.outputTcpAccelerations[i],
            recorder.quaternionTime[i],
            *recorder.quaternion[i]
        ]
        
        motion_data_rows.append(row)

    # Create filename
    now = datetime.datetime.now()
    if filename is None:
        motion_filename = f"data/{now.year}-{now.month}-{now.day}/robotData_motion_pose{run_index}_axis{move_axis}.csv"
    else:
        motion_filename = f"data/{now.year}-{now.month}-{now.day}/motion_{filename}"

    # Create directories if they don't exist
    motion_directory = os.path.dirname(motion_filename)
    os.makedirs(motion_directory, exist_ok=True) 

    # Store motion data in csv
    with open(file=motion_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(motion_headers)     # header row
        writer.writerows(motion_data_rows)    # all data rows
    
    # Static parameters
    static_headers = [
        'input_v',
        'input_r',
        *[f'j{i}_inertia' for i in range(len(recorder.outputMassDiagonals))],
        *[f'end_index_{i}' for i in range(len(recorder.endIndices))] # (unknown size) -- must be the last elements 
    ]

    static_data_rows = []
    static_data_rows.append(
        [
            *recorder.inputV,
            *recorder.inputR,
            *recorder.outputMassDiagonals,
            *recorder.endIndices

        ]
    )

    # Create filename
    static_filename = f"data/{now.year}-{now.month}-{now.day}/robotData_static_pose{run_index}_axis{move_axis}.csv"
    if filename is None:
        static_filename = f"data/{now.year}-{now.month}-{now.day}/robotData_static_pose{run_index}_axis{move_axis}.csv"
    else:
        static_filename = f"data/{now.year}-{now.month}-{now.day}/static_{filename}"
    
    # Create directories if they don't exist
    static_directory = os.path.dirname(static_filename)
    os.makedirs(static_directory, exist_ok=True) 

    # Store static data in csv
    with open(file=static_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(static_headers)     # header row
        writer.writerows(static_data_rows)    # all data rows

def normalized_interpolation(time_list: List[np.ndarray], to_interp_lists: List[list]) -> Tuple[List, np.ndarray]:
        # Determine the maximum number of trajectory points across joints
        max_traj_pts = max(len(tj) for tj in time_list)

        # Compute normalized time for each joint trajectory (range [0,1])
        normalized_time_trajectories = []
        for tj in time_list:
            if tj[-1] > 0:
                norm_tj = tj / tj[-1]  # normalize each time vector by its final time
                normalized_time_trajectories.append(norm_tj)
            else:
                normalized_time_trajectories.append(tj)

        # Create a common normalized time vector from 0 to 1 with max_traj_pts samples
        common_norm_time = np.linspace(0, 1, max_traj_pts)

        # Optionally, define a common overall duration (e.g., the maximum final time)
        max_duration = max(tj[-1] for tj in time_list)
        path_time = common_norm_time * max_duration

        # Interpolate each joint's trajectory using the normalized time vector
        for _list in to_interp_lists: # interp_lists = [joint_trajetories, joint_velocities, joint_accelerations]
            for i in range(len(_list)): # _list = joint_trajectories list
                arr = np.squeeze(_list[i])
                _list[i] = np.interp(common_norm_time, normalized_time_trajectories[i], arr)

        # Return altered lists
        return to_interp_lists, path_time

def transpose_list(original_list: List) -> List:
    return list(map(list, zip(*original_list)))