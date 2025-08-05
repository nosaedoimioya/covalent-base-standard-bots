# util/Utility.py
# This module provides utility objects and functions for other scripts.

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