# src/util/Robot.py
# This module defines an abstract base class for robots, which can be extended by specific robot implementations
# to provide functionality for movement, command execution, and trajectory handling.

from abc import ABC, abstractmethod
from typing import List
from util.Utility import TrajParams, SystemIdParams, DataRecorder
from util.Trajectory import Trajectory

class Robot(ABC):
    def __init__(self, name: str):
        self.name = name
        self.robot_states = {}
        self.log = {}
        self.mode = {}
        self.recorder = DataRecorder()
        self.robot_info = {}
        # Number of joints in the robot
        # default [j0, j1, j2, j3, j4, j5]
        self.num_joints = 6 
        # Pose vector length 
        # default [x, y, z, qx, qy, qz, qw]
        self.pose_length = 7 

    @property
    @abstractmethod
    def in_sim_mode(self) -> bool:
        """Check if the robot is in simulation mode."""
        pass

    @property
    @abstractmethod
    def urdf_path(self) -> str:
        """URDF path for the robot."""
        pass

    @abstractmethod
    def initialize_model_from_urdf(self, urdf_path: str):
        """Load the robot kinematic and dynamic model from a URDF file."""
        pass

    @abstractmethod
    def move_home(self, home_sign: int = 1):
        pass

    @abstractmethod
    def command_robot(self, t_params: TrajParams, s_params: SystemIdParams, Ts: float, axes_to_command: int, start_pose: int = 0):
        pass

    @abstractmethod
    def move_point_to_point_xyz(self, current_pose: List[float], target_xyz: List[float]):
        pass

    @abstractmethod
    def rt_periodic_task(self, in_task_space: bool, Ts: float,
                        current_pose: List[float], 
                        trajectory: Trajectory, 
                        traj_params: TrajParams):
        pass