# robots/TestRobot.py
# This module defines a TestRobot class for testing purposes.

from typing import List
from util.Utility import DataRecorder
from util.Robot import Robot, TrajParams, SystemIdParams, Trajectory

class TestRobot(Robot):
    def __init__(self, robot_ip: str, local_ip: str):
        super().__init__("Test Robot")
        # Initialize Robot specific attributes
        self.robot_states = {}
        self.log = {}
        self.mode = {}
        self.recorder = DataRecorder()
        self.robot_info = {}
        self.num_joints = 6
        self.pose_length = 7
        
        # Print ip addresses
        print(f"Robot IP Address: {robot_ip}")
        print(f"Local IP address: {local_ip}")

    def move_home(self, home_sign: int = 1):
        print(f"Moving Test Robot to home position with sign {home_sign}.")
        pass

    def command_robot(self, t_params: TrajParams, s_params: SystemIdParams, Ts: float, axes_to_command: int, start_pose: int = 0):
        print(f"Commanding Test Robot with parameters: {t_params}, {s_params}, Ts: {Ts}, axes to command: {axes_to_command}, start pose: {start_pose}.")
        pass

    def move_point_to_point_xyz(self, current_pose: List[float], target_xyz: List[float]):
        pass

    def rt_periodic_task(self, in_task_space: bool, Ts: float,
                        current_pose: List[float], 
                        trajectory: Trajectory, 
                        traj_params: TrajParams):
        pass