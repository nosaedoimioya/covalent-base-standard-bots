# robots/TestRobot.py
# This module defines a TestRobot class for testing purposes.

from util.Utility import DataRecorder

class TestRobot:
    def __init__(self, robot_ip: str, local_ip: str):
        # super().__init__("Test Robot")
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