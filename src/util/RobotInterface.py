# util/RobotInterface.py
# This module provides an interface to create and manage different types of robots.

from robots.TestRobot import TestRobot

class RobotInterface:
    def __init__(self, robot_name: str, robot_ip: str = "", local_ip: str = "", robot_id: str = ""):
        self.robot = self._create_robot(robot_name.lower(), robot_ip, local_ip)

    def _create_robot(self, robot_name: str, robot_ip: str, local_ip: str):
        if robot_name == "test":
            return TestRobot(robot_ip, local_ip)
        # add more robot types as needed
        else:
            raise ValueError(f"Unknown robot type '{robot_name}'. Please use a valid robot name.")



# Example usage
# interface = RobotInterface("flexiv", "192.168.1.1", "192.168.1.2")
