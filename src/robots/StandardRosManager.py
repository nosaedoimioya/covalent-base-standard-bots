# src/robots/StandardRosManager.py
# This module defines the StandardRosManager class for managing the ROS interface for the Standard robot.

import numpy as np
import time, threading
from collections import deque
from pydantic import BaseModel
import rclpy

from rclpy.node import Node
from rclpy.time import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Imu
from builtin_interfaces.msg import Duration

class JointStateSample(BaseModel):
    send_time: float
    recv_time: float
    positions: list[float]
    velocities: list[float]
    efforts: list[float]

class ImuSample(BaseModel):
    send_time: float
    recv_time: float
    orientation: list[float]  # Quaternion as a list [x, y, z, w]
    orientation_covariance: list[float]
    angular_velocity: list[float]  # Angular velocity as a list [x, y, z]
    angular_velocity_covariance: list[float]
    linear_acceleration: list[float]  # Linear acceleration as a list [x, y, z]
    linear_acceleration_covariance: list[float]

class JointTrajectoryController(Node):
    def __init__(
            self,
            bot_id: str,
            Ts: float = 0.005,
            time_data: list = [],
            position_data: list = [],
            velocity_data: list = [],
            acceleration_data: list = [],
            publish_complete_event: threading.Event = None,
            buffer_len_s: float = 3600,  # 60 minutes
    ):
        super().__init__("joint_trajectory_controller")
        self.bot_id = bot_id

        # Callback groups so timer and subs can run in parallel
        cb = ReentrantCallbackGroup()

        # Publisher for joint trajectory commands
        self.joint_publisher = self.create_publisher(
            JointTrajectory,
            f"/{bot_id}/ro1/hardware/joint_trajectory",
            10,
            callback_group=cb
        )

        qos = QoSProfile(
            depth=10,  # equivalent to keep_last(1)
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        ) 

        # Subscription for joint state feedback
        self.joint_subscriber = self.create_subscription(
            JointState,
            f"/{bot_id}/ro1/hardware/joint_state",
            self.joint_state_callback,
            qos,
            callback_group=cb
        )

        # Subscription for imu data
        self.imu_subscriber = self.create_subscription(
            Imu,
            f"/{bot_id}/ro1/hardware/end_effector_imu",
            self.imu_callback,
            10,
            callback_group=cb
        )

        # Buffers to store incoming data
        maxlen = int(buffer_len_s * (1.0 / Ts))
        self.state_data: deque[JointStateSample] = deque(maxlen=maxlen)
        self.imu_data: deque[ImuSample] = deque(maxlen=maxlen)

        # We'll keep raw command-time stamps as well
        self.cmd_times  = []
        self.cmd_points = []

        # aligned logs: tuples of (timestamp, command_point, state_sample, imu_sample)
        self.aligned_log = deque(maxlen=maxlen)

        # Trajectory to send
        self.time_data = time_data
        self.position_data = position_data
        self.velocity_data = velocity_data
        self.acceleration_data = acceleration_data
        self.publish_complete_event = publish_complete_event
        

        # Timer to publish joint trajectory points and immediately sample data at Ts
        self.index = 0
        self.timer = self.create_timer(Ts, self.publish_and_record)
        self.get_logger().info("Joint Controller has Started.")

    def joint_state_callback(self, msg: JointState) -> None:
        sample = JointStateSample(
            send_time=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            recv_time=self.get_clock().now().nanoseconds * 1e-9,
            positions=msg.position,
            velocities=msg.velocity,
            efforts=msg.effort,
        )
        self.state_data.append(sample)

    def encoder_is_ready(self) -> bool:
        return len(self.state_data) > 0 and self.state_data[-1].recv_time > time.time() - 1

    def imu_callback(self, msg: Imu) -> None:
        sample = ImuSample(
            send_time=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            recv_time=self.get_clock().now().nanoseconds * 1e-9,
            orientation=[msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            orientation_covariance=list(msg.orientation_covariance),
            angular_velocity=[msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            angular_velocity_covariance=list(msg.angular_velocity_covariance),
            linear_acceleration=[msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            linear_acceleration_covariance=list(msg.linear_acceleration_covariance),
        )
        self.imu_data.append(sample)

    def imu_is_ready(self) -> bool:
        return len(self.imu_data) > 0 and self.imu_data[-1].recv_time > time.time() - 1
    
    def publish_and_record(self):
        # wait until we're recording data before starting trajectory
        if (not self.encoder_is_ready()) or (not self.imu_is_ready()):
            return

        # 1) if there’s still trajectory left, publish next point
        if self.index < len(self.position_data):
            now = self.get_clock().now()
            point = JointTrajectoryPoint()
            point.positions = self.position_data[self.index]
            point.velocities = self.velocity_data[self.index] if len(self.velocity_data) > 0 else []
            point.accelerations = self.acceleration_data[self.index] if len(self.acceleration_data) > 0 else []

            # Use provided time_data to set the time_from_start field [seconds]
            point.time_from_start = Duration(
                sec=int(now.nanoseconds*1e-9), 
                nanosec=int(now.nanoseconds % 1e9)
            )  

            msg = JointTrajectory()
            msg.joint_names = [f"joint{i}" for i in range(len(point.positions))]
            msg.points.append(point)
            self.joint_publisher.publish(msg)

            # record raw command timestamp and point
            self.cmd_times.append(now.nanoseconds * 1e-9)
            self.cmd_points.append(point)

            self.index += 1
        else:
            # Once the trajectory is complete, cancel the timer
            self.timer.cancel()
            self.get_logger().info("Trajectory publishing complete")
            # Align the data
            self.align_offline()
            self.get_logger().info("Aligned data")
            self.publish_complete_event.set()

    def align_offline(self):
        """Interpolate state_data and imu_data onto uniform self.time_data."""
        if self.time_data is None:
            raise RuntimeError("time_data must be provided for offline alignment")
        
        # Helper to extract arrays from deque of dicts
        def extract(deq, key):
            arr = np.array([getattr(d, key) for d in deq])
            stamps = np.array([getattr(d, 'send_time') for d in deq])
            return stamps, arr

        # Joint states
        t_state, pos_state = extract(self.state_data, 'positions')
        _, vel_state = extract(self.state_data, 'velocities')
        _, eff_state = extract(self.state_data, 'efforts')
        # IMU
        t_imu,  accel_imu = extract(self.imu_data, 'linear_acceleration')
        _,      angvel_imu = extract(self.imu_data, 'angular_velocity')
        _, quat_imu = extract(self.imu_data, 'orientation')


        # For each command time, interpolate latest sensor readings
        for t_cmd, point in zip(self.cmd_times, self.cmd_points):
            # find nearest or interpolate
            def interp(t, x):
                return np.interp(t_cmd, t, x, left=np.nan, right=np.nan)

            pos_i   = np.array([interp(t_state, pos_state[:,j]) for j in range(pos_state.shape[1])])
            vel_i   = np.array([interp(t_state, vel_state[:,j]) for j in range(vel_state.shape[1])])
            eff_i   = np.array([interp(t_state, eff_state[:,j]) for j in range(eff_state.shape[1])])
            accel_i = np.array([interp(t_imu, accel_imu[:,j]) for j in range(accel_imu.shape[1])])
            angv_i  = np.array([interp(t_imu, angvel_imu[:,j]) for j in range(angvel_imu.shape[1])])
            quat_i  = np.array([interp(t_imu, quat_imu[:,j]) for j in range(quat_imu.shape[1])])

            self.aligned_log.append({
                    'cmd_time' : t_cmd,
                    'input_positions'  : point.positions,
                    'output_positions': pos_i,
                    'velocities': vel_i,
                    'efforts'   : eff_i,
                    'imu_time' : t_cmd,
                    'linear_acceleration' : accel_i,
                    'angular_velocity'    : angv_i,
                    'orientation' : quat_i,
                })
        