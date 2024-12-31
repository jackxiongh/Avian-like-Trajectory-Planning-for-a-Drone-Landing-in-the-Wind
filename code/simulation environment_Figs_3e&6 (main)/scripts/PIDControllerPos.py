#!/usr/bin/env python3
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Vector3, Quaternion
from nav_msgs.msg import Odometry
from mavros_msgs.msg import AttitudeTarget
from tf import transformations
from simple_pid import PID
from MPCQueue import MPCQueue


class PositionController:
    def __init__(self):
        # Flag to check if the parameters are initialized
        self.initialized_params = False
        # Flag to activate/deactivate the controller
        self.controller_active = False
        # Initialize MPCQueue object for trajectory control
        self.mpc_queue = MPCQueue("", "")
        # Initialize other variables for odometry, command pose, and attitude target
        self.odometry = Odometry()
        self.command_pose = PoseStamped()
        self.command_vel = Vector3()
        self.target_attitude = AttitudeTarget()
        self.yaw_rate = 0

        self.gravity = 9.81  # Gravitational constant
        self.sample_time = 0.02  # Sample time for PID controllers

        # Initialize PID controllers for position (x, y, z) and velocity (x, y, z)
        self.pose_pid_x = PID(6.0, 1.2, 3.5, sample_time=self.sample_time, output_limits=(-5, 5))
        self.pose_pid_y = PID(6.0, 1.2, 3.5, sample_time=self.sample_time, output_limits=(-5, 5))
        self.pose_pid_z = PID(1.8, 0.36, 0.9, sample_time=self.sample_time, output_limits=(-4, 4))

        self.vel_pid_x = PID(3, 0.0, 1.5, sample_time=self.sample_time, output_limits=(-4, 4))
        self.vel_pid_y = PID(3, 0.0, 1.5, sample_time=self.sample_time, output_limits=(-4, 4))
        self.vel_pid_z = PID(3.0, 0.0, 0.5, sample_time=self.sample_time, output_limits=(-3, 3))

        # PID controller for yaw rate
        self.yaw_rate_pid = PID(1.5, 0.05, 0.1, sample_time=self.sample_time, output_limits=(-3, 3))

        self.mass = 0.964  # Mass of the drone (kg)

        # Initialize parameters for control
        self.initialize_parameters()

        # Subscribe to odometry topic to get position and velocity data
        self.sub_odometry = rospy.Subscriber(
            "Env/mavros/local_position/odom", Odometry, self.callback_odometry
        )

        # Publisher to send attitude control commands to the drone
        self.pub_attitude = rospy.Publisher(
            "/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1, tcp_nodelay=True
        )

        # Set initial target position for the drone
        self.command_pose.pose.position.x = 0
        self.command_pose.pose.position.y = 0
        self.command_pose.pose.position.z = 2.0
        self.pose_pid_x.setpoint = self.command_pose.pose.position.x
        self.pose_pid_y.setpoint = self.command_pose.pose.position.y
        self.pose_pid_z.setpoint = self.command_pose.pose.position.z
        self.command_pose.pose.orientation.w = 1
        self.yaw_rate_pid.setpoint = 0  # Initial yaw rate
        self.start_time = rospy.get_rostime()

    # Callback function to process odometry data and update control commands
    def callback_odometry(self, odometry: Odometry):
        assert self.initialized_params  # Ensure that parameters are initialized
        self.odometry = odometry  # Update odometry data
        self.update_control()  # Update control based on new odometry data
        acceleration = self.compute_desired_acceleration()  # Compute desired acceleration
        desired_quat = self.compute_desired_attitude(acceleration)  # Compute desired attitude

        # Compute thrust based on desired acceleration and current orientation
        rotation_matrix = transformations.quaternion_matrix(
            [self.odometry.pose.pose.orientation.x, self.odometry.pose.pose.orientation.y,
             self.odometry.pose.pose.orientation.z, self.odometry.pose.pose.orientation.w]
        )
        thrust = +self.mass * np.dot(acceleration, rotation_matrix[:3, 2])

        # Update attitude target message with desired orientation, yaw rate, and thrust
        self.target_attitude.orientation.x = desired_quat[0]
        self.target_attitude.orientation.y = desired_quat[1]
        self.target_attitude.orientation.z = desired_quat[2]
        self.target_attitude.orientation.w = desired_quat[3]
        self.target_attitude.body_rate.z = self.yaw_rate
        self.target_attitude.thrust = thrust
        self.target_attitude.header.stamp = rospy.get_rostime()

        # Publish the attitude command to control the drone
        self.pub_attitude.publish(self.target_attitude)

    # Initialize the parameters for the controller
    def initialize_parameters(self):
        self.initialized_params = True

    # Compute desired acceleration using PID controllers for position and velocity
    def compute_desired_acceleration(self):
        # Compute the desired velocity setpoints from position PID controllers
        self.command_vel.x = self.pose_pid_x(self.odometry.pose.pose.position.x)
        self.command_vel.y = self.pose_pid_y(self.odometry.pose.pose.position.y)
        self.command_vel.z = self.pose_pid_z(self.odometry.pose.pose.position.z)

        # Set the velocity PID controllers' setpoints
        self.vel_pid_x.setpoint = self.command_vel.x
        self.vel_pid_y.setpoint = self.command_vel.y
        self.vel_pid_z.setpoint = self.command_vel.z

        acceleration = np.zeros([3])  # Initialize acceleration vector

        # Compute desired accelerations from velocity PID controllers
        acceleration[0] = self.vel_pid_x(self.odometry.twist.twist.linear.x)
        acceleration[1] = self.vel_pid_y(self.odometry.twist.twist.linear.y)
        acceleration[2] = self.vel_pid_z(self.odometry.twist.twist.linear.z) + self.gravity

        return acceleration

    # Compute desired attitude (roll, pitch, yaw) from the desired acceleration
    def compute_desired_attitude(self, acceleration: np.ndarray):
        # Extract the current yaw angle from the quaternion orientation
        _, _, yaw = transformations.euler_from_quaternion(
            [self.odometry.pose.pose.orientation.x, self.odometry.pose.pose.orientation.y,
             self.odometry.pose.pose.orientation.z, self.odometry.pose.pose.orientation.w]
        )

        self.yaw_rate = self.yaw_rate_pid(yaw)  # Update yaw rate

        # Compute desired thrust based on the desired acceleration
        u1 = self.mass * np.sqrt(acceleration[0] ** 2 + acceleration[1] ** 2 + acceleration[2] ** 2)

        # Compute the desired roll angle
        roll_des = np.arcsin(
            (acceleration[0] * np.sin(yaw) - acceleration[1] * np.cos(yaw)) / np.linalg.norm(acceleration))

        # Compute the desired pitch angle
        pitch_des = np.arcsin((acceleration[0] / np.linalg.norm(acceleration) - np.sin(roll_des) * np.sin(yaw)) /
                              (np.cos(roll_des) * np.cos(yaw)))

        # Convert the desired roll, pitch, and yaw angles to a quaternion
        quat_ = transformations.quaternion_from_euler(roll_des, pitch_des, 0)

        return quat_

    # Update control commands based on the latest point from MPC queue
    def update_control(self):
        latest_point = self.mpc_queue.get_latest_point()
        if latest_point.size > 0:
            # Use the latest trajectory point for control
            self.pose_pid_x.setpoint = latest_point[0]
            self.pose_pid_y.setpoint = latest_point[1]
            self.pose_pid_z.setpoint = latest_point[2]
        else:
            pass  # No new trajectory points available

# Main function to initialize ROS node and controller
if __name__ == "__main__":
    rospy.init_node("PositionController", log_level=rospy.INFO)  # Initialize ROS node
    pid_controller = PositionController()  # Create PositionController object
    rospy.spin()  # Keep the node running
