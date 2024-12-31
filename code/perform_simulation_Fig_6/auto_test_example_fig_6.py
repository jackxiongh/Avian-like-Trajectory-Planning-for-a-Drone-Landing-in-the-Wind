"""
Example Code for Optimization and Evaluation with Data Saving

This script demonstrates how optimization and evaluation are carried out in a simplified scenario. 
It simulates the optimization of a trajectory, evaluates the trajectory using a mock evaluation method, 
and saves the results to both `.npy` and `.csv` files.

Key Features:
1. **Optimization**: The `Optimization` class simulates optimizing a trajectory by generating random data for positions, velocities, and orientations.
2. **Evaluation**: The `Evaluate` class simulates the process of evaluating a trajectory, providing feedback such as RMS error, accelerations, and wind forces.
3. **Saving Data**: The results (trajectories, feedback, etc.) are saved to `.npy` files for later use and `.csv` files for easy inspection.

How the code works:
- The `AutoTest` class orchestrates the entire process. It generates an optimized trajectory, evaluates it, and then saves the results to files.
- The script runs through a simulation, but in a real-world scenario, this could be connected to actual data from optimization algorithms or hardware.
- The generated data is saved in both `.npy` (for numerical data) and `.csv` (for easy human-readable inspection).

In this example:
1. **Random Data Generation**: The optimization and evaluation use random data to simulate real calculations.
2. **Saving Results**: The results are saved as both `npy` files (which can be loaded into Python later for further analysis) and `csv` files (which are convenient for reporting or visualizing results).

This code is self-contained and can be run independently to see how trajectory optimization and evaluation might function in a simplified scenario.

Example Usage:
- The script runs through one trial and simulates an optimization and evaluation process. It outputs the results as files.

To run this code, simply execute the script, and the optimization process will start with random data. The results will be saved in the current directory.

"""

import numpy as np
import os
import csv
import rospy
import random
import time
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform


class Optimization:
    def __init__(self):
        """Initialization of the optimization parameters"""
        self.px = [1.0, 1.0, 1.0]  # Example initial position
        self.cfg = self.Config()

    class Config:
        def __init__(self):
            self.optimization = self.OptimizationSetting()

        class OptimizationSetting:
            def __init__(self):
                self.T_high = 10  # Time horizon
                self.T_low = 2  # Time horizon
                self.dynamics = self.Dynamics()

            class Dynamics:
                def __init__(self):
                    self.px = [1.0, 1.0, 1.0]  # Position

    def optimize(self):
        """ Simulate the optimization process. In real scenarios, this would calculate the trajectory. """
        T = 100  # Simulation time steps
        p_value = np.random.randn(T, 3)  # Random position data
        v_value = np.random.randn(T, 3)  # Random velocity data
        phi_value = np.random.randn(T, 3)  # Random orientation data
        return p_value, v_value, phi_value, None, None, T

class Evaluate:
    def __init__(self):
        """Initialize the evaluation process"""
        pass

    def evaluate(self):
        """ Simulate the evaluation of the trajectory """
        # Simulated results for evaluation
        fbk = np.random.randn(10)  # Feedback data (e.g., RMS, cost, etc.)
        traj = np.random.randn(100, 3)  # Simulated trajectory
        traj_ref = np.random.randn(100, 3)  # Reference trajectory
        a_com = np.random.randn(100, 3)  # Commanded acceleration
        a = np.random.randn(100, 3)  # Actual acceleration
        thrust = np.random.randn(100)  # Thrust values
        wind = np.random.randn(100, 3)  # Wind forces
        return fbk, traj, traj_ref, a_com, a, thrust, wind

class AutoTest:
    def __init__(self):
        """Initialize AutoTest with configurations and components"""
        self.use_generated_traj = False  # Example configuration
        self.npy_list = []  # List to hold npy file paths
        self.evaluate = Evaluate()  # Evaluation object
        self.opt = Optimization()  # Optimization object

        # Initialize ROS publishers
        
        # Corresponds to the topic used in PIDControllerPos.py for trajectory control
        self.pose_pub_opt = rospy.Publisher("", MultiDOFJointTrajectory, queue_size=1)
        self.pose_pub_trad = rospy.Publisher("", MultiDOFJointTrajectory, queue_size=1)
        # Corresponds to the topic used in PIDControllerPos.py for position control
        self.reset_pub = rospy.Publisher("", PoseStamped, queue_size=1)

        self.rl_signal_pub = rospy.Publisher("/rl/start_signal", PoseStamped, queue_size=1)

    def run(self):
        """Run the optimization test."""
        for trial in range(0, 50):
            self.auto_test(trial)

    def auto_test(self, trial):
        """Perform a single test iteration."""
        # Reset the target point randomly
        self.reset_target_point()
        time.sleep(10)  # Pause for 10 seconds to allow the reset process to complete

        # Start RL signal
        self.publish_rl_signal()
        time.sleep(10)  # Pause for RL program to execute

        # Collect data for evaluation
        """Please complete the data collection program here.
        Of course, you can use rosbag to collect all topics,
        which achieves the same result."""
        fbk, traj, traj_ref, a_com, a, thrust, wind = self.evaluate()

        # Save the evaluation data
        self.to_csv(traj, traj_ref, a_com, a, thrust, wind)

        print("Evaluation Feedback:", fbk)
        print("Trajectory Data:", traj)

        # Saving the evaluation data
        self.to_csv(traj, traj_ref, a_com, a, thrust, wind)

    def publish_trajectory(self, traj, publisher):
        """Publish the trajectory using ROS publisher with MultiDOFJointTrajectory."""
        trajectory_msg = MultiDOFJointTrajectory()
        trajectory_msg.header.stamp = rospy.Time.now()
        trajectory_msg.header.frame_id = "world"

        for i in range(len(traj)):
            point = MultiDOFJointTrajectoryPoint()

            # Set transform for position and orientation
            transform = Transform()
            transform.translation.x = traj[i][0]
            transform.translation.y = traj[i][1]
            transform.translation.z = traj[i][2]
            transform.rotation.x = 0  # Simulated orientation
            transform.rotation.y = 0
            transform.rotation.z = 0
            transform.rotation.w = 1

            # Add the transform to the trajectory point
            point.transforms.append(transform)

            # (Optional) You can also set velocities and accelerations here if needed
            # velocity = Twist()
            # acceleration = Twist()
            # point.velocities.append(velocity)
            # point.accelerations.append(acceleration)

            # Add the point to the trajectory message
            trajectory_msg.points.append(point)

        # Publish the trajectory message
        publisher.publish(trajectory_msg)
        rospy.loginfo("Trajectory published.")


    def reset_target_point(self):
        """Publish a random target point using reset_pub."""
        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()
        target_pose.header.frame_id = "world"

        # Randomly generate a target position
        target_pose.pose.position = Point(
            random.uniform(-3.0, 3.0),  # X-coordinate
            random.uniform(-3.0, 3.0),  # Y-coordinate
            random.uniform(1.5, 3.0)    # Z-coordinate
        )

        # Set a fixed orientation for simplicity
        target_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        # Publish the reset target point
        self.reset_pub.publish(target_pose)
        rospy.loginfo(f"Reset target point published: {target_pose.pose.position}")



    def publish_rl_signal(self):
        """Publish a start signal for the RL program"""
        signal = PoseStamped()
        signal.header.stamp = rospy.Time.now()
        signal.header.frame_id = "world"
        signal.pose.position = Point(0, 0, 0)  # Signal for RL to start
        signal.pose.orientation = Quaternion(0, 0, 0, 1)  # Simulate orientation
        self.rl_signal_pub.publish(signal)
        print("RL start signal published.")

    @staticmethod
    def save_numpy_array(array, file_path):
        """Save the numpy array to a given file path."""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(file_path, array)
        print(f"Array saved at {file_path}")

    def save_data_to_csv(self, filename, data):
        """Save data to CSV file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print(f"Data saved to {filename}")

    def to_csv(self, traj, traj_ref, a_com, a, thrust, wind):
        """Save trajectory and evaluation data to CSV."""
        t = np.arange(len(traj[:, 0])) * 0.02  # Time step
        csv_filename = f"traj_x{self.opt.px[0]:.2f}_y{self.opt.px[1]:.2f}_z{self.opt.px[2]:.2f}_result.csv"
        
        headers = ['t', 'x', 'y', 'z', 'ref_x', 'ref_y', 'ref_z', 'a_com_x', 'a_com_y', 'a_com_z',
                   'a_x', 'a_y', 'a_z', 'thrust', 'wind_x', 'wind_y', 'wind_z']
        
        data = [
            [t[i], traj[i, 0], traj[i, 1], traj[i, 2], traj_ref[i, 0], traj_ref[i, 1], traj_ref[i, 2], 
             a_com[i, 0], a_com[i, 1], a_com[i, 2], a[i, 0], a[i, 1], a[i, 2], thrust[i], wind[i, 0], wind[i, 1], wind[i, 2]]
            for i in range(len(traj))
        ]
        
        self.save_data_to_csv(csv_filename, [headers] + data)

# Example usage:
if __name__ == "__main__":
    rospy.init_node("auto_test")  # Initialize ROS node
    test = AutoTest()
    test.run()
