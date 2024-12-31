import numpy as np
import rospy
from std_msgs.msg import Header
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist
from tf import transformations
from geometry_msgs.msg import PoseStamped, Vector3, Quaternion


class MPCQueue:
    def __init__(self, tra_topic_name, pos_topic_name):
        self.sub_tra = rospy.Subscriber(tra_topic_name, MultiDOFJointTrajectory, self.trajectory_callback)
        self.sub_command_pose = rospy.Subscriber(
            pos_topic_name, PoseStamped, self.callback_command_pose
        )
        self.trajectory_points = []
        self.interpolation_step = 0.02  # 20ms
        rospy.loginfo("MPCQueue initialized and listening to {}".format(topic_name))

    def trajectory_callback(self, data: MultiDOFJointTrajectory):
        rospy.loginfo("Received trajectory message")
        self.trajectory_points = self.interpolate_trajectory(data)

    def callback_command_pose(self, pose):
        point = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, 0, 0, 0, 0, 0, 0])
        self.trajectory_points = [point]

    def interpolate_trajectory(self, trajectory):
        interpolated_points = []
        for i in range(len(trajectory.points) - 1):
            start_point = trajectory.points[i]
            end_point = trajectory.points[i + 1]
            start_time = start_point.time_from_start.to_sec()
            end_time = end_point.time_from_start.to_sec()
        
            duration = end_time - start_time
            num_steps = int(np.ceil(duration / self.interpolation_step))
            for step in range(num_steps):
                t = step * self.interpolation_step
                interp_point = self.interpolate_point(start_point, end_point, t / duration)
                interpolated_points.append(interp_point)


        return interpolated_points

    def interpolate_point(self, start_point, end_point, alpha):
        interp_pos = (1 - alpha) * np.array(
            [start_point.transforms[0].translation.x, start_point.transforms[0].translation.y,
             start_point.transforms[0].translation.z]) + \
                     alpha * np.array([end_point.transforms[0].translation.x, end_point.transforms[0].translation.y,
                                       end_point.transforms[0].translation.z])
        # Assuming simple interpolation for velocity and ignoring orientation for simplicity
        interp_vel = (1 - alpha) * np.array([start_point.velocities[0].linear.x, start_point.velocities[0].linear.y,
                                             start_point.velocities[0].linear.z]) + \
                     alpha * np.array(
            [end_point.velocities[0].linear.x, end_point.velocities[0].linear.y, end_point.velocities[0].linear.z])
        # Convert to your desired format
        return np.hstack((interp_pos, interp_vel, np.zeros(3)))  # Roll, pitch, yaw are zeros for simplification

    def get_latest_point(self):
        if len(self.trajectory_points) > 1:
            return self.trajectory_points.pop(
                0)  # Retrieve the next point and remove from list if there are multiple points left
        elif self.trajectory_points:
            return self.trajectory_points[0]  # Return the last point without removing it from the list
        return np.array([])  # Return an empty array if no points are available
