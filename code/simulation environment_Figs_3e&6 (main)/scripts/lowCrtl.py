import random

import numpy
import numpy as np
import mujoco
import rospy
import tf.transformations
from typing import List
from mavros_msgs.msg import AttitudeTarget
from nav_msgs.msg import Odometry
from dynamic_reconfigure.server import Server
from mujocoEnv.cfg import uavConfig
from geometry_msgs.msg import Vector3
from neuralWind import NNet


class LowCtrl(object):
    """
    Low level controller
    """

    def __init__(self, handle_name: str, inertial, mass,  use_wind=False, use_param=True):
        # state pos x y z, vel x y z, rpy roll pitch yaw, pwm 1 2 3 4
        self.use_param = use_param
        self.neural_generator = NNet()

        self.state_odom = Odometry()
        self.state_desired_attitude = AttitudeTarget()
        self.state_rpy = Vector3()  # r p y
        self.state_desired_rpy = Vector3()

        self.init_ok = False
        self.use_neural = False
        self.use_wind = use_wind
        self.wind_choices = [0, 1, 2, 3, 4]
        self.wind_choice = 0

        self.ctrl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # PWM 1234 WIND 567 force
        self.pwm_min = 1000
        self.pwm_max = 2000
        self.pwm_k = 100.
        self.pwm_b = 1000.
        self.now_pwm = np.ones([4], dtype=np.float64) * self.pwm_min

        self.roll_gain = 60
        self.pitch_gain = 60
        self.roll_integrator_gain = 1.0
        self.pitch_integrator_gain = 1.0

        self.roll_error_integration = 0.0
        self.pitch_error_integration = 0.0
        self.max_integrator_error = 1.0

        self.p_gain = 25
        self.q_gain = 22
        self.r_gain = 1.0

        self.roll_error = 0.0
        self.pitch_error = 0.0

        self.inertia = np.diag(inertial)
        self.mass = mass
        self.allocation_matrix = np.array(
            [[0.707, -0.707, 1.0, 1.0],
             [-0.707, 0.707, 1.0, 1.0],
             [0.707, 0.707, -1.0, 1.0],
             [-0.707, -0.707, -1.0, 1.0]]
        ).T

        self.angular_acc_to_rotor_velocities = numpy.array([])

        self.n_rotors = 4
        self.g = 9.81
        self.rotor_force_constant = 1.0  # 8.54858e-6  # [kg.m/s^2]
        self.rotor_moment_constant = 0.016000  # m
        self.arm_length = 0.125

        self.srv_dyn = Server(uavConfig, self.callback_parameter)

        self.init_parameter()

    def get_state(self) -> list:
        return [self.now_pwm.tolist(), self.state_odom]

    def reset_wind(self):
        self.wind_choice = random.choice(self.wind_choices)

    def set_wind(self, choice):
        self.wind_choice = choice

    def init_parameter(self):
        if self.use_param:
            if rospy.get_param("/use_neural_wind"):
                self.use_neural = rospy.get_param("/use_neural_wind")
            else:
                rospy.logerr("use_neural_wind is not set")
            if rospy.get_param("/Env/rotor_force_constant"):
                self.rotor_force_constant = rospy.get_param("/Env/rotor_force_constant")
            else:
                rospy.logerr("rotor_force_constant is not set")
            if rospy.get_param("/Env/rotor_moment_constant"):
                self.rotor_moment_constant = rospy.get_param("/Env/rotor_moment_constant")
            else:
                rospy.logerr("rotor_moment_constant is not set")
            if rospy.get_param("/Env/arm_length"):
                self.arm_length = rospy.get_param("/Env/arm_length")
            else:
                rospy.logerr("arm_length is not set")
            if rospy.get_param("/Env/allocation_matrix"):
                self.allocation_matrix = np.array(rospy.get_param("/Env/allocation_matrix")).T
                rospy.loginfo("allocation_matrix: {}".format(self.allocation_matrix))
            else:
                rospy.logerr("/Env/allocation_matrix is not set")
            if rospy.get_param("/Env/n_rotors"):
                self.n_rotors = rospy.get_param("/Env/n_rotors")
            else:
                rospy.logerr("n_rotors is not set")
            if rospy.get_param("/Env/roll_gain"):
                self.roll_gain = rospy.get_param("/Env/roll_gain")
                self.srv_dyn.update_configuration({"roll_gain": self.roll_gain})
            else:
                rospy.logerr("roll_gain is not set")
            if rospy.get_param("/Env/pitch_gain"):
                self.pitch_gain = rospy.get_param("/Env/pitch_gain")
                self.srv_dyn.update_configuration({"pitch_gain": self.pitch_gain})
            else:
                rospy.logerr("pitch_gain is not set")
            if rospy.get_param("/Env/roll_int_gain"):
                self.roll_integrator_gain = rospy.get_param("/Env/roll_int_gain")
                self.srv_dyn.update_configuration({"roll_int_gain": self.roll_integrator_gain})
            else:
                rospy.logerr("roll_int_gain is not set")
            if rospy.get_param("/Env/pitch_int_gain"):
                self.pitch_integrator_gain = rospy.get_param("/Env/pitch_int_gain")
                self.srv_dyn.update_configuration({"pitch_int_gain": self.pitch_integrator_gain})
            else:
                rospy.logerr("pitch_int_gain is not set")
            if rospy.get_param("/Env/p_gain"):
                self.p_gain = rospy.get_param("/Env/p_gain")
                self.srv_dyn.update_configuration({"p_gain": self.p_gain})
            else:
                rospy.logerr("p_gain is not set")
            if rospy.get_param("/Env/q_gain"):
                self.q_gain = rospy.get_param("/Env/q_gain")
                self.srv_dyn.update_configuration({"q_gain": self.q_gain})
            else:
                rospy.logerr("q_gain is not set")
            if rospy.get_param("/Env/r_gain"):
                self.r_gain = rospy.get_param("/Env/r_gain")
                self.srv_dyn.update_configuration({"r_gain": self.r_gain})
            else:
                rospy.logerr("r_gain is not set")

        I = np.zeros([4, 4], dtype=np.float64)
        I[:3, :3] = self.inertia
        I[3, 3] = 1.0

        K = np.zeros([4, 4], dtype=np.float64)
        K[0, 0] = self.arm_length * self.rotor_force_constant
        K[1, 1] = self.arm_length * self.rotor_force_constant
        K[2, 2] = self.rotor_force_constant * self.rotor_moment_constant
        K[3, 3] = self.rotor_force_constant

        self.angular_acc_to_rotor_velocities = (self.allocation_matrix.T @
                                                np.linalg.inv(self.allocation_matrix @ self.allocation_matrix.T) @
                                                np.linalg.inv(K)) @ I
        rospy.loginfo(f"angular_acc_to_rotor_velocities:\n{self.angular_acc_to_rotor_velocities}")

        self.init_ok = True

    def callback_parameter(self, config, level):
        if level == 0:
            self.roll_gain = config["roll_gain"]
            self.pitch_gain = config["pitch_gain"]
            self.roll_integrator_gain = config["roll_int_gain"]
            self.pitch_integrator_gain = config["pitch_int_gain"]
            self.p_gain = config["p_gain"]
            self.q_gain = config["q_gain"]
            self.r_gain = config["r_gain"]
        return config

    def update_ctrl(self, cmd: AttitudeTarget) -> None:
        # rospy.loginfo(f"cmd: {cmd}")
        self.state_desired_attitude = cmd
        self.state_desired_rpy.x, self.state_desired_rpy.y, self.state_desired_rpy.z = tf.transformations.euler_from_quaternion(
            [self.state_desired_attitude.orientation.x,
             self.state_desired_attitude.orientation.y,
             self.state_desired_attitude.orientation.z,
             self.state_desired_attitude.orientation.w])

    def compute_ang_acc(self) -> Vector3:
        """Compute angular acceleration based on attitude error and control gains."""
        c_ang_acc = Vector3()
        
        # Calculate roll and pitch errors
        error_roll = self.state_desired_rpy.x - self.state_rpy.x
        error_pitch = self.state_desired_rpy.y - self.state_rpy.y

        # Integrate errors for roll and pitch
        self.roll_error_integration += error_roll
        self.pitch_error_integration += error_pitch

        # Clamp error integration to max values
        if abs(self.roll_error_integration) > self.max_integrator_error:
            self.roll_error_integration = self.max_integrator_error * self.roll_error_integration / abs(self.roll_error_integration)
        if abs(self.pitch_error_integration) > self.max_integrator_error:
            self.pitch_error_integration = self.max_integrator_error * self.pitch_error_integration / abs(self.pitch_error_integration)

        # Calculate body rate errors
        error_p = 0 - self.state_odom.twist.twist.angular.x
        error_q = 0 - self.state_odom.twist.twist.angular.y
        error_r = self.state_desired_attitude.body_rate.z - self.state_odom.twist.twist.angular.z

        # Calculate control torques for angular acceleration
        c_ang_acc.x = (self.roll_gain * error_roll + self.roll_integrator_gain * self.roll_error_integration + self.p_gain * error_p)
        c_ang_acc.y = (self.pitch_gain * error_pitch + self.pitch_integrator_gain * self.pitch_error_integration + self.q_gain * error_q)
        c_ang_acc.z = self.r_gain * error_r

        return c_ang_acc

    def calculate_rotor_thrust(self) -> list:
        """Calculate rotor thrust based on angular acceleration and body rates."""
        # Compute angular acceleration
        ang_acc = self.compute_ang_acc()

        # Construct thrust command from angular acceleration and desired thrust
        ang_acc_thrust = np.array([ang_acc.x, ang_acc.y, ang_acc.z, self.state_desired_attitude.thrust]).reshape([4, 1])

        # Get angular velocity
        ang_v = np.array([self.state_odom.twist.twist.angular.x, self.state_odom.twist.twist.angular.y, self.state_odom.twist.twist.angular.z]).reshape([3, 1])

        # Compute cross-term for rotor dynamics
        cross_term = np.zeros([4, 1], dtype=np.float64)
        cross_term[:3, 0] = (np.linalg.inv(self.inertia) @ np.cross(ang_v, self.inertia @ ang_v, axis=0))[:3, 0]

        # Calculate rotor velocities
        rotor_velocities2 = self.angular_acc_to_rotor_velocities @ (ang_acc_thrust - cross_term)
        
        # Clip rotor velocities to avoid negative values
        rotor_velocities2 = np.clip(rotor_velocities2, a_min=0, a_max=None)

        # Convert rotor velocities to thrust
        rotor_thrust = (rotor_velocities2 * self.rotor_force_constant).flatten().tolist()

        return rotor_thrust

    @staticmethod
    def cal_pwm(x):
        return 7.283e-11 * pow(x, 6) + 1.381e-07 * pow(x, 5) + 9.582e-05 * pow(x, 4) + 2.99e-02 * pow(x, 3) \
            + 3.649 * pow(x, 2) + 126.1 * x + 1017

    def convent_pwm(self, rotor_thrust: list) -> None:
        for i in range(self.now_pwm.size):
            self.now_pwm[i] = self.cal_pwm(rotor_thrust[i])
        self.now_pwm = self.now_pwm.clip(self.pwm_min, self.pwm_max)

    def update_state(self, mjdata: mujoco.MjData) -> list:

        self.state_odom.pose.pose.position.x = mjdata.qpos[0]
        self.state_odom.pose.pose.position.y = mjdata.qpos[1]
        self.state_odom.pose.pose.position.z = mjdata.qpos[2]
        self.state_odom.pose.pose.orientation.w = mjdata.qpos[3]
        self.state_odom.pose.pose.orientation.x = mjdata.qpos[4]
        self.state_odom.pose.pose.orientation.y = mjdata.qpos[5]
        self.state_odom.pose.pose.orientation.z = mjdata.qpos[6]
        self.state_rpy.x, self.state_rpy.y, self.state_rpy.z = tf.transformations.euler_from_quaternion(
            [self.state_odom.pose.pose.orientation.x,
             self.state_odom.pose.pose.orientation.y,
             self.state_odom.pose.pose.orientation.z,
             self.state_odom.pose.pose.orientation.w]
        )
        self.state_odom.twist.twist.linear.x = mjdata.qvel[0]
        self.state_odom.twist.twist.linear.y = mjdata.qvel[1]
        self.state_odom.twist.twist.linear.z = mjdata.qvel[2]
        self.state_odom.twist.twist.angular.x = mjdata.qvel[3]
        self.state_odom.twist.twist.angular.y = mjdata.qvel[4]
        self.state_odom.twist.twist.angular.z = mjdata.qvel[5]

        # assert self.init_ok
        rotor_thrust = self.calculate_rotor_thrust()
        self.ctrl[:4] = rotor_thrust
        self.convent_pwm(rotor_thrust)
        if self.use_wind:
            self.ctrl[4], self.ctrl[5], self.ctrl[6] = \
                self.neural_generator.update_adapt(self.state_odom, self.now_pwm, self.mass, self.wind_choice)
        else:
            self.ctrl[4], self.ctrl[5], self.ctrl[6] = 0.0, 0.0, 0.0
        # rospy.loginfo(f"ctrl: {self.ctrl}")
        return self.ctrl
