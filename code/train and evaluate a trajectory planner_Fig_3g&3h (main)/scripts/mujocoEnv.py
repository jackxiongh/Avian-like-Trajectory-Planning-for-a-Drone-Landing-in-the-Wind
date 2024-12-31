import math
import random
from typing import Optional, Tuple

import rospy
import numpy as np
import gymnasium as gym
import tf.transformations

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, Transform, Twist, Vector3, PoseStamped, Quaternion
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from rl_policy.srv import reset
from omegaconf import DictConfig

"""
NOTE: the attitude of quadrotor should be constant during training and evaluation
"""


class mujocoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: DictConfig) -> None:
        # parameter and limitation

        self.tra_time = cfg.tra_time  # s
        self.tra_sample_time = cfg.mode.tra_sample_time  # s

        self.max_a = cfg.max_a  # m/s2
        self.max_step = cfg.max_step
        self.pre_length = cfg.mode.pre_length
        self.max_v = cfg.max_v  # m/s
        self.area_max_x = cfg.area_max_x
        self.area_max_y = cfg.area_max_y
        self.area_max_z = cfg.area_max_z
        self.area_min_z = cfg.area_min_z
        self.max_wind_force = cfg.max_wind_force

        ## reward parameter
        self.omega_p = cfg.omega_p
        self.omega_p_decay = cfg.omega_p_decay
        self.omega_v = cfg.omega_v
        self.omega_tp = cfg.omega_tp
        self.omega_tv = cfg.omega_tv
        self.omega_t = cfg.omega_t

        # ------------------------------------------------------------------- #

        self.reset_point = Vector3()
        self.goal_point = np.zeros([3])
        self.goal_point = np.zeros([3])
        self.goal_count = 0
        self.trajectory = MultiDOFJointTrajectory()
        self.quat_ = Quaternion(w=1)  # the attitude of all waypoint (constant during the episode)
        self.reset_pose = PoseStamped()

        # ------------------------------------------------------------------- #
        act_dim = cfg.act_dim
        obs_dim = cfg.obs_dim

        # use one point
        self.use_one_action = cfg.mode.use_one_action
        self.num = cfg.mode.num

        self.observation = np.zeros([obs_dim])  # observation from vision odom, [pos, vel, wind force], NOTE: update when there is message in queue
        self.observation_uav = np.zeros([obs_dim])  # observation, NOTE: only update in 10Hz

        # observation_space uav_state[x,y,z,u_x,u_y,u_z] wind_state [f_x, f_y, f_z]

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)

        # action_space uav_action[delta_v_x, delta_v_y, delta_v_z]

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(act_dim * self.num,), dtype=np.float32)
        # ------------------------------------------------------------------- #

        # ROS setting
        rospy.init_node("learning_node", anonymous=True)
        rospy.wait_for_service('/Env/reset')
        # self.rate = rospy.Rate(int(1.0 / self.tra_time))
        self.rate = rospy.Rate(cfg.Hz)
        self.srv_proxy = rospy.ServiceProxy('/Env/reset', reset)
        self.sub_odom = rospy.Subscriber('/Env/mavros/local_position/odom', Odometry, self._callback_odom,
                                         tcp_nodelay=True)

        self.sub_neural = rospy.Subscriber('/wind_force', Vector3Stamped, self._callback_wind, tcp_nodelay=True)
        self.pub_tra = rospy.Publisher("", MultiDOFJointTrajectory, queue_size=1,
                                       tcp_nodelay=True)
        self.pub_pose = rospy.Publisher("", PoseStamped, queue_size=1)

        # ------------------------------------------------------------------- #
        self.reward_p = 0
        self.reward_v = 0
        self.reward_t = 0
        self.reward_tp = 0
        self.reward_tv = 0

        self.steps = 0
        random.seed(cfg.seed)

    def step(self,
             action: np.ndarray
             ):
        '''
        Input:
        action: [the normalized acceleration] x num

        Return:
        next observation, reward, terminated, truncated, info
        '''
        # get next waypoint
        # setup
        acc = action * self.max_a  # calculation the acc
        waypoint = self.observation_uav[:6]
        self.trajectory.points.clear()
        z_acc = 9.8 + acc[2]
        desired_p = math.atan((acc[0]) / z_acc)
        desired_r = - math.atan((acc[1]) / z_acc)
        quad = tf.transformations.quaternion_from_euler(desired_r, desired_p, 0)
        self.quat_ = Quaternion(x=quad[0], y=quad[1], z=quad[2], w=quad[3])

        if self.use_one_action:
            for i in range(1, self.pre_length + 1):
                waypoint = self._make_a_step(waypoint, acc, self.tra_sample_time,
                                             i)  # make a step and append the waypoint to trajectory
        else:
            for i in range(1, self.num + 1):
                waypoint = self._make_a_step(waypoint, acc[(i - 1) * 3:i * 3], self.tra_time,
                                             i)  # make a step and append the waypoint to trajectory

        # publish waypoints
        self.pub_tra.publish(self.trajectory)
        self.rate.sleep()
        self.steps += 1

        # get next real state
        self.observation_uav = self.observation
        obs = self.preprocess_state(self.observation_uav)

        # check done or truncated
        done, truncated, info = self.check_terminate(self.observation_uav, self.steps)

        # get reward
        reward = self.get_reward(acc[:3], self.observation_uav, self.goal_point, done, truncated, info)
        info["rew_p"] = self.reward_p
        info["rew_v"] = self.reward_v
        info["rew_t"] = self.reward_t
        info["rew_tp"] = self.reward_tp
        info["rew_tv"] = self.reward_tv

        print(f"action: {acc}, reward: {reward}, count: {self.goal_count}")

        return obs, reward, done, truncated, info

    def check_terminate(self, state: np.ndarray, step: int) -> Tuple[bool, bool, dict]:
        '''
        Input:
        state: observation (not normalized)
        step: current step

        Return:
        done: reach goal
        truncated: out of region or reach max steps
        '''
        done = False
        truncated = False
        info = {}
        info["return_terminal_reward"] = False
        info["is_success"] = False
        info["close2Goal"] = False

        # reach goal
        tmp_dist = np.linalg.norm(state[:3] - self.goal_point)
        tmp_vel = np.linalg.norm(state[3:6])
        print(f"tmp_dist: {tmp_dist}, vel: {tmp_vel}")
        if (np.linalg.norm(state[:3] - self.goal_point) < 0.1) and (np.linalg.norm(state[3:6]) < 0.1):
            info["close2Goal"] = True
            if self.goal_count >= 0:
                print("------------------------------")
                print("reach goal point")
                done = True
                info["is_success"] = True
                info["return_terminal_reward"] = True
            else:
                self.goal_count += 1

        # truncated
        ## out of region
        if state[0] > self.area_max_x or state[0] < -self.area_max_x:
            truncated = True
            info["is_success"] = False
        if state[1] > self.area_max_y or state[1] < -self.area_max_y:
            truncated = True
            info["is_success"] = False
        if state[2] > self.area_max_z or state[2] < self.area_min_z:
            truncated = True
            info["is_success"] = False
        truncated = False
        if truncated:
            print(f"out of area, state: {state[:3]}")
            info["return_terminal_reward"] = True

        ## reach max steps
        if step >= self.max_step:
            print(f"reach max step")
            truncated = True
            info["is_success"] = False
            info["return_terminal_reward"] = True

        return done, truncated, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,

    ) -> Tuple[np.ndarray, dict]:
        '''
        reset the environment

        Return:
        observation
        '''
        # set reset point
        self.reset_point = Vector3(
            x=random.uniform(-self.area_max_x + 0.3, self.area_max_x - 0.3),
            y=random.uniform(-self.area_max_y + 0.3, self.area_max_y - 0.3),
            z=random.uniform(self.area_min_z + 0.3, self.area_max_z - 0.3)
        )

        self.goal_count = 0
        
        self.goal_point = np.array([
            random.uniform(-self.area_max_x + 0.5, self.area_max_x - 0.5),
            random.uniform(-self.area_max_y + 0.5, self.area_max_y - 0.5),
            random.uniform(self.area_min_z + 0.5, self.area_max_z - 0.5)
        ])

        
        self.reset_pose.pose.position = self.reset_point
        self.reset_pose.pose.orientation = self.quat_
        self.steps = 0

        self.pub_pose.publish(self.reset_pose)
        # reset in simulation
        res = self.srv_proxy(self.reset_point)
        self.rate.sleep()  # sleep for a while to update the observation

        # self._get_wind_force()
        self.observation_uav = self.observation
        obs = self.preprocess_state(self.observation_uav)  # get normalized state and goal position in body frame
        info = {}

        self.omega_p = self.omega_p * self.omega_p_decay
        return obs, info

    def get_reward(self,
                   acc: np.ndarray,
                   state: np.ndarray,
                   goal_point: np.ndarray,
                   done: bool,
                   truncated: bool,
                   info: dict) -> float:
        '''
        delta_v: the change of velocity
        state: [pos, vel, wind force]
        goal_point: goal position
        done: true if the quadrotor reach goal (and hover a while)
        truncated: true if the quadrotor reach max steps or outside of experiment area
        info: [is_success, return_terminal_reward]
        '''
        reward = 0
        self.reward_p = -self.omega_p * np.linalg.norm(goal_point - state[:3])
        self.reward_v = - self.omega_v * np.linalg.norm(acc)
        gravity = np.array([0, 0, -9.8])
        wind_force = state[6:9] * self.max_wind_force
        self.reward_t = - self.omega_t * np.linalg.norm(acc - gravity - wind_force)
        reward += self.reward_p + self.reward_v + self.reward_t

        self.reward_tp = 0
        self.reward_tv = 0

        if info["return_terminal_reward"]:
            self.reward_tp = -self.omega_tp * np.linalg.norm(goal_point - state[:3])
            self.reward_tv = -self.omega_tv * np.linalg.norm(state[3:6])
            reward += self.reward_tp + self.reward_tv

        if info["is_success"]:
            reward += 100

        return reward

    def preprocess_state(self, observation: np.ndarray) -> np.ndarray:
        '''
        Input:
        observation: the received observation 
        Return:
        the normalized observation [the postion is the goal position in body frame]
        '''
        observation_uav = np.zeros([9])
        observation_uav[:3] = self.goal_point - observation[:3]  # calculate position in body frame
        observation_uav[3:] = observation[3:]

        # NOTE: normalize to [-1, 1]
        observation_uav[0] = observation_uav[0] / (2 * self.area_max_x)
        observation_uav[1] = observation_uav[1] / (2 * self.area_max_y)
        middle = (self.area_max_z + self.area_min_z) / 2
        observation_uav[2] = (observation_uav[2] - middle) / (self.area_max_z - self.area_min_z)
        observation_uav[3:6] = observation_uav[3:6] / self.max_v
        observation_uav[6:9] = observation_uav[6:9] / self.max_wind_force
        return observation_uav

    def _callback_odom(self, msg: Odometry):
        # receive position and velocity
        self.observation[:6] = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                                msg.pose.pose.position.z, msg.twist.twist.linear.x,
                                msg.twist.twist.linear.y, msg.twist.twist.linear.z)

    def _callback_wind(self, msg: Vector3Stamped):
        # receive wind force
        self.observation[6:9] = msg.vector.x, msg.vector.y, msg.vector.z
        self.observation[6:9] = np.clip(self.observation[6:9], -self.max_wind_force,
                                        self.max_wind_force)

    def _make_a_step(self, waypoint: np.ndarray, acc: np.ndarray, dt: float, step: int) -> np.ndarray:
        '''
        desc: make a step using euler method and append a waypoint
        input: waypoint: [pos, vel], acc: [acc]
        return: next waypoint
        '''
        tra_point = MultiDOFJointTrajectoryPoint()

        # get next waypoint
        next_waypoint = np.zeros_like(waypoint, dtype=float)
        # NOTE: divide 2, not jump change
        next_waypoint[:3] = waypoint[0:3] + dt * waypoint[3:6]  # next position
        next_waypoint[3:6] = np.clip(waypoint[3:6] + dt * acc, a_min=-self.max_v, a_max=self.max_v)  # next velocity

        # set next waypoint
        tra_point.transforms.append(Transform(translation=Vector3(
            x=next_waypoint[0], y=next_waypoint[1], z=next_waypoint[2]
        ), rotation=self.quat_))
        tra_point.velocities.append(Twist(linear=Vector3(
            x=next_waypoint[3], y=next_waypoint[4], z=next_waypoint[5])
        ))
        tra_point.time_from_start = rospy.Duration.from_sec(step * dt)

        # append waypoint
        self.trajectory.points.append(tra_point)

        return next_waypoint

    def _get_reward_component(self) -> Tuple[float, float, float, float]:
        return self.reward_p, self.reward_v, self.reward_tp, self.reward_tv
