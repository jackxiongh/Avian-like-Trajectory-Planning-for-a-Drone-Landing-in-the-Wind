#!/usr/bin/env python
import copy
import mujoco
import mujoco.viewer
import numpy as np
import rospy
import time

from geometry_msgs.msg import Vector3, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from mavros_msgs.msg import AttitudeTarget
from lowCrtl import LowCtrl
from mujocoEnv.srv import reset, resetResponse


class Env(object):
    pub_rate = 50  # simulator pub rate 50Hz

    odometry_name = "/mavros/local_position/odom"
    attitude_name = "/mavros/setpoint_raw/attitude"
    servo_output_name = "/mavros/servo_output"
    wind_set_name = "/wind_set"

    def __init__(self, handle_name: str) -> None:
        self.node = rospy.init_node("mujocoEnv", log_level=rospy.INFO)

        modelName = rospy.get_param("/Env/uav_model_path",
                                    "/home/hoshino/data/mujoco_quadrotor_wind/src/mujocoEnv/Models/quadrotor_ground.xml")
        self.model = mujoco.MjModel.from_xml_path(modelName)
        self.mjdata = mujoco.MjData(self.model)
        self.inertial = self.model.body_inertia[1, :] 

        self.low_ctrl = LowCtrl(handle_name, use_wind=True, use_param=True, inertial=self.inertial,
                                mass=self.model.body_mass[1])

        self.sleep_time = 2
        self.servo_output_state = Float64MultiArray()
        self.odom_state = Odometry()
        self.odom_state.header.frame_id = "world"

        self.timer = rospy.Timer(rospy.Duration(nsecs=int(1.0 / self.pub_rate * 1e9)), self.timer_callback)

        self.srv_reset = rospy.Service(
            handle_name + "/reset", reset, self.handle_reset
        )
        self.pub_odom = rospy.Publisher(
            handle_name + self.odometry_name, Odometry, queue_size=1, tcp_nodelay=True
        )
        self.pub_servo_output = rospy.Publisher(
            handle_name + self.servo_output_name, Float64MultiArray, queue_size=1, tcp_nodelay=True
        )
        self.sub_attitude = rospy.Subscriber(
            self.attitude_name, AttitudeTarget, self.attitude_callback, tcp_nodelay=True
        )
        self.sub_wind_set = rospy.Subscriber(
            self.wind_set_name, PoseStamped, self.wind_set_callback, tcp_nodelay=True
        )

    def attitude_callback(self, data: AttitudeTarget) -> None:
        self.low_ctrl.update_ctrl(data)

    def timer_callback(self, data) -> None:
        # publish the state
        rostime = rospy.get_rostime()
        self.odom_state.header.stamp = rostime
        self.pub_odom.publish(self.odom_state)
        self.pub_servo_output.publish(self.servo_output_state)

    def wind_set_callback(self, data: PoseStamped) -> None:
        self.low_ctrl.set_wind(int(data.pose.position.x))

    def handle_reset(self, uav_req: reset) -> resetResponse:
        uav_point = uav_req.reset_pose
        self.mjdata.qpos[0] = uav_point.x
        self.mjdata.qpos[1] = uav_point.y
        self.mjdata.qpos[2] = uav_point.z
        self.mjdata.qvel[0] = 0.
        self.mjdata.qvel[1] = 0.
        self.mjdata.qvel[2] = 0.
        self.mjdata.ctrl = np.zeros(self.model.nu)
        self.low_ctrl.reset_wind()
        res_point = Vector3(x=self.odom_state.pose.pose.position.x, y=self.odom_state.pose.pose.position.y,
                            z=self.odom_state.pose.pose.position.z)
        time.sleep(self.sleep_time)
        return resetResponse(pose=res_point, success=True)

    def get_observation(self):
        return np.array([self.odom_state.pose.pose.position.x, self.odom_state.pose.pose.position.y,
                         self.odom_state.pose.pose.position.z, self.odom_state.twist.twist.linear.x,
                         self.odom_state.twist.twist.linear.y, self.odom_state.twist.twist.linear.z, ])

    def run(self) -> None:
        """

        :return: None
        """
        with mujoco.viewer.launch_passive(self.model, self.mjdata) as viewer:
            start_time = time.time()
            count = 0
            with viewer.lock():
                # Set the visual option and camera position
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
                # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True

            while not rospy.is_shutdown() and viewer.is_running():
                step_start = time.time()
                self.mjdata.ctrl = self.low_ctrl.update_state(self.mjdata)
                mujoco.mj_step(self.model, self.mjdata)
                viewer.sync()
                count += 1
                now_time = time.time()
                if now_time - start_time > 5: 
                    # rospy.loginfo("Hz = {}\n".format(count / (now_time - start_time)))
                    print("Hz = {}\n".format(count / (now_time - start_time)))
                    # viewer.sync()
                    # rospy.loginfo(f"{self.mjdata.qfrc_passive}")
                    start_time = time.time()
                    count = 0

                # update output
                output = self.low_ctrl.get_state()  # List[Float64MultiArray, Odometry]
                self.servo_output_state.data = output[0]
                self.odom_state.pose = output[1].pose
                self.odom_state.twist = output[1].twist

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start) - 0.00005 # 
                if time_until_next_step > 0:
                    rospy.sleep(time_until_next_step)
                else:
                    print("Time until next step: {}".format(time_until_next_step))


if __name__ == "__main__":
    env = Env("Env")
    env.run()
