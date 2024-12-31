#!/home/firefly/anaconda3/envs/torch/bin/python

import collections
import rospy
import message_filters
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kalman_filter import KalmanAdaptive
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from mavros_msgs.msg import AttitudeTarget  # ServoOutput
from std_msgs.msg import Float64MultiArray
from tf import transformations
from mavros_msgs.msg import AttitudeTarget, RCOut

torch.set_default_tensor_type('torch.DoubleTensor')

# Define a named tuple to store the model structure
Model = collections.namedtuple('Model', 'phi h options')

# Phi_Net class: Neural network model, takes the state features of the UAV as input and predicts the wind force
class Phi_Net(nn.Module):
    def __init__(self, options):
        super(Phi_Net, self).__init__()

        # Define the layers of the neural network
        self.fc1 = nn.Linear(options['dim_x'], 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        # The last layer has output dimension dim_a-1, appending a constant bias term
        self.fc4 = nn.Linear(50, options['dim_a'] - 1)

    def forward(self, x):
        # Apply ReLU activation function after each layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # If input is a single data point, return the data and the constant bias term; if batch input, concatenate the result
        if len(x.shape) == 1:
            return torch.cat([x, torch.ones(1)])
        else:
            return torch.cat([x, torch.ones([x.shape[0], 1])], dim=-1)

# NNet class: Loads and uses the trained neural network model to estimate wind force
class NNet(object):
    _softmax = nn.Softmax(dim=1)

    def __init__(self, modelname):
        self.thr2acc_ = 1.0
        self.prev_vel = np.zeros(3)
        self.prev_t = -1
        self.constant = False
        self.gravity = 9.81
        # Load the trained neural network model
        self.model = torch.load(modelname)
        self.kalman_filter = KalmanAdaptive()
        self.wind_force = Vector3Stamped()
        self.pwm = [np.array([1000, 1000, 1000, 1000])] * 5
        self.thrust = [0] * 5

        options = self.model['options']
        print(options)
        # Create and load the state dictionary for the neural network
        self.phi_net = Phi_Net(options=options)
        self.phi_net.load_state_dict(self.model['phi_net_state_dict'])
        self.phi_net.eval()

        # ROS publisher to publish wind force data
        self.wind_force_pub = rospy.Publisher('/wind_force', Vector3Stamped, queue_size=1)
  
        # ROS subscribers to receive UAV odometry data and RC input data
        self.odom_sub = rospy.Subscriber("Env/mavros/local_position/odom", Odometry, self._callback_odom, tcp_nodelay=True)
        self.pwm_sub = rospy.Subscriber("/mavros/rc/out", RCOut, self.pwm_cb_, tcp_nodelay=True)

    def _callback_odom(self, msg: Odometry):
        # Callback function to receive UAV position and velocity data
        t = msg.header.stamp.to_sec()
        pwm = self.pwm[0]
        thrust = self.thrust[0]
        self.update_adapt(msg, pwm, thrust, t)

    @staticmethod
    def cal_pwm(x):
        # Convert the input PWM signal to thrust using a polynomial function
        f = 1.125e-17 * pow(x, 6) - 2.689e-13 * pow(x, 5) + 1.512e-09 * pow(x, 4) - 3.783e-06 * pow(x, 3) + \
            0.00485 * pow(x, 2) - 3.121 * x + 799.8
        return f

    def pwm_cb_(self, data: RCOut):
        # Callback function to receive RC output, calculate thrust
        thrust = self.cal_pwm(data.channels[0]) + self.cal_pwm(data.channels[1]) + \
                 self.cal_pwm(data.channels[2]) + self.cal_pwm(data.channels[3])
        # Store PWM and thrust data
        self.pwm.append(np.array([data.channels[0], data.channels[1], data.channels[2], data.channels[3]]))
        self.pwm.pop(0)
        self.thrust.append(thrust)
        self.thrust.pop(0)

    def update_adapt(self, odom: Odometry, pwm: np.ndarray, thrust: float, t: float):
        with (torch.no_grad()):
            # Extract velocity from the Odometry message and construct a feature vector
            v = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z])
            feature = np.zeros((11, 1))
            feature[:3, 0] = v
            feature[3, 0] = odom.pose.pose.orientation.x
            feature[4, 0] = odom.pose.pose.orientation.y
            feature[5, 0] = odom.pose.pose.orientation.z
            feature[6, 0] = odom.pose.pose.orientation.w
            hover_ = 1.0 / 1000.0
            feature[7:, 0] = (pwm - 1000) * hover_
            # Pass the feature vector through the Phi network to get the predicted wind force
            X = torch.from_numpy(feature.flatten())
            Phi = self.phi_net(X)
            phi_output = np.array([1, 1, 1]).reshape([3, 1]) if self.constant else Phi.numpy()

            # Calculate acceleration (wind force)
            a_w = (v - self.prev_vel) / (t - self.prev_t) if self.prev_t != -1 else np.zeros(3)
            if np.isnan(a_w).any():
                return np.array([0, 0, 0])

            fu = thrust * np.array([0, 0, 1, 0]).reshape([4, 1])
            f_measurement = a_w + np.array([0, 0, self.gravity]) - (transformations.quaternion_matrix([
                odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]) @ fu).flatten()[:3]
            Kp = np.array([1, 1, 1]) * 2.0  # Gain
            s = np.zeros([3, 1])
            self.kalman_filter.update(f_measurement.reshape([3, 1]), s, phi_output)
            a = self.kalman_filter.get_a()

            # Compute the wind force
            f_ = np.array([
                phi_output.T @ a[:3, 0],
                phi_output.T @ a[3:6, 0],
                phi_output.T @ a[6:, 0]
            ])
            f_ = f_.clip(-2.5, 2.5)

            # Update the stored velocity and time information
            self.prev_t = t
            self.prev_vel = v

            # Publish the wind force data
            self.wind_force.vector.x = f_[0]
            self.wind_force.vector.y = f_[1]
            self.wind_force.vector.z = f_[2]
            self.wind_force.header.stamp = rospy.get_rostime()
            return f_

if __name__ == '__main__':
    # Initialize the ROS node and start the process
    rospy.init_node('mlmodel')
    path = rospy.get_param("net_model_path", "")
    nnet = NNet(modelname=path)
    rospy.spin()
