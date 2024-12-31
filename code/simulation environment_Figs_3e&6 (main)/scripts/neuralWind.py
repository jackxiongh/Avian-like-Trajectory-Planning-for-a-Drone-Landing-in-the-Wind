#!/usr/bin/env python3
import collections
import random
import time

import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from tf import transformations

torch.set_default_tensor_type('torch.DoubleTensor')

# Define the model structure
Model = collections.namedtuple('Model', 'phi h options')


class Phi_Net(nn.Module):
    """Neural network for generating force adaptation coefficients."""
    def __init__(self, options):
        super(Phi_Net, self).__init__()
        self.fc1 = nn.Linear(options['dim_x'], 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        self.fc4 = nn.Linear(50, options['dim_a'] - 1)  # Exclude bias, added later

    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # Add constant bias term to output
        if len(x.shape) == 1:
            return torch.cat([x, torch.ones(1)])
        else:
            return torch.cat([x, torch.ones([x.shape[0], 1])], dim=-1)


class NNet(object):
    """Main neural network model for wind force estimation."""
    _softmax = nn.Softmax(dim=1)

    def __init__(self):
        self.thr2acc_ = 1.0
        self.prev_vel = np.zeros(3)
        self.prev_t = -1
        self.constant = False
        self.gravity = 9.81
        self.max_wind_force = 2.5

        # Load the trained model
        modelname = rospy.get_param("/Env/neural_model_path", "")
        self.model = torch.load(modelname)
        self.wind_force = Vector3Stamped()

        # Initialize neural network
        options = self.model['options']
        self.phi_net = Phi_Net(options=options)
        self.phi_net.load_state_dict(self.model['phi_net_state_dict'])
        self.phi_net.eval()

        # Wind force publisher
        self.wind_force_pub = rospy.Publisher('wind_force', Vector3Stamped, queue_size=1)

        # Adaptation matrices
        self.a = [
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
        ]

    def update_adapt(self, odom: Odometry, pwm: np.ndarray, mass: float, choose: int):
        """Estimate wind force based on odometry, PWM, and mass."""
        with torch.no_grad():
            t = rospy.get_rostime().to_sec()
            v = np.array([odom.twist.twist.linear.x, 
                          odom.twist.twist.linear.y, 
                          odom.twist.twist.linear.z])

            # Prepare input feature vector
            feature = np.zeros((11, 1))
            feature[:3, 0] = v
            feature[3, 0] = odom.pose.pose.orientation.x
            feature[4, 0] = odom.pose.pose.orientation.y
            feature[5, 0] = odom.pose.pose.orientation.z
            feature[6, 0] = odom.pose.pose.orientation.w
            feature[7:, 0] = (pwm - 1000.0) / 1000.0

            # Neural network inference
            X = torch.from_numpy(feature.flatten())
            Phi = self.phi_net(X)
            phi_output = np.array([1, 1, 1]).reshape([1, 3]) if self.constant else Phi.numpy()
            choose_a = self.a[choose]

            # Compute wind force in drone frame
            f_ = np.array([
                [phi_output @ choose_a[:, 0]],
                [phi_output @ choose_a[:, 1]],
                [phi_output @ choose_a[:, 2]],
                [1.0]
            ])

            # Transform to world frame
            R = transformations.quaternion_matrix([odom.pose.pose.orientation.x, 
                                                   odom.pose.pose.orientation.y,
                                                   odom.pose.pose.orientation.z, 
                                                   odom.pose.pose.orientation.w])
            wind_force_t = (R.T @ f_).flatten()
            wind_force_t = np.clip(wind_force_t, -self.max_wind_force, self.max_wind_force)

            # Publish wind force
            self.wind_force.vector.x = wind_force_t[0]
            self.wind_force.vector.y = wind_force_t[1]
            self.wind_force.vector.z = wind_force_t[2]
            self.wind_force.header.stamp = rospy.get_rostime()
            self.wind_force_pub.publish(self.wind_force)
            
            return wind_force_t[0] * mass, wind_force_t[1] * mass, wind_force_t[2] * mass
