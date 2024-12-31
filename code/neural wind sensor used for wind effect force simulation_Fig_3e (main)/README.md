## neuralNet

### Description
This folder contains the *wind sense* module. 
This module is based on aerodynamic data on the interaction between wind and the real drone. 
This module publishes the wind-effect force on the drone in simulation through ROS topics.
This module can be used by other modules, such as reinforcement learning or control systems.

The key Python files in this folder are:

- **`kalman_filter.py`**: This file implements the Kalman filter for estimating the wind force. The filter combines noisy sensor measurements with a dynamic model to produce more accurate estimates of the wind force. It is a crucial part of the wind sensing system, ensuring that the wind force data used by other modules is as accurate and reliable as possible.

- **`mlmodel.py`**: This file contains the machine learning model for predicting wind force. It uses historical data to train a neural network or another model to predict the wind forces acting on the drone based on various inputs (e.g., speed, position). The predictions can be used for better decision-making or adaptation in the control system.

### How to Use
1. **Running the Wind Sensing Module**
   - This module runs automatically once the appropriate ROS topics are subscribed to. It listens for data from the simulation environment and calculates the wind force.
   - The wind force is then published through the `wind_force` topic.

2. **Integration with Other Modules**
   - Other modules (e.g., the RL module) can subscribe to the `wind_force` topic to receive real-time wind data, which can influence decision-making or control actions.
