## mujocoEnv

### Description
The `mujocoEnv` folder contains the simulation-related code for running the Mujoco-based environment. 
It includes components required to simulate the drone's behavior and interactions with its environment, as well as the PID controller that can be used for control tasks. 
This folder integrates with ROS (Robot Operating System) to enable easy communication and control.

The key Python files in this folder are:

- **`env.py`**: This is the main environment simulation file. It sets up the virtual environment for the drone and handles the interactions between the drone and its surroundings.
  
- **`lowCtrl.py`**: This file implements the low-level control logic for the drone. It includes functions for controlling the drone's motors and responding to commands.

- **`MPCQueue.py`**: This file handles the reception and processing of trajectory data. It manages the interaction with the Model Predictive Controller (MPC) and ensures smooth execution of the drone's path.

- **`neuralWind.py`**: This file contains the neural network model for generating the wind forces acting on the drone. The network predicts the wind force and applies it to the drone's simulation.

- **`PIDControllerPos.py`**: This file implements the PID controller used for controlling the drone's position. It adjusts the drone's movements to maintain the desired trajectory based on sensor feedback.

### How to Use
1. **Starting the Simulation Environment**
   - To start the simulation, use the following ROS command:
     ```bash
     roslaunch mujocoEnv env.launch
     ```
   - This command will initialize the simulation environment in Mujoco, allowing the drone to interact with its virtual surroundings.

2. **PID Controller**
   - A PID controller is provided for controlling the drone's movement. To use it, you need to set up the correct ROS interface names.
   - To start the PID controller, use the following command:
     ```bash
     roslaunch mujocoEnv pid.launch
     ```
   - **Note**: You need to specify the relevant ROS topic names and interface parameters in the `pid.launch` file before running it.
