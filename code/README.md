# Avian-like Trajectory Planning for a Drone Landing in the Wind: Code Documentation

This repository provides code for the paper *Avian-like Trajectory Planning for a Drone Landing in the Wind: Bio-inspired Objectives, Senses, and Skill Acquisition Method*. 
The code includes three main folders: **mujocoEnv**, **neuralNet**, and **rl_policy**, each serving a distinct purpose for simulation, wind sensing, and reinforcement learning tasks.

## Table of Contents
- [Avian-like Trajectory Planning for a Drone Landing in the Wind: Code Documentation](#avian-like-trajectory-planning-for-a-drone-landing-in-the-wind-code-documentation)
  - [Table of Contents](#table-of-contents)
  - [mujocoEnv](#mujocoenv)
    - [Description](#description)
    - [How to Use](#how-to-use)
  - [neuralNet](#neuralnet)
    - [Description](#description-1)
    - [How to Use](#how-to-use-1)
  - [rl\_policy](#rl_policy)
    - [Description](#description-2)
    - [How to Use](#how-to-use-2)
  - [Dependencies](#dependencies)
    - [Required Software](#required-software)
    - [Installation](#installation)

---

## mujocoEnv (simulation environment_Figs_3&6)

### Description
The `mujocoEnv` folder contains the simulation-related code for running the Mujoco-based environment. It includes components required to simulate the drone’s behavior and interactions with its environment, as well as the PID controller that can be used for control tasks. This folder integrates with ROS (Robot Operating System) to enable easy communication and control.

- The simulation environment is set up using the `env.launch` file.
- The PID controller can be used for controlling the drone’s movement within the environment.

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

---

## neuralNet (neural wind sensor used for wind effect force simulation_Fig_3)

### Description
The `neuralNet` folder contains the *wind sense* module. This module detects the wind force experienced by the drone and publishes the information through ROS topics, which can be used by other modules, such as reinforcement learning or control systems.

### How to Use
1. **Running the Wind Sensing Module**
   - This module runs automatically once the appropriate ROS topics are subscribed to. It listens for data from the simulation environment and calculates the wind force.
   - The wind force is then published through the `wind_force` topic.

2. **Integration with Other Modules**
   - Other modules (e.g., the RL module) can subscribe to the `wind_force` topic to receive real-time wind data, which can influence decision-making or control actions.

---

## rl_policy (train a trajectory planner_Fig_3)

### Description
The `rl_policy` folder contains the reinforcement learning (RL) module. This module is responsible for training the drone to land in the presence of wind by learning an optimal trajectory through interactions with the environment.

- **train.py**: This script is used to train the RL agent. It sets up the environment, the policy network, and handles training iterations.
- **experiment.py**: This script is used for inference. It runs the trained RL policy and applies it in real-time to control the drone's landing behavior.

### How to Use
1. **Training the RL Agent**
   - To train the RL agent, use the following command:
     ```bash
     python rl_policy/train.py
     ```
   - This command will train the RL agent in the specified environment, updating the policy network through interactions with the simulation.

2. **Running the Trained Policy for Inference**
   - To run inference using the trained policy, use:
     ```bash
     python rl_policy/experiment.py
     ```
   - This will load the trained model and apply the learned policy to control the drone during landing.

---

## Dependencies

### Required Software
- **ROS (Robot Operating System)**: The code is designed to work with ROS, and many modules communicate via ROS topics.
- **Mujoco**: Used for simulation of the drone's flight and interaction with the environment.
- **Python**: Python is used to write the reinforcement learning and neural network components. Ensure you have Python 3.x installed.
- **PyTorch**: Required for running neural networks in the `neuralNet` folder.
- **Stable Baselines3**: Used for the reinforcement learning algorithms in `rl_policy`.

### Installation
1. Install **ROS**: Follow the instructions for your operating system from the [official ROS installation guide](http://wiki.ros.org/ROS/Installation).
2. Install **Mujoco**: Follow the installation instructions from the [Mujoco website](https://www.roboti.us/index.html).
3. Install **Python dependencies**:
   ```bash
   pip install torch stable-baselines3 gym
