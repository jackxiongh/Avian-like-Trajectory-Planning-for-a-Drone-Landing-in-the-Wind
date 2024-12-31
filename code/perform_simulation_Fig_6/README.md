# Example Code for Trajectory Planning, Evaluation, and Data Saving

## Overview
This script demonstrates how trajectory planning and evaluation are carried out in the simulation presented in Fig. 6.
The results are saved to both `.npy` and `.csv` files for further analysis and inspection.

## Key Features
1. **Optimization**: 
   - The `Optimization` class simulates the optimization of a trajectory by generating random data for positions, velocities, and orientations. It models a simplified optimization process.
   
2. **Evaluation**:
   - The `Evaluate` class simulates the process of evaluating a trajectory, providing feedback such as RMS error, accelerations, and wind forces.
   
3. **Saving Data**:
   - Results (e.g., trajectories, feedback) are saved into `.npy` files for later use and `.csv` files for easy inspection and analysis.
   
## How the Code Works

We create a instance of class `AutoTest`, and then we execute `run` function to launch the entire process to generate the data. 

The details are presented here:

- **`AutoTest` Class**: This class orchestrates the entire process. It generates an optimized trajectory, evaluates it, and then saves the results to files. 
   
- **Simulation Process**: The script runs a simulation using randomly generated data, which serves as a mock for real-world optimization algorithms or hardware systems.
  
- **Data Saving**: 
   - The generated data is saved in `.npy` format (ideal for numerical data processing in Python) and `.csv` format (convenient for human-readable inspection or visualizing results).