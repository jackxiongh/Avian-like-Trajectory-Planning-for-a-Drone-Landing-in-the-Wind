 # Data Folder Structure

The `data` folder contains experimental data used for the training and evaluation of a neural wind sensor (Figs. 3c&3d), the training of a bio-inspired trajectory planner (Fig. 3g), and the evaluation of the bio-inspired trajectory planner (Figs. 4, 5, 6, and 3h). 
Each subfolder includes data of experiments shown in corresponding figure/s.


Where:
- `trajectory planning method` can be one of the following:
  - **optimization**: Refers to data generated using the time-optimal trajectory planning method.
  - **rl**: Refers to data generated using the Reinforcement Learning (RL) trajectory planning method.
  - **tradition**: Refers to data generated using the conventional trajectory planning method.

- `flight control method` can be one of the following:
  - `MPC` is the default flight controller. 
  - `PID` is a flight controller that can be used for the study of generalization capability.


## File Example

Each file in the subfolders follows the naming format:
- `x`, `y`, `z` represent the initial position of a test.
- `method` can be either `opt`, `rl`, or `trad` depending on the trajectory planning method used.



### Description of files in folders related to Figs. 4, 5, 6, and 3h

1. **t**: Time, unit: seconds (s).  
2. **x, y, z**: Drone position, unit: meters (m).  
3. **roll**: Roll angle, unit: radians (rad).  
4. **pitch**: Pitch angle, unit: radians (rad).  
5. **yaw**: Yaw angle, unit: radians (rad).  
6. **v_x, v_y, v_z**: Drone velocity components in the x, y, and z directions, unit: meters per second (m/s).  
7. **ref_x, ref_y, ref_z**: Reference position in the x, y, and z coordinates, unit: meters (m).  
8. **ref_v_x, ref_v_y, ref_v_z**: Reference velocity components in the x, y, and z directions, unit: meters per second (m/s).  
9. **a_com**: Magnitude of drone acceleration, unit: meters per second squared (m/s²).  
10. **a_x, a_y, a_z**: Drone acceleration components in the x, y, and z directions, unit: meters per second squared (m/s²).  
11. **thrust**: Thrust-effect acceleration, unit: meters per second squared (m/s²).  
12. **wind_x, wind_y, wind_z**: Wind-effect acceleration in the x, y, and z directions, unit: meters per second squared (m/s²).  

### Description of files in folders related to Figs. 3b and 3c

1. **num**: Index or sequence number, unit: dimensionless.  
2. **t**: Time of a reference trajectory, unit: seconds (s).  
3. **tt**: Recorded timestamp, unit: seconds (s).  
4. **p**: Position of the drone in the x, y, and z coordinates, unit: meters (m).  
5. **v**: Velocity of the drone in the x, y, and z directions, unit: meters per second (m/s).  
6. **q**: Quaternion representing the drone's orientation \([qx, qy, qz, qw]\), unit: dimensionless.  
7. **T**: Thrust-effect acceleration, unit: meters per second squared (m/s²).  
8. **T_sp**: Target thrust acceleration, unit: meters per second squared (m/s²).  
9. **q_sp**: Target orientation in quaternion \([qx, qy, qz, qw]\), unit: dimensionless.  
10. **wind_effect**: Wind0-effect acceleration in the x, y, and z directions, unit: meters per second squared (m/s²).  
11. **pwm**: Pulse Width Modulation (PWM) signals for motors \([motor1, motor2, motor3, motor4]\)

