## Training: Neural-Fly Model

### Overview
The `training.ipynb` file is used to learn a neural wind sensor for our self-made drone.
The code is adopted from the study of Neural-Fly (https://github.com/aerorobotics/neural-fly).
The **Domain Adversarially Invariant Meta Learning (DAIML)** algorithm is used. 
DAIML is an offline learning process designed to train a wind-invariant representation of aerodynamic effects on a quadrotor.

This notebook trains a model to learn and adapt to the wind-effect force that affects the quadrotor's motion, which is crucial for robust flight performance in windy environments. 
The goal of the DAIML approach is to develop a neural network that can predict the effects of wind on a quadrotor, even when the model has not encountered similar wind conditions during training.

### Workflow
1. **Training Process**: 
   - The notebook starts by loading and preprocessing the training and testing datasets. These datasets contain various wind conditions that affect the drone's flight dynamics.
   - The DAIML algorithm is applied to train a wind-invariant representation of the aerodynamic effects. This is done using a neural network model that learns to predict the wind forces based on the quadrotor’s state.

2. **Model Evaluation**: 
   - After training, the model is validated on test data to evaluate its generalization capabilities across different wind conditions.
   - The notebook generates simple statistics and plots that show how well the model fits the training and testing data. These visualizations are helpful for understanding the model's performance and how effectively it has learned the wind effects.

3. **Output**: 
   - The model performance is evaluated using metrics such as loss and accuracy.
   - Plots are generated to visualize the fit of the model to both the training and testing data. This helps in assessing the model's ability to predict wind effects in a variety of scenarios.

### Additional Notes
- The notebook demonstrates how DAIML can be used to train models that are robust to varying wind conditions.
- The trained model can be used for prediction and control tasks, particularly for improving the robustness of quadrotor flight in real-world wind conditions.

### References
- [Neural-Fly GitHub Repository](https://github.com/aerorobotics/neural-fly)
