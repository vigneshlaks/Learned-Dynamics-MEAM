# Investigating Smoothing Methods of Learned Dynamics for Trajectory Optimization

This project explores the integration of smoothing techniques into learned dynamics models for robotic control and trajectory optimization. The aim is to improve stability, accuracy, and computational efficiency in tasks like CartPole and 2D Quadrotor stabilization.

## Key Contributions and Challenges

### 1. Working with Safe Control Environment
- Integrated randomized controllers into the Safe Control Gym framework despite its rigid architecture.
- Enabled random initialization for LQR data generation after extended debugging and architectural understanding.
- Used LQR policies to collect diverse state-action-next state tuples, critical for training dynamics models.

### 2. Data Augmentation
- Parsed and aligned observations into state-action-next state tuples for model training while ensuring data integrity.
- Balanced fidelity and computational feasibility for efficient model training.

### 3. Incorporating Dynamics with Controllers
- Coupled smoothed learned dynamics with the controller framework
- Implemented regularization techniques and randomized smoothing to improve trajectory optimization performance.

## Methods
- **Regularization Techniques**: Included L2 regularization, gradient norm penalties, and Hessian norm penalties to enforce smoothness.
- **Randomized Smoothing**: Applied Monte Carlo-based smoothing to stabilize gradient estimates.

## Results
- **CartPole**: Improvements in average return and RMSE compared to unsmoothed models, though performance fell short of LQR baselines.
- **2D Quadrotor**: Smoothing methods showed promise in stabilizing trajectories but required further tuning for optimal results.

For more details, refer to our [paper](https://github.com/vigneshlaks/Learned-Dynamics-MEAM) and [codebase](https://github.com/vigneshlaks/Learned-Dynamics-MEAM).
