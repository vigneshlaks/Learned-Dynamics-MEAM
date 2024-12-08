import torch
import numpy as np
from dynamics_learning.learner import SimpleDynamicsModel, L2RegularizedDynamicsModel, GradientNormPenaltyDynamicsModel
import torch.nn as nn


def load_learned_dynamics(model_type, model_path, obs_dim, action_dim, device):
    if model_type == 'l2':
        model = L2RegularizedDynamicsModel(obs_dim, action_dim).to(device)
    elif model_type == 'gradient_norm':
        model = GradientNormPenaltyDynamicsModel(obs_dim, action_dim).to(device)
    else:
        model = SimpleDynamicsModel(obs_dim, action_dim).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def smoothed_dynamics(f_model, x, u, epsilon, M, device):
    """
    Compute smoothed dynamics using Monte Carlo sampling.
    Args:
        f_model: Trained dynamics model.
        x: torch.Tensor, current state.
        u: torch.Tensor, current control input.
        epsilon: float, smoothing scale.
        M: int, number of Monte Carlo samples.
    Returns:
        Smoothed dynamics: torch.Tensor
    """
    noise = torch.randn((M, *u.shape), device=device)  # Sample Gaussian noise
    u_noisy = u + epsilon * noise  # Add noise to control inputs
    dynamics = torch.stack([f_model(x, u_n) for u_n in u_noisy])
    return dynamics.mean(dim=0)

# Trajectory Optimization
def trajectory_optimization(model_path, model_type, x0, N, epsilon, M, lr, max_iter, tol):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    obs_dim, action_dim = len(x0), 2  # Adjust action_dim based on your data
    f_model = load_learned_dynamics(model_type, model_path, obs_dim, action_dim, device)

    # Initialize control inputs
    u = torch.zeros((N, action_dim), requires_grad=True, device=device)
    x = torch.zeros((N + 1, obs_dim), device=device)
    x[0] = torch.tensor(x0, device=device)

    optimizer = torch.optim.Adam([u], lr=lr)

    for iteration in range(max_iter):
        total_cost = 0.0

        # Forward pass
        for t in range(N):
            x_t = x[t]
            u_t = u[t]
            x_next = smoothed_dynamics(f_model, x_t, u_t, epsilon, M, device)
            x[t + 1] = x_next
            total_cost += torch.sum(x_t**2 + u_t**2)  # Example quadratic cost

        # Backward pass
        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()

        print(f"Iteration {iteration}, Cost: {total_cost.item()}")
        if total_cost.item() < tol:
            print("Converged!")
            break

    return u.detach().cpu(), x.detach().cpu()

# Run Trajectory Optimization
if __name__ == "__main__":
    x0 = [1.0, 0.0, 0.0]  # Initial state
    N = 10  # Trajectory length
    epsilon = 0.1  # Smoothing scale
    M = 50  # Monte Carlo samples
    lr = 0.01  # Learning rate
    max_iter = 100
    tol = 1e-3
    model_path = "simple_dynamics_model.pth"
    model_type = "simple" 

    u_opt, x_opt = trajectory_optimization(model_path, model_type, x0, N, epsilon, M, lr, max_iter, tol)
