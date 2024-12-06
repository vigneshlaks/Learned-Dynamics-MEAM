import numpy as np
import torch

# Learned Dynamics Model (Replace this with your actual model)
class LearnedDynamicsModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LearnedDynamicsModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )

    def forward(self, x, u):
        return self.network(torch.cat((x, u), dim=-1))

# Cost function (example: quadratic cost)
def cost_function(x, u):
    return torch.sum(x**2 + u**2)

# Randomized smoothing for the dynamics
def smoothed_dynamics(f_model, x, u, epsilon, M):
    """
    Compute smoothed dynamics using Monte Carlo sampling.
    Args:
        f_model: Learned dynamics model.
        x: torch.Tensor, current state.
        u: torch.Tensor, current control input.
        epsilon: float, smoothing scale.
        M: int, number of Monte Carlo samples.
    Returns:
        Smoothed dynamics: torch.Tensor
    """
    noise = torch.randn(M, *u.shape)  # Sample noise Z ~ N(0, I)
    u_noisy = u + epsilon * noise  # Add noise to control input
    dynamics = torch.stack([f_model(x, u_n) for u_n in u_noisy])
    return dynamics.mean(dim=0)

# Trajectory Optimization Algorithm
def trajectory_optimization(f_model, x0, N, epsilon, M, learning_rate, max_iter, tol):
    """
    Trajectory optimization with randomized smoothing.
    Args:
        f_model: Learned dynamics model.
        x0: torch.Tensor, initial state.
        N: int, trajectory length.
        epsilon: float, smoothing scale.
        M: int, number of Monte Carlo samples.
        learning_rate: float, gradient descent step size.
        max_iter: int, maximum iterations.
        tol: float, convergence threshold.
    Returns:
        Optimized control inputs and state trajectory.
    """

    u = torch.zeros(N, x0.shape[0], requires_grad=True)  # Control inputs
    x = torch.zeros(N + 1, x0.shape[0])  
    x[0] = x0

    optimizer = torch.optim.Adam([u], lr=learning_rate)

    for iteration in range(max_iter):
        total_cost = 0.0


        for t in range(N):
            x_t = x[t]
            u_t = u[t]

            x_next = smoothed_dynamics(f_model, x_t, u_t, epsilon, M)
            x[t + 1] = x_next

            total_cost += cost_function(x_t, u_t)


        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()


        if total_cost.item() < tol:
            print(f"Converged at iteration {iteration}.")
            break

        print(f"Iteration {iteration}, Cost: {total_cost.item()}")

    return u.detach(), x.detach()

if __name__ == "__main__":
    torch.manual_seed(0)

    # Initialize dynamics model
    input_dim = 6  # State + control input dimension
    output_dim = 3  # Output dimension of the dynamics
    f_model = LearnedDynamicsModel(input_dim, output_dim)

    # Initial state
    x0 = torch.tensor([1.0, 0.0, 0.0])


    N = 10  # Trajectory length
    epsilon = 0.1  # Smoothing scale
    M = 50  # Number of Monte Carlo samples
    learning_rate = 0.01
    max_iter = 100
    tol = 1e-3


    optimized_u, optimized_x = trajectory_optimization(
        f_model, x0, N, epsilon, M, learning_rate, max_iter, tol
    )

    print("Optimized control inputs:")
    print(optimized_u)
    print("Optimized state trajectory:")
    print(optimized_x)
