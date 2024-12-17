import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse


# Augment Data Function
def augment_data(pkl_path='/Users/valaksh/Desktop/thingyforcont/safe-control-gym/examples/temp-data/mpc_data_quadrotor_stabilization.pkl',
                 output_path='augmented_data.npz'):
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    obs = np.concatenate(data['trajs_data']['obs'], axis=0)[:-1]
    actions = np.concatenate(data['trajs_data']['action'], axis=0)
    if len(obs) != len(actions):
        raise ValueError("Mismatch between the number of observations and actions.")

    next_obs = obs[1:]
    obs = obs[:-1]
    actions = actions[:-1]

    np.savez(output_path, obs=obs, actions=actions, next_obs=next_obs)
    print(f"Data saved to {output_path}")


# Dataset Class
class DynamicsDataset(Dataset):
    def __init__(self, obs, actions, next_obs):
        self.obs = obs
        self.actions = actions
        self.next_obs = next_obs

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs_sample = self.obs[idx].astype(np.float32)
        action_sample = self.actions[idx].astype(np.float32)
        next_obs_sample = self.next_obs[idx].astype(np.float32)
        return obs_sample, action_sample, next_obs_sample


# Models
class SimpleDynamicsModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super(SimpleDynamicsModel, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, obs_dim)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class GradientNormPenaltyDynamicsModel(SimpleDynamicsModel):
    def gradient_norm_penalty(self, inputs, outputs, threshold=1.0):
        grads = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                                    create_graph=True, retain_graph=True)[0]
        penalty = torch.mean((grads.norm(2, dim=-1) - threshold) ** 2)
        return penalty


class HessianNormPenaltyDynamicsModel(SimpleDynamicsModel):
    def hessian_norm_penalty(self, inputs, outputs):
        hessians = torch.autograd.functional.hessian(lambda x: self.forward(x[:, :inputs.shape[1]], x[:, inputs.shape[1]:]), inputs)
        penalty = sum(torch.sum(h ** 2) for h in hessians)
        return penalty


# Training Function
def train_dynamics_model(model_type='simple', data_path='augmented_data.npz', epochs=50, batch_size=64, learning_rate=1e-3):
    # Load the data
    data = np.load(data_path)
    obs, actions, next_obs = data['obs'], data['actions'], data['next_obs']

    obs_train, obs_val, actions_train, actions_val, next_obs_train, next_obs_val = train_test_split(
        obs, actions, next_obs, test_size=0.2, random_state=42
    )

    # Data loaders
    train_loader = DataLoader(DynamicsDataset(obs_train, actions_train, next_obs_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DynamicsDataset(obs_val, actions_val, next_obs_val), batch_size=batch_size, shuffle=False)

    # Model Selection
    obs_dim, action_dim = obs.shape[1], actions.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'l2':
        model = L2RegularizedDynamicsModel(obs_dim, action_dim).to(device)
    elif model_type == 'gradient_norm':
        model = GradientNormPenaltyDynamicsModel(obs_dim, action_dim).to(device)
    elif model_type == 'hessian_norm':
        model = HessianNormPenaltyDynamicsModel(obs_dim, action_dim).to(device)
    else:
        model = SimpleDynamicsModel(obs_dim, action_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4 if model_type == 'l2' else 0)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for obs_batch, action_batch, next_obs_batch in train_loader:
            obs_batch, action_batch, next_obs_batch = obs_batch.to(device), action_batch.to(device), next_obs_batch.to(device)
            optimizer.zero_grad()

            pred_next_obs = model(obs_batch, action_batch)
            loss = criterion(pred_next_obs, next_obs_batch)

            # Add penalties for specific models
            if model_type == 'gradient_norm':
                loss += 0.1 * model.gradient_norm_penalty(obs_batch, pred_next_obs)
            elif model_type == 'hessian_norm':
                loss += 0.05 * model.hessian_norm_penalty(torch.cat([obs_batch, action_batch], dim=-1), pred_next_obs)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * obs_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, action_batch, next_obs_batch in val_loader:
                obs_batch, action_batch, next_obs_batch = obs_batch.to(device), action_batch.to(device), next_obs_batch.to(device)
                pred_next_obs = model(obs_batch, action_batch)
                val_loss += criterion(pred_next_obs, next_obs_batch).item() * obs_batch.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    torch.save(model.state_dict(), f'{model_type}_dynamics_model.pth')
    print(f"Model saved as {model_type}_dynamics_model.pth")


# Main Function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'l2', 'gradient_norm', 'hessian_norm'],
                        help="Choose the dynamics model: 'simple', 'l2', 'gradient_norm', or 'hessian_norm'")
    args = parser.parse_args()

    augment_data()
    train_dynamics_model(model_type=args.model)
