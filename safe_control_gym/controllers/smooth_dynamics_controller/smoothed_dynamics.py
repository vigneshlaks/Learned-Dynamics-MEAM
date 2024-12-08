import os
import torch
import numpy as np
from collections import defaultdict

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.ddpg.ddpg_utils import DDPGAgent, DDPGBuffer, make_action_noise_process
from smoothed_dynamics import smoothed_dynamics  # Import your smoothed dynamics function

class SmoothedDynamicsDDPG(BaseController):
    '''DDPG Controller with Smoothed Dynamics.'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='smoothed_model_latest.pt',
                 output_dir='temp',
                 use_gpu=False,
                 seed=0,
                 smoothing_scale=0.1,
                 num_samples=50,
                 **kwargs):
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)

        # Task setup
        if self.training:
            self.env = env_func(seed=seed)
        else:
            self.env = env_func()

        # Hyperparameters
        self.smoothing_scale = smoothing_scale  # ε: Smoothing scale
        self.num_samples = num_samples  # M: Number of Monte Carlo samples

        # Agent
        self.agent = DDPGAgent(self.env.observation_space,
                               self.env.action_space,
                               hidden_dim=self.hidden_dim,
                               gamma=self.gamma,
                               tau=self.tau,
                               actor_lr=self.actor_lr,
                               critic_lr=self.critic_lr,
                               activation=self.activation)
        self.agent.to(self.device)

        # Noise for exploration
        self.noise_process = None
        if self.random_process:
            self.noise_process = make_action_noise_process(self.random_process, self.env.action_space)

        # Replay buffer
        self.buffer = DDPGBuffer(self.env.observation_space, self.env.action_space,
                                 self.max_buffer_size, self.train_batch_size)

        # Logging
        self.logger = ExperimentLogger(output_dir, log_file_out=training, use_tensorboard=training)

    def compute_smoothed_dynamics(self, obs, action):
        """
        Compute the smoothed dynamics using Monte Carlo sampling.
        Args:
            obs (torch.Tensor): Current state/observation.
            action (torch.Tensor): Action input.
        Returns:
            Smoothed next state prediction.
        """
        return smoothed_dynamics(self.agent.ac.actor, obs, action, self.smoothing_scale, self.num_samples, self.device)

    def train_step(self):
        '''Performs a single training step with smoothed dynamics.'''
        self.agent.train()

        # Sample current batch
        batch = self.buffer.sample(self.train_batch_size, self.device)
        obs, act, rew, next_obs, mask = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['mask']

        # Smoothed next states
        with torch.no_grad():
            smoothed_next_states = self.compute_smoothed_dynamics(obs, act)

        # Update the agent
        results = self.agent.update({
            'obs': obs,
            'act': act,
            'rew': rew,
            'next_obs': smoothed_next_states,
            'mask': mask
        })

        return results

    def learn(self, **kwargs):
        '''Training loop for smoothed dynamics-based DDPG.'''
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)

        for step in range(self.max_env_steps):
            # Select action with exploration noise
            with torch.no_grad():
                action = self.agent.ac.act(obs)
                if self.noise_process:
                    action += torch.FloatTensor(self.noise_process.sample()).to(self.device)

            # Step environment
            next_obs, reward, done, info = self.env.step(action.cpu().numpy())
            next_obs = torch.FloatTensor(next_obs).to(self.device)

            # Store transition
            self.buffer.push({
                'obs': obs.cpu().numpy(),
                'act': action.cpu().numpy(),
                'rew': reward,
                'next_obs': next_obs.cpu().numpy(),
                'mask': 1 - float(done)
            })

            # Train step
            if len(self.buffer) >= self.train_batch_size:
                results = self.train_step()

                if step % self.log_interval == 0:
                    self.log_step(results)

            obs = next_obs if not done else torch.FloatTensor(self.env.reset()[0]).to(self.device)

            # Save checkpoint
            if step % self.save_interval == 0:
                self.save(self.checkpoint_path)

    def select_action(self, obs, info=None):
        '''Select an action using the current policy.'''
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action = self.agent.ac.act(obs)
        return action.cpu().numpy()

    def save(self, path):
        '''Saves the model.'''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.agent.state_dict(), path)

    def load(self, path):
        '''Loads the model.'''
        self.agent.load_state_dict(torch.load(path, map_location=self.device))
