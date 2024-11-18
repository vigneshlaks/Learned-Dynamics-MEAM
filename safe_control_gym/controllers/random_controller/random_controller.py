# random_controller_experiment.py

import numpy as np

import casadi as cs
import numpy as np
from termcolor import colored

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.mpc_utils import (compute_discrete_lqr_gain_from_cont_linear_system,
                                                        compute_state_rmse, get_cost_weight_matrix,
                                                        reset_constraints, rk_discrete)
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list
from safe_control_gym.utils.utils import timing

class RandomController(BaseController):
    '''A controller that selects random actions within the action space of the environment.'''

    def __init__(
            self,
            env_func,
            # runner args
            output_dir: str = 'results/temp',
            additional_constraints: list = None,
            use_gpu: bool = False,
            seed: int = 0,
            **kwargs
    ):
        '''Initializes the RandomController.

        Args:
            env_func (callable): A function that creates an instance of the environment.
            **kwargs: Additional arguments (not used here but kept for compatibility).
        '''
        # CHECK TO MAKE SURE THIS IS SOMETHING WE REALLY WANT TO DO
        super().__init__(env_func=env_func, output_dir=output_dir, use_gpu=use_gpu, seed=seed, **kwargs)

        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})

        # Task.
        self.env = env_func()

        '''
        if additional_constraints is not None:
            additional_ConstraintsList = create_constraint_list(additional_constraints,
                                                                GENERAL_CONSTRAINTS,
                                                                self.env)
            self.additional_constraints = additional_ConstraintsList.constraints
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(self.env.constraints.constraints + self.additional_constraints)
        else:
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(self.env.constraints.constraints)
            self.additional_constraints = []
        '''

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def select_action(self, obs, info=None):
        '''Selects a random action.

        Args:
            obs (ndarray): The current observation.
            info (dict): Additional info (not used here).

        Returns:
            action (ndarray): A random action within the action space.
        '''
        random_action = self.env.action_space.sample()  
        if hasattr(self, "input_constraints_sym"):
            for constraint in self.input_constraints_sym:
                if not constraint(random_action):
                    random_action = self.env.action_space.sample()  
        return random_action

    def reset(self):
        '''Resets the controller state if necessary (not used here).'''
        print(colored('Resetting Random Controller', 'green'))
        
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.setup_results_dict()

    def reset_before_run(self, obs, info=None, env=None):
        '''Resets the controller state before a new run (not used here).

        Args:
            obs (ndarray): The initial observation.
            info (dict): Additional info.
            env (gym.Env): The environment instance.
        '''
        self.reset()

    def learn(self, env, **kwargs):
        '''No learning for the RandomController.'''
        pass

    def close(self):
        '''Cleans up resources if necessary (not used here).'''
        self.env.close()

    def save(self, path):
        '''Saves the controller state (not applicable for RandomController).

        Args:
            path (str): The path to save the controller.
        '''
        pass

    def load(self, path):
        '''Loads the controller state (not applicable for RandomController).

        Args:
            path (str): The path to load the controller.
        '''
        pass
