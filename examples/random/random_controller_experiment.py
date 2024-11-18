# random_controller_experiment.py

import os
import pickle
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

# Import the RandomController
from safe_control_gym.controllers.random_controller.random_controller import RandomController

import yaml
def run(gui=True, n_episodes=1, n_steps=None, save_data=False):
    '''Runs the experiment with the RandomController.

    Args:
        gui (bool): Whether to display the GUI and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    random_env = env_func(gui=False)
    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )


    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes

    # Run the experiment
    for _ in range(n_episodes):
        # Get initial state and create environments
        init_state = random_env.reset()
        static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

        # Create the experiment
        experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)

        # No training for RandomController
        # experiment.launch_training()

        # Run evaluation
        if n_steps is None:
            trajs_data, _ = experiment.run_evaluation(training=False, n_episodes=1)
        else:
            trajs_data, _ = experiment.run_evaluation(training=False, n_steps=n_steps)

        if gui:
            post_analysis(trajs_data['obs'][0], trajs_data['action'][0], static_env)

        # Close environments
        static_env.close()
        static_train_env.close()

        # Collect trajectory data
        for key, value in trajs_data.items():
            all_trajs[key] += value

    ctrl.close()
    random_env.close()
    metrics = experiment.compute_metrics(all_trajs)
    all_trajs = dict(all_trajs)

    # Save data if required
    if save_data:
        results = {'trajs_data': all_trajs, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/random_controller_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    # Print final metrics
    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))

def post_analysis(state_stack, input_stack, env):
    '''Plots the input and states to analyze performance.

    Args:
        state_stack (ndarray): The list of observations from the latest run.
        input_stack (ndarray): The list of actions from the latest run.
        env (gym.Env): The environment instance.
    '''
    model = env.symbolic
    stepsize = model.dt

    plot_length = min(len(input_stack), len(state_stack))
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

    # Plot states
    fig, axs = plt.subplots(model.nx, 1, figsize=(10, 8))
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, :plot_length], label='Actual')
        axs[k].plot(times, reference.transpose()[k, :plot_length], color='r', label='Desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
        if k != model.nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend()
    axs[-1].set(xlabel='Time (sec)')

    # Plot inputs
    fig, axs = plt.subplots(model.nu, 1, figsize=(10, 6))
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, np.array(input_stack).transpose()[k, :plot_length])
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='Time (sec)')

    plt.show()

if __name__ == '__main__':
    run()
