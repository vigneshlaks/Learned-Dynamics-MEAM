import pickle
import numpy as np
import os

def augment_data(pkl_path='/Users/valaksh/Desktop/thingyforcont/safe-control-gym/examples/temp-data/mpc_data_quadrotor_stabilization.pkl',
                 output_path='augmented_data.npz'):
    """
    Load .pkl file, extract obs, actions, and next_obs, and save for ML training.
    """
    # Load the data from the pickle file
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    print("Data Type:", type(data))

    # Check if required keys are present in the data
    if 'trajs_data' not in data or 'obs' not in data['trajs_data'] or 'action' not in data['trajs_data']:
        raise ValueError("The provided pickle file does not have the expected keys.")
    
    # Convert 'obs' and 'actions' from lists of arrays to single NumPy arrays
    obs = np.concatenate(data['trajs_data']['obs'], axis=0)[:-1]  # Exclude the last observation
    actions = np.concatenate(data['trajs_data']['action'], axis=0)  # Combine all actions

    # Ensure obs and actions are consistent
    if len(obs) != len(actions):
        print(len(obs))
        print(len(actions))
        raise ValueError("Mismatch between the number of observations and actions.")

    # Create next_obs by shifting obs
    next_obs = obs[1:]  # All observations except the first
    obs = obs[:-1]  # All observations except the last
    actions = actions[:-1]  # Ensure actions align with obs and next_obs

    # Print shapes for debugging
    #print("Observation Shape:", obs.shape)
    #print("Action Shape:", actions.shape)
    #print("Next Observation Shape:", next_obs.shape)

    #for i in range(5):  # Compare the first 5 examples
        #print(f"obs[{i}]: {obs[i]}")
        #print(f"next_obs[{i}]: {next_obs[i]}")
        #print("---")

    # Save the processed data as .npz
    np.savez(output_path, obs=obs, actions=actions, next_obs=next_obs)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    augment_data()
