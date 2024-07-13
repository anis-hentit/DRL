import tensorflow as tf
import numpy as np
import time
from environment import FogEnvironment
from agent import Agent, flatten_state

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

def has_converged(rewards, threshold=0.01, window=30):
    if len(rewards) < 2 * window:
        return False
    recent_rewards = rewards[-window:]
    previous_rewards = rewards[-2*window:-window]
    improvement = np.mean(recent_rewards) - np.mean(previous_rewards)
    variance = np.var(recent_rewards)
    return improvement < threshold and variance < threshold

def test_initialization():
    env = FogEnvironment()
    state = env.reset()
    input_dim = 3
    hidden_dim = 128
    max_components = env.max_components
    num_hosts = env.num_hosts
    output_dim = max_components * num_hosts
<<<<<<< HEAD
    agent = Agent(input_dim, hidden_dim, output_dim, max_components, num_hosts, learning_rate=0.001)

    rewards = []
    start_time = time.time()
    max_training_time = 10*3600  # 10 hours in seconds
=======
    agent = Agent(input_dim, hidden_dim, output_dim, max_components, num_hosts, learning_rate=0.002)

    rewards = []
    start_time = time.time()
    max_training_time = 1200000000000  # 3 hour in seconds
>>>>>>> cb6bf5738e4f5823784affb5113bb4ecfac2527c
    window = 30
    convergence_threshold = 0.1

    e = 0
    while True:
        state = env.reset()
        done = False
        total_reward = 0
<<<<<<< HEAD

        for app_index in range(len(state['applications'])):
            while not done:
                action, action_index, explore = agent.choose_action(state, app_index)
=======
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_app_indices = []

        for app_index in range(len(state['applications'])):
            while not done:
                action = agent.choose_action(state, app_index)
>>>>>>> cb6bf5738e4f5823784affb5113bb4ecfac2527c
                if action is None:
                    print(f"No valid actions available for app {app_index}. Ending deployment for this app.")
                    break  # Move to the next application
                next_state, reward, done, _ = env.step(action, app_index)
<<<<<<< HEAD
                
                # Store the experience in the appropriate replay buffer
                agent.store_experience(state, action_index, reward, next_state, done, app_index, explore)
                
=======
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_app_indices.append(app_index)
>>>>>>> cb6bf5738e4f5823784affb5113bb4ecfac2527c
                state = next_state
                total_reward += reward

                # Break if all components of the current app are deployed
                if all(comp['deployed'] for comp in state['applications'][app_index]['components'].values()):
                    break

<<<<<<< HEAD
        agent.learn(batch_size=64)  # Call learn after each episode
=======
        agent.learn(episode_states, episode_actions, episode_rewards, episode_app_indices)
>>>>>>> cb6bf5738e4f5823784affb5113bb4ecfac2527c
        rewards.append(total_reward)
        e += 1
        elapsed_time = time.time() - start_time

        print(f"Episode {e}, Total Reward: {total_reward}")

        if has_converged(rewards, convergence_threshold, window) or elapsed_time >= max_training_time:
            break

    print(f"Training completed in {e} episodes and {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    test_initialization()