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
    topology_files = ["DRL/data/scaled_infrastructure_H100_L7250.json", "DRL/data/scaled_infrastructure_H50_L885.json"]
    input_dim = 3
    hidden_dim = 256
    max_components = 6  # Set this to the maximum number of components we expect can make it dynamic from the env class ( "env.max_components")
    num_hosts = 100  # Set to the maximum number of hosts across all topologies
    agent = Agent(input_dim, hidden_dim, max_components, num_hosts, learning_rate=0.002)

    rewards = []
    start_time = time.time()
    max_training_time = 5*3600  # 5 hours in seconds
    max_episodes = 3000 # Training for 1500 episodes
    window = 30
    convergence_threshold = 0.1

    e = 0
    while e < max_episodes:
        for topology_file in topology_files:
            if e >= max_episodes:
                break

            env = FogEnvironment(topology_file)
            state = env.reset()
            done = False
            total_reward = 0

            for app_index in range(len(state['applications'])):
                while not done:
                    action, action_index, explore = agent.choose_action(state, app_index)
                    if action is None:
                        print(f"No valid actions available for app {app_index}. Ending deployment for this app.")
                        break  # Move to the next application
                    next_state, reward, done, _ = env.step(action, app_index)
                    
                    agent.store_experience(state, action_index, reward, next_state, done, app_index, explore)
                    
                    state = next_state
                    total_reward += reward

                    if all(comp['deployed'] for comp in state['applications'][app_index]['components'].values()):
                        break

            agent.learn(batch_size=64)  # Call learn after each episode
            rewards.append(total_reward)
            e += 1
            elapsed_time = time.time() - start_time

            print(f"Episode {e} with topology {topology_file}, Total Reward: {total_reward}")

            if has_converged(rewards, convergence_threshold, window) or elapsed_time >= max_training_time:
                break

        if has_converged(rewards, convergence_threshold, window) or elapsed_time >= max_training_time:
            break

    print(f"Training completed in {e} episodes and {elapsed_time:.2f} seconds.")
    agent.save('gnn_agent_model')  # Save the trained model

if __name__ == "__main__":
    test_initialization()
