import tensorflow as tf
import numpy as np
import time
import os
from environment import FogEnvironment
from agent import Agent, flatten_state
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    topology_files = ["DRL/data/scaled_infrastructure_H50_L808T.json"]
    input_dim = 3
    hidden_dim = 64
    max_components = 6  # Set this to the maximum number of components we expect can make it dynamic from the env class ( "env.max_components")
    num_hosts = 50  # Set to the maximum number of hosts across all topologies

    agent = Agent(input_dim, hidden_dim, max_components, num_hosts, learning_rate=0.002, num_layers=3,attention_type="gat_v2",attention_num_heads=4)

    load_path = 'gnn_agent_model_256_boltz_H50L808TRY'
    save_path = 'gnn_agent_model_256_boltz_H50L808TRY'
    if os.path.exists(load_path):
        print("Loading previous model...")
        agent.load(load_path)
    else:
        print("No previous model found. Starting from scratch.")

    rewards = []
    start_time = time.time()
    max_training_time = 100 * 3600  # 100 hours in seconds
    max_episodes = 7000  # Training for 6000 episodes
    window = 30
    convergence_threshold = 0.1

    e = 0
    while e < max_episodes:
        for topology_file in topology_files:
            if e >= max_episodes:
                break

            env = FogEnvironment(topology_file)
            state = env.reset()
            total_reward = 0
            done = False
            current_num_hosts = env.get_num_hosts()  # Get the current number of hosts

            while not done:
                for app_index in range(len(state['applications'])):
                    while not all(comp['deployed'] for comp in state['applications'][app_index]['components'].values()):
                        action, action_index, explore = agent.choose_action(state, app_index, num_hosts=current_num_hosts)
                        if action is None:
                            print(f"No valid actions available for app {app_index}. Ending deployment for this app.")
                            break  # Move to the next application
                        next_state, reward, done, _ = env.step(action, app_index)

                        agent.store_experience(state, action_index, reward, next_state, done, app_index, explore)

                        state = next_state
                        total_reward += reward

                done = env.check_all_deployed() or env.current_step >= env.max_steps_per_episode

            agent.learn(batch_size=64)  # Call learn after each episode
            rewards.append(total_reward)
            e += 1
            elapsed_time = time.time() - start_time

            print(f"Episode {e} with topology {topology_file}, Total Reward: {total_reward}")

            if e % 500 == 0:
                print(f"Saving model at episode {e}...")
                agent.save(save_path)

            if has_converged(rewards, convergence_threshold, window) or elapsed_time >= max_training_time:
                break

        if has_converged(rewards, convergence_threshold, window) or elapsed_time >= max_training_time:
            break

    print(f"Training completed in {e} episodes and {elapsed_time:.2f} seconds.")
    agent.save(save_path)  # Save the trained model

if __name__ == "__main__":
    test_initialization()
