import tensorflow as tf
from environment import FogEnvironment
from agent import Agent, flatten_state
import numpy as np
import time
from InfrastructureGenerator import main

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

def measure_execution_time(num_hosts, density_factor, num_applications):
    main(num_hosts, density_factor)  # Generate the network with specified parameters
    env = FogEnvironment()
    input_dim = 3
    hidden_dim = 128
    max_components = env.max_components
    num_hosts = env.num_hosts
    output_dim = max_components * num_hosts
    agent = Agent(input_dim, hidden_dim, output_dim, max_components, num_hosts, learning_rate=0.002)
    episodes = 1  # Increase this to average over multiple episodes

    total_time = 0
    for e in range(episodes):
        start_time = time.time()
        state = env.reset()
        done = False
        total_reward = 0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_app_indices = []

        for app_index in range(num_applications):
            while not done:
                action = agent.choose_action(state, app_index)
                if action is None:
                    print(f"No valid actions available for app {app_index}. Ending deployment for this app.")
                    break  # Move to the next application
                next_state, reward, done, _ = env.step(action, app_index)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_app_indices.append(app_index)
                state = next_state
                total_reward += reward

                # Break if all components of the current app are deployed
                if all(comp['deployed'] for comp in state['applications'][app_index]['components'].values()):
                    break

        agent.learn(episode_states, episode_actions, episode_rewards, episode_app_indices)
        end_time = time.time()
        total_time += end_time - start_time
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Time: {end_time - start_time:.2f} seconds")

    avg_time_per_episode = total_time / episodes
    return avg_time_per_episode

if __name__ == "__main__":
    num_hosts_list = [50,100,200]
    density_factors = [0.5,1.0,1.5]  # Example density factors
    num_applications_list = [10] # currently max is 10 since there is 10 apps in the json

    results = []

    for num_hosts in num_hosts_list:
        for density in density_factors:
            for num_apps in num_applications_list:
                avg_time = measure_execution_time(num_hosts, density, num_apps)
                results.append((num_hosts, density, num_apps, avg_time))
                print(f"Hosts: {num_hosts}, Density: {density}, Applications: {num_apps}, Avg Time: {avg_time:.2f} seconds")

    print("Results:", results)
