import tensorflow as tf
import numpy as np
import time
from environment import FogEnvironment
from agent import Agent, flatten_state
from InfrastructureGenerator import main

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

def measure_execution_time(num_hosts, density_factor, num_applications, episodes=1000):
    main(num_hosts, density_factor)  # Generate the network with specified parameters
    env = FogEnvironment(num_applications=num_applications)
    input_dim = 3
    hidden_dim = 256
    max_components = env.max_components
    num_hosts = env.num_hosts
    output_dim = max_components * num_hosts
    agent = Agent(input_dim, hidden_dim, output_dim, max_components, num_hosts, learning_rate=0.002)

    total_time = 0
    total_rewards = []

    for e in range(episodes):
        start_time = time.time()
        state = env.reset()
        done = False
        total_reward = 0
        episode_rewards = []

        for app_index in range(num_applications):
            while not done:
                action, action_index, explore = agent.choose_action(state, app_index)
                if action is None:
                    print(f"No valid actions available for app {app_index}. Ending deployment for this app.")
                    break  # Move to the next application
                next_state, reward, done, _ = env.step(action, app_index)
                
                # Store the experience in the appropriate replay buffer
                agent.store_experience(state, action_index, reward, next_state, done, app_index, explore)
                
                state = next_state
                total_reward += reward

                # Break if all components of the current app are deployed
                if all(comp['deployed'] for comp in state['applications'][app_index]['components'].values()):
                    break

        agent.learn(batch_size=64)  # Call learn after each episode
        end_time = time.time()
        total_time += end_time - start_time
        total_rewards.append(total_reward)
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Time: {end_time - start_time:.2f} seconds")

    avg_time_per_episode = total_time / episodes
    avg_reward_last_50 = np.mean(total_rewards[-50:])
    return avg_time_per_episode, avg_reward_last_50

if __name__ == "__main__":
    num_hosts_list = [50, 100]
    density_factors = [0.5, 1.5]  # Example density factors
    num_applications_list = [1, 5, 10]  # Varying the number of applications

    results = []

    for num_hosts in num_hosts_list:
        for density in density_factors:
            for num_apps in num_applications_list:
                avg_time, avg_reward_last_50 = measure_execution_time(num_hosts, density, num_apps)
                results.append((num_hosts, density, num_apps, avg_time, avg_reward_last_50))
                print(f"Hosts: {num_hosts}, Density: {density}, Applications: {num_apps}, Avg Time: {avg_time:.2f} seconds, Avg Reward (last 50): {avg_reward_last_50:.2f}")

    # Save results to a file
    with open('benchmark.txt', 'w') as f:
        for result in results:
            num_hosts, density, num_apps, avg_time, avg_reward_last_50 = result
            f.write(f"Hosts: {num_hosts}, Density: {density}, Applications: {num_apps}, Avg Time: {avg_time:.2f} seconds, Avg Reward (last 50): {avg_reward_last_50:.2f}\n")

    print("Results saved to benchmark.txt")
