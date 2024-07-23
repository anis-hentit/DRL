import tensorflow as tf
import numpy as np
import random
import time
from environment import FogEnvironment
from agent import flatten_state, Agent, compute_action_mask

def run_inference(agent, env, num_episodes, num_applications):
    total_time = 0
    rewards = []
    latency_penalties = []

    all_app_indices = list(range(len(env.state['applications'])))

    for episode in range(num_episodes):
        state = env.reset()
        selected_apps = random.sample(all_app_indices, num_applications)
        random.shuffle(selected_apps)
        episode_reward = 0
        episode_latency_penalty = 0
        done = False
        start_time = time.time()

        while not done:
            all_deployed = True
            for app_index in selected_apps:
                num_hosts = len(state['hosts'])
                action, action_index, explore = agent.choose_action(state, app_index, mode='inference', num_hosts=num_hosts)
                if action is None:
                    continue  # Skip to next app if no valid actions available

                next_state, reward, _, _ = env.step(action, app_index)
                episode_reward += reward
                state = next_state

                # Check if all components are deployed
                app_state = state['applications'][app_index]
                for comp_id, comp in app_state['components'].items():
                    if not comp['deployed']:
                        all_deployed = False

            if all_deployed:
                done = True  # End the episode if all components are deployed

        for app_index in selected_apps:
            app_state = state['applications'][app_index]
            num_links = len(app_state['links'])
            latency_penalty = sum(penalty for penalty in app_state['latency_penalties'].values())
            episode_latency_penalty += latency_penalty / num_links if num_links > 0 else 0

        end_time = time.time()
        total_time += end_time - start_time
        rewards.append(episode_reward / num_applications)
        latency_penalties.append(episode_latency_penalty / num_applications)

        # Log the deployment status of each component
        print(f"\n--- End of Episode {episode} ---")
        for app_index in selected_apps:
            app_state = state['applications'][app_index]
            for comp_id, comp in app_state['components'].items():
                status = "Deployed" if comp['deployed'] else "Not Deployed"
                print(f"App {app_index} Component {comp_id}: {status} on Host {comp.get('host', 'N/A')}")

    avg_time = total_time / num_episodes
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    avg_latency_penalty = np.mean(latency_penalties)
    std_latency_penalty = np.std(latency_penalties)
    min_latency_penalty = np.min(latency_penalties)
    max_latency_penalty = np.max(latency_penalties)

    return (avg_time, avg_reward, std_reward, min_reward, max_reward,
            avg_latency_penalty, std_latency_penalty, min_latency_penalty, max_latency_penalty)

# Load the pre-trained model including dense layers
input_dim = 3
hidden_dim = 258
max_components = 6  # Set this to the maximum number of components we expect can make it dynamic from the env class ( "env.max_components")
num_hosts = 100  # Should be set to the maximum number of hosts across all topologies

agent = Agent(input_dim, hidden_dim, max_components, num_hosts, learning_rate=0.002)
agent.load('gnn_agent_model_256')

num_episodes = 10
application_configs = [1]
topology_files = ["DRL/data/scaled_infrastructure_H100_L3625.json","DRL/data/scaled_infrastructure_H100_L1551.json","DRL/data/scaled_infrastructure_H50_L885.json",
"DRL/data/scaled_infrastructure_H50_L542.json"]

metrics_filename = "metrics_log.txt"

with open(metrics_filename, "a") as file:
    for topology_file in topology_files:
        env = FogEnvironment(topology_file, mode='inference')
        file.write(f"\n--- Running inference on topology {topology_file} ---\n")
        print(f"\n--- Running inference on topology {topology_file} ---\n")

        for num_applications in application_configs:
            print(f"\nRunning inference with {num_applications} applications...")
            (avg_time, avg_reward, std_reward, min_reward, max_reward,
             avg_latency_penalty, std_latency_penalty, min_latency_penalty, max_latency_penalty) = run_inference(agent, env, num_episodes, num_applications)

            print(f"Avg Time per Episode: {avg_time:.4f} seconds")
            print(f"Avg Reward: {avg_reward:.4f} per application")
            print(f"Std Reward: {std_reward:.4f}")
            print(f"Min Reward: {min_reward:.4f}")
            print(f"Max Reward: {max_reward:.4f}")
            print(f"Avg Latency Penalty: {avg_latency_penalty:.4f} per application")
            print(f"Std Latency Penalty: {std_latency_penalty:.4f}")
            print(f"Min Latency Penalty: {min_latency_penalty:.4f}")
            print(f"Max Latency Penalty: {max_latency_penalty:.4f}")

            file.write(f"\nRunning inference with {num_applications} applications...\n")
            file.write(f"Avg Time per Episode: {avg_time:.4f} seconds\n")
            file.write(f"Avg Reward: {avg_reward:.4f} per application\n")
            file.write(f"Std Reward: {std_reward:.4f}\n")
            file.write(f"Min Reward: {min_reward:.4f}\n")
            file.write(f"Max Reward: {max_reward:.4f}\n")
            file.write(f"Avg Latency Penalty: {avg_latency_penalty:.4f} per application\n")
            file.write(f"Std Latency Penalty: {std_latency_penalty:.4f}\n")
            file.write(f"Min Latency Penalty: {min_latency_penalty:.4f}\n")
            file.write(f"Max Latency Penalty: {max_latency_penalty:.4f}\n")
