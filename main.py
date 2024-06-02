from environment import FogEnvironment
from agent import Agent
from agent import flatten_state
import numpy as np

def calculate_state_size(state):
    flattened_state = flatten_state(state)
    return len(flattened_state)

def test_initialization():
    env = FogEnvironment()
    state = env.reset()
    print(f"Initial state: {state}")
    state_size = calculate_state_size(state)
    num_components = len(state['components'])
    num_hosts = len(state['hosts'])
    agent = Agent(state_size, num_components, num_hosts, env.infra_config)

    episodes = 10
    for e in range(episodes):
        state = env.reset()
        print(f"Reset state: {state}")
        print(f"Reset state flattened size: {flatten_state(state).shape}")
        done = False
        total_reward = 0
        
        while not done:
            print(f"iteration number: {e}")
            action = agent.choose_action(state)
            print(f"Action chosen: {action}")
            next_state, reward, done, _ = env.step(action)
            print(f"Next state after action: {next_state}")
            agent.learn([state], [action], [reward])
            state = next_state
            total_reward += reward
            print(f"Total reward: {total_reward}")

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    test_initialization()
