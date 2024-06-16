import tensorflow as tf
from environment import FogEnvironment
from agent import Agent
from agent import flatten_state
import numpy as np

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

def test_initialization():
    env = FogEnvironment()
    state = env.reset()
    input_dim = 3  # CPU, RAM, deployed for components; CPU, RAM for hosts
    hidden_dim = 24
    output_dim = len(state['components']) * len(state['hosts'])
    agent = Agent(input_dim, hidden_dim, output_dim)

    episodes = 3
    for e in range(episodes):
        state = env.reset()
        print(f"Reset state: {state}")
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            if action is None:
                print("No valid actions available. Ending episode.")
                done = True
                continue

            next_state, reward, done, _ = env.step(action)
            agent.learn([state], [action], [reward])
            state = next_state
            total_reward += reward

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    test_initialization()
