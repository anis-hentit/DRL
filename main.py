import tensorflow as tf
from environment import FogEnvironment  # Import the custom environment class
from agent import Agent  # Import the custom agent class
from agent import flatten_state  # Import the utility function to flatten the state
import numpy as np

# Print TensorFlow and Keras versions for verification
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

def test_initialization():
    """
    Initialize the environment and the agent, and run multiple training episodes.
    """
    # Create an instance of the FogEnvironment
    env = FogEnvironment()

    # Reset the environment to get the initial state
    state = env.reset()

    # Define the dimensions for the agent
    input_dim = 3  # Number of features per node: CPU, RAM, deployed status for components; CPU, RAM for hosts
    hidden_dim = 12  # Number of hidden units in the GNN layers
    output_dim = len(state['components']) * len(state['hosts'])  # Total number of possible actions (component-host pairs)

    # Initialize the Agent with specified dimensions and learning rate
    agent = Agent(input_dim, hidden_dim, output_dim, learning_rate=0.0001)

    # Number of episodes to train the agent
    episodes = 1000
    for e in range(episodes):
        # Reset the environment at the start of each episode
        state = env.reset()
        print(f"Reset state: {state}")  # Print the reset state for debugging

        done = False  # Flag to indicate if the episode has ended
        total_reward = 0  # Initialize total reward for the episode
        episode_states = []  # List to store states encountered in the episode
        episode_actions = []  # List to store actions taken in the episode
        episode_rewards = []  # List to store rewards received in the episode
        
        while not done:
            # Agent selects an action based on the current state
            action = agent.choose_action(state)

            if action is None:
                print("No valid actions available. Ending episode.")  # Debugging output for no valid actions
                done = True
                continue

            # Perform the chosen action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)

            # Store the state, action, and reward for the current step
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            # Update the current state to the next state
            state = next_state

            # Accumulate the total reward for the episode
            total_reward += reward

        # After each episode, update the agent using the experiences (states, actions, rewards)
        agent.learn(episode_states, episode_actions, episode_rewards)
        
        # Print the total reward for the current episode for monitoring
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    test_initialization()
