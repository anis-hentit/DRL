

## DRL

This project involves deploying components in a fog computing environment using a reinforcement learning (RL) agent. The RL agent learns optimal deployment strategies to maximize resource utilization and minimize latency and bandwidth penalties. The project consists of three main modules: the agent, the environment, and the main execution script.

## File Structure

- `agent.py`: Contains the implementation of the RL agent.
- `environment.py`: Defines the fog computing environment where the agent operates.
- `main.py`: The main script to initialize the environment, train the agent, and test the deployment strategies.

## Agent

### Overview

The agent is implemented using TensorFlow and Keras. It employs a neural network to predict the best actions based on the current state of the environment.

### Hyperparameters

| Hyperparameter   | Description                                               | Value  |
|------------------|-----------------------------------------------------------|--------|
| `learning_rate`  | Learning rate for the neural network optimizer            | 0.001  |
| `gamma`          | Discount factor for future rewards                        | 0.99   |
| `epsilon`        | Initial exploration rate for the epsilon-greedy policy    | 1.0    |
| `epsilon_decay`  | Decay rate for epsilon after each episode                 | 0.995  |
| `epsilon_min`    | Minimum value for epsilon to ensure exploration           | 0.01   |

### Methods

- `__init__(self, state_size, num_components, num_hosts, infra_config, learning_rate=0.001)`: Initializes the agent with the given parameters.
- `build_model(self)`: Constructs the neural network model.
- `choose_action(self, state)`: Chooses an action based on the current state using an epsilon-greedy policy.
- `decode_action(self, action_index, state)`: Decodes the action index into a specific component-host deployment.
- `encode_action(self, action)`: Encodes the action for learning.
- `generate_paths(self, component_id, host_id, state)`: Generates valid paths between components.
- `learn(self, states, actions, rewards)`: Updates the neural network based on the experience.
- `dijkstra(self, source, target, latency_req, bandwidth_req)`: Finds the shortest path between hosts considering latency and bandwidth requirements using Dijkstra's algorithm.

### State Representation

The state is represented as a combination of host resources and component requirements, including deployment status:

- **Hosts**: Represented as a dictionary with CPU and RAM resources.
- **Components**: Represented as a dictionary with CPU, RAM requirements, and deployment status.
- **Fixed Positions**: Components that are fixed to specific hosts.
- **Links**: Logical links between components with bandwidth and latency requirements.
- **Paths**: Physical paths between hosts for each logical link.

#### Flattening the State

The `flatten_state` function converts the state dictionary into a flat numpy array for input into the neural network. It concatenates host values, component values (excluding deployment status), and deployment statuses into a single array.

## Environment

### Overview

The `FogEnvironment` class represents the fog computing environment where components are deployed to hosts. It simulates the interaction between the agent and the environment, providing rewards based on the deployment's efficiency.

### Methods

- `__init__(self)`: Initializes the environment with application and infrastructure configurations.
- `define_action_space(self)`: Defines the action space as a combination of components and hosts.
- `define_observation_space(self)`: Defines the observation space representing the state.
- `initialize_state(self)`: Initializes the environment state, deploying fixed components.
- `step(self, action)`: Executes an action and updates the state.
- `validate_path(self, path, latency_req, bandwidth_req)`: Validates a path based on latency and bandwidth requirements.
- `calculate_state_size(self, state)`: Calculates the size of the flattened state.
- `update_state(self, component_id, host_id, path_ids)`: Updates the state after deploying a component.
- `check_all_deployed(self)`: Checks if all components have been deployed.
- `calculate_reward(self, action, path_ids)`: Calculates the reward based on energy consumption and penalties.
- `calculate_latency_penalty(self, path_ids)`: Calculates latency penalties for the paths.
- `calculate_bandwidth_penalty(self, path_ids)`: Calculates bandwidth penalties for the paths.
- `reset(self)`: Resets the environment to the initial state.
- `render(self, mode='console')`: Renders the current state of the environment.
- `close(self)`: Closes the environment.

### State Initialization

The `initialize_state` method sets up the initial state by:

1. **Hosts**: Setting up each host with its CPU and RAM resources.
2. **Components**: Setting up each component with its CPU and RAM requirements and marking them as not deployed.
3. **Fixed Positions**: Deploying components fixed to specific hosts.
4. **Links**: Setting up logical links between components.
5. **Paths**: Initializing paths for each logical link as empty.

## Main Script

### Overview

The `main.py` script initializes the environment and agent, and runs training episodes to test the deployment strategies.

### Functions

- `calculate_state_size(state)`: Calculates the size of the flattened state.
- `test_initialization()`: Initializes the environment, creates the agent, and runs training episodes.

### Execution

The script runs for a specified number of episodes. In each episode, the agent interacts with the environment, choosing actions and learning from the rewards received. The total reward for each episode is printed to monitor the agent's performance.

## State Representation and Interaction

1. **State Representation**: 
   - Hosts and components are represented using dictionaries to store their respective CPU, RAM, and deployment status.
   - The `flatten_state` function converts this dictionary format into a flat array for neural network input.

2. **Agent-Environment Interaction**:
   - The agent observes the current state, selects an action (deploying a component to a host), and receives a reward.
   - The environment updates the state based on the agent's action, calculates rewards, and provides feedback to the agent.
   - The agent learns from this interaction to improve future decisions.

### Summary of Interactions

1. **Initialization**:
   - The environment is initialized with hosts and components.
   - Fixed components are deployed to their respective hosts.

2. **Action Selection**:
   - The agent selects a component and host pair for deployment.
   - The agent uses a neural network to predict the best action based on the current state.

3. **State Update and Reward Calculation**:
   - The environment updates the state based on the selected action.
   - The environment calculates rewards based on resource utilization, latency, and bandwidth constraints.

4. **Learning**:
   - The agent updates its neural network based on the rewards received, improving its future action selections.

## Deep Learning Model

### Overview

The deep learning model used in this project is a neural network implemented using TensorFlow and Keras. The model takes the flattened state as input and outputs probabilities for each possible action.

### Model Architecture

1. **Input Layer**: 
   - The input layer receives the flattened state representation.
   
2. **Hidden Layers**: 
   - Two hidden layers with 24 neurons each and ReLU activation functions.
   
3. **Output Layer**: 
   - The output layer has `action_size` neurons with softmax activation to output probabilities for each action.

### Input

The input to the model is a flattened state array that includes:

- Host resources (CPU, RAM) for each host.
- Component requirements (CPU, RAM) for each component.
- Deployment status of each component.

### Output

The output of the model is a probability distribution over all possible actions (component-host pairs).

### Loss Function

The loss function used is categorical cross-entropy, which is appropriate for multi-class classification problems. The agent uses this loss to update its neural network based on the rewards received from the environment.

### Interaction with the Reward System

The learning process in reinforcement learning involves updating the neural network to maximize the cumulative reward. Here's how the model interacts with the reward system:

1. **Action Selection**: 
   - The agent selects an action based on the probabilities output by the model.

2. **Reward Calculation**: 
   - The environment calculates the reward based on the selected action's impact on resource utilization, latency, and bandwidth constraints.

3. **Learning**:
   - The agent updates the model using the rewards received. The loss function (categorical cross-entropy) is minimized, guiding the neural network to predict actions that lead to higher rewards.
   - The rewards are used to adjust the probabilities output by the model, encouraging actions that result in higher rewards and discouraging actions that result in lower rewards.

