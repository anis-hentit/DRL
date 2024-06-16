## DRL - Deep Reinforcement Learning for Fog Computing

This project involves deploying application components in a fog computing environment using a reinforcement learning (RL) agent enhanced with a Graph Neural Network (GNN). The RL agent learns optimal deployment strategies to maximize resource utilization and minimize latency, bandwidth, and energy consumption penalties. The project consists of three main modules: the agent, the environment, and the main execution script.

### File Structure

- `agent.py`: Contains the implementation of the RL agent using GNN.
- `environment.py`: Defines the fog computing environment where the agent operates.
- `main.py`: The main script to initialize the environment, train the agent, and test the deployment strategies.

### Agent

#### Overview

The agent is implemented using TensorFlow and TensorFlow GNN. It employs a Graph Neural Network to predict the best actions based on the current state of the environment.

#### Hyperparameters

| Hyperparameter   | Description                                               | Value  |
|------------------|-----------------------------------------------------------|--------|
| `learning_rate`  | Learning rate for the neural network optimizer            | 0.001  |
| `gamma`          | Discount factor for future rewards                        | 0.99   |
| `epsilon`        | Initial exploration rate for the epsilon-greedy policy    | 1.0    |
| `epsilon_decay`  | Decay rate for epsilon after each episode                 | 0.995  |
| `epsilon_min`    | Minimum value for epsilon to ensure exploration           | 0.01   |

#### Methods

- **`__init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001)`**: Initializes the agent with the given parameters.
- **`choose_action(self, state)`**: Chooses an action based on the current state using an epsilon-greedy policy.
- **`learn(self, states, actions, rewards)`**: Updates the neural network based on the experiences.
- **`GNNAgent`**: Defines the Graph Neural Network architecture used by the agent.

#### State Representation

The state is represented as a combination of host resources, component requirements, deployment status, logical links, and infrastructure links:

- **Hosts**: Represented with their CPU and RAM resources.
- **Components**: Represented with their CPU, RAM requirements, and deployment status.
- **Links**: Logical links between components with bandwidth and latency requirements.
- **Infrastructure Links**: Physical links between hosts in the fog network.
- **Paths**: Physical paths between hosts for each logical link.

The state is converted into a graph data structure for input into the neural network using the `generate_graph_data` function.

### Environment

#### Overview

The `FogEnvironment` class represents the fog computing environment where components are deployed to hosts. It simulates the interaction between the agent and the environment, providing rewards based on the deployment's efficiency.

#### Methods

- **`__init__(self)`**: Initializes the environment with application and infrastructure configurations.
- **`define_action_space(self)`**: Defines the action space as a combination of components and hosts.
- **`define_observation_space(self)`**: Defines the observation space representing the state.
- **`initialize_state(self)`**: Initializes the environment state, deploying fixed components.
- **`step(self, action)`**: Executes an action and updates the state.
- **`deploy_component(self, component_id, host_id)`**: Deploys a component to a specified host.
- **`find_path(self, link_id)`**: Finds a path for a logical link between two deployed components.
- **`constrained_dijkstra(self, source, target, max_latency, min_bandwidth)`**: Finds a path with latency and bandwidth constraints using Dijkstra's algorithm.
- **`validate_path(self, path, latency_req, bandwidth_req)`**: Validates a path based on latency and bandwidth requirements.
- **`calculate_reward(self, action)`**: Calculates the reward for the current state and action.
- **`reset(self)`**: Resets the environment to the initial state.
- **`render(self, mode='console')`**: Renders the current state of the environment.

### Main Script

#### Overview

The `main.py` script initializes the environment and agent, and runs training episodes to test the deployment strategies.

#### Functions

- **`calculate_state_size(state)`**: Calculates the size of the flattened state.
- **`test_initialization()`**: Initializes the environment, creates the agent, and runs training episodes.

#### Execution

The script runs for a specified number of episodes. In each episode, the agent interacts with the environment, choosing actions and learning from the rewards received. The total reward for each episode is printed to monitor the agent's performance.

### Future Directions

1. **Multi-Application Handling**: Extend the environment and agent to handle the deployment of multiple applications in parallel.
2. **Improved Reward Mechanism**: Refine the reward calculation to better balance the trade-offs between resource utilization, latency, bandwidth, and energy consumption.
3. **Advanced GNN Models**: Experiment with different GNN architectures and hyperparameters to enhance the agent's performance.
4. **Real-Time Deployment**: Explore the integration of the trained agent in real-time fog computing scenarios.


