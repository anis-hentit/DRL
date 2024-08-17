## DRL - Deep Reinforcement Learning for Fog Computing

This project involves deploying multiple application components in a fog computing environment using a reinforcement learning (RL) agent enhanced with a Graph Neural Network (GNN). The RL agent learns optimal deployment strategies to maximize resource utilization and minimize latency, bandwidth, and energy consumption penalties. The project supports multiple applications and dynamic infrastructure generation, enhancing its versatility and scalability.

### File Structure

- `agent.py`: Contains the implementation of the RL agent using GNN, including the decision-making process, learning mechanism, and the dynamic masking technique for valid actions.
- `environment.py`: Defines the fog computing environment where the agent operates, including state initialization, reward calculation, and pathfinding algorithms.
- `main.py`: The main script to initialize the environment, train the agent, and evaluate the deployment strategies.
- `utils.py`: Utility functions for loading configuration files and supporting functions.
- `InfrastructureGenerator.py`: Script for generating scalable and dynamic infrastructure configurations.
- `benchmark.py`: Script to benchmark the execution time of different configurations on multiple topologies.

### Agent

#### Overview

The agent is implemented using TensorFlow and TensorFlow GNN. It employs a Graph Neural Network to predict the best actions based on the current state of the environment. The agent's decision-making is guided by a policy learned through interaction with the environment, using a Boltzmann exploration policy that gradually shifts from exploration to exploitation as training progresses.

#### Hyperparameters

| Hyperparameter   | Description                                               | Value  |
|------------------|-----------------------------------------------------------|--------|
| `learning_rate`  | Learning rate for the neural network optimizer            | 0.002  |
| `gamma`          | Discount factor for future rewards                        | 0.80   |
| `temperature`    | Initial temperature for the Boltzmann exploration policy  | 1.0    |
| `temperature_decay` | Decay rate for temperature after each episode           | 0.996  |
| `temperature_min` | Minimum value for temperature to ensure exploration       | 0.0001 |

#### Methods

- **`__init__(self, input_dim, hidden_dim, max_components, num_hosts, learning_rate=0.002)`**: Initializes the agent with the given parameters.
- **`choose_action(self, state, app_index, mode='train', num_hosts=None)`**: Chooses an action based on the current state using a Boltzmann exploration policy.
- **`store_experience(self, state, action, reward, next_state, done, app_index, explore)`**: Stores the experience in the replay buffer.
- **`learn(self, batch_size=64)`**: Updates the neural network based on the experiences.
- **`GNNAgent`**: Defines the Graph Neural Network architecture used by the agent.
- **`discount_rewards(self, rewards)`**: Computes discounted rewards to provide feedback on actions over time.
- **`sample_experiences(self, batch_size)`**: Samples experiences from both model and random replay buffers, adjusting the ratio based on the current temperature value.

#### Replay Buffer

The agent uses two types of replay buffers:

- **Model Replay Buffer**: Stores experiences where actions were chosen by the model.
- **Random Replay Buffer**: Stores experiences where actions were chosen randomly (exploration).

As training progresses, the sampling ratio adjusts to favor experiences from the model replay buffer more, while still maintaining a minimum level of exploration.

#### State Representation

The state is represented as a combination of host resources, multiple application requirements, deployment status, logical links, and infrastructure links:

- **Hosts**: Represented with their CPU and RAM resources.
- **Applications**: Each application consists of:
  - **Components**: Represented with their CPU, RAM requirements, deployment status, and application ID.
  - **components_start_idx**: Starting index of the components for the application.
  - **Links**: Logical links between components with bandwidth and latency requirements.
  - **Paths**: Paths initialized as `None`, will store the physical paths for each logical link.
  - **Fixed Positions**: Deployment zones (DZ) indicating fixed positions for components.

The state is converted into a graph data structure for input into the neural network using the `generate_graph_data` function.

### Environment

#### Overview

The `FogEnvironment` class represents the fog computing environment where components are deployed to hosts. It simulates the interaction between the agent and the environment, providing rewards based on the deployment's efficiency and the satisfaction of constraints like latency, bandwidth, and energy consumption.

#### Methods

- **`__init__(self, topology_file, num_applications=None, mode='train')`**: Initializes the environment with application and infrastructure configurations.
- **`define_action_space(self)`**: Defines the action space as a combination of components and hosts.
- **`define_observation_space(self)`**: Defines the observation space representing the state.
- **`initialize_state(self)`**: Initializes the environment state, deploying fixed components.
- **`step(self, action, app_index)`**: Executes an action and updates the state.
- **`deploy_component(self, app_state, component_id, host_id)`**: Deploys a component to a specified host.
- **`find_path(self, link_id, link, app_index)`**: Finds a path for a logical link between two deployed components.
- **`constrained_dijkstra(self, source, target, max_latency, min_bandwidth)`**: Finds a path with latency and bandwidth constraints using a constrained Dijkstra's algorithm.
- **`reduce_bandwidth(self, path, required_bandwidth)`**: Reduces the bandwidth on the path after deploying a component.
- **`calculate_energy_cost(self, host)`**: Calculates the energy cost based on the host's type and resources used.
- **`calculate_total_energy_cost(self)`**: Calculates the total energy cost for the entire deployment.
- **`calculate_reward(self, action, app_index)`**: Calculates the reward for the current state and action.
- **`reset(self)`**: Resets the environment to the initial state.
- **`check_all_deployed(self)`**: Checks if all components have been deployed.
- **`save_deployment_strategy(self, reward)`**: Saves the current deployment strategy to a text file.
- **`render(self, mode='console')`**: Renders the current state of the environment.

### Infrastructure Generator

#### Overview

The `InfrastructureGenerator.py` script dynamically generates scalable infrastructure configurations, including hosts and links. This allows the system to adapt to different sizes and densities of fog computing environments.

#### Methods

- **`generate_hosts(num_hosts)`**: Generates a list of hosts with specified resources.
- **`generate_links(num_hosts, density_factor)`**: Generates network topology with specified density.
- **`main(num_hosts, density_factor)`**: Main function to generate and save the infrastructure configuration.

### Benchmark

#### Overview

The `benchmark.py` script measures the execution time and performance of different configurations to evaluate the deployment strategies across multiple topologies. It runs the environment for a specified number of episodes and records various metrics.

#### Methods

- **`run_inference(agent, env, num_episodes, num_applications)`**: Runs inference and collects metrics for the given number of episodes and applications.
- **`main()`**: Main function to run the benchmark with different configurations and topologies.

### Main Script

#### Overview

The `main.py` script initializes the environment and agent, and runs training episodes to test and evaluate the deployment strategies. The agent interacts with the environment, making deployment decisions and learning from the outcomes to improve its policy over time.

#### Functions

- **`calculate_state_size(state)`**: Calculates the size of the flattened state.
- **`test_initialization()`**: Initializes the environment, creates the agent, and runs training episodes to evaluate the deployment strategies.

#### Execution

The script runs for a specified number of episodes (default is 7000). In each episode, the agent interacts with the environment, choosing actions and learning from the rewards received. The total reward for each episode is printed to monitor the agent's performance.

### Dependencies

- TensorFlow version: 2.12.0
- Keras version: 2.12.0
- TensorFlow GNN
- Gym
- NumPy
