import gym
import numpy as np
from gym import spaces
from utils import load_json_config
from agent import flatten_state, generate_graph_data
import heapq

def calculate_state_size(state):
    """
    Calculate the size of the flattened state.
    """
    flattened_state = flatten_state(state)
    return len(flattened_state)

class FogEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        """
        Initialize the FogEnvironment with application and infrastructure configurations.
        """
        super(FogEnvironment, self).__init__()
        # Load application and infrastructure configurations from JSON files
        self.app_config, self.infra_config = load_json_config("DRL/data/Application.json", "DRL/data/Graph_Infra.json")
        # Initialize the environment state
        self.state = self.initialize_state()
        self.num_logical_links = len(self.state['links'])
        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()
        self.current_step = 0
        self.max_steps_per_episode = 100

    def define_action_space(self):
        """
        Define the action space as a tuple of (component, host).
        """
        num_components = len(self.state['components'])
        num_hosts = len(self.state['hosts'])
        return spaces.Tuple((spaces.Discrete(num_components), spaces.Discrete(num_hosts)))

    def define_observation_space(self):
        """
        Define the observation space as dictionaries for hosts and components.
        """
        num_components = len(self.state['components'])
        num_hosts = len(self.state['hosts'])
        return spaces.Dict({
            'hosts': spaces.Box(low=0, high=100, shape=(num_hosts, 2), dtype=np.float32),
            'components': spaces.Box(low=0, high=100, shape=(num_components, 3), dtype=np.float32)
        })

    def initialize_state(self):
        """
        Initialize the state of the environment.
        Deploy components with fixed positions and setup the initial state.
        """
        state = {
            'hosts': {i: {'CPU': host['CPU'], 'RAM': host['BW']} for i, host in enumerate(self.infra_config['hosts']['configuration'])},
            'components': {i: {'CPU': comp['CPU'], 'RAM': comp['RAM'], 'deployed': False} for i, comp in enumerate(self.app_config['application']['requirements'])},
            'fixed_positions': {comp['component_id']: comp['host_id'] for comp in self.app_config['application'].get('DZ', [])},
            'links': {i: link for i, link in enumerate(self.app_config['application']['links'])},
            'infra_links': {i: dict(link) for i, link in enumerate(self.infra_config['network']['topology'])},
            'paths': {i: None for i in range(len(self.app_config['application']['links']))}  # Initialize paths as None
        }
        
        # Deploy components that have fixed positions specified
        for comp_id, host_id in state['fixed_positions'].items():
            if comp_id in state['components']:
                comp = state['components'][comp_id]
                host = state['hosts'][host_id]
                if comp['CPU'] <= host['CPU'] and comp['RAM'] <= host['RAM']:
                    host['CPU'] -= comp['CPU']
                    host['RAM'] -= comp['RAM']
                    comp['deployed'] = True
                    comp['host'] = host_id
                    print(f"Initialized: Component {comp_id} deployed to Host {host_id}")
                else:
                    raise ValueError(f"Host {host_id} does not have enough resources to deploy component {comp_id} at initialization.")
        return state

    def step(self, action):
        """
        Execute the given action and update the environment's state.
        """
        self.current_step += 1
        component_id, host_id = action

        # Log the action attempt
        print(f"\nStep {self.current_step}: Trying to deploy component {component_id} to host {host_id}")

        # Log the bandwidths before taking the action
        print("\nBandwidths before action:")
        for link_id, link in self.state['infra_links'].items():
            print(f"  Link {link_id}: Source={link['source']}, Destination={link['destination']}, Bandwidth={link['bandwidth']}")

        # Check for invalid actions
        if component_id == -1 or host_id == -1:
            print(f"Invalid action received: component_id={component_id}, host_id={host_id}")
            return self.state, -50, self.current_step >= self.max_steps_per_episode, {}

        # Deploy the component if not already deployed
        if not self.state['components'][component_id]['deployed']:
            component = self.state['components'][component_id]
            host = self.state['hosts'][host_id]
            if component['CPU'] <= host['CPU'] and component['RAM'] <= host['RAM']:
                self.deploy_component(component_id, host_id)
                print(f"Deployed component {component_id} to host {host_id}")
            else:
                print(f"Failed to deploy component {component_id} to host {host_id} due to insufficient resources.")
                return self.state, -20, self.current_step >= self.max_steps_per_episode, {}

        # Find and validate paths for each logical link if both endpoints are deployed
        for link_id in self.state['links']:
            if self.state['components'][self.state['links'][link_id]['source']]['deployed'] and \
               self.state['components'][self.state['links'][link_id]['destination']]['deployed']:
                path = self.find_path(link_id)
                if self.validate_path(path, self.state['links'][link_id]['latency'], self.state['links'][link_id]['bandwidth']):
                    self.state['paths'][link_id] = path
                    if path:  # Only reduce bandwidth if there is a valid non-empty path
                        self.reduce_bandwidth(path, self.state['links'][link_id]['bandwidth'])
                    print(f"Mapped link {link_id} to path {path}")
                else:
                    print(f"Path for link {link_id} does not meet constraints. Skipping.")
                    self.state['paths'][link_id] = None  # Set to None if no valid path is found

        done = self.check_all_deployed()  # Check if all components are deployed
        reward = self.calculate_reward(action)  # Calculate reward for the action
        if done:
            self.save_deployment_strategy(reward)  # Save deployment strategy to file
        done = done or self.current_step >= self.max_steps_per_episode  # Check if the episode should end
        next_state = self.state

        # Log the bandwidths after taking the action
        print("\nBandwidths after action:")
        for link_id, link in self.state['infra_links'].items():
            print(f"  Link {link_id}: Source={link['source']}, Destination={link['destination']}, Bandwidth={link['bandwidth']}")

        # Log final state for this step
        print("\n--- Final State After Step ---")
        self.print_state()

        return next_state, reward, done, {}

    def deploy_component(self, component_id, host_id):
        """
        Deploy a component to a specified host.
        """
        component = self.state['components'][component_id]
        host = self.state['hosts'][host_id]
        host['CPU'] -= component['CPU']
        host['RAM'] -= component['RAM']
        self.state['components'][component_id]['deployed'] = True
        self.state['components'][component_id]['host'] = host_id
        print(f"Component {component_id} deployed to Host {host_id}")

    def find_path(self, link_id):
        """
        Find a path for a logical link between two deployed components.
        """
        source_comp = self.state['links'][link_id]['source']
        destination_comp = self.state['links'][link_id]['destination']
        source_host = self.state['components'][source_comp]['host']
        destination_host = self.state['components'][destination_comp]['host']
        print(f"Finding path for link {link_id} from host {source_host} to host {destination_host}")
        if source_host == destination_host:
            print(f"Link {link_id} is intra-host communication. Returning empty path.")
            return []
        path = self.constrained_dijkstra(source_host, destination_host, 
                                         self.state['links'][link_id]['latency'], 
                                         self.state['links'][link_id]['bandwidth'])
        print(f"Path found for link {link_id}: {path}")
        return path

    def constrained_dijkstra(self, source, target, max_latency, min_bandwidth):
        """
        Implement a constrained Dijkstra's algorithm to find a path with latency and bandwidth constraints.
        """
        print(f"Running constrained Dijkstra from {source} to {target} with max latency {max_latency} and min bandwidth {min_bandwidth}")
        
        # Construct the graph from the infrastructure links
        graph = {i: [] for i in range(len(self.state['hosts']))}
        for link in self.infra_config['network']['topology']:
            if link['source'] == link['destination']:
                continue  # Ignore self-links
            if 'latency' not in link or 'bandwidth' not in link:
                print(f"Link missing required keys: {link}")
                continue
            graph[link['source']].append((link['destination'], link['latency'], link['bandwidth']))
        print(f"Constructed graph: {graph}")

        # Priority queue for Dijkstra's algorithm
        queue = [(0, source, [])]
        seen = set()
        while queue:
            (cost, node, path) = heapq.heappop(queue)
            if node in seen:
                continue
            new_path = path + [node]
            seen.add(node)
            if node == target:
                # Ensure the path meets the overall constraints
                if self.validate_path_with_constraints(new_path, max_latency, min_bandwidth):
                    valid_path = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
                    print(f"Valid path found: {valid_path}")
                    return valid_path
            for next_node, latency, bandwidth in graph.get(node, []):
                if next_node not in seen and bandwidth >= min_bandwidth:
                    print(f"Evaluating path to {next_node}: current latency {cost + latency}, bandwidth {bandwidth}")
                    heapq.heappush(queue, (cost + latency, next_node, new_path))

        print(f"No valid path found from {source} to {target} meeting constraints.")
        return None  # Return None to indicate no valid path was found

    def validate_path_with_constraints(self, path, max_latency, min_bandwidth):
        """
        Validate a path to ensure it meets latency and bandwidth constraints.
        """
        total_latency = 0
        for i in range(len(path) - 1):
            link_data = next((item for item in self.infra_config['network']['topology']
                              if item['source'] == path[i] and item['destination'] == path[i+1]), None)
            if not link_data:
                print(f"No link data found for segment from {path[i]} to {path[i+1]}")
                return False
            total_latency += link_data['latency']
            print(f"Checking link from {path[i]} to {path[i+1]}: latency={link_data['latency']}, total latency so far={total_latency}, bandwidth={link_data['bandwidth']}")
            if link_data['bandwidth'] < min_bandwidth:
                print(f"Link from {path[i]} to {path[i+1]} fails bandwidth requirement: {link_data['bandwidth']} < {min_bandwidth}")
                return False
        print(f"Total latency for path {path}: {total_latency} (max allowed: {max_latency})")
        return total_latency <= max_latency

    def reduce_bandwidth(self, path, required_bandwidth):
        """
        Reduce the available bandwidth along the given path.
        """
        if path is None or not path:  # Skip if path is None or empty
            return
        for link in path:
            link_data = next((item for item in self.state['infra_links'].values()
                              if item['source'] == link[0] and item['destination'] == link[1]), None)
            if link_data and 'bandwidth' in link_data and link_data['bandwidth'] > 0:  # Ensure bandwidth is positive
                print(f"Before reducing: Link {link} has bandwidth {link_data['bandwidth']}")
                link_data['bandwidth'] -= required_bandwidth
                print(f"After reducing: Link {link} now has bandwidth {link_data['bandwidth']}")

    def validate_path(self, path, latency_req, bandwidth_req):
        """
        Validate if a given path meets the specified latency and bandwidth requirements.
        """
        if path is None:
            print(f"Path is None. Invalid path.")
            return False
        if not path:  # Check if the path is an empty list, meaning intra-host communication
            print(f"Path is an empty list, meaning intra-host communication. Valid path.")
            return True

        total_latency = 0
        min_bandwidth = float('inf')
        for link in path:
            link_data = next((item for item in self.state['infra_links'].values()
                              if item['source'] == link[0] and item['destination'] == link[1]), None)
            if not link_data or link_data['source'] == link_data['destination']:
                continue
            link_bandwidth = link_data['bandwidth']
            link_latency = link_data['latency']
            total_latency += link_latency
            min_bandwidth = min(min_bandwidth, link_bandwidth)
            print(f"Link from {link[0]} to {link[1]}: latency={link_latency}, bandwidth={link_bandwidth}")
        print(f"Total latency: {total_latency}, Min bandwidth: {min_bandwidth}")

        return total_latency <= latency_req and min_bandwidth >= bandwidth_req

    def calculate_reward(self, action):
        """
        Calculate the reward for the current state and action.
        """
        energy = 0
        active_hosts = set(comp['host'] for comp in self.state['components'].values() if comp['deployed'])
        for host_id in active_hosts:
            host = self.state['hosts'][host_id]
            energy += host['CPU'] * 0.02 + host['RAM'] * 0.01

        latency_penalty = self.calculate_latency_penalty()
        bandwidth_penalty = self.calculate_bandwidth_penalty()

        # Base reward/penalty for maintaining energy efficiency and respecting constraints
        total_reward = -energy - latency_penalty - bandwidth_penalty

        # Additional reward for successful deployments
        total_reward += 15 * len(active_hosts)

        # Count the number of valid paths
        valid_paths = sum(1 for path in self.state['paths'].values() if path is not None and (path != [] or not self.check_if_inter_host(path)))

        # Reward for each valid link mapping (successful path mapping)
        total_reward += 10 * valid_paths  

        # Additional penalty for each failed path mapping (paths set to None or empty list not for intra-host communication)
        failed_paths = sum(1 for path in self.state['paths'].values() if path is None or (path == [] and self.check_if_inter_host(path)))
        total_reward -= 20 * failed_paths

        # Additional penalty for invalid actions
        if action == (-1, -1):
            total_reward -= 50

        print(f"Energy cost: {-energy}, Latency penalty: {-latency_penalty}, Bandwidth penalty: {-bandwidth_penalty}, Failed paths: {failed_paths}")
        print(f"Valid paths: {valid_paths}, Components deployed: {len(active_hosts)}")
        print(f"Total calculated reward: {total_reward}")
        return total_reward

    def calculate_latency_penalty(self):
        """
        Calculate penalties for paths exceeding the latency requirements.
        """
        penalty = 0
        for link_id, path in self.state['paths'].items():
            if path is None or not path:  # Skip if path is None or empty
                continue
            total_latency = sum(next(link_data['latency'] for link_data in self.infra_config['network']['topology']
                                     if link_data['source'] == link[0] and link_data['destination'] == link[1])
                                for link in path)
            latency_req = self.state['links'][link_id]['latency'] if link_id in self.state['links'] else 0
            if total_latency > latency_req:
                penalty += (total_latency - latency_req)
        return penalty

    def calculate_bandwidth_penalty(self):
        """
        Calculate penalties for paths not meeting the bandwidth requirements.
        """
        penalty = 0
        for link_id, path in self.state['paths'].items():
            if path is None or not path:  # Skip if path is None or empty
                continue
            min_bandwidth = min(next(link_data['bandwidth'] for link_data in self.infra_config['network']['topology']
                                    if link_data['source'] == link[0] and link_data['destination'] == link[1])
                               for link in path)
            bandwidth_req = self.state['links'][link_id]['bandwidth'] if link_id in self.state['links'] else 0
            if min_bandwidth < bandwidth_req:
                penalty += (bandwidth_req - min_bandwidth)
        return penalty

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.state = self.initialize_state()
        self.current_step = 0
        flattened_state = flatten_state(self.state)
        print(f"Reset state: {self.state}")
        print(f"Reset state flattened size: {flattened_state.shape}")
        assert flattened_state.shape[0] == calculate_state_size(self.state), "State size mismatch on reset!"

        # Log initial bandwidths for verification
        print("\nInitial bandwidths after reset:")
        for link_id, link in self.state['infra_links'].items():
            print(f"  Link {link_id}: Source={link['source']}, Destination={link['destination']}, Bandwidth={link['bandwidth']}")

        return self.state

    def check_all_deployed(self):
        """
        Check if all components have been deployed.
        """
        return all(comp['deployed'] for comp in self.state['components'].values())

    def check_if_inter_host(self, path):
        """
        Check if an empty path corresponds to inter-host communication.
        """
        if not path:
            return False  # empty path implies intra-host communication
        source_host = path[0][0] if path else None
        dest_host = path[-1][1] if path else None
        return source_host is not None and dest_host is not None and source_host != dest_host

    def save_deployment_strategy(self, reward):
        """
        Save the current deployment strategy to a text file.
        """
        with open('deployment_strategy.txt', 'a') as f:  # Append to the file instead of overwriting
            f.write("\n--- Deployment Strategy ---\n")
            f.write(f"Step {self.current_step}:\n")
            f.write("Components:\n")
            for comp_id, comp in self.state['components'].items():
                host_id = comp.get('host', None)
                status = "Deployed" if comp['deployed'] else "Not Deployed"
                f.write(f"  Component {comp_id}: CPU={comp['CPU']}, RAM={comp['RAM']}, Status={status}, Host={host_id}\n")

            f.write("Paths:\n")
            for link_id, path in self.state['paths'].items():
                f.write(f"  Link {link_id}: Path={path}\n")

            f.write(f"Total reward: {reward}\n")

    def render(self, mode='console'):
        """
        Render the current state of the environment.
        """
        if mode == 'console':
            self.print_state()

    def print_state(self):
        """
        Print the current state of components, hosts, and paths.
        """
        print("\n--- Current State ---")
        print("Components:")
        for comp_id, comp in self.state['components'].items():
            host_id = comp.get('host', None)
            status = "Deployed" if comp['deployed'] else "Not Deployed"
            print(f"  Component {comp_id}: CPU={comp['CPU']}, RAM={comp['RAM']}, Status={status}, Host={host_id}")
        
        print("Hosts:")
        for host_id, host in self.state['hosts'].items():
            print(f"  Host {host_id}: CPU={host['CPU']}, RAM={host['RAM']}")
        
        print("Paths:")
        for link_id, path in self.state['paths'].items():
            print(f"  Link {link_id}: Path={path}")
        
        print("Infra Links:")
        for link_id, link in self.state['infra_links'].items():
            latency = link.get('latency', 'N/A')  # Safely get latency, defaulting to 'N/A' if not present
            bandwidth = link['bandwidth']
            print(f"  Link {link_id}: Source={link['source']}, Destination={link['destination']}, Bandwidth={bandwidth}, Latency={latency}")
        
        print("----------------------")

    def close(self):
        """
        Close the environment.
        """
        pass
