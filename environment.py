import gym
import numpy as np
from gym import spaces
from utils import load_json_config
from agent import flatten_state, generate_graph_data, compute_action_mask
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
        self.app_configs, self.infra_config = load_json_config("DRL/data/Application.json", "DRL/data/scaled_infrastructure.json")
        # Initialize the environment state
        self.state = self.initialize_state()
        self.path_cache = {}  # Cache for storing computed paths
        self.num_logical_links = sum(len(app['links']) for app in self.state['applications'])
        self.max_components = max(len(app['components']) for app in self.state['applications'])
        self.num_hosts = len(self.state['hosts'])
        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()
        self.current_step = 0
        self.max_steps_per_episode = 100

    def define_action_space(self):
        """
        Define the action space as a tuple of (component, host).
        """
        return spaces.Tuple((spaces.Discrete(self.max_components), spaces.Discrete(self.num_hosts)))

    def define_observation_space(self):
        """
        Define the observation space as dictionaries for hosts and components.
        """
        num_components = sum(len(app['components']) for app in self.state['applications'])
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
            'infra_links': {i: dict(link) for i, link in enumerate(self.infra_config['network']['topology'])},
            'applications': []
        }

        global_comp_idx = 0
        for app_id, app_config in enumerate(self.app_configs['applications']):
            app_state = {
                'components': {i: {'CPU': comp['CPU'], 'RAM': comp['RAM'], 'deployed': False, 'app_id': app_id} for i, comp in enumerate(app_config['requirements'])},
                'components_start_idx': global_comp_idx,
                'fixed_positions': {comp['component_id']: comp['host_id'] for comp in app_config.get('DZ', [])},
                'links': {i: link for i, link in enumerate(app_config['links'])},
                'paths': {i: None for i in range(len(app_config['links']))}
            }

            global_comp_idx += len(app_state['components'])

            # Deploy components that have fixed positions specified
            for comp_id, host_id in app_state['fixed_positions'].items():
                if comp_id in app_state['components']:
                    comp = app_state['components'][comp_id]
                    host = state['hosts'][host_id]
                    if comp['CPU'] <= host['CPU'] and comp['RAM'] <= host['RAM']:
                        host['CPU'] -= comp['CPU']
                        host['RAM'] -= comp['RAM']
                        comp['deployed'] = True
                        comp['host'] = host_id
                        print(f"Initialized: Application {app_id}, Component {comp_id} deployed to Host {host_id}")
                    else:
                        raise ValueError(f"Host {host_id} does not have enough resources to deploy component {comp_id} of application {app_id} at initialization.")
            state['applications'].append(app_state)

        return state

    def step(self, action, app_index):
        """
        Execute the given action and update the environment's state.
        """
        self.current_step += 1
        component_id, host_id = action

        # Log the action attempt
        print(f"\nStep {self.current_step}: Trying to deploy component {component_id} to host {host_id}")

        app_state = self.state['applications'][app_index]
        num_components = len(app_state['components'])

        # Compute the action mask to check if the action is valid
        mask = compute_action_mask(self.state, app_index, self.max_components)
        action_index = component_id * self.num_hosts + host_id
        if component_id == -1 or host_id == -1 or mask[action_index] == 0:
            print(f"Invalid action received: component_id={component_id}, host_id={host_id}")
            return self.state, -50, self.current_step >= self.max_steps_per_episode, {}

        # Deploy the component if not already deployed
        if not app_state['components'][component_id]['deployed']:
            component = app_state['components'][component_id]
            host = self.state['hosts'][host_id]
            if component['CPU'] <= host['CPU'] and component['RAM'] <= host['RAM']:
                self.deploy_component(app_state, component_id, host_id)
                print(f"Deployed component {component_id} to host {host_id}")
            else:
                print(f"Failed to deploy component {component_id} to host {host_id} due to insufficient resources.")
                return self.state, -20, self.current_step >= self.max_steps_per_episode, {}

        # Find and validate paths for each logical link if both endpoints are deployed
        for link_id, link in app_state['links'].items():
            if app_state['components'][link['source']]['deployed'] and app_state['components'][link['destination']]['deployed']:
                if app_state['paths'][link_id] is None:
                    path = self.find_path(link_id, link, app_index)
                    if path is not None:
                        app_state['paths'][link_id] = path
                        if path:
                            self.reduce_bandwidth(path, link['bandwidth'])
                        print(f"Mapped link {link_id} to path {path}")
                        print(f"Path for link {link_id} between hosts {self.get_component_host(link['source'], app_index)} and {self.get_component_host(link['destination'], app_index)} saved as: {path}")
                    else:
                        print(f"Path for link {link_id} does not meet constraints. Skipping.")
                        app_state['paths'][link_id] = None

        done = self.check_all_deployed()  # Check if all components are deployed
        reward = self.calculate_reward(action)  # Calculate reward for the action
        done = done or self.current_step >= self.max_steps_per_episode  # Check if the episode should end
        next_state = self.state

        if done:
            self.save_deployment_strategy(reward)  # Save the deployment strategy at the end of the episode

        # Log final state for this step
        print("\n--- Final State After Step ---")
        self.print_state()

        return next_state, reward, done, {}

    def deploy_component(self, app_state, component_id, host_id):
        """
        Deploy a component to a specified host.
        """
        component = app_state['components'][component_id]
        host = self.state['hosts'][host_id]
        host['CPU'] -= component['CPU']
        host['RAM'] -= component['RAM']
        app_state['components'][component_id]['deployed'] = True
        app_state['components'][component_id]['host'] = host_id
        print(f"Component {component_id} deployed to Host {host_id}")

    def find_path(self, link_id, link, app_index):
        """
        Find a path for a logical link between two deployed components.
        """
        source_comp = link['source']
        destination_comp = link['destination']
        source_host = self.get_component_host(source_comp, app_index)
        destination_host = self.get_component_host(destination_comp, app_index)
        print(f"Finding path for link {link_id} from host {source_host} to host {destination_host}")
        if source_host == destination_host:
            print(f"Link {link_id} is intra-host communication. Returning empty path.")
            return []
        path = self.constrained_dijkstra(source_host, destination_host, link['latency'], link['bandwidth'])
        print(f"Path found for link {link_id}: {path}")
        return path

    def get_component_host(self, component_id, app_index):
        """
        Get the host to which a component is deployed.
        """
        app_state = self.state['applications'][app_index]
        if component_id in app_state['components']:
            component = app_state['components'][component_id]
            if 'host' in component:
                return component['host']
        return None

    def constrained_dijkstra(self, source, target, max_latency, min_bandwidth):
        """
        Implement a modified Dijkstra's algorithm to find the optimal path with latency and bandwidth constraints.
        """
        with open('pathfinding_log.txt', 'a') as f:
            f.write(f"Running constrained Dijkstra from {source} to {target} with max latency {max_latency} and min bandwidth {min_bandwidth}\n")
            
            graph = {i: [] for i in range(len(self.state['hosts']))}
            for link in self.state['infra_links'].values():
                if link['source'] == link['destination']:
                    continue
                if 'latency' not in link or 'bandwidth' not in link:
                    continue
                
                graph[link['source']].append((link['destination'], link['latency'], link['bandwidth']))
                graph[link['destination']].append((link['source'], link['latency'], link['bandwidth']))

            queue = [(0, source, [])]
            seen = set()
            all_valid_paths = []
            while queue:
                (cost, node, path) = heapq.heappop(queue)
                if node in seen:
                    continue
                new_path = path + [node]
                seen.add(node)
                if node == target:
                    if self.validate_path_with_constraints(new_path, max_latency, min_bandwidth):
                        valid_path = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
                        all_valid_paths.append((cost, valid_path))
                        f.write(f"Valid path found: {valid_path} with cost {cost}\n")
                for next_node, latency, bandwidth in graph.get(node, []):
                    if next_node not in seen and bandwidth >= min_bandwidth:
                        heapq.heappush(queue, (cost + latency, next_node, new_path))
                        f.write(f"Queueing path: {new_path} -> {next_node} with added cost {cost + latency}\n")

            if all_valid_paths:
                best_path = min(all_valid_paths, key=lambda x: x[0])[1]
                f.write(f"Best path found: {best_path}\n")
                return best_path

            f.write(f"No valid path found from {source} to {target} with the given constraints.\n")
            return None

    def validate_path_with_constraints(self, path, max_latency, min_bandwidth):
        """
        Validate a path to ensure it meets latency and bandwidth constraints.
        """
        if not path:
            # Allow empty path for intra-host communication
            return True
        total_latency = 0
        for i in range(len(path) - 1):
            link_data = next((item for item in self.state['infra_links'].values()
                              if item['source'] == path[i] and item['destination'] == path[i+1]), None)
            if not link_data:
                print(f"Missing link data for path segment {path[i]} to {path[i+1]}")
                return False
            total_latency += link_data['latency']
            if link_data['bandwidth'] < min_bandwidth:
                print(f"Path segment {path[i]} to {path[i+1]} does not meet bandwidth constraint")
                return False
        if total_latency > max_latency:
            print(f"Path exceeds latency constraint: total_latency={total_latency}, max_latency={max_latency}")
            return False
        return True

    def reduce_bandwidth(self, path, required_bandwidth):
        """
        Reduce the available bandwidth along the given path.
        """
        if path is None or not path:
            return
        for link in path:
            link_data = next((item for item in self.state['infra_links'].values()
                              if item['source'] == link[0] and item['destination'] == link[1]), None)
            if link_data and 'bandwidth' in link_data and link_data['bandwidth'] > 0:
                link_data['bandwidth'] -= required_bandwidth
                print(f"Reduced bandwidth for link ({link[0]}, {link[1]}) by {required_bandwidth}. New bandwidth: {link_data['bandwidth']}")

    def calculate_reward(self, action):
        """
        Calculate the reward for the current state and action.
        """
        energy = 0
        active_hosts = set(comp['host'] for app in self.state['applications'] for comp in app['components'].values() if comp['deployed'])
        for host_id in active_hosts:
            host = self.state['hosts'][host_id]
            energy += host['CPU'] * 0.02 + host['RAM'] * 0.01

        latency_penalty = self.calculate_latency_penalty()
        total_reward = -energy - latency_penalty

        total_reward += 5 * len(active_hosts)
        valid_paths = sum(1 for app in self.state['applications'] for path in app['paths'].values() if path is not None and (path != [] or not self.check_if_inter_host(path)))
        total_reward += 15 * valid_paths  
        failed_paths = sum(1 for app in self.state['applications'] for path in app['paths'].values() if path is None or (path == [] and not self.check_if_inter_host(path)))
        total_reward -= 15 * failed_paths

        if action == (-1, -1):
            print("BAD ACTION -50")
            total_reward -= 50

        print(f"Energy cost: {-energy}, Latency penalty: {-latency_penalty}, Failed paths: {failed_paths}")
        print(f"Valid paths: {valid_paths}, Components deployed: {len(active_hosts)}")
        print(f"Total calculated reward: {total_reward}")
        return total_reward

    def calculate_latency_penalty(self):
        """
        Calculate penalties for paths exceeding the latency requirements.
        """
        penalty = 0
        for app in self.state['applications']:
            for link_id, path in app['paths'].items():
                if path is None or not path:
                    continue
                total_latency = sum(next(link_data['latency'] for link_data in self.state['infra_links'].values()
                                         if link_data['source'] == link[0] and link_data['destination'] == link[1])
                                    for link in path)
                latency_req = app['links'][link_id]['latency'] if link_id in app['links'] else 0
                if total_latency > latency_req:
                    penalty += (total_latency - latency_req)
        return penalty

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.state = self.initialize_state()
        self.current_step = 0
        self.path_cache.clear()  # Clear the path cache on reset
        flattened_state = flatten_state(self.state)
        assert flattened_state.shape[0] == calculate_state_size(self.state), "State size mismatch on reset!"
        return self.state

    def check_all_deployed(self):
        """
        Check if all components have been deployed.
        """
        return all(comp['deployed'] for app in self.state['applications'] for comp in app['components'].values())

    def check_if_inter_host(self, path):
        """
        Check if an empty path corresponds to inter-host communication.
        """
        if not path:
            return False
        source_host = path[0][0] if path else None
        dest_host = path[-1][1] if path else None
        return source_host is not None and dest_host is not None and source_host != dest_host

    def log_path(self, link_id, path):
        """
        Log the path for a link to a text file.
        """
        with open('deployment_strategy.txt', 'a') as f:
            f.write(f"Link {link_id}: Path={path}\n")

    def save_deployment_strategy(self, reward):
        """
        Save the current deployment strategy to a text file.
        """
        with open('deployment_strategy.txt', 'a') as f:
            f.write("\n--- Deployment Strategy ---\n")
            f.write(f"Total Steps: {self.current_step}\n")
            for app_id, app in enumerate(self.state['applications']):
                f.write(f"Application {app_id}:\n")
                f.write("Components:\n")
                for comp_id, comp in app['components'].items():
                    host_id = comp.get('host', None)
                    status = "Deployed" if comp['deployed'] else "Not Deployed"
                    f.write(f"  Component {comp_id}: CPU={comp['CPU']}, RAM={comp['RAM']}, Status={status}, Host={host_id}\n")

                f.write("Paths:\n")
                for link_id, path in app['paths'].items():
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
        for app in self.state['applications']:
            for comp_id, comp in app['components'].items():
                host_id = comp.get('host', None)
                status = "Deployed" if comp['deployed'] else "Not Deployed"
                print(f"  Component {comp_id}: CPU={comp['CPU']}, RAM={comp['RAM']}, Status={status}, Host={host_id}")

        print("Hosts:")
        for host_id, host in self.state['hosts'].items():
            print(f"  Host {host_id}: CPU={host['CPU']}, RAM={host['RAM']}")

        print("Paths:")
        for app in self.state['applications']:
            for link_id, path in app['paths'].items():
                print(f"  Link {link_id}: Path={path}")

        print("----------------------")

    def close(self):
        """
        Close the environment.
        """
        pass
