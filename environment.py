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
                    else:
                        app_state['paths'][link_id] = None

        done = self.check_all_deployed()  # Check if all components are deployed
        reward = self.calculate_reward(action, app_index)  # Calculate reward for the action
        done = done or self.current_step >= self.max_steps_per_episode  # Check if the episode should end
        next_state = self.state

        if done:
            final_reward = self.calculate_final_reward()
            self.save_deployment_strategy(final_reward)  # Save the deployment strategy at the end of the episode

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

    def find_path(self, link_id, link, app_index):
        """
        Find a path for a logical link between two deployed components.
        """
        source_comp = link['source']
        destination_comp = link['destination']
        source_host = self.get_component_host(source_comp, app_index)
        destination_host = self.get_component_host(destination_comp, app_index)
        if source_host == destination_host:
            return []
        path = self.constrained_dijkstra(source_host, destination_host, link['latency'], link['bandwidth'])
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
                    
            for next_node, latency, bandwidth in graph.get(node, []):
                if next_node not in seen and bandwidth >= min_bandwidth:
                    heapq.heappush(queue, (cost + latency, next_node, new_path))
                    

        if all_valid_paths:
            best_path = min(all_valid_paths, key=lambda x: x[0])[1]
            
            return best_path

           
            return None

    def validate_path_with_constraints(self, path, max_latency, min_bandwidth):
        """
        Validate a path to ensure it meets latency and bandwidth constraints.
        """
        if not path:
            return True
        total_latency = 0
        for i in range(len(path) - 1):
            link_data = next((item for item in self.state['infra_links'].values()
                              if item['source'] == path[i] and item['destination'] == path[i+1]), None)
            if not link_data:
                return False
            total_latency += link_data['latency']
            if link_data['bandwidth'] < min_bandwidth:
                return False
        if total_latency > max_latency:
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

    def calculate_reward(self, action, app_index):
        """
        Calculate the reward for the current state and action.
        """
        app_state = self.state['applications'][app_index]

        energy = 0
        active_hosts = set(comp['host'] for comp in app_state['components'].values() if comp['deployed'])
        for host_id in active_hosts:
            host = self.state['hosts'][host_id]
            energy += host['CPU'] * 0.02 + host['RAM'] * 0.01

        latency_penalty = self.calculate_latency_penalty(app_index)
        total_reward = -energy - latency_penalty
    
        total_reward += 2 * len(active_hosts)
        num_links = len(app_state['links'])
        if num_links > 0:
            valid_paths = sum(1 for path in app_state['paths'].values() if path is not None and (path != [] or not self.check_if_inter_host(path)))
            failed_paths = sum(1 for path in app_state['paths'].values() if path is None or (path == [] and not self.check_if_inter_host(path)))
    
            valid_paths_percentage = valid_paths / num_links
            failed_paths_percentage = failed_paths / num_links

            total_reward += 10 * valid_paths_percentage
            total_reward -= 10 * failed_paths_percentage

        # Additional reward for deploying components even if not all are deployed yet
        if not all(comp['deployed'] for comp in app_state['components'].values()):
            total_reward += 5  # Positive reward for each component deployment action

        if action == (-1, -1):
            total_reward -= 50

        # Detailed debug statements for reward components
        print(f"App {app_index} Reward Calculation Debug:")
        print(f"  Energy cost: {-energy}")
        print(f"  Latency penalty: {-latency_penalty}")
        print(f"  Active hosts bonus: {1 * len(active_hosts)}")
        print(f"  Valid paths bonus: {10 * valid_paths_percentage }")
        print(f"  Failed paths penalty: {-15 * failed_paths_percentage }")
        print(f"  Total calculated reward: {total_reward}")

        return total_reward


    def calculate_latency_penalty(self, app_index):
        """
        Calculate penalties for paths exceeding the latency requirements.
        """
        penalty = 0
        app = self.state['applications'][app_index]
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

    def calculate_final_reward(self):
        """
        Calculate the final reward for the overall state of deployment at the end of an episode.
        """
        final_reward = 0
        for app_index in range(len(self.state['applications'])):
            app_reward = self.calculate_reward(None, app_index)
            final_reward += app_reward
        return final_reward

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
        for app in self.state['applications']:
            for comp_id, comp in app['components'].items():
                host_id = comp.get('host', None)
                status = "Deployed" if comp['deployed'] else "Not Deployed"
                print(f"  Component {comp_id}: CPU={comp['CPU']}, RAM={comp['RAM']}, Status={status}, Host={host_id}")

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
