import gym
import numpy as np
from gym import spaces
from utils import load_json_config
from agent import flatten_state, generate_graph_data, compute_action_mask
import heapq

def calculate_state_size(state):
    flattened_state = flatten_state(state)
    return len(flattened_state)

class FogEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, topology_file, num_applications=None, mode='train'):
        super(FogEnvironment, self).__init__()
        self.topology_file = topology_file
        self.app_configs, self.infra_config = load_json_config("DRL/data/Application.json", topology_file)
        
        if num_applications is not None:
            self.app_configs['applications'] = self.app_configs['applications'][:num_applications]

        self.mode = mode
        self.log_file = 'deployment_strategy.txt' if self.mode == 'train' else 'inference_strategy.txt'
        
        self.state = self.initialize_state()
        self.path_cache = {}
        self.num_logical_links = sum(len(app['links']) for app in self.state['applications'])
        self.max_components = max(len(app['components']) for app in self.state['applications'])
        self.num_hosts = 100  # Set to the maximum number of hosts across all topologies
        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()
        self.current_step = 0
        self.max_steps_per_episode = 100

    def define_action_space(self):
        return spaces.Tuple((spaces.Discrete(self.max_components), spaces.Discrete(self.num_hosts)))

    def define_observation_space(self):
        num_components = sum(len(app['components']) for app in self.state['applications'])
        return spaces.Dict({
            'hosts': spaces.Box(low=0, high=100, shape=(self.num_hosts, 2), dtype=np.float32),
            'components': spaces.Box(low=0, high=100, shape=(num_components, 3), dtype=np.float32)
        })

    def initialize_state(self):
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
                'paths': {i: None for i in range(len(app_config['links']))},
                'latency_penalties': {i: 0 for i in range(len(app_config['links']))}
            }

            global_comp_idx += len(app_state['components'])

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
        self.current_step += 1
        component_id, host_id = action
        print(f"\nStep {self.current_step}: Trying to deploy component {component_id} to host {host_id}")

        app_state = self.state['applications'][app_index]
        num_components = len(app_state['components'])

        mask = compute_action_mask(self.state, app_index, self.max_components, max_hosts=self.num_hosts)
        action_index = component_id * self.num_hosts + host_id
        if component_id == -1 or host_id == -1 or mask[action_index] == 0:
            print(f"Invalid action received: component_id={component_id}, host_id={host_id}")
            return self.state, -50, self.current_step >= self.max_steps_per_episode, {}

        if not app_state['components'][component_id]['deployed']:
            component = app_state['components'][component_id]
            host = self.state['hosts'][host_id]
            if component['CPU'] <= host['CPU'] and component['RAM'] <= host['RAM']:
                self.deploy_component(app_state, component_id, host_id)
                print(f"Deployed component {component_id} to host {host_id}")
            else:
                print(f"Failed to deploy component {component_id} to host {host_id} due to insufficient resources.")
                return self.state, -20, self.current_step >= self.max_steps_per_episode, {}

        for link_id, link in app_state['links'].items():
            if app_state['components'][link['source']]['deployed'] and app_state['components'][link['destination']]['deployed']:
                if app_state['paths'][link_id] is None:
                    path = self.find_path(link_id, link, app_index)
                    app_state['paths'][link_id] = path
                    if path:
                        self.reduce_bandwidth(path, link['bandwidth'])
                        self.calculate_latency_penalty(app_index, link_id, path)

        done = self.check_all_deployed()
        reward = self.calculate_reward(action, app_index)
        done = done or self.current_step >= self.max_steps_per_episode
        next_state = self.state

        if done:
            final_reward = self.calculate_final_reward()
            self.save_deployment_strategy(final_reward)

        print("\n--- Final State After Step ---")
        self.print_state()
        if done:
            print(f"Final episode reward: {final_reward}")

        return next_state, reward, done, {}

    def deploy_component(self, app_state, component_id, host_id):
        component = app_state['components'][component_id]
        host = self.state['hosts'][host_id]
        host['CPU'] -= component['CPU']
        host['RAM'] -= component['RAM']
        app_state['components'][component_id]['deployed'] = True
        app_state['components'][component_id]['host'] = host_id

    def find_path(self, link_id, link, app_index):
        source_comp = link['source']
        destination_comp = link['destination']
        source_host = self.get_component_host(source_comp, app_index)
        destination_host = self.get_component_host(destination_comp, app_index)
        if source_host == destination_host:
            return []
        path = self.constrained_dijkstra(source_host, destination_host, link['latency'], link['bandwidth'])
        return path

    def get_component_host(self, component_id, app_index):
        app_state = self.state['applications'][app_index]
        if component_id in app_state['components']:
            component = app_state['components'][component_id]
            if 'host' in component:
                return component['host']
        return None

    def constrained_dijkstra(self, source, target, max_latency, min_bandwidth):
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
        if path is None or not path:
            return
        for link in path:
            link_data = next((item for item in self.state['infra_links'].values()
                              if item['source'] == link[0] and item['destination'] == link[1]), None)
            if link_data and 'bandwidth' in link_data and link_data['bandwidth'] > 0:
                link_data['bandwidth'] -= required_bandwidth

    def calculate_reward(self, action, app_index):
        app_state = self.state['applications'][app_index]
        num_components = len(app_state['components'])

        if action is not None:
            component_id, host_id = action
            component = app_state['components'][component_id]
            host = self.state['hosts'][host_id]
            energy_cost = host['CPU'] * 0.04 + host['RAM'] * 0.02
        else:
            energy_cost = 0

        latency_penalty = sum(app_state['latency_penalties'].values())
        total_reward = -energy_cost - latency_penalty

        if all(comp['deployed'] for comp in app_state['components'].values()):
            num_links = len(app_state['links'])
            failed_paths = sum(1 for path in app_state['paths'].values() if path is None)
            failed_paths_percentage = failed_paths / num_links
            total_reward -= 10 * failed_paths_percentage

        if not all(comp['deployed'] for comp in app_state['components'].values()):
            total_reward += 5 / num_components

        if action == (-1, -1):
            total_reward -= 50

        print(f"App {app_index} Reward Calculation Debug:")
        print(f"  Energy cost: {energy_cost}")
        print(f"  Latency penalty: {latency_penalty}")
        print(f"  Total calculated reward for the action: {total_reward}")

        return total_reward

    def calculate_latency_penalty(self, app_index, link_id, path):
        app = self.state['applications'][app_index]
        total_latency = sum(next(link_data['latency'] for link_data in self.state['infra_links'].values()
                                 if link_data['source'] == link[0] and link_data['destination'] == link[1])
                            for link in path)
        if link_id in app['links']:
            latency_req = app['links'][link_id]['latency']
        else:
            raise ValueError("Link id not in app")
        penalty = 0
        if total_latency > latency_req:
            penalty += (total_latency - latency_req)
        else:
            penalty -= (latency_req - total_latency)
        app['latency_penalties'][link_id] = penalty

    def calculate_final_reward(self):
        final_reward = 0
        total_energy_cost = 0

        for app_index in range(len(self.state['applications'])):
            app_reward = self.calculate_reward(None, app_index)
            final_reward += app_reward

        active_hosts = set(comp['host'] for app in self.state['applications'] for comp in app['components'].values() if comp['deployed'])
        for host_id in active_hosts:
            host = self.state['hosts'][host_id]
            total_energy_cost += host['CPU'] * 0.04 + host['RAM'] * 0.02
        
        final_reward -= total_energy_cost
        final_reward /= len(self.state['applications'])

        return final_reward

    def reset(self, topology_file=None):
        if topology_file:
            self.app_configs, self.infra_config = load_json_config("DRL/data/Application.json", topology_file)
        self.state = self.initialize_state()
        self.current_step = 0
        self.path_cache.clear()
        flattened_state = flatten_state(self.state)
        assert flattened_state.shape[0] == calculate_state_size(self.state), "State size mismatch on reset!"
        return self.state

    def check_all_deployed(self):
        return all(comp['deployed'] for app in self.state['applications'] for comp in app['components'].values())

    def check_if_inter_host(self, path):
        if not path:
            return False
        source_host = path[0][0] if path else None
        dest_host = path[-1][1] if path else None
        return source_host is not None and dest_host is not None and source_host != dest_host

    def log_path(self, link_id, path, penalty):
        with open(self.log_file, 'a') as f:
            f.write(f"Link {link_id}: Path={path}, Latency Penalty/Reward={penalty}\n")

    def save_deployment_strategy(self, reward):
        with open(self.log_file, 'a') as f:
            f.write("\n--- Deployment Strategy ---\n")
            f.write(f"Topology: {self.topology_file}\n")
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
                    penalty = app['latency_penalties'][link_id]
                    f.write(f"  Link {link_id}: Path={path}, Latency Penalty/Reward={penalty}\n")

            f.write(f"Total reward: {reward}\n")

    def render(self, mode='console'):
        if mode == 'console':
            self.print_state()

    def print_state(self):
        print("\n--- Current State ---")
        for app in self.state['applications']:
            for comp_id, comp in app['components'].items():
                host_id = comp.get('host', None)
                status = "Deployed" if comp['deployed'] else "Not Deployed"
                print(f"  Component {comp_id}: CPU={comp['CPU']}, RAM={comp['RAM']}, Status={status}, Host={host_id}")

        print("Paths:")
        for app in self.state['applications']:
            for link_id, path in app['paths'].items():
                penalty = app['latency_penalties'][link_id]
                print(f"  Link {link_id}: Path={path}, Latency Penalty (negative means positive reward)={penalty}")

        print("----------------------")

    def close(self):
        pass
