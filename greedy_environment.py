import gym
import numpy as np
import random
from gym import spaces
from utils import load_json_config
import heapq

class GreedyEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, topology_file, num_applications=None, mode='train'):
        super(GreedyEnvironment, self).__init__()
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
        self.num_hosts = len(self.infra_config['hosts']['configuration'])
        self.current_step = 0
        self.max_steps_per_episode = 100

    def initialize_state(self):
        shuffled_hosts = list(self.infra_config['hosts']['configuration'])
        #random.shuffle(shuffled_hosts)
        state = {
            'hosts': {i: {'CPU': host['CPU'], 'RAM': host['BW'], 'total_CPU': host['CPU'], 'total_RAM': host['BW']} for i, host in enumerate(shuffled_hosts)},
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

    def infer_host_type(self, host):
        if host['total_CPU'] <= 15:
            return 'iot'
        elif host['total_CPU'] <= 200:
            return 'edge'
        else:
            return 'cloud'

    def step(self, action, app_index):
        self.current_step += 1
        component_id, host_id = action
        print(f"\nStep {self.current_step}: Trying to deploy component {component_id} to host {host_id}")

        app_state = self.state['applications'][app_index]
        num_components = len(app_state['components'])

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
                    if len(path) != 0:
                        self.reduce_bandwidth(path, link['bandwidth'])
                        self.calculate_latency_penalty(app_index, link_id, path)
                    else :
                        self.calculate_latency_penalty(app_index, link_id, path)

        done = self.check_all_deployed()
        reward = self.calculate_reward(action, app_index)
        done = done or self.current_step >= self.max_steps_per_episode
        next_state = self.state

        if done:
            final_reward = self.calculate_final_reward()

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
            if link_data and 'bandwidth' in link_data and link_data['bandwidth'] >= required_bandwidth:
                link_data['bandwidth'] -= required_bandwidth

    def calculate_energy_cost(self, host):
        # Infer the host type based on its total resources
        host_type = self.infer_host_type(host)
        if host_type == 'iot':
            base_power = 1
            power_per_cpu = 0.08
            power_per_ram = 0.04
        elif host_type == 'edge':
            base_power = 3
            power_per_cpu = 0.4
            power_per_ram = 0.2
        else:  # 'cloud'
            base_power = 8
            power_per_cpu = 1.0
            power_per_ram = 0.5
        
        used_cpu = host['total_CPU'] - host['CPU']
        used_ram = host['total_RAM'] - host['RAM']
        energy_cost = base_power + used_cpu * power_per_cpu + used_ram * power_per_ram
        
        return energy_cost

    def calculate_total_energy_cost(self):
        total_energy_cost = 0
        calculated_hosts = set()

        for app_state in self.state['applications']:
            for comp in app_state['components'].values():
                if comp['deployed']:
                    host_id = comp['host']
                    if host_id not in calculated_hosts:
                        host = self.state['hosts'][host_id]
                        total_energy_cost += self.calculate_energy_cost(host)
                        calculated_hosts.add(host_id)

        # Return the total energy cost for the hosts used, without normalization
        return total_energy_cost

    def calculate_total_latency_penalty(self, app_index):
        app_state = self.state['applications'][app_index]
        total_latency_penalty = sum(app_state['latency_penalties'].values())

        return total_latency_penalty

    def calculate_reward(self, action, app_index):
        app_state = self.state['applications'][app_index]
        num_components = len(app_state['components'])

        # Calculate energy cost
        if action is not None:
            component_id, host_id = action
            component = app_state['components'][component_id]
            host = self.state['hosts'][host_id]
            energy_cost = self.calculate_energy_cost(host) / (num_components or 1)
        else:
            energy_cost = self.calculate_total_energy_cost()

        # Calculate latency penalty
        latency_penalty = 0
        if action is not None:
            component_id, host_id = action
            for link_id, link in app_state['links'].items():
                if link['source'] == component_id or link['destination'] == component_id:
                    if app_state['components'][link['source']]['deployed'] and app_state['components'][link['destination']]['deployed']:
                        if app_state['paths'][link_id] is not None:
                            latency_penalty += app_state['latency_penalties'][link_id]
        else:
            latency_penalty = self.calculate_total_latency_penalty(app_index)

        # Calculate total reward
        total_reward = 0

        total_reward += 20 / (1 + energy_cost) 

        # Penalize latency if it exceeds the required latency
        total_reward -= latency_penalty

       

        # Additional penalty for failed paths after all components are deployed
        if all(comp['deployed'] for comp in app_state['components'].values()):
            num_links = len(app_state['links'])
            failed_paths = sum(1 for path in app_state['paths'].values() if path is None)
            failed_paths_percentage = failed_paths / num_links
            total_reward -= 10 * failed_paths_percentage

        # Large penalty for invalid actions
        if action == (-1, -1):
            total_reward -= 50

        print(f"App {app_index} Reward Calculation Debug:")
        print(f"  Energy efficiency: {100 / (1 + energy_cost)}")
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
            penalty -= ((latency_req - total_latency) + (20 / len(app["links"])))

    
        if len(path)==0 :
           
            app['latency_penalties'][link_id] = -(20 / len(app["links"])) # still give a reward for intra host paths []
        else:
            app['latency_penalties'][link_id] = penalty
           

        app['latency_penalties'][link_id] = penalty

    def calculate_final_reward(self):
        final_reward = 0

        for app_index in range(len(self.state['applications'])):
            app_reward = self.calculate_reward(None, app_index)
            final_reward += app_reward

        #final_reward /= len(self.state['applications'])

        return final_reward

    def reset(self, topology_file=None):
        if topology_file:
            self.app_configs, self.infra_config = load_json_config("DRL/data/Application.json", topology_file)
        self.state = self.initialize_state()
        self.current_step = 0
        self.path_cache.clear()
        return self.state

    def check_all_deployed(self):
        return all(comp['deployed'] for app in self.state['applications'] for comp in app['components'].values())

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
