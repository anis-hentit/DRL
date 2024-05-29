import gym
import numpy as np
from utils import load_json_config

class FogEnvironment:
    def __init__(self):
        # Load configurations using the JSON loader
        self.app_config, self.infra_config = load_json_config('data/Application.json', 'data/Graph_Infra.json')
        self.state = self.initialize_state()

    def initialize_state(self):
        state = {
            'hosts': {i: {'CPU': host['CPU'], 'RAM': host['BW']} for i, host in enumerate(self.infra_config['hosts']['configuration'])},
            'components': {i: {'CPU': comp['CPU'], 'RAM': comp['RAM'], 'deployed': False} for i, comp in enumerate(self.app_config['application']['requirements'])},
            'fixed_positions': {comp['component_id']: comp['host_id'] for comp in self.app_config['application'].get('DZ', [])},
            'links': {i: link for i, link in enumerate(self.app_config['application']['links'])},
            'paths': {i: [] for i in range(len(self.app_config['application']['links']))}
            }
        # Automatically deploy and subtract resources for fixed position components
        for comp_id, host_id in state['fixed_positions'].items():
            if comp_id in state['components']:
                comp = state['components'][comp_id]
                host = state['hosts'][host_id]
                if comp['CPU'] <= host['CPU'] and comp['RAM'] <= host['RAM']:
                    # Update host resources
                    host['CPU'] -= comp['CPU']
                    host['RAM'] -= comp['RAM']
                    # Mark component as deployed
                    comp['deployed'] = True
                    comp['host'] = host_id
                else:
                    raise ValueError(f"Host {host_id} does not have enough resources to deploy component {comp_id} at initialization.")
        return state


    def step(self, action):
        component_id, host_id, path_ids = action
        if self.state['components'][component_id]['deployed']:
            return self.state, -10, True, {}  # Penalize re-deployment attempts

        component = self.state['components'][component_id]
        host = self.state['hosts'][host_id]
        valid_paths = all(self.validate_path(path, latency_req, bandwidth_req) 
                        for path_id, (path, latency_req, bandwidth_req) in path_ids.items())

        if component['CPU'] <= host['CPU'] and component['RAM'] <= host['RAM'] and valid_paths:
            success = self.update_state(component_id, host_id, path_ids)
            if success:
                done = self.check_all_deployed()
                reward = self.calculate_reward()
            else:
                reward = -100  # Severe penalty for attempting an impossible deployment
                done = False
        else:
            reward = -100  # Penalize for resource or path constraints
            done = False

        return self.state, reward, done, {}


    def validate_path(self, path, latency_req, bandwidth_req):
        total_latency = sum(self.infra_config['network']['topology'][link]['latency'] for link in path)
        min_bandwidth = min(self.infra_config['network']['topology'][link]['bandwidth'] for link in path)
        return total_latency <= latency_req and min_bandwidth >= bandwidth_req

    def update_state(self, component_id, host_id, path_ids):
        component = self.state['components'][component_id]
        host = self.state['hosts'][host_id]

        # Update resources if possible, else handle resource shortfall
        if component['CPU'] <= host['CPU'] and component['RAM'] <= host['RAM']:
            host['CPU'] -= component['CPU']
            host['RAM'] -= component['RAM']
            self.state['components'][component_id]['deployed'] = True
            self.state['components'][component_id]['host'] = host_id
            for path_id, path_data in path_ids.items():
                self.state['paths'][path_id] = path_data
            return True  # Successful deployment
        else:
            return False  # Failed deployment due to insufficient resources


    def check_all_deployed(self):
        return all(comp['deployed'] for comp in self.state['components'].values())

    def calculate_reward(self):
        energy = 0
        active_hosts = set(comp['host'] for comp in self.state['components'].values() if comp['deployed'])
        for host_id in active_hosts:
            host = self.state['hosts'][host_id]
            energy += host['CPU'] * 0.1 + host['RAM'] * 0.05
        return -energy

    def reset(self):
        self.state = self.initialize_state()
        return self.state
