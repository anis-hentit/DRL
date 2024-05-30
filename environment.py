import gym
import numpy as np
from gym import spaces
from utils import load_json_config

class FogEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(FogEnvironment, self).__init__()
        self.app_config, self.infra_config = load_json_config('data/Application.json', 'data/Graph_Infra.json')
        self.state = self.initialize_state()
        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()

    def define_action_space(self):
        num_components = len(self.state['components'])
        num_hosts = len(self.state['hosts'])
        return spaces.Tuple((spaces.Discrete(num_components), spaces.Discrete(num_hosts)))

    def define_observation_space(self):
        num_components = len(self.state['components'])
        num_hosts = len(self.state['hosts'])
        return spaces.Dict({
            'hosts': spaces.Box(low=0, high=100, shape=(num_hosts, 2), dtype=np.float32),
            'components': spaces.Box(low=0, high=100, shape=(num_components, 3), dtype=np.float32)
        })

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
                    host['CPU'] -= comp['CPU']
                    host['RAM'] -= comp['RAM']
                    comp['deployed'] = True
                    comp['host'] = host_id
                else:
                    raise ValueError(f"Host {host_id} does not have enough resources to deploy component {comp_id} at initialization.")
        return state

    def step(self, action):
        component_id, host_id = action
        path_ids = {}  # Example path_ids, this should be defined based on your action

        if self.state['components'][component_id]['deployed']:
            return self.state, -10, True, {}

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
                reward = -100
                done = False
        else:
            reward = -100
            done = False

        return self.state, reward, done, {}

    def validate_path(self, path, latency_req, bandwidth_req):
        total_latency = 0
        min_bandwidth = float('inf')
        
        for link in path:
            link_data = next((item for item in self.infra_config['network']['topology'] 
                              if item['source'] == link[0] and item['destination'] == link[1]), None)
            if not link_data or link_data['source'] == link_data['destination']:
                continue  # Skip self-links and invalid links
            
            link_bandwidth = link_data['bandwidth']
            link_latency = link_data['latency']
            
            total_latency += link_latency
            min_bandwidth = min(min_bandwidth, link_bandwidth)
        
        return total_latency <= latency_req and min_bandwidth >= bandwidth_req

    def update_state(self, component_id, host_id, path_ids):
        component = self.state['components'][component_id]
        host = self.state['hosts'][host_id]
        if component['CPU'] <= host['CPU'] and component['RAM'] <= host['RAM']:
            host['CPU'] -= component['CPU']
            host['RAM'] -= component['RAM']
            self.state['components'][component_id]['deployed'] = True
            self.state['components'][component_id]['host'] = host_id
            for path_id, path_data in path_ids.items():
                self.state['paths'][path_id] = path_data
            return True
        else:
            return False

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

    def render(self, mode='console'):
        if mode == 'console':
            print("Current state:", self.state)

    def close(self):
        pass
