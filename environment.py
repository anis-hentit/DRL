import gym
import numpy as np
from gym import spaces
from utils import load_json_config
from agent import flatten_state

class FogEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(FogEnvironment, self).__init__()
        # Load configuration files
        self.app_config, self.infra_config = load_json_config('data/Application.json', 'data/Graph_Infra.json')
        # Initialize state
        self.state = self.initialize_state()
        # Define action and observation spaces
        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()
        self.current_step = 0
        self.max_steps_per_episode = 100

    def define_action_space(self):
        # Define action space as a tuple of discrete spaces (component, host)
        num_components = len(self.state['components'])
        num_hosts = len(self.state['hosts'])
        return spaces.Tuple((spaces.Discrete(num_components), spaces.Discrete(num_hosts)))

    def define_observation_space(self):
        # Define observation space as a dictionary of Box spaces for hosts and components
        num_components = len(self.state['components'])
        num_hosts = len(self.state['hosts'])
        return spaces.Dict({
            'hosts': spaces.Box(low=0, high=100, shape=(num_hosts, 2), dtype=np.float32),
            'components': spaces.Box(low=0, high=100, shape=(num_components, 3), dtype=np.float32)
        })

    def initialize_state(self):
        # Initialize the state with hosts and components
        state = {
            'hosts': {i: {'CPU': host['CPU'], 'RAM': host['BW']} for i, host in enumerate(self.infra_config['hosts']['configuration'])},
            'components': {i: {'CPU': comp['CPU'], 'RAM': comp['RAM'], 'deployed': False} for i, comp in enumerate(self.app_config['application']['requirements'])},
            'fixed_positions': {comp['component_id']: comp['host_id'] for comp in self.app_config['application'].get('DZ', [])},
            'links': {i: link for i, link in enumerate(self.app_config['application']['links'])},
            'paths': {i: [] for i in range(len(self.app_config['application']['links']))}
        }
        # Deploy components with fixed positions
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
        # Execute an action and update the environment state
        self.current_step += 1
        component_id, host_id, path_ids = action

        if self.state['components'][component_id]['deployed']:
            print("ALREADY DEPLOYED -20")
            return self.state, -20, self.current_step >= self.max_steps_per_episode, {}

        component = self.state['components'][component_id]
        host = self.state['hosts'][host_id]

        print(f"Attempting to deploy Component {component_id} to Host {host_id}")
        print(f"Component requirements: CPU={component['CPU']}, RAM={component['RAM']}")
        print(f"Host available resources: CPU={host['CPU']}, RAM={host['RAM']}")

        # Validate paths
        valid_paths = all(self.validate_path(path, latency_req, bandwidth_req) 
                          for path_id, (path, latency_req, bandwidth_req) in path_ids.items())

        if component['CPU'] <= host['CPU'] and component['RAM'] <= host['RAM'] and valid_paths:
            success = self.update_state(component_id, host_id, path_ids)
            if success:
                done = self.check_all_deployed()
                reward = self.calculate_reward(action, path_ids) + 15
                print("INCREASE OF 15")
            else:
                reward = -5
                print("DECREASE OF 5")
                done = False
        else:
            reward = -20
            print("DECREASE OF 20")
            done = False

        done = done or self.current_step >= self.max_steps_per_episode

        next_state = self.state
        flattened_state = flatten_state(next_state)

        print(f"Action taken: Deploy Component {component_id} to Host {host_id}")
        print(f"Current deployment status: {self.state['components']}")
        print(f"Paths chosen: {path_ids}")
        print(f"Reward this step: {reward}")

        return self.state, reward, done, {}

    def generate_paths(self, component_id, host_id):
        return {}

    def validate_path(self, path, latency_req, bandwidth_req):
        # Validate a path based on latency and bandwidth requirements
        total_latency = 0
        min_bandwidth = float('inf')

        for link in path:
            link_data = next((item for item in self.infra_config['network']['topology'] 
                              if item['source'] == link[0] and item['destination'] == link[1]), None)
            if not link_data or link_data['source'] == link_data['destination']:
                continue

            link_bandwidth = link_data['bandwidth']
            link_latency = link_data['latency']

            total_latency += link_latency
            min_bandwidth = min(min_bandwidth, link_bandwidth)

        is_valid = total_latency <= latency_req and min_bandwidth >= bandwidth_req
        print(f"Validating path {path} with total_latency={total_latency}, min_bandwidth={min_bandwidth}, "
              f"latency_req={latency_req}, bandwidth_req={bandwidth_req} => is_valid={is_valid}")
        return is_valid

    def calculate_state_size(self, state):
        # Calculate the size of the flattened state
        flattened_state = flatten_state(state)
        return len(flattened_state)

    def update_state(self, component_id, host_id, path_ids):
        # Update the state after deploying a component
        component = self.state['components'][component_id]
        host = self.state['hosts'][host_id]
        if component['CPU'] <= host['CPU'] and component['RAM'] <= host['RAM']:
            host['CPU'] -= component['CPU']
            host['RAM'] -= component['RAM']
            self.state['components'][component_id]['deployed'] = True
            self.state['components'][component_id]['host'] = host_id
            for path_id, path_data in path_ids.items():
                self.state['paths'][path_id] = path_data
            print(f"Update: Component {component_id} deployed to Host {host_id}")
            return True
        else:
            print(f"Update failed: Component {component_id} could not be deployed to Host {host_id}")
            return False

    def check_all_deployed(self):
        # Check if all components have been deployed
        return all(comp['deployed'] for comp in self.state['components'].values())

    def calculate_reward(self, action, path_ids):
        # Calculate the reward based on energy consumption and penalties
        energy = 0
        active_hosts = set(comp['host'] for comp in self.state['components'].values() if comp['deployed'])
        for host_id in active_hosts:
            host = self.state['hosts'][host_id]
            energy += host['CPU'] * 0.1 + host['RAM'] * 0.05

        latency_penalty = self.calculate_latency_penalty(path_ids)
        bandwidth_penalty = self.calculate_bandwidth_penalty(path_ids)

        total_reward = -energy - latency_penalty - bandwidth_penalty
        print(f"Energy cost: {-energy}, Latency penalty: {-latency_penalty}, Bandwidth penalty: {-bandwidth_penalty}")
        print(f"Total calculated reward: {total_reward}")
        return total_reward

    def calculate_latency_penalty(self, path_ids):
        # Calculate latency penalties for the paths
        penalty = 0
        for path_id, (path, latency_req, _) in path_ids.items():
            total_latency = sum(next(link_data['latency'] for link_data in self.infra_config['network']['topology']
                                     if link_data['source'] == link[0] and link_data['destination'] == link[1])
                                for link in path)
            if total_latency > latency_req:
                penalty += (total_latency - latency_req)
        return penalty

    def calculate_bandwidth_penalty(self, path_ids):
        # Calculate bandwidth penalties for the paths
        penalty = 0
        for path_id, (path, _, bandwidth_req) in path_ids.items():
            min_bandwidth = min(next(link_data['bandwidth'] for link_data in self.infra_config['network']['topology']
                                    if link_data['source'] == link[0] and link_data['destination'] == link[1])
                               for link in path)
            if min_bandwidth < bandwidth_req:
                penalty += (bandwidth_req - min_bandwidth)
        return penalty

    def reset(self):
        # Reset the environment to the initial state
        self.state = self.initialize_state()
        self.current_step = 0
        flattened_state = flatten_state(self.state)
        print(f"Reset state: {self.state}")
        print(f"Reset state flattened size: {flattened_state.shape}")
        assert flattened_state.shape[0] == self.calculate_state_size(self.state), "State size mismatch on reset!"
        return self.state

    def render(self, mode='console'):
        # Render the current state of the environment
        if mode == 'console':
            print("Current state:", self.state)

    def close(self):
        # Close the environment
        pass
