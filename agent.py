import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import heapq

def flatten_state(state):
    hosts = state['hosts']
    components = state['components']
    
    # Flatten hosts
    host_values = np.concatenate([list(host.values()) for host in hosts.values()])
    
    # Flatten components excluding 'deployed' attribute
    component_values = np.concatenate([list(component.values())[:2] for component in components.values()])
    
    # make sure 'deployed' attribute is always 0 or 1
    deployed_status = np.array([1.0 if component['deployed'] else 0.0 for component in components.values()], dtype=np.float32)
    
    # concatenate all parts
    flattened_state = np.concatenate([host_values, component_values, deployed_status])
    
    # Debug: Print detailed information
    print(f"host_values: {host_values} (size: {host_values.size})")
    print(f"component_values: {component_values} (size: {component_values.size})")
    print(f"deployed_status: {deployed_status} (size: {deployed_status.size})")
    
    return flattened_state

class Agent:
    def __init__(self, state_size, num_components, num_hosts, infra_config, learning_rate=0.001):
        self.state_size = state_size
        self.num_components = num_components
        self.num_hosts = num_hosts
        self.infra_config = infra_config
        self.action_size = num_components * num_hosts
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        layer1 = layers.Dense(24, activation='relu')(inputs)
        layer2 = layers.Dense(24, activation='relu')(layer1)
        outputs = layers.Dense(self.action_size, activation='softmax')(layer2)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy')
        
        return model

    def choose_action(self, state):
        flattened_state = flatten_state(state)
        flattened_state = np.reshape(flattened_state, [1, self.state_size])
        
        valid_actions = []
        for component_id in range(self.num_components):
            if not state['components'][component_id]['deployed']:
                for host_id in range(self.num_hosts):
                    if state['components'][component_id]['CPU'] <= state['hosts'][host_id]['CPU'] and state['components'][component_id]['RAM'] <= state['hosts'][host_id]['RAM']:
                        valid_actions.append((component_id, host_id))
        
        if np.random.rand() <= self.epsilon:
            if valid_actions:
                component_id, host_id = valid_actions[np.random.choice(len(valid_actions))]
                action = component_id * self.num_hosts + host_id
            else:
                action = np.random.choice(self.action_size)  # If no valid actions, choose randomly to avoid crash
        else:
            probabilities = self.model.predict(flattened_state)[0]
            sorted_actions = np.argsort(probabilities)[::-1]
            for action in sorted_actions:
                component_id = action // self.num_hosts
                host_id = action % self.num_hosts
                if (component_id, host_id) in valid_actions:
                    break
        
        return self.decode_action(action, state)

    def decode_action(self, action_index, state):
        component_id = action_index // self.num_hosts
        host_id = action_index % self.num_hosts
        path_ids = self.generate_paths(component_id, host_id, state)
        return (component_id, host_id, path_ids)

    def encode_action(self, action):
        component_id, host_id, _ = action
        return component_id * self.num_hosts + host_id

    def generate_paths(self, component_id, host_id, state):
        paths = {}
        link_id_counter = 0
        deployed_components = {comp_id: comp['host'] for comp_id, comp in state['components'].items() if comp['deployed']}
        
        for link in state['links'].values():
            source_comp = link['source']
            destination_comp = link['destination']
            if source_comp in deployed_components and destination_comp in deployed_components:
                source_host = deployed_components[source_comp]
                destination_host = deployed_components[destination_comp]
                if source_host != destination_host:  # Only generate paths if components are on different hosts
                    path, latency = self.dijkstra(source_host, destination_host, link['latency'], link['bandwidth'])
                    if path:
                        paths[link_id_counter] = (path, latency, link['bandwidth'])
                        link_id_counter += 1
                
        print(f"Generated paths: {paths}")
        return paths

    def learn(self, states, actions, rewards):
        flattened_states = np.vstack([flatten_state(state) for state in states])
        
        print(f"learn - states shape: {flattened_states.shape}")
        
        actions = np.array([self.encode_action(action) for action in actions])
        rewards = np.array(rewards)

        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)

        with tf.GradientTape() as tape:
            predictions = self.model(flattened_states)
            action_masks = tf.one_hot(actions, self.action_size)
            log_probs = tf.reduce_sum(action_masks * tf.math.log(predictions), axis=1)
            loss = -tf.reduce_sum(log_probs * rewards)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def dijkstra(self, source, target, latency_req, bandwidth_req):
        graph = {i: [] for i in range(len(self.infra_config['network']['topology']))}
        for link in self.infra_config['network']['topology']:
            if link['bandwidth'] >= bandwidth_req:
                graph[link['source']].append((link['destination'], link['latency'], link['bandwidth']))

        queue = [(0, source, [])]
        seen = set()
        while queue:
            (cost, node, path) = heapq.heappop(queue)
            if node in seen:
                continue
            new_path = path + [node]
            seen.add(node)
            if node == target:
                complete_path = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
                return complete_path, cost
            for next_node, latency, bandwidth in graph.get(node, []):
                if next_node not in seen and cost + latency <= latency_req:
                    heapq.heappush(queue, (cost + latency, next_node, new_path))

        print(f"No valid path found from {source} to {target} with latency requirement {latency_req} and bandwidth requirement {bandwidth_req}")
        return None, float('inf')
