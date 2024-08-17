import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import mt_albis
import numpy as np
import random
import os

def flatten_state(state):
    hosts = state['hosts']
    applications = state['applications']
    infra_links = state['infra_links']

<<<<<<< HEAD
    # Ensure consistency in ordering of values
=======
>>>>>>> 1678550bf925cc138a94f5f33500cc4e4e487090
    host_values = np.concatenate([list(host.values()) for host in hosts.values()])
    component_values = np.concatenate([list(component.values())[:2] for app in applications for component in app['components'].values()])
    deployed_status = np.array([1.0 if component['deployed'] else 0.0 for app in applications for component in app['components'].values()], dtype=np.float32)
    link_values = np.concatenate([list(link.values()) for app in applications for link in app['links'].values()])
    infra_link_values = np.concatenate([list(link.values()) for link in infra_links.values()])

    flattened_state = np.concatenate([host_values, component_values, deployed_status, link_values, infra_link_values])
    return flattened_state

def generate_graph_data(state, app_index):
    hosts = state['hosts']
    applications = state['applications']
    infra_links = state['infra_links']
    app = applications[app_index]

    node_features = []
    node_indices = {}
    idx = 0

<<<<<<< HEAD
    # Add host nodes
=======
>>>>>>> 1678550bf925cc138a94f5f33500cc4e4e487090
    for host_id, host in hosts.items():
        node_features.append([host['CPU'], host['RAM'], 0.0])
        node_indices[('host', host_id)] = idx
        idx += 1

<<<<<<< HEAD
    # Add component nodes
=======
>>>>>>> 1678550bf925cc138a94f5f33500cc4e4e487090
    for comp_id, comp in app['components'].items():
        node_features.append([comp['CPU'], comp['RAM'], 1.0 if comp['deployed'] else 0.0])
        node_indices[('component', app_index, comp_id)] = idx
        idx += 1

    edge_list = []
    edge_features = []

<<<<<<< HEAD
    # Add infrastructure links as edges
=======
>>>>>>> 1678550bf925cc138a94f5f33500cc4e4e487090
    for link in infra_links.values():
        if link['source'] in hosts and link['destination'] in hosts:
            latency = link.get('latency', 0)
            bandwidth = link.get('bandwidth', 0)
            edge_list.append([node_indices[('host', link['source'])], node_indices[('host', link['destination'])]])
            edge_features.append([latency, bandwidth])
<<<<<<< HEAD
            
        else:
            print(f"Skipping invalid link with source {link['source']} and destination {link['destination']}")

    # Add application links as edges
    for link in app['links'].values():
        if link['source'] in app['components'] and link['destination'] in app['components']:
            latency = link.get('latency', 0)
            bandwidth = link.get('bandwidth', 0)
            edge_list.append([node_indices[('component', app_index, link['source'])], node_indices[('component', app_index, link['destination'])]])
            edge_features.append([latency, bandwidth])
            
        else:
            print(f"Skipping invalid link with source {link['source']} and destination {link['destination']}")

    #print("Node Features:")
    #print(node_features)
    #print("Edge List:")
    #print(edge_list)
    #print("Edge Features:")
    #print(edge_features)

    node_features = np.array(node_features, dtype=np.float32)
    edge_list = np.array(edge_list, dtype=np.int32).T
    edge_features = np.array(edge_features, dtype=np.float32)

    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={'nodes': tfgnn.NodeSet.from_fields(
            sizes=[len(node_features)],
            features={'hidden_state': tf.constant(node_features)}
        )},
        edge_sets={'edges': tfgnn.EdgeSet.from_fields(
            sizes=[len(edge_features)],
            adjacency=tfgnn.Adjacency.from_indices(
                source=('nodes', tf.constant(edge_list[0])),
                target=('nodes', tf.constant(edge_list[1]))
            ),
            features={'features': tf.constant(edge_features)}
        )}
    )

    return graph



def compute_action_mask(state, app_index, max_components, max_hosts):
    num_hosts = len(state['hosts'])
    mask = np.zeros((max_components, max_hosts))

    app = state['applications'][app_index]
    for comp_id, comp in app['components'].items():
        if not comp['deployed']:
            for host_id, host in state['hosts'].items():
                if comp['CPU'] <= host['CPU'] and comp['RAM'] <= host['RAM']:
                    mask[comp_id, host_id] = 1

    mask = mask.flatten()
    if len(mask) < max_components * max_hosts:
        mask = np.pad(mask, (0, max_components * max_hosts - len(mask)), 'constant', constant_values=0)

    return mask


class GNNAgent(tf.keras.Model):
    def __init__(self, hidden_dim, max_components, max_hosts, num_layers=2, attention_type="gat_v2", attention_num_heads=4):
        super(GNNAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_components = max_components
        self.max_hosts = max_hosts

        self.projection_layer = tf.keras.layers.Dense(hidden_dim,activation='relu')
        self.graph_update_layers = [
            mt_albis.MtAlbisGraphUpdate(
                units=64,
                message_dim=512,
                receiver_tag=tfgnn.TARGET,
                simple_conv_reduce_type="mean",
                normalization_type="layer",
                next_state_type="residual",
                state_dropout_rate=0.1,
                
                
                
                
            ) for _ in range(num_layers)
        ]
        self.dense_layer = tf.keras.layers.Dense(max_components * max_hosts)

    @tf.function
    def call(self, graph, training=False):
        node_features = graph.node_sets['nodes']['hidden_state']
        projected_features = self.projection_layer(node_features)
        graph = graph.replace_features(node_sets={'nodes': {'hidden_state': projected_features}})

        for graph_update in self.graph_update_layers:
            graph = graph_update(graph)

        updated_features = graph.node_sets['nodes']['hidden_state']
        aggregated_features = tf.reduce_mean(updated_features, axis=0)
        aggregated_features = tf.expand_dims(aggregated_features, axis=0)

        return self.dense_layer(aggregated_features)

    def save(self, filepath):
        self.save_weights(os.path.join(filepath, 'gnn_weights'))

    def load(self, filepath):
        self.load_weights(os.path.join(filepath, 'gnn_weights'))



class Agent:
    def __init__(self, input_dim, hidden_dim, max_components, num_hosts, learning_rate=0.002, num_layers=2, attention_type="gat_v2", attention_num_heads=4):
        
        self.model = GNNAgent(hidden_dim, max_components, num_hosts, num_layers, attention_type, attention_num_heads)
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate,weight_decay=0.002)
        self.gamma = 0.80
        self.temperature = 1.0
        self.temperature_decay = 0.996
        self.temperature_min = 0.0001
        self.max_components = max_components
        self.num_hosts = num_hosts

        self.replay_buffer_model = []
        self.replay_buffer_random = []
        self.temp_experiences = []

    def choose_action(self, state, app_index, mode='train', num_hosts=None):
        if num_hosts is None:
            num_hosts = self.num_hosts

        num_components = len(state['applications'][app_index]['components'])
        temperature = 1e-8 if mode == 'inference' else self.temperature
        graph = generate_graph_data(state, app_index)
        logits = self.model(graph, training=False).numpy().flatten()

        # Print logits
        #print(f"Logits: {logits}")

        # Adjust the size of logits based on the current number of hosts and components
        logits = logits[:num_components * num_hosts]

        mask = compute_action_mask(state, app_index, num_components, num_hosts)
        masked_logits = np.where(mask, logits, -np.inf)

        # Print masked logits
        #print(f"Masked Logits: {masked_logits}")

        if np.all(masked_logits == -np.inf):
            return None, None, None

        try:
            max_logit = np.max(masked_logits[np.isfinite(masked_logits)])
            stabilized_logits = masked_logits - max_logit
            exp_logits = np.exp(stabilized_logits / temperature)
            action_probs = exp_logits / np.sum(exp_logits)
        except ZeroDivisionError:
            return None, None, None

        # Print action probabilities
        #print(f"Action Probabilities: {action_probs}")

        if np.any(np.isnan(action_probs)) or np.any(np.isinf(action_probs)):
            print(f"Action probabilities contain NaN or Inf values: {action_probs}")
            return None, None, None

        valid_indices = np.where(mask)[0]
        #self.model.summary()
        if valid_indices.size == 0:
            print(f"No valid actions available for app {app_index}. Ending episode.")
            return None, None, None

        if mode == 'inference':
            
            action_index = np.argmax(action_probs)
            explore = False
            print(f"Inference mode for app {app_index}: Chose best action index {action_index}")
        else:
            action_index = np.random.choice(valid_indices, p=action_probs[valid_indices])
            explore = action_probs[action_index] < np.max(action_probs)
            print(f"Training mode for app {app_index}: Chose action index {action_index} with temperature {self.temperature}, explore: {explore}")

        component_id, host_id = divmod(action_index, num_hosts)

        if component_id >= self.max_components or host_id >= num_hosts or mask[action_index] == 0:
            print(f"Invalid action for app {app_index}: Trying to deploy component {component_id} to host {host_id} with insufficient resources.")
            return None, None, None

        print(f"Chosen action for app {app_index} - Component: {component_id}, Host: {host_id}")
        return (component_id, host_id), action_index, explore


    def store_experience(self, state, action, reward, next_state, done, app_index, explore):
        self.temp_experiences.append((state, action, reward, next_state, done, app_index, explore))
        if done:
            self._flush_experiences()

    def _flush_experiences(self):
        rewards = [exp[2] for exp in self.temp_experiences]
        discounted_rewards = self.discount_rewards(rewards)
        for i, (state, action, reward, next_state, done, app_index, explore) in enumerate(self.temp_experiences):
            reward = discounted_rewards[i]
            if explore:
                self.replay_buffer_random.append((state, action, reward, next_state, done, app_index))
                if len(self.replay_buffer_random) > 10000:
                    self.replay_buffer_random.pop(0)
            else:
                self.replay_buffer_model.append((state, action, reward, next_state, done, app_index))
                if len(self.replay_buffer_model) > 10000:
                    self.replay_buffer_model.pop(0)
        self.temp_experiences = []

    def learn(self, batch_size=32):
        print(f"Replay buffer sizes - Model: {len(self.replay_buffer_model)}, Random: {len(self.replay_buffer_random)}")
        
        total_experiences = len(self.replay_buffer_model) + len(self.replay_buffer_random)
        if total_experiences < batch_size:
            print("Not enough samples to learn.")
            #if self.temperature > self.temperature_min:
               # self.temperature *= self.temperature_decay
               #self.temperature = max(self.temperature, self.temperature_min)
            print(f"Temperature after checking buffer size: {self.temperature}")
=======
            edge_list.append([node_indices[('host', link['destination'])], node_indices[('host', link['source'])]])
            edge_features.append([latency, bandwidth])
        else:
            print(f"Skipping invalid link with source {link['source']} and destination {link['destination']}")

    for link in app['links'].values():
        if link['source'] in app['components'] and link['destination'] in app['components']:
            latency = link.get('latency', 0)
            bandwidth = link.get('bandwidth', 0)
            edge_list.append([node_indices[('component', app_index, link['source'])], node_indices[('component', app_index, link['destination'])]])
            edge_features.append([latency, bandwidth])
            edge_list.append([node_indices[('component', app_index, link['destination'])], node_indices[('component', app_index, link['source'])]])
            edge_features.append([latency, bandwidth])
        else:
            print(f"Skipping invalid link with source {link['source']} and destination {link['destination']}")

    node_features = np.array(node_features, dtype=np.float32)
    edge_list = np.array(edge_list, dtype=np.int32).T
    edge_features = np.array(edge_features, dtype=np.float32)

    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={'nodes': tfgnn.NodeSet.from_fields(
            sizes=[len(node_features)],
            features={'hidden_state': tf.constant(node_features)}
        )},
        edge_sets={'edges': tfgnn.EdgeSet.from_fields(
            sizes=[len(edge_features)],
            adjacency=tfgnn.Adjacency.from_indices(
                source=('nodes', tf.constant(edge_list[0])),
                target=('nodes', tf.constant(edge_list[1]))
            ),
            features={'features': tf.constant(edge_features)}
        )}
    )

    return graph

def compute_action_mask(state, app_index, max_components, max_hosts):
    num_hosts = len(state['hosts'])
    mask = np.zeros((max_components, max_hosts))

    app = state['applications'][app_index]
    for comp_id, comp in app['components'].items():
        if not comp['deployed']:
            for host_id, host in state['hosts'].items():
                if comp['CPU'] <= host['CPU'] and comp['RAM'] <= host['RAM']:
                    mask[comp_id, host_id] = 1

    mask = mask.flatten()
    if len(mask) < max_components * max_hosts:
        mask = np.pad(mask, (0, max_components * max_hosts - len(mask)), 'constant', constant_values=0)

 

    return mask


class GNNAgent(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(GNNAgent, self).__init__()
        self.hidden_dim = hidden_dim

        self.projection_layer = tf.keras.layers.Dense(hidden_dim)
        self.graph_update = mt_albis.MtAlbisGraphUpdate(
            units=hidden_dim,
            message_dim=hidden_dim,
            simple_conv_reduce_type="mean",
            normalization_type="layer",
            next_state_type="residual",
            edge_dropout_rate=0.2,
            state_dropout_rate=0.2,
            l2_regularization=1e-5,
            receiver_tag=tfgnn.TARGET
        )
        self.dense_layer = tf.keras.layers.Dense(100 * hidden_dim)

    @tf.function
    def call(self, graph, training=False):
        node_features = graph.node_sets['nodes']['hidden_state']
        projected_features = self.projection_layer(node_features)
        graph = graph.replace_features(node_sets={'nodes': {'hidden_state': projected_features}})

        graph = self.graph_update(graph)
        updated_features = graph.node_sets['nodes']['hidden_state']
        
        aggregated_features = tf.reduce_mean(updated_features, axis=0)
        aggregated_features = tf.expand_dims(aggregated_features, axis=0)

        return self.dense_layer(aggregated_features)

    def save(self, filepath):
        self.save_weights(os.path.join(filepath, 'gnn_weights'))

    def load(self, filepath):
        self.load_weights(os.path.join(filepath, 'gnn_weights'))

class Agent:
    def __init__(self, input_dim, hidden_dim, max_components, num_hosts, learning_rate=0.001):
        learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            alpha=0.0
        )
        self.model = GNNAgent(hidden_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.max_components = max_components
        self.num_hosts = num_hosts  # This is the number of hosts the agent was trained with

        self.replay_buffer_model = []
        self.replay_buffer_random = []
        self.temp_experiences = []

    def choose_action(self, state, app_index, mode='train', num_hosts=None):
        if num_hosts is None:
            num_hosts = self.num_hosts

        epsilon = 0 if mode == 'inference' else self.epsilon
        graph = generate_graph_data(state, app_index)
        logits = self.model(graph, training=False).numpy().flatten()
    
        output_dim = self.max_components * num_hosts  # Fixing the output dimension to the max number of hosts
        logits = logits[:output_dim]

        # Compute the mask based on the current number of hosts
        mask = compute_action_mask(state, app_index, self.max_components, num_hosts)
        masked_logits = np.where(mask, logits[:len(mask)], -np.inf)

        

        # Check for NaN or infinite values in masked_logits
        if np.all(masked_logits == -np.inf):
            
            return None, None, None

        # Calculate action probabilities with a small epsilon to avoid division by zero
        try:
            action_probs = np.exp(masked_logits) / (np.sum(np.exp(masked_logits)) + 1e-8)
        except ZeroDivisionError:
            
            return None, None, None

        # Ensure action_probs are valid probabilities
        if np.any(np.isnan(action_probs)) or np.any(np.isinf(action_probs)):
            
            return None, None, None

        valid_indices = np.where(mask)[0]

        if valid_indices.size == 0:
            print(f"No valid actions available for app {app_index}. Ending episode.")
            return None, None, None

        if np.random.rand() <= epsilon:
            action_index = np.random.choice(valid_indices)
            print(f"Exploring for app {app_index}: Chose random action index {action_index} with epsilon {self.epsilon}")
            explore = True
        else:
            action_index = np.argmax(action_probs)
            print(f"Exploiting for app {app_index}: Chose best action index {action_index} with epsilon {self.epsilon}")
            explore = False

        component_id, host_id = divmod(action_index, num_hosts)

        if component_id >= self.max_components or host_id >= num_hosts or mask[action_index] == 0:
            print(f"Invalid action for app {app_index}: Trying to deploy component {component_id} to host {host_id} with insufficient resources.")
            return None, None, None

        print(f"Chosen action for app {app_index} - Component: {component_id}, Host: {host_id}")
        return (component_id, host_id), action_index, explore


    def store_experience(self, state, action, reward, next_state, done, app_index, explore):
        self.temp_experiences.append((state, action, reward, next_state, done, app_index, explore))
        if done:
            self._flush_experiences()

    def _flush_experiences(self):
        rewards = [exp[2] for exp in self.temp_experiences]
        discounted_rewards = self.discount_rewards(rewards)
        for i, (state, action, reward, next_state, done, app_index, explore) in enumerate(self.temp_experiences):
            reward = discounted_rewards[i]
            if explore:
                self.replay_buffer_random.append((state, action, reward, next_state, done, app_index))
                if len(self.replay_buffer_random) > 10000:
                    self.replay_buffer_random.pop(0)
            else:
                self.replay_buffer_model.append((state, action, reward, next_state, done, app_index))
                if len(self.replay_buffer_model) > 10000:
                    self.replay_buffer_model.pop(0)
        self.temp_experiences = []

    def learn(self, batch_size=32):
        print(f"Replay buffer sizes - Model: {len(self.replay_buffer_model)}, Random: {len(self.replay_buffer_random)}")
        
        total_experiences = len(self.replay_buffer_model) + len(self.replay_buffer_random)
        if total_experiences < batch_size:
            print("Not enough samples to learn.")
            # Ensure epsilon decays even when not learning
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            print(f"Epsilon after checking buffer size: {self.epsilon}")
>>>>>>> 1678550bf925cc138a94f5f33500cc4e4e487090
            return

        experiences = self.sample_experiences(batch_size)
        total_grads = [tf.zeros_like(var) for var in self.model.trainable_weights]

        rewards = [exp[2] for exp in experiences]
<<<<<<< HEAD
        #print(f"Rewards before normalization: {rewards}")
=======
>>>>>>> 1678550bf925cc138a94f5f33500cc4e4e487090
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        if std_reward > 0:
            rewards = (rewards - mean_reward) / std_reward
<<<<<<< HEAD
        #print(f"Rewards after normalization: {rewards}")
=======
>>>>>>> 1678550bf925cc138a94f5f33500cc4e4e487090

        with tf.GradientTape() as tape:
            total_loss = 0

            for i, (state, action_index, reward, next_state, done, app_index) in enumerate(experiences):
                if action_index is None:
                    continue
                graph = generate_graph_data(state, app_index)
                logits = self.model(graph, training=True)
<<<<<<< HEAD

                # Print logits for each experience
                #print(f"Logits for experience {i}: {logits.numpy().flatten()}")

                action_index = int(action_index)
                action_prob = tf.nn.softmax(logits)[0, action_index]
                log_prob = tf.math.log(action_prob + 1e-8)

                loss = -log_prob * rewards[i]
                #print(f"Loss for experience {i}: {loss.numpy()}")  # Print individual loss
                total_loss += loss

            grads = tape.gradient(total_loss, self.model.trainable_weights)
            for i in range(len(total_grads)):
                total_grads[i] += grads[i]
            
            for i, grad in enumerate(grads):
                if grad is not None:
                    print(f"Gradient for variable {i}: {tf.reduce_sum(grad).numpy()}")
                else:
                    print(f"Gradient for variable {i} is None")

        total_grads, _ = tf.clip_by_global_norm(total_grads, 1.0)
        self.optimizer.apply_gradients(zip(total_grads, self.model.trainable_weights))

        if self.temperature > self.temperature_min:
            self.temperature *= self.temperature_decay

        print(f"Temperature after learning: {self.temperature}")

        return total_loss / batch_size

    def sample_experiences(self, batch_size):
        model_size = len(self.replay_buffer_model)
        random_size = len(self.replay_buffer_random)
        random_ratio = max(self.temperature, 0.01)
=======
                action_index = int(action_index)
                action_prob = tf.nn.softmax(logits)[0, action_index]
                log_prob = tf.math.log(action_prob + 1e-8)

                loss = -log_prob * rewards[i]
                total_loss += loss

            grads = tape.gradient(total_loss, self.model.trainable_weights)
            for i in range(len(total_grads)):
                total_grads[i] += grads[i]

        total_grads, _ = tf.clip_by_global_norm(total_grads, 1.0)
        self.optimizer.apply_gradients(zip(total_grads, self.model.trainable_weights))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(f"Epsilon after learning: {self.epsilon}")

        return total_loss / batch_size

    def sample_experiences(self, batch_size):
        model_size = len(self.replay_buffer_model)
        random_size = len(self.replay_buffer_random)
        random_ratio = max(self.epsilon, 0.1)
>>>>>>> 1678550bf925cc138a94f5f33500cc4e4e487090

        model_sample_size = int(batch_size * (1 - random_ratio))
        random_sample_size = batch_size - model_sample_size

        model_samples = random.sample(self.replay_buffer_model, min(model_sample_size, model_size))
        random_samples = random.sample(self.replay_buffer_random, min(random_sample_size, random_size))

        return model_samples + random_samples

    def discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0.0
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[i]
            discounted[i] = cumulative
        return discounted

    def save(self, filepath):
        self.model.save_weights(os.path.join(filepath, 'gnn_weights'))

    def load(self, filepath):
<<<<<<< HEAD
        self.model.load_weights(os.path.join(filepath, 'gnn_weights'))
=======
        self.model.load_weights(os.path.join(filepath, 'gnn_weights'))
>>>>>>> 1678550bf925cc138a94f5f33500cc4e4e487090
