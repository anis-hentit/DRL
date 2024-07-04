import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import mt_albis
import numpy as np

def flatten_state(state):
    """
    Flatten the state dictionary into a single numpy array.
    This array combines host resources, component requirements and deployment status,
    logical links, and infrastructure links to form a comprehensive state representation.
    """
    hosts = state['hosts']
    applications = state['applications']
    infra_links = state['infra_links']

    # Flatten host values: CPU and RAM for each host
    host_values = np.concatenate([list(host.values()) for host in hosts.values()])
    
    # Flatten component values: CPU and RAM for each component, ignoring deployment status
    component_values = np.concatenate([list(component.values())[:2] for app in applications for component in app['components'].values()])
    
    # Create an array for the deployment status of each component (1 if deployed, else 0)
    deployed_status = np.array([1.0 if component['deployed'] else 0.0 for app in applications for component in app['components'].values()], dtype=np.float32)
    
    # Flatten link values: logical link attributes (source, destination, latency, bandwidth)
    link_values = np.concatenate([list(link.values()) for app in applications for link in app['links'].values()])
    
    # Flatten infrastructure link values: physical link attributes
    infra_link_values = np.concatenate([list(link.values()) for link in infra_links.values()])

    # Combine all flattened parts into a single array
    flattened_state = np.concatenate([host_values, component_values, deployed_status, link_values, infra_link_values])
    return flattened_state

def generate_graph_data(state, app_index):
    """
    Generate a GraphTensor from the state dictionary for a specific application.
    The GraphTensor represents the physical and logical structure of the system
    including hosts and components as nodes, and links as edges.
    """
    hosts = state['hosts']
    applications = state['applications']
    infra_links = state['infra_links']
    app = applications[app_index]

    node_features = []
    node_indices = {}
    idx = 0

    # Add host nodes to the graph, each with its CPU and RAM as features
    for host_id, host in hosts.items():
        node_features.append([host['CPU'], host['RAM'], 0.0])
        node_indices[('host', host_id)] = idx
        idx += 1

    # Add component nodes to the graph, each with its CPU, RAM, and deployment status as features
    for comp_id, comp in app['components'].items():
        node_features.append([comp['CPU'], comp['RAM'], 1.0 if comp['deployed'] else 0.0])
        node_indices[('component', app_index, comp_id)] = idx
        idx += 1

    edge_list = []
    edge_features = []

    # Add physical links (infrastructure links) to the graph
    for link in infra_links.values():
        if link['source'] in hosts and link['destination'] in hosts:
            latency = link.get('latency', 0)
            bandwidth = link.get('bandwidth', 0)
            edge_list.append([node_indices[('host', link['source'])], node_indices[('host', link['destination'])]])
            edge_features.append([latency, bandwidth])
            edge_list.append([node_indices[('host', link['destination'])], node_indices[('host', link['source'])]])
            edge_features.append([latency, bandwidth])
        else:
            print(f"Skipping invalid link with source {link['source']} and destination {link['destination']}")

    # Add logical links (application links) to the graph
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

    # Convert node features to a NumPy array
    node_features = np.array(node_features, dtype=np.float32)

    # Convert edge list and edge features to NumPy arrays and transpose edge list for adjacency
    edge_list = np.array(edge_list, dtype=np.int32).T
    edge_features = np.array(edge_features, dtype=np.float32)

    # Create a GraphTensor from the node and edge data
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

def compute_action_mask(state, app_index, max_components):
    """
    Compute a mask for valid actions based on the current state for a specific application.
    Each action corresponds to deploying a component to a host,
    and the mask indicates if an action is valid (1) or invalid (0).
    """
    num_hosts = len(state['hosts'])
    mask = np.zeros((max_components, num_hosts))

    app = state['applications'][app_index]
    for comp_id, comp in app['components'].items():
        if not comp['deployed']:
            for host_id, host in state['hosts'].items():
                if comp['CPU'] <= host['CPU'] and comp['RAM'] <= host['RAM']:
                    mask[comp_id, host_id] = 1

    # Flatten the mask and pad to the maximum size (max_components x num_hosts)
    mask = mask.flatten()
    if len(mask) < max_components * num_hosts:
        mask = np.pad(mask, (0, max_components * num_hosts - len(mask)), 'constant', constant_values=0)

    return mask

class GNNAgent(tf.keras.Model):
    """
    A Graph Neural Network (GNN) based agent model for learning deployment strategies.
    """
    def __init__(self, hidden_dim, output_dim):
        super(GNNAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layer to project node features to the hidden dimension
        self.projection_layer = tf.keras.layers.Dense(hidden_dim)

        # GNN update mechanism using the MT-Albis architecture
        self.graph_update = mt_albis.MtAlbisGraphUpdate(
            units=hidden_dim,                 # Number of units in each layer
            message_dim=hidden_dim,           # Dimension of the message passing
            attention_type="none",            # No attention mechanism
            simple_conv_reduce_type="mean",   # Mean reduction for convolution
            normalization_type="layer",       # Layer normalization
            next_state_type="residual",       # Residual connections
            state_dropout_rate=0.2,           # Dropout rate
            l2_regularization=1e-5,           # L2 regularization
            receiver_tag=tfgnn.TARGET         # Tag indicating message receiver
        )
        # Final dense layer to output action probabilities
        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, graph):
        """
        Forward pass through the GNN.
        Project node features, apply graph update, and compute action probabilities.
        """
        # Project node features to hidden dimension
        node_features = graph.node_sets['nodes']['hidden_state']
        projected_features = self.projection_layer(node_features)
        graph = graph.replace_features(node_sets={'nodes': {'hidden_state': projected_features}})

        # Apply the graph update to integrate node and edge information
        graph = self.graph_update(graph)
        updated_features = graph.node_sets['nodes']['hidden_state']
        
        # Aggregate node features by averaging
        aggregated_features = tf.reduce_mean(updated_features, axis=0)
        
        # Expand dimensions to match the expected input shape for the dense layer
        aggregated_features = tf.expand_dims(aggregated_features, axis=0)

        # Return the probabilities for each action
        return self.dense(aggregated_features)

class Agent:
    """
    Agent class for managing the training and action selection process using GNN.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, max_components, num_hosts, learning_rate=0.001):
        self.model = GNNAgent(hidden_dim, output_dim)  # Initialize the GNN model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)  # Adam optimizer for training
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.986  # Decay rate for epsilon
        self.epsilon_min = 0.02  # Minimum value for epsilon
        self.max_components = max_components  # Maximum number of components
        self.num_hosts = num_hosts  # Number of hosts

    def choose_action(self, state, app_index):
        """
        Choose an action based on the current state for a specific application.
        This function uses an epsilon-greedy strategy for exploration and exploitation.
        """
        graph = generate_graph_data(state, app_index)  # Generate the graph data from the state
        logits = self.model(graph).numpy().flatten()  # Get action probabilities from the GNN

        # Compute the action mask for valid actions
        mask = compute_action_mask(state, app_index, self.max_components)
        print(f"Action Mask for app {app_index}: {mask}")  # Debug log for action mask

        # Apply the mask to the logits, invalid actions are set to -inf
        masked_logits = np.where(mask, logits, -np.inf)
        action_probs = np.exp(masked_logits) / np.sum(np.exp(masked_logits))  # Compute probabilities using softmax

        valid_indices = np.where(mask)[0]  # Indices of valid actions

        if valid_indices.size == 0:
            print(f"No valid actions available for app {app_index}. Ending episode.")
            return None

        if np.random.rand() <= self.epsilon:
            # Explore: choose a random valid action
            action_index = np.random.choice(valid_indices)
            print(f"Exploring for app {app_index}: Chose random action index {action_index} with epsilon {self.epsilon}")
        else:
            # Exploit: choose the best action based on the model's prediction
            action_index = np.argmax(action_probs)
            print(f"Exploiting for app {app_index}: Chose best action index {action_index} with epsilon {self.epsilon}")

        component_id, host_id = divmod(action_index, self.num_hosts)

        # Validate the selected component and host
        if component_id >= self.max_components or host_id >= self.num_hosts or mask[action_index] == 0:
            print(f"Invalid action for app {app_index}: Trying to deploy component {component_id} to host {host_id} with insufficient resources.")
            return None

        print(f"Chosen action for app {app_index} - Component: {component_id}, Host: {host_id}")
        return component_id, host_id

    def learn(self, states, actions, rewards, app_indices):
        """
        Update the model based on the experiences (states, actions, rewards, app indices).
        Uses policy gradient method to update the policy based on the cumulative rewards.
        """
        discounted_rewards = self.discount_rewards(rewards)
        total_loss = 0

        for state, action, reward, app_index in zip(states, actions, discounted_rewards, app_indices):
            if action is None:
                continue
            with tf.GradientTape() as tape:
                graph = generate_graph_data(state, app_index)  # Generate the graph from the state
                logits = self.model(graph)  # Get action probabilities from the model
                action_index = action[0] * self.num_hosts + action[1]  # Encode the action into a single index
                if action_index >= 0:  # Only update if the action is valid
                    action_prob = tf.nn.softmax(logits)[0, action_index]  # Get the probability of the taken action
                    log_prob = tf.math.log(action_prob + 1e-8)  # Add small value to prevent log(0)
                    
                    # Policy gradient loss (negative log probability scaled by reward)
                    loss = -log_prob * reward  
                    grads = tape.gradient(loss, self.model.trainable_weights)  # Compute gradients
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))  # Apply gradients
                    total_loss += loss

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay epsilon for exploration-exploitation trade-off

        # Return the average loss
        return total_loss / len(states) if len(states) > 0 else 0

    def discount_rewards(self, rewards):
        """
        Compute discounted rewards.
        Apply discount factor to rewards to emphasize immediate rewards over distant future rewards.
        """
        discounted = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0.0
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[i]
            discounted[i] = cumulative
        return discounted
