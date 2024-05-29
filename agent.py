import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Agent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = 0.99  # Discount factor for future rewards
        self.model = self.build_model()

    def build_model(self):
        # Neural network for the policy gradient
        inputs = layers.Input(shape=(self.state_size,))
        layer1 = layers.Dense(24, activation='relu')(inputs)
        layer2 = layers.Dense(24, activation='relu')(layer1)
        outputs = layers.Dense(self.action_size, activation='softmax')(layer2)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                      loss='categorical_crossentropy')
        return model

    def choose_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        probabilities = self.model.predict(state)[0]
        action = np.random.choice(self.action_size, p=probabilities)
        return self.decode_action(action)  # Implement a method to convert actions to your specific format

    def decode_action(self, action_index):
        # Translate an action index back into the (component_id, host_id, path_ids) format
        # This part needs your specific logic based on how you've structured actions
        return (component_id, host_id, path_ids)


    def learn(self, states, actions, rewards):
        # Convert list to numpy array for batch processing
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Normalize the rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)

        # Compute gradient of the actions taken
        with tf.GradientTape() as tape:
            # Get model predictions
            predictions = self.model(states)
            # Create a mask: 1 for actions taken, 0 otherwise
            action_masks = tf.one_hot(actions, self.action_size)
            # Compute the cross entropy loss
            log_probs = tf.reduce_sum(action_masks * tf.math.log(predictions), axis=1)
            loss = -tf.reduce_sum(log_probs * rewards)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

