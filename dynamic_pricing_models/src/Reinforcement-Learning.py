# src/reinforcement_learning.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class PricingEnvironment:
    def __init__(self, data):
        self.data = data
        self.state_features = ['sales', 'price', 'rolling_mean_7', 'rolling_std_7']
        self.scaler = MinMaxScaler()
        self.states = self.scaler.fit_transform(data[self.state_features].values)
        self.actions = np.linspace(data['price'].min()*0.8, data['price'].max()*1.2, 10)
        self.current_step = 0
        self.max_steps = len(data) - 1

    def reset(self):
        self.current_step = 0
        return self.states[0].reshape(1, -1)

    def step(self, action_idx):
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done:
            next_state = self.states[self.current_step - 1].reshape(1, -1)
            return next_state, 0.0, done

        next_state = self.states[self.current_step].reshape(1, -1)

        price = self.actions[action_idx]
        cost = price * 0.8
        reward = next_state[0][0] * (price - cost)

        return next_state, reward, done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            tf.keras.Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0][0] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.amax(next_q_values[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

def main():
    # Paths
    processed_data_path = 'data/processed/processed_data.csv'
    model_path = 'models/rl_pricing_model.h5'
    results_path = 'results/rl_results.txt'
    figures_path = 'figures/training_progress.png'

    # Ensure directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(os.path.dirname(figures_path), exist_ok=True)

    try:
        print("1. Loading data...")
        data = pd.read_csv(processed_data_path)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        print("2. Creating environment...")
        env = PricingEnvironment(data)

        print("3. Initializing agent...")
        agent = DQNAgent(env.states.shape[1], len(env.actions))

        print("4. Starting training...")
        episodes = 100  # Adjust as needed
        rewards_history = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                agent.replay()

            rewards_history.append(total_reward)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                print(f"Episode: {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        print("5. Saving model...")
        agent.save(model_path)

        print("6. Saving results...")
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_history)
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(figures_path)
        plt.close()

        with open(results_path, 'w') as f:
            f.write(f"Final Reward: {rewards_history[-1]:.2f}\n")
            f.write(f"Average Reward: {np.mean(rewards_history):.2f}\n")

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error in training: {str(e)}")

if __name__ == "__main__":
    main()