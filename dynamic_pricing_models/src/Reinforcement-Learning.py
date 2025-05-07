# src/reinforcement_learning.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List

# Suppress TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class PriceEnvironment:
    """Custom environment for pricing problem"""
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.states = self._prepare_states()
        self.actions = self._prepare_actions()
        self.current_step = 0
        self.max_steps = len(data) - 1
        
    def _prepare_states(self) -> np.ndarray:
        """Normalize state features"""
        features = self.data[['sales', 'price', 'rolling_mean_7', 'rolling_std_7']].values
        scaler = MinMaxScaler()
        return scaler.fit_transform(features)
    
    def _prepare_actions(self) -> np.ndarray:
        """Create discrete action space"""
        min_price = self.data['price'].min()
        max_price = self.data['price'].max()
        return np.linspace(min_price * 0.9, max_price * 1.1, 15)  # 15 price points
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        return self.states[0].reshape(1, -1)
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool]:
        """Execute one step in the environment"""
        self.current_step += 1
        next_state = self.states[self.current_step].reshape(1, -1)
        done = self.current_step == self.max_steps
        
        # Reward calculation (can be customized)
        price = self.actions[action_idx]
        demand = next_state[0][0]  # Normalized sales
        reward = demand * price  # Revenue as reward
        
        return next_state, reward, done

class DQNAgent:
    """Deep Q-Network Agent for dynamic pricing"""
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.update_target_model()
        
    def _build_model(self) -> Sequential:
        """Build the neural network model"""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def update_target_model(self):
        """Update target network weights"""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0][0] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])
        
        # Batch prediction for efficiency
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        for i, (_, action, reward, _, done) in enumerate(minibatch):
            if done:
                current_q[i][action] = reward
            else:
                current_q[i][action] = reward + self.gamma * np.amax(next_q[i])
        
        # Train on batch
        self.model.fit(states, current_q, batch_size=self.batch_size, 
                      epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path: str):
        """Save model weights"""
        self.model.save_weights(path)
        
    def load(self, path: str):
        """Load model weights"""
        self.model.load_weights(path)

def train_agent(env: PriceEnvironment, agent: DQNAgent, 
               episodes: int = 1000) -> List[float]:
    """Train DQN agent"""
    episode_rewards = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action_idx = agent.act(state)
            next_state, reward, done = env.step(action_idx)
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            
        # Update target network periodically
        if e % 10 == 0:
            agent.update_target_model()
            
        episode_rewards.append(total_reward)
        
        # Print progress
        if (e+1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {e+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return episode_rewards

def evaluate_agent(env: PriceEnvironment, agent: DQNAgent) -> Tuple[float, List[float]]:
    """Evaluate trained agent"""
    state = env.reset()
    total_reward = 0
    rewards = []
    done = False
    
    while not done:
        action_idx = agent.act(state)
        next_state, reward, done = env.step(action_idx)
        state = next_state
        total_reward += reward
        rewards.append(reward)
        
    return total_reward, rewards

def plot_results(rewards: List[float], path: str):
    """Plot and save training results"""
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    window_size = max(10, len(rewards) // 20)
    moving_avg = pd.Series(rewards).rolling(window_size).mean()
    plt.plot(moving_avg)
    plt.title(f'Moving Average (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    # Define file paths
    processed_data_path = 'data/processed/processed_data.csv'
    rl_model_path = 'models/rl_pricing_model.h5'
    results_path = 'results/rl_pricing_results.txt'
    figures_path = 'figures/rewards_plot.png'
    
    # Create directories
    os.makedirs(os.path.dirname(rl_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(os.path.dirname(figures_path), exist_ok=True)
    
    try:
        # Load and prepare data
        print("Loading and preparing data...")
        data = pd.read_csv(processed_data_path)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        # Create environment and agent
        env = PriceEnvironment(data)
        agent = DQNAgent(env.states.shape[1], len(env.actions))
        
        # Train agent
        print("Training DQN agent...")
        episode_rewards = train_agent(env, agent, episodes=1000)
        
        # Save trained model
        agent.save(rl_model_path)
        print(f"Model saved to {rl_model_path}")
        
        # Evaluate agent
        print("Evaluating agent...")
        total_reward, test_rewards = evaluate_agent(env, agent)
        print(f"Total Evaluation Reward: {total_reward:.2f}")
        
        # Save results
        with open(results_path, 'w') as f:
            f.write(f"Final Evaluation Reward: {total_reward:.2f}\n")
            f.write(f"Average Training Reward: {np.mean(episode_rewards):.2f}\n")
            f.write(f"Max Training Reward: {np.max(episode_rewards):.2f}\n")
        
        # Plot results
        plot_results(episode_rewards, figures_path)
        print(f"Results plot saved to {figures_path}")
        
        print("Reinforcement learning completed successfully!")
        
    except Exception as e:
        print(f"Error in reinforcement learning: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()