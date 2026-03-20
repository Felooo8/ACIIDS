# main.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import gym
from gym import spaces
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Additional imports for volatility models and sentiment analysis
from arch import arch_model
from transformers import pipeline

# Constants
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.1  # 10% of data for future use
BATCH_SIZE = 64
EPOCHS = 10
PATIENCE = 10
LEARNING_RATE = 0.001

# Function to calculate GARCH volatility
def calculate_garch_volatility(prices):
    am = arch_model(prices, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    forecasts = res.forecast(horizon=1)
    garch_vol = np.sqrt(forecasts.variance.values[-1, :])[0]
    return garch_vol

# Simplified Heston volatility (using implied volatility as a proxy)
def calculate_heston_volatility(iv):
    return iv  # Placeholder for actual Heston model implementation

# Sentiment analysis using a pre-trained transformer model
# sentiment_pipeline = pipeline('sentiment-analysis')

# def get_sentiment_score(text):
#     try:
#         result = sentiment_pipeline(text)[0]
#         score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
#     except:
#         score = 0  # Default to neutral sentiment if analysis fails
#     return score

def load_and_preprocess_data(market_data_path, text_data_path=None):
    """
    Load data from CSV and preprocess it.
    """
    # Load market data
    data = pd.read_csv(market_data_path, parse_dates=['Date', 'expiration_date'])

    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)  # For any remaining NaNs

    # Include 'call_put' as a feature
    data['is_call'] = data['call_put'].map({'C': 1, 'P': 0})

    # Calculate GARCH volatility
    data['garch_volatility'] = data.groupby('Date')['price'].transform(calculate_garch_volatility)

    # Calculate Heston volatility (simplified)
    data['heston_volatility'] = calculate_heston_volatility(data['iv'])

    # Load and process sentiment data
    # text_data = pd.read_csv(text_data_path)  # Assume it has 'Date' and 'text' columns
    # text_data['sentiment_score'] = text_data['text'].apply(get_sentiment_score)
    
    data['sentiment_score'] = data['positive_articles'] - data['negative_articles']
    # data = data.merge(text_data[['Date', 'sentiment_score']], on='Date', how='left')
    data['sentiment_score'].fillna(0, inplace=True)

    # Feature selection
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'delta', 'gamma', 'vega', 'theta', 'rho',
        'iv', 'dte',
        'RSI', 'MACD',
        'garch_volatility', 'heston_volatility', 'sentiment_score',
        'is_call'
    ]

    # Target variable
    target = 'price'

    # Ensure correct data types
    data[features] = data[features].astype(float)
    data[target] = data[target].astype(float)

    # Initialize scalers
    standard_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()

    # Standard scaling for continuous features
    standard_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'iv', 'dte',
                         'garch_volatility', 'heston_volatility', 'sentiment_score']
    data[standard_features] = standard_scaler.fit_transform(data[standard_features])

    # Min-Max scaling for bounded features
    min_max_features = [
        'delta', 'gamma', 'vega', 'theta', 'rho',
        'RSI', 'MACD', 'is_call'
    ]
    data[min_max_features] = min_max_scaler.fit_transform(data[min_max_features])

    return data, features, target

def split_data(data, test_size=TEST_SIZE):
    """
    Split data into training and future datasets.
    """
    split_index = int(len(data) * (1 - test_size))
    training_data = data.iloc[:split_index]
    future_data = data.iloc[split_index:]

    return training_data, future_data

def create_sequences(data, features, target, seq_length=SEQUENCE_LENGTH):
    """
    Create sequences of data for time series prediction.
    """
    X = []
    y = []

    # Sort data by date
    data = data.sort_values('Date').reset_index(drop=True)

    # Extract features and target
    feature_data = data[features].values
    target_data = data[target].values

    # Create sequences
    for i in range(len(data) - seq_length):
        X.append(feature_data[i:i+seq_length])
        y.append(target_data[i+seq_length])

    return np.array(X), np.array(y)

def build_lstm_model(seq_length, num_features):
    """
    Build an LSTM model.
    """
    inputs = Input(shape=(seq_length, num_features))
    x = LSTM(256)(inputs)
    x = Dropout(0.3)(x)
    # x = GRU(64)(x)
    # x = Dropout(0.3)(x)
    # x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the model with early stopping.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mean_squared_error')

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

class OptionPricingEnv(gym.Env):
    def __init__(self, data, model, features, target, max_steps=200):
        super(OptionPricingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.model = model
        self.features = features
        self.target = target
        self.sequence_length = SEQUENCE_LENGTH
        self.current_step = 0
        self.total_steps = len(self.data) - self.sequence_length - 1
        self.max_steps = max_steps  # Set maximum steps for each episode

        # Define action space: 0 - Buy, 1 - Sell, 2 - Hold
        self.action_space = spaces.Discrete(3)

        # Observation space: All features plus predicted price
        num_state_features = len(features) + 1  # +1 for predicted price
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_state_features,), dtype=np.float64
        )

        # Initialize sequence buffer
        self.sequence_buffer = self._initialize_sequence_buffer()

        # Position management
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.entry_price = 0
        self.trade_history = []

    def _initialize_sequence_buffer(self):
        seq_data = self.data.loc[self.current_step:self.current_step + self.sequence_length - 1, self.features].values
        return seq_data.tolist()

    def reset(self):
        self.current_step = 0
        self.sequence_buffer = self._initialize_sequence_buffer()
        self.position = 0
        self.entry_price = 0
        self.trade_history = []
        return np.array(self._get_observation(), dtype=np.float64)

    def _get_observation(self):
        # Current feature set
        current_features = self.sequence_buffer[-1]

        # Predicted price
        seq = np.array(self.sequence_buffer, dtype=np.float64)
        seq = seq[np.newaxis, :]
        predicted_price = self.model.predict(seq)[0, 0]

        # Combine features and predicted price
        obs = np.concatenate([current_features, [predicted_price]])
        return obs

    def step(self, action):
        seq = np.array(self.sequence_buffer, dtype=np.float64)
        seq = seq[np.newaxis, :]
        predicted_price = self.model.predict(seq)[0, 0]

        current_price = self.data.loc[self.current_step + self.sequence_length - 1, self.target]
        next_price = self.data.loc[self.current_step + self.sequence_length, self.target]

        reward = 0
        done = False

        # Apply action
        if action == 0:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            else:
                reward = -1  # Penalty for invalid action
        elif action == 1:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            else:
                reward = -1  # Penalty for invalid action
        elif action == 2:  # Hold
            pass  # No action

        # Update reward based on position and next day's price movement
        profit = 0
        if self.position == 1:
            profit = next_price - self.entry_price
            self.position = 0
            self.entry_price = 0
        elif self.position == -1:
            profit = self.entry_price - next_price
            self.position = 0
            self.entry_price = 0

        # Include prediction error in reward
        prediction_error = abs(predicted_price - next_price)
        alpha = 0.1  # Weight for prediction error
        reward += profit - alpha * prediction_error

        # Record trade
        if profit != 0:
            self.trade_history.append({
                'step': self.current_step,
                'action': 'Buy' if action == 0 else 'Sell',
                'entry_price': self.entry_price,
                'exit_price': next_price,
                'profit': profit
            })

        # Move to next step
        self.current_step += 1

        # Check if done
        if self.current_step >= self.total_steps:
            done = True

        # Check if done
        if self.current_step >= self.total_steps or self.current_step >= self.max_steps:
            done = True

        # Update the sequence buffer
        if not done:
            next_data_point = self.data.loc[self.current_step + self.sequence_length - 1, self.features].values
            self.sequence_buffer.append(next_data_point)
            self.sequence_buffer.pop(0)
            obs = self._get_observation()
        else:
            obs = np.zeros(len(self.features) + 1, dtype=np.float64)

        info = {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'next_price': next_price,
            'action_taken': action,
            'profit': profit
        }

        return np.array(obs, dtype=np.float64), reward, done, info

def train_rl_agent(env, total_timesteps=1000):
    """
    Train the reinforcement learning agent.
    """
    agent_path = "option_pricing_agent_new"
    if not os.path.exists(agent_path + ".zip"):
        agent = DQN('MlpPolicy', env, verbose=1)
        agent.learn(total_timesteps=total_timesteps)
        agent.save(agent_path)
    else:
        agent = DQN.load(agent_path, env, verbose=1)
    return agent

def evaluate_rl_agent(env, agent, episodes=10):
    """
    Evaluate the RL agent over a number of episodes.
    """
    total_rewards = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            print(f"Step {step}: Action {action}, Reward {reward:.2f}, "
                  f"Predicted Price {info['predicted_price']:.2f}, "
                  f"Current Price {info['current_price']:.2f}, "
                  f"Next Price {info['next_price']:.2f}")
            step += 1
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Total Reward = {episode_reward}")
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward}")

def visualize_agent_performance(env, target, features):
    """
    Visualize the agent's performance over time.
    """
    trade_history = pd.DataFrame(env.trade_history)
    if trade_history.empty:
        print("No trades were made by the agent.")
        return

    # Cumulative profit over time
    trade_history['cumulative_profit'] = trade_history['profit'].cumsum()

    # Plot cumulative profit
    plt.figure(figsize=(12, 6))
    plt.plot(trade_history['step'], trade_history['cumulative_profit'], label='Cumulative Profit')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Profit')
    plt.title('Agent Cumulative Profit Over Time')
    plt.legend()
    plt.show()

    # Actions taken over time
    plt.figure(figsize=(12, 6))
    plt.scatter(trade_history['step'], trade_history['action'].map({'Buy': 1, 'Sell': -1}), marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Action (1=Buy, -1=Sell)')
    plt.title('Actions Taken Over Time')
    plt.show()

    # Predicted vs Actual Prices
    plt.figure(figsize=(12, 6))
    plt.plot(env.data['Date'][env.sequence_length:], env.data[target][env.sequence_length:], label='Actual Prices')
    predicted_prices = []
    for i in range(env.sequence_length, len(env.data)):
        seq = env.data[features].values[i - env.sequence_length:i]
        seq = seq[np.newaxis, :]
        predicted_price = env.model.predict(seq)[0, 0]
        predicted_prices.append(predicted_price)
    plt.plot(env.data['Date'][env.sequence_length:], predicted_prices, label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Predicted vs Actual Prices')
    plt.legend()
    plt.show()

def main():
    # Load and preprocess data
    market_data_path = 'data/final_merged_all_data.csv'
    # text_data_path = 'data/text_data.csv'
    data, features, target = load_and_preprocess_data(market_data_path)

    # Split data into training and future datasets
    training_data, future_data = split_data(data)

    # Create sequences for training data
    X, y = create_sequences(training_data, features, target)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, shuffle=False
    )

    # Build and train the model
    model_path = 'models/option_pricing_model_new_2.keras'
    if not os.path.exists(model_path):
        model = build_lstm_model(SEQUENCE_LENGTH, len(features))
        history = train_model(model, X_train, y_train, X_val, y_val)
        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    # Evaluate the model on validation data
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation RMSE: {rmse}, MAE: {mae}")

    # Plot predictions vs actual prices
    plt.figure(figsize=(12,6))
    plt.plot(y_val, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.legend()
    plt.title('Model Predictions vs Actual Prices on Validation Set')
    plt.show()

    # Use future data for RL agent
    if not future_data.empty:
        # Create environment
        env = OptionPricingEnv(future_data, model, features, target)
        # Train RL agent
        agent = train_rl_agent(env)
        # Evaluate RL agent
        evaluate_rl_agent(env, agent)
        # Visualize agent performance
        visualize_agent_performance(env, target, features)
    else:
        print("No future data available for RL agent training.")

if __name__ == '__main__':
    main()
