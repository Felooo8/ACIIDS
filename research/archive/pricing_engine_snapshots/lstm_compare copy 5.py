# main.py


import os, sys
sys.path.insert(0, os.path.abspath("...."))
from src.pricing_engine.traditional_models import black_scholes, heston_model_price, monte_carlo_option_price
import matplotlib
from datetime import datetime, timedelta
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
from stable_baselines3 import DQN,PPO
import matplotlib.pyplot as plt
import os
import warnings
import QuantLib as ql

warnings.filterwarnings('ignore')
from stable_baselines3.common.callbacks import EvalCallback
import shap
import seaborn as sns
# Additional imports for volatility models and sentiment analysis
from arch import arch_model
from transformers import pipeline

from scipy.stats import norm
# Constants
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.2  # 10% of data for future use
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 20
LEARNING_RATE = 0.001

PLOT_COLORS = {
    'niebieski': (0, 0.447, 0.741),
    'czerwony': (0.85, 0.325, 0.098),
    'zolty': (0.929, 0.694, 0.125),
    'fioletowy': (0.494, 0.184, 0.556),
    'zielony': (0.466, 0.674, 0.188),
    'jasnoniebieski': (0.301, 0.745, 0.933),
    'ciemnioczerwony': (0.635, 0.078, 0.184)
}


# Initialize scalers
feature_standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
target_standard_scaler = StandardScaler()


def load_and_preprocess_data(market_data_path, text_data_path=None):
    """
    Load data from CSV and preprocess it.
    """
    # Load market data
    data = pd.read_csv(market_data_path, parse_dates=['Date', 'expiration_date'])
    
    # Filter option_ids with at least 150 records and an average price over 100
    data = data.groupby('option_id').filter(lambda x: len(x) >= 50 and x['price'].mean() > 3)
    # data = data[data['option_id'] == 124140401]
    
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)  # For any remaining NaNs

    # Include 'call_put' as a feature
    data['is_call'] = data['call_put'].map({'C': 1, 'P': 0})

    # Calculate sentiment score
    data['sentiment_score'] = data['positive_articles'] - data['negative_articles']
    data['sentiment_score'].fillna(0, inplace=True)

    # Select only relevant columns
    selected_columns = [
        'Date', 'expiration_date', 'option_id', 'price', 'underlying_price', 'price_strike', 
        'iv', 'dte', 'is_call', 'sentiment_score'
    ]
    data = data[selected_columns]
    
    # Feature selection
    features = ['underlying_price', 'price_strike', 'is_call', 'iv', 'dte', 'sentiment_score']

    # Target variable
    target = 'price'

    # Ensure correct data types
    data[features] = data[features].astype(float)
    data[target] = data[target].astype(float)

    # Save unscaled data
    data_unscaled = data[['Date', 'option_id', 'underlying_price', 'price_strike', 'iv', 'dte', 'price', 'is_call', 'sentiment_score']].copy()

    return data, features, target, data_unscaled


def split_data(data, test_size=TEST_SIZE):
    """
    Split data into training and future datasets.
    """
    split_index = int(len(data) * (1 - test_size))
    training_data = data.iloc[:split_index]
    future_data = data.iloc[split_index:]

    return training_data, future_data

def create_samples(data, features, target):
    """
    Create samples without sequences.
    """
    # Sort data by date
    data = data.sort_values('Date').reset_index(drop=True)
    X = data[features].values
    y = data[target].values
    return X, y

def create_sequences(data, features, target=None, seq_length=30):
    X = []
    y = []
    
    # Ensure data is sorted by option_id and date
    data = data.sort_values(['option_id', 'Date']).reset_index(drop=True)
    
    # Group data by option_id
    for option_id, group in data.groupby('option_id'):
        # Ensure the group is long enough for sequence creation
        if len(group) >= seq_length:
            for i in range(len(group) - seq_length):
                X.append(group[features].iloc[i:i+seq_length].values)
                y.append(group[target].iloc[i + seq_length])  # Target is the value after the end of the sequence
    
    return np.array(X), np.array(y)

def build_model(num_features, seq_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(seq_length, num_features)))
    model.add(LSTM(128))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
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


def plot_learning_curves(history):
    plt.figure(figsize=(12,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

import gym
from gym import spaces
import pickle

class OptionTradingEnv(gym.Env):
    def __init__(self, data, model, features, seq_length, min_max_scaler, target_standard_scaler, mode='train'):
        super(OptionTradingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.model = model
        self.features = features
        self.seq_length = seq_length
        self.mode = mode  # 'train' or 'test'
        self.current_step = 0
        self.total_steps = len(self.data) - 1

        # Sort data by date and option_id
        self.data.sort_values(['Date', 'option_id'], inplace=True)
        self.option_ids = self.data['option_id'].unique()
        
        # Action space: 0 - Hold, 1 - Buy, 2 - Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.seq_length * len(self.features) + 1,), dtype=np.float32
        )
        
        # Trading parameters
        self.position = 0  # 1 for long, -1 for short, 0 for neutral
        self.cash = 100000  # Starting cash
        self.shares_held = 0
        self.net_worth = self.cash
        self.max_shares = 1000  # Maximum shares to trade
        self.trade_history = []  # Record of trades

        # Initialize history buffer
        self.history = []

        # Scalers
        self.min_max_scaler = min_max_scaler
        self.target_standard_scaler = target_standard_scaler

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.cash = 100000
        self.shares_held = 0
        self.net_worth = self.cash
        self.trade_history = []
        self.history = []

        if self.mode == 'train':
            # For training, shuffle data to randomize starting point
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        
        # Pre-fill the history with initial data
        for _ in range(self.seq_length):
            if self.current_step < len(self.data):
                self.history.append(self.data.iloc[self.current_step])
                self.current_step += 1
            else:
                break
        return self._next_observation()

    def _next_observation(self):
        # Ensure we have enough data
        if len(self.history) < self.seq_length:
            pad_size = self.seq_length - len(self.history)
            pad_data = [self.history[0]] * pad_size
            sequence_data = pad_data + self.history
        else:
            sequence_data = self.history[-self.seq_length:]
        
        # Extract features and scale
        features = [row[self.features].values for row in sequence_data]
        features_scaled = self.min_max_scaler.transform(features)
        features_scaled = np.array(features_scaled)
        
        # Prepare input for prediction
        features_scaled_input = features_scaled.reshape(1, self.seq_length, len(self.features))
        # Predict price
        predicted_price_scaled = self.model.predict(features_scaled_input)
        predicted_price = self.target_standard_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0, 0]
        
        # Flatten features and append predicted price
        obs = np.concatenate([features_scaled.flatten(), [predicted_price]], axis=0).astype(np.float32)
        return obs

    def step(self, action):
        # Execute action
        current_row = self.data.iloc[self.current_step]
        current_price = current_row['price']
        
        done = False
        reward = 0
        
        # Calculate profit/loss
        prev_net_worth = self.net_worth

        if action == 1:  # Buy
            if self.position <= 0:
                # Buy max shares
                can_buy = self.cash // current_price
                shares_bought = min(can_buy, self.max_shares)
                cost = shares_bought * current_price
                self.cash -= cost
                self.shares_held += shares_bought
                self.position = 1
                self.trade_history.append({'step': self.current_step, 'action': 'Buy', 'price': current_price, 'shares': shares_bought})
        elif action == 2:  # Sell
            if self.position >= 0 and self.shares_held > 0:
                # Sell all shares
                revenue = self.shares_held * current_price
                self.cash += revenue
                self.shares_held = 0
                self.position = -1
                self.trade_history.append({'step': self.current_step, 'action': 'Sell', 'price': current_price, 'shares': self.shares_held})
        else:  # Hold
            self.trade_history.append({'step': self.current_step, 'action': 'Hold', 'price': current_price, 'shares': self.shares_held})

        # Update net worth
        self.net_worth = self.cash + self.shares_held * current_price

        # Calculate reward (change in net worth)
        reward = self.net_worth - prev_net_worth

        self.current_step += 1

        if self.current_step < len(self.data):
            self.history.append(self.data.iloc[self.current_step])
        else:
            done = True  # End of data

        # Get next observation
        if not done:
            obs = self._next_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, {}


def calculate_cumulative_profit(trade_history, actual_prices):
    cumulative_profit = 0
    for trade in trade_history:
        step = trade['step']
        action = trade['action']
        price = trade['price']
        if action == 'Buy':
            cumulative_profit -= price
        elif action == 'Sell':
            cumulative_profit += price
    return cumulative_profit

def simulate_trading_strategy(predicted_prices, actual_prices, dates, initial_cash=100000):
    cash = initial_cash
    position = 0  # Number of shares held
    net_worths = []
    trade_history = []
    net_worths_dates = []

    max_index = min(len(predicted_prices), len(actual_prices)) - 1
    for i in range(max_index):
        predicted_next_price = predicted_prices[i + 1]
        actual_current_price = actual_prices[i]
        actual_next_price = actual_prices[i + 1]
        current_date = dates[i]

        # Decide to buy or sell one unit
        if predicted_next_price > actual_current_price:
            # Buy one unit
            if cash >= actual_current_price:
                cash -= actual_current_price
                position += 1
                trade_history.append({'date': current_date, 'step': i, 'action': 'Buy', 'price': actual_current_price, 'shares': 1})
        else:
            # Sell one unit if holding
            if position > 0:
                cash += actual_current_price
                position -= 1
                trade_history.append({'date': current_date, 'step': i, 'action': 'Sell', 'price': actual_current_price, 'shares': 1})

        net_worth = cash + position * actual_current_price
        net_worths.append(net_worth)
        net_worths_dates.append(current_date)

    return net_worths, net_worths_dates, trade_history


def simulate_trading_strategy_multiple_options(predicted_prices, actual_prices, option_ids, initial_cash=100000):
    results = []
    unique_option_ids = np.unique(option_ids)
    for oid in unique_option_ids:
        indices = np.where(option_ids == oid)[0]
        pred_prices = predicted_prices[indices]
        act_prices = actual_prices[indices]
        net_worths, trade_history = simulate_trading_strategy(pred_prices, act_prices, initial_cash)
        profit = net_worths[-1] - initial_cash
        results.append({'option_id': oid, 'profit': profit, 'net_worths': net_worths})
    return results


def create_sequences_with_future_target(training_data, future_data, features, target, seq_length=30):
    """
    Create sequences for future data by including the last SEQUENCE_LENGTH - 1 days from training data.
    """
    X = []
    y = []
    # Ensure data is sorted by option_id and date
    training_data = training_data.sort_values(['option_id', 'Date']).reset_index(drop=True)
    future_data = future_data.sort_values(['option_id', 'Date']).reset_index(drop=True)
    
    # Concatenate last SEQUENCE_LENGTH - 1 days from training_data to future_data for each option_id
    combined_data = []
    for option_id in future_data['option_id'].unique():
        training_group = training_data[training_data['option_id'] == option_id]
        future_group = future_data[future_data['option_id'] == option_id]
        if len(training_group) >= seq_length - 1:
            last_training = training_group.iloc[-(seq_length - 1):]
            combined_group = pd.concat([last_training, future_group], ignore_index=True)
        else:
            # If not enough data in training, use all available data
            combined_group = future_group
        # Create sequences from combined_group
        if len(combined_group) >= seq_length:
            for i in range(len(combined_group) - seq_length):
                X.append(combined_group[features].iloc[i:i+seq_length].values)
                y.append(combined_group[target].iloc[i + seq_length])  # Target is the value after the end of the sequence
    return np.array(X), np.array(y)

def print_metrics(model_name, y_true, y_pred):
    """
    Print RMSE and MAE metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f'{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}')


def plot_feature_importance_heatmap(model, X, feature_names):
    """
    Plots a heatmap of feature importance using SHAP values.
    
    Parameters:
    - model: The trained model (e.g., an LSTM model).
    - X: The input data for which SHAP values will be computed.
    - feature_names: List of feature names corresponding to the columns in X.
    """
    # Use SHAP DeepExplainer or KernelExplainer depending on the model type
    explainer = shap.DeepExplainer(model, X) if hasattr(model, "predict") else shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)
    
    # Compute the mean absolute SHAP values for each feature across all predictions
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Create a DataFrame for better visualization with seaborn
    importance_df = pd.DataFrame(feature_importance, index=feature_names, columns=["Importance"])
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(importance_df.T, cmap="YlGnBu", annot=True, cbar=True, square=True)
    plt.title("Feature importance heatmap showing the contribution of different inputs to the model’s predictions")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    # Save the plot to a file
    plt.savefig('feature_importance_heatmap.pdf')

    # Save the feature importance data for reuse
    importance_df.to_pickle('feature_importance.pkl')

def simulate_trading_strategy_multi_option(combined_data, model_name, initial_cash=100000):
    cash = initial_cash
    positions = {}  # Positions per option_id
    net_worths = []
    dates = []
    trade_history = []
    
    # We assume combined_data is sorted by date
    unique_dates = combined_data['date'].unique()
    
    for current_date in unique_dates:
        daily_data = combined_data[combined_data['date'] == current_date]
        # For each option available on this date
        for idx, row in daily_data.iterrows():
            option_id = row['option_id']
            actual_price = row['actual_price']
            if model_name == 'LSTM':
                predicted_price = row['predicted_price_nn']
            elif model_name == 'Black-Scholes':
                predicted_price = row['predicted_price_bs']
            elif model_name == 'Heston':
                predicted_price = row['predicted_price_heston']
            elif model_name == 'Monte-Carlo':
                predicted_price = row['predicted_price_mc']
            else:
                raise ValueError('Invalid model name')
            
            # Simple trading strategy: Buy if predicted_price > actual_price
            if predicted_price > actual_price:
                # Buy one unit
                if cash >= actual_price:
                    cash -= actual_price
                    positions[option_id] = positions.get(option_id, 0) + 1
                    trade_history.append({'date': current_date, 'option_id': option_id, 'action': 'Buy', 'price': actual_price, 'shares': 1})
            else:
                # Sell one unit if holding
                if positions.get(option_id, 0) > 0:
                    cash += actual_price
                    positions[option_id] -= 1
                    trade_history.append({'date': current_date, 'option_id': option_id, 'action': 'Sell', 'price': actual_price, 'shares': 1})
        
        # Calculate net worth
        net_worth = cash
        for oid, shares in positions.items():
            # Get the latest actual price for this option
            option_price = daily_data[daily_data['option_id'] == oid]['actual_price'].iloc[0]
            net_worth += shares * option_price
        net_worths.append(net_worth)
        dates.append(current_date)
    
    profit = net_worths[-1] - initial_cash
    return net_worths, dates, profit, trade_history

def main():
    market_data_path = 'data/final_merged_all_data.csv'
    # Load and preprocess data
    data, features, target, data_unscaled = load_and_preprocess_data(market_data_path)

    # Split data into training and future sets
    training_data, future_data = split_data(data)
    training_data_unscaled, future_data_unscaled = split_data(data_unscaled)

    # Step 1: Group by option_id and get max DTE for each option
    option_dte = training_data.groupby('option_id')['dte'].max().reset_index()
    # Sort options by DTE in descending order
    option_dte = option_dte.sort_values(by='dte', ascending=False)
    # Select the top 10 options with the longest DTE
    top_10_options = option_dte.head(10)['option_id'].tolist()

    # Filter data for the selected options
    data_selected = training_data[training_data['option_id'].isin(top_10_options)].reset_index(drop=True)
    data_selected_unscaled = data_unscaled[data_unscaled['option_id'].isin(top_10_options)].reset_index(drop=True)

    # Similarly, filter future data
    future_data_selected = future_data[future_data['option_id'].isin(top_10_options)].reset_index(drop=True)
    future_data_selected_unscaled = future_data_unscaled[future_data_unscaled['option_id'].isin(top_10_options)].reset_index(drop=True)

    # Fit scalers on training data
    training_data[features] = min_max_scaler.fit_transform(training_data[features])
    training_data[target] = target_standard_scaler.fit_transform(training_data[[target]])

    # Transform future data
    future_data[features] = min_max_scaler.transform(future_data[features])
    future_data[target] = target_standard_scaler.transform(future_data[[target]])

    # Transform data for the selected options
    data_selected[features] = min_max_scaler.transform(data_selected[features])
    data_selected[target] = target_standard_scaler.transform(data_selected[[target]])

    future_data_selected[features] = min_max_scaler.transform(future_data_selected[features])
    future_data_selected[target] = target_standard_scaler.transform(future_data_selected[[target]])

    # Build and train the model
    model_path = 'models/option_pricing_model_lstm_yes_sentiment.keras'
    if not os.path.exists(model_path):
        # Create sequences for training data
        X, y = create_sequences(data_selected, features, target, seq_length=SEQUENCE_LENGTH)
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        model = build_model(len(features), SEQUENCE_LENGTH)
        history = train_model(model, X_train, y_train, X_val, y_val)
        plot_learning_curves(history)
        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    # Initialize the RL environment with training data
    env = OptionTradingEnv(future_data_unscaled, model, features, SEQUENCE_LENGTH, min_max_scaler, target_standard_scaler, mode='train')

    agent_path = 'models/option_trading_agent_adv'
    if not os.path.exists(agent_path + '.zip'):
        policy_kwargs = dict(net_arch=[128, 32, 1])
        agent = DQN('MlpPolicy', env, learning_rate=LEARNING_RATE, buffer_size=20000, batch_size=BATCH_SIZE, gamma=0.95, verbose=0, policy_kwargs=policy_kwargs)
        eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                            log_path='./logs/', eval_freq=200,
                            deterministic=True, render=False)
        agent.learn(total_timesteps=2000, callback=eval_callback)
        agent.save(agent_path)
    else:
        agent = DQN.load(agent_path)


    # Initialize variables for trading simulation
    cumulative_net_worths = {
        'Black-Scholes': {},
        'Heston': {},
        'Monte-Carlo': {},
        'LSTM': {},
        'Hybrid Model (with RL)': {}
    }

    r = 0.04  # Risk-free interest rate
    kappa = 2.0        # Mean reversion rate
    theta = 0.01       # Long-term variance
    xi = 0.1           # Volatility of volatility
    rho = -0.5         # Correlation between the underlying asset and its volatility
    print("Starting trading simulation...")
    # Loop over each option
    for option_id in top_10_options:
        print(f"Processing Option ID: {option_id}")
        # Filter data for the specific option 
        data_option = data_selected[data_selected['option_id'] == option_id].reset_index(drop=True)
        data_option_unscaled = data_selected_unscaled[data_selected_unscaled['option_id'] == option_id].reset_index(drop=True)
        future_data_option = future_data_selected[future_data_selected['option_id'] == option_id].reset_index(drop=True)
        future_data_option_unscaled = future_data_selected_unscaled[future_data_selected_unscaled['option_id'] == option_id].reset_index(drop=True)

        # Create sequences for the specific option
        x_option, y_option = create_sequences(data_option, features, target, SEQUENCE_LENGTH)
        future_x_option, future_y_option = create_sequences_with_future_target(
            data_option, future_data_option, features, target, seq_length=SEQUENCE_LENGTH
        )

        if len(future_x_option) == 0:
            print(f"No future data available for option {option_id}, skipping.")
            continue

        # Predict on future data for the specific option
        future_pred = model.predict(future_x_option)
        future_pred_real = target_standard_scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        future_y_real = target_standard_scaler.inverse_transform(future_y_option.reshape(-1, 1)).flatten()

        # Extract unscaled features for the specific option
        S_option = future_data_option_unscaled['underlying_price'].values
        K_option = future_data_option_unscaled['price_strike'].values
        sigma_option = future_data_option_unscaled['iv'].values
        T_option = future_data_option_unscaled['dte'].values / 252  # Convert to years
        option_type_option = future_data_option_unscaled['is_call'].values
        actual_prices_option = future_data_option_unscaled['price'].values
        dates_option = pd.to_datetime(future_data_option_unscaled['Date'].values)

        # Calculate prices using different models
        bs_prices_option = np.array([black_scholes(S_option[i], K_option[i], T_option[i], r, sigma_option[i], option_type_option[i]) for i in range(len(S_option))])
        heston_prices_option = np.array([heston_model_price(S_option[i], K_option[i], T_option[i], r, sigma_option[i], kappa, theta, xi, rho, option_type_option[i]) for i in range(len(S_option))])
        mc_prices_option = np.array([monte_carlo_option_price(S_option[i], K_option[i], T_option[i], r, sigma_option[i], option_type=option_type_option[i]) for i in range(len(S_option))])

        # Simulate trading strategies for the specific option
        # Ensure that dates, predicted prices, and actual prices have the same length
        min_length = min(len(future_pred_real), len(future_y_real), len(dates_option), len(bs_prices_option), len(heston_prices_option), len(mc_prices_option))
        future_pred_real = future_pred_real[:min_length]
        future_y_real = future_y_real[:min_length]
        dates_option = dates_option[:min_length]
        bs_prices_option = bs_prices_option[:min_length]
        heston_prices_option = heston_prices_option[:min_length]
        mc_prices_option = mc_prices_option[:min_length]

        # For LSTM Model
        nn_net_worths, nn_dates, nn_trade_history = simulate_trading_strategy(
            future_pred_real, future_y_real, dates_option
        )
        # For Black-Scholes Model
        bs_net_worths, bs_dates, bs_trade_history = simulate_trading_strategy(
            bs_prices_option, future_y_real, dates_option
        )
        # For Heston Model
        heston_net_worths, heston_dates, heston_trade_history = simulate_trading_strategy(
            heston_prices_option, future_y_real, dates_option
        )
        # For Monte Carlo Model
        mc_net_worths, mc_dates, mc_trade_history = simulate_trading_strategy(
            mc_prices_option, future_y_real, dates_option
        )

        # For RL Agent
        # Initialize environment with unscaled data for the specific option, mode='test'
        env_test = OptionTradingEnv(future_data_option_unscaled, model, features, SEQUENCE_LENGTH, min_max_scaler, target_standard_scaler, mode='test')
        
        obs = env_test.reset()
        net_worths_rl = []
        dates_rl = []   
        print(env_test.total_steps)
        for _ in range(env_test.total_steps):
            action, _states = agent.predict(obs)
            obs, reward, done, info = env_test.step(action)
            net_worths_rl.append(env_test.net_worth)
            dates_rl.append(env_test.data.iloc[env_test.current_step - 1]['Date'])
            if done:
                break

        # Accumulate net worths over time for each model
        # Sum net worths day by day
        models = {
            'Black-Scholes': (bs_dates, bs_net_worths),
            'Heston': (heston_dates, heston_net_worths),
            'Monte-Carlo': (mc_dates, mc_net_worths),
            'LSTM': (nn_dates, nn_net_worths),
            'Hybrid Model (with RL)': (dates_rl, net_worths_rl)
        }

        for model_name, (dates_model, net_worths_model) in models.items():
            for date, net_worth in zip(dates_model, net_worths_model):
                date_str = date.strftime('%Y-%m-%d')
                if date_str not in cumulative_net_worths[model_name]:
                    cumulative_net_worths[model_name][date_str] = 0
                cumulative_net_worths[model_name][date_str] += net_worth


            # After processing all options, sort the dates and net worths
    for model_name in cumulative_net_worths:
        # Convert to list of tuples and sort by date
        net_worths_over_time = sorted(cumulative_net_worths[model_name].items())
        dates_model, net_worths_model = zip(*net_worths_over_time)
        cumulative_net_worths[model_name] = (dates_model, net_worths_model)

    # Plot cumulative net worth over time for each model
    plt.figure(figsize=(12, 6))
    for model_name, color in zip(cumulative_net_worths.keys(), PLOT_COLORS.values()):
        dates_model, net_worths_model = cumulative_net_worths[model_name]
        dates_model = pd.to_datetime(dates_model)
        plt.plot(dates_model, net_worths_model, label=model_name, color=color)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Net Worth')
    plt.title('Cumulative Net Worth Over Time for Different Models Trading Multiple Options')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cumulative_net_worth_over_time_multiple_options.png')
    plt.show()

    # Calculate final cumulative profits for each model
    cumulative_profits = {}
    initial_net_worth = 100000 * len(top_10_options)  # Initial cash per option
    for model_name in cumulative_net_worths:
        dates_model, net_worths_model = cumulative_net_worths[model_name]
        final_net_worth = net_worths_model[-1]
        profit = final_net_worth - initial_net_worth
        cumulative_profits[model_name] = profit

    # Print the table
    print("\nTrading performance comparison based on cumulative profit:")
    print("{:<30} {:<15}".format('Model', 'Cumulative Profit'))
    for model, profit in cumulative_profits.items():
        print("{:<30} {:<15.2f}".format(model, profit))




if __name__ == '__main__':
    main()