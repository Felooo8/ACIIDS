# main.py


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
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import os
import warnings
import QuantLib as ql
warnings.filterwarnings('ignore')
from stable_baselines3.common.callbacks import EvalCallback

# Additional imports for volatility models and sentiment analysis
from arch import arch_model
from transformers import pipeline

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

def heston_model_price(S, K, T, r, sigma, kappa, theta, xi, rho, option_type=1):
    """
    Calculate option price using the Heston model.
    """
    # Set evaluation date
    evaluation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = evaluation_date

    # Option parameters   
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == 1 else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(evaluation_date + int(T * 365))

    # Market data
    risk_free_rate = ql.FlatForward(evaluation_date, r, ql.Actual365Fixed())
    dividend_rate = ql.FlatForward(evaluation_date, 0.0, ql.Actual365Fixed())
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))

    # Heston process
    v0 = sigma**2  # Initial volatility
    heston_process = ql.HestonProcess(
        ql.YieldTermStructureHandle(risk_free_rate),
        ql.YieldTermStructureHandle(dividend_rate),
        spot_handle,
        v0,
        kappa,
        theta,
        xi,
        rho
    )

    # Heston model
    model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(model)

    # Option
    european_option = ql.VanillaOption(payoff, exercise)
    european_option.setPricingEngine(engine)

    # Price
    price = european_option.NPV()
    return price

def binomial_option_price(S, K, T, r, sigma, N=100, option_type=1):
    """
    Calculate option price using the Binomial Option Pricing model.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))      # Up factor
    d = 1 / u                            # Down factor
    p = (np.exp(r * dt) - d) / (u - d)   # Risk-neutral probability

    # Initialize asset prices at maturity
    asset_prices = np.zeros(N+1)
    asset_prices[0] = S * d**N
    for i in range(1, N+1):
        asset_prices[i] = asset_prices[i-1] * u / d

    # Initialize option values at maturity
    option_values = np.zeros(N+1)
    if option_type == 1:
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)

    # Backward induction
    for i in range(N-1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])

    return option_values[0]

def monte_carlo_option_price(S, K, T, r, sigma, option_type=1, num_simulations=10000):
    """
    Calculate option price using Monte Carlo simulation.
    """
    # Simulate end-of-period asset price
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type == 1:
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type=1):
    """
    Calculate Black-Scholes option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 1:
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

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
    def __init__(self, data, model, features, seq_length, min_max_scaler, target_standard_scaler):
        super(OptionTradingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.model = model
        self.features = features
        self.seq_length = seq_length
        self.current_step = 0
        self.total_steps = len(self.data) - 1
        
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
            if self.position >= 0:
                # Sell all shares
                revenue = self.shares_held * current_price
                self.cash += revenue
                self.shares_held = 0
                self.position = -1
                self.trade_history.append({'step': self.current_step, 'action': 'Sell', 'price': current_price, 'shares': self.shares_held})
        else:  # Hold
            pass

        # Update net worth
        self.net_worth = self.cash + self.shares_held * current_price


        # Calculate reward (change in net worth)
        reward = self.net_worth - prev_net_worth

        self.current_step += 1

        if self.current_step < len(self.data):
            self.history.append(self.data.iloc[self.current_step])
        self.current_step += 1

        if self.current_step >= self.total_steps:
            done = True

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
    position = 0  # Number of shares held (can go negative for short positions)
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
            # Buy one unit (cover if currently short)
            if position < 0:  # If currently short, cover a short
                cash -= actual_current_price
                position += 1
                trade_history.append({'date': current_date, 'step': i, 'action': 'Cover', 'price': actual_current_price, 'shares': 1})
            else:  # Regular buy
                if cash >= actual_current_price:
                    cash -= actual_current_price
                    position += 1
                    trade_history.append({'date': current_date, 'step': i, 'action': 'Buy', 'price': actual_current_price, 'shares': 1})
        else:
            # Sell one unit (go short if currently no position or long)
            if position > 0:  # If currently holding, sell a long position
                cash += actual_current_price
                position -= 1
                trade_history.append({'date': current_date, 'step': i, 'action': 'Sell', 'price': actual_current_price, 'shares': 1})
            else:  # Short sell
                cash += actual_current_price
                position -= 1
                trade_history.append({'date': current_date, 'step': i, 'action': 'Short', 'price': actual_current_price, 'shares': 1})

        # Calculate net worth, considering short positions
        net_worth = cash + position * actual_current_price
        net_worths.append(net_worth)
        net_worths_dates.append(current_date)

    return net_worths, net_worths_dates, trade_history



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


def main():
    market_data_path = 'data/final_merged_all_data.csv'
    # Load and preprocess data
    data, features, target, data_unscaled = load_and_preprocess_data(market_data_path)

    # Split data into training and future sets
    training_data, future_data = split_data(data)
    training_data_unscaled, future_data_unscaled = split_data(data_unscaled)


    # Step 1: Group by option_id and get max DTE for each option
    option_dte = training_data.groupby('option_id')['dte'].max().reset_index()
    highest_dte = option_dte['dte'].max()
    options_with_highest_dte = option_dte[option_dte['dte'] == highest_dte]

    # Step 2: Filter the options data to only include options with the highest DTE
    data_highest_dte = training_data[training_data['option_id'].isin(options_with_highest_dte['option_id'])]

    # Step 3: Calculate the average price for each option_id and sort by it
    data_highest_dte['average_price'] = data_highest_dte.groupby('option_id')['price'].transform('mean')
    median_price = data_highest_dte['average_price'].median()

    # Select the option with the average price closest to the median
    selected_option = data_highest_dte.loc[(data_highest_dte['average_price'] - median_price).abs().idxmin()]
    selected_option_id = selected_option['option_id']

    # Filter data for the selected option
    data_selected = training_data[training_data['option_id'] == selected_option_id].reset_index(drop=True)
    data_selected_unscaled = data_unscaled[data_unscaled['option_id'] == selected_option_id].reset_index(drop=True)


    # Implement RL for a specific option ID
    specific_option_id = 124140401  # Replace with desired option ID
    future_data_specific = future_data[future_data['option_id'] == specific_option_id].reset_index(drop=True)
    future_data_specific_unscaled = future_data_unscaled[future_data_unscaled['option_id'] == specific_option_id].reset_index(drop=True)

    # Fit scalers on training data
    training_data[features] = min_max_scaler.fit_transform(training_data[features])
    training_data[target] = target_standard_scaler.fit_transform(training_data[[target]])

    # Transform future data
    future_data[features] = min_max_scaler.transform(future_data[features])
    future_data[target] = target_standard_scaler.transform(future_data[[target]])

    future_data_specific[features] = min_max_scaler.transform(future_data_specific[features])
    future_data_specific[target] = target_standard_scaler.transform(future_data_specific[[target]])

    # Transform data for the selected option
    data_selected[features] = min_max_scaler.transform(data_selected[features])
    data_selected[target] = target_standard_scaler.transform(data_selected[[target]])
    
    x_long, y_long = create_sequences(data_selected, features, target, SEQUENCE_LENGTH)

    # Create sequences for training data
    sequences_file = 'sequences_yes_sentiment.pkl'
    if os.path.exists(sequences_file):
        with open(sequences_file, 'rb') as f:
            X, y = pickle.load(f)
    else:
        # Filter out groups with fewer than SEQUENCE_LENGTH records
        training_data_filtered = training_data.groupby('option_id').filter(lambda x: len(x) >= SEQUENCE_LENGTH)
        X, y = create_sequences(training_data_filtered, features, target, seq_length=SEQUENCE_LENGTH)
        with open(sequences_file, 'wb') as f:
            pickle.dump((X, y), f)

    # Create sequences for future data including last SEQUENCE_LENGTH - 1 days from training data
    future_data_x, future_data_y = create_sequences_with_future_target(
        training_data, future_data, features, target, seq_length=SEQUENCE_LENGTH
    )

    future_data_specific_x, future_data_specific_y = create_sequences_with_future_target(
        training_data, future_data_specific, features, target, seq_length=SEQUENCE_LENGTH
    )

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Build and train the model
    model_path = 'models/option_pricing_model_lstm_yes_sentiment_2.keras'
    if not os.path.exists(model_path):
        model = build_model(len(features), SEQUENCE_LENGTH)
        history = train_model(model, X_train, y_train, X_val, y_val)
        plot_learning_curves(history)
        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    
    # Evaluate the model on validation data
    y_pred = model.predict(X_val)
    y_val_real = target_standard_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_real = target_standard_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    print_metrics("Neural Network Model (Validation)", y_val_real, y_pred_real)
    
    # Predict on future data for the specific option
    future_specific_pred = model.predict(future_data_specific_x)
    future_specific_pred_real = target_standard_scaler.inverse_transform(future_specific_pred.reshape(-1, 1)).flatten()
    future_data_specific_y_real = target_standard_scaler.inverse_transform(future_data_specific_y.reshape(-1, 1)).flatten()
    
    # Extract unscaled features for the specific option
    S_specific = future_data_specific_unscaled['underlying_price'].values
    K_specific = future_data_specific_unscaled['price_strike'].values
    sigma_specific = future_data_specific_unscaled['iv'].values
    T_specific = future_data_specific_unscaled['dte'].values / 252  # Convert to years
    option_type_specific = future_data_specific_unscaled['is_call'].values
    actual_prices_specific = future_data_specific_unscaled['price'].values

    r = 0.04  # Risk-free interest rate
    kappa = 2.0        # Mean reversion rate
    theta = 0.01       # Long-term variance
    xi = 0.1           # Volatility of volatility
    rho = -0.5         # Correlation between the underlying asset and its volatility

    # Calculate prices using different models
    bs_prices_specific = np.array([black_scholes(S_specific[i], K_specific[i], T_specific[i], r, sigma_specific[i], option_type_specific[i]) for i in range(len(S_specific))])
    heston_prices_specific = np.array([heston_model_price(S_specific[i], K_specific[i], T_specific[i], r, sigma_specific[i], kappa, theta, xi, rho, option_type_specific[i]) for i in range(len(S_specific))])
    mc_prices_specific = np.array([monte_carlo_option_price(S_specific[i], K_specific[i], T_specific[i], r, sigma_specific[i], option_type=option_type_specific[i]) for i in range(len(S_specific))])
    bop_prices_specific = np.array([binomial_option_price(S_specific[i], K_specific[i], T_specific[i], r, sigma_specific[i], option_type=option_type_specific[i]) for i in range(len(S_specific))])

    # Print RMSE and MAE for all models
    print_metrics("Neural Network Model", future_data_specific_y_real, future_specific_pred_real)
    print_metrics("Black-Scholes Model", future_data_specific_y_real, bs_prices_specific[1:])
    print_metrics("Heston Model", future_data_specific_y_real, heston_prices_specific[1:])
    print_metrics("Monte Carlo Model", future_data_specific_y_real, mc_prices_specific[1:])
    print_metrics("Binomial Option Pricing Model", future_data_specific_y_real, bop_prices_specific[1:])
    # Simulate trading strategies for all models
    # For LSTM Model# Extract dates for the specific option
    dates_specific = future_data_specific_unscaled['Date'].values

    # Ensure that dates are in datetime format
    dates_specific = pd.to_datetime(dates_specific)

    # Ensure that dates, predicted prices, and actual prices have the same length
    min_length = min(len(future_specific_pred_real), len(future_data_specific_y_real), len(dates_specific))
    future_specific_pred_real = future_specific_pred_real[:min_length]
    future_data_specific_y_real = future_data_specific_y_real[:min_length]
    dates_specific = dates_specific[:min_length]

    # For LSTM Model
    nn_net_worths, nn_net_worths_dates, nn_trade_history = simulate_trading_strategy(
        future_specific_pred_real, future_data_specific_y_real, dates_specific
    )
    profit_nn = nn_net_worths[-1] - 100000

    # For Black-Scholes Model
    bs_net_worths, bs_net_worths_dates, bs_trade_history = simulate_trading_strategy(
        bs_prices_specific, future_data_specific_y_real, dates_specific
    )
    profit_bs = bs_net_worths[-1] - 100000

    # For Heston Model
    heston_net_worths, heston_net_worths_dates, heston_trade_history = simulate_trading_strategy(
        heston_prices_specific, future_data_specific_y_real, dates_specific
    )
    profit_heston = heston_net_worths[-1] - 100000

    # For Monte Carlo Model
    mc_net_worths, mc_net_worths_dates, mc_trade_history = simulate_trading_strategy(
        mc_prices_specific, future_data_specific_y_real, dates_specific
    )
    profit_mc = mc_net_worths[-1] - 100000

    # For Monte Carlo Model
    bop_net_worths, bop_net_worths_dates, bop_trade_history = simulate_trading_strategy(
        bop_prices_specific, future_data_specific_y_real, dates_specific
    )
    profit_bop = bop_net_worths[-1] - 100000

    # For RL Agent
    # Initialize environment with unscaled data
    env = OptionTradingEnv(future_data_specific_unscaled, model, features, SEQUENCE_LENGTH, min_max_scaler, target_standard_scaler)
    agent_path = 'models/option_trading_agent'
    if not os.path.exists(agent_path + '.zip'):
        policy_kwargs = dict(net_arch=[128, 32, 1])
        agent = DQN('MlpPolicy', env, learning_rate=LEARNING_RATE, buffer_size=20000, batch_size=BATCH_SIZE, gamma=0.95, verbose=0, policy_kwargs=policy_kwargs)
        eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=200,
                             deterministic=True, render=False)
        agent.learn(total_timesteps=5000, callback=eval_callback)
        agent.save(agent_path)
    else:
        agent = DQN.load(agent_path, env)

    # For RL Agent
    obs = env.reset()
    net_worths_rl = []
    dates_rl = []
    for _ in range(env.total_steps):
        action, _states = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        net_worths_rl.append(env.net_worth)
        dates_rl.append(env.data.iloc[env.current_step - 1]['Date'])
        if done:
            break
    profit_rl = net_worths_rl[-1] - 100000

    # Convert dates to datetime objects if they are not already
    dates_rl = pd.to_datetime(dates_rl)

    # Adjust lengths to match if necessary
    min_length_bs = min(len(bs_net_worths), len(dates_specific))
    bs_net_worths = bs_net_worths[:min_length_bs]
    dates_bs = dates_specific[:min_length_bs]

    min_length_heston = min(len(heston_net_worths), len(dates_specific))
    heston_net_worths = heston_net_worths[:min_length_heston]
    dates_heston = dates_specific[:min_length_heston]

    min_length_mc = min(len(mc_net_worths), len(dates_specific))
    mc_net_worths = mc_net_worths[:min_length_mc]
    dates_mc = dates_specific[:min_length_mc]

    min_length_nn = min(len(nn_net_worths), len(dates_specific))
    nn_net_worths = nn_net_worths[:min_length_nn]
    dates_nn = dates_specific[:min_length_nn]

    min_length_rl = min(len(net_worths_rl), len(dates_rl))
    net_worths_rl = net_worths_rl[:min_length_rl]
    dates_rl = dates_rl[:min_length_rl]

    min_length_bop = min(len(bop_net_worths), len(dates_specific))
    bop_net_worths = bop_net_worths[:min_length_bop]
    dates_bop = dates_specific[:min_length_bop]


    # Print net worths for all models
    print("Net Worths:")
    print("Neural Network Model Net Worths:", nn_net_worths[-1])
    print("Black-Scholes Model Net Worths:", bs_net_worths[-1])
    print("Heston Model Net Worths:", heston_net_worths[-1])
    print("Monte Carlo Model Net Worths:", mc_net_worths[-1])
    print("RL Agent Model Net Worths:", net_worths_rl[-1])
    print("Binomal:", bop_net_worths[-1])

    # Print RL agent's trade history
    print("\nRL Agent Trade History:")
    for trade in env.trade_history:
        print(trade)
    # Print the price of the option day by day
    # for date, price in zip(dates_specific, actual_prices_specific):
    #     print(f"Date: {date}, Price: {price}")
    # Plot net worth over time for each model
    # Calculate profits as a percentage
    initial_cash = 100000
    profit_percentage_nn = (nn_net_worths[-1] - initial_cash) / initial_cash * 100
    profit_percentage_bs = (bs_net_worths[-1] - initial_cash) / initial_cash * 100
    profit_percentage_heston = (heston_net_worths[-1] - initial_cash) / initial_cash * 100
    profit_percentage_mc = (mc_net_worths[-1] - initial_cash) / initial_cash * 100
    profit_percentage_rl = (net_worths_rl[-1] - initial_cash) / initial_cash * 100
    profit_percentage_bop = (bop_net_worths[-1] - initial_cash) / initial_cash * 100

    # Print profits as a percentage
    print("Profits as a Percentage:")
    print(f"Neural Network Model: {profit_percentage_nn:.2f}%")
    print(f"Black-Scholes Model: {profit_percentage_bs:.2f}%")
    print(f"Heston Model: {profit_percentage_heston:.2f}%")
    print(f"Monte Carlo Model: {profit_percentage_mc:.2f}%")
    print(f"RL Agent Model: {profit_percentage_rl:.2f}%")
    print(f"BOP Agent Model: {profit_percentage_bop:.2f}%")
    plt.figure(figsize=(6, 6))
    plt.plot(dates_bs, bs_net_worths, label='Black-Scholes', color=PLOT_COLORS['niebieski'])
    plt.plot(dates_heston, heston_net_worths, label='Heston', color=PLOT_COLORS['czerwony'])
    plt.plot(dates_mc, mc_net_worths, label='Monte-Carlo', color=PLOT_COLORS['zielony'])
    plt.plot(dates_nn, nn_net_worths, label='LSTM', color=PLOT_COLORS['fioletowy'])
    plt.plot(dates_rl, net_worths_rl, label='Hybrid Model (with RL)', color=PLOT_COLORS['zolty'])
    plt.plot(dates_bop, bop_net_worths, label='Binomial Option Pricing', color=PLOT_COLORS['ciemnioczerwony'])
    plt.xlabel('Date')
    plt.ylabel('Net Worth')
    plt.title('Net Worth Over Time for Different Models')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('net_worth_over_time_plot3.pdf')
    plt.show()


if __name__ == '__main__':
    main()