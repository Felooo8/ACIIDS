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
            self.trade_history.append({'step': self.current_step, 'action': 'Hold', 'price': current_price, 'shares': self.shares_held})

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

def simulate_trading_strategy(predicted_prices, actual_prices, initial_cash=100000):
    cash = initial_cash
    position = 0  # Number of shares held
    net_worths = []
    trade_history = []
    
    for i in range(len(predicted_prices) - 1):
        predicted_next_price = predicted_prices[i + 1]
        actual_current_price = actual_prices[i]
        actual_next_price = actual_prices[i + 1]
        
        # Decide to buy or sell one unit
        if predicted_next_price > actual_current_price:
            # Buy one unit
            if cash >= actual_current_price:
                cash -= actual_current_price
                position += 1
                trade_history.append({'step': i, 'action': 'Buy', 'price': actual_current_price, 'shares': 1})
        else:
            # Sell one unit if holding
            if position > 0:
                cash += actual_current_price
                position -= 1
                trade_history.append({'step': i, 'action': 'Sell', 'price': actual_current_price, 'shares': 1})
        
        net_worth = cash + position * actual_current_price
        net_worths.append(net_worth)
    
    return net_worths, trade_history


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
    model_path = 'models/option_pricing_model_lstm_yes_sentiment.keras'
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
    return
    # Predict prices for the selected option
    y_long_pred = model.predict(x_long)
    y_long_real = target_standard_scaler.inverse_transform(y_long.reshape(-1, 1)).flatten()
    y_long_pred_real = target_standard_scaler.inverse_transform(y_long_pred.reshape(-1, 1)).flatten()

    # Example data for illustration purposes
    dates = pd.bdate_range(start=datetime(2023, 1, 1), periods=len(y_long_real)).to_pydatetime().tolist()


    x_long_unscaled = min_max_scaler.inverse_transform(x_long.reshape(-1, x_long.shape[2]))
    x_long_unscaled = x_long_unscaled.reshape(x_long.shape)  # Reshape back to (116, 30, 6)
    x_long_final_timestep = x_long_unscaled[:, -1, :]  # Select the last timestep for each sequence

    # Extract relevant columns for pricing models
    S = x_long_final_timestep[:, 0]  # underlying_price
    K = x_long_final_timestep[:, 1]  # price_strike
    T = x_long_final_timestep[:, 4] / 252  # dte converted to years
    sigma = x_long_final_timestep[:, 3]  # iv
    option_type = x_long_final_timestep[:, 2]  # is_call

    r = 0.04  # Risk-free interest rate
    kappa, theta, xi, rho = 2.0, 0.01, 0.1, -0.5  # Heston model parameters

    bs_prices = np.array([black_scholes(S[i], K[i], T[i], r, sigma[i], option_type[i]) for i in range(len(S))])
    heston_prices = np.array([heston_model_price(S[i], K[i], T[i], r, sigma[i], kappa, theta, xi, rho, option_type[i]) for i in range(len(S))])
    bop_prices = np.array([binomial_option_price(S[i], K[i], T[i], r, sigma[i], N=100, option_type=option_type[i]) for i in range(len(S))])
    mc_prices = np.array([monte_carlo_option_price(S[i], K[i], T[i], r, sigma[i], option_type=option_type[i]) for i in range(len(S))])
    # LSTM model
    print_metrics("LSTM Model", y_long_real, y_long_pred_real)

    # Black-Scholes model
    print_metrics("Black-Scholes Model", y_long_real, bs_prices)

    # Heston model
    print_metrics("Heston Model", y_long_real, heston_prices)

    # Binomial Option Pricing model
    print_metrics("Binomial Option Pricing Model", y_long_real, bop_prices)

    # Monte Carlo model
    print_metrics("Monte Carlo Model", y_long_real, mc_prices)
    # Plot predicted vs actual prices for the selected option with custom colors
    plt.figure(figsize=(6, 6))
    plt.plot(dates, y_long_real, label='Actual Prices', color=PLOT_COLORS['niebieski'])
    plt.plot(dates, y_long_pred_real, label='Predicted Prices', color=PLOT_COLORS['czerwony'])
    plt.plot(dates, bs_prices, label='Black-Scholes Prices', color=PLOT_COLORS['zolty'])
    plt.plot(dates, heston_prices, label='Heston Prices', color=PLOT_COLORS['fioletowy'])
    plt.plot(dates, bop_prices, label='Binomial Option Pricing Model', color=PLOT_COLORS['zielony'])
    plt.plot(dates, mc_prices, label='Monte Carlo Model', color=PLOT_COLORS['jasnoniebieski'])
    plt.xlabel('Date')
    plt.ylabel('Option Price')
    plt.title(f'Predicted vs. Actual Prices for one specific option contract')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate dates for better readability
    plt.tight_layout()
    # Save as PGF file
    plt.savefig('option_price_plot_all.pdf')
    plt.show()
    return
    # Evaluate the model on validation data
    y_pred = model.predict(X_val)
    y_val_real = target_standard_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_real = target_standard_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    print_metrics("Neural Network Model (Validation)", y_val_real, y_pred_real)

    # Evaluate the model on future data
    future_pred = model.predict(future_data_x)
    future_data_y_real = target_standard_scaler.inverse_transform(future_data_y.reshape(-1, 1)).flatten()
    future_pred_real = target_standard_scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()

    print_metrics("Neural Network Model (Future Data)", future_data_y_real, future_pred_real)

    # Evaluate the model on future data specific option
    future_specific_pred = model.predict(future_data_specific_x)
    future_data_specific_y_real = target_standard_scaler.inverse_transform(future_data_specific_y.reshape(-1, 1)).flatten()
    future_specific_pred_real = target_standard_scaler.inverse_transform(future_specific_pred.reshape(-1, 1)).flatten()

    print_metrics("Neural Network Model (Specific Option)", future_data_specific_y_real, future_specific_pred_real)

    # Prepare data for traditional models
    # For validation data
    X_val_reshaped = X_val.reshape(-1, X_val.shape[2])
    X_val_real = min_max_scaler.inverse_transform(X_val_reshaped)
    X_val_real = X_val_real.reshape(X_val.shape)
    S = X_val_real[:, -1, 0]  # underlying_price
    K = X_val_real[:, -1, 1]  # price_strike
    T = X_val_real[:, -1, 4] / 252  # dte converted to years
    sigma = X_val_real[:, -1, 3]  # iv
    option_type = X_val_real[:, -1, 2]  # is_call

    r = 0.04  # Risk-free interest rate

    # Set Heston parameters (example values)
    kappa = 2.0        # Mean reversion rate
    theta = 0.01       # Long-term variance
    xi = 0.1           # Volatility of volatility
    rho = -0.5         # Correlation between the underlying asset and its volatility

    # Calculate prices using different models
    bs_prices = np.array([black_scholes(S[i], K[i], T[i], r, sigma[i], option_type[i]) for i in range(len(S))])
    heston_prices = np.array([heston_model_price(S[i], K[i], T[i], r, sigma[i], kappa, theta, xi, rho, option_type[i]) for i in range(len(S))])
    bop_prices = np.array([binomial_option_price(S[i], K[i], T[i], r, sigma[i], N=100, option_type=option_type[i]) for i in range(len(S))])
    mc_prices = np.array([monte_carlo_option_price(S[i], K[i], T[i], r, sigma[i], option_type=option_type[i]) for i in range(len(S))])

    print(f"Length of y_val_real: {len(y_val_real)}")
    print(f"Length of bs_prices: {len(bs_prices)}")
    # Compare models
    print_metrics("Black-Scholes Model", y_val_real, bs_prices)
    print_metrics("Heston Model", y_val_real, heston_prices)
    print_metrics("Binomial Option Pricing Model", y_val_real, bop_prices)
    print_metrics("Monte Carlo Model", y_val_real, mc_prices)

    # Assuming option_ids_in_validation should be extracted from the validation data
    option_ids_in_validation = training_data_unscaled['option_id'].values[:len(y_val_real)]
    results = simulate_trading_strategy_multiple_options(y_pred_real, y_val_real, option_ids_in_validation)

    profits = [res['profit'] for res in results]
    average_profit = np.mean(profits)
    std_profit = np.std(profits)

    print(f"Average Cumulative Profit: {average_profit}")
    print(f"Standard Deviation of Profits: {std_profit}")

    # Select an option_id to plot
    selected_option_id = specific_option_id  # Or any other option_id
    selected_result = next(res for res in results if res['option_id'] == selected_option_id)

    plt.figure(figsize=(12,6))
    plt.plot(selected_result['net_worths'], label='Net Worth Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Net Worth')
    plt.title(f'Net Worth Over Time for Option ID {selected_option_id}')
    plt.legend()
    plt.show()

    # Plot predicted vs actual prices for validation data
    plt.figure(figsize=(12,6))
    plt.plot(y_val_real[:SEQUENCE_LENGTH], label='Actual Prices')
    plt.plot(y_pred_real[:SEQUENCE_LENGTH], label='LSTM Predicted Prices')
    plt.plot(bs_prices[:SEQUENCE_LENGTH], label='Black-Scholes Prices')
    plt.plot(heston_prices[:SEQUENCE_LENGTH], label='Heston Prices')
    plt.plot(y_val_real[:SEQUENCE_LENGTH], label='Binomial Option Pricing Model')
    plt.plot(mc_prices[:SEQUENCE_LENGTH], label='Monte Carlo Model')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Option Price')
    plt.title('Predicted vs. Actual Prices on Validation Data')
    plt.show()

if __name__ == '__main__':
    main()