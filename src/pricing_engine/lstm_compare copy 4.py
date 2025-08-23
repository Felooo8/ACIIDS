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
import shap
import seaborn as sns
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

    # Option Greeks
    data['delta'] = data['delta'].fillna(0)
    data['gamma'] = data['gamma'].fillna(0)
    data['theta'] = data['theta'].fillna(0)
    data['vega'] = data['vega'].fillna(0)
    data['rho'] = data['rho'].fillna(0)

    # Technical indicators
    data['RSI'] = data['RSI'].fillna(0)
    data['MACD'] = data['MACD'].fillna(0)

    # Fundamental indicators
    data['PE_Ratio'] = data['PE Ratio'].fillna(0)
    data['Revenue_per_Share'] = data['Revenue per Share'].fillna(0)

    selected_columns = [
        'Date', 'expiration_date', 'option_id', 'price', 'underlying_price', 'price_strike',
        'iv', 'dte', 'is_call', 'sentiment_score', 'delta', 'gamma', 'theta', 'vega', 'rho',
        'RSI', 'MACD', 'PE_Ratio', 'Revenue_per_Share'
    ]
    data = data[selected_columns]
    
    # Feature selection
    features = [
        'underlying_price', 'price_strike', 'is_call', 'iv', 'dte', 'sentiment_score',
        'delta', 'gamma', 'theta', 'vega', 'rho', 'RSI', 'MACD', 'PE_Ratio', 'Revenue_per_Share'
    ]

    # Target variable
    target = 'price'

    # Ensure correct data types
    data[features] = data[features].astype(float)
    data[target] = data[target].astype(float)

    # Save unscaled data
    data_unscaled = data[['Date', 'option_id', 'underlying_price', 'price_strike', 'iv', 'dte', 'price', 'is_call', 'sentiment_score', 'delta', 'gamma', 'theta', 'vega', 'rho', 'RSI', 'MACD', 'PE_Ratio', 'Revenue_per_Share']].copy()

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
        
        # Combine data for all options and sort by date
        self.data.sort_values(['Date', 'option_id'], inplace=True)
        self.total_steps = len(self.data) - 1
        
        # Action space: For each option, 0 - Hold, 1 - Buy, 2 - Sell
        self.option_ids = self.data['option_id'].unique()
        self.num_options = len(self.option_ids)
        self.action_space = spaces.MultiDiscrete([3] * self.num_options)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.seq_length * len(self.features) * self.num_options + self.num_options,), dtype=np.float32
        )
        
        # Trading parameters
        self.positions = {oid: 0 for oid in self.option_ids}  # Positions per option_id
        self.cash = 100000  # Starting cash
        self.net_worth = self.cash
        self.trade_history = []  # Record of trades
    
        # Initialize history buffer per option
        self.history = {oid: [] for oid in self.option_ids}
    
        # Scalers
        self.min_max_scaler = min_max_scaler
        self.target_standard_scaler = target_standard_scaler
    
    def reset(self):
        self.current_step = 0
        self.positions = {oid: 0 for oid in self.option_ids}
        self.cash = 100000
        self.net_worth = self.cash
        self.trade_history = []
        self.history = {oid: [] for oid in self.option_ids}
    
        # Pre-fill the history with initial data
        for oid in self.option_ids:
            option_data = self.data[self.data['option_id'] == oid]
            for _ in range(self.seq_length):
                if self.current_step < len(option_data):
                    self.history[oid].append(option_data.iloc[self.current_step])
                else:
                    break
        return self._next_observation()
    
    def _next_observation(self):
        obs = []
        predicted_prices = []
        for oid in self.option_ids:
            history = self.history[oid]
            # Ensure we have enough data
            if len(history) < self.seq_length:
                pad_size = self.seq_length - len(history)
                pad_data = [history[0]] * pad_size
                sequence_data = pad_data + history
            else:
                sequence_data = history[-self.seq_length:]
            
            # Extract features and scale
            features = [row[self.features].values for row in sequence_data]
            features_scaled = self.min_max_scaler.transform(features)
            features_scaled = np.array(features_scaled)
            obs.extend(features_scaled.flatten())
            
            # Prepare input for prediction
            features_scaled_input = features_scaled.reshape(1, self.seq_length, len(self.features))
            # Predict price
            predicted_price_scaled = self.model.predict(features_scaled_input)
            predicted_price = self.target_standard_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0, 0]
            predicted_prices.append(predicted_price)
        obs.extend(predicted_prices)
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        # Execute actions
        done = False
        reward = 0
        prev_net_worth = self.net_worth
    
        # Get current data for all options
        current_data = self.data.iloc[self.current_step]
        for idx, oid in enumerate(self.option_ids):
            current_row = current_data[current_data['option_id'] == oid]
            if current_row.empty:
                continue
            current_price = current_row['price'].values[0]
            action = actions[idx]
            
            if action == 1:  # Buy
                # Buy one unit
                if self.cash >= current_price:
                    self.cash -= current_price
                    self.positions[oid] += 1
                    self.trade_history.append({'step': self.current_step, 'option_id': oid, 'action': 'Buy', 'price': current_price, 'shares': 1})
            elif action == 2:  # Sell
                # Sell one unit if holding
                if self.positions[oid] > 0:
                    self.cash += current_price
                    self.positions[oid] -= 1
                    self.trade_history.append({'step': self.current_step, 'option_id': oid, 'action': 'Sell', 'price': current_price, 'shares': 1})
            else:
                # Hold
                pass
    
        # Update net worth
        self.net_worth = self.cash
        for oid in self.option_ids:
            current_row = current_data[current_data['option_id'] == oid]
            if current_row.empty:
                continue
            current_price = current_row['price'].values[0]
            self.net_worth += self.positions[oid] * current_price
    
        # Calculate reward (change in net worth)
        reward = self.net_worth - prev_net_worth
    
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

    # Filter future data
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

    # Create sequences for training data
    x_long, y_long = create_sequences(data_selected, features, target, SEQUENCE_LENGTH)

    # Create sequences for future data including last SEQUENCE_LENGTH - 1 days from training data
    future_data_x, future_data_y = create_sequences_with_future_target(
        training_data, future_data_selected, features, target, seq_length=SEQUENCE_LENGTH
    )

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

    # Extract unscaled features for the selected options
    S = future_data_selected_unscaled['underlying_price'].values
    K = future_data_selected_unscaled['price_strike'].values
    sigma = future_data_selected_unscaled['iv'].values
    T = future_data_selected_unscaled['dte'].values / 252  # Convert to years
    option_type = future_data_selected_unscaled['is_call'].values
    actual_prices = future_data_selected_unscaled['price'].values
    option_ids = future_data_selected_unscaled['option_id'].values
    dates = pd.to_datetime(future_data_selected_unscaled['Date'].values)

    # Predict using the LSTM model
    future_pred = model.predict(future_data_x)
    future_pred_real = target_standard_scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
    future_data_y_real = target_standard_scaler.inverse_transform(future_data_y.reshape(-1, 1)).flatten()

    r = 0.04  # Risk-free interest rate
    kappa = 2.0        # Mean reversion rate
    theta = 0.01       # Long-term variance
    xi = 0.1           # Volatility of volatility
    rho = -0.5         # Correlation between the underlying asset and its volatility

    # Calculate prices using different models
    bs_prices = np.array([black_scholes(S[i], K[i], T[i], r, sigma[i], option_type[i]) for i in range(len(S))])
    heston_prices = np.array([heston_model_price(S[i], K[i], T[i], r, sigma[i], kappa, theta, xi, rho, option_type[i]) for i in range(len(S))])
    mc_prices = np.array([monte_carlo_option_price(S[i], K[i], T[i], r, sigma[i], option_type=option_type[i]) for i in range(len(S))])

    # Combine all data into a single DataFrame
    combined_data = pd.DataFrame({
        'date': dates,
        'option_id': option_ids,
        'predicted_price_nn': future_pred_real,
        'predicted_price_bs': bs_prices,
        'predicted_price_heston': heston_prices,
        'predicted_price_mc': mc_prices,
        'actual_price': actual_prices,
        'option_type': option_type,
        'S': S,
        'K': K,
        'sigma': sigma,
        'T': T,
    })
    # Sort combined data by date
    combined_data.sort_values('date', inplace=True)
    combined_data.reset_index(drop=True, inplace=True)

    # Simulate trading strategies for all models
    # For LSTM Model
    nn_net_worths, nn_dates, profit_nn, nn_trade_history = simulate_trading_strategy_multi_option(
        combined_data, 'LSTM'
    )

    # For Black-Scholes Model
    bs_net_worths, bs_dates, profit_bs, bs_trade_history = simulate_trading_strategy_multi_option(
        combined_data, 'Black-Scholes'
    )

    # For Heston Model
    heston_net_worths, heston_dates, profit_heston, heston_trade_history = simulate_trading_strategy_multi_option(
        combined_data, 'Heston'
    )

    # For Monte Carlo Model
    mc_net_worths, mc_dates, profit_mc, mc_trade_history = simulate_trading_strategy_multi_option(
        combined_data, 'Monte-Carlo'
    )

    # For RL Agent
    # Initialize environment with combined data
    env = OptionTradingEnv(future_data_selected_unscaled, model, features, SEQUENCE_LENGTH, min_max_scaler, target_standard_scaler)
    agent_path = 'models/option_trading_agent_multi_option'
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

    obs = env.reset()
    net_worths_rl = []
    dates_rl = []
    for _ in range(env.total_steps):
        action, _states = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        net_worths_rl.append(env.net_worth)
        dates_rl.append(env.data.iloc[env.current_step]['Date'])
        if done:
            break
    profit_rl = net_worths_rl[-1] - 100000

    # Plot net worth over time for each model
    plt.figure(figsize=(12, 6))
    plt.plot(nn_dates, nn_net_worths, label='LSTM', color=PLOT_COLORS['fioletowy'])
    plt.plot(bs_dates, bs_net_worths, label='Black-Scholes', color=PLOT_COLORS['niebieski'])
    plt.plot(heston_dates, heston_net_worths, label='Heston', color=PLOT_COLORS['czerwony'])
    plt.plot(mc_dates, mc_net_worths, label='Monte-Carlo', color=PLOT_COLORS['zielony'])
    plt.plot(dates_rl, net_worths_rl, label='Hybrid Model (with RL)', color=PLOT_COLORS['zolty'])
    plt.xlabel('Date')
    plt.ylabel('Net Worth')
    plt.title('Net Worth Over Time for Different Models Trading Multiple Options')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('net_worth_over_time_multiple_options.png')
    plt.show()

    # Collect cumulative profits
    cumulative_profits = {
        'Black-Scholes': profit_bs,
        'Heston': profit_heston,
        'Monte-Carlo': profit_mc,
        'LSTM': profit_nn,
        'Hybrid Model (with RL)': profit_rl
    }

    # Print the table
    print("\nTrading performance comparison based on cumulative profit:")
    print("{:<30} {:<15}".format('Model', 'Cumulative Profit'))
    for model, profit in cumulative_profits.items():
        print("{:<30} {:<15.2f}".format(model, profit))


if __name__ == '__main__':
    main()