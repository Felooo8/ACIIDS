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
import QuantLib as ql
warnings.filterwarnings('ignore')

# Additional imports for volatility models and sentiment analysis
from arch import arch_model
from transformers import pipeline

# Constants
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.2  # 10% of data for future use
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 50
LEARNING_RATE = 0.001

# Initialize scalers
feature_standard_scaler = StandardScaler()
target_standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()


def heston_model_price(S, K, T, r, sigma, kappa, theta, xi, rho, option_type='call'):
    """
    Calculate option price using the Heston model.
    """
    # Set evaluation date
    evaluation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = evaluation_date

    # Option parameters   
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == 'call' else ql.Option.Put, K)
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

def binomial_option_price(S, K, T, r, sigma, N=100, option_type='call'):
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
    if option_type == 'call':
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)

    # Backward induction
    for i in range(N-1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])

    return option_values[0]

def monte_carlo_option_price(S, K, T, r, sigma, option_type='call', num_simulations=10000):
    """
    Calculate option price using Monte Carlo simulation.
    """
    # Simulate end-of-period asset price
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
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
    features = ['underlying_price', 'price_strike', 'iv', 'dte']

    # Target variable
    target = 'price'

    # Ensure correct data types
    data[features] = data[features].astype(float)
    data[target] = data[target].astype(float)

    # Save unscaled data
    data_unscaled = data[['Date', 'option_id', 'underlying_price', 'price_strike', 'iv', 'dte', 'price']].copy()

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
    
    # Ensure data is sorted by date
    data = data.sort_values('Date').reset_index(drop=True)
    
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:i+seq_length].values)
        if target:
            y.append(data[target].iloc[i+seq_length])
    if target:
        return np.array(X), np.array(y)
    else:
        return np.array(X), None

def build_model(num_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(num_features,)))
    model.add(Dense(128, activation='relu'))
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
import random


def main():
    market_data_path = 'data/final_merged_all_data.csv'
    # Load and preprocess data
    data, features, target, data_unscaled = load_and_preprocess_data(market_data_path)

    # Split data into training and validation sets
    training_data, validation_data = split_data(data)
    training_data_unscaled, validation_data_unscaled = split_data(data_unscaled)

    # Define features and target
    standard_features = ['underlying_price', 'price_strike', 'iv', 'dte']
    target = 'price'

    # Initialize scalers
    min_max_scaler = MinMaxScaler()
    target_standard_scaler = StandardScaler()

    # Scale features
    training_data[standard_features] = min_max_scaler.fit_transform(training_data[standard_features])
    validation_data[standard_features] = min_max_scaler.transform(validation_data[standard_features])

    # Scale target
    training_data[target] = target_standard_scaler.fit_transform(training_data[[target]])
    validation_data[target] = target_standard_scaler.transform(validation_data[[target]])

    # Create samples for training data
    X, y = create_samples(training_data, features, target)
    X_unscaled, Y_unscaled = create_samples(training_data_unscaled, ['underlying_price', 'price_strike', 'iv', 'dte'], target)
    # X, y = create_sequences(training_data, features, target, seq_length=SEQUENCE_LENGTH)
    # X_unscaled, _ = create_sequences(training_data_unscaled, ['underlying_price', 'price_strike', 'iv', 'dte'], target, seq_length=SEQUENCE_LENGTH)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    X_unscaled_train, X_unscaled_val, Y_unscaled_train, Y_unscaled_val = train_test_split(
        X_unscaled, Y_unscaled, test_size=0.2, shuffle=False
    )
    X_unscaled_val = np.array(X_unscaled_val)

    # Build and train the model
    model_path = 'models/option_pricing_model_asd333121.keras'
    if not os.path.exists(model_path):
        model = build_model(len(features))
        history = train_model(model, X_train, y_train, X_val, y_val)
        plot_learning_curves(history)
        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    # Evaluate the model on validation data
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation RMSE: {rmse}, MAE: {mae}")
    # Evaluate the model on validation data
    y_pred = model.predict(X_val)
    y_val_real = target_standard_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_real = target_standard_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    results_df = pd.DataFrame()
    results_df['predicted_price'] = y_pred_real
    results_df['actual_price'] = y_val_real

    # Optionally, save results to a CSV for manual inspection
    results_df.to_csv('predictions_vs_actuals.csv', index=False)

    # Display some random 20 results
    results_df = results_df.sample(20)
    print(results_df)

    # Plot predictions vs actual prices over time
    plt.figure(figsize=(12,6))
    plt.plot(results_df['actual_price'], label='Actual Prices')
    plt.plot(results_df['predicted_price'], label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Option Price')
    plt.title('Predicted vs Actual Option Prices Over Time')
    plt.legend()
    plt.show()

    # Calculate Black-Scholes prices for validation data using unscaled features
    S = X_unscaled_val[:, 0]  # underlying_price
    K = X_unscaled_val[:, 1]  # price_strike
    T = X_unscaled_val[:, 3] / 252  # dte converted to years
    sigma = X_unscaled_val[:, 2]  # iv
    ########### LSTM: 3D array, BS: 1D array
    # S = X_unscaled_val[:, -1, 0]  # underlying_price
    # K = X_unscaled_val[:, -1, 1]  # price_strike
    # T = X_unscaled_val[:, -1, 3] / 252  # dte converted to years
    # sigma = X_unscaled_val[:, -1, 2]  # iv


    r = 0.04  # Risk-free interest rate
    option_type = 'call' if data['is_call'].iloc[0] == 1 else 'put'
    # Set Heston parameters (example values)
    kappa = 2.0        # Mean reversion rate
    theta = 0.01       # Long-term variance
    xi = 0.1           # Volatility of volatility
    rho = -0.5         # Correlation between the underlying asset and its volatility

    # Calculate prices using different models
    bs_prices = np.array([black_scholes(S[i], K[i], T[i], r, sigma[i], option_type) for i in range(len(S))])
    # heston_prices = np.array([heston_model_price(S[i], K[i], T[i], r, sigma[i], kappa, theta, xi, rho, option_type) for i in range(len(S))])
    # bop_prices = np.array([binomial_option_price(S[i], K[i], T[i], r, sigma[i], N=100, option_type=option_type) for i in range(len(S))])
    # mc_prices = np.array([monte_carlo_option_price(S[i], K[i], T[i], r, sigma[i], option_type=option_type) for i in range(len(S))])
    heston_prices = []
    bop_prices = []
    mc_prices = []
    for i in range(len(S)):
        # Extract the last time step's data from the sequence
        # S = X_unscaled_val[:, -1, 0]  # underlying_price
        # K = X_unscaled_val[:, -1, 1]  # price_strike
        # T = X_unscaled_val[:, -1, 3] / 252  # dte converted to years
        # sigma = X_unscaled_val[:, -1, 2]  # iv

        # LSTM: 3D array, BS: 1D array
        S_scalar = float(S[i])
        K_scalar = float(K[i])
        T_scalar = float(T[i])
        sigma_scalar = float(sigma[i])
        # Ensure values are floats
        S_scalar = float(S_scalar)
        K_scalar = float(K_scalar)
        T_scalar = float(T_scalar)
        sigma_scalar = float(sigma_scalar)
        
        # Call the pricing function
        try:
            price = heston_model_price(S_scalar, K_scalar, T_scalar, r, sigma_scalar, kappa, theta, xi, rho, option_type)
            bop_price = binomial_option_price(S_scalar, K_scalar, T_scalar, r, sigma_scalar, N=100, option_type=option_type)
            mc_price = monte_carlo_option_price(S_scalar, K_scalar, T_scalar, r, sigma_scalar, option_type=option_type)
        except Exception as e:
            print(f"Error at index {i}: {e}")
            price = np.nan
        heston_prices.append(price)
        bop_prices.append(bop_price)
        mc_prices.append(mc_price)
    heston_prices = np.array(heston_prices)
    bop_prices = np.array(bop_prices)
    mc_prices = np.array(mc_prices)

    # Calculate RMSE and MAE for each model
    def print_metrics(model_name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f'{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}')
    
    print_metrics("Neural Network Model", y_val_real, y_pred_real)
    print_metrics("Black-Scholes Model", y_val_real, bs_prices)
    print_metrics("Heston Model", y_val_real, heston_prices)
    print_metrics("Binomial Option Pricing Model", y_val_real, bop_prices)
    print_metrics("Monte Carlo Model", y_val_real, mc_prices)
    # Plot predictions vs actual prices vs traditional models
    plt.figure(figsize=(12,6))
    plt.plot(y_val_real, label='Actual Prices')
    plt.plot(y_pred_real, label='Predicted Prices')
    plt.plot(bs_prices, label='Black-Scholes Prices')
    plt.plot(heston_prices, label='Heston Prices')
    plt.plot(bop_prices, label='Binomial Prices')
    plt.plot(mc_prices, label='Monte Carlo Prices')
    plt.legend()
    plt.title('Model Predictions vs Actual Prices vs Traditional Models')
    plt.show()


if __name__ == '__main__':
    main()
