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
TEST_SIZE = 0.2  # 10% of data for future use
BATCH_SIZE = 32
EPOCHS = 100    
PATIENCE = EPOCHS*0.1
LEARNING_RATE = 0.001

# Initialize scalers
feature_standard_scaler = StandardScaler()
target_standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()

# Function to calculate GARCH volatility on a rolling basis
def calculate_garch_volatility(prices, window=30):
    garch_volatility = []
    
    for i in range(len(prices)):
        # Use all available past data (from the start up to current row), or the last 'window' days
        available_window = min(i, window)
        
        # If fewer than 2 data points are available, append None
        if available_window < 1:
            garch_volatility.append(None)
        else:
            # Use available data for GARCH calculation
            window_prices = prices[i-available_window:i]
            log_returns = np.log(window_prices / window_prices.shift(1)).dropna()

            if len(log_returns) > 1:  # Proceed if there's enough data for GARCH calculation
                try:
                    model = arch_model(log_returns, vol='Garch', p=1, q=1)
                    result = model.fit(disp='off')
                    garch_volatility.append(result.conditional_volatility.iloc[-1])
                except Exception as e:
                    # If any error occurs during GARCH fitting, append None
                    garch_volatility.append(None)
                    print(f"Error at index {i}: {e}")
            else:
                garch_volatility.append(None)
    
    return garch_volatility


# Simplified Heston volatility (using implied volatility as a proxy)
def calculate_heston_volatility(iv):
    return iv  # Placeholder for actual Heston model implementation
import QuantLib as ql

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

# Sentiment analysis using a pre-trained transformer model
# sentiment_pipeline = pipeline('sentiment-analysis')

# def get_sentiment_score(text):
#     try:
#         result = sentiment_pipeline(text)[0]
#         score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
#     except:
#         score = 0  # Default to neutral sentiment if analysis fails
#     return score
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
    # Filter data for option_id = 124140401
    data = data[data['option_id'] == 124140401]

    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)  # For any remaining NaNs

    # Include 'call_put' as a feature
    data['is_call'] = data['call_put'].map({'C': 1, 'P': 0})

    # Calculate GARCH volatility
    data['garch_volatility'] = calculate_garch_volatility(data['price'], window=30)
    
    # Replace NaN values in 'garch_volatility' with 'iv'
    data['garch_volatility'].fillna(data['iv'], inplace=True)

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
            'price',  # Include price as a feature
            'price_strike', 'underlying_price',  # Add these features
            'delta', 'gamma', 'vega', 'theta', 'rho',
            'iv', 'dte',
            'RSI', 'MACD',
            'garch_volatility', 'sentiment_score',
            'is_call'
        ]

    # Target variable
    target = 'price'

    # Ensure correct data types
    data[features] = data[features].astype(float)
    data[target] = data[target].astype(float)

    unscaled_features = ['Date', 'underlying_price', 'price_strike', 'iv', 'dte', 'price']
    data_unscaled = data[unscaled_features].copy()

    # Standard scaling for continuous features (excluding 'price')
    standard_features = ['underlying_price', 'price_strike', 'iv', 'dte',
                        'garch_volatility', 'heston_volatility', 'sentiment_score']
    # data[standard_features] = feature_standard_scaler.fit_transform(data[standard_features])
    data[standard_features] = feature_standard_scaler.fit_transform(data[standard_features])

    # Scale the target variable 'price'
    data[['price']] = target_standard_scaler.fit_transform(data[['price']])

    # Min-Max scaling for bounded features
    min_max_features = [
        'delta', 'gamma', 'vega', 'theta', 'rho',
        'RSI', 'MACD', 'is_call'
    ]
    # data[min_max_features] = min_max_scaler.fit_transform(data[min_max_features])

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

def build_model(num_features, time_steps):
    model = tf.keras.Sequential()
    # Input shape should now include the time steps
    model.add(tf.keras.Input(shape=(time_steps, num_features)))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.4))  # Prevent overfitting
    model.add(tf.keras.layers.GRU(64, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1))  # Predicting a single output value (option price)
    model.compile(optimizer='adam', loss='mse')  # Use Adam optimizer with MSE
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
        self.current_step = 0
        self.total_steps = len(self.data) - 1
        self.max_steps = max_steps

        # Define action space: 0 - Buy, 1 - Sell, 2 - Hold
        self.action_space = spaces.Discrete(3)

        # Observation space: All features plus predicted price
        num_state_features = len(features) + 1  # +1 for predicted price
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_state_features,), dtype=np.float64
        )

        # Position management
        self.position = 0
        self.entry_price = 0
        self.trade_history = []

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.trade_history = []
        return self._get_observation()

    def _get_observation(self):
        # Current feature set
        current_features = self.data.loc[self.current_step, self.features].values

        # Predicted price
        current_features_reshaped = current_features.reshape(1, -1)
        predicted_price = self.model.predict(current_features_reshaped)[0, 0]

        # Combine features and predicted price
        obs = np.concatenate([current_features, [predicted_price]])
        return obs
    
    def step(self, action):
        current_features = self.data.loc[self.current_step, self.features].values
        current_features_reshaped = current_features.reshape(1, -1)
        predicted_price = self.model.predict(current_features_reshaped)[0, 0]

        current_price = self.data.loc[self.current_step, self.target]
        next_price = self.data.loc[self.current_step + 1, self.target] if self.current_step + 1 < len(self.data) else current_price

        
        # Find the next valid price for the same option_id
        while next_step < len(self.data) and self.data.loc[next_step, 'option_id'] != current_option_id:
            next_step += 1
        
        if next_step < len(self.data):
            next_price = self.data.loc[next_step, self.target]
        else:
            next_price = current_price  # If no valid next step, fallback to current price


        # Retrieve contract ID and date
        option_id = self.data.loc[self.current_step + self.sequence_length - 1, 'option_id']  # assuming 'option_id' exists
        date = self.data.loc[self.current_step + self.sequence_length - 1, 'Date']  # assuming 'Date' exists

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
        alpha = 0.0  # Weight for prediction error
        reward += profit - alpha * prediction_error

        # Record trade
        if profit != 0:
            self.trade_history.append({
                'step': self.current_step,
                'action': 'Buy' if action == 0 else 'Sell',
                'entry_price': self.entry_price,
                'exit_price': next_price,
                'profit': profit,
                'option_id': option_id,
                'date': date
            })

        # Move to next step
        self.current_step += 1

        done = self.current_step >= self.total_steps or self.current_step >= self.max_steps

        # Get next observation
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(len(self.features) + 1, dtype=np.float64)

        info = {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'next_price': next_price,
            'action_taken': action,
            'profit': profit,
            'option_id': option_id,  # Add contract ID to the info
            'date': date  # Add date to the info
        }

        # Return obs, reward, done, info
        return obs, reward, done, info


def train_rl_agent(env, total_timesteps=1000):
    """
    Train the reinforcement learning agent.
    """
    agent_path = "option_pricing_agent_new_13"
    if not os.path.exists(agent_path + ".zip"):
        agent = DQN('MlpPolicy', env, verbose=1)
        agent.learn(total_timesteps=total_timesteps)
        # agent.save(agent_path)
    else:
        agent = DQN.load(agent_path, env, verbose=1)
    return agent

def evaluate_rl_agent(env, agent, episodes=10):
    """
    Evaluate the RL agent over a number of episodes.
    """
    total_rewards = []
    all_predicted_prices = []
    all_actual_prices = []
    all_steps = []
    all_dates = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            info['predicted_price'] = target_standard_scaler.inverse_transform(info['predicted_price'].reshape(-1, 1)).reshape(-1)[0]
            info['current_price'] = target_standard_scaler.inverse_transform(info['current_price'].reshape(-1, 1)).reshape(-1)[0]
            info['next_price'] = target_standard_scaler.inverse_transform(info['next_price'].reshape(-1, 1)).reshape(-1)[0]
            reward = target_standard_scaler.inverse_transform(reward.reshape(-1, 1)).reshape(-1)[0]
            print(f"Step {step}: Contract ID {info['option_id']}, Date {info['date']}, "
                f"Action {action}, Reward {reward:.2f}, "
                f"Predicted Price {info['predicted_price']:.2f}, "
                f"Current Price {info['current_price']:.2f}, "
                f"Next Price {info['next_price']:.2f}")
            # Collect data for plotting
            all_predicted_prices.append(info['predicted_price'])
            all_actual_prices.append(info['current_price'])
            all_steps.append(env.current_step)
            all_dates.append(env.data.loc[env.current_step + env.sequence_length - 1, 'Date'])
            step += 1
        total_rewards.append(episode_reward)
        episode_reward = target_standard_scaler.inverse_transform(episode_reward.reshape(-1, 1)).reshape(-1)[0]
        print(f"Episode {episode+1}: Total Reward = {episode_reward}")
    avg_reward = target_standard_scaler.inverse_transform(np.mean(total_rewards).reshape(-1, 1)).reshape(-1)[0]
    print(f"Average Reward over {episodes} episodes: {avg_reward}")
    # Store the collected data in the environment for visualization
    env.evaluation_data = {
        'dates': all_dates,
        'steps': all_steps,
        'predicted_prices': all_predicted_prices,
        'actual_prices': all_actual_prices
    }

def visualize_agent_performance(env):
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
    evaluation_data = env.evaluation_data
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_data['dates'], evaluation_data['actual_prices'], label='Actual Prices')
    plt.plot(evaluation_data['dates'], evaluation_data['predicted_prices'], label='Predicted Prices')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.title('Predicted vs Actual Prices During Evaluation')
    plt.legend()
    plt.show()

def plot_learning_curves(history):
    plt.figure(figsize=(12,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    # Load and preprocess data
    market_data_path = 'data/final_merged_all_data.csv'
    data, features, target, data_unscaled = load_and_preprocess_data(market_data_path)

    # Split data into training and future datasets
    training_data, future_data = split_data(data)
    training_data_unscaled, future_data_unscaled = split_data(data_unscaled)

    # Create samples for training data
    X, y = create_samples(training_data, features, target)
    X_unscaled, _ = create_samples(training_data_unscaled, ['underlying_price', 'price_strike', 'iv', 'dte'], target)
    X, y = create_sequences(training_data, features, target, seq_length=SEQUENCE_LENGTH)
    X_unscaled, _ = create_sequences(training_data_unscaled, ['underlying_price', 'price_strike', 'iv', 'dte'], target, seq_length=SEQUENCE_LENGTH)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    X_unscaled_train, X_unscaled_val = train_test_split(
        X_unscaled, test_size=0.2, shuffle=False
    )
    X_unscaled_val = np.array(X_unscaled_val)

    # Build and train the model
    model_path = 'models/option_pricing_model_asd.keras'
    if not os.path.exists(model_path):
        model = build_model(len(features), SEQUENCE_LENGTH)
        history = train_model(model, X_train, y_train, X_val, y_val)
        # plot_learning_curves(history)
        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    # Evaluate the model on validation data
    y_pred = model.predict(X_val)
    y_val_real = target_standard_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_real = target_standard_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate Black-Scholes prices for validation data using unscaled features
    S = X_unscaled_val[:, -1, 0]  # underlying_price
    K = X_unscaled_val[:, -1, 1]  # price_strike
    T = X_unscaled_val[:, -1, 3] / 252  # dte converted to years
    sigma = X_unscaled_val[:, -1, 2]  # iv
    r = 0.01  # Risk-free interest rate
    option_type = 'call' if data['call_put'].iloc[0] == 'C' else 'put'
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

    # ... rest of the code ...


    # Use future data for RL agent
    if not future_data.empty:
        # Create environment
        env = OptionPricingEnv(future_data, model, features, target)
        # Train RL agent
        agent = train_rl_agent(env)
        # Evaluate RL agent
        evaluate_rl_agent(env, agent)
        # Visualize agent performance
        visualize_agent_performance(env)
    else:
        print("No future data available for RL agent training.")


if __name__ == '__main__':
    main()
