# Data manipulation
import pandas as pd
import numpy as np

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Flatten,
    MultiHeadAttention, Add
)
from tensorflow.keras.callbacks import EarlyStopping

# Reinforcement Learning
import gym
from gym import spaces
from stable_baselines3 import DQN

# Sentiment Analysis
from transformers import pipeline

# Other
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Load your data
data = pd.read_csv('data/final_merged_all_data.csv', parse_dates=['Date', 'expiration_date'])


# Handle missing values
data.fillna(method='ffill', inplace=True)
data.fillna(0, inplace=True)  # For any remaining NaNs

# Include 'call_put' as a feature
data['is_call'] = data['call_put'].map({'C': 1, 'P': 0})
data['sentiment'] = data['positive_articles'] - data['negative_articles']

# Feature selection
features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'delta', 'gamma', 'vega', 'theta', 'rho',
    'iv', 'dte', 'underlying_price',
    'RSI', 'MACD', 'sentiment',
    'is_call'
]

# Target variable
target = 'price'

# Ensure that the data types are correct
data[features] = data[features].astype(float)
data[target] = data[target].astype(float)
# Initialize scalers
standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()

# Standard scaling for continuous features
standard_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'underlying_price', 'iv', 'dte']
data[standard_features] = standard_scaler.fit_transform(data[standard_features])

# Min-Max scaling for bounded features
min_max_features = [
    'delta', 'gamma', 'vega', 'theta', 'rho',
    'RSI', 'MACD', 'sentiment', 'is_call'
]
data[min_max_features] = min_max_scaler.fit_transform(data[min_max_features])

def build_transformer_model(seq_length, num_features, d_model=64):
    inputs = Input(shape=(seq_length, num_features))
    
    # Project inputs to d_model dimensions
    x = Dense(d_model)(inputs)
    
    # Transformer Encoder Block
    attention_output = MultiHeadAttention(num_heads=4, key_dim=d_model // 4)(x, x)
    attention_output = Dropout(0.1)(attention_output)
    attention_output = Add()([x, attention_output])  # Residual connection
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
    # Feed Forward Network
    ffn_output = Dense(128, activation='relu')(attention_output)
    ffn_output = Dropout(0.1)(ffn_output)
    ffn_output = Dense(d_model)(ffn_output)  # Project back to d_model dimensions
    ffn_output = Add()([attention_output, ffn_output])  # Residual connection
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)
    
    # Flatten and Output
    flatten = Flatten()(ffn_output)
    dense = Dense(64, activation='relu')(flatten)
    outputs = Dense(1)(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_sequences(data, seq_length):
    X = []
    y = []
    option_ids = data['option_id'].unique()
    
    for option_id in option_ids:
        option_data = data[data['option_id'] == option_id]
        option_data = option_data.sort_values('Date')
        
        # Extract features and target
        feature_data = option_data[features].values
        target_data = option_data[target].values
        
        # Create sequences
        for i in range(len(option_data) - seq_length):
            X.append(feature_data[i:i+seq_length])
            y.append(target_data[i+seq_length])
    
    return np.array(X), np.array(y)

# Create sequences with 30-day window
sequence_length = 30
X_seq_30, y_seq_30 = create_sequences(data, sequence_length)

# Split data into training and validation sets
X_train_30, X_val_30, y_train_30, y_val_30 = train_test_split(
    X_seq_30, y_seq_30, test_size=0.2, shuffle=False)

# Build model for 30-day window
model_30 = build_transformer_model(sequence_length, len(features))
model_30.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='mean_squared_error')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history_30 = model_30.fit(
    X_train_30, y_train_30,
    epochs=10,
    batch_size=64,
    validation_data=(X_val_30, y_val_30),
    callbacks=[early_stopping]
)

# Make predictions
y_pred_30 = model_30.predict(X_val_30)

# Compare predicted prices with actual prices
plt.figure(figsize=(12,6))
plt.plot(y_val_30, label='Actual Prices (30-day window)')
plt.plot(y_pred_30, label='Predicted Prices (30-day window)')
plt.legend()
plt.show()

# Calculate performance metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse_30 = np.sqrt(mean_squared_error(y_val_30, y_pred_30))
mae_30 = mean_absolute_error(y_val_30, y_pred_30)

print(f"30-day Window - RMSE: {rmse_30}, MAE: {mae_30}")

from sklearn.model_selection import train_test_split

class OptionPricingEnv(gym.Env):
    def __init__(self, data, model):
        super(OptionPricingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.model = model
        self.current_step = 0
        self.total_steps = len(self.data) - 1
        
        # Define action space: Adjust price up, down, or hold
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Adjust shape according to the number of features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(features),), dtype=np.float32)
        
    def reset(self):
        self.current_step = 0
        return self._get_observation()
    
    def _get_observation(self):
        obs = self.data.loc[self.current_step, features].values.astype(np.float32)
        return obs
    
    def step(self, action):
        # Apply action
        adjustment = 0
        if action == 0:
            adjustment = -0.01  # Decrease price by 1%
        elif action == 1:
            adjustment = 0.01   # Increase price by 1%
        # Else action == 2: Hold (no adjustment)
        
        # Predict price
        seq = self._get_sequence(self.current_step)
        predicted_price = self.model.predict(seq[np.newaxis, :])[0, 0]
        adjusted_price = predicted_price * (1 + adjustment)
        
        # Get actual market price
        actual_price = self.data.loc[self.current_step, target]
        
        # Calculate reward
        reward = -abs(adjusted_price - actual_price)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.total_steps
        
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        return obs, reward, done, {}
    
    def _get_sequence(self, idx):
        # Ensure we don't go out of bounds
        start_idx = max(0, idx - sequence_length + 1)
        seq_data = self.data.loc[start_idx:idx, features].values
        # Pad sequence if needed
        if len(seq_data) < sequence_length:
            padding = np.zeros((sequence_length - len(seq_data), len(features)))
            seq_data = np.vstack((padding, seq_data))
        return seq_data

# Create the environment
env = OptionPricingEnv(data, model_30)

# Define the RL agent
agent = DQN('MlpPolicy', env, verbose=1)

# Train the agent
agent.learn(total_timesteps=5000)

# Save the agent
agent.save("option_pricing_agent")


import shap

# Use the correct variable names
# Note: SHAP can be computationally intensive. Consider using a smaller sample.
# Initialize the DeepExplainer
explainer = shap.DeepExplainer(model_30, X_train_30[:100])

# Compute SHAP values
shap_values = explainer.shap_values(X_val_30[:10])

# Plot SHAP values
shap.summary_plot(shap_values, X_val_30[:10], feature_names=features)


def objective(trial):
    # Suggest hyperparameters
    num_heads = trial.suggest_int('num_heads', 2, 8)
    key_dim = trial.suggest_int('key_dim', 16, 64)
    ff_dim = trial.suggest_int('ff_dim', 64, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    d_model = trial.suggest_int('d_model', 32, 128)
    
    # Build model with suggested hyperparameters
    inputs = Input(shape=(sequence_length, len(features)))
    x = Dense(d_model)(inputs)
    
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = Add()([x, attention_output])  # Residual connection
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
    ffn_output = Dense(ff_dim, activation='relu')(attention_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Add()([attention_output, ffn_output])  # Residual connection
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)
    
    flatten = Flatten()(ffn_output)
    dense = Dense(64, activation='relu')(flatten)
    outputs = Dense(1)(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    
    # Train model
    model.fit(X_train_30, y_train_30, epochs=10, batch_size=64, verbose=0)
    
    # Evaluate model
    y_pred = model.predict(X_val_30)
    rmse = np.sqrt(mean_squared_error(y_val_30, y_pred))
    return rmse
