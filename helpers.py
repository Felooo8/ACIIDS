import warnings
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from traditinal_models import calculate_garch_volatility


# Constants
SEQUENCE_LENGTH = 10
TEST_SIZE = 0.2  # 20% of data for future use
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.0001

TARGET_INDEX = 0

# Load and preprocess data
def load_and_preprocess_data(market_data_path):
    """
    Load data from CSV and preprocess it.
    """
    # Load market data
    data = pd.read_csv(market_data_path, parse_dates=['QUOTE_DATE', 'EXPIRE_DATE'])

    # Filter option_ids with at least 50 records and minimum volume over 1
    data = data.groupby('option_id').filter(lambda x: len(x) > SEQUENCE_LENGTH and x['VOLUME'].min() > 1)
    print(f"Number of unique option_ids: {len(data.option_id.unique())}")

    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)  # For any remaining NaNs

    # Calculate GARCH volatility
    data['garch_volatility'] = calculate_garch_volatility(data, column='UNDERLYING_LAST')
    # Calculate 'dte' (Days to Expiration)
    data['dte'] = (data['EXPIRE_DATE'] - data['QUOTE_DATE']).dt.days

    # Select only relevant columns
    selected_columns = [
        'QUOTE_DATE', 'EXPIRE_DATE', 'option_id', 'LAST', 'UNDERLYING_LAST', 'STRIKE',
        'IV', 'dte', 'is_call', 'sentiment_score', 'garch_volatility', 'RSI','MACD','MACD_Signal','PE Ratio','Price to Sales Ratio'
    ]
    data = data[selected_columns]

    # Feature selection
    features = ['UNDERLYING_LAST', 'STRIKE', 'is_call', 'IV', 'dte', 'sentiment_score', 'garch_volatility', 'RSI','MACD','MACD_Signal','PE Ratio','Price to Sales Ratio']

    # Target variable
    target = 'LAST'

    # Ensure correct data types
    data[features] = data[features].astype(float)

    # Save unscaled data
    data_unscaled = data[['QUOTE_DATE', 'option_id', 'UNDERLYING_LAST', 'STRIKE', 'IV', 'dte', 'LAST', 'is_call', 'sentiment_score', 'garch_volatility', 'RSI','MACD','MACD_Signal','PE Ratio','Price to Sales Ratio']].copy()

    return data, features, target, data_unscaled

# Split data into training and future sets
def split_data(data, test_size=TEST_SIZE):
    """
    Split data into training and future datasets.
    """
    split_index = int(len(data) * (1 - test_size))
    training_data = data.iloc[:split_index]
    future_data = data.iloc[split_index:]

    return training_data, future_data

# Create sequences for training
def create_sequences(data, features, target=None, seq_length=SEQUENCE_LENGTH):
    X = []
    y = []
    indices = []

    # Sort data by date
    data = data.sort_values('QUOTE_DATE').reset_index(drop=True)

    # Group data by option_id
    for option_id, group in data.groupby('option_id'):
        # Ensure the group is long enough for sequence creation
        if len(group) >= seq_length:
            for i in range(0, len(group) - seq_length):
                X.append(group[features].iloc[i:i+seq_length].values)
                y.append(group[target].iloc[i + seq_length])  # Target is the value after the end of the sequence
                indices.append(group.index[i + seq_length])  # Save the index of the target row

    return np.array(X), np.array(y), indices

# Build LSTM model
def build_model(num_features, seq_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(seq_length, num_features)))
    model.add(LSTM(64))
    model.add(Dropout(0.12))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the model
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

# Plot learning curves
def plot_learning_curves(history):
    plt.figure(figsize=(12,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Print metrics
def print_metrics(model_name, y_true, y_pred):
    """
    Print RMSE and MAE metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f'{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}')

# Create full array for inverse transformation
def create_full_array_for_inverse(y, target_index, num_features):
    full_array = np.zeros((y.shape[0], num_features))
    full_array[:, target_index] = y.flatten()
    return full_array