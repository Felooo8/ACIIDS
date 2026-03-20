import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU,LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# Features to use for prediction
X_FEATURES = [
    'positive_articles', 'dte', 'iv', 
    'volume', 'openinterest', 'underlying_price'
]

X_FEATURES = [
    'dte', 'iv', 'underlying_price', 
    'volume', 'openinterest', 'positive_articles'
]

# Preprocessing: prepare data
def prepare_data_for_next_day_prediction(df, lookback=5):
    """
    Prepares data for next day option price prediction using 'lookback' days.

    Args:
    df (pd.DataFrame): DataFrame containing option and stock data.
    lookback (int): Number of past days to use for predicting the next day's price.

    Returns:
    X (np.array): Feature data in shape (num_samples, lookback, num_features).
    Y (np.array): Target next-day prices.
    """
    X, Y = [], []
    
    # Sort by Date to ensure time-series order
    df = df.sort_values('Date')

    # We need at least 'lookback' + 1 rows to create an input-output pair
    for i in range(lookback, len(df)):
        past_data = df.iloc[i-lookback:i][X_FEATURES].values
        next_day_price = df.iloc[i]['price']  # Ensure you're selecting a single 'price' column
        X.append(past_data)
        Y.append(next_day_price)  # Make sure Y gets only one value per row
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)  # Reshape Y to (samples, 1)

# Main function
if __name__ == '__main__':
    # Load your dataset
    df = pd.read_csv('data/final_merged_all_data.csv')

    # Filter data for a specific contract (option_id=124140323)
    df = df[df['option_id'] == 124140323]

    # Label Encoding for 'call_put' column
    label_encoder = LabelEncoder()
    df['call_put'] = label_encoder.fit_transform(df['call_put'])  # Encode 'C' and 'P' to numerical values
    
    # Select relevant features and prepare the data
    df = df[['Date', 'option_id', 'price'] + X_FEATURES]  # Ensure 'Date' and 'option_id' are included

    # Split the data: save 10% for future prediction, 90% for training/testing
    df_train, df_future = train_test_split(df, test_size=0.2, shuffle=False)  # Keep future data for prediction

    # Prepare data with lookback (e.g., 5 days of past data to predict the next day's price)
    lookback = 7
    X_train, Y_train = prepare_data_for_next_day_prediction(df_train, lookback=lookback)
    
    # Split the data into training and testing sets (ensure X and Y are consistent)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    # Normalize the features (scale X data)
    scaler_X = MinMaxScaler()

    # Reshape for scaling (scaler expects 2D input, so we flatten the 3D data)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    # Fit and transform training data, and transform testing data
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

    # Normalize the target variable (price) separately# Normalize the target variable (price) separately
    scaler_Y = MinMaxScaler()

    # Fit and transform the training target data (reshape Y_train to 2D)
    Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1))  # Reshape to (samples, 1)

    # Transform the test target data (reshape Y_test to 2D)
    Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, 1))  # Reshape to (samples, 1)


    # Neural Network Model (LSTM)
    # model = Sequential()
    # # model.add(LSTM(128, input_shape=(lookback, X_train.shape[2]), return_sequences=True))  # LSTM layer
    # # model.add(Dropout(0.3))  # Dropout to prevent overfitting
    # model.add(GRU(32, return_sequences=False))
    # # model.add(Dense(64, activation='relu'))  # Dense layer
    # model.add(Dense(16, activation='relu'))  # Dense layer
    # model.add(Dense(1, activation='linear'))  # Output layer for regression
    
    model = Sequential()
    model.add(LSTM(128, input_shape=(lookback, X_train.shape[2])))
    model.add(Dropout(0.2))
    # model.add(Dense(4, activation='linear'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    # model.summary()

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train_scaled, Y_train_scaled, validation_data=(X_test_scaled, Y_test_scaled), epochs=20, batch_size=64,  callbacks=[early_stopping], verbose=0)
    # Save the model
    # model.save('models/simple_model.keras')
    # model = tf.keras.models.load_model('models/simple_model.keras')

    # Evaluate the model
    loss = model.evaluate(X_test_scaled, Y_test_scaled)
    print(f"Test Loss (MSE): {loss}")

    # Prepare the data for future predictions

    X_future, Y_future_real = prepare_data_for_next_day_prediction(df_future, lookback=lookback)  # Also get real prices

    # Scale the future data
    X_future_reshaped = X_future.reshape(-1, X_future.shape[-1])
    X_future_scaled = scaler_X.transform(X_future_reshaped).reshape(X_future.shape)

    # Make predictions on the future data
# Verify shape of X_future_scaled before predicting

    # Ensure the shape matches (samples, timesteps, features)
    Y_future_pred_scaled = model.predict(X_future_scaled)


    # Denormalize the predicted prices
    Y_future_pred_real = scaler_Y.inverse_transform(Y_future_pred_scaled)
    # Print the real vs predicted future prices
    print("\nReal vs Predicted Future Prices:")
    for real, pred in zip(Y_future_real[:10], Y_future_pred_real[:10]):
        print(f"Real: {real:.4f}, Predicted: {pred[0]:.4f}")  # Access the first (and only) element

    # Calculate MSE for real prices
    mse_future = mean_squared_error(Y_future_real, Y_future_pred_real)
    print(f"\nMean Squared Error for Future Predictions: {mse_future:.4f}")
