import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# Features to use for prediction
X_FEATURES = [
    'dte', 'iv', 'underlying_price', 
    'volume', 'openinterest', 'positive_articles', 'option_id'  # Adding option_id as a feature
]
X_FEATURES = ['dte']


# Preprocessing: prepare data with sliding window
def prepare_data_for_next_day_prediction(df, lookback=5):
    """
    Prepares data for next day option price prediction using 'lookback' days.

    Args:
    df (pd.DataFrame): DataFrame containing option and stock data.
    lookback (int): Number of past days to use for predicting the next day's price.

    Returns:
    X (np.array): Feature data in shape (num_samples, lookback, num_features).
    Y (np.array): Target next-day prices.
    price_history (list): History of prices leading up to the prediction for each sample.
    """
    X, Y, price_history = [], [], []
    
    # Sort by Date to ensure time-series order
    df = df.sort_values('Date')

    # We need at least 'lookback' + 1 rows to create an input-output pair
    for i in range(lookback, len(df)):
        past_data = df.iloc[i-lookback:i][X_FEATURES].values
        next_day_price = df.iloc[i]['price']  # Ensure you're selecting a single 'price' column
        price_hist = df.iloc[i-lookback:i]['price'].values  # Historical prices leading up to the prediction
        X.append(past_data)
        Y.append(next_day_price)
        price_history.append(price_hist)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(price_history)

# Main function to handle predictions for the entire dataset
def predict_for_all_options(df, lookback=5, test_size=0.2, patience=5, epochs=20, batch_size=64):
    """
    Handles prediction for all options by training a single model on the combined dataset.

    Args:
    df (pd.DataFrame): The full dataset containing all option data.
    lookback (int): The number of days to look back for time series prediction.
    test_size (float): The proportion of the data to use for testing.
    patience (int): Early stopping patience.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.

    Returns:
    model: The trained model.
    mse_results: Mean Squared Error for the model on the test set.
    Y_test: Real test prices.
    Y_pred_real: Predicted prices for the test set.
    price_history_test: History of prices leading up to each test sample prediction.
    """
    # Label Encoding for 'call_put' column
    label_encoder = LabelEncoder()
    df['call_put'] = label_encoder.fit_transform(df['call_put'])  # Encode 'C' and 'P' to numerical values
    
    # Select relevant features and prepare the data
    df = df[['Date', 'option_id', 'price'] + X_FEATURES]  # Ensure 'Date' and 'option_id' are included
    df = df[['Date', 'price'] + X_FEATURES]  # Ensure 'Date' and 'option_id' are included

    # Split the data: save 20% for testing
    df_train, df_test = train_test_split(df, test_size=test_size, shuffle=False)

    # Prepare data with lookback (e.g., 7 days of past data to predict the next day's price)
    X_train, Y_train, _ = prepare_data_for_next_day_prediction(df_train, lookback=lookback)
    X_test, Y_test, price_history_test = prepare_data_for_next_day_prediction(df_test, lookback=lookback)

    # Normalize the features (scale X data)
    scaler_X = MinMaxScaler()

    # Reshape for scaling (scaler expects 2D input, so we flatten the 3D data)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    # Fit and transform training data, and transform testing data
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

    # Normalize the target variable (price) separately
    scaler_Y = MinMaxScaler()

    # Fit and transform the training target data (reshape Y_train to 2D)
    Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1))  # Reshape to (samples, 1)

    # Transform the test target data (reshape Y_test to 2D)
    Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, 1))  # Reshape to (samples, 1)

    # Neural Network Model (LSTM)
    model = Sequential()
    model.add(LSTM(128, input_shape=(lookback, X_train.shape[2])))
    model.add(Dropout(0.3))  # Increased dropout for regularization
    model.add(Dense(64, activation='relu'))  # Added dense layer for better learning
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train_scaled, Y_train_scaled, validation_data=(X_test_scaled, Y_test_scaled), 
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

    # Evaluate the model
    loss = model.evaluate(X_test_scaled, Y_test_scaled, verbose=0)
    print(f"Test Loss (MSE): {loss}")

    # Make predictions on the test data
    Y_pred_scaled = model.predict(X_test_scaled)

    # Denormalize the predicted prices
    Y_pred_real = scaler_Y.inverse_transform(Y_pred_scaled)

    # Denormalize the real test prices
    Y_test_real = scaler_Y.inverse_transform(Y_test_scaled)

    # Calculate MSE for real prices
    mse_results = mean_squared_error(Y_test_real, Y_pred_real)
    print(f"Mean Squared Error on Test Data: {mse_results:.4f}")

    return model, mse_results, Y_test_real, Y_pred_real, price_history_test

# Example usage for predicting with all options
if __name__ == '__main__':
    # Load your dataset
    df = pd.read_csv('data/final_merged_all_data.csv')
    
    # Filter data to include only options with at least 30 rows of data
    valid_option_ids = df['option_id'].value_counts()
    valid_option_ids = valid_option_ids[valid_option_ids >= 30].index.tolist()  # Only keep option_ids with at least 30 rows
    df = df[df['option_id'].isin(valid_option_ids)]

    # Predict for all options and get the MSE results and price history
    model, mse_results, Y_test_real, Y_pred_real, price_history_test = predict_for_all_options(df, lookback=7, test_size=0.2, patience=5, epochs=20, batch_size=64)

    # Print out the MSE result
    print(f"\nFinal MSE on Test Set: {mse_results:.4f}")

    # Print predicted vs real prices with price history
    print("\nPredicted vs Real Prices with Price History on Test Set:")
    for real, pred, history in zip(Y_test_real[:10], Y_pred_real[:10], price_history_test[:10]):
        print(f"Real: {real[0]:.4f}, Predicted: {pred[0]:.4f}, Price History: {history}")
