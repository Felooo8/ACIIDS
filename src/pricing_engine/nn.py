import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

X_FEATURES = [
    "Close",
    "iv",
]


# Data Preparation Function for LSTM/GRU
def create_windowed_data(df, window_size=30, offset=0):
    """
    Create sliding window of data grouped by option_id (each contract) for LSTM/GRU.
    
    Args:
    df (pd.DataFrame): The input DataFrame containing all the data.
    window_size (int): Number of days in the sliding window (default is 30 days).
    offset (int): Offset to shift the window across the data.
    
    Returns:
    X (np.array): Feature data in shape (num_windows, window_size, num_features).
    Y (np.array): Target variable (option price) in shape (num_windows, 1).
    """
    X, Y = [], []
    # Group data by option_id (each unique contract)
    grouped = df.groupby(['option_id', 'call_put'])
    
    # Iterate over each contract (option_id)
    for option_id, group in grouped:
        # Sort by Date to ensure time-series order
        group = group.sort_values('Date')
        # Get unique dates for the current contract
        unique_days = group['Date'].unique()
        # Create sliding windows for each contract (option_id)
        for i in range(offset, min(offset+window_size, len(unique_days))):
            # Get the window of dates (30 unique days)
            window_dates = unique_days[i:i + window_size]
            
            # Filter the data for the current window of this contract
            window_data = group[group['Date'].isin(window_dates)]
            
            # Ensure the window data is reshaped properly (window_size, num_features)
            window_data_filtered = window_data.drop(columns=['Date', 'option_id'])
            
            
            # Pad the data if the window has fewer than `window_size` days
            if window_data_filtered.shape[0] < window_size:
                # Number of missing rows
                missing_rows = window_size - window_data_filtered.shape[0]
                
                # Create zero padding for the missing days
                padding = np.zeros((missing_rows, len(window_data_filtered.columns)))
                
                # Append the padding to the window data
                window_data_filtered = np.vstack([window_data_filtered.values, padding])
            
            # Reshape the data for LSTM/GRU (samples, window_size, num_features)
            window_data_filtered = np.array(window_data_filtered)
            window_reshaped = window_data_filtered.reshape(window_size, len(window_data_filtered[0]))
            X.append(window_reshaped)
            
            # The target is the option price on the last day of the window
            last_day_price = window_data.iloc[-1]['price']
            Y.append(last_day_price)
    
    return np.array(X), np.array(Y)

if __name__ == '__main__':
    # Load your dataset
    df = pd.read_csv('data/final_merged_all_data.csv')

    # Label Encoding for 'call_put' column
    label_encoder = LabelEncoder()
    df['call_put'] = label_encoder.fit_transform(df['call_put'])  # Encode 'C' and 'P' to numerical values
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    # Select important features
    features = X_FEATURES  # Your list of important features here
    df = df[['Date', 'option_id', 'dte', 'price', 'call_put'] + features]  # Ensure 'Date' is included for windowing

    # Create windowed data
    X, Y = create_windowed_data(df, window_size=30)
    df.drop(columns=['Date', 'option_id'], inplace=True)  # Drop non-feature columns
    # Print the columns of X to verify the features
    print("X columns (features):", df.columns.tolist())
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Normalize the data (scale features)
    scaler = MinMaxScaler()

    # Save the feature names
    feature_names = [str(i) for i in range(X_train.shape[-1])]  # Example feature names
    joblib.dump(feature_names, 'feature_names.joblib')  # Saving feature names for later use
    print(feature_names)
    # Reshape for scaling (scaler expects 2D input)
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    # Fit and transform training data, and transform testing data
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train_shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test_shape)

    # Save the scaler to use later in testing
    joblib.dump(scaler, 'scaler.joblib')

    # Neural Network Model (LSTM/GRU)
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))  # Dropout to prevent overfitting
    model.add(GRU(64, return_sequences=False))  # You can stack more GRUs or LSTMs
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output for regression

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.02), loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train_scaled, Y_train, validation_data=(X_test_scaled, Y_test), epochs=10, batch_size=64)

    # Evaluate the model
    loss = model.evaluate(X_test_scaled, Y_test)
    print(f"Test Loss (MSE): {loss}")

    # Save the model
    model.save('models/lstm_gru_option_pricing_final_3.keras')


