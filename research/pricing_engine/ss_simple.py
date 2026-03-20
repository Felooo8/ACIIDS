import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

X_FEATURES = [
    "Close",
    "iv",
]

if __name__ == '__main__':
    # Load your dataset
    df = pd.read_csv('data/final_merged_all_data.csv')

    # Label Encoding for 'call_put' column
    label_encoder = LabelEncoder()
    df['call_put'] = label_encoder.fit_transform(df['call_put'])  # Encode 'C' and 'P' to numerical values
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    
    # Select important features
    features = X_FEATURES  # Your list of important features here
    df = df[['dte', 'price', 'call_put'] + features]  # Ensure the target 'price' is included for prediction

    # Split the data into features (X) and target (Y)
    X = df.drop(columns=['price'])  # Features
    Y = df['price']  # Target variable (option price)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Normalize the data (scale features)
    scaler = MinMaxScaler()

    # Fit and transform training data, and transform testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler to use later in testing
    joblib.dump(scaler, 'scaler.joblib')

    # Neural Network Model (Fully Connected/Feedforward)
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Input layer and first hidden layer
    model.add(Dropout(0.3))  # Dropout to prevent overfitting
    model.add(Dense(64, activation='relu'))  # Second hidden layer
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))  # Third hidden layer
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.02), loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train_scaled, Y_train, validation_data=(X_test_scaled, Y_test), epochs=10, batch_size=64)

    # Evaluate the model
    loss = model.evaluate(X_test_scaled, Y_test)
    print(f"Test Loss (MSE): {loss}")

    # Save the model
    model.save('models/fully_connected_option_pricing_final_simple.keras')
