import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# List of model architectures to test

model_configs = [
    {
        "layers": [
            ("LSTM", 128, False),
            ("Dropout", 0.3),
            ("Dense", 64, "relu"),
            ("Dense", 32, "relu"),
            ("Dense", 1, "linear")
        ],
        "learning_rate": 0.005,
        "batch_size": 32,
        "epochs": 20
    },
    {
        "layers": [
            ("LSTM", 64, False),
            ("Dropout", 0.2),
            ("Dense", 32, "relu"),
            ("Dense", 16, "relu"),
            ("Dense", 1, "linear")
        ],
        "learning_rate": 0.002,
        "batch_size": 32,
        "epochs": 25
    },
    {
        "layers": [
            ("GRU", 64, False),
            ("Dropout", 0.2),
            ("Dense", 64, "relu"),
            ("Dense", 32, "relu"),
            ("Dense", 1, "linear")
        ],
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 30
    },
    {
        "layers": [
            ("GRU", 128, False),
            ("Dropout", 0.25),
            ("Dense", 32, "relu"),
            ("Dense", 1, "linear")
        ],
        "learning_rate": 0.004,
        "batch_size": 32,
        "epochs": 20
    },
    {
        "layers": [
            ("LSTM", 256, False),
            ("Dropout", 0.4),
            ("Dense", 128, "relu"),
            ("Dense", 1, "linear")
        ],
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 20
    },
]

# Load and preprocess the data

def prepare_data_for_next_day_prediction(df, lookback=5):
    X, Y = [], []
    df = df.sort_values('Date')
    for i in range(lookback, len(df)):
        past_data = df.iloc[i-lookback:i][X_FEATURES].values
        next_day_price = df.iloc[i]['price']
        X.append(past_data)
        Y.append(next_day_price)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Assuming df, X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, scaler_X, scaler_Y, df_future, lookback are already defined

# Features to use for prediction
X_FEATURES = [
    'positive_articles', 'dte', 'iv', 
    'volume', 'openinterest', 'underlying_price'
]

X_FEATURES = [
    'positive_articles', 'dte', 'iv', 'underlying_price'
]

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


# Iterate through each model configuration
for idx, config in enumerate(model_configs):
    print(f"\nTraining model {idx + 1}/{len(model_configs)}")
    
    # Build the model
    model = Sequential()
    for layer in config["layers"]:
        if layer[0] == "LSTM":
            model.add(LSTM(layer[1], input_shape=(lookback, X_train_scaled.shape[2]), return_sequences=layer[2]))
        elif layer[0] == "GRU":
            model.add(GRU(layer[1], return_sequences=layer[2]))
        elif layer[0] == "Dropout":
            model.add(Dropout(layer[1]))
        elif layer[0] == "Dense":
            model.add(Dense(layer[1], activation=layer[2]))
    
    # Compile the model
    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train_scaled, Y_train_scaled, 
                        validation_data=(X_test_scaled, Y_test_scaled), 
                        epochs=config["epochs"], batch_size=config["batch_size"], 
                        callbacks=[early_stopping], verbose=0)
    
    # Evaluate the model
    loss = model.evaluate(X_test_scaled, Y_test_scaled, verbose=0)
    print(f"Model {idx + 1} Test Loss (MSE): {loss:.4f}")
    
    # Prepare the data for future predictions
    X_future, Y_future_real = prepare_data_for_next_day_prediction(df_future, lookback=lookback)
    X_future_reshaped = X_future.reshape(-1, X_future.shape[-1])
    X_future_scaled = scaler_X.transform(X_future_reshaped).reshape(X_future.shape)
    
    # Make predictions on the future data
    Y_future_pred_scaled = model.predict(X_future_scaled, verbose=0)
    Y_future_pred_real = scaler_Y.inverse_transform(Y_future_pred_scaled)
    if np.all(Y_future_pred_real == Y_future_pred_real[0]):
        print(f"Mean Squared Error for Future Predictions (Model {idx + 1}): 9999999")
    else:
        # Calculate and print MSE for future predictions
        mse_future = mean_squared_error(Y_future_real, Y_future_pred_real)
        print(f"Mean Squared Error for Future Predictions (Model {idx + 1}): {mse_future:.4f}")
    
    # Save the model
    # model.save(f'models/model_{idx + 1}.keras')
