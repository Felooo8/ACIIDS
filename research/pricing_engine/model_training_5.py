import os, sys
sys.path.insert(0, os.path.abspath("...."))
from src.preprocessing.data_preprocessing_5 import prepare_features_and_labels, clean_and_preprocess
from src.preprocessing.data_loader_5 import load_option_data
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_neural_network(input_shape):
    """
    Build a simple neural network model with improvements.

    Args:
        input_shape (tuple): Shape of the input data (number of features).

    Returns:
        model: Compiled Keras model.
    """
    model = models.Sequential()

    # Add layers to the model
    model.add(layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.3))  # Add dropout to prevent overfitting
    model.add(layers.BatchNormalization())  # Add Batch Normalization for faster convergence

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(1, activation='linear'))  # Output layer (single regression value)

    # Compile the model with both MSE and MAE for tracking
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])

    return model

def train_and_evaluate_model(X_train, X_test, Y_train, Y_test):
    """
    Train the neural network model and evaluate it on the test data.

    Args:
        X_train, X_test, Y_train, Y_test: Training and testing datasets.
    """
    # Get input shape (number of features)
    input_shape = (X_train.shape[1],)

    # Build and compile the model
    model = build_neural_network(input_shape)

    # Early stopping and learning rate reduction on plateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # Train the model
    model.fit(X_train, Y_train, 
              epochs=120, 
              batch_size=32, 
              validation_split=0.2, 
              verbose=1, 
              callbacks=[early_stopping, lr_scheduler])

    # Evaluate the model on the test data
    test_loss, test_mse, test_mae = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Test MSE: {test_mse}, Test MAE: {test_mae}")

    return model

if __name__ == "__main__":
    # Load the data using data_loading_5
    df = load_option_data()

    # Clean and preprocess the data
    cleaned_df = clean_and_preprocess(df, 365, 10, 10)

    # Prepare features and labels for model training
    X_train, X_test, Y_train, Y_test = prepare_features_and_labels(cleaned_df)

    # Train and evaluate the model
    model = train_and_evaluate_model(X_train, X_test, Y_train, Y_test)

    # Save the model
    model.save('models/option_model.keras')
    print("Model saved to models/option_model.keras")
