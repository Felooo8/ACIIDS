import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Add your project path
sys.path.insert(0, os.path.abspath("...."))

# Import functions from your modules
from src.preprocessing.data_preprocessing_4 import prepare_features_and_labels, clean_and_preprocess
from src.preprocessing.data_loader_4 import load_option_data
import tensorflow as tf

def load_trained_model(model_path):
    """
    Load the pretrained model from the specified path.

    Args:
        model_path (str): Path to the saved model file (.h5).

    Returns:
        model: Loaded Keras model.
    """
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    return model

def test_model(model, X_test, Y_test):
    """
    Test the trained model and evaluate performance.

    Args:
        model: Trained Keras model.
        X_test: Features for testing.
        Y_test: Actual target values for testing.

    Returns:
        Y_pred: Predicted values from the model.
    """
    # Make predictions using the trained model
    Y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error for the predictions
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse}")

    return Y_pred

def print_comparison_table(X_test, Y_test, Y_pred, options_test):
    """
    Print a comparison table showing the actual and predicted values with option details.

    Args:
        X_test: Test features (for reference).
        Y_test: Actual target values (implied volatility or prices).
        Y_pred: Predicted target values from the model.
        options_test: Option details to include in the table.
    """
    # Prepare DataFrame to hold results
    options_test['Actual Future Volatility'] = Y_test
    options_test['Predicted Future Volatility'] = Y_pred

    # Print the table with relevant details
    print("\nPredicted vs Actual Future Volatilities:")
    print(options_test[['date', 'strike', 'type_put', 'expiration', 'implied_volatility',
                        'Actual Future Volatility', 'Predicted Future Volatility']].head())

if __name__ == "__main__":
    # Load the pretrained model
    model_path = 'models/option_volatility_model.keras'
    model = load_trained_model(model_path)

    # Load the test data
    df = load_option_data()
    cleaned_df = clean_and_preprocess(df)
    X_train, X_test, Y_train, Y_test, options_test = prepare_features_and_labels(cleaned_df, return_options_test=True)

    # Test the model and make predictions
    Y_pred = test_model(model, X_test, Y_test)

    # Print comparison table of actual vs predicted volatility
    print_comparison_table(X_test, Y_test, Y_pred, options_test)
