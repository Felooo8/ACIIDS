import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import norm

# Add your project path
sys.path.insert(0, os.path.abspath("...."))

# Import functions from your modules
from src.preprocessing.data_preprocessing_5 import prepare_features_and_labels, clean_and_preprocess
from src.preprocessing.data_loader_5 import load_option_data
import tensorflow as tf

# Define constants
RISK_FREE_RATE = 4.72  # Approximate U.S. 10-Year Treasury rate (you can update this dynamically)

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes option price.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Implied volatility
        option_type: 'call' or 'put'
    
    Returns:
        float: The calculated option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

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
    print(f"Neural Network Test RMSE: {rmse}")

    return Y_pred

def calculate_black_scholes_predictions(options_test):
    """
    Calculate option prices using the Black-Scholes model for comparison.
    
    Args:
        options_test (pd.DataFrame): Test data containing option details.
    
    Returns:
        np.ndarray: Black-Scholes predicted prices.
    """
    bs_prices = []
    
    for _, row in options_test.iterrows():
        S = row['Close']  # Stock price from the test set
        K = row['strike']
        T = row['days_to_expiration'] / 365  # Convert days to years
        sigma = row['implied_volatility']  # Use implied volatility as a proxy
        option_type = 'call' if row['type_put'] == 0 else 'put'
        
        price = black_scholes(S, K, T, RISK_FREE_RATE, sigma, option_type)
        bs_prices.append(price)
    
    return np.array(bs_prices)

def calculate_rmse_nrmse(Y_test, Y_pred):
    """
    Calculate RMSE and normalized RMSE (nRMSE).

    Args:
        Y_test: Actual values.
        Y_pred: Predicted values.

    Returns:
        rmse: Root Mean Squared Error.
        nrmse: Normalized RMSE (nRMSE).
    """
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    nrmse = rmse / (Y_test.max() - Y_test.min())  # Normalized RMSE using the range of actual values
    return rmse, nrmse

def print_comparison_table(Y_test, Y_pred, bs_pred, options_test):
    """
    Print a comparison table showing the actual, predicted values from NN, and Black-Scholes prices.

    Args:
        Y_test: Actual target values (option prices).
        Y_pred: Predicted target values from the model.
        bs_pred: Black-Scholes predicted option prices.
        options_test: Option details to include in the table.
    """
    # Prepare DataFrame to hold results
    options_test['Actual Option Price'] = Y_test
    options_test['Predicted Option Price'] = Y_pred
    options_test['Black-Scholes Price'] = bs_pred

    # Print the table with relevant details
    print("\nPredicted vs Actual Option Prices (NN and Black-Scholes):")
    print(options_test[['date', 'strike', 'type_put', 'expiration', 'last_price',
                        'Actual Option Price', 'Predicted Option Price', 'Black-Scholes Price']].head())

if __name__ == "__main__":
    # Load the pretrained model
    model_path = 'models/option_model.keras'
    model = load_trained_model(model_path)

    # Load the test data
    df = load_option_data()
    cleaned_df = clean_and_preprocess(df)
    X_train, X_test, Y_train, Y_test, options_test = prepare_features_and_labels(cleaned_df, return_options_test=True)

    # Test the model and make predictions
    Y_pred = test_model(model, X_test, Y_test)

    # Calculate Black-Scholes prices for comparison
    bs_pred = calculate_black_scholes_predictions(options_test)

    # Print comparison table of actual vs predicted option prices and Black-Scholes prices
    print_comparison_table(Y_test, Y_pred, bs_pred, options_test)

    # Calculate RMSE and nRMSE for Neural Network predictions
    nn_rmse, nn_nrmse = calculate_rmse_nrmse(Y_test, Y_pred)
    print(f"Neural Network RMSE: {nn_rmse}, nRMSE: {nn_nrmse}")

    # Calculate RMSE and nRMSE for Black-Scholes predictions
    bs_rmse, bs_nrmse = calculate_rmse_nrmse(Y_test, bs_pred)
    print(f"Black-Scholes RMSE: {bs_rmse}, nRMSE: {bs_nrmse}")
