# use_model.py
import os, sys
import pandas as pd
sys.path.insert(0, os.path.abspath("...."))
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from src.preprocessing.data_preprocessing_2 import preprocess_data
from config import MODEL_DIR

from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_3d_predictions(options_test):
    """
    Plot a 3D scatter plot to visualize current, predicted, and actual future prices against strike and expiration.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert expiration to numeric (ordinal) for plotting, but keep the original for labels
    expiration_numeric = pd.to_datetime(options_test['expiration']).map(lambda x: x.toordinal())
    expiration_labels = pd.to_datetime(options_test['expiration']).dt.strftime('%Y-%m-%d')  # For axis ticks

    # Plot actual future prices
    ax.scatter(options_test['strike'], expiration_numeric, options_test['Actual Future Price'], 
               c='blue', label='Actual Future Price', marker='o', s=30)

    # Plot predicted future prices
    ax.scatter(options_test['strike'], expiration_numeric, options_test['Predicted Future Price'], 
               c='red', label='Predicted Future Price', marker='^', s=30)

    # Plot current prices
    ax.scatter(options_test['strike'], expiration_numeric, options_test['last'], 
               c='green', label='Current Price (at prediction time)', marker='s', s=30)

    # Set axis labels and title
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Expiration Date')
    ax.set_zlabel('Option Price')
    ax.set_title('3D Scatter Plot of Option Prices: Current, Predicted, and Actual')

    # Set Y-axis ticks to display actual date labels
    ax.set_yticks(expiration_numeric)
    ax.set_yticklabels(expiration_labels, rotation=45)  # Rotate for better readability

    plt.legend()
    plt.show()


# Plotting actual vs predicted future prices
def plot_predictions(Y_test, Y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, Y_pred, label='Predicted vs Actual', color='blue', alpha=0.6)
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')  # Diagonal line
    plt.xlabel('Actual Future Price')
    plt.ylabel('Predicted Future Price')
    plt.title('Predicted vs Actual Future Option Prices')
    plt.legend()
    plt.show()

def load_model():
    """
    Load the trained Random Forest Regressor model.
    """
    model_file = os.path.join(MODEL_DIR, 'option_price_model_2.pkl')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found at {model_file}. Please train the model first.")
    
    model = joblib.load(model_file)
    print(f"Model loaded from {model_file}")
    return model

def predict_and_visualize():
    """
    Load the model, predict on the test set, compare with actual future prices,
    and visualize the prediction performance.
    """
    # Preprocess data to get test set (includes future prices)
    X_train, X_test, Y_train, Y_test, options_test = preprocess_data()

    # Load the saved model
    model = load_model()

    # Predict on the test data
    Y_pred = model.predict(X_test)

    # Create a DataFrame to display options with actual and predicted prices
    options_test['Actual Future Price'] = Y_test
    options_test['Predicted Future Price'] = Y_pred

    # Print out a sample of the predicted vs actual future prices
    print("\nPredicted vs Actual Future Prices:")
    print(options_test[['date', 'strike', 'type', 'expiration', 'last', 'Actual Future Price', 'Predicted Future Price']].head())

    # Calculate RMSE
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = mse ** 0.5
    print(f"\nRMSE: {rmse}")


    # Call the function to plot
    plot_predictions(Y_test, Y_pred)
    # Assuming options_test is available from your model
    plot_3d_predictions(options_test)

    import matplotlib.pyplot as plt

if __name__ == "__main__":
    predict_and_visualize()

