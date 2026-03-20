# use_model.py
import os, sys
import pandas as pd
sys.path.insert(0, os.path.abspath("...."))
import joblib
import matplotlib.pyplot as plt
from src.preprocessing.data_preprocessing import preprocess_data
from config import MODEL_DIR

def load_model():
    """
    Load the trained Random Forest Regressor model.
    """
    model_file = os.path.join(MODEL_DIR, 'option_price_model.pkl')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found at {model_file}. Please train the model first.")
    
    model = joblib.load(model_file)
    print(f"Model loaded from {model_file}")
    return model

def predict_and_display():
    """
    Load the model, predict on the test set, and display results including option details.
    """
    # Preprocess data to get test set
    X_train, X_test, Y_train, Y_test, options_test = preprocess_data()

    # Load the saved model
    model = load_model()

    # Predict on the test data
    Y_pred = model.predict(X_test)

    # Create a DataFrame to display options with actual and predicted prices
    options_test['Actual Price'] = Y_test
    options_test['Predicted Price'] = Y_pred

    print("\nPredicted Options Details:")
    print(options_test[['date', 'strike', 'type', 'expiration', 'Actual Price', 'Predicted Price']].head())

    # Plotting actual vs predicted prices
    def plot_predictions(Y_test, Y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(Y_test)), Y_test, label='Actual Prices', color='blue', alpha=0.6)
        plt.scatter(range(len(Y_pred)), Y_pred, label='Predicted Prices', color='red', alpha=0.6)
        plt.xlabel('Data Points')
        plt.ylabel('Option Prices')
        plt.title('Actual vs Predicted Option Prices')
        plt.legend()
        plt.show()

    # Call the function to plot
    plot_predictions(Y_test, Y_pred)

if __name__ == "__main__":
    predict_and_display()
