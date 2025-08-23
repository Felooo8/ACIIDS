# train_model.py
import os, sys
sys.path.insert(0, os.path.abspath("...."))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from config import MODEL_DIR, N_ESTIMATORS
from src.preprocessing.data_preprocessing import preprocess_data
import matplotlib.pyplot as plt

def train_model():
    """
    Train a Random Forest Regressor model and save it to the models directory.
    """
    # Preprocess data
    X_train, X_test, Y_train, Y_test, options_test = preprocess_data()  # Add options_test to keep track of option details

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42)
    model.fit(X_train, Y_train)

    # Predict on the test set
    Y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = mse ** 0.5
    print(f"RMSE: {rmse}")

    # Save the trained model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model_file = os.path.join(MODEL_DIR, 'option_price_model.pkl')
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

    return Y_test, Y_pred, options_test  # Return the test predictions and the options details

if __name__ == "__main__":
    Y_test, Y_pred, options_test = train_model()

    # Plotting function
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
