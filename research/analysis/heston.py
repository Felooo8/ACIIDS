import os
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from pandas.errors import EmptyDataError
from datetime import datetime

# Define bounds for each parameter: (min, max)
bounds = [
    (0.01, 10),  # kappa: positive, reasonable range
    (0.01, 1),   # theta: positive, reasonable range for long-term variance
    (0.01, 2),   # sigma: positive, reasonable range for volatility of variance
    (-1, 1),     # rho: must be between -1 and 1
    (0.01, 1)    # v0: positive, reasonable initial variance
]


class HestonModel:
    def __init__(self, params):
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = params

    def characteristic_function(self, u, S, K, r, T):
        complex_i = 1j
        d = np.sqrt((self.rho * self.sigma * u * complex_i - self.kappa) ** 2 + (u * u * complex_i + u) * self.sigma ** 2)
        g = (self.kappa - self.rho * self.sigma * u * complex_i - d) / (self.kappa - self.rho * self.sigma * u * complex_i + d)
        
        term1 = np.exp(complex_i * u * np.log(S / K) + complex_i * u * r * T)
        term2 = np.exp((self.v0 / self.sigma ** 2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))) * d * u * complex_i)
        term3 = np.exp((self.kappa * self.theta * T / self.sigma ** 2) * (self.kappa - self.rho * self.sigma * u * complex_i - d))
        
        return term1 * term2 * term3

    def option_price(self, S, K, r, T, u_value=0.5):
        return np.real(self.characteristic_function(u_value, S, K, r, T))

    def calculate_volatility(self, T):
        # Using mean reversion and time T to calculate expected variance
        expected_variance_T = self.theta + (self.v0 - self.theta) * np.exp(-self.kappa * T)
        # Return the square root of variance, which is the volatility
        return np.sqrt(expected_variance_T) if expected_variance_T.gt(0).all() else 0.0


def heston_loss_function(params, market_prices, strikes, maturities, S, r):
    heston_model = HestonModel(params)
    total_loss = 0.0
    for i in range(len(market_prices)):
        K = strikes[i]
        T = maturities[i]
        market_price = market_prices[i]
        
        # Estimate option price from Heston model
        estimated_price = heston_model.option_price(S=S, K=K, r=r, T=T)
        total_loss += (market_price - estimated_price) ** 2
    
    return total_loss

def calibrate_heston_parameters(option_data, stock_price, risk_free_rate):
    strikes = option_data['price_strike']
    market_prices = option_data['price']
    maturities = option_data['dte'] / 365  # Convert days to years

    # Initial guess for Heston parameters
    initial_params = [1.0, 0.04, 0.2, -0.7, 0.04]

    # Minimize the loss function
    result = minimize(heston_loss_function, initial_params, args=(market_prices, strikes, maturities, stock_price, risk_free_rate))

    # Return calibrated parameters
    return result.x

def save_parameters_to_csv(params, date, file_name):
    """Save Heston parameters to a CSV file, appending new rows for each date."""
    
    # Ensure date is formatted correctly
    if isinstance(date, pd.Series):
        date = date.iloc[0]  # Extract the first value if it's a series

    new_row = {
        'Date': date,
        'kappa': params[0],
        'theta': params[1],
        'sigma': params[2],
        'rho': params[3],
        'v0': params[4]
    }
    
    # If the CSV file doesn't exist, create it with headers
    if not os.path.exists(file_name):
        df = pd.DataFrame([new_row])
        df.to_csv(file_name, index=False)
    else:
        # Append to the existing CSV without duplicating columns
        df = pd.read_csv(file_name)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(file_name, index=False)


def load_parameters_from_csv(date, file_name):
    """Load Heston parameters for a specific date from a CSV file."""
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        param_row = df[df['Date'] == str(date)]
        if not param_row.empty:
            return param_row.iloc[0, 1:].values.tolist()  # Return the parameters as a list
    return None

def dynamic_recalibration(option_files_folder, risk_free_rate, save_folder, params_file):
    file_list = sorted([f for f in os.listdir(option_files_folder) if f.endswith('.csv')])
    
    for file in file_list:
        file_path = os.path.join(option_files_folder, file)
        try:
            option_data = pd.read_csv(file_path)
        except EmptyDataError:
            print(f"Empty file encountered: {file_path}. Skipping.")
            continue

        # Extract date from filename (you can adjust this to match your filename structure)
        date = pd.to_datetime(option_data['c_date']).dt.strftime('%Y-%m-%d')[0]
        stock_price = option_data['underlying_price'].mean()

        # Check if parameters already exist for this date
        params = load_parameters_from_csv(date, params_file)
        if params:
            print(f"Loaded saved parameters for {date}: {params}")
            heston_model = HestonModel(params)
        else:
            # Calibrate parameters if not available
            calibrated_params = calibrate_heston_parameters(option_data, stock_price, risk_free_rate)
            save_parameters_to_csv(calibrated_params, date, params_file)
            print(f"Calibrated and saved parameters for {date}: {calibrated_params}")
            heston_model = HestonModel(calibrated_params)

        # Add new columns for volatility and option price
        option_data['heston_volatility'] = heston_model.calculate_volatility(option_data['dte'] / 365)
        option_data['heston_price'] = heston_model.option_price(S=stock_price, K=option_data['price_strike'], r=risk_free_rate, T=option_data['dte'] / 365)


        # Save the updated option data with volatility and price
        updated_file_path = os.path.join(save_folder, f"{file}_with_volatility_and_price.csv")
        option_data.to_csv(updated_file_path, index=False)
        print(f"Saved option data with Heston volatilities and prices for {file}.")

if __name__ == "__main__":
    option_files_folder = 'data/options/'
    save_folder = 'data/options/heston/'  # Folder to save recalibration results and option data with volatilities and prices
    params_file = 'data/options/heston/heston_params.csv'  # File to save Heston parameters

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Example risk-free rate
    risk_free_rate = 0.04  # You can dynamically fetch it from financial data APIs

    # Run dynamic recalibration and save results
    dynamic_recalibration(option_files_folder, risk_free_rate, save_folder, params_file)
