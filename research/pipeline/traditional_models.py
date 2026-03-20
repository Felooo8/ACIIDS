import QuantLib as ql
import numpy as np
from arch import arch_model
import pandas as pd

def calculate_garch_volatility(data, column='UNDERLYING_LAST'):
    """
    Calculate GARCH(1,1) volatility for a given time series column.
    :param data: DataFrame containing the time series data.
    :param column: The column for which to calculate GARCH volatility.
    :return: A Series containing GARCH volatility values.
    """
    # Fit GARCH(1,1) model
    returns = np.log(data[column] / data[column].shift(1)).dropna()  # Calculate log returns
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp="off")

    # Predict conditional volatility
    conditional_volatility = garch_fit.conditional_volatility
    volatility_series = pd.Series(conditional_volatility, index=returns.index)

    # Reindex to match the original data
    volatility_series = volatility_series.reindex(data.index).fillna(method='ffill').fillna(0)

    return volatility_series


def heston_model_price(S, K, T, r, sigma, kappa, theta, xi, rho, option_type=1):
    """
    Calculate option price using the Heston model.
    """
    # Set evaluation date
    evaluation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = evaluation_date

    epsilon = 1e-8
    sigma = max(sigma, epsilon)
    T = max(T, epsilon)
    # Option parameters   
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == 1 else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(evaluation_date + int(T * 365))

    # Market data
    risk_free_rate = ql.FlatForward(evaluation_date, r, ql.Actual365Fixed())
    dividend_rate = ql.FlatForward(evaluation_date, 0.0, ql.Actual365Fixed())
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))

    # Heston process
    v0 = sigma**2  # Initial volatility
    heston_process = ql.HestonProcess(
        ql.YieldTermStructureHandle(risk_free_rate),
        ql.YieldTermStructureHandle(dividend_rate),
        spot_handle,
        v0,
        kappa,
        theta,
        xi,
        rho
    )

    # Heston model
    model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(model)

    # Option
    european_option = ql.VanillaOption(payoff, exercise)
    european_option.setPricingEngine(engine)

    # Price
    price = european_option.NPV()
    return price

def binomial_option_price(S, K, T, r, sigma, N=100, option_type=1):
    """
    Calculate option price using the Binomial Option Pricing model.
    """
    epsilon = 1e-8
    sigma = max(sigma, epsilon)
    T = max(T, epsilon)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))      # Up factor
    d = 1 / u                            # Down factor
    p = (np.exp(r * dt) - d) / (u - d)   # Risk-neutral probability

    # Initialize asset prices at maturity
    asset_prices = np.zeros(N+1)
    asset_prices[0] = S * d**N
    for i in range(1, N+1):
        asset_prices[i] = asset_prices[i-1] * u / d

    # Initialize option values at maturity
    option_values = np.zeros(N+1)
    if option_type == 1:
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)

    # Backward induction
    for i in range(N-1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])

    return option_values[0]

def monte_carlo_option_price(S, K, T, r, sigma, option_type=1, num_simulations=10000):
    """
    Calculate option price using Monte Carlo simulation.
    """
    # Simulate end-of-period asset price
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type == 1:
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type=1):
    """
    Calculate Black-Scholes option price.
    """
    
    epsilon = 1e-8
    sigma = max(sigma, epsilon)
    T = max(T, epsilon)

    d1 = (np.log(S / K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 1:
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

class BlackScholesModel:
    def predict(self, input_data):
        # Extract required parameters from input_data
        # Assuming input_data is of shape (batch_size, num_features)
        S = input_data[self.features.index('UNDERLYING_LAST')]
        K = input_data[self.features.index('STRIKE')]
        T = input_data[self.features.index('dte')] / 252
        sigma = input_data[self.features.index('IV')]
        is_call = input_data[self.features.index('is_call')]
        r = 0.04  # Risk-free rate


        price = black_scholes(S, K, T, r, sigma, is_call)
        return price

# Constants
SEQUENCE_LENGTH = 10
TEST_SIZE = 0.2  # 20% of data for future use
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.0001

TARGET_INDEX = 0
FIXED_COST = 0.20          # $0.70 per contract
PROPORTIONAL_COST = 0.005  # 0.5% of current price

# Compute traditional model prices on validation data
R = 0.04  # Risk-free interest rate
KAPPA = 4.0        # Mean reversion rate
THETA = 0.03       # Long-term variance
XI = 0.2           # Volatility of volatility
RHO = -0.1         # Correlation between the underlying asset and its volatility


class BlackScholesEnvironment:
    def __init__(self, data_scaled, data_unscaled, features, scaler, initial_cash=1000):
        self.data_scaled = data_scaled.reset_index(drop=True)
        self.data_unscaled = data_unscaled.reset_index(drop=True)
        self.features = features
        self.scaler = scaler
        self.initial_cash = initial_cash
        self.sequence_length = SEQUENCE_LENGTH
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0  # 0 or 1
        self.total_asset = self.cash
        self.trade_history = []
        self.profits = []
        # Initialize sequence buffer
        self.sequence_scaled = []
        for i in range(self.sequence_length):
            self.sequence_scaled.append(self.data_scaled.loc[self.current_step + i, self.features].values)
        state = self._get_state()
        return state
    
    def get_current_price(self):
        # Get current price (unscaled)
        current_price_scaled = self.sequence_scaled[-1][TARGET_INDEX]
        current_price_full = np.zeros((1, len(self.features)))
        current_price_full[0, TARGET_INDEX] = current_price_scaled
        current_price = self.scaler.inverse_transform(current_price_full)[0, TARGET_INDEX]
        return current_price

    def get_predicted_price(self):
        # Get predicted price (unscaled)
        state = self._get_state()
        predicted_price = state[0, -1]
        return predicted_price

    def _get_state(self):
        # Black-Scholes price prediction
        current_features_scaled = self.sequence_scaled[-1]
        current_features_unscaled = self.scaler.inverse_transform(current_features_scaled.reshape(1, -1))[0]

        underlying_price = current_features_unscaled[self.features.index("UNDERLYING_LAST")]
        strike_price = current_features_unscaled[self.features.index("STRIKE")]
        time_to_maturity = current_features_unscaled[self.features.index("dte")] / 252  # Convert to years
        volatility = current_features_unscaled[self.features.index("IV")]
        is_call = int(current_features_unscaled[self.features.index("is_call")]) == 1
        risk_free_rate = 0.04  # Constant risk-free rate

        predicted_price_unscaled = black_scholes(
            underlying_price, strike_price, time_to_maturity, risk_free_rate, volatility, is_call
        )

        # State includes unscaled features and Black-Scholes predicted price
        state = np.concatenate([current_features_unscaled, [predicted_price_unscaled]])
        state = state.reshape(1, -1)
        return state

    def step(self, action):
        done = False

        # Get current and next actual prices (unscaled)
        current_price_scaled = self.sequence_scaled[-1][TARGET_INDEX]
        current_price_full = np.zeros((1, len(self.features)))
        current_price_full[0, TARGET_INDEX] = current_price_scaled
        current_price = self.scaler.inverse_transform(current_price_full)[0, TARGET_INDEX]

        if self.current_step + self.sequence_length >= len(self.data_unscaled) - 1:
            done = True
            next_price = current_price
        else:
            next_price_scaled = self.data_scaled.loc[self.current_step + self.sequence_length, self.features].values[TARGET_INDEX]
            next_price_full = np.zeros((1, len(self.features)))
            next_price_full[0, TARGET_INDEX] = next_price_scaled
            next_price = self.scaler.inverse_transform(next_price_full)[0, TARGET_INDEX]

        prev_total_asset = self.cash + self.position * current_price

        # Action Logic
        cost = 0
        if action == 1:  # Buy one unit
            if self.cash >= current_price and self.position == 0:
                self.position = 1
                self.cash -= current_price
                cost = FIXED_COST + PROPORTIONAL_COST * current_price
                self.cash -= cost
                self.trade_history.append({'step': self.current_step, 'action': 'buy', 'price': current_price, 'cash': self.cash, 'position': self.position, 'cost': cost})
        elif action == 2:  # Sell one unit
            if self.position == 1:
                self.position = 0
                self.cash += current_price
                cost = FIXED_COST + PROPORTIONAL_COST * current_price
                self.cash -= cost
                self.trade_history.append({'step': self.current_step, 'action': 'sell', 'price': current_price, 'cash': self.cash, 'position': self.position, 'cost': cost})
        elif action == 0:  # Hold
            pass  # Do nothing

        # Update total asset value after action
        total_asset = self.cash + self.position * next_price

        # Reward is the change in total asset
        reward = total_asset - prev_total_asset

        # Record profit
        self.profits.append(total_asset - self.initial_cash)

        # Move to the next step
        self.current_step += 1

        # Update sequence buffer
        if not done:
            self.sequence_scaled.pop(0)
            self.sequence_scaled.append(self.data_scaled.loc[self.current_step + self.sequence_length - 1, self.features].values)
            next_state = self._get_state()
        else:
            next_state = np.zeros((1, len(self.features) + 1))  # State size is features + Black-Scholes price

        return next_state, reward, done, {}


class HestonEnvironment:
    def __init__(self, data_scaled, data_unscaled, features, scaler, initial_cash=1000):
        self.data_scaled = data_scaled.reset_index(drop=True)
        self.data_unscaled = data_unscaled.reset_index(drop=True)
        self.features = features
        self.scaler = scaler
        self.initial_cash = initial_cash
        self.sequence_length = SEQUENCE_LENGTH
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0
        self.total_asset = self.cash
        self.trade_history = []
        self.profits = []
        self.sequence_scaled = []
        for i in range(self.sequence_length):
            self.sequence_scaled.append(self.data_scaled.loc[self.current_step + i, self.features].values)
        state = self._get_state()
        return state

    def _get_state(self):
        current_features_scaled = self.sequence_scaled[-1]
        current_features_unscaled = self.scaler.inverse_transform(current_features_scaled.reshape(1, -1))[0]

        underlying_price = current_features_unscaled[self.features.index("UNDERLYING_LAST")]
        strike_price = current_features_unscaled[self.features.index("STRIKE")]
        time_to_maturity = current_features_unscaled[self.features.index("dte")] / 252
        volatility = current_features_unscaled[self.features.index("IV")]
        is_call = int(current_features_unscaled[self.features.index("is_call")]) == 1
        risk_free_rate = 0.04

        predicted_price_unscaled = heston_model_price(
            underlying_price, strike_price, time_to_maturity, risk_free_rate, volatility, KAPPA, THETA, XI, RHO, is_call
        )

        state = np.concatenate([current_features_unscaled, [predicted_price_unscaled]])
        state = state.reshape(1, -1)
        return state

    def step(self, action):
        done = False

        # Get current and next actual prices (unscaled)
        current_price_scaled = self.sequence_scaled[-1][TARGET_INDEX]
        current_price_full = np.zeros((1, len(self.features)))
        current_price_full[0, TARGET_INDEX] = current_price_scaled
        current_price = self.scaler.inverse_transform(current_price_full)[0, TARGET_INDEX]

        if self.current_step + self.sequence_length >= len(self.data_unscaled) - 1:
            done = True
            next_price = current_price
        else:
            next_price_scaled = self.data_scaled.loc[self.current_step + self.sequence_length, self.features].values[TARGET_INDEX]
            next_price_full = np.zeros((1, len(self.features)))
            next_price_full[0, TARGET_INDEX] = next_price_scaled
            next_price = self.scaler.inverse_transform(next_price_full)[0, TARGET_INDEX]

        prev_total_asset = self.cash + self.position * current_price
        cost = 0
        # Action Logic
        if action == 1:  # Buy one unit
            if self.cash >= current_price and self.position == 0:
                self.position = 1
                self.cash -= current_price
                cost = FIXED_COST + PROPORTIONAL_COST * current_price
                self.cash -= cost
                self.trade_history.append({'step': self.current_step, 'action': 'buy', 'price': current_price, 'cash': self.cash, 'position': self.position, 'cost': cost})
        elif action == 2:  # Sell one unit
            if self.position == 1:
                self.position = 0
                self.cash += current_price
                cost = FIXED_COST + PROPORTIONAL_COST * current_price
                self.cash -= cost
                self.trade_history.append({'step': self.current_step, 'action': 'sell', 'price': current_price, 'cash': self.cash, 'position': self.position, 'cost': cost})
        elif action == 0:  # Hold
            pass  # Do nothing

        # Update total asset value after action
        total_asset = self.cash + self.position * next_price

        # Reward is the change in total asset
        reward = total_asset - prev_total_asset

        # Record profit
        self.profits.append(total_asset - self.initial_cash)

        # Move to the next step
        self.current_step += 1

        # Update sequence buffer
        if not done:
            self.sequence_scaled.pop(0)
            self.sequence_scaled.append(self.data_scaled.loc[self.current_step + self.sequence_length - 1, self.features].values)
            next_state = self._get_state()
        else:
            next_state = np.zeros((1, len(self.features) + 1))  # State size is features + Black-Scholes price

        return next_state, reward, done, {}

    def get_current_price(self):
        # Get current price (unscaled)
        current_price_scaled = self.sequence_scaled[-1][TARGET_INDEX]
        current_price_full = np.zeros((1, len(self.features)))
        current_price_full[0, TARGET_INDEX] = current_price_scaled
        current_price = self.scaler.inverse_transform(current_price_full)[0, TARGET_INDEX]
        return current_price

    def get_predicted_price(self):
        # Get predicted price (unscaled)
        state = self._get_state()
        predicted_price = state[0, -1]
        return predicted_price

class MonteCarloEnvironment:
    def __init__(self, data_scaled, data_unscaled, features, scaler, initial_cash=1000):
        self.data_scaled = data_scaled.reset_index(drop=True)
        self.data_unscaled = data_unscaled.reset_index(drop=True)
        self.features = features
        self.scaler = scaler
        self.initial_cash = initial_cash
        self.sequence_length = SEQUENCE_LENGTH
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0
        self.total_asset = self.cash
        self.trade_history = []
        self.profits = []
        self.sequence_scaled = []
        for i in range(self.sequence_length):
            self.sequence_scaled.append(self.data_scaled.loc[self.current_step + i, self.features].values)
        state = self._get_state()
        return state

    def _get_state(self):
        current_features_scaled = self.sequence_scaled[-1]
        current_features_unscaled = self.scaler.inverse_transform(current_features_scaled.reshape(1, -1))[0]

        underlying_price = current_features_unscaled[self.features.index("UNDERLYING_LAST")]
        strike_price = current_features_unscaled[self.features.index("STRIKE")]
        time_to_maturity = current_features_unscaled[self.features.index("dte")] / 252
        volatility = current_features_unscaled[self.features.index("IV")]
        is_call = int(current_features_unscaled[self.features.index("is_call")]) == 1
        risk_free_rate = 0.04

        predicted_price_unscaled = monte_carlo_option_price(
            underlying_price, strike_price, time_to_maturity, risk_free_rate, volatility, is_call
        )

        state = np.concatenate([current_features_unscaled, [predicted_price_unscaled]])
        state = state.reshape(1, -1)
        return state


    def step(self, action):
        done = False

        # Get current and next actual prices (unscaled)
        current_price_scaled = self.sequence_scaled[-1][TARGET_INDEX]
        current_price_full = np.zeros((1, len(self.features)))
        current_price_full[0, TARGET_INDEX] = current_price_scaled
        current_price = self.scaler.inverse_transform(current_price_full)[0, TARGET_INDEX]

        if self.current_step + self.sequence_length >= len(self.data_unscaled) - 1:
            done = True
            next_price = current_price
        else:
            next_price_scaled = self.data_scaled.loc[self.current_step + self.sequence_length, self.features].values[TARGET_INDEX]
            next_price_full = np.zeros((1, len(self.features)))
            next_price_full[0, TARGET_INDEX] = next_price_scaled
            next_price = self.scaler.inverse_transform(next_price_full)[0, TARGET_INDEX]

        prev_total_asset = self.cash + self.position * current_price
        cost = 0
        # Action Logic
        if action == 1:  # Buy one unit
            if self.cash >= current_price and self.position == 0:
                self.position = 1
                self.cash -= current_price
                cost = FIXED_COST + PROPORTIONAL_COST * current_price
                self.cash -= cost
                self.trade_history.append({'step': self.current_step, 'action': 'buy', 'price': current_price, 'cash': self.cash, 'position': self.position, 'cost': cost})
        elif action == 2:  # Sell one unit
            if self.position == 1:
                self.position = 0
                self.cash += current_price
                cost = FIXED_COST + PROPORTIONAL_COST * current_price
                self.cash -= cost
                self.trade_history.append({'step': self.current_step, 'action': 'sell', 'price': current_price, 'cash': self.cash, 'position': self.position, 'cost': cost})
        elif action == 0:  # Hold
            pass  # Do nothing

        # Update total asset value after action
        total_asset = self.cash + self.position * next_price

        # Reward is the change in total asset
        reward = total_asset - prev_total_asset

        # Record profit
        self.profits.append(total_asset - self.initial_cash)

        # Move to the next step
        self.current_step += 1

        # Update sequence buffer
        if not done:
            self.sequence_scaled.pop(0)
            self.sequence_scaled.append(self.data_scaled.loc[self.current_step + self.sequence_length - 1, self.features].values)
            next_state = self._get_state()
        else:
            next_state = np.zeros((1, len(self.features) + 1))  # State size is features + Black-Scholes price

        return next_state, reward, done, {}

    def get_current_price(self):
        # Get current price (unscaled)
        current_price_scaled = self.sequence_scaled[-1][TARGET_INDEX]
        current_price_full = np.zeros((1, len(self.features)))
        current_price_full[0, TARGET_INDEX] = current_price_scaled
        current_price = self.scaler.inverse_transform(current_price_full)[0, TARGET_INDEX]
        return current_price

    def get_predicted_price(self):
        # Get predicted price (unscaled)
        state = self._get_state()
        predicted_price = state[0, -1]
        return predicted_price

