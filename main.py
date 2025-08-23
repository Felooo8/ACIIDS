import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
import pickle
import random

from traditinal_models import (HestonEnvironment, MonteCarloEnvironment, BlackScholesEnvironment,
                               FIXED_COST, PROPORTIONAL_COST)
from ppo_model import PPO, PPOTRadingEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
from helpers import load_and_preprocess_data

SEED = 8
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PLOT_COLORS = {
    'LSTM': (0, 0.447, 0.741),
    'Black-Scholes': (0.85, 0.325, 0.098),
    'Heston': (0.466, 0.674, 0.188),
    'Monte Carlo': (0.494, 0.184, 0.556),
    'PPO': (0.929, 0.694, 0.125),
}

PLOT_COLORS_rgb = {
    'blue': (0, 0.447, 0.741),
    'red': (0.85, 0.325, 0.098),
    'yellow': (0.929, 0.694, 0.125),
    'purple': (0.494, 0.184, 0.556),
    'green': (0.466, 0.674, 0.188),
    'light_blue': (0.301, 0.745, 0.933),
    'dark_red': (0.635, 0.078, 0.184)
}
# Constants
SEQUENCE_LENGTH = 10
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.0001

TARGET_INDEX = 0

min_max_scaler = MinMaxScaler()

class TradingEnvironment:
    def __init__(self, data_scaled, data_unscaled, model, features, scaler, initial_cash=1000):
        self.data_scaled = data_scaled.reset_index(drop=True)
        self.data_unscaled = data_unscaled.reset_index(drop=True)
        self.model = model
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
    def get_current_price(self):
        current_price_scaled = self.sequence_scaled[-1][TARGET_INDEX]
        current_price_full = np.zeros((1, len(self.features)))
        current_price_full[0, TARGET_INDEX] = current_price_scaled
        current_price = self.scaler.inverse_transform(current_price_full)[0, TARGET_INDEX]
        return current_price

    def get_predicted_price(self):
        state = self._get_state()
        predicted_price = state[0, -1]
        return predicted_price

    def _get_state(self):
        seq_scaled = np.array(self.sequence_scaled)
        seq_scaled = seq_scaled.reshape(1, self.sequence_length, len(self.features))
        seq_scaled_tensor = tf.convert_to_tensor(seq_scaled, dtype=tf.float32)
        predicted_price_scaled = self.model.predict(seq_scaled_tensor, verbose=0)[0, 0]

        predicted_price_full = np.zeros((1, len(self.features)))
        predicted_price_full[0, TARGET_INDEX] = predicted_price_scaled
        predicted_price_unscaled = self.scaler.inverse_transform(predicted_price_full)[0, TARGET_INDEX]

        current_features_scaled = self.sequence_scaled[-1].reshape(1, -1)
        current_features_unscaled = self.scaler.inverse_transform(current_features_scaled)[0]

        state = np.concatenate([current_features_unscaled, [predicted_price_unscaled]])
        state = state.reshape(1, -1)
        return state

    def step(self, action):
        done = False

        current_price_scaled = self.sequence_scaled[-1][TARGET_INDEX]
        min_price = self.scaler.data_min_[TARGET_INDEX]
        max_price = self.scaler.data_max_[TARGET_INDEX]
        current_price = current_price_scaled * (max_price - min_price) + min_price

        if self.current_step + self.sequence_length >= len(self.data_unscaled) - 1:
            done = True
            next_price = current_price
        else:
            next_price_scaled = self.data_scaled.loc[self.current_step + self.sequence_length, self.features].values[TARGET_INDEX]
            next_price = next_price_scaled * (max_price - min_price) + min_price

        prev_total_asset = self.cash + self.position * current_price
        cost = 0
        
        if action == 1:  # Buy: Set position to +1
            if self.position != 1 and self.cash >= current_price * (1 + abs(self.position)):
                self.cash += self.position * current_price
                self.position = 1
                self.cash -= current_price * self.position
                cost = FIXED_COST + PROPORTIONAL_COST * current_price
                self.cash -= cost
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'price': current_price,
                    'cash': self.cash,
                    'position': self.position,
                    'cost': cost
                })
        elif action == 2:  # Sell: Set position to -1
            if self.position != -1 and self.cash >= current_price * (1 + abs(self.position)):
                self.cash += self.position * current_price
                self.position = -1
                self.cash -= current_price * self.position
                cost = FIXED_COST + PROPORTIONAL_COST * current_price
                self.cash -= cost
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'cash': self.cash,
                    'position': self.position,
                    'cost': cost
                })
        elif action == 0:  # Hold
            pass

        total_asset = self.cash + self.position * next_price

        reward = total_asset - prev_total_asset

        self.profits.append(total_asset - self.initial_cash)

        self.current_step += 1

        if not done:
            self.sequence_scaled.pop(0)
            self.sequence_scaled.append(self.data_scaled.loc[self.current_step + self.sequence_length - 1, self.features].values)
            next_state = self._get_state()
        else:
            next_state = np.zeros((1, len(self.features) + 1))

        return next_state, reward, done, {}


class QLearningAgent:
    def __init__(self, state_bins, action_size, learning_rate=0.0001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.9, save_path="q_table_final.pkl"):
        self.state_bins = state_bins
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

        self.save_path = save_path
        self.q_table = self._load_q_table()

    def discretize_state(self, state):
        price_difference = state[1] - state[0]

        max_diff = 10
        normalized_state = np.clip(price_difference / max_diff, -1, 1)
        discrete_state = int((normalized_state + 1) / 2 * (self.state_bins - 1))
        return discrete_state

    def act(self, state):
        discrete_state = self.discretize_state(state)
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_size)
            return action
        action = np.argmax(self.q_table[discrete_state])
        return action

    def _load_q_table(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                print(f"Loading Q-table from {self.save_path}")
                return pickle.load(f)
        else:
            print("Initializing new Q-table")
            return np.zeros((self.state_bins, self.action_size))

    def save_q_table(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self.q_table, f)
            print(f"Q-table saved to {self.save_path}")


    def learn(self, state, action, reward, next_state, done):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        q_current = self.q_table[discrete_state, action]
        q_next = 0 if done else np.max(self.q_table[discrete_next_state])
        q_target = reward + self.discount_factor * q_next

        self.q_table[discrete_state, action] += self.learning_rate * (q_target - q_current)

        if done:
            self.exploration_rate = max(0.1, self.exploration_rate * self.exploration_decay)
            

# Constants
SEQUENCE_LENGTH = 10
INITIAL_CASH = 1000
EPISODES = 50
RUNS = 3
ID = 14
def main():
    # Load and Preprocess Data
    market_data_path = 'enriched_dataset.csv'
    filtered_data_path = 'filtered_dataset.csv'
    data, features, target, data_unscaled = load_and_preprocess_data(market_data_path)

    option_id_counts = data['option_id'].value_counts()
    valid_option_ids = option_id_counts.nlargest(1).index.tolist()
    print(f"Selected top {len(valid_option_ids)} option_ids with most data points: {valid_option_ids}")

    model_path = 'models/best_model.keras'
    model = tf.keras.models.load_model(model_path)

    filtered_data = data[data['option_id'].isin(valid_option_ids)]
    filtered_data.to_csv(filtered_data_path, index=False)
    data = pd.read_csv(filtered_data_path)

    # Prepare Data Structures for Storing Results
    cumulative_rewards_across_options = {m: [] for m in PLOT_COLORS.keys()}
    total_profits_across_options = {m: [] for m in PLOT_COLORS.keys()}

    # Define Models/Agents (including PPO)
    models_dict = {
        'LSTM': {
            'agent_class': QLearningAgent,
            'environment_class': lambda data_scaled, data_unscaled, features, scaler: TradingEnvironment(
                data_scaled, data_unscaled, model, features, scaler
            ),
            'is_ppo': False
        },
        'Black-Scholes': {
            'agent_class': QLearningAgent,
            'environment_class': BlackScholesEnvironment,
            'is_ppo': False
        },
        'Heston': {
            'agent_class': QLearningAgent,
            'environment_class': HestonEnvironment,
            'is_ppo': False
        },
        'Monte Carlo': {
            'agent_class': QLearningAgent,
            'environment_class': MonteCarloEnvironment,
            'is_ppo': False
        },
        'PPO': {
            'agent_class': None,
            'environment_class': PPOTRadingEnv,
            'is_ppo': True
        }
    }

    # Main Loop over Option IDs
    for option_id in valid_option_ids:
        print(f"\nProcessing Option ID: {option_id}")

        option_data = data[data['option_id'] == option_id].reset_index(drop=True)
        option_data_unscaled = data_unscaled[data_unscaled['option_id'] == option_id].reset_index(drop=True)
        
        split_index = int(len(option_data) * 0.6)
        training_data = option_data.iloc[:split_index].reset_index(drop=True)
        testing_data = option_data.iloc[split_index:].reset_index(drop=True)

        training_data_unscaled = option_data_unscaled.iloc[:split_index].reset_index(drop=True)
        testing_data_unscaled = option_data_unscaled.iloc[split_index:].reset_index(drop=True)

        scaler = MinMaxScaler()
        scaler.fit(training_data[features])

        training_data[features] = scaler.transform(training_data[features])
        testing_data[features] = scaler.transform(testing_data[features])

        training_results = {m: [] for m in models_dict.keys()}
        testing_results = {m: [] for m in models_dict.keys()}
        cumulative_returns_across_options_per_option = {m: [] for m in models_dict.keys()}
        all_trade_points = []

        # Multiple Runs
        for run in range(RUNS):
            print(f"\nRun {run+1}/{RUNS} for Option ID {option_id}")

            for model_name, model_info in models_dict.items():
                print(f"\nTraining {model_name} on Option ID {option_id}, Run {run+1}")

                if model_info['is_ppo']:
                    env_train = model_info['environment_class'](
                        training_data,
                        training_data_unscaled,
                        features,
                        scaler,
                        initial_cash=INITIAL_CASH,
                        sequence_length=SEQUENCE_LENGTH
                    )
                    ppo_agent = PPO(
                        "MlpPolicy",
                        env_train,
                        verbose=0,
                        batch_size=256,
                        learning_rate=0.001,
                        gamma=0.9,
                        ent_coef=0.03,
                        n_steps=(len(training_data)),
                        n_epochs=(EPISODES)
                    )
                    ppo_agent.learn(total_timesteps=int(len(training_data)*(EPISODES)), progress_bar=True)

                    train_episode_rewards = getattr(env_train, "episode_rewards", [])
                    if not train_episode_rewards:
                        train_episode_rewards = [getattr(env_train, "total_reward", 0.0)]

                    training_results[model_name].append(train_episode_rewards)

                    # Testing Phase (PPO)
                    print(f"\nTesting {model_name} on Option ID {option_id}, Run {run+1}")
                    env_test = model_info['environment_class'](
                        testing_data,
                        testing_data_unscaled,
                        features,
                        scaler,
                        initial_cash=INITIAL_CASH,
                        sequence_length=SEQUENCE_LENGTH
                    )
                    obs, _ = env_test.reset()
                    done = False
                    total_profit = 0
                    cumulative_rewards = []

                    while not done:
                        action, _states = ppo_agent.predict(obs)
                        obs, reward, done, _, _ = env_test.step(action)
                        total_profit += reward
                        cumulative_rewards.append(total_profit)

                    testing_results[model_name].append(cumulative_rewards)
                    print(f"{model_name} - Option ID {option_id}, Total Profit: {total_profit:.2f}")

                    dates = testing_data_unscaled['QUOTE_DATE'].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
                    if len(cumulative_rewards) > len(dates):
                        cumulative_rewards = cumulative_rewards[:len(dates)]
                    else:
                        dates = dates[:len(cumulative_rewards)]

                    initial_cash = 1000
                    cum_perc_returns = (np.array(cumulative_rewards) / initial_cash) * 100
                    df = pd.DataFrame({
                        'Date': dates,
                        'CumulativePercentageReturn': cum_perc_returns
                    })
                    cumulative_returns_across_options_per_option[model_name].append(df)
                    total_profits_across_options[model_name].append(total_profit)

                else:
                    # Q-Learning Approach
                    agent_class = model_info['agent_class']
                    environment_class = model_info['environment_class']

                    agent = agent_class(
                        state_bins=50,
                        action_size=3,
                        save_path=f"agents/q_table_{model_name}_{ID}.pkl"
                    )
                    agent.exploration_rate = 1.0

                    episode_profits = []
                    for e in range(EPISODES):
                        if callable(environment_class):
                            env = environment_class(training_data, training_data_unscaled, features, scaler)
                        else:
                            env = environment_class(training_data, training_data_unscaled, features, scaler)

                        state = env.reset()
                        done = False
                        episode_profit = 0

                        while not done:
                            current_price = env.get_current_price()
                            predicted_price = env.get_predicted_price()
                            state = np.array([current_price, predicted_price])

                            action = agent.act(state)
                            next_state_raw, reward, done, _ = env.step(action)

                            next_price = env.get_current_price()
                            next_predicted = env.get_predicted_price()
                            next_state = np.array([next_price, next_predicted])

                            agent.learn(state, action, reward, next_state, done)
                            episode_profit += reward

                        episode_profits.append(episode_profit)
                        print(f"{model_name} - Episode {e+1}/{EPISODES}, Profit: {episode_profit:.2f}")

                    agent.save_q_table()
                    training_results[model_name].append(episode_profits)

                    # Testing Phase (Q-Learning)
                    print(f"\nTesting {model_name} on Option ID {option_id}, Run {run+1}")
                    if callable(environment_class):
                        env = environment_class(testing_data, testing_data_unscaled, features, scaler)
                    else:
                        env = environment_class(testing_data, testing_data_unscaled, features, scaler)

                    state = env.reset()
                    done = False
                    total_profit = 0
                    cumulative_rewards = []
                    agent.exploration_rate = 0.1
                    while not done:
                        current_price = env.get_current_price()
                        predicted_price = env.get_predicted_price()
                        state = np.array([current_price, predicted_price])

                        action = agent.act(state)
                        next_state_raw, reward, done, _ = env.step(action)

                        total_profit += reward
                        cumulative_rewards.append(total_profit)

                    testing_results[model_name].append(cumulative_rewards)
                    print(f"{model_name} Agent - Option ID {option_id}, Total Profit: {total_profit:.2f}")
                    trade_points = [
                        {
                            'run': run + 1,
                            'model': model_name,
                            'option_id': option_id,
                            'step': t['step'],
                            'action': t['action'],
                            'price': t['price']
                        }
                        for t in env.trade_history
                        if t['action'] in ['buy', 'sell']
                    ]
                    all_trade_points.extend(trade_points)

                    dates = testing_data_unscaled['QUOTE_DATE'].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
                    if len(cumulative_rewards) > len(dates):
                        cumulative_rewards = cumulative_rewards[:len(dates)]
                    else:
                        dates = dates[:len(cumulative_rewards)]

                    initial_cash = 1000
                    cum_perc_returns = (np.array(cumulative_rewards) / initial_cash) * 100
                    df = pd.DataFrame({
                        'Date': dates,
                        'CumulativePercentageReturn': cum_perc_returns
                    })
                    cumulative_returns_across_options_per_option[model_name].append(df)
                    total_profits_across_options[model_name].append(total_profit)

        # Plot Average Training Profit per Episode
        avg_training_results = {}
        for model_name in models_dict.keys():
            arr_runs = training_results[model_name]
            if len(arr_runs) == 0:
                continue
            max_len = max(len(x) for x in arr_runs)
            padded = [np.pad(x, (0, max_len - len(x)), 'edge') for x in arr_runs]
            mean_profits = np.mean(padded, axis=0)
            std_profits = np.std(padded, axis=0)
            avg_training_results[model_name] = (mean_profits, std_profits)

        plt.figure(figsize=(12, 8))
        plt.tight_layout()
        plt.xticks(rotation=45)
        episodes = np.arange(1, EPISODES+1)
        for model_name, (avg_profits, std_profits) in avg_training_results.items():
            ep_len = len(avg_profits)
            ep_axis = episodes[:ep_len]
            ep_len = len(avg_profits)
            ep_axis = np.arange(1, ep_len + 1)
            plt.plot(ep_axis, avg_profits, label=model_name, color=PLOT_COLORS[model_name])
            plt.fill_between(ep_axis, avg_profits - std_profits, avg_profits + std_profits,
                             alpha=0.2, color=PLOT_COLORS[model_name])
        plt.title("Average Training Profit per Episode for One Specific Option Contract")
        plt.xlabel("Episode")
        plt.ylabel("Average Profit")
        plt.legend()
        plt.savefig(f"plots/training_profit_option_{option_id}.pdf")
        plt.savefig(f"plots/training_profit_option_{option_id}.png")
        plt.show()

        # Plot Average Testing Results
        avg_testing_results = {}
        for model_name in models_dict.keys():
            arr_runs = testing_results[model_name]
            if len(arr_runs) == 0:
                continue
            max_len = max(len(x) for x in arr_runs)
            padded = [np.pad(x, (0, max_len - len(x)), 'edge') for x in arr_runs]
            mean_test = np.mean(padded, axis=0)
            std_test = np.std(padded, axis=0)
            avg_testing_results[model_name] = (mean_test, std_test)

        plt.figure(figsize=(12, 8))
        plt.tight_layout()
        plt.xticks(rotation=45)

        for model_name, (mean_test, std_test) in avg_testing_results.items():
            dates = testing_data_unscaled['QUOTE_DATE'].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
            if len(mean_test) > len(dates):
                mean_test = mean_test[:len(dates)]
                std_test = std_test[:len(dates)]
            else:
                dates = dates[:len(mean_test)]

            plt.plot(dates, mean_test, label=model_name, color=PLOT_COLORS[model_name])
            plt.fill_between(dates, mean_test - std_test, mean_test + std_test,
                             alpha=0.2, color=PLOT_COLORS[model_name])
        plt.title(f"Average Cumulative Profit for One Specific Option Contract")
        plt.xlabel("Date")
        plt.ylabel("Average Cumulative Profit")
        plt.legend()
        plt.savefig(f"plots/cumulative_profit_option_{option_id}.pdf")
        plt.savefig(f"plots/cumulative_profit_option_{option_id}.png")
        plt.show()

        # Collect Cumulative Returns for All Models
        for model_name in models_dict.keys():
            cumulative_rewards_across_options[model_name].extend(
                cumulative_returns_across_options_per_option[model_name]
            )
    option_prices = testing_data_unscaled["LAST"].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
    dates = testing_data_unscaled["QUOTE_DATE"].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, option_prices, label="Option Price", color="blue")
    
    buy_steps = [tp['step'] for tp in trade_points if tp['action'] == 'buy']
    buy_cum   = [tp['price'] for tp in trade_points if tp['action'] == 'buy']

    sell_steps = [tp['step'] for tp in trade_points if tp['action'] == 'sell']
    sell_cum   = [tp['price'] for tp in trade_points if tp['action'] == 'sell']

    buy_dates  = [dates[s] for s in buy_steps if s < len(dates)]
    sell_dates = [dates[s] for s in sell_steps if s < len(dates)]

    plt.scatter(buy_dates, buy_cum, marker='^', color='g', label='Buy', zorder=5)
    plt.scatter(sell_dates, sell_cum, marker='v', color='r', label='Sell', zorder=5)
    plt.title("Option Price with Buy/Sell Actions")
    plt.xlabel("Date")
    plt.ylabel("Option Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/option_price_with_trades.png")
    plt.show()

    # Combine Cumulative Percentage Returns Across Options
    all_models_df = None
    for model_name in models_dict.keys():
        df_list = cumulative_rewards_across_options[model_name]
        if not df_list:
            continue
        combined_df = pd.concat(df_list)
        combined_df = combined_df.groupby('Date')['CumulativePercentageReturn'].mean().reset_index()
        combined_df = combined_df.sort_values('Date')
        combined_df.rename(columns={'CumulativePercentageReturn': model_name}, inplace=True)

        if all_models_df is None:
            all_models_df = combined_df
        else:
            all_models_df = pd.merge(all_models_df, combined_df, on='Date', how='outer')

    if all_models_df is not None:
        all_models_df = all_models_df.sort_values('Date')
        plt.figure(figsize=(12, 8))
        plt.tight_layout()
        plt.xticks(rotation=45)
        for model_name in models_dict.keys():
            if model_name in all_models_df.columns:
                plt.plot(all_models_df['Date'], all_models_df[model_name],
                         label=model_name, color=PLOT_COLORS[model_name])
        plt.title("Combined Cumulative Percentage Returns Across All Test Options")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Percentage Return (%)")
        plt.legend()
        plt.grid()
        plt.savefig("plots/cumulative_returns_all_options.pdf")
        plt.savefig("plots/cumulative_returns_all_options.png")
        plt.show()
    else:
        print("No data available for plotting combined cumulative percentage returns.")


    cum_training_results = {}

    for model_name, runs_list in training_results.items():
        if not runs_list:
            continue

        cumsums = [np.cumsum(run_array) for run_array in runs_list]

        max_len = max(len(x) for x in cumsums)
        padded = [np.pad(x, (0, max_len - len(x)), 'edge') for x in cumsums]

        mean_cum = np.mean(padded, axis=0)
        std_cum = np.std(padded, axis=0)

        cum_training_results[model_name] = (mean_cum, std_cum)

    plt.figure(figsize=(12, 8))
    for model_name, (mean_cum, std_cum) in cum_training_results.items():
        ep_axis = np.arange(1, len(mean_cum) + 1)
        plt.plot(ep_axis, mean_cum, label=model_name, color=PLOT_COLORS[model_name])
        plt.fill_between(ep_axis, mean_cum - std_cum, mean_cum + std_cum,
                        alpha=0.2, color=PLOT_COLORS[model_name])

    plt.title("Cumulative Profits Across Models Over the Training Period")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Profit")
    plt.legend()
    plt.grid()
    plt.savefig("plots/cumulative_profit_training_period.pdf")
    plt.savefig("plots/cumulative_profit_training_period.png")
    plt.show()

    # Print Total Profits
    print("\nTotal Profits across all options:")
    for model_name in models_dict.keys():
        total_profit = sum(total_profits_across_options[model_name])
        print(f"{model_name} Agent: {total_profit:.2f}")


    # Convert training results dictionary to a DataFrame and save
    df_train_data = []
    for model_name, (avg_profits, std_profits) in avg_training_results.items():
        for episode, (avg_profit, std_profit) in enumerate(zip(avg_profits, std_profits), start=1):
            df_train_data.append({
                "model": model_name,
                "episode": episode,
                "avg_profit": avg_profit,
                "std_profit": std_profit
            })
    df_train = pd.DataFrame(df_train_data)
    df_train.to_csv(f"results/training_results_option_{option_id}.csv", index=False)
    print(f"Saved training results for option {option_id} to CSV")
    
    # Convert testing results dictionary to a DataFrame and save
    df_test_data = []
    for model_name, (mean_test, std_test) in avg_testing_results.items():
        for date, cum_profit, std_profit in zip(dates, mean_test, std_test):
            df_test_data.append({
                "model": model_name,
                "date": date,
                "cumulative_profit": cum_profit,
                "std_profit": std_profit
            })
    df_test = pd.DataFrame(df_test_data)
    df_test.to_csv(f"results/testing_results_option_{option_id}.csv", index=False)
    print(f"Saved testing results for option {option_id} to CSV")
    df_total_profits = pd.DataFrame({
    "model": list(total_profits_across_options.keys()),
    "total_profit": [sum(total_profits_across_options[m]) for m in total_profits_across_options.keys()]
})
    df_total_profits.to_csv("results/total_profits.csv", index=False)
    print("Saved total profits to CSV")
    # Convert cum_training_results into a DataFrame
    rows = []
    for model_name, (mean_cum, std_cum) in cum_training_results.items():
        for ep_idx, (m_val, s_val) in enumerate(zip(mean_cum, std_cum)):
            rows.append({
                "model": model_name,
                "episode": ep_idx + 1,
                "cumulative_profit_mean": m_val,
                "cumulative_profit_std": s_val
            })

    df_cum_train = pd.DataFrame(rows)
    df_cum_train.to_csv("results/cumulative_train_profits.csv", index=False)
    print("Saved cumulative training profits to CSV.")
    df_trades = pd.DataFrame(all_trade_points)
    df_trades.to_csv("results/trade_points.csv", index=False)
    print("Saved executed trade points to results/trade_points.csv")


if __name__ == '__main__':
    main()