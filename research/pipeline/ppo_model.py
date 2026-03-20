import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PLOT_COLORS_rgb = {
    'blue': (0, 0.447, 0.741),
    'red': (0.85, 0.325, 0.098),
    'yellow': (0.929, 0.694, 0.125),
    'purple': (0.494, 0.184, 0.556),
    'green': (0.466, 0.674, 0.188),
    'light_blue': (0.301, 0.745, 0.933),
    'dark_red': (0.635, 0.078, 0.184)
}

FIXED_COST = 0.70
PROPORTIONAL_COST = 0.005

FEATURES = ['UNDERLYING_LAST', 'STRIKE', 'is_call', 'IV', 'dte', 
            'sentiment_score', 'garch_volatility', 'RSI', 'MACD', 'MACD_Signal', 
            'PE Ratio', 'Price to Sales Ratio']

SEQUENCE_LENGTH = 10
TEST_SIZE = 0.2
INITIAL_CASH = 1000

class PPOTRadingEnv(gym.Env):
    """
    A Gym environment for trading options with PPO.
    Actions: 0=Hold, 1=Buy, 2=Sell
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, data_scaled, data_unscaled, features, scaler, initial_cash=INITIAL_CASH, sequence_length=SEQUENCE_LENGTH):
        super(PPOTRadingEnv, self).__init__()
        self.data_scaled = data_scaled.reset_index(drop=True)
        self.data_unscaled = data_unscaled.reset_index(drop=True)
        self.features = features
        self.scaler = scaler
        self.initial_cash = initial_cash
        self.sequence_length = sequence_length

        obs_dim = len(features) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.episode_rewards = []

        self.reset()

    def reset(self, seed=None):
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0
        self.trade_history = []
        self.profits = []
        self.episode_reward = 0.0
        
        self.sequence_scaled = []
        for i in range(self.sequence_length):
            self.sequence_scaled.append(self.data_scaled.loc[self.current_step + i, self.features].values)
        return self._get_obs(), {}

    def _get_current_price(self):
        idx = self.features.index("UNDERLYING_LAST")
        scaled_val = self.sequence_scaled[-1][idx]
        dummy = np.zeros((1, len(self.features)))
        dummy[0, idx] = scaled_val
        price = self.scaler.inverse_transform(dummy)[0, idx]
        return price

    def _get_obs(self):
        idx = np.arange(len(self.features))
        last_scaled = self.sequence_scaled[-1].reshape(1, -1)
        last_unscaled = self.scaler.inverse_transform(last_scaled)[0]
        current_price = self._get_current_price()
        obs = np.concatenate([last_unscaled, [current_price]])
        return obs.astype(np.float32)

    def step(self, action):
        done = False
        current_price = self._get_current_price()

        if self.current_step + self.sequence_length >= len(self.data_unscaled) - 1:
            done = True
            next_price = current_price
        else:
            idx = self.features.index("UNDERLYING_LAST")
            next_scaled = self.data_scaled.loc[self.current_step + self.sequence_length, self.features].values[idx]
            dummy = np.zeros((1, len(self.features)))
            dummy[0, idx] = next_scaled
            next_price = self.scaler.inverse_transform(dummy)[0, idx]

        prev_total_asset = self.cash + self.position * current_price
        cost = 0

        if action == 1:  # Buy
            if self.position == 0 and self.cash >= current_price:
                self.position = 1
                self.cash -= current_price
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
        elif action == 2:  # Sell
            if self.position == 1:
                self.position = 0
                self.cash += current_price
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

        total_asset = self.cash + self.position * next_price
        reward = total_asset - prev_total_asset
        self.profits.append(total_asset - self.initial_cash)
        self.current_step += 1
        self.episode_reward += reward

        if not done:
            self.sequence_scaled.pop(0)
            self.sequence_scaled.append(self.data_scaled.loc[self.current_step + self.sequence_length - 1, self.features].values)
            obs = self._get_obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self.episode_rewards.append(self.episode_reward)

        return obs, reward, done, done, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, Position: {self.position}, Profit: {self.profits[-1] if self.profits else 0:.2f}")

from research.pipeline.helpers import load_and_preprocess_data

def main():
    data, features, target, data_unscaled = load_and_preprocess_data('enriched_dataset.csv')

    option_id_counts = data['option_id'].value_counts()
    valid_option_ids = option_id_counts.nlargest(1).index.tolist()
    print(f"Selected top {len(valid_option_ids)} option_ids with most data points: {valid_option_ids}")
    data = data[data['option_id'].isin(valid_option_ids)]

    RUNS = 3

    all_train_episode_rewards = []
    all_test_cumulative_rewards = []
    final_profits = []

    final_results = []
    time_series_results = []

    for run in range(RUNS):
        print(f"\n=== RUN {run+1}/{RUNS} ===")

        for option_id in valid_option_ids:
            print(f"Processing Option ID: {option_id} (run {run+1})")

            option_data = data[data['option_id'] == option_id].reset_index(drop=True)
            option_data_unscaled = data_unscaled[data_unscaled['option_id'] == option_id].reset_index(drop=True)

            split_index = int(len(option_data) * 0.6)
            training_data = option_data.iloc[:split_index].reset_index(drop=True)
            testing_data  = option_data.iloc[split_index:].reset_index(drop=True)

            training_data_unscaled = option_data_unscaled.iloc[:split_index].reset_index(drop=True)
            testing_data_unscaled  = option_data_unscaled.iloc[split_index:].reset_index(drop=True)

            scaler = MinMaxScaler()
            scaler.fit(training_data[features])

            training_data[features] = scaler.transform(training_data[features])
            testing_data[features]  = scaler.transform(testing_data[features])

            env_train = PPOTRadingEnv(
                training_data,
                training_data_unscaled,
                features,
                scaler,
                initial_cash=1000,
                sequence_length=10
            )
            ppo_agent = PPO(
                "MlpPolicy", 
                env_train, 
                verbose=0, 
                batch_size=8, 
                gamma=0.99, 
                n_steps=len(training_data), 
                n_epochs=50
            )
            ppo_agent.learn(total_timesteps=len(training_data)*50, progress_bar=True)

            train_episode_rewards = getattr(env_train, "episode_rewards", [])
            if not train_episode_rewards:
                train_episode_rewards = [getattr(env_train, "total_reward", 0.0)]

            env_test = PPOTRadingEnv(
                testing_data,
                testing_data_unscaled,
                features,
                scaler,
                initial_cash=1000,
                sequence_length=10
            )
            obs, _ = env_test.reset()
            done = False
            test_cumulative_rewards = []
            total_profit = 0
            step_count = 0

            while not done:
                action, _states = ppo_agent.predict(obs)
                obs, reward, done, _, _ = env_test.step(action)
                total_profit += reward
                test_cumulative_rewards.append(total_profit)

                time_series_results.append({
                    "option_id": option_id,
                    "run": run+1,
                    "step": step_count,
                    "cumulative_profit": total_profit
                })
                step_count += 1

            final_profits.append(total_profit)
            all_train_episode_rewards.append(np.array(train_episode_rewards))
            all_test_cumulative_rewards.append(np.array(test_cumulative_rewards))

            final_results.append({
                "option_id": option_id,
                "run": run+1,
                "final_profit": total_profit
            })

            print(f"PPO Agent - Option ID {option_id}, Final Test Profit: {total_profit:.2f}")

    if all_train_episode_rewards:
        min_len_train = min(len(arr) for arr in all_train_episode_rewards)
        truncated_train = [arr[:min_len_train] for arr in all_train_episode_rewards]
        train_matrix = np.vstack(truncated_train)

        mean_train = np.mean(train_matrix, axis=0)
        std_train  = np.std(train_matrix, axis=0)
        episodes   = np.arange(len(mean_train))

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, mean_train, label="PPO")
        plt.fill_between(episodes, mean_train - std_train, mean_train + std_train, alpha=0.2)
        plt.title("Average Training Profit per Episode (PPO)")
        plt.xlabel("Episode")
        plt.ylabel("Profit")
        plt.legend()
        plt.grid(True)
        plt.show()

    if all_test_cumulative_rewards:
        min_len_test = min(len(arr) for arr in all_test_cumulative_rewards)
        truncated_test = [arr[:min_len_test] for arr in all_test_cumulative_rewards]
        test_matrix = np.vstack(truncated_test)

        mean_test = np.mean(test_matrix, axis=0)
        std_test  = np.std(test_matrix, axis=0)
        steps     = np.arange(len(mean_test))

        plt.figure(figsize=(10, 6))
        plt.plot(steps, mean_test, label="PPO")
        plt.fill_between(steps, mean_test - std_test, mean_test + std_test, alpha=0.2)
        plt.title("Average Cumulative Profit on Test (PPO)")
        plt.xlabel("Test Step")
        plt.ylabel("Cumulative Profit")
        plt.legend()
        plt.grid(True)
        plt.show()

    plt.figure(figsize=(10, 6))
    x_labels = [f"Run {i+1}" for i in range(len(final_profits))]
    plt.bar(x_labels, final_profits, color='blue')
    plt.title("Final Test Profits (All Runs, PPO)")
    plt.xlabel("Run (and Option ID if multiple)")
    plt.ylabel("Final Cumulative Profit")
    plt.grid(axis='y')
    plt.show()

    avg_final_profit = np.mean(final_profits)
    std_final_profit = np.std(final_profits)
    print(f"\nPPO - Average Final Profit over {len(final_profits)} results: "
          f"{avg_final_profit:.2f} ± {std_final_profit:.2f}")

    df_final = pd.DataFrame(final_results)
    df_final.to_csv("ppo_final_results.csv", index=False)
    print("Saved PPO final results to ppo_final_results.csv")

    df_ts = pd.DataFrame(time_series_results)
    df_ts.to_csv("ppo_time_series_results.csv", index=False)
    print("Saved PPO time series results to ppo_time_series_results.csv")

if __name__ == '__main__':
    main()