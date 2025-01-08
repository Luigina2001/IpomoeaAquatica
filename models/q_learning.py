import hashlib
import os
from collections import defaultdict

import cv2
import numpy as np
import os.path as osp

import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.functions import initialize_early_stopping, handle_video, handle_early_stopping, log_results
from .rl_agent import RLAgent
from utils.constants import PATIENCE, MAX_STEPS
from wrappers import Action, Observation, Reward
from sklearn.preprocessing import minmax_scale


def init_q_table():
    return 1


class QLearning(RLAgent):
    def __init__(self, env, lr, gamma, eps_start, eps_end, decay_steps, normalize_reward: bool = False, *args, **kwargs):
        super().__init__(env, lr, gamma, eps_start, eps_end, decay_steps, *args, **kwargs)

        self.q_table = defaultdict(init_q_table)
        self.normalize_reward = normalize_reward

        self.initialize_env()

    def initialize_env(self):
        # disable frame skipping in original env if enabled
        if self.env.unwrapped._frameskip > 1:
            self.env.unwrapped._frameskip = 3

        self.env = Reward(self.env, normalize_reward=self.normalize_reward)
        self.env = Observation(self.env)
        self.env = Action(self.env)

    def serialize(self):
        model_state = super().serialize()

        model_state.update({
            'extra_parameters': {'q_table': self.q_table, 'normalize_reward': self.normalize_reward}
        })

        return model_state

    def policy(self, state):
        if self.env is None:
            raise ValueError("Environment not set. Please set the environment before calling the policy method.")

        if self.train_mode and np.random.uniform(0, 1) < self.eps:
            # Exploration
            return int(self.env.action_space.sample())

        # Exploitation
        max_q = float("-inf")
        best_action = None
        encoded_state = self.encode_state(state)

        for action in range(self.env.action_space.n):
            if (encoded_state, action) in self.q_table:
                if self.q_table[(encoded_state, action)] > max_q:
                    max_q = self.q_table[(encoded_state, action)]
                    best_action = action

        if max_q != float("-inf"):
            if best_action == 0:
                self.count_noop += 1
            if self.count_noop == 10:
                self.count_noop = 0
                best_action = np.random.choice([1, 2, 3, 4, 5])
            return int(best_action)

        return int(np.random.choice([1, 2, 3, 4, 5]))

    @classmethod
    def load_model(cls, env, checkpoint_path: str, return_params: bool = False):
        instance, model_state = super().load_model(env, checkpoint_path, True)

        instance.q_table = model_state["extra_parameters"]["q_table"]
        instance.normalize_reward = model_state["extra_parameters"]["normalize_reward"]

        if return_params:
            return instance, model_state
        return instance

    @staticmethod
    def smooth_data(data, window_size=10):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def encode_state(self, state):
        frame, _ = state
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return hashlib.sha256(frame.tobytes()).hexdigest()

    @staticmethod
    def log_3d_plot(consecutive_dbs_values, mmavg_values, wandb_run):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        episodes = range(len(consecutive_dbs_values))
        scatter = ax.scatter(episodes, consecutive_dbs_values, mmavg_values,
                             c=consecutive_dbs_values, cmap='viridis', alpha=0.8)
        ax.set_xlabel('Episodes', fontsize=14)
        ax.set_ylabel('DBS', fontsize=14)
        ax.set_zlabel('MMAVG', fontsize=14)
        plt.title("3D Plot of DBS and MMAVG", fontsize=16)
        cbar = fig.colorbar(scatter, shrink=0.5, aspect=10)
        cbar.set_label('DBS Intensity', fontsize=12)
        log_results(wandb_run, {"DBS and MMAVG 3D Plot": wandb.Image(plt)})
        plt.close()

    @staticmethod
    def calculate_delta_q(q_values_prev, q_values_current):
        # Q-value saturation refers to a condition where:
        # 1. The Q-values Q(s, a) for states and actions do not change significantly between updates.
        # 2. The gradient updates for the Q-network approach zero.
        # 3. The agent has learned a nearly stable policy,
        #    and subsequent episodes lead to only minimal variations in Q-values.

        # If Q-values saturate too early, it might indicate a problem such as:
        # - Non-explorative policy (low epsilon), limiting the agent's ability to discover new states and rewards.
        # - Under-dimensioned network or excessively low learning rate, preventing effective learning.
        # - Lack of relevant experience in the replay buffer, leading to suboptimal training data.

        if q_values_prev is None:
            return float('inf')  # first step

        delta_q = abs(q_values_current - q_values_prev)
        return delta_q

    def train_step(self, n_episodes: int, var_threshold: float, val_every_ep: int, max_steps: int = MAX_STEPS,
                   patience: int = PATIENCE, wandb_run=None, video_dir=None, checkpoint_dir=None,
                   epsilon: float = 1e-3):

        avg_playtime = 0
        total_steps = 0
        patience_counter = 0
        cumulative_rewards = []

        raw_rewards = []
        avg_rewards = []
        consecutive_dbs_values = []
        dbs_values = []
        wdc_n, wdc_p = 0, 0
        mmavg_values = []
        action_values = []

        q_values_prev = None
        q_values_current = None

        with tqdm(range(n_episodes)) as pg_bar:
            for episode in pg_bar:
                state, _ = self.env.reset()
                cumulative_reward = 0

                for _ in range(max_steps):
                    action = self.policy(state)
                    next_state, reward_info, truncated, terminated, info = self.env.step(action)
                    cumulative_reward += reward_info['reward']

                    self.eps_schedule(total_steps)

                    # update q-value
                    enc_state = self.encode_state(state)
                    enc_next_state = self.encode_state(next_state)
                    max_q = max(self.q_table[(enc_next_state, a)]
                                for a in range(self.env.action_space.n))

                    self.q_table[(enc_state, action)] = self.q_table[
                                                            (enc_state, action)] + self.lr * (reward_info['reward']
                                                                                              + self.gamma * max_q
                                                                                              - self.q_table[
                                                                                                  (enc_state, action)])
                    action_values.append(self.q_table[(enc_state, action)])

                    if truncated or terminated:
                        break

                    state = next_state

                    pg_bar.set_description(
                        f"Episode: {episode + 1}, Step: {_}, Cumulative Reward: {cumulative_reward}, Current Score: {reward_info['score']}")
                    raw_rewards.append(reward_info['score'])

                total_steps += _

                if action_values:
                    normalized_q_values = minmax_scale(
                        action_values, feature_range=(0, 1))
                    avg_q_value = np.mean(normalized_q_values)

                if self.env.has_wrapper_attr("recorded_frames"):
                    avg_playtime += len(self.env.get_wrapper_attr("recorded_frames"))

                episode_data = {
                    "Average Q-Value": avg_q_value,
                    "Cumulative Reward": cumulative_reward,
                    "Game Score": reward_info['score']
                }

                if (episode + 1) % val_every_ep == 0:
                    cumulative_rewards.append(cumulative_reward)

                    q_values_current = action_values[_]

                    '''if len(cumulative_rewards) > 1:
                        variance = np.var(cumulative_rewards)
                        print(f"\n=========\nVariance over last {val_every_ep} episodes: {variance}\n=========")
                        episode_data.update({f"Performance variance over {val_every_ep} episodes": variance})

                        if np.var(cumulative_rewards) <= var_threshold:
                            print(f"\n=========\nQLearning agent reached convergence! Total steps needed: {total_steps}\n=========")
                            episode_data.update({"Convergence steps": total_steps})
                            log_results(wandb_run, episode_data)
                            break'''

                    # Delta Q
                    delta_q = self.calculate_delta_q(q_values_prev, q_values_current)
                    log_results(wandb_run, {"Delta Q": delta_q})
                    q_values_prev = q_values_current

                    # Saturation monitoring
                    if (episode + 1) % val_every_ep and delta_q < epsilon:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(
                                f"Early stopping triggered at episode {episode} after {patience} consecutive stable episodes.")
                            break
                    else:
                        patience_counter = 0

                log_results(wandb_run, episode_data)

                # Cumulative Reward
                plt.figure(figsize=(12, 8))
                plt.scatter(range(len(cumulative_rewards)), cumulative_rewards, alpha=0.7)
                plt.xlabel("Episode")
                plt.ylabel("Cumulative Reward")
                plt.title("Cumulative Reward per Episode")
                log_results(wandb_run, {"Cumulative Reward Scatter": wandb.Image(plt)})
                plt.close()

                # Raw Reward
                plt.figure(figsize=(12, 8))
                plt.scatter(range(len(raw_rewards)), raw_rewards, alpha=0.7)
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.title("Reward per Episode")
                log_results(wandb_run, {"Reward Scatter": wandb.Image(plt)})
                plt.close()

                # AvgReward
                if len(raw_rewards) >= val_every_ep:
                    avg_reward = np.mean(raw_rewards[-val_every_ep:])
                    avg_rewards.append(avg_reward)

                # DBS
                if len(raw_rewards) > 1:
                    dbs = [raw_rewards[i + 1] - raw_rewards[i] for i in range(len(raw_rewards) - 1)]
                    dbs_values.extend(dbs[-val_every_ep:])
                    consecutive_dbs = raw_rewards[-1] - raw_rewards[-2]
                    consecutive_dbs_values.append(consecutive_dbs)

                # WDC
                if len(dbs_values) > 0:
                    wdc_n = sum([x for x in dbs if x < 0])
                    wdc_p = sum([x for x in dbs if x > 0])

                # MMAVG
                if len(raw_rewards) >= val_every_ep:
                    mmavg = (max(raw_rewards[-val_every_ep:]) - min(raw_rewards[-val_every_ep:])) / avg_reward
                    mmavg_values.append(mmavg)

                # Smoothed Avg Rewards
                smoothed_avg_rewards = self.smooth_data(avg_rewards, window_size=10)

                episode_data = {
                    f"Avg Reward of {val_every_ep}": avg_reward / val_every_ep,
                    "Smoothed AvgReward": smoothed_avg_rewards[-1] if len(smoothed_avg_rewards) > 0 else 0,
                    f"Avg Q (held-out) of {val_every_ep}:": avg_q_value,
                    "WDCn": wdc_n,
                    "WDCp": wdc_p,
                    "MMAVG": mmavg if len(mmavg_values) > 0 else 0,
                }

                log_results(wandb_run, episode_data)

                if len(consecutive_dbs_values) == len(mmavg_values):
                    self.log_3d_plot(consecutive_dbs_values, mmavg_values, wandb_run)

            if action_values:
                q_min = min(action_values)
                q_max = max(action_values)

                normalized_q_values = [10 * (q - q_min) / (q_max - q_min) if q_max > q_min else 10 for q in
                                       action_values]

                for q in normalized_q_values:
                    wandb.log({"Normalized Avg Q Values": q})

            if len(dbs_values) > 0:
                for v in dbs_values:
                    wandb.log({"DBS": v})

                plt.figure(figsize=(12, 8))
                colors = ["red" if v < 0 else "blue" for v in dbs_values]

                episodes = range(len(dbs_values))
                plt.bar(episodes, dbs_values, color=colors, width=0.95)

                plt.xlabel("Episode")
                plt.ylabel("DBS")
                plt.title("DBS Histogram")

                log_results(wandb_run, {"DBS Histogram": wandb.Image(plt)})
                plt.close()

                data = [[_ * val_every_ep, dbs_values[_]] for _ in range(0, len(dbs_values), val_every_ep)]
                table = wandb.Table(data=data, columns=["Episode", "DBS"])
                log_results(wandb_run, {f"DBS Table of {val_every_ep}": wandb.plot.bar(table, "Episode", "DBS")})

            if len(raw_rewards) > 0:
                data = [[(_ + 1), raw_rewards[_]] for _ in range(episode)]
                table = wandb.Table(data=data, columns=["Episode", "Raw Reward"])
                log_results(wandb_run, {"Raw Reward Table": wandb.plot.scatter(table, "Episode", "Raw Reward")})

            log_results(wandb_run, {"Convergence steps": total_steps})

            checkpoint_path = os.path.join(checkpoint_dir, f"QLearning_ep_{episode}.pkl")
            print(f"Saving model to {checkpoint_path}")
            self.save_model(checkpoint_path)
            print("Model saved successfully!")

            if video_dir and avg_playtime > 0:
                log_results(wandb_run, {"Playtime": avg_playtime // n_episodes})

            self.env.close()
