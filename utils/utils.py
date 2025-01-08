import os

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt

from .constants import PATIENCE
from .functions import smooth_data


class EarlyStopping:
    def __init__(self, threshold: float, checkpoint_dir: str, checkpoint_ext: str = "ckpt",
                 patience: int = PATIENCE, trace_func=print):

        self.counter = 0
        self.patience = patience
        self.trace_func = trace_func
        self.threshold = threshold
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_ext = checkpoint_ext
        self.best_value = float('inf')

    def __call__(self, delta_q, model, episode, video_path=None):
        if delta_q <= self.threshold:
            self.trace_func(f"Delta Q reached converged!")
            self.counter = 0
            self.save_checkpoint(delta_q, model, episode)
            return True
        elif delta_q >= self.best_value:
            self.counter += 1
            if video_path and os.path.exists(video_path):
                self.trace_func(f"Removing video from episode {episode}: {video_path}")
                os.remove(video_path)

            if self.counter >= self.patience:
                self.trace_func(f"Delta Q did not improve for {self.counter} episodes. Stopping...")
                return True

        self.best_value = delta_q
        self.save_checkpoint(delta_q, model, episode)
        self.trace_func(f"Delta Q improved from {self.best_value} to {delta_q}!")
        return False

    def save_checkpoint(self, metric_value, model, episode):
        model_name = os.path.dirname(
            self.checkpoint_dir).split(os.path.sep)[-1]
        checkpoint_filename = f"{model_name}_ep_{episode}_delta_q_{metric_value:.4f}.{self.checkpoint_ext}"
        checkpoint_path = os.path.join(
            self.checkpoint_dir, checkpoint_filename)

        self.trace_func(f"Saving model checkpoint to: {checkpoint_path}...")

        model.save_model(checkpoint_path)

        self.trace_func("Model checkpoint saved!")

        # remove all previous checkpoints
        for filename in os.listdir(self.checkpoint_dir):
            if filename != checkpoint_filename and filename.endswith(self.checkpoint_ext):
                os.remove(os.path.join(self.checkpoint_dir, filename))


class MetricLogger:
    def __init__(self, wandb_run, val_every_ep):
        self.wandb_run = wandb_run
        self.val_every_ep = val_every_ep

        self.avg_rewards = []
        self.dbs_values = []
        self.consecutive_dbs_values = []
        self.raw_rewards = []
        self.mmavg_values = []
        self.q_values = []

        if self.wandb_run is None:
            print("Warning: Wandb run is not initialized. Logging and metric computation will be disabled.")

    def calculate_delta_q(self, prev_q_values, curr_q_values):
        # Q-value saturation refers to a condition where:
        # 1. The Q-values Q(s, a) for states and actions do not change significantly between updates.
        # 2. The gradient updates for the Q-network approach zero.
        # 3. The agent has learned a nearly stable policy,
        #    and subsequent episodes lead to only minimal variations in Q-values.

        # If Q-values saturate too early, it might indicate a problem such as:
        # - Non-explorative policy (low epsilon), limiting the agent's ability to discover new states and rewards.
        # - Under-dimensioned network or excessively low learning rate, preventing effective learning.
        # - Lack of relevant experience in the replay buffer, leading to suboptimal training data.
        if prev_q_values is None:
            return float('inf')

        if isinstance(curr_q_values, torch.Tensor):
            delta_q = torch.abs(curr_q_values - prev_q_values).mean().item()
        else:
            delta_q =  np.abs(curr_q_values - prev_q_values)

        if self.wandb_run:
            self.wandb_run.log({"Delta Q": delta_q})

        return delta_q

    def plot(self, data, title, x_label, y_label, figsize=(12, 8)):
        if self.wandb_run is None:
            return

        plt.figure(figsize=figsize)
        plt.scatter(range(len(data)), data, alpha=0.7)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        self.wandb_run.log({title: wandb.Image(plt)})

        plt.close()

    def compute_log_metrics(self, avg_q_value, avg_playtime=None, loss=None, figsize=(12, 8)):
        if self.wandb_run is None:
            return None

        # AvgReward
        avg_reward = np.mean(self.raw_rewards[-self.val_every_ep:])
        self.avg_rewards.append(avg_reward)

        # DBS
        dbs = [self.raw_rewards[i + 1] - self.raw_rewards[i] for i in range(len(self.raw_rewards) - 1)]
        self.dbs_values.extend(dbs[-self.val_every_ep:])
        self.consecutive_dbs_values.append(self.raw_rewards[-1] - self.raw_rewards[-2])

        # WDC
        wdc_n = sum([x for x in dbs if x < 0])
        wdc_p = sum([x for x in dbs if x > 0])

        # MMAVG
        mmavg = (max(self.raw_rewards[-self.val_every_ep:]) - min(self.raw_rewards[-self.val_every_ep:])) / avg_reward
        self.mmavg_values.append(mmavg)

        # Smoothed Avg Rewards
        smoothed_avg_rewards = smooth_data(self.avg_rewards, window_size=10)

        episode_data = {
            f"Avg Reward of {self.val_every_ep}": avg_reward / self.val_every_ep,
            "Smoothed AvgReward": smoothed_avg_rewards[-1] if len(smoothed_avg_rewards) > 0 else 0,
            f"Avg Q (held-out) of {self.val_every_ep}:": avg_q_value,
            "WDCn": wdc_n,
            "WDCp": wdc_p,
            "MMAVG": mmavg if len(self.mmavg_values) > 0 else 0,
        }

        if avg_playtime is not None and avg_playtime > 0:
            episode_data.update({f"Avg Playtime of {self.val_every_ep}": avg_playtime // self.val_every_ep})

        if loss is not None:
            episode_data.update({f"Avg Loss of {self.val_every_ep}": loss / self.val_every_ep})

        self.wandb_run.log(episode_data)

        if len(self.consecutive_dbs_values) == len(self.mmavg_values):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            episodes = range(len(self.consecutive_dbs_values))
            scatter = ax.scatter(episodes, self.consecutive_dbs_values, self.mmavg_values,
                                 c=self.consecutive_dbs_values, cmap='viridis', alpha=0.8)
            ax.set_xlabel('Episodes', fontsize=14)
            ax.set_ylabel('DBS', fontsize=14)
            ax.set_zlabel('MMAVG', fontsize=14)
            plt.title("3D Plot of DBS and MMAVG", fontsize=16)
            cbar = fig.colorbar(scatter, shrink=0.5, aspect=10)
            cbar.set_label('DBS Intensity', fontsize=12)

            self.wandb_run.log({"DBS and MMAVG 3D Plot": wandb.Image(plt)})

            plt.close()

    def log_final_metrics(self, episode, convergence_steps, figsize=(12, 8)):
        if self.wandb_run is None:
            return

        summary_data = {"Convergence Steps": convergence_steps}

        if len(self.q_values):
            q_min = min(self.q_values)
            q_max = max(self.q_values)

            normalized_q_values = [10 * (q - q_min) / (q_max - q_min) if q_max > q_min else 10 for q in self.q_values]

            for q in normalized_q_values:
                self.wandb_run.log({"Normalized Avg Q Values": q})

        if len(self.dbs_values) > 0:
            for v in self.dbs_values:
                self.wandb_run.log({"DBS": v})

            plt.figure(figsize=figsize)
            colors = ["red" if v < 0 else "blue" for v in self.dbs_values]

            episodes = range(len(self.dbs_values))
            plt.bar(episodes, self.dbs_values, color=colors, width=0.95)

            plt.xlabel("Episode")
            plt.ylabel("DBS")
            plt.title("DBS Histogram")

            self.wandb_run.log({"DBS Histogram": wandb.Image(plt)})

            plt.close()

            data = [[_ * self.val_every_ep, self.dbs_values[_]] for _ in range(0, len(self.dbs_values), self.val_every_ep)]
            table = wandb.Table(data=data, columns=["Episode", "DBS"])
            summary_data.update({f"DBS Table of {self.val_every_ep}": wandb.plot.bar(table, "Episode", "DBS")})

        if len(self.raw_rewards) > 0:
            data = [[(_ + 1), self.raw_rewards[_]] for _ in range(episode)]
            table = wandb.Table(data=data, columns=["Episode", "Raw Reward"])
            summary_data.update({"Raw Reward Table": wandb.plot.scatter(table, "Episode", "Raw Reward")})

        self.wandb_run.log(summary_data)
