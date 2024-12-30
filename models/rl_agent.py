import os
import pickle
import numpy as np
import os.path as osp

from abc import abstractmethod
from utils import EarlyStopping
from utils.constants import PATIENCE


class RLAgent:
    def __init__(self, env, lr, gamma, eps_start, eps_end, decay_steps, *args, **kwargs):
        self.lr = lr
        self.env = env
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.decay_steps = decay_steps
        self.decay_rate = (self.eps_start - self.eps_end) / self.decay_steps
        self.train_mode = True
        self.rewards_per_episode = []

    def reset_environment(self):
        return self.env.reset()

    def close_environment(self):
        self.env.close()

    def handle_video(self, video_dir, episode, prefix="video"):
        if video_dir:
            return osp.join(video_dir, f"{prefix}_episode-{episode}.mp4")
        return None

    def initialize_early_stopping(self, checkpoint_dir: str, patience: int = PATIENCE, metric: str = "Reward", objective: str = "maximize"):
        if checkpoint_dir:
            return EarlyStopping(metric, objective, checkpoint_dir=checkpoint_dir, patience=patience)
        return None

    def log_results(self, wandb_run, episode_data):
        if wandb_run:
            wandb_run.log(episode_data)

    def handle_early_stopping(self, early_stopping, reward, agent, episode, video_path):
        if early_stopping and early_stopping(reward, agent, episode):
            if video_path and osp.exists(video_path):
                os.remove(video_path)
            return True
        return False

    def policy(self, state):
        if self.env is None:
            raise ValueError(
                "Environment not set. Please set the environment before calling the policy method.")

        if self.train_mode and np.random.uniform(0, 1) < self.eps:
            # Exploration
            return int(self.env.action_space.sample())

        # Exploitation
        econded_state = self.encode_state(state)
        for action in range(self.env.action_space.n):
            if (econded_state, action) in self.q_table:
                return action

        return 0

    def eps_schedule(self, current_step):
        self.eps = max(self.eps_end, self.eps_start -
                       self.decay_rate * current_step)

    @abstractmethod
    def initialize_env(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    def serialize(self):
        model_state = {
            'std_parameters': {
                'env': self.env,
                'lr': self.lr,
                'gamma': self.gamma,
                'eps_start': self.eps_start,
                'eps_end': self.eps_end,
                'decay_steps': self.decay_steps,
            }
        }

        if hasattr(self, 'train_mode'):
            model_state.update(
                {'extra_parameters': {'train_mode': self.train_mode}})

        return model_state

    def save_model(self, checkpoint_path: str):
        model_state = self.serialize()
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(model_state, f)
        except (IOError, pickle.PickleError) as e:
            raise ValueError(f"Error saving model to {checkpoint_path}: {e}")

    @classmethod
    def load_model(cls, checkpoint_path: str, return_params: bool = False):
        if not osp.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file {checkpoint_path} not found.")

        try:
            with open(checkpoint_path, "rb") as f:
                model_state = pickle.load(f)

            instance = cls(**model_state['std_parameters'])
            instance.train_mode = model_state['extra_parameters']['train_mode']

            if return_params:
                return instance, model_state
            return instance

        except (IOError, pickle.PickleError) as e:
            raise ValueError(
                f"Error loading model from {checkpoint_path}: {e}")

    def evaluation_mode(self):
        self.train_mode = False
