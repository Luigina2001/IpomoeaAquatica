import pickle
import os.path as osp
from collections import defaultdict

import numpy as np

from abc import abstractmethod

from utils.constants import SHIELD_ROW_RANGE, SHIELD_COLUMN_RANGE, SHIELD_TOTAL_WIDTH, \
    SHIELD_COLUMN_SEPARATION_WIDTH, MAX_NUMBER_OF_STATES, FIRE, LEFT, RIGHT, LEFTFIRE, RIGHTFIRE


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
        self.q_table = defaultdict(int)

    def encode_state(self, state):
        """Encode the state to an integer."""
        frame, info = state
        players_pos = (np.min(info["player_pos"][1]), np.max(info["player_pos"][1])) if info["player_pos"] else (-1, -1)
        mothership_pos = (np.min(info["mothership_pos"][1]) + np.max(info["mothership_pos"][1])) if info[
            "mothership_pos"] else (-1, -1)
        shields_state = []

        if info["shields"]:
            shields_region = frame[SHIELD_ROW_RANGE[0]:SHIELD_ROW_RANGE[1] + 1,
                             SHIELD_COLUMN_RANGE[0]:SHIELD_COLUMN_RANGE[1] + 1]

            start_indices = [
                i * SHIELD_TOTAL_WIDTH + i * SHIELD_COLUMN_SEPARATION_WIDTH
                for i in range(3)
            ]

            shields_state = [
                np.sum(shields_region[:, start:start + SHIELD_TOTAL_WIDTH])
                for start in start_indices
            ]

        lives = info["lives"]

        bullet_pos = (-1, -1)

        if info["bullet_pos"]:
            bullet_curr_pos_x = (np.min(info["bullet_pos"][0]), np.max(info["bullet_pos"][0]))
            bullet_curr_pos_y = (np.min(info["bullet_pos"][1]), np.max(info["bullet_pos"][1]))
            bullet_pos = (bullet_curr_pos_x, bullet_curr_pos_y)

        return hash((players_pos, mothership_pos, tuple(shields_state), tuple(map(tuple, info["invaders_matrix"])),
                     bullet_pos, lives)) % MAX_NUMBER_OF_STATES

    def get_q_value(self, state, action, is_state_encoded: bool = True):
        """Get the Q-value for a given state-action pair. Initialize to 0 if not present."""
        return self.q_table[(state if is_state_encoded else self.encode_state(state), int(action))]

    def set_q_value(self, state, action, value, is_state_encoded: bool = True):
        """Set the Q-value for a given state-action pair."""
        self.q_table[(state if is_state_encoded else self.encode_state(state), int(action))] = value

    def policy(self, state):
        if self.env is None:
            raise ValueError("Environment not set. Please set the environment before calling the policy method.")

        if self.train_mode and np.random.uniform(0, 1) < self.eps:
            # Exploration
            return self.env.action_space.sample()

        # Exploitation
        econded_state = self.encode_state(state)
        for action in range(self.env.action_space.n):
            if (econded_state, action) in self.q_table:
                return action

        return

    def eps_schedule(self, current_step):
        self.eps = max(self.eps_end, self.eps_start -
                       self.decay_rate * current_step)

    @abstractmethod
    def initialize_env(self, *args, **kwargs):
        raise NotImplementedError

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
