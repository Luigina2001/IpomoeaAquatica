
import os
import hashlib
import os.path as osp

from tqdm import tqdm
from collections import defaultdict

from .rl_agent import RLAgent
from utils import EarlyStopping
from utils.constants import PATIENCE, MAX_STEPS
from wrappers import Action, Observation, Reward


class QLearning(RLAgent):
    def __init__(self, env, lr, gamma, eps_start, eps_end, decay_steps, *args, **kwargs):
        super().__init__(env, lr, gamma, eps_start, eps_end, decay_steps, *args, **kwargs)

        self.q_table = defaultdict(int)
        self.initialize_env()

    def initialize_env(self):
        # disable frame skipping in original env if enabled
        if self.env.unwrapped._frameskip > 1:
            self.env.unwrapped._frameskip = 3

        self.env = Reward(self.env)
        self.env = Observation(self.env)
        self.env = Action(self.env)

    def encode_state(self, state):
        frame, _ = state
        return hashlib.sha256(frame.tobytes()).hexdigest()

    def train(self, n_episodes: int, max_steps: int = MAX_STEPS, patience: int = PATIENCE, wandb_run=None, video_dir=None, checkpoint_dir=None):
        total_steps = 0

        if checkpoint_dir:
            early_stopping = EarlyStopping(
                "Reward", "maximize", checkpoint_dir=checkpoint_dir, patience=patience)

        with tqdm(range(n_episodes)) as pg_bar:
            for episode in pg_bar:

                if video_dir:
                    video_path = osp.join(
                        video_dir, f"video_QLearning_episode-{episode}.mp4")

                state, _ = self.env.reset()

                for _ in range(max_steps):
                    action = self.policy(state)
                    next_state, reward_info, truncated, terminated, info = self.env.step(
                        action)
                    total_steps += 1

                    self.eps_schedule(total_steps)

                    # update q-value
                    enc_state = self.encode_state(state)
                    enc_next_state = self.encode_state(next_state)
                    max_q = max(self.q_table[(enc_next_state, action)]
                                for action in range(self.env.action_space.n))

                    self.q_table[(enc_state, action)] = self.q_table[(
                        enc_state, action)] + self.lr * (reward_info['reward'] + self.gamma * max_q - self.q_table[(enc_state, action)])

                    if truncated or terminated:
                        break

                    state = next_state

                if wandb_run:
                    wandb_run.log({"game_score": reward_info['score']})

                if checkpoint_dir:
                    counter = early_stopping.counter
                    if early_stopping(reward_info['score'], self, episode):
                        if video_dir and osp.exists(video_path):
                            os.remove(video_path)
                        break
                    elif video_dir and counter < early_stopping.counter:
                        if osp.exists(video_path):
                            os.remove(video_path)

            self.env.close()
