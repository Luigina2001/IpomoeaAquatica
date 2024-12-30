
import os
import hashlib
import os.path as osp

from tqdm import tqdm
from collections import defaultdict

from .rl_agent import RLAgent
from utils.constants import PATIENCE, MAX_STEPS
from wrappers import Action, Observation, Reward


def init_q_table():
    return 1


class QLearning(RLAgent):
    def __init__(self, env, lr, gamma, eps_start, eps_end, decay_steps, *args, **kwargs):
        super().__init__(env, lr, gamma, eps_start, eps_end, decay_steps, *args, **kwargs)

        self.q_table = defaultdict(init_q_table)
        self.initialize_env()

    def initialize_env(self):
        # disable frame skipping in original env if enabled
        if self.env.unwrapped._frameskip > 1:
            self.env.unwrapped._frameskip = 3

        self.env = Reward(self.env)
        self.env = Observation(self.env)
        self.env = Action(self.env)

    def serialize(self):
        model_state = super().serialize()

        model_state.update({
            'extra_parameters': {'q_table': self.q_table}
        })

        return model_state

    @classmethod
    def load_model(cls, checkpoint_path: str, return_params: bool = False):
        instance, model_state = super().load_model(checkpoint_path, True)

        instance.q_table = model_state["extra_parameters"]["q_table"]

        if return_params:
            return instance, model_state
        return instance

    def encode_state(self, state):
        frame, _ = state
        return hashlib.sha256(frame.tobytes()).hexdigest()

    def train(self, n_episodes: int, max_steps: int = MAX_STEPS, patience: int = PATIENCE, wandb_run=None, video_dir=None, checkpoint_dir=None):
        total_steps = 0

        early_stopping = self.initialize_early_stopping(checkpoint_dir, patience, metric="Cumulative Reward")

        with tqdm(range(n_episodes)) as pg_bar:
            for episode in pg_bar:

                video_path = self.handle_video(
                    video_dir, episode, prefix="QLearning")

                state, _ = self.reset_environment()
                cumulative_reward = 0

                for _ in range(max_steps):
                    action = self.policy(state)
                    next_state, reward_info, truncated, terminated, info = self.env.step(
                        action)
                    total_steps += 1
                    cumulative_reward += reward_info['reward']

                    self.eps_schedule(total_steps)

                    # update q-value
                    enc_state = self.encode_state(state)
                    enc_next_state = self.encode_state(next_state)
                    max_q = max(self.q_table[(enc_next_state, a)]
                                for a in range(self.env.action_space.n))

                    self.q_table[(enc_state, action)] = self.q_table[(
                        enc_state, action)] + self.lr * (reward_info['reward'] + self.gamma * max_q - self.q_table[(enc_state, action)])

                    if truncated or terminated:
                        break

                    state = next_state

                    pg_bar.set_description(
                        f"Episode: {episode}, Step: {_}, Cumulative Reward: {cumulative_reward}, Current Score: {reward_info['score']}")

                self.log_results(
                    wandb_run, {"game_score": reward_info['score']})

                if self.handle_early_stopping(
                    early_stopping=early_stopping, reward=reward_info['reward'], agent=self, episode=episode, video_path=video_path):
                    break

            self.close_environment()
