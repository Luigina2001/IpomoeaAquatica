import hashlib
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from utils import EarlyStopping, MetricLogger
from utils.constants import PATIENCE, MAX_STEPS
from utils.functions import handle_video
from wrappers import Action, Observation, Reward
from .rl_agent import RLAgent


def init_q_table():
    return 1


class QLearning(RLAgent):
    def __init__(self, env, lr, gamma, eps_start, eps_end, decay_steps, normalize_reward: bool = False, *args,
                 **kwargs):
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
    def encode_state(state):
        frame, _ = state
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return hashlib.sha256(frame.tobytes()).hexdigest()

    def train_step(self, n_episodes: int, val_every_ep: int, max_steps: int = MAX_STEPS,
                   patience: int = PATIENCE, wandb_run=None, video_dir=None, checkpoint_dir=None,
                   epsilon: float = 1e-3):

        early_stopping = EarlyStopping(threshold=epsilon, checkpoint_dir=checkpoint_dir, patience=PATIENCE)
        metric_logger = MetricLogger(wandb_run, val_every_ep)

        avg_playtime = 0
        processed_frames = 0
        cumulative_rewards = []

        q_values_prev = None
        q_values_current = None

        with tqdm(range(1, n_episodes + 1)) as pg_bar:
            for episode in pg_bar:
                state, _ = self.env.reset()
                cumulative_reward = 0

                video_path = handle_video(video_dir, episode, prefix="QLearning")

                for _ in range(max_steps):
                    action = self.policy(state)
                    next_state, reward_info, truncated, terminated, info = self.env.step(action)
                    cumulative_reward += reward_info['reward']

                    # update q-value
                    enc_state = self.encode_state(state)
                    enc_next_state = self.encode_state(next_state)
                    max_q = max(self.q_table[(enc_next_state, a)]
                                for a in range(self.env.action_space.n))

                    self.q_table[(enc_state, action)] = self.q_table[(enc_state, action)] + self.lr * (
                                reward_info['reward'] + self.gamma * max_q - self.q_table[(enc_state, action)])
                    metric_logger.q_values.append(self.q_table[(enc_state, action)])

                    processed_frames += 1

                    # linearly decay eps
                    if processed_frames < self.decay_steps:
                        self.eps_schedule(processed_frames)

                    if truncated or terminated:
                        break

                    state = next_state

                    pg_bar.set_description(
                        f"Episode: {episode}, Step: {_}, Cumulative Reward: {cumulative_reward}, Current Score: {reward_info['score']}")

                    if len(metric_logger.raw_rewards) > 0:
                        metric_logger.raw_rewards.append(abs(reward_info['score'] - metric_logger.raw_rewards[-1]))
                    else:
                        metric_logger.raw_rewards.append(reward_info['score'])

                cumulative_rewards.append(cumulative_reward)
                # Cumulative Reward
                metric_logger.plot(cumulative_rewards, "Cumulative Reward per Episode", "Episode", "Cumulative Reward")
                # Raw Rewards
                metric_logger.plot(metric_logger.raw_rewards, "Raw Rewards", "Episode", "Cumulative Reward")

                if self.env.has_wrapper_attr("recorded_frames"):
                    avg_playtime += len(self.env.get_wrapper_attr("recorded_frames"))

                if episode % val_every_ep == 0:
                    avg_q_value = np.std(metric_logger.q_values[-_ * val_every_ep:])
                    metric_logger.compute_log_metrics(avg_q_value, avg_playtime)
                    q_values_current = avg_q_value

                    # Delta Q
                    delta_q = metric_logger.calculate_delta_q(q_values_prev, q_values_current)
                    q_values_prev = q_values_current

                    # Saturation monitoring
                    if episode % val_every_ep == 0 and early_stopping(delta_q, self, episode, video_path):
                        break

            metric_logger.log_final_metrics(episode, processed_frames)

            self.env.close()
