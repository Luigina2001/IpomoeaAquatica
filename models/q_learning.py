import hashlib
import os
from collections import defaultdict

import cv2
import numpy as np
import os.path as osp
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

    def encode_state(self, state):
        frame, _ = state
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return hashlib.sha256(frame.tobytes()).hexdigest()

    def train_step(self, n_episodes: int, var_threshold: float, val_every_ep: int, max_steps: int = MAX_STEPS,
                   patience: int = PATIENCE, wandb_run=None, video_dir=None, checkpoint_dir=None):

        avg_playtime = 0
        total_steps = 0
        cumulative_rewards = []

        with tqdm(range(n_episodes)) as pg_bar:
            for episode in pg_bar:
                state, _ = self.env.reset()
                cumulative_reward = 0
                action_values = []

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

                    if len(cumulative_rewards) > 1:
                        variance = np.var(cumulative_rewards)
                        print(f"\n=========\nVariance over last {val_every_ep} episodes: {variance}\n=========")
                        episode_data.update({f"Performance variance over {val_every_ep} episodes": variance})

                        if np.var(cumulative_rewards) <= var_threshold:
                            print(f"\n=========\nQLearning agent reached convergence! Total steps needed: {total_steps}\n=========")
                            episode_data.update({"Convergence steps": total_steps})
                            log_results(wandb_run, episode_data)
                            break

                log_results(wandb_run, episode_data)

            checkpoint_path = os.path.join(checkpoint_dir, f"QLearning_ep_{episode}.pkl")
            print(f"Saving model to {checkpoint_path}")
            self.save_model(checkpoint_path)
            print("Model saved successfully!")

            if video_dir and avg_playtime > 0:
                log_results(wandb_run, {"Playtime": avg_playtime // n_episodes})

            self.env.close()
