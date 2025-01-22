import copy
import os
import random
from collections import namedtuple, deque

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from tqdm import tqdm

from utils import EarlyStopping, MetricLogger
from utils.constants import PATIENCE, MAX_STEPS
from utils.functions import handle_video
from .rl_agent import RLAgent

# Fix 'NSInternalInconsistencyException' on macOS
matplotlib.use('agg')

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, capacity, held_out_ratio):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        self.held_out_memory = []
        self.held_out_ratio = held_out_ratio

    def push(self, *args):
        if len(self.held_out_memory) < int((self.held_out_ratio * self.capacity)) and random.random() >= 0.5:
            self.held_out_memory.append(Transition(*args))
        else:
            self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_held_out(self, batch_size=None):
        if batch_size is None:
            return self.held_out_memory
        return random.sample(self.held_out_memory, min(batch_size, len(self.held_out_memory)))

    def __len__(self):
        return len(self.memory)


class DQN(RLAgent, nn.Module):
    def __init__(self, n_channels: int = 4, n_actions: int = 6, gamma: int = 0.99, eps_start: int = 1,
                 eps_end: int = 0.01, decay_steps: int = 1_000_000, lr: int = 0.000025, frame_skip: int = 4,
                 noop_max: int = 30):
        RLAgent.__init__(self, lr, gamma, eps_start, eps_end, decay_steps)
        nn.Module.__init__(self)

        self.noop_max = noop_max
        self.n_actions = n_actions
        self.n_channels = n_channels
        self.frame_skip = frame_skip
        self.dummy_param = nn.Parameter(torch.empty(0))

        ####################
        # CNN architecture #
        ####################

        # H_out = [((H_in + 2 x padding[0] - dilation[0] x (kernel_size[0] - 1) - 1)) / stride[0] + 1]
        # W_out = [((W_in + 2 x padding[1] - dilation[1] x (kernel_size[1] - 1) - 1)) / stride[1] + 1]

        # inpt_size_conv1 = (B, C_in = 1, H_in = 84, W_in = 84)
        self.conv1 = nn.Conv2d(in_channels=n_channels,
                               out_channels=32, kernel_size=8, stride=4)
        # H_out = W_out = [((84 + 2 x 0 - 1 x 7 - 1) / 4) + 1] = [(76 / 4) + 1] = 20
        # out_size_conv1 = (C_out = 32, H_out = 20, W_out = 20)
        # inpt_size_conv2 = (B, C_out = 32, H_out = 20, W_out = 20)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # H_out = W_out = [((20 + 2 x 0 - 1 x 3 - 1) / 2) + 1] = [(16 / 2) + 1] = 9
        # inpt_size_conv2 = (C_out = 32, H_out = 20, W_out = 20)
        # out_size_conv2 = (B, C_out = 64, H_out = 9, W_out = 9)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # H_out = W_out = [((9 + 2 x 0 - 1 x 2 - 1) / 1) + 1] = [(6 / 1) + 1] = 7
        # out_size_conv3 = (B, C_out = 64, H_out = 7, W_out = 7)
        # inpt_size_fc1 = C_out * H_out * W_out = 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out_layer = nn.Linear(512, self.n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)

        del self.train_mode

    def policy(self, state):
        if self.env is None:
            raise ValueError("Environment not set. Please set the environment before calling the policy method.")

        if self.training and np.random.uniform(0, 1) < self.eps:
            # Exploration: pull random action
            return torch.tensor(self.env.action_space.sample())
        # Exploitation: pull the best greedy action
        with torch.no_grad():
            # Add batch dimension
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.dummy_param.device)
            return torch.argmax(self.forward(state))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        # flatten tensor
        if len(x.shape) == 4:
            x = x.view(x.shape[0], -1)  # keep batch dim and flatten the rest
        else:
            x = x.view(-1)

        x = self.fc1(x)
        x = self.out_layer(x)

        return x

    def initialize_env(self, env, frame_skip=None, noop_max=None):
        # disable frame skipping in original env if enabled
        if env.unwrapped._frameskip > 1:
            env.unwrapped._frameskip = 1

        frame_skip = frame_skip if frame_skip is not None else self.frame_skip
        noop_max = noop_max if noop_max is not None else self.noop_max

        env = AtariPreprocessing(env=env, noop_max=noop_max, frame_skip=frame_skip, grayscale_obs=True, scale_obs=True)
        self.env = FrameStackObservation(env, stack_size=4)

    def serialize(self):
        model_state = RLAgent.serialize(self)

        model_state['std_parameters'].update({
            'noop_max': self.noop_max,
            'n_actions': self.n_actions,
            'n_channels': self.n_channels,
            'frame_skip': self.frame_skip,
        })

        return model_state

    def save_model(self, checkpoint_path: str):
        model_state_dict = self.state_dict()
        model_state_dict.update(self.serialize())

        torch.save(model_state_dict, checkpoint_path)

    @classmethod
    def load_model(cls, env, checkpoint_path: str, return_params: bool = False):
        device = torch.device("cuda" if torch.cuda.is_available()
                              else ("mps" if torch.backends.mps.is_available() else "cpu"))
        model_state_dict = torch.load(checkpoint_path, map_location=device)

        instance = cls(**model_state_dict['std_parameters'])
        del model_state_dict['std_parameters']

        if 'extra_parameters' in model_state_dict:
            del model_state_dict['extra_parameters']

        instance.load_state_dict(model_state_dict)
        instance.initialize_env(env, instance.frame_skip, instance.noop_max)

        if return_params:
            return instance, model_state_dict
        return instance

    def optimize(self, target_q_network, replay_memory, batch_size: int = 32):
        if len(replay_memory) < batch_size:
            raise ValueError(
                f"Not enough experience. Current experience: {len(self.replay_memory)} "
                f"- Current batch size: {batch_size}")

        transitions = replay_memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        device = self.dummy_param.device

        state_batch = torch.stack(tuple(torch.tensor(s) for s in batch.state), dim=0).to(device)
        action_batch = torch.IntTensor(batch.action).squeeze().to(device)
        reward_batch = torch.FloatTensor(batch.reward).squeeze().to(device)
        non_terminal_mask = torch.FloatTensor(batch.done).to(device)
        next_states_batch = torch.stack([torch.from_numpy(s) for s in batch.next_state]).to(dtype=torch.float32).to(
            device)

        action_distribution = self(state_batch)
        action_batch = action_batch.unsqueeze(1).long()
        current_q_values = action_distribution.gather(1, action_batch)

        with torch.no_grad():
            next_q_values = target_q_network(next_states_batch).max(1)[0]
            target_q_values = reward_batch + (self.gamma * next_q_values * (1 - non_terminal_mask))

        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

        return loss.item()

    def calculate_avg_q_value(self, replay_memory):
        if not replay_memory.held_out_memory:
            raise ValueError("Held-out set is empty!")

        with torch.no_grad():
            transitions = replay_memory.sample_held_out()  # held-out set states
            states = torch.stack([torch.tensor(t.state, dtype=torch.float32) for t in transitions])
            states = states.to(self.dummy_param.device)

            q_values = self(states)
            avg_q_value = q_values.mean().item()

        return avg_q_value


def dq_learning(policy_network, target_network, n_episodes: int, val_every_ep: int, batch_size: int = 32,
                target_update_freq: int = 10_000,
                replay_start_size: int = 50_000, max_steps: int = MAX_STEPS, wandb_run=None, video_dir=None,
                checkpoint_dir=None, patience: int = PATIENCE, epsilon: float = 1e-3,
                memory_capacity: int = 1_000_000, held_out_ratio: float = 0.1):
    policy_network.train()
    target_network.eval()

    replay_memory = ReplayMemory(memory_capacity, held_out_ratio=held_out_ratio)

    early_stopping = EarlyStopping(threshold=epsilon, checkpoint_dir=checkpoint_dir, patience=patience)
    metric_logger = MetricLogger(wandb_run, val_every_ep)

    processed_frames = 0
    avg_loss = 0.0
    avg_playtime = 0

    q_values_prev = None
    q_values_current = None

    with tqdm(range(1, n_episodes + 1)) as pg_bar:
        for episode in pg_bar:
            state, _ = policy_network.env.reset()
            score = 0

            video_path = handle_video(video_dir, episode, prefix="DQN")

            for t in range(max_steps):
                action = policy_network.policy(state)
                next_state, reward, truncated, terminated, _ = policy_network.env.step(action)
                done = (truncated or terminated)

                # clip reward between -1 and 1 for training stability
                replay_memory.push(state, action.item(), reward, next_state, done)

                score += reward

                # a uniform random policy is run for 'replay_start_size' frames to accumulate experience
                # before learning starts
                processed_frames += 1
                pg_desc = f"Episode: {episode}, Processed Frames: {processed_frames}, Step: {t}, " \
                          f"Current Score: {score}"

                if processed_frames >= replay_start_size:
                    loss = policy_network.optimize(target_network, replay_memory, batch_size)
                    pg_desc += f", Loss: {loss:.2f}"
                    avg_loss += loss

                    q_values_current = policy_network(
                        torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(policy_network.dummy_param.device))

                # linearly decay eps
                if processed_frames < policy_network.decay_steps:
                    policy_network.eps_schedule(processed_frames)

                # reset Q^ = Q
                if processed_frames % target_update_freq:
                    target_network.load_state_dict(policy_network.state_dict())

                if done:
                    break

                state = next_state
                pg_bar.set_description(pg_desc)

            # RawRewards
            metric_logger.raw_rewards.append(score)
            metric_logger.plot(metric_logger.raw_rewards, "Raw Rewards", "Episode", "Raw Reward")

            if processed_frames >= replay_start_size and policy_network.env.has_wrapper_attr("recorded_frames"):
                avg_playtime += len(policy_network.env.get_wrapper_attr("recorded_frames")) / 30

            if episode % val_every_ep == 0:
                avg_q_value = policy_network.calculate_avg_q_value(replay_memory)
                metric_logger.q_values.append(avg_q_value)

                metric_logger.compute_log_metrics(avg_q_value, avg_playtime,
                                                  avg_loss if processed_frames >= replay_start_size else None)
                avg_loss = 0
                avg_playtime = 0

                # Delta Q
                delta_q = metric_logger.calculate_delta_q(q_values_prev, q_values_current)
                q_values_prev = q_values_current

                # Saturation monitoring
                if processed_frames >= replay_start_size and early_stopping(delta_q, policy_network, episode,
                                                                            video_path):
                    break

    metric_logger.log_final_metrics(processed_frames)

    policy_network.env.close()
