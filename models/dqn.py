import os
import torch
import random

import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from .rl_agent import RLAgent
from utils.constants import PATIENCE, MAX_STEPS
from utils.functions import initialize_early_stopping, handle_early_stopping, handle_video, log_results

from gymnasium.wrappers import AtariPreprocessing
from collections import namedtuple, deque

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory:
    def __init__(self, capacity, held_out_ratio):
        self.memory = deque([], maxlen=capacity)
        self.held_out_memory = []
        self.held_out_ratio = held_out_ratio

    def push(self, *args):
        if random.random() < self.held_out_ratio:
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
    def __init__(self, env, n_channels: int = 1, n_actions: int = 6, gamma: int = 0.99, eps_start: int = 1,
                 eps_end: int = 0.01,
                 decay_steps: int = 1_000_000, memory_capacity: int = 1_000_000, held_out_ratio=0.1,
                 lr: int = 0.000025, frame_skip: int = 3, noop_max: int = 30):
        RLAgent.__init__(self, env, lr, gamma, eps_start, eps_end, decay_steps)
        nn.Module.__init__(self)

        self.noop_max = noop_max
        self.n_actions = n_actions
        self.n_channels = n_channels
        self.frame_skip = frame_skip
        self.memory_capacity = memory_capacity
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

        self.replay_memory = ReplayMemory(memory_capacity, held_out_ratio)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.initialize_env(frame_skip=self.frame_skip, noop_max=self.noop_max)

        del self.train_mode

    def policy(self, state):
        if self.training and np.random.uniform(0, 1) < self.eps:
            # Exploration: pull random action
            return torch.tensor(np.random.choice(self.n_actions))
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

    def optimize_model(self, target_q_network, batch_size: int = 32):
        if len(self.replay_memory) < batch_size:
            raise ValueError(
                f"Not enough experience. Current experience: {len(self.replay_memory)} - Current batch size: {batch_size}")

        transitions = self.replay_memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(tuple(torch.tensor(state)
                                        for state in batch.state), dim=0).to(self.dummy_param.device)
        state_batch = state_batch.unsqueeze(1)
        action_batch = torch.tensor(batch.action, dtype=torch.int64).to(self.dummy_param.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.dummy_param.device)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.dummy_param.device)
        non_final_next_states = torch.stack(
            [torch.from_numpy(s) for s in batch.next_state if s is not None]).to(dtype=torch.float32).to(
            self.dummy_param.device)
        non_final_next_states = non_final_next_states.unsqueeze(1)

        action_distribution = self(state_batch)
        action_batch = action_batch.unsqueeze(1).long()
        current_q_values = action_distribution.gather(1, action_batch)

        with torch.no_grad():
            next_q_values = torch.zeros(batch_size, device=state_batch.device)
            next_q_values[non_final_mask] = target_q_network(
                non_final_next_states).max(1)[0]

            target_q_values = reward_batch + (next_q_values * self.gamma)

        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def calculate_avg_q_value(self):
        if not self.replay_memory.held_out_memory:
            raise ValueError("Held-out set is empty!")

        with torch.no_grad():
            transitions = self.replay_memory.sample_held_out()  # held-out set states
            states = torch.stack([torch.tensor(t.state, dtype=torch.float32) for t in transitions])
            states = states.unsqueeze(1).to(self.dummy_param.device)

            q_values = self.forward(states)

            avg_q_value = q_values.mean().item()

        return avg_q_value

    def train_step(self, n_episodes: int, val_every_ep: int, batch_size: int = 32, target_update_freq: int = 10_000,
                   replay_start_size: int = 50_000, max_steps: int = MAX_STEPS, wandb_run=None, video_dir=None,
                   checkpoint_dir=None, patience: int = PATIENCE):

        early_stopping = initialize_early_stopping(checkpoint_dir, patience)

        target_q_network = DQN(
            env=self.env, n_channels=self.n_channels, n_actions=self.n_actions)
        target_q_network.load_state_dict(self.state_dict())
        target_q_network.to(self.dummy_param.device)
        target_q_network.eval()

        processed_frames = 0
        curr_loss = 0
        avg_playtime = 0
        prev_counter = 0

        with tqdm(range(n_episodes)) as pg_bar:
            for episode in pg_bar:

                state, _ = self.env.reset()
                avg_loss = 0.0
                score = 0.0  # it is the same as the game score

                if prev_counter < early_stopping.counter:
                    if video_path and osp.exists(video_path):
                        os.remove(video_path)

                if episode > 1 and episode % val_every_ep == 0:
                    self.eval()

                video_path = handle_video(video_dir, episode, prefix="DQN")

                prev_counter = early_stopping.counter

                for T in range(max_steps):
                    action = self.policy(state)
                    next_state, reward, terminated, truncated, _ = self.env.step(
                        action)
                    done = truncated or terminated
                    next_state = next_state if not done else None

                    self.replay_memory.push(
                        state,
                        action.item(),
                        # clip between -1 and 1 for training stability
                        max(-1.0, min(reward, 1.0)),
                        next_state
                    )

                    score += reward

                    # a uniform random policy is run for 'replay_start_size' frames to accumulate experience
                    # before learning starts
                    processed_frames += 1
                    if processed_frames >= replay_start_size:
                        curr_loss = self.optimize_model(target_q_network, batch_size)
                        avg_loss += curr_loss

                    # linearly decay eps
                    if processed_frames < self.decay_steps:
                        self.eps_schedule(processed_frames)

                    # reset Q^ = Q
                    if processed_frames % target_update_freq == 0:
                        avg_q_value = self.calculate_avg_q_value()
                        pg_bar.set_description(
                            f"Episode: {episode}, Avg Q (held-out): {avg_q_value:.2f}, Processed Frames: {processed_frames}"
                        )
                        log_results(wandb_run, {"avg_q_held_out": avg_q_value})
                        target_q_network.load_state_dict(self.state_dict())

                    if done:
                        break

                    state = next_state

                    pg_bar.set_description(
                        f"Episode: {episode}, Processed Frames: {processed_frames}, Step: {T}, Current Loss: {curr_loss:.2f}, Current Score: {score}")

                log_results(wandb_run, {"avg_loss": avg_loss / T, "avg_reward": score / T,
                                        "game_score": score})

                if processed_frames >= replay_start_size and self.env.has_wrapper_attr("recorded_frames"):
                    avg_playtime += len(self.env.get_wrapper_attr("recorded_frames"))

                if processed_frames >= replay_start_size and episode > 1 and episode % val_every_ep == 0:
                    if handle_early_stopping(early_stopping=early_stopping, reward=score, agent=self,
                                             episode=episode,
                                             video_path=video_path):
                        break
                    self.train()

            if video_dir and avg_playtime > 0:
                log_results(wandb_run, {"Playtime": avg_playtime // n_episodes})

            self.env.close()

    def initialize_env(self, frame_skip, noop_max):
        # disable frame skipping in original env if enabled
        if self.env.unwrapped._frameskip > 1:
            self.env.unwrapped._frameskip = 1

        self.env = AtariPreprocessing(
            env=self.env, noop_max=noop_max, frame_skip=frame_skip, scale_obs=True)

    def serialize(self):
        model_state = RLAgent.serialize(self)

        model_state['std_parameters'].update({
            'noop_max': self.noop_max,
            'n_actions': self.n_actions,
            'n_channels': self.n_channels,
            'frame_skip': self.frame_skip,
            'memory_capacity': self.memory_capacity
        })

        return model_state

    def save_model(self, checkpoint_path: str):
        model_state_dict = self.state_dict()
        model_state_dict.update(self.serialize())

        torch.save(model_state_dict, checkpoint_path)

    @classmethod
    def load_model(cls, env, checkpoint_path: str, return_params: bool = False):
        model_state_dict = torch.load(checkpoint_path)

        instance = cls(env=env, **model_state_dict['std_parameters'])
        del model_state_dict['std_parameters']

        if 'extra_parameters' in model_state_dict:
            del model_state_dict['extra_parameters']

        instance.load_state_dict(model_state_dict)

        if return_params:
            return instance, model_state_dict
        return instance
