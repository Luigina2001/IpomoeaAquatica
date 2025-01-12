from abc import ABC
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from utils.functions import seed_everything
from .rl_agent import RLAgent

# Based on: https://github.com/MorvanZhou/pytorch-A3C

class Agent(nn.Module):
    def __init__(self, n_channels: int, n_actions: int, lr: float, model_type: Literal['Actor', 'Critic']):
        super().__init__()

        self.n_channels = n_channels
        self.model_type = model_type
        self.n_actions = n_actions
        # inpt_size_conv1 = (B, C_in = 1, H_in = 84, W_in = 84)
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=8, stride=4)
        # H_out = W_out = [((84 + 2 x 0 - 1 x 7 - 1) / 4) + 1] = [(76 / 4) + 1] = 20
        # out_size_conv1 = (C_out = 32, H_out = 20, W_out = 20)
        # inpt_size_conv2 = (B, C_out = 32, H_out = 20, W_out = 20)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # H_out = W_out = [((20 + 2 x 0 - 1 x 3 - 1) / 2) + 1] = [(16 / 2) + 1] = 9
        # inpt_size_conv2 = (C_out = 32, H_out = 20, W_out = 20)
        # out_size_conv2 = (B, C_out = 64, H_out = 9, W_out = 9)
        # inpt_size_out = C_out * H_out * W_out = 64 * 9 * 9 = 5184
        self.lstm_cell = nn.LSTMCell(5184, 128)

        # TODO: critic should have 1 output unit for the value
        self.out_layer = nn.Linear(128, self.n_actions)

        if self.model_type == 'Actor':
            self.last_layer = nn.Softmax(dim=1)
        else:
            self.last_layer = nn.Identity()

    def forward(self, x):
        if self.model_type == 'Actor':
            return self._actor_forward(x)
        return self._critic_forward(x)

    def _critic_forward(self, x):
        pass

    def _actor_forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)

        if len(x.shape) == 4:
            x = x.view(x.shape[0], -1)
        else:
            x = x.view(-1)

        x = self.lstm_cell(x)
        x = self.out_layer(x)
        x = self.last_layer(x)

        return x


class A3C(RLAgent, nn.Module, ABC):
    def __init__(self, val_every_ep: int = 100, n_channels: int = 4, n_actions: int = 6, gamma: float = 0.99,
                 lr: float = 0.0001, beta: float = 0.01, t_max: int = 5, n_threads: int = 4, eps_start: int = 1,
                 eps_end: int = 0.01, decay_steps: int = 1_000_000, frame_skip: int = 4):

        RLAgent.__init__(self, lr, gamma, eps_start, eps_end, decay_steps)
        nn.Module.__init__(self)

        self.n_actions = n_actions
        self.n_channels = n_channels
        self.beta = beta
        self.t_max = t_max
        self.n_threads = n_threads
        self.frame_skip = frame_skip
        self.val_every_ep = val_every_ep

        self.actor = Agent(n_channels=n_channels, n_actions=n_actions, lr=lr, model_type='Actor')
        self.critic = Agent(n_channels=n_channels, n_actions=n_actions, lr=lr, model_type='Critic')

    def initialize_env(self, env, frame_skip=None):
        # disable frame skipping in original env if enabled
        if env.unwrapped._frameskip > 1:
            env.unwrapped._frameskip = 1

        frame_skip = frame_skip if frame_skip is not None else self.frame_skip

        env = AtariPreprocessing(env=env, frame_skip=frame_skip,
                                 grayscale_obs=True, scale_obs=True)
        self.env = FrameStackObservation(env, stack_size=4)

    def serialize(self):
        model_state = RLAgent.serialize(self)

        model_state['std_parameters'].update({
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
        model_state_dict = torch.load(checkpoint_path)

        instance = cls(**model_state_dict['std_parameters'])
        del model_state_dict['std_parameters']

        if 'extra_parameters' in model_state_dict:
            del model_state_dict['extra_parameters']

        instance.load_state_dict(model_state_dict)
        instance.initialize_env(env, instance.frame_skip)

        if return_params:
            return instance, model_state_dict
        return instance

    def policy(self, state):
        if self.env is None:
            raise ValueError("Environment not set. Please set the environment before calling the policy method.")

        if self.training and np.random.uniform(0, 1) < self.eps:
            # Exploration: pull random action
            return torch.tensor(self.env.action_space.sample())
        # Exploitation: pull the best greedy action
        with torch.no_grad():
            # Add batch dimension
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return torch.argmax(self.actor(state))

    def compute_returns(self, rewards, next_value, dones):
        returns = []
        R = next_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.dummy_param.device)

    def forward(self, x, value_only: bool = False):
        value = self.critic(x)

        if value_only:
            return value

        probs = self.actor(x)

        return probs, value


class Worker(mp.Process):
    def __init__(self, global_network, optimizer, rank, t_max, global_episode, n_episodes, seed, n_threads):
        super(Worker, self).__init__()

        self.global_network = global_network
        self.optimizer = optimizer
        self.t_max = t_max
        self.rank = rank
        self.global_episode = global_episode
        self.n_episodes = n_episodes
        self.seed = seed
        self.n_threads = n_threads

        self.local_network = A3C(global_network.n_actions, global_network.n_channels, global_network.lr)

        # generate unique env for each process
        seed_everything(self.seed + self.rank)
        self.local_network.initialize_env(gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array"),
                                          frame_skip=self.local_network.frame_skip)

    """
        Algorithm S3 Asynchronous advantage actor-critic - pseudocode for each actor-learner thread.
            // Assume global shared parameter vectors θ and θv and global shared counter T = 0 
            // Assume thread-specific parameter vectors θ′ and θv′
            Initialize thread step counter t ← 1
            repeat
                1. Reset gradients: dθ ← 0 and dθv ← 0.
                   Synchronize thread-specific parameters θ′ = θ and θv′ = θv 
                   t_start = t
                2. Get state st
                repeat
                    3. Perform at according to policy π(at|st; θ′) 
                    4. Receive reward rt and new state st+1 
                       t←t+1
                       T←T+1
                until terminal st or t − tstart == tmax 
                    5. R = 0 for terminal st
                    5. R = V (st , θv′ ) for non-terminal st --> Bootstrap from last state 
                for i ∈ {t − 1, . . . , t_start} do
                    6. R ← ri + γR
                    7. Accumulate gradients wrt θ′: dθ ← dθ + ∇θ′ log π(ai|si; θ′)(R − V (si; θv′ )) 
                    7. Accumulate gradients wrt θv′ : dθv ← dθv + ∂ (R − V (si; θv′ ))2/∂θv′
                end for
                Perform asynchronous update of θ using dθ and of θv using dθv . 
            until T > Tmax
    """

    def run(self):
        print(f"Worker {self.rank} started. Current global episode: {self.global_episode}")

        # Handle CPU Oversubscription: https://pytorch.org/docs/stable/notes/multiprocessing.html#cpu-oversubscription
        torch.set_num_threads(self.n_threads)

        while self.global_episode < self.n_episodes:
            """1. Reset gradients: dθ ← 0 and dθv ← 0"""
            rewards, log_probs, values, dones = [], [], [], []
            """2. Get state st"""
            state, _ = self.local_network.env.reset()
            done = False
            action_probs = None

            for t in range(self.t_max):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, value = self.local_network(state_tensor)

                """3. Perform at according to policy π(at|st; θ')"""
                action = self.local_network.policy(state)

                """4. Receive reward rt and new state st+1"""
                next_state, reward, terminated, truncated, _ = self.local_network.env.step(action)
                done = terminated or truncated

                rewards.append(reward)
                log_probs.append(torch.log(action_probs[action]))
                values.append(value.squeeze())
                dones.append(done)

                if done:
                    break

                state = next_state

            """5.   R = 0 for terminal st
                    R = V (st , θv′ ) for non-terminal st --> Bootstrap from last state """

            next_value = 0 if done else self.local_network.critic(torch.FloatTensor(state).unsqueeze(0), True)
            """6. R ← ri + γR """
            returns = self.local_network.compute_returns(rewards, next_value, dones)

            """7. Accumulate gradients for policy and value"""
            advantages = returns - torch.stack(values)
            policy_loss = -(torch.stack(log_probs) * advantages).mean()
            value_loss = F.mse_loss(torch.stack(values), returns, reduction='mean')
            entropy = -1 * (action_probs * torch.log(action_probs)).sum().mean()
            entropy_loss = self.local_network.beta * entropy

            loss = policy_loss + value_loss + entropy_loss

            # Update local and global network
            self.optimizer.zero_grad()
            loss.backward()

            for lp, gp in zip(self.local_network.parameters(), self.global_network.parameters()):
                gp.grad = lp.grad

            self.optimizer.step()

            self.local_network.load_state_dict(self.global_network.state_dict())

        self.local_network.env.close()
