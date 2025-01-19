import os
from abc import ABC
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.distributions import Categorical

from utils.functions import seed_everything
from .rl_agent import RLAgent

from tqdm import tqdm


# Based on: https://github.com/MorvanZhou/pytorch-A3C

class Agent(nn.Module):
    def __init__(self, n_channels: int, n_actions: int, model_type: Literal['Actor', 'Critic']):
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

        if self.model_type == 'Actor':
            self.out_layer = nn.Linear(128, self.n_actions)
            self.softmax = nn.Softmax(dim=1)
        else:
            self.out_layer = nn.Linear(128, 1)  # Value function

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)

        if len(x.shape) == 4:
            x = x.view(x.shape[0], -1)
        else:
            x = x.view(-1)

        x = self.lstm_cell(x)[0]  # get hidden state
        x = self.out_layer(x)

        if self.model_type == 'Actor':
            x = self.softmax(x)

        return x


class A3C(RLAgent, nn.Module, ABC):
    def __init__(self, n_channels: int = 4, n_actions: int = 6, gamma: float = 0.99,
                 lr: float = 0.0001, beta: float = 0.01, eps_start: int = 1,
                 eps_end: int = 0.01, decay_steps: int = 1_000_000, frame_skip: int = 4):

        RLAgent.__init__(self, lr, gamma, eps_start, eps_end, decay_steps)
        nn.Module.__init__(self)

        self.n_actions = n_actions
        self.n_channels = n_channels
        self.beta = beta
        self.frame_skip = frame_skip

        self.actor = Agent(n_channels=n_channels, n_actions=n_actions, model_type='Actor')
        self.critic = Agent(n_channels=n_channels, n_actions=n_actions, model_type='Critic')

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
        return torch.tensor(returns, dtype=torch.float32)

    def forward(self, x, value_only: bool = False):
        value = self.critic(x)

        if value_only:
            return value

        probs = self.actor(x)

        return probs, value


class Worker(mp.Process):
    def __init__(self, global_network, optimizer, queue, condition, stop_signal, rank, t_max, global_episode,
                 n_episodes, seed, n_threads, val_every_ep, wandb_run):
        super(Worker, self).__init__()

        self.global_network = global_network
        self.optimizer = optimizer
        self.t_max = t_max
        self.rank = rank
        self.queue = queue
        self.condition = condition
        self.stop_signal = stop_signal

        self.global_episode = global_episode
        self.n_episodes = n_episodes + 1
        self.seed = seed
        self.n_threads = n_threads
        self.val_every_ep = val_every_ep
        self.wandb_run = wandb_run

        self.local_network = A3C(n_actions=global_network.n_actions, n_channels=global_network.n_channels,
                                 eps_start=global_network.eps_start, eps_end=global_network.eps_end,
                                 decay_steps=global_network.decay_steps, gamma=global_network.gamma,
                                 frame_skip=global_network.frame_skip)

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
        pid = os.getpid()
        print(f"Worker {self.rank} started. - PID: {pid} Current global episode: {self.global_episode.value}",
              flush=True)

        # Handle CPU Oversubscription:
        # https://pytorch.org/docs/stable/notes/multiprocessing.html#cpu-oversubscription

        torch.set_num_threads(self.n_threads)

        while self.global_episode.value < self.n_episodes and not self.stop_signal.value:

            """1. Reset gradients: dθ ← 0 and dθv ← 0"""
            rewards, log_probs, values, dones = [], [], [], []
            """2. Get state st"""
            state, _ = self.local_network.env.reset()
            done = False
            action_probs = None
            processed_frames = 0

            with tqdm(range(self.t_max), desc=f"Worker {self.rank}") as pg_bar:
                for t in pg_bar:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs, value = self.local_network(state_tensor)

                    """3. Perform at according to policy π(at|st; θ')"""
                    policy = Categorical(action_probs)
                    action = policy.sample()

                    if t % 10 == 0:
                        self.wandb_run.log({
                            f"Worker {self.rank}/action_probs_mean": action_probs.mean().item(),
                            f"Worker {self.rank}/value_step": value.item()
                        })

                    """4. Receive reward rt and new state st+1"""
                    next_state, reward, terminated, truncated, _ = self.local_network.env.step(action)
                    done = terminated or truncated

                    self.wandb_run.log({f"Worker {self.rank}/reward_raw": reward})

                    processed_frames += 1
                    self.local_network.eps_schedule(processed_frames)

                    rewards.append(reward)
                    log_probs.append(torch.log(action_probs[0, action]))
                    values.append(value.squeeze())
                    dones.append(done)

                    pg_bar.set_description(
                        f"Worker {self.rank} - Global episode: {self.global_episode.value} "
                        f"- Step: {t + 1}/{self.t_max}, Score: {sum(rewards)}")

                    if done:
                        break

                    state = next_state

            """5.   R = 0 for terminal st
                    R = V (st , θv′ ) for non-terminal st --> Bootstrap from last state """
            next_value = 0 if done else self.local_network.critic(torch.FloatTensor(state).unsqueeze(0))
            """6. R ← ri + γR """
            returns = self.local_network.compute_returns(rewards, next_value, dones)
            """7. Accumulate gradients for policy and value"""
            advantages = returns - torch.stack(values)

            # Log advantages and returns
            self.wandb_run.log({
                f"Worker {self.rank}/advantage_mean": advantages.mean().item(),
                f"Worker {self.rank}/value_mean": torch.stack(values).mean().item(),
                f"Worker {self.rank}/return_mean": returns.mean().item()
            })


            """Compute loss"""
            policy_loss = -(torch.stack(log_probs) * advantages).mean()
            value_loss = F.mse_loss(torch.stack(values), returns, reduction='mean')
            # entropy = -1 * (action_probs * torch.log(action_probs)).sum().mean()
            # entropy_loss = self.local_network.beta * entropy
            entropy_loss = -self.local_network.beta * Categorical(action_probs).entropy().mean()
            loss = policy_loss + value_loss + entropy_loss

            # Log losses and entropy
            entropy = Categorical(action_probs).entropy().mean()
            self.wandb_run.log({
                f"Worker {self.rank}/policy_loss": policy_loss.item(),
                f"Worker {self.rank}/value_loss": value_loss.item(),
                f"Worker {self.rank}/entropy_loss": entropy_loss.item(),
                f"Worker {self.rank}/total_loss": loss.item(),
                f"Worker {self.rank}/entropy": entropy.item()
            })

            # Update local and global network
            self.optimizer.zero_grad()
            loss.backward()

            for lp, gp in zip(self.local_network.parameters(), self.global_network.parameters()):
                gp.grad = lp.grad

            for name, param in self.global_network.named_parameters():
                if param.grad is not None:
                    self.wandb_run.log({f"grad_norm/{name}": param.grad.norm().item()})

            self.optimizer.step()
            self.local_network.load_state_dict(self.global_network.state_dict())

            # notify main process that the episode has ended
            self.queue.put(self.rank)

            with self.condition:
                self.condition.wait()

        self.local_network.env.close()
