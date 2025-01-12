import os
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.distributions import Categorical
from tqdm import tqdm

from utils import MetricLogger, PATIENCE
from utils.constants import MAX_STEPS
from .rl_agent import RLAgent


class A3C(RLAgent, nn.Module, ABC):
    def __init__(self, val_every_ep: int = 100, n_channels: int = 4, n_actions: int = 6, gamma: float = 0.99,
                 lr: float = 0.0001,beta: float = 0.01, tmax: int = 5, n_threads: int = 4, eps_start: int = 1,
                 eps_end: int = 0.01, decay_steps: int = 1_000_000, frame_skip: int = 4, noop_max: int = 30):

        RLAgent.__init__(self, lr, gamma, eps_start, eps_end, decay_steps)
        nn.Module.__init__(self)

        self.n_actions = n_actions
        self.n_channels = n_channels
        self.beta = beta
        self.tmax = tmax
        self.n_threads = n_threads
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.frame_skip = frame_skip
        self.noop_max = noop_max
        self.val_every_ep = val_every_ep

        ####################
        # CNN architecture #
        ####################
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out_layer = nn.Linear(512, n_actions + 1)  # [Policy logits | Value]

        self.optimizer = optim.RMSprop(params=self.parameters(), lr=self.lr, centered=False, alpha=0.99, eps=1e-5)

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

        env = AtariPreprocessing(env=env, noop_max=noop_max, frame_skip=frame_skip,
                                 grayscale_obs=True, scale_obs=True)
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
        model_state_dict = torch.load(checkpoint_path)

        instance = cls(**model_state_dict['std_parameters'])
        del model_state_dict['std_parameters']

        if 'extra_parameters' in model_state_dict:
            del model_state_dict['extra_parameters']

        instance.load_state_dict(model_state_dict)
        instance.initialize_env(env, instance.frame_skip, instance.noop_max)

        if return_params:
            return instance, model_state_dict
        return instance

    def compute_returns(self, rewards, next_value, dones):
        returns = []
        R = next_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.dummy_param.device)


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
def a3c_learning(network, n_episodes: int, val_every_ep: int, wandb_run=None, video_dir=None, tmax: int = MAX_STEPS,
                 checkpoint_dir=None, patience: int = PATIENCE, n_threads: int = os.cpu_count()):

    metric_logger = MetricLogger(wandb_run, val_every_ep)
    processed_frames = 0

    with tqdm(range(1, n_episodes + 1)) as pg_bar:
        for episode in pg_bar:
            thread_rewards = []
            for _ in range(n_threads):
                """1. Reset gradients: dθ ← 0 and dθv ← 0"""
                rewards, log_probs, values, dones = [], [], [], []

                """2. Get state st"""
                state, _ = network.env.reset()

                for t in range(tmax):
                    state_tensor = torch.tensor(state, dtype=torch.float32,
                                                device=network.dummy_param.device).unsqueeze(0)
                    logits, value = network(state_tensor).split([network.n_actions, 1], dim=1)

                    """3. Perform at according to policy π(at|st; θ')"""
                    policy = Categorical(logits=logits)
                    action = policy.sample()

                    """4. Receive reward rt and new state st+1"""
                    next_state, reward, terminated, truncated, _ = network.env.step(action.item())
                    done = terminated or truncated
                    rewards.append(reward)
                    log_probs.append(policy.log_prob(action))
                    values.append(value.squeeze())
                    dones.append(done)

                    state = next_state
                    processed_frames += 1

                    pg_desc = f"Episode: {episode}, Processed Frames: {processed_frames}, Step: {t}, " \
                              f"Current Score: {sum(rewards)}"
                    pg_bar.set_description(pg_desc)

                    if done:
                        break


                """5.   R = 0 for terminal st
                        R = V (st , θv′ ) for non-terminal st --> Bootstrap from last state """
                next_value = network(torch.tensor(next_state, dtype=torch.float32,
                                                  device=network.dummy_param.device).unsqueeze(0))
                next_value = next_value.split([network.n_actions, 1], dim=1)[1]

                """6. R ← ri + γR """
                returns = network.compute_returns(rewards, next_value, dones)

                """7. Accumulate gradients for policy and value"""
                advantages = returns - torch.stack(values)
                policy_loss = -(torch.stack(log_probs) * advantages).mean()
                value_loss = F.mse_loss(torch.stack(values), returns)
                entropy_loss = -network.beta * Categorical(logits=logits).entropy().mean()
                loss = policy_loss + value_loss + entropy_loss

                # Backpropagation
                network.optimizer.zero_grad()
                loss.backward()
                network.optimizer.step()

                thread_rewards.append(sum(rewards))
                metric_logger.log_thread_rewards(thread_rewards, n_threads)

        """T > T_max"""
        metric_logger.log_final_metrics(n_episodes, processed_frames)
        if checkpoint_dir:
            torch.save(network.state_dict(), os.path.join(checkpoint_dir, "a3c_checkpoint.pth"))
        network.env.close()
