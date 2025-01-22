import copy
import os
import time
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from torch.distributions import Categorical
from torch.multiprocessing import Manager, Queue, Value, Condition

import yaml
import torch
from tqdm import tqdm

import wandb
import models
import ale_py
import argparse
import os.path as osp
import gymnasium as gym
import math

from functools import partial

from models.dqn import dq_learning
from models.a3c import A3C, Worker
from utils import MetricLogger, EarlyStopping
from utils.functions import seed_everything, handle_video
from gymnasium.wrappers import RecordVideo
from utils.constants import N_EPISODES, N_STEPS, PATIENCE

import torch.nn.functional as F

gym.register_envs(ale_py)


def train(args):
    seed_everything(args.seed)

    agent_name = args.agent
    run = None

    experiment_dir = args.experiment_dir

    if agent_name not in args.experiment_dir:
        experiment_dir = str(osp.join(args.experiment_dir, agent_name))

    if not args.no_log_to_wandb:
        with open("wandb_key.txt", "r") as f:
            api_key = f.read().strip()
        wandb.login(key=api_key)

        run_name = args.run_name if args.run_name else (agent_name.lower() if args.no_log_to_wandb else None)
        run = wandb.init(entity=args.entity, project=args.project, name=run_name)
        run.name = agent_name + "-" + run.name if args.sweep_id is None \
            else agent_name + f"-sweep-{args.sweep_id}-" + run.name
        experiment_dir = str(osp.join(experiment_dir if experiment_dir else args.experiment_dir, run.id))

        wandb.define_metric("episode")
        wandb.define_metric("*", step_metric="episode", step_sync=True)
    else:
        experiment_dir = str(
            osp.join(experiment_dir if experiment_dir else args.experiment_dir, str(time.time())))

    os.makedirs(experiment_dir, exist_ok=True)

    video_dir = os.path.join(experiment_dir, f"video/")
    os.makedirs(video_dir, exist_ok=True)

    agent_parameters = {
        "lr": args.lr,
        "gamma": args.gamma,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "decay_steps": args.decay_steps
    }

    train_args = {
        "n_episodes": args.n_episodes,
        "wandb_run": run,
        "video_dir": video_dir,
        "checkpoint_dir": experiment_dir,
        "val_every_ep": args.val_every_ep
    }

    if agent_name == "QLearning":
        agent_parameters.update({
            'normalize_reward': args.normalize_reward,
            "max_steps": args.n_steps
        })

    elif agent_name == "DQN":
        train_args.update({
            'batch_size': args.batch_size,
            'patience': args.patience,
            'replay_start_size': args.replay_start_size,
            'target_update_freq': args.target_update_freq,
            "memory_capacity": args.memory_capacity,
            "held_out_ratio": args.held_out_ratio,
            "epsilon": args.epsilon,
            "max_steps": args.n_steps,
        })

    elif agent_name == "A3C":
        agent_parameters.update({
            "beta": args.beta
        })

    if args.tune_hyperparameters:
        agent_parameters["lr"] = wandb.config['lr']
        agent_parameters["gamma"] = wandb.config['gamma']
        agent_parameters["eps_start"] = wandb.config['eps_start']

    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")

    if agent_name == "A3C":
        with Manager() as _:
            lr = agent_parameters["lr"]
            del agent_parameters["lr"]

            global_network = A3C(**agent_parameters)
            global_network.initialize_env(env)
            global_network.share_memory()  # share parameters between processes
            global_episode = Value('i', 1)
            stop_signal = Value('b', False)
            optimizer = optim.RMSprop(global_network.parameters(), lr=lr)
            queue = Queue()
            condition = Condition()

            metric_logger = MetricLogger(run, args.val_every_ep)
            wandb_run = run

            early_stopping = EarlyStopping(float('-inf'), experiment_dir, patience=args.patience)
            stop_training = False

            n_threads_per_worker = math.floor(
                args.n_workers / args.n_vcpus) if args.n_threads is None else args.n_threads

            print(f"\n====\nStarting A3C training with {args.n_workers} workers...\n====\n")

            workers = [
                Worker(global_network, optimizer, queue, condition, stop_signal, rank, args.n_steps, global_episode,
                       args.n_episodes,
                       args.seed, n_threads_per_worker, args.val_every_ep, wandb_run) for rank in range(args.n_workers)]

            [worker.start() for worker in workers]

            global_network.env = RecordVideo(global_network.env,
                                             episode_trigger=lambda t: t % args.val_every_ep == 0,
                                             video_folder=video_dir,
                                             name_prefix=f"video_{agent_name}")

            raw_rewards = []
            avg_rewards = []
            dbs_values = []
            consecutive_dbs_values = []
            mmavg_values = []
            wdc_n = 0
            wdc_p = 0
            curr_score = 0
            last_score = 0

            # Main loop for managing episode counter
            while True:
                # Collect messages from workers
                messages = set()
                while len(messages) < args.n_workers:
                    worker_id = queue.get()
                    messages.add(worker_id)

                with condition:
                    if global_episode.value % args.val_every_ep == 0:
                        print(
                            f"\n\n====\nMain process: Starting validation for episode {global_episode.value}\n====\n\n")

                        global_network.eval()

                        # Disable epsilon during validation
                        if global_network.eps > 0:
                            global_network.eps = 0

                        rewards, log_probs, values, dones = [], [], [], []
                        state, _ = global_network.env.reset()

                        video_path = handle_video(video_dir, global_episode.value, prefix="A3C")

                        with tqdm(range(args.n_steps)) as pg_bar:
                            for t in pg_bar:
                                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                                action_probs, value = global_network(state_tensor)
                                policy = Categorical(action_probs)
                                action = policy.sample()

                                next_state, reward, terminated, truncated, _ = global_network.env.step(action)
                                done = terminated or truncated

                                rewards.append(reward)
                                log_probs.append(torch.log(action_probs[0, action]))
                                values.append(value.squeeze())
                                dones.append(done)

                                pg_bar.set_description(
                                    f"Main thread - Global episode: {global_episode.value} - "
                                    f"Step: {t + 1}/{args.n_steps}, "
                                    f"Score: {sum(rewards)}")

                                if done:
                                    break

                                state = next_state

                        curr_score = sum(rewards)
                        next_value = 0 if done else global_network.critic(torch.FloatTensor(state).unsqueeze(0))
                        returns = global_network.compute_returns(rewards, next_value, dones)

                        advantages = returns - torch.stack(values)
                        policy_loss = -(torch.stack(log_probs) * advantages).mean()
                        value_loss = F.mse_loss(torch.stack(values), returns, reduction='mean')
                        entropy_loss = -global_network.beta * Categorical(action_probs).entropy().mean()

                        loss = policy_loss + value_loss + entropy_loss

                        wandb_run.log({
                            "Policy loss": policy_loss.item(),
                            "Value loss": value_loss.item(),
                            "Entropy loss": entropy_loss.item(),
                            "Total loss": loss.item(),
                            "Advantage mean": advantages.mean().item(),
                            "Return mean": returns.mean().item(),
                            "Reward/Score mean": np.mean(rewards),
                            "Score": sum(rewards)
                        })

                        # Avg reward
                        avg_reward = np.mean(rewards)
                        avg_rewards.append(avg_reward)

                        # DBS
                        if len(rewards) > 1:
                            dbs = [curr_score - last_score]
                            dbs_values.extend([curr_score - last_score])

                            # WDC
                            wdc_n = sum([x for x in dbs if x < 0])
                            wdc_p = sum([x for x in dbs if x > 0])

                            # MMAVG
                            mmavg = (np.max(rewards) - np.min(rewards)) / avg_reward
                            mmavg_values.append(mmavg)

                            wandb_run.log({
                                f"Avg Reward": avg_reward,
                                f"MMAVG": mmavg if len(mmavg_values) > 0 and mmavg is not None else 0,
                                f"WDC_n": wdc_n,
                                f"WDC_p": wdc_p
                            })

                        global_network.train()

                        stop_training = early_stopping(loss.item(), global_network, global_episode.value,
                                                       video_path=video_path)

                    # Increment global counter
                    with global_episode.get_lock():
                        global_episode.value += 1
                        stop_training = stop_training or global_episode.value >= args.n_episodes

                        with stop_signal.get_lock():
                            stop_signal.value = stop_training

                        if not stop_training:
                            print(
                                f"\n\n====\nMain process incremented global episode counter: "
                                f"{global_episode.value}\n====\n\n")

                    condition.notify_all()

                # If all episodes are done, exit
                if stop_training:
                    break

            [worker.join() for worker in workers]

            # metric_logger.log_final_metrics(global_episode.value - 1)
            if len(dbs_values) > 0:
                for v in dbs_values:
                    wandb_run.log({"DBS": v})

                plt.figure(figsize=(12, 8))
                colors = ["red" if v < 0 else "blue" for v in dbs_values]

                episodes = range(len(dbs_values))
                plt.bar(episodes, dbs_values, color=colors, width=0.95)

                plt.xlabel("Episode")
                plt.ylabel("DBS")
                plt.title("DBS Histogram")

                wandb_run.log({"DBS Histogram": wandb.Image(plt)})

                plt.close()

                data = [[_, dbs_values[_]] for _ in
                        range(0, len(dbs_values), args.val_every_ep)]
                table = wandb.Table(data=data, columns=["Episode", "DBS"])
                wandb_run.log({f"DBS Table of {args.val_every_ep}": wandb.plot.bar(table, "Episode", "DBS")})

            print(f"\n\n====\nA3C training finished!\n====\n")
    else:
        agent = getattr(models, args.agent)(**agent_parameters)

        agent.initialize_env(env)

        agent.env = RecordVideo(agent.env, episode_trigger=lambda t: t % args.val_every_ep == 0,
                                video_folder=video_dir,
                                name_prefix=f"video_{agent_name}")

        if agent_name == "DQN":
            device = torch.device("cuda" if torch.cuda.is_available()
                                  else ("mps" if torch.backends.mps.is_available()
                                        else "cpu")
                                  )
            agent.to(device)
            target_network = copy.deepcopy(agent)
            dq_learning(target_network=target_network, policy_network=agent, **train_args)
        else:
            agent.train_step(**train_args)

    if not args.no_log_to_wandb:
        run.finish()


def main(args):
    # run wandb sweep to tune hyperparameters
    args.sweep_id = None
    if args.tune_hyperparameters:
        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(
            sweep=sweep_config, entity=args.entity, project=args.project)
        sweep_config.update(vars(args))
        args.sweep_id = sweep_id
        wandb.agent(sweep_id, partial(train, args), count=args.sweep_count)
    else:
        train(args=args)


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_log_to_wandb",
                        action="store_true", default=False)
    parser.add_argument('--eps', default=0.05, type=float,
                        help='Probability of choosing random action')
    parser.add_argument('--lr', default=0.000025,
                        type=float, help='Learning Rate')
    parser.add_argument('--gamma', default=0.99,
                        type=float, help='Discounting Factor')
    parser.add_argument("--n_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--val_every_ep", type=int, default=100)
    parser.add_argument("--var_threshold", type=float, default=100)
    parser.add_argument("--n_steps", type=int, default=N_STEPS,
                        help="Max number of steps per episode")
    parser.add_argument("--patience", type=int,
                        default=PATIENCE, help="Patience for early stopping")
    parser.add_argument("--tune_hyperparameters",
                        action="store_true", default=False)
    parser.add_argument("--project", type=str, default="ipomoea-aquatica")
    parser.add_argument("--entity", type=str, default="ipomoea_aquatica3")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sweep_config", type=str,
                        default=osp.join(os.getcwd(), "sweep.yaml"))
    parser.add_argument("--sweep_count", type=int, default=10)
    parser.add_argument("--experiment_dir", type=str,
                        default=osp.join(os.getcwd(), "experiments"))
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--agent", type=str, default="DQN",
                        choices=["QLearning", "DQN", "A3C"])
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DQN optimization")
    parser.add_argument("--target_update_freq", type=int,
                        default=10, help="Frequency to update the target network")
    parser.add_argument("--frame_skip", type=int, default=4,
                        help="Number of frames to skip in FrameSkipEnv")
    parser.add_argument("--normalize_reward", action="store_true",
                        default=False, help="Normalize agent reward")
    parser.add_argument("--eps_start", type=float, default=1,
                        help="Starting value of epsilon for epsilon-greedy policy")
    parser.add_argument("--eps_end", type=float, default=0.01,
                        help="Ending value of epsilon for epsilon-greedy policy")
    parser.add_argument("--decay_steps", type=int, default=1_000_000,
                        help="Number of steps over which epsilon is decayed")
    parser.add_argument("--memory_capacity", type=int, default=1_000_000,
                        help="Capacity of the replay memory")
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--replay_start_size", type=int, default=50_000,
                        help="Number of processed frames before learning")
    parser.add_argument("--held_out_ratio", type=float, default=0.1,
                        help="Probability of putting a state into the hold-out set in the DQN class")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Threshold for Q-values saturation")
    parser.add_argument("--tmax", type=int, default=20, help="Maximum steps for A3C updates")
    parser.add_argument("--beta", type=float, default=0.01, help="Entropy regularization factor")
    parser.add_argument("--n_workers", type=int, default=os.cpu_count(), help="Number of workers for A3C")
    parser.add_argument("--n_vcpus", type=int, default=os.cpu_count(), help="Number of vCPUS for A3C")
    parser.add_argument("--n_threads", type=Optional[int], default=None, help="Number of threads for vCPUS for A3C")

    return parser


if __name__ == '__main__':
    main(argument_parser().parse_args())
