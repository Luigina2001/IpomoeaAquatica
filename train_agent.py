import os
import time
import yaml
import torch
import wandb
import models
import ale_py
import argparse
import os.path as osp
import gymnasium as gym

from functools import partial
from utils import seed_everything
from gymnasium.wrappers import RecordVideo
from utils.constants import N_EPISODES, N_STEPS, PATIENCE

gym.register_envs(ale_py)


def episode_trigger(t):
    return True


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

        run_name = args.run_name if args.run_name else agent_name.lower()
        run = wandb.init(entity=args.entity,
                         project=args.project, name=run_name)
        experiment_dir = str(
            osp.join(experiment_dir if experiment_dir else args.experiment_dir, run.id))
    else:
        experiment_dir = str(
            osp.join(experiment_dir if experiment_dir else args.experiment_dir, str(time.time())))

    os.makedirs(experiment_dir, exist_ok=True)

    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")

    video_dir = os.path.join(experiment_dir, f"video/")
    os.makedirs(video_dir, exist_ok=True)

    agent_parameters = {
        "env": env,
        "lr": args.lr,
        "gamma": args.gamma,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "decay_steps": args.decay_steps
    }

    train_args = {
        "n_episodes": args.n_episodes,
        "max_steps": args.n_steps,
        "wandb_run": run,
        "video_dir": video_dir,
        "checkpoint_dir": experiment_dir,
        "val_every_ep": args.val_every_ep
    }

    if agent_name == "QLearning":
        agent_parameters.update({
            'normalize_reward': args.normalize_reward
        })

        train_args.update({
            'var_threshold': args.var_threshold,
        })
    elif agent_name == "DQN":
        agent_parameters.update({
            "memory_capacity": args.memory_capacity
        })

        train_args.update({
            'batch_size': args.batch_size,
            'patience': args.patience,
            'replay_start_size': args.replay_start_size,
            'target_update_freq': args.target_update_freq
        })

    if args.tune_hyperparameters:
        agent_parameters["lr"] = wandb.config['lr']
        agent_parameters["gamma"] = wandb.config['gamma']
        agent_parameters["eps_start"] = wandb.config['eps_start']

    agent = getattr(models, args.agent)(**agent_parameters)

    episode_trigger_func = episode_trigger

    if agent_name != "DQN":
        episode_trigger_func = lambda t: t % 100 == 0

    agent.env = RecordVideo(agent.env, episode_trigger=episode_trigger_func, video_folder=video_dir,
                            name_prefix=f"video_{agent_name}")

    if agent_name == "DQN":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        agent.to(device)

    agent.train_step(**train_args)

    if not args.no_log_to_wandb:
        run.finish()


def main(args):
    # run wandb sweep to tune hyperparameters
    if args.tune_hyperparameters:
        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(
            sweep=sweep_config, entity=args.entity, project=args.project)
        sweep_config.update(vars(args))
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
    parser.add_argument("--entity", type=str, default="ipomoea-aquatica")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sweep_config", type=str,
                        default=osp.join(os.getcwd(), "sweep.yaml"))
    parser.add_argument("--sweep_count", type=int, default=10)
    parser.add_argument("--experiment_dir", type=str,
                        default=osp.join(os.getcwd(), "experiments"))
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--agent", type=str, default="DQN",
                        choices=["QLearning", "DQN"])
    parser.add_argument("--every_visit", action="store_true", default=False,
                        help="Boolean to discern between first-visit and every-visit Monte Carlo methods")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DQN optimization")
    parser.add_argument("--target_update_freq", type=int,
                        default=10, help="Frequency to update the target network")
    parser.add_argument("--frame_skip", type=int, default=3,
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
    parser.add_argument("--replay_start_size", type=int, default=50_000,
                        help="Number of processed frames before learning")

    return parser


if __name__ == '__main__':
    main(argument_parser().parse_args())
