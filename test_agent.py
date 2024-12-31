import os
import torch
import ale_py
import argparse
import os.path as osp

import gymnasium as gym

from gymnasium.wrappers import RecordVideo

import models
import time
from utils.constants import N_EPISODES, N_STEPS

from utils.functions import seed_everything

gym.register_envs(ale_py)


def test(args):
    evaluation_dir = str(os.path.join(args.evaluation_dir, args.agent + str(time.time())))

    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)

    seed_everything(seed=args.seed)

    print(f"Seed everything to: {args.seed}\n"f"Evaluation dir: {evaluation_dir} \n")

    if args.checkpoint_path:
        print(f"Loading agent {args.agent} from checkpoint {args.checkpoint_path}\n")

    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")

    video_dir = os.path.join(evaluation_dir, f"video/")
    os.makedirs(video_dir, exist_ok=True)
    env = RecordVideo(env, video_folder=video_dir, name_prefix=f"video_{args.agent}_test")

    agent = getattr(models, args.agent).load_model(env, args.checkpoint_path)

    if args.agent == "DQN":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        agent.to(device)

    total_rewards = []
    total_scores = []
    for episode in range(args.n_episodes):
        done = False
        current_state, _ = agent.env.reset()
        cumulative_reward = 0
        reward = 0
        score = 0
        while not done:
            action = agent.policy(current_state)
            next_state, score_info, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated

            if isinstance(score_info, dict):
                reward = score_info['reward']
                score = score_info['score']
            else:
                reward = score_info
                score = reward

            cumulative_reward += reward
            current_state = next_state

        print(f"Episode {episode + 1}: Total Reward: {cumulative_reward} --- Game Score: {score}")
        total_rewards.append(cumulative_reward)
        total_scores.append(score)

    print(f"Average Reward: {sum(total_rewards) / len(total_rewards)}")
    print(f"Average Game Score: {sum(total_scores) / (args.n_episodes)}")


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--n_steps", type=int, default=N_STEPS,
                        help="Max number of steps per episode")
    parser.add_argument("--evaluation_dir", type=str,
                        default=osp.join(os.getcwd(), "evaluation_dir"))
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--agent", type=str, default="DQN",
                        choices=["QLearning", "DQN"])
    parser.add_argument("--every_visit", action="store_true", default=False,
                        help="Boolean to discern between first-visit and every-visit Monte Carlo methods")

    return parser


if __name__ == '__main__':
    test(argument_parser().parse_args())