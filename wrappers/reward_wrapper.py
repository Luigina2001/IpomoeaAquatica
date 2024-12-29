import numpy as np
from gymnasium import RewardWrapper
from utils.constants import RIGHT, LEFT, RIGHTFIRE, LEFTFIRE


MAX_REWARD = 200
MIN_REWARD = -100


class Reward(RewardWrapper):
    def __init__(self, env, normalize_reward: bool = False):
        super().__init__(env)

        self.score = 0
        self.last_lives = 3
        self.player_pos = None
        self.shields_pos = None
        self.invaders_pos = None
        self.shields_hit_by = None
        self.invaders_matrix = None
        self.prev_invaders_matrix = None
        self.og_invaders_player_dist = None
        self.normalize_reward = normalize_reward
        self.got_player_head = False

    def update_reward_info(self, shields_hit_by, player_pos, shields_pos, invaders_pos, invaders_matrix):
        """
        Updates game state variables.
        """
        self.shields_hit_by = shields_hit_by
        self.player_pos = player_pos
        self.invaders_pos = invaders_pos
        self.shields_pos = shields_pos
        self.invaders_matrix = invaders_matrix

        if not self.got_player_head and self.player_pos is not None:
            self.player_head = np.max(self.player_pos[0])
            closest_invader_pos = np.max(self.invaders_matrix)
            self.og_invaders_player_dist = self.player_head - closest_invader_pos
            self.got_player_head = False

    def reset_score(self):
        self.score = 0

    def set_last_action(self, action):
        self.last_action = action

    def reward(self, reward):
        """
        Modifies the reward system based on the player's actions and game state.
        """

        self.score += reward

        current_lives = self.env.unwrapped.ale.lives()
        # 1. Penalize loss of lives
        if current_lives < self.last_lives:
            reward = -100
            self.last_lives = current_lives

        # 2. Penalize if the player hits the shields
        if self.shields_pos is not None and self.shields_hit_by:
            if self.shields_hit_by == "player":
                reward -= 25

        # 3. Penalize invader advancement toward the player

        # If the invaders' matrix is not None, calculate a progressive penalty
        # based on how close the invaders are to the player

        # Additional penalty if invaders have advanced relative to the previous frame
        if self.prev_invaders_matrix is not None:
            invaders_penalty = 0
            closest_invader_pos = np.max(self.invaders_matrix)
            prev_closest_invader_pos = np.max(self.prev_invaders_matrix)

            if closest_invader_pos > prev_closest_invader_pos:
                invaders_player_dist = self.player_head - closest_invader_pos
                invaders_penalty = np.count_nonzero(
                    self.invaders_matrix) + (invaders_player_dist - self.og_invaders_player_dist)
                reward -= invaders_penalty
                

        self.prev_invaders_matrix = np.copy(self.invaders_matrix)

        if self.normalize_reward:
            reward = (reward - MIN_REWARD) / (MAX_REWARD -
                                              MIN_REWARD)  # minmax normalization

        return {"score": self.score, "reward": reward}
