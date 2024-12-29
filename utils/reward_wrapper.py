import numpy as np
from gymnasium import RewardWrapper
from .constants import RIGHT, LEFT, RIGHTFIRE, LEFTFIRE


class Reward(RewardWrapper):
    def __init__(self, env, normalize_reward=False):
        super().__init__(env)
        self.last_lives = None
        self.last_enemy_positions = None
        self.shields_hit_by = None
        self.player_pos = None
        self.invaders_pos = None
        self.shields = None
        self.invaders_matrix = None
        self.prev_invaders_matrix = None
        self.score = 0
        self.time_step = 0
        self.last_action = None
        self.normalize_reward = normalize_reward

    def update_reward_info(self, shields_hit_by, player_pos, shields, invaders_pos, invaders_matrix):
        """
        Updates game state variables.
        """
        self.shields_hit_by = shields_hit_by
        self.player_pos = player_pos
        self.invaders_pos = invaders_pos
        self.shields = shields
        self.invaders_matrix = invaders_matrix

    def reset_score(self):
        self.score = 0
        self.time_step = 0

    def set_last_action(self, action):
        self.last_action = action

    def reward(self, reward):
        """
        Modifies the reward system based on the player's actions and game state.
        """
        current_lives = self.env.unwrapped.ale.lives()
        self.score += reward

        # 1. Penalize loss of lives
        if self.last_lives is None:
            self.last_lives = current_lives
        elif current_lives < self.last_lives:
            reward = -100
            self.last_lives = current_lives

        # 2. Penalize if the player hits the shields
        if self.shields and self.shields_hit_by and self.shields_hit_by == "player":
            reward -= 25
        # 3. Reward for using shields as protection
        elif self.shields and self.shields_hit_by and self.player_pos is not None:
            shield_rows, shield_cols = self.shields
            player_row, player_col_min = np.min(self.player_pos[0]), np.min(self.player_pos[1])
            player_col_max = np.max(self.player_pos[1])

            # Check if the player is under the shield
            is_under_shield = np.any((shield_cols >= player_col_min) & (shield_cols <= player_col_max))

            if is_under_shield and self.shields_hit_by == "invader":
                reward += 20

        # 4. Penalize invader advancement toward the player
        invaders_penalty = 0

        # Time-based scaling factor: starts high and decreases over time
        time_factor = max(0.2, 1.0 - self.time_step * 0.001)  # Minimum factor is 0.2
        self.time_step += 1

        # If the invaders' matrix is not None, calculate a progressive penalty
        # based on how close the invaders are to the player

        # Additional penalty if invaders have advanced relative to the previous frame
        if self.prev_invaders_matrix is not None:
            current_max_row = np.max(self.invaders_matrix[0])
            prev_max_row = np.max(self.prev_invaders_matrix[0])

            if current_max_row > prev_max_row:
                invader_rows = np.any(self.invaders_matrix, axis=1)
                for idx, row in enumerate(invader_rows):
                    if row:
                        # Penalty for invaders advancing
                        # Penalty increases for rows closer to the player
                        invaders_penalty += (self.invaders_matrix.shape[0] - idx) * time_factor
                reward -= invaders_penalty
            # invaders_penalty += 10 * time_factor

        self.prev_invaders_matrix = np.copy(self.invaders_matrix)

        if self.last_action in {RIGHT, LEFT, RIGHTFIRE, LEFTFIRE}:
            reward += 3
        # elif self.last_action == 0:
        #     reward -= 1000

        if self.normalize_reward:
            reward = (reward + 100)/(203 + 100)

        return {"score": self.score, "reward": reward}
