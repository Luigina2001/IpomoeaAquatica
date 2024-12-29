import numpy as np
from gymnasium import ObservationWrapper

from utils.constants import INVADERS_TOTAL_HEIGHT, INVADERS_TOTAL_WIDTH, INVADER_ROW_SEPARATION_WIDTH, PROJECTILE_COLOR
from utils.functions import (
    get_element_pos,
    determine_bounding_box,
    get_frame,
    calculate_frame_difference,
    get_shields,
    get_mothership_pos,
    get_player_pos,
    get_invaders_pos,
)


class Observation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.player_pos = None 
        self.prev_frame = None
        self.bullet_pos = None
        self.did_agent_shoot = False
        self.shields_hit = False
        self.invaders_pos = None
        self.shields_hit_by = None
        self.mothership_pos = None
        self.invaders_matrix = None
        self.curr_shields_pos = None
        self.prev_shields_pos = None

    def observation(self, obs):
        curr_frame = obs[0] if isinstance(obs, tuple) else obs
        self.curr_shields_pos = None

        self.invaders_pos = get_invaders_pos(curr_frame)

        if self.invaders_pos is not None:
            invaders_region = determine_bounding_box(curr_frame, self.invaders_pos)
            # 6 x 6 = (100, 80)
            # column length = 16
            # row height = 18

            rows = int(invaders_region.shape[0] / 18) + 1
            cols = int(invaders_region.shape[1] / 16) + 1

            self.invaders_matrix = np.ones((rows, cols))

            for i in range(rows):
                row_start = i * INVADER_ROW_SEPARATION_WIDTH + i * INVADERS_TOTAL_HEIGHT
                row_end = row_start + INVADERS_TOTAL_HEIGHT
                col_end = 0
                for j in range(cols):
                    col_start = col_end + INVADERS_TOTAL_WIDTH if j > 0 else 0
                    col_end = col_start + INVADERS_TOTAL_WIDTH
                    if np.any(invaders_region[row_start:row_end, col_start:col_end]):
                        invaders_region_start = np.min(self.invaders_pos[0])
                        self.invaders_matrix[i, j] = invaders_region_start + row_end
                    else:
                        self.invaders_matrix[i, j] = 0
        else:
            self.invaders_matrix = np.zeros((6, 6))

        if self.prev_frame is not None and self.prev_shields_pos is not None:
            self.curr_shields_pos = get_shields(curr_frame)

            if self.curr_shields_pos is None:
                self.shields_hit = True
            else:
                curr_shield_bb = determine_bounding_box(curr_frame, self.curr_shields_pos)
                prev_shield_bb = determine_bounding_box(self.prev_frame, self.prev_shields_pos)

                if curr_shield_bb.shape != prev_shield_bb.shape:
                    self.shields_hit = True
                else:
                    diff = calculate_frame_difference(curr_shield_bb, prev_shield_bb)
                    self.shields_hit = np.any(diff)

            if self.shields_hit:
                self.shields_hit_by = self._determine_shields_hit_source(curr_frame)
            else:
                self.shields_hit_by = None

        self.player_pos = get_player_pos(curr_frame)
        self.mothership_pos = get_mothership_pos(curr_frame)

        if isinstance(obs, tuple):
            obs = list(obs)
            info = obs[-1]
        else:
            info = {}

        # Update observation with shield status, player and invaders position
        info.update({"shields_hit": self.shields_hit,
                     "curr_shields_pos": self.curr_shields_pos,
                     "shields_hit_by": self.shields_hit_by,
                     "player_pos": self.player_pos,
                     "invaders_pos": self.invaders_pos,
                     "mothership_pos": self.mothership_pos,
                     "invaders_matrix": self.invaders_matrix,
                     "bullet_pos": self.bullet_pos,
                     "lives": self.env.unwrapped.ale.lives()})
        
        if isinstance(obs, tuple):
            obs[-1] = info
            obs = tuple(obs)
        else:
            obs = (curr_frame, info)

        self.prev_frame = curr_frame
        self.prev_shields_pos = self.curr_shields_pos

        if self.env.has_wrapper_attr('update_reward_info'):
            self.env.get_wrapper_attr('update_reward_info')(self.shields_hit_by, self.player_pos, self.invaders_pos, self.curr_shields_pos, self.invaders_matrix)

        return obs

    def _determine_shields_hit_source(self, curr_frame,):
        """
        Determines who hit the shield: player or invaders, based on player position and shield hit event.
        """
        _, shield_cols = self.shields_pos
        _, player_col_min = np.min(self.player_pos[0]), np.min(self.player_pos[1])
        player_col_max = np.max(self.player_pos[1])

        # Check if the player is under the shield
        is_under_shield = np.any(
            (shield_cols >= player_col_min) &
            (shield_cols <= player_col_max)
        )

        # Determine the source of the hit
        if self.did_agent_shoot and is_under_shield:
            self.bullet_pos = get_element_pos(curr_frame, PROJECTILE_COLOR)

            # Check if projectile_positions_current has enough elements before accessing index 1
            if (self.bullet_pos is None) or (self.bullet_pos[1].size == 0):
                return 'player'
            else:
                projectile_columns = self.bullet_pos[1]
                prev_shield_columns = set(self.prev_shields_pos[1])
                common_columns = list(set(projectile_columns.tolist()) & prev_shield_columns)

                if common_columns:
                    self.did_agent_shoot = False
                    return 'player'
                else:
                    return 'invader'

        else:
            return 'invader'

    def notify_agent_shot(self):
        """
            Notify when player shoots.
        """
        self.did_agent_shoot = True
