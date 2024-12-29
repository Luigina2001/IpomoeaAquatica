import numpy as np
from gymnasium import ObservationWrapper

from .constants import INVADERS_TOTAL_HEIGHT, INVADERS_TOTAL_WIDTH, INVADER_ROW_SEPARATION_WIDTH, PROJECTILE_COLOR
from .functions import (
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
        self._prev_frame = None
        self._player_pos = None
        self._invaders_pos = None
        self._shields = None
        self._mothership_pos = None
        self._bullet_pos = None
        self._shields_hit = False
        self._shields_hit_by = None
        self._last_agent_shot = False
        self._invaders_matrix = np.ones((6, 6))

    def observation(self, obs):
        """
        Processes the observation to include shield hit status and player position.
        """
        current_frame = get_frame(obs)

        self._invaders_pos = get_invaders_pos(current_frame)

        if self._invaders_pos is None:
            self._invaders_matrix = np.zeros((6, 6))
        else:
            invaders_region = determine_bounding_box(current_frame, self._invaders_pos)
            # 6x6 = (100, 80)
            # column = 16
            # row = 18
            i = 0

            rows = int(invaders_region.shape[0] / 18) + 1
            cols = int(invaders_region.shape[1] / 16) + 1

            self._invaders_matrix = np.ones((rows, cols))

            while i < rows:
                row_start = i * INVADER_ROW_SEPARATION_WIDTH + i * INVADERS_TOTAL_HEIGHT
                row_end = row_start + INVADERS_TOTAL_HEIGHT
                j = 0
                col_end = 0
                while j < cols:
                    col_start = col_end + INVADERS_TOTAL_WIDTH if j > 0 else 0
                    col_end = col_start + INVADERS_TOTAL_WIDTH
                    if np.any(invaders_region[row_start:row_end, col_start:col_end]):
                        invaders_region_start = np.min(self._invaders_pos[0])
                        self._invaders_matrix[i, j] = invaders_region_start + row_end
                    else:
                        self._invaders_matrix[i, j] = 0
                    j += 1
                i += 1

        # Check for shield hit events if a previous frame exists
        if self._prev_frame is not None:
            curr_shields_pos = get_shields(current_frame)
            prev_shields_pos = get_shields(self._prev_frame)
            self._shields = curr_shields_pos

            if curr_shields_pos is None and prev_shields_pos is not None:
                self._shields_hit = True
            elif curr_shields_pos is None and prev_shields_pos is None:
                self._shields_hit = False
            elif curr_shields_pos is not None and prev_shields_pos is None:
                self._shields_hit = False
            else:
                current_frame_shield_bounding_box = determine_bounding_box(current_frame, curr_shields_pos)
                prev_frame_shield_bounding_box = determine_bounding_box(self._prev_frame, prev_shields_pos)

                if current_frame_shield_bounding_box.shape != prev_frame_shield_bounding_box.shape:
                    self._shields_hit = True
                else:
                    diff = calculate_frame_difference(current_frame_shield_bounding_box,
                                                      prev_frame_shield_bounding_box)
                    self._shields_hit = np.any(diff)

            if self._shields_hit:
                self._shields_hit_by = self._determine_hit_source(current_frame, prev_shields_pos)
            else:
                self._shields_hit_by = None

        if isinstance(obs, tuple):
            obs = list(obs)
            info = obs[-1]
        else:
            info = {}

        self._player_pos = get_player_pos(current_frame)
        self._mothership_pos = get_mothership_pos(current_frame)

        # Update observation with shield status, player and invaders position
        info.update({"shields_hit": self._shields_hit,
                     "shields": self._shields,
                     "shields_hit_by": self._shields_hit_by,
                     "player_pos": self._player_pos,
                     "invaders_pos": self._invaders_pos,
                     "mothership_pos": self._mothership_pos,
                     "invaders_matrix": self._invaders_matrix,
                     "bullet_pos": self._bullet_pos,
                     "lives": self.env.unwrapped.ale.lives()})

        if isinstance(obs, tuple):
            obs[-1] = info
            obs = tuple(obs)
        else:
            obs = (current_frame, info)

        self._prev_frame = current_frame
        self.env.update_reward_info(self._shields_hit_by, self._player_pos, self._invaders_pos, self._shields,
                                    self._invaders_matrix)
        return obs

    def _determine_hit_source(self, frame, prev_shields_pos):
        """
        Determines who hit the shield: player or invaders, based on player position and shield hit event.
        """
        if self._shields is not None and self._player_pos is not None:
            shield_rows, shield_cols = self._shields
            player_row, player_col_min = np.min(self._player_pos[0]), np.min(self._player_pos[1])
            player_col_max = np.max(self._player_pos[1])

            # Check if the player is under the shield
            is_under_shield = np.any(
                (shield_cols >= player_col_min) &
                (shield_cols <= player_col_max)
            )

            # Determine the source of the hit
            if self._last_agent_shot and is_under_shield:
                self._bullet_pos = get_element_pos(frame, PROJECTILE_COLOR)

                # Check if projectile_positions_current has enough elements before accessing index 1
                if (self._bullet_pos is None) or (self._bullet_pos[1].size == 0):
                    return 'player'
                else:
                    projectile_columns = self._bullet_pos[1]
                    prev_shield_columns = set(prev_shields_pos[1])
                    common_columns = list(set(projectile_columns.tolist()) & prev_shield_columns)

                    if common_columns:
                        self._last_agent_shot = False
                        return 'player'
                    else:
                        return 'invader'

            else:
                return 'invader'
        return None

    def notify_agent_shot(self):
        """
          Notify when player shoots.
        """
        self._last_agent_shot = True

    @property
    def shields_hit(self):
        """
        Property to check if shields were hit in the last frame transition.
        """
        return self._shields_hit

    @property
    def shields_hit_by(self):
        """
        Property to check who hit the shield
        """
        return self._shields_hit_by