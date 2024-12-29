import os
import random

import cv2
import numpy as np

from utils.constants import PLAYER_COLOR, EARTH_COLOR, INVADERS_COLOR, SHIELD_COLOR, MOTHERSHIP_COLOR


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def remove_region_from_frame(frame, elem_pos):
    """
    Removes specified regions from the frame.
    """
    return np.delete(frame, elem_pos[0], axis=0)


def determine_bounding_box(frame, pos):
    min_row, max_row = np.min(pos[0]), np.max(pos[0])
    min_col, max_col = np.min(pos[1]), np.max(pos[1])

    return frame[min_row:max_row + 1, min_col:max_col + 1]


def positions_match(pos1, pos2):
    """
    Checks if two position tuples are identical.
    """
    return np.array_equal(pos1[0], pos2[0]) and np.array_equal(pos1[1], pos2[1])


def get_frame(obs):
    """
    Retrieves the frame from the observation.
    """
    return obs[0] if isinstance(obs, tuple) else obs


def calculate_frame_difference(frame1, frame2):
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frames must have same shape. frame1.shape: {frame1.shape} - frame2.shape{frame2.shape}")

    return cv2.absdiff(frame1, frame2)


def remove_score_and_earth(frame):
    """
    Removes score and earth regions from the frame.
    """
    # Remove score region
    score_pos = np.where(np.all(frame == PLAYER_COLOR, axis=-1))
    frame = np.delete(frame, score_pos[0][score_pos[0] <= 19], axis=0)

    # Remove earth region
    earth_pos = np.where(np.all(frame == EARTH_COLOR, axis=-1))
    frame = np.delete(frame, earth_pos[0], axis=0)

    return frame


def get_element_pos(frame, color, filter_earth_and_score=False, condition="equal", return_frame=False):
    """
    Extracts positions of a specific color in the frame and optionally removes unwanted regions.
    """
    if filter_earth_and_score:
        frame = remove_score_and_earth(frame)

    # Find positions matching or differing from the color
    if condition == "equal":
        mask = np.all(frame == color, axis=-1)
    elif condition == "not_equal":
        mask = np.all(frame != color, axis=-1)
    else:
        raise ValueError(f"Invalid condition: {condition}")

    elem_pos = np.where(mask)
    # elem_pos = (np.unique(elem_pos[0]), np.unique(elem_pos[1]))

    if elem_pos[0].size == 0 and elem_pos[1].size == 0:
        elem_pos = None

    if return_frame:
        frame = remove_region_from_frame(frame, elem_pos)
        return frame, elem_pos

    return elem_pos


def get_player_pos(current_frame, return_frame=False):
    """
    Identifies the player's position and optionally returns the updated frame.
    """
    return get_element_pos(current_frame, PLAYER_COLOR, filter_earth_and_score=True,
                           return_frame=return_frame)


def get_invaders_pos(current_frame, return_frame=False):
    """
    Identifies the invaders' positions (non-black elements) and optionally returns the updated frame.
    """
    return get_element_pos(current_frame, INVADERS_COLOR, return_frame=return_frame)


def get_shields(frame, return_frame=False):
    """
    Identifies the shield positions and optionally returns the updated frame.
    """
    return get_element_pos(frame, SHIELD_COLOR, return_frame=return_frame)


def get_mothership_pos(frame, return_frame=False):
    """
    Identifies the mother ship position and optionally returns the updated frame.
    """
    return get_element_pos(frame, MOTHERSHIP_COLOR, return_frame=return_frame)