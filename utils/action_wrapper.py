from gymnasium import ActionWrapper

from .constants import FIRE, RIGHTFIRE, LEFTFIRE


class Action(ActionWrapper):
    """
    Custom wrapper to handle player actions and notify when the agent fires a shot.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        """
        Overrides the step method to notify when the agent shoots.
        The agent shooting corresponds to specific action IDs (1, 4, or 5).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.env.get_wrapper_attr('set_last_action')(action)

        if action in {FIRE, RIGHTFIRE, LEFTFIRE}:
            if self.env.has_wrapper_attr('notify_agent_shot'):
                self.env.get_wrapper_attr('notify_agent_shot')()

        if terminated or truncated:
            if self.env.has_wrapper_attr('reset_score'):
                self.env.get_wrapper_attr('reset_score')()

        return obs, reward, terminated, truncated, info
