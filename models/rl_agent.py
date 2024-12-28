import pickle
import os.path as osp

from abc import abstractmethod


class RLAgent:
    def __init__(self, env, lr, gamma, eps_start, eps_end, decay_steps, *args, **kwargs):
        self.lr = lr
        self.env = env
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.decay_steps = decay_steps
        self.decay_rate = (self.eps_start - self.eps_end) / self.decay_steps
        self.train_mode = True

    def eps_schedule(self, current_step):
        self.eps = max(self.eps_end, self.eps_start -
                       self.decay_rate * current_step)

    @abstractmethod
    def initialize_env(self, *args, **kwargs):
        raise NotImplementedError

    def serialize(self):
        model_state = {
            'std_parameters': {
                'env': self.env,
                'lr': self.lr,
                'gamma': self.gamma,
                'eps_start': self.eps_start,
                'eps_end': self.eps_end,
                'decay_steps': self.decay_steps,
            }
        }

        if hasattr(self, 'train_mode'):
            model_state.update(
                {'extra_parameters': {'train_mode': self.train_mode}})

        return model_state

    def save_model(self, checkpoint_path: str):
        model_state = self.serialize()
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(model_state, f)
        except (IOError, pickle.PickleError) as e:
            raise ValueError(f"Error saving model to {checkpoint_path}: {e}")

    @classmethod
    def load_model(cls, checkpoint_path: str, return_params: bool = False):
        if not osp.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file {checkpoint_path} not found.")

        try:
            with open(checkpoint_path, "rb") as f:
                model_state = pickle.load(f)

            instance = cls(**model_state['std_parameters'])
            instance.train_mode = model_state['extra_parameters']['train_mode']

            if return_params:
                return instance, model_state
            return instance

        except (IOError, pickle.PickleError) as e:
            raise ValueError(
                f"Error loading model from {checkpoint_path}: {e}")

    def evaluation_mode(self):
        self.train_mode = False
