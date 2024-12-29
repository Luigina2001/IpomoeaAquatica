import os 
import torch
import random 
import numpy as np 

from .constants import PATIENCE

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



class EarlyStopping: 
    def __init__(self, metric_to_track: str, objective: str, checkpoint_dir: str, checkpoint_ext: str = "ckpt", patience: int = PATIENCE, trace_func=print):

        if objective not in ["minimize", "maximize"]:
            raise ValueError("Objective can only be 'maximize' or 'minimize'.")

        self.counter = 0
        self.patience = patience 
        self.objective = objective 
        self.trace_fun = trace_func
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_ext = checkpoint_ext
        self.metric_to_track = metric_to_track
        self.best_value = float('inf' if self.objective == 'minimize' else '-inf')

    def __call__(self, metric_value, model, episode):
        has_improved = (
            (self.objective == "minimize" and metric_value < self.best_value) or
            (self.objective == "maximize" and metric_value > self.best_value)
        )

        if has_improved:
            self.counter = 0
            self.best_value = metric_value
            self.save_checkpoint(metric_value, model, episode)
        else: 
            self.counter += 1

            if self.counter >= self.patience: 
                self.trace_func( f"{self.metric_to_track} did not improve for {self.counter} episodes. Stopping...")
                return True 
        return False 

    def save_checkpoint(self, metric_value, model, episode):
        model_name = os.path.dirname(
            self.checkpoint_dir).split(os.path.sep)[-1]
        checkpoint_filename = f"{model_name}_ep_{episode}_{self.metric_to_track}_{metric_value}.{self.checkpoint_ext}"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

        self.trace_fun(f"Saving model checkpoint to: {checkpoint_path}...")

        model.save_model(checkpoint_path)

        self.trace_fun("Model checkpoint saved!")

        # remove all previous checkpoints
        for filename in os.listdir(self.checkpoint_dir):
            if filename != checkpoint_filename and filename.endswith(self.checkpoint_ext):
                os.remove(os.path.join(self.checkpoint_dir, filename))
