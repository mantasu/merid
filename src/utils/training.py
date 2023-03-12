import os
import torch
import random
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint

def compute_gamma(num_epochs: int, start_lr: float = 1e-3, end_lr: float = 5e-5) -> float:
    if num_epochs < 1:
        return 1
    
    return (end_lr / start_lr) ** (1 / num_epochs)

def get_checkpoint_callback(model_name: str = "my-model") -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath="checkpoints",
        filename=model_name + "-{epoch:02d}",
        every_n_epochs=1,
        monitor="val_loss",
        mode="min"
    )