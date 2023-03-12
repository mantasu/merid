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
        save_last=False,
        every_n_epochs=1,
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )

def seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# TODO: Add augmentation