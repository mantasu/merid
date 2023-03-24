import os
import torch
import random
import numpy as np
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from .image_tools import tensor_to_image, unnormalize as unnormalize_fn


def compute_gamma(num_epochs: int, start_lr: float = 1e-3, end_lr: float = 5e-5) -> float:
    if num_epochs < 1:
        return 1
    
    return (end_lr / start_lr) ** (1 / num_epochs)

def get_checkpoint_callback(model_name: str = "my-model", monitor="val_loss") -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath="checkpoints",
        filename=model_name + "-{epoch:02d}-{" + monitor + ":.5f}'",
        every_n_epochs=1,
        monitor=monitor,
        mode="min"
    )

def get_default_transform_list() -> list[A.DualTransform]:
    return [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ShiftScaleRotate(),
        A.OneOf([
            A.RandomResizedCrop(256, 256, p=0.5),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1),
            A.PiecewiseAffine(),
            A.Perspective()
        ]),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.RandomGamma(),
            A.CLAHE(),
            A.ColorJitter(),
        ]),
        A.OneOf([
            A.Blur(blur_limit=3),
            A.GaussianBlur(),
            A.MedianBlur(),
            A.GaussNoise(),
        ]),
        A.CoarseDropout(max_holes=5, p=0.3)
    ]

def create_augmentation(
    transform_list: str | list[A.DualTransform] = "default",
    include_normalize: bool = False,
    norm_mean_and_std: tuple[tuple[float], tuple[float]] | None = None,
    include_to_tensor: bool = False,
    additional_targets: dict[str, str] | int | tuple[int, int] | None = None,
) -> A.Compose:
    if isinstance(additional_targets, int):
        additional_targets = (additional_targets, additional_targets)
    
    if isinstance(additional_targets, tuple):
        additional_targets = {
            **{f"image{i+1}": "image" for i in range(additional_targets[0])},
            **{f"mask{i+1}": "mask" for i in range(additional_targets[0])},
        }
    
    if isinstance(transform_list, str) and transform_list == "default":
        transform_list = get_default_transform_list()
    elif isinstance(transform_list, str):
        raise ValueError(f"Transform type {transform_list} is not supported.")
    
    if include_normalize and norm_mean_and_std is not None:
        transform_list.append(A.Normalize(*norm_mean_and_std))
    elif include_normalize:
        transform_list.append(A.Normalize())
    
    if include_to_tensor:
        transform_list.append(ToTensorV2())

    return A.Compose(transform_list, additional_targets=additional_targets)


def test(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
):
    trainer = pl.Trainer()
    trainer.test(model, datamodule)


def train(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    model_name: str = "my_model",
    monitored_loss_name: str = "val_loss",
    checkpoint_dir: str | os.PathLike = "checkpoints",
    ckpt_path: str | None = None,
    seed: int = 0,
    test_at_the_end: bool = True,
    **trainer_kwargs
):
    # Set up default trainer kwargs
    DEFAULT_TRAINER_KWARGS = {
        "max_epochs": 20,
        "accelerator": "gpu",
    }

    # Create a custom checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=model_name + "-{epoch:02d}-{" + monitored_loss_name + ":.5f}",
        monitor=monitored_loss_name,
        mode="min",
    )
    
    # Add the checkpoint callback as a default trainer kwarg
    DEFAULT_TRAINER_KWARGS["callbacks"] = [checkpoint_callback]
    trainer_kwargs = {**DEFAULT_TRAINER_KWARGS, **trainer_kwargs}

    # Seed everything and set precision
    seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    # Init trainer, train with datamodule
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if test_at_the_end:
        # If testing is desired after the full training
        trainer.test(ckpt_path="best", datamodule=datamodule)

    # Load the best model from the saved best checkpoint and save it
    best = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    save_path = os.path.join(checkpoint_dir, model_name + "-best.pth")
    torch.save(best.state_dict(), save_path)


def plot_results(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    num_batch_inputs: int = 2,
    weights_path: str | None = None,
    checkpoint_path: str | None = None,
    max_samples: int = 4,
    unnormalize: list[bool] = [],
    is_grayscale: list[bool] = [],
):
    batch = next(iter(datamodule.test_dataloader()))

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    else:
        model.load_from_checkpoint(checkpoint_path)

    outputs = model(*batch[:num_batch_inputs])
    batch = (*batch, outputs)

    num_rows = min(max_samples, len(outputs))
    num_cols = len(batch)

    for i in range(num_rows):
        for j, out in enumerate(batch):
            sample = out[i]

            if unnormalize != [] and unnormalize[j]:
                sample = unnormalize_fn(sample)

            image = tensor_to_image(sample)

            if is_grayscale != [] and is_grayscale[j]:
                image = image[..., None].repeat(3, axis=-1)
                
            plt.subplot(num_rows, num_cols, i * num_cols + j + 1)
            plt.imshow(image)
    
    plt.show()
