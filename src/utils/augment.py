import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def unnormalize(x: torch.Tensor,
                mean: list[float, float, float] = [0.485, 0.456, 0.406], 
                std: list[float, float, float] = [0.229, 0.224, 0.225]):
    return x * std + mean

def create_default_augmentation() -> A.Compose:
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([
                A.RandomResizedCrop(256, 256, p=0.5),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1),
                A.PiecewiseAffine(),
                A.Perspective()
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.RandomGamma(gamma_limit=(80, 120)),
                A.CLAHE(clip_limit=4.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ], p=0.5),
            A.OneOf([
                A.Blur(blur_limit=3),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=3),
                A.GaussNoise(var_limit=(10.0, 50.0)),
            ], p=0.5),
            A.CoarseDropout(max_holes=5, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])