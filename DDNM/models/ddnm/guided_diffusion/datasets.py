import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class DummyDataset(Dataset):
    def __init__(self, image, image_size):
        super().__init__()
        self.image = image

        # self.transforms = T.Compose([
        #     T.ToTensor(),
        #     T.Resize(image_size),
        # ])
    
    def __getitem__(self, index):
        # image = self.transforms(self.image)
        image = self.image
        return image, torch.zeros_like(image)
    
    def __len__(self):
        return 1

def get_dataset(args, config):
    return None, DummyDataset(config.image, config.data.image_size)

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)