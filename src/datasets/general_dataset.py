import os
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

class GeneralDataset(Dataset):
    def __init__(self, path_x=None, path_y=None, use_labels=False,
                 transforms_train=None, transforms_val=None):
        """Initializes the dataset

        Args:
            path_x (str, optional): The path to the first folder with
                images or `None` if only second path is used. Defaults
                to None.
            path_y (str, optional): The path to the second folder with
                images or `None` if only first path is used. Defaults
                to None.
            use_labels (bool, optional): Whether to include glasses and 
                shadows masks as labels of the synthetic dataset if it
                is specified as path_y. Defaults to False.
            transforms_train (albumentations.Compose): The transform to
                apply to the train images (and masks). Defaults to None.
            transforms_val (albumentations.Compose): The transform to
                apply to the val images (and masks). Defaults to None.
        """
        # Assign the arguments
        self.path_x = path_x
        self.path_y = path_y
        self.use_labels = use_labels
        self.transforms_train = transforms_train
        self.transforms_val = transforms_val

        # Auxilary parameters
        self.is_train = False
        self.t = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        if path_x is not None:
            # If the 1st path is not none, read it
            self.image_paths_x = os.listdir(path_x)
        
        if path_y is not None:
            # If the 2nd path is not none, read it
            self.image_paths_y = os.listdir(path_y)
        
        if path_x is None and path_y is None:
            raise ValueError("At least one path must be specified!")
        
        if use_labels and path_y is None:
            raise ValueError("2nd path must be specified for labels!")

    def __getitem__(self, idx):
        transform_kwargs, aug = {}, None

        if self.path_x is not None:
            image_path_x = os.path.join(self.path_x, self.image_paths_x[idx])
            image_x = cv2.cvtColor(cv2.imread(image_path_x), cv2.COLOR_BGR2RGB)
            transform_kwargs["image"] = image_x
        
        if self.path_y is not None:
            image_path_y = os.path.join(self.path_y, self.image_paths_y[idx])
            image_y = cv2.cvtColor(cv2.imread(image_path_y), cv2.COLOR_BGR2RGB)
            
            if self.path_x is None:
                transform_kwargs["image"] = image_y
            else:
                transform_kwargs["image1"] = image_y
        
        if self.use_labels:
            mask_path_a = image_path_y.replace("train_x", "train_y").replace("-all", "-seg")
            mask_path_b = image_path_y.replace("train_x", "train_y").replace("-all", "-shseg")
            mask_a = cv2.imread(mask_path_a, cv2.IMREAD_GRAYSCALE)
            mask_b = cv2.imread(mask_path_b, cv2.IMREAD_GRAYSCALE)
            transform_kwargs["mask"] = mask_a
            transform_kwargs["mask1"] = mask_b
        
        if not self.is_train and self.transforms_val is not None:
            aug = self.transforms_val(**transform_kwargs)
        elif self.transforms_train is not None:
            aug = self.transforms_train(**transform_kwargs)
        
        if aug is not None:
            for key in transform_kwargs.keys():
                if "image" in key:
                    val = self.t(Image.fromarray(aug[key]))
                else:
                    val = torch.from_numpy(aug[key]).bool().float()
                
                transform_kwargs[key] = val
        
        return tuple(transform_kwargs.values())

    def __len__(self):
        if self.path_x is None:
            return len(self.image_paths_y)
        elif self.path_y is None:
            return len(self.image_paths_x)
        else:
            return min(len(self.image_paths_x), len(self.image_paths_y))
    
    def set_train(self, is_train=True):
        self.is_train = True