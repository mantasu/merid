import sys
sys.path.append("src/architectures")
sys.path.append("src/datasets")

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A

from collections import OrderedDict
from torch.utils.data import random_split
from general_dataset import GeneralDataset
from remglass.segmentation import ResnetGeneratorMask
from remglass.domain_adaption import DomainAdapter, PatchGAN


def load_single(config):
    match config.pop("name"):
        # MODULES
        case "DomainAdapter":
            return DomainAdapter(**config)
        case "PatchGAN":
            return PatchGAN(**config)
        case "ResnetGeneratorMask":
            return ResnetGeneratorMask(**config)
        
        # OPTIMIZERS
        case "Adam":
            return lambda x: optim.Adam(x, **config)
        case "AdamW":
            return lambda x: optim.Adam(x, **config)
        
        # CRITERIONS
        case "MSE":
            return nn.MSELoss(**config)
        case "BCEWithLogits":
            if (k := "pos_weight") in config.keys():
                config[k] = torch.tensor(config[k])
            
            return nn.BCEWithLogitsLoss(**config)
        
        # ALBUMENTATIONS
        case "Resize":
            return A.Resize(**config, interpolation=cv2.INTER_NEAREST)
        case "HorizontalFlip":
            return A.HorizontalFlip(**config)
        case "VerticalFlip":
            return A.VerticalFlip(**config)
        case "GridDistortion":
            return A.GridDistortion(**config)
        case "RandomBrightnessContrast":
            if (k := "brightness_limit") in config.keys():
                config[k] = tuple(config[k])
            
            if (k := "contrast_limit") in config.keys():
                config[k] = tuple(config[k])
            
            return A.RandomBrightnessContrast(**config)
        case "GaussNoise":
            return A.GaussNoise(**config)
        
        # DATASETS
        case "GeneralDataset":
            return GeneralDataset(**config)

def load_multi(config):
    return dict(map(lambda kv: (kv[0], load_single(kv[1])), config.items()))

def load_weights(config, weight_fn=None):
    weights = {}
    loaded_checkpoints = {}

    for key, val in config.items():
        if isinstance(val, str):
            weights[key] = torch.load(val)
        elif isinstance(val, list):
            if val[0] not in loaded_checkpoints.keys():
                checkpoint = torch.load(val[0])

                if weight_fn is not None:
                    checkpoint = weight_fn(checkpoint)
                
                loaded_checkpoints[val[0]] = checkpoint
            
            weights[key] = loaded_checkpoints[val[0]][val[1]]
        else:
            weights[key] = val
    
    return weights

def init_module_weights(modules, weights):
    loaded_modules = {}

    for name, module in modules.items():
        if name in weights.keys():
            module.load_state_dict(weights[name])
        
        if (k := f"freeze_{name}") in weights.keys() and weights[k]:
            for param in module.parameters():
                param.requires_grad = False
        
        loaded_modules[name] = module
    
    return loaded_modules

def parse_model_config(config, weight_fn=None, flatten=False):
    parsed_config = {}
    
    if "modules" in config.keys():
        modules = load_multi(config["modules"])
        parsed_config["modules"] = modules
    
    if "weights" in config.keys() and "modules" in config.keys():
        weights = load_weights(config["weights"], weight_fn)
        modules = init_module_weights(modules, weights)
        parsed_config["modules"] = modules

    if "training" in config.keys():
        training = load_multi(config["training"])
        parsed_config["training"] = training
    
    if flatten:
        flat_config = {}

        for value in parsed_config.values():
            for key, val in value.items():
                flat_config[key] = val
        
        parsed_config = flat_config
    
    return parsed_config

def create_transforms(configs):
    transform = A.Compose([
       load_single(transform_config) for transform_config in configs
    ], additional_targets={"image1": "image", "mask1": "mask"})
    return transform
    

def parse_data_config(config):
    dataset = config["dataset"].copy()
    
    val_size = dataset.pop("val_size", None)
    split_seed = dataset.pop("split_seed", 42)
    transforms_train = config.get("transforms_train", [])
    transforms_val = config.get("transforms_val", [])

    if "img_size" in dataset.keys():
        [height, width] = dataset.pop("img_size")
        albumentation = {"name": "Resize", "height": height, "width": width}
        transforms_train.insert(0, albumentation.copy())
        transforms_val.insert(0, albumentation.copy())
    
    if transforms_train != []:
        transforms_train = create_transforms(transforms_train)
    else:
        transforms_train = None
    
    if transforms_val != []:
        transforms_val = create_transforms(transforms_val)
    else:
        transforms_val = None
    
    dataset["transforms_train"] = transforms_train
    dataset["transforms_val"] = transforms_val

    dataset = load_single(dataset)

    val_size = int(len(dataset) * val_size)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(split_seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator)
    train_set.is_val = False
    val_set.is_val = True

    parsed_data_config = {
        "train_dataset": train_set,
        "val_dataset": val_set
    }

    if "dataloader" in config.keys():
        parsed_data_config["loader_kwargs"] = config["dataloader"].copy()
    
    return parsed_data_config

def fix_weights(ckpt: dict) -> dict:
    """Prepares the weights to be loaded from the original module

    If DomainAdapter is created in the same way as described in the
    paper (uses pretrained normalised VGG and 6 ResNet blocks), and the
    original state dictionary is used to initialize its parameters, then
    this fixes the pretrained state dictionary to contain only the
    relevant weights for the domain adapter. In the original repository,
    weights of subsequent VGG layers were saved, even though they were
    not used in `forward` method.

    Args:
        ckpt (dict): The original loaded checkpoint
    
    Returns:
        dict: A modified checkpoint at 'DA' entry
    """
    # Init the DA modules
    modules = OrderedDict()
    
    # Copy over the initial VGG encodings but change the naming
    modules["vgg_encoding.0.weight"] = ckpt["DA"]["enc_1.0.weight"]
    modules["vgg_encoding.0.bias"] = ckpt["DA"]["enc_1.0.bias"]
    modules["vgg_encoding.2.weight"] = ckpt["DA"]["enc_1.2.weight"]
    modules["vgg_encoding.2.bias"] = ckpt["DA"]["enc_1.2.bias"]

    for key in ckpt["DA"].keys():
        if "enc_" not in key:
            # Only keep non-enc modules
            modules[key] = ckpt["DA"][key]
    
    # Reassign DA state
    ckpt["DA"] = modules

    return ckpt
        