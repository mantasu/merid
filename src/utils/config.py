import sys
sys.path.append("src/architectures")
sys.path.append("src/datasets")

import cv2
import inspect
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from typing import Type, Any

from .io_and_types import load_json
from copy import deepcopy
from collections import OrderedDict
from torch.utils.data import random_split
from general_dataset import GeneralDataset
from pesr.segmentation import ResnetGeneratorMask
from pesr.domain_adaption import DomainAdapter, PatchGAN
from glasses_labeled_dataset import GlassesLabeledDataset
from glasses_and_no_glasses_mix_dataset import GlassesAndNoGlassesMixDataset
from labeled_and_non_labeled_mix_dataset import LabeledAndNonLabeledMixDataset

AVAILABLE_CLASSES = dict((
    *inspect.getmembers(A, inspect.isclass),
    *inspect.getmembers(nn, inspect.isclass),
    *inspect.getmembers(optim, inspect.isclass),
    ("ResnetGeneratorMask", ResnetGeneratorMask),
    ("DomainAdapter", DomainAdapter),
    ("PatchGAN", PatchGAN),
    ("GeneralDataset", GeneralDataset),
    ("GlassesLabeledDataset", GlassesLabeledDataset),
    ("GlassesAndNoGlassesMixDataset", GlassesAndNoGlassesMixDataset),
    ("LabeledAndNonLabeledMixDataset", LabeledAndNonLabeledMixDataset)
))

def verify_types(config: dict[str, Any]) -> dict[str, Any]:
    if "Resize" in config.keys():
        config["interpolation"] = cv2.INTER_NEAREST
    
    if (k := "pos_weight") in config.keys():
        config[k] = torch.tensor(config[k])
    
    if (k := "brightness_limit") in config.keys():
        config[k] = tuple(config[k])
    
    if (k := "contrast_limit") in config.keys():
        config[k] = tuple(config[k])
    
    return config

def load_single(config: dict[str, Any]) -> Type | callable:
    # Verify config, pop name, rest is kwargs
    config = verify_types(deepcopy(config))
    class_name = config.pop("name")
    optim_names = [x for x, _ in inspect.getmembers(optim, inspect.isclass)]

    if class_name in optim_names:
        # If class is optim, return lambda (needs model parameters)
        return lambda x: AVAILABLE_CLASSES[class_name](x, **config)
    
    return AVAILABLE_CLASSES[class_name](**config)

def load_multi(config: dict[str, Any]) -> dict[str, Any]:
    return dict(map(lambda kv: (kv[0], load_single(kv[1])), config.items()))

def load_weights(config: dict[str | list[str]],
                 weight_fn: callable = None) -> \
                 dict[str, dict | OrderedDict]:
    def recurse_by_key(checkpoint, remaining_keys):
        if remaining_keys == []:
            # Ending condition
            return checkpoint
        
        return recurse_by_key(checkpoint[remaining_keys[0]], remaining_keys[1:])
    
    # Store already loaded checkpoints (in case they are reused)
    loaded = {}

    for key in config.keys():
        if isinstance(config[key], bool):
            continue

        if isinstance(config[key], str):
            # Set to list to generalize
            config[key] = [config[key]]
        
        if config[key][0] not in loaded.keys():
            # First str in a list is path to weights - load it
            loaded[config[key][0]] = torch.load(config[key][0])

            if weight_fn is not None:
                # If weight pre-processing function is not none, apply it
                loaded[config[key][0]] = weight_fn(loaded[config[key][0]])
        
        # Reassign config item to loaded weights (further specified by keys)
        config[key] = recurse_by_key(loaded[config[key][0]], config[key][1:])
    
    return config

def load_modules_with_weights(modules: dict[str, nn.Module],
                              weights: dict[str, dict | OrderedDict]) -> \
                              dict[str, nn.Module]:
    for name in modules.keys():
        if name in weights.keys():
            # Load module with corresponding weights
            modules[name].load_state_dict(weights[name])
        
        if (k := f"freeze_{name}") in weights.keys() and weights[k]:
            for param in modules[name].parameters():
                # Freeze module weights if required
                param.requires_grad = False
    
    return modules

def parse_model_config(config: dict[str, Any],
                       weight_fn: callable = None) -> dict[str, Any]:
    if "modules" in config.keys():
        # Load the modules from modules config
        modules = load_multi(config["modules"])
        config["modules"] = modules
    
    if "weights" in config.keys() and "modules" in config.keys():
        # Load the weights from weights config, use for modules
        weights = load_weights(config["weights"], weight_fn)
        modules = load_modules_with_weights(modules, weights)
        config["modules"] = modules

    if "training" in config.keys():
        # Load training props from training config
        training = load_multi(config["training"])
        config["training"] = training

    del config["weights"]
    
    # Flatten config (remove primary keys shown in if statements above)
    config = {k: v for sub in config.values() for k, v in sub.items()}

    return config

def create_transforms(configs):
    transform = A.Compose([
       load_single(transform_config) for transform_config in configs
    ], additional_targets={"image1": "image", "mask1": "mask"})
    return transform

def parse_config(config: dict[str, Any]) -> dict[str, Any]:
    # Create parsing map
    PARSING_MAP = {
        "mask_generator": lambda x: parse_model_config(x, fix_weights),
        "mask_inpainter": parse_model_config,
        "data": parse_data_config,
    }

    for key, _config in config.items():        
        if (macro := _config.get("macro", None)) is not None:
            # If contains macro, load it
            config[key] = load_json(macro)
        
        # Parse current config and reassign
        config[key] = PARSING_MAP[key](_config)
    
    return config

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
        