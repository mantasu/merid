import sys
sys.path.append("src/architectures")
sys.path.append("src/datasets")

import os
import cv2
import json
import torch
import inspect
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import albumentations as A
from typing import Type, Any

from . import guest
from copy import deepcopy
from models.pesr.segmentation import ResnetGeneratorMask
from models.pesr.domain_adaption import DomainAdapter, PatchGAN
from models.merid.sunglasses_classifier import SunglassesClssifier
from models.merid.sunglasses_segmenter import GlassesSegmenter
from models.trash.nafnet_denoiser_old import NAFNetArtefactRemover
from models.lafin.lafin_inpainter import LafinInpainter
from models.ddnm.ddnm_inpainter import DDNMInpainter
from models.nafnet.nafnet_denoiser import NAFNetDenoiser
from models.merid.recolorizer import Recolorizer


AVAILABLE_CLASSES = dict((
    *inspect.getmembers(A, inspect.isclass),
    *inspect.getmembers(nn, inspect.isclass),
    *inspect.getmembers(optim, inspect.isclass),
    ("ResnetGeneratorMask", ResnetGeneratorMask),
    ("DomainAdapter", DomainAdapter),
    ("PatchGAN", PatchGAN),
    ("SunglassesClassifier", SunglassesClssifier),
    ("GlassesSegmenter", GlassesSegmenter),
    ("NAFNetArtefactRemover", NAFNetArtefactRemover),
    ("LafinInpainter", LafinInpainter),
    ("DDNMInpainter", DDNMInpainter),
    ("NAFNetDenoiser", NAFNetDenoiser),
    ("Recolorizer", Recolorizer)
))

def load_json(path: str | os.PathLike) -> dict[str, Any]:
    """Loads JSON file

    Simply reads the specified file that ends with ".json" and parses
    its contents to python values to make up a python dictionary.
    
    Args:
        path (str | os.PathLike): The path to .json file
    
    Returns:
        dict[str, Any]: A python dictionary
    """
    with open(path, 'r') as f:
        # Load JSON file to py
        py_dict = json.load(f)
    
    return py_dict

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

def load_weights(config: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    def recurse_by_key(checkpoint, remaining_keys):
        if remaining_keys == []:
            # Ending condition
            return checkpoint
        
        return recurse_by_key(checkpoint[remaining_keys[0]], remaining_keys[1:])
    
    # Store already loaded checkpoints (in case they are reused)
    loaded = {}

    for module_name, weights_config in config.items():
        if (weights_path := weights_config.pop("path", None)) is None:
            # No path
            continue

        if not isinstance(weights_path, list):
            # Convert to list for convenience
            weights_path = [weights_path]

        if weights_path[0] not in loaded.keys():
            # Load the actual weights (0th element is actual path)
            loaded[weights_path[0]] = torch.load(weights_path[0])

        # Get the actual state dictionary from the loaded weights file
        weights = recurse_by_key(loaded[weights_path[0]], weights_path[1:])
        
        if (guest_fn := weights_config.pop("guest_fn", None)) is not None:
            # Apply a function to fix the weights if needed
            weights = getattr(guest, guest_fn)(weights)

        # Add weights key (after deleting path)
        config[module_name]["weights"] = weights
    
    return config

def load_modules_with_weights(
        modules: dict[str, nn.Module | pl.LightningModule | object],
        weights: dict[str, dict[str, Any]]) -> \
        dict[str, nn.Module | pl.LightningModule | object]:
    """Loads modules with weights and optionally freezes

    Loads modules in the provided dictionary with weights provided in
    weights dictionary. Each key in modules dictionary should match the
    key in weights dictionary, for the module instance that the
    corresponding weights should be loaded.

    Note:
        If the any entry in weights dictionary contains "freeze" and it
        is specified as true, please ensure that the created model (as
        specified by the previous module config) has the functions
        `eval` and `freeze`. These are already available in
        :class:`~pl.LightningModule`, if your model is not of the same
        type, please create the functions manually.

    Args:
        modules (dict[str, nn.Module | pl.LightningModule | object]):
            The modules dictionary with keys specifying module names and
            values specifying actual instances of those modules.
        weights (dict[str, dict[str, Any]]): The weights dictionary with
            keys specifying module names and values specifying weights
            config dictionary. The latter typically contains 2 items:
            "weights" - loaded torch weights, and "freeze" - whether
            the module with or without loaded weights should be frozen.

    Returns:
        dict[str, nn.Module | pl.LightningModule | object]: A modules
            dictionary which is the same as the one passed in arguments,
            except the modules may optionally be loaded with weights
            and/or frozen.
    """
    for name in modules.keys():
        if name in weights.keys() and weights[name].get("weights") is not None:
            # Load module with corresponding weights (based on name)
            # print(name, weights[name])
            modules[name].load_state_dict(weights[name]["weights"])
        
        if weights.get("freeze", False):
            # Ensure these exist
            modules[name].eval()
            modules[name].freeze()
    
    return modules

def parse_model_config(config: dict[str, Any]) -> dict[str, Any]:
    if (modules_config := config.pop("modules", None)) is not None:
        # Load the modules from config (save var for weights)
        config.update(modules := load_multi(modules_config))
    
    if (weights_config := config.pop("weights", None)) is not None:
        # Load the weights and use for modules
        weights = load_weights(weights_config)
        config.update(load_modules_with_weights(modules, weights))

    if (training_config := config.pop("training", None)) is not None:
        # Load training props from training config
        config.update(load_multi(training_config))

    return config

def parse_config(config_path: str | os.PathLike) -> dict[str, Any]:
    # Create parsing map
    PARSING_MAP = {
        "mask_generator": parse_model_config,
        "mask_retoucher": parse_model_config,
        "mask_inpainter": parse_model_config
    }

    # Load the config from given path
    config = load_json(config_path)

    for key, _config in config.items():        
        if (macro := _config.get("macro", None)) is not None:
            # If contains macro, load it
            _config.update(load_json(macro))
        
        # Parse current config and reassign
        config[key] = PARSING_MAP[key](_config)
    
    return config
