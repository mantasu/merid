import argparse
import torch
import pytorch_lightning as pl

from datasets.data_module import GeneralModule

from architectures.mask_generator import MaskGenerator

from utils.io_and_types import load_json
from utils.config import parse_model_config, fix_weights, parse_data_config

torch.set_float32_matmul_precision("medium")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/main.json",
        help="path to the JSON config file"
    )

    return parser.parse_args()

def prepare_model(model_name, model_config):
    model_config = parse_model_config(model_config, fix_weights)

    if model_name == "mask_generator":
        return MaskGenerator(model_config)
    elif model_name == "mask_inpainter":
        return None

def prepare_datamodule(data_config):
    data_config = parse_data_config(data_config)
    datamodule = GeneralModule(**data_config)
    return datamodule

if __name__ == "__main__":
    config = load_json(parse_arguments().config)

    model = prepare_model("mask_generator", config["mask_generator"])
    datamodule = prepare_datamodule(config["data"])

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=100)
    trainer.fit(model, datamodule=datamodule)

