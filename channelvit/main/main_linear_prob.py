# Script for linear evaluation of pre-trained DINO embeddings

import logging

import boto3
import hydra
import pytorch_lightning as pl
import torch
from amlssl.meta_arch import LinearProb
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy


@hydra.main(version_base=None, config_path="../config", config_name="main_linear_prob")
def linear_prob(cfg: DictConfig) -> None:
    # Load the linear probing model (pl-LightningModule)
    model = LinearProb(cfg)

    # Define the trainer.
    # We will use the configurations under cfg.trainer.
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[LearningRateMonitor(logging_interval="step")],
        **cfg.trainer
    )

    trainer.fit(model=model)


if __name__ == "__main__":
    # Filter useless logging info
    boto3.set_stream_logger(name="botocore.credentials", level=logging.ERROR)

    # set precision to high on A100
    torch.set_float32_matmul_precision("high")

    linear_prob()
