# Script for linear evaluation of pre-trained DINO embeddings

import hydra
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import logging
import boto3
import torch
# from lightning_fabric.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy

from amlssl.meta_arch import SupervisedProb



@hydra.main(version_base=None, config_path="../config", config_name="main_supervised_prob")
def supervised_prob(cfg: DictConfig) -> None:
    wandb_logger = pl_loggers.WandbLogger(**cfg.wandb)
    wandb_logger.log_hyperparams({
        "nickname": cfg.nickname,
        "task_name": cfg.task_name
    })

    # Load the linear probing model (pl-LightningModule)
    model = SupervisedProb(cfg)

    # Define the trainer.
    # We will use the configurations under cfg.trainer.
    trainer = pl.Trainer(
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[LearningRateMonitor(logging_interval="step")],
        **cfg.trainer
    )

    trainer.fit(model=model)


if __name__ == "__main__":
    # Filter useless logging info
    boto3.set_stream_logger(name='botocore.credentials', level=logging.ERROR)

    # set precision to high on A100
    torch.set_float32_matmul_precision("high")

    supervised_prob()
