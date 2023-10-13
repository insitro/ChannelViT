# Script for training the DINO embeddings from scratch using ContextViT

import logging

import boto3
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

import channelvit.data as data
from channelvit.meta_arch import DINO


def get_train_loader(cfg: DictConfig):
    # Define the training data.
    # cfg.data is a dictionary with key=data_name, value=data_cfg.
    # In the DINO training, we support training on one data.
    assert len(cfg.data) == 1
    train_data_cfg = next(iter(cfg.data.values()))
    with open_dict(cfg):
        cfg.data = train_data_cfg

    train_data = getattr(data, train_data_cfg.name)(
        is_train=True, transform_cfg=cfg.transformations, **train_data_cfg.args
    )
    train_loader = DataLoader(
        train_data, **train_data_cfg.loader, collate_fn=train_data.collate_fn
    )

    # We also need to pre-compute the number of batches for each epoch.
    # We will use this inforamtion for the learning rate schedule.
    with open_dict(cfg):
        # get number of batches per epoch (many optimizers use this information to schedule
        # the learning rate)
        cfg.data.loader.num_batches = len(train_loader) // cfg.trainer.devices + 1

    return train_loader


@hydra.main(version_base=None, config_path="../config", config_name="main_dino")
def train(cfg: DictConfig) -> None:
    # get the train data loader
    train_loader = get_train_loader(cfg)

    # Load the DINO model (pl-LightningModule)
    model = DINO(cfg)

    # set precision to high on A100
    torch.set_float32_matmul_precision("high")

    # Define the trainer.
    # We will use the configurations under cfg.trainer.
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(dirpath=cfg.trainer.default_root_dir, save_top_k=-1),
    ]
    trainer = pl.Trainer(strategy="ddp", callbacks=callbacks, **cfg.trainer)

    # Fit the model. if cfg.checkpoint is specified, we will start from the saved
    # checkpoint.
    trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=cfg.checkpoint)


if __name__ == "__main__":
    boto3.set_stream_logger(name="botocore.credentials", level=logging.ERROR)
    torch.set_float32_matmul_precision("high")
    train()
