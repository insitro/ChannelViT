import logging

import boto3
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import channelvit.data as data
from channelvit.meta_arch import CorrelationComputer


@hydra.main(version_base=None, config_path="../config", config_name="main_correlation")
def compute_correlation(cfg: DictConfig) -> None:
    model = CorrelationComputer()

    # load val data
    assert len(cfg.val_data_dict) == 1
    val_data_cfg = next(iter(cfg.val_data_dict.values()))
    val_data = getattr(data, val_data_cfg.name)(
        is_train=False, transform_cfg=cfg.val_transformations, **val_data_cfg.args
    )
    val_loader = DataLoader(
        val_data, **val_data_cfg.loader, collate_fn=val_data.collate_fn
    )

    trainer = pl.Trainer(strategy="ddp", **cfg.trainer)

    trainer.validate(model=model, dataloaders=val_loader)


if __name__ == "__main__":
    boto3.set_stream_logger(name="botocore.credentials", level=logging.ERROR)
    torch.set_float32_matmul_precision("high")
    compute_correlation()
