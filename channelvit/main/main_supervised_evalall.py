import logging
from itertools import combinations

import boto3
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

import channelvit.data as data
from channelvit.meta_arch import Supervised


@hydra.main(
    version_base=None, config_path="../config", config_name="main_supervised_evalall"
)
def predict(cfg: DictConfig) -> None:
    print(f"loading model from ckpt {cfg.checkpoint}")
    model = Supervised.load_from_checkpoint(cfg.checkpoint)

    channels = [int(c) for c in cfg.channels]

    # only one val data
    assert len(cfg.val_data) == 1
    val_data_cfg = next(iter(cfg.val_data.values()))

    for n_channels in range(len(channels), 0, -1):
        for selected in combinations(channels, n_channels):
            selected = list(selected)
            if cfg.transformation_mask:
                print(f"selecting {selected} through transformations")
                with open_dict(cfg):
                    cfg.transformations.args.channel_mask = [
                        c for c in channels if c not in selected
                    ]
                val_data_cfg.args.scale = float(len(channels)) / len(selected)
                print("Scale: ", val_data_cfg.args.scale)
            else:
                print(f"selecting {selected} through inputs")
                val_data_cfg.args.channels = selected
                with open_dict(cfg):
                    cfg.val_data = val_data_cfg

            val_data = getattr(data, val_data_cfg.name)(
                is_train=False, transform_cfg=cfg.transformations, **val_data_cfg.args
            )
            val_loader = DataLoader(
                val_data, **val_data_cfg.loader, collate_fn=val_data.collate_fn
            )

            trainer = pl.Trainer(strategy="ddp", **cfg.trainer)
            trainer.validate(model=model, dataloaders=val_loader)


if __name__ == "__main__":
    boto3.set_stream_logger(name="botocore.credentials", level=logging.ERROR)
    torch.set_float32_matmul_precision("high")
    predict()
