# Script for training the DINO embeddings from scratch using ContextViT

import hydra
import pytorch_lightning as pl
import logging
import random
import wandb
import boto3
import wandb
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from itertools import combinations

import amlssl.data as data
from amlssl.meta_arch import SupervisedMulticlass


def get_checkpoint_path(ckpt):
    '''Get the exact checkpoint path based on the prefix and the epoch number.
    Note that the original checkpoint is named after both the epochs and the number of
    updates. This method enables us to automatically fetch the latest ckpt based on the
    epoch number.
    '''
    assert ckpt.startswith("s3://insitro-user/")
    prefix = ckpt[len("s3://insitro-user/"):]

    try:
        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket="insitro-user",
                                      Prefix=prefix,
                                      MaxKeys=100)
        assert (len(response['Contents']) == 1,
            "Received more than one files with the given ckpt prefix")

        ckpt = "s3://insitro-user/" + response['Contents'][0]['Key']
        print(f"Getting ckpt from {ckpt}")

        return ckpt

    except Exception as e:
        print(prefix)
        raise e


@hydra.main(version_base=None, config_path="../config",
            config_name="main_supervised_evalall")
def predict(cfg: DictConfig) -> None:
    ckpt = get_checkpoint_path(cfg.checkpoint)
    print(f"loading model from ckpt {ckpt}")
    model = SupervisedMulticlass.load_from_checkpoint(ckpt)

    channels = [int(c) for c in cfg.channels]

    # only one val data
    assert len(cfg.val_data) == 1
    val_data_cfg = next(iter(cfg.val_data.values()))

    random.seed(0)

    flag = False

    for n_channels in range(len(channels), 0, -1):
        current_combinations = list(combinations(channels, n_channels))
        random.shuffle(current_combinations)
        current_combinations = current_combinations[:min(len(current_combinations), 20)]
        print("Sample combinations")
        print(current_combinations)

        for selected in current_combinations:

            selected = list(selected)
            if selected == [2, 11]:
                flag = True
            elif flag is False:
                continue

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

            # create a new wandb entry
            wandb_logger = pl_loggers.WandbLogger(**cfg.wandb)
            nickname = (cfg.nickname + "-evaluation-correct-ch-"
                        + "-".join([str(c) for c in selected]))

            wandb_logger.log_hyperparams({"nickname": nickname})

            val_data = getattr(data, val_data_cfg.name)(
                is_train=False,
                transform_cfg=cfg.transformations,
                **val_data_cfg.args,
            )
            val_loader = DataLoader(val_data, **val_data_cfg.loader,
                                    collate_fn=val_data.collate_fn)

            trainer = pl.Trainer(
                logger=wandb_logger, strategy="ddp", **cfg.trainer
            )

            trainer.validate(model=model, dataloaders=val_loader)
            wandb.finish()

            wandb.finish()


if __name__ == "__main__":
    boto3.set_stream_logger(name='botocore.credentials', level=logging.ERROR)
    torch.set_float32_matmul_precision("high")
    predict()
