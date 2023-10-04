# Script for training the DINO embeddings from scratch using ContextViT

import hydra
import pytorch_lightning as pl
import logging
import boto3
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader
from typing import Union, Dict, Dict, Iterator
import  torch.nn.functional as F

import amlssl.data as data
from amlssl.meta_arch import Supervised


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


@hydra.main(version_base=None, config_path="../config", config_name="main_supervised")
def visualize(cfg: DictConfig) -> None:
    ckpt = get_checkpoint_path(cfg.checkpoint)
    print(f"loading model from ckpt {ckpt}")
    model = Supervised.load_from_checkpoint(ckpt)
    channel_embed = model.backbone.patch_embed.channel_embed.weight
    print(channel_embed.size())
    print(torch.corrcoef(channel_embed))

    print(F.pairwise_distance(channel_embed.unsqueeze(0), channel_embed.unsqueeze(1)))



if __name__ == "__main__":
    boto3.set_stream_logger(name='botocore.credentials', level=logging.ERROR)
    torch.set_float32_matmul_precision("high")
    visualize()
