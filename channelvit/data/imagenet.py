import os
from typing import List, Union

import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image

from channelvit import transformations
from channelvit.data.s3dataset import S3Dataset


class ImageNet(S3Dataset):
    def __init__(
        self,
        path: str,
        is_train: bool,
        transform_cfg: DictConfig,
        channels: List[int] = [0, 1, 2],  # use all rgb channels
        scale: float = 1,
    ):
        super().__init__()

        self.df = pd.read_parquet(path)
        self.is_train = is_train

        self.transform = getattr(transformations, transform_cfg.name)(
            is_train=is_train, **transform_cfg.args
        )
        self.channels = torch.tensor([c for c in channels])
        self.scale = scale  # scale the input to compensate for input channel masking

        self.s3_client = None

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        img_hwc = Image.fromarray(self.get_image(row.path))

        # data augmentation is applied on all channels (since we have color jitter in
        # the data augmentation)
        img_chw = self.transform(img_hwc)

        # only use the selected channels as the input
        if type(img_chw) is list:
            img_chw = [img[self.channels, :, :] for img in img_chw]
        else:
            img_chw = img_chw[self.channels, :, :]

        # sample channels
        channels = self.channels
        if self.scale != 1:
            if type(img_chw) is list:
                # multi crop for DINO training
                img_chw = [c * self.scale for c in img_chw]
            else:
                # single view for linear probing
                img_chw *= self.scale

        return img_chw, {"ID": row.label, "channels": channels}

    def __len__(self) -> int:
        return len(self.df)
