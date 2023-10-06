from typing import List, Union

import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image

from channelvit import transformations
from channelvit.data.s3dataset import S3Dataset

TRAIN_DATASET_PATH = (
    "s3://insitro-curated-data/public/imagenet/indexed_df/train/dataframe.pq"
)
VALIDATION_DATASET_PATH = (
    "s3://insitro-curated-data/public/imagenet/indexed_df/validation/dataframe.pq"
)
TEST_DATASET_PATH = (
    "s3://insitro-curated-data/public/imagenet/indexed_df/test/dataframe.pq"
)


class ImageNet(S3Dataset):
    def __init__(
        self,
        split: str,
        is_train: bool,
        transform_cfg: DictConfig,
        sample_channels: int = -1,
        channels: List[int] = [0, 1, 2],  # use all rgb channels
        scale: float = 1,
    ):
        super().__init__()

        if split == "train":
            self.df = pd.read_parquet(TRAIN_DATASET_PATH)
        elif split == "valid":
            self.df = pd.read_parquet(VALIDATION_DATASET_PATH)
        elif split == "test":
            self.df = pd.read_parquet(TEST_DATASET_PATH)
        else:
            raise ValueError(f"Unknown split {split}")

        self.is_train = is_train
        self.sample_channels = sample_channels

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
        if self.sample_channels != -1:
            # sample a subset of the channels
            channel_indicies = torch.randperm(len(channels))[:self.sample_channels]
            channels = channels[channel_indicies]
            if type(img_chw) is list:
                # multi crop for DINO training
                img_chw = [c[channel_indicies,:,:] for c in img_chw]
            else:
                # single view for linear probing
                img_chw = img_chw[channel_indicies,:,:]

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
