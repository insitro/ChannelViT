from typing import List, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from amlssl import transformations
from amlssl.data.s3dataset import S3Dataset


class BigEarth(S3Dataset):
    """BigEarth dataset"""
    # base_path = "s3://insitro-user/yujia/datasets/BigEarthNet/"
    base_path = "s3://insitro-user/yujia/datasets/BigEarthNetMM/"

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    labels = [
        'Urban fabric',
        'Industrial or commercial units',
        'Arable land',
        'Permanent crops',
        'Pastures',
        'Complex cultivation patterns',
        'Land principally occupied by agriculture, with significant areas of natural vegetation',
        'Agro-forestry areas',
        'Broad-leaved forest',
        'Coniferous forest',
        'Mixed forest',
        'Natural grassland and sparsely vegetated areas',
        'Moors, heathland and sclerophyllous vegetation',
        'Transitional woodland, shrub',
        'Beaches, dunes, sands',
        'Inland wetlands',
        'Coastal wetlands',
        'Inland waters',
        'Marine waters',
    ]

    def __init__(
        self,
        path: str,
        split: str,  # train, valid or test
        is_train: bool,
        transform_cfg: DictConfig,
        channels: Union[List[int], None],
        upsample: int = 1,
        channel_mask: bool = False,
        scale: float = 1,
    ) -> None:
        """Initialize the dataset."""
        super().__init__()

        # read the cyto mask df
        self.df = pd.read_parquet(path)

        if upsample != 1:
            # upsample the dataset to increase num of batches per epoch --> match
            # optimization statistics  with imagenet
            print(f"Upsampling each epoch by {upsample}")
            print(f"Original size {len(self.df)}")
            self.df = pd.concat([self.df for _ in range(int(upsample))], ignore_index=True)
            print(f"After upsample size {len(self.df)}")

        self.channels = torch.tensor([c for c in channels])
        self.scale = scale  # scale the input to compensate for input channel masking

        self.transform = getattr(transformations, transform_cfg.name)(
            is_train,
            **transform_cfg.args,
            normalization_mean=transform_cfg.normalization.mean,
            normalization_std=transform_cfg.normalization.std,
        )

        self.channel_mask = channel_mask

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_chw = self.get_image(f"{self.base_path}{row['path']}").astype('float32')
        if img_chw is None:
            return None

        img_chw = self.transform(img_chw)

        channels = self.channels.numpy()

        if self.scale != 1:
            if type(img_chw) is list:
                # multi crop for DINO training
                img_chw = [img * self.scale for img in img_chw]
            else:
                # single view for linear probing
                img_chw *= self.scale

        # mask out channels
        if type(img_chw) is list:
            if self.channel_mask:
                unselected = [c for c in range(len(img_chw[0])) if c not in channels]
                for i in range(len(img_chw)):
                    img_chw[i][unselected] = 0
            else:
                img_chw = [img[channels] for img in img_chw]
        else:
            if self.channel_mask:
                unselected = [c for c in range(len(img_chw)) if c not in channels]
                img_chw[unselected] = 0
            else:
                img_chw = img_chw[channels]

        labels = np.array([row[l] for l in self.labels])
        labels_original = np.array([row[f'original_{l}'] for l in range(43)])

        return (
            torch.tensor(img_chw).float(),
            {
                "channels": channels,
                "label": labels,
                "label_original": labels_original,
            },
        )

    def __len__(self) -> int:
        return len(self.df)
