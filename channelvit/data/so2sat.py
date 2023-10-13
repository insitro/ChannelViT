from typing import List, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from channelvit import transformations
from channelvit.data.s3dataset import S3Dataset


class So2Sat(S3Dataset):
    """So2Sat"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self,
        path: str,
        split: -1,  # train, valid or test
        is_train: bool,
        transform_cfg: DictConfig,
        channels: Union[List[int], None],
        channel_mask: bool = False,
        scale: float = 1,
    ) -> None:
        """Initialize the dataset."""
        super().__init__()

        # read the cyto mask df
        self.df = pd.read_parquet(path)

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
        img_chw = self.get_image(row["path"]).astype("float32")
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

        label = row.label.astype(int)
        if sum(label) > 1:
            raise ValueError("More than one positive")

        for i, y in enumerate(label):
            if y == 1:
                label = i
                break

        return (
            torch.tensor(img_chw.copy()).float(),
            {"channels": channels, "label": label},
        )

    def __len__(self) -> int:
        return len(self.df)
