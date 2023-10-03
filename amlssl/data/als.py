from typing import List, Union

import torch
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig

from amlssl import transformations
from amlssl.data.s3dataset import S3Dataset


donor_to_id = {
    "Dins390": 0,
    "Dins532": 1,
    "Dins604": 2,
    "Dins605": 3,
    "Dins022": 4,
    "Dins023": 5,
    "Dins032": 6,
    "Dins033": 7,
    "Dins025": 8,
}

class ALS(S3Dataset):
    """ALS data"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None



    def __init__(
        self,
        path,
        is_train: bool,
        transform_cfg: DictConfig,
        channels: Union[List[int]],
        split: None,
    ) -> None:
        """Initialize the dataset."""
        super().__init__()

        if type(path) == ListConfig:
            self.df = pd.concat([pd.read_parquet(p) for p in path], ignore_index=True)
        else:
            self.df = pd.read_parquet(path)

        cell_line = list(self.df['cell_line'].unique())
        self.cell_line_to_id = dict(zip(cell_line, range(len(cell_line))))

        if split is not None:
            # random split the df into train and val
            np.random.seed(0)
            mask = np.random.rand(len(self.df)) < 0.8
            if split == 'train':
                self.df = self.df[mask]
            elif split == 'val':
                self.df = self.df[~mask]
            elif split == 'all':
                self.df = self.df
            else:
                raise ValueError("Unknown split")

        self.channels = torch.tensor([c for c in channels])

        normalization_mean = [transform_cfg.normalization.mean[c] for c in channels]
        normalization_std = [transform_cfg.normalization.std[c] for c in channels]

        self.transform = getattr(transformations, transform_cfg.name)(
            is_train,
            **transform_cfg.args,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )

        self.is_train = is_train

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_chw = self.get_image(row["masked_tile_path"])
        if img_chw is None:
            return None

        if self.channels is not None:
            # only load the selected channels
            img_chw = img_chw[self.channels.numpy()]

        # transpose the tile
        channels = np.array(self.channels)
        img_hwc = img_chw.transpose(1, 2, 0)
        img_chw = self.transform(img_hwc)

        donor_id = -1 if 'donor_registry_id_x' not in row else donor_to_id[row["donor_registry_id_x"]]

        return (
            img_chw,
            {
                "channels": channels,
                "cell_line": self.cell_line_to_id[row["cell_line"]],
                "disease_state": int(row["is_mutant"]) if "is_mutant" in row else -1,
                "donor_id": donor_id
            },
        )

    def __len__(self) -> int:
        return len(self.df)
