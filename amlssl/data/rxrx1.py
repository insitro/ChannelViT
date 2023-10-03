# Get the RxRx1 dataset from wilds
import os

import pandas as pd
import numpy as np
import torch
from omegaconf import DictConfig

from amlssl import transformations
from amlssl.data.s3dataset import S3Dataset

DATASET_PATH = "s3://insitro-user/yujia/wilds/rxrx1_v1.0/"

TEST_CENTER = 2
VAL_CENTER = 1


class RxRx1(S3Dataset):
    def __init__(self, split: str, is_train: bool, transform_cfg: DictConfig,
                 channels, scale: float = 1):
        """
        Official Wilds split
        https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/wilds/datasets/rxrx1_dataset.py#LL79C1-L79C1
            Training:   33 experiments, 1 site per experiment (site 1)
            Validation: 4 experiments, 2 sites per experiment
            Test OOD:   14 experiments, 2 sites per experiment
            Test ID:    Same 33 experiments from training set
                        1 site per experiment (site 2)

        Input (x):
            256 x 256 cell painting image with 3 channels

        Each image is annotated with:
            cell_type (4 different values):
                'HEPG2', 'HUVEC', 'RPE', 'U2OS'
            experiment (51 different values):
                'HEPG2-01', 'HEPG2-02', 'HEPG2-03', 'HEPG2-04', 'HEPG2-05', 'HEPG2-06',
                'HEPG2-07', 'HEPG2-08', 'HEPG2-09', 'HEPG2-10', 'HEPG2-11', 'HUVEC-01',
                'HUVEC-02', 'HUVEC-03', 'HUVEC-04', 'HUVEC-05', 'HUVEC-06', 'HUVEC-07',
                'HUVEC-08', 'HUVEC-09', 'HUVEC-10', 'HUVEC-11', 'HUVEC-12', 'HUVEC-13',
                'HUVEC-14', 'HUVEC-15', 'HUVEC-16', 'HUVEC-17', 'HUVEC-18', 'HUVEC-19',
                'HUVEC-20', 'HUVEC-21', 'HUVEC-22', 'HUVEC-23', 'HUVEC-24', 'RPE-01',
                'RPE-02', 'RPE-03', 'RPE-04', 'RPE-05', 'RPE-06', 'RPE-07', 'RPE-08',
                'RPE-09', 'RPE-10', 'RPE-11', 'U2OS-01', 'U2OS-02', 'U2OS-03',
                'U2OS-04', 'U2OS-05'
            plate (4 different values each experiment has four plates):
                1, 2, 3, 4
            well (308 different values):
                'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11',
                'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21',
                ...
            site (2 different values):
                1, 2
            sirna_id (1139 different values): This is the prediction target!
        """
        super().__init__()
        self.is_train = is_train
        self.transform = getattr(transformations, transform_cfg.name)(
            is_train=is_train, **transform_cfg.args
        )

        self.base_path = DATASET_PATH
        self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))

        # map covariate values into integer
        for k in ["cell_type", "experiment", "plate", "well", "site"]:
            self.df[f"{k}_id"], _ = pd.factorize(self.df[k], sort=True)

        if split == "train":
            self.df = self.df[self.df.dataset == "train"]
            self.df = self.df[self.df.site != 2]
            self.df = pd.concat([self.df for _ in range(10)], ignore_index=True)
        elif split == "val":
            self.df = self.df[self.df.dataset == "val"]
        elif split == "id_test":
            self.df = self.df[self.df.dataset == "train"]
            self.df = self.df[self.df.site == 2]
        elif split == "test":
            self.df = self.df[self.df.dataset == "test"]
        else:
            raise ValueError(f"Unknown split {split}")

        self.channels = np.array([int(c) for c in channels])
        print(self.channels)
        self.scale = scale

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(
            self.base_path,
            "images",
            row.experiment,
            f"Plate{row.plate}",
            f"{row.well}_s{row.site}.png",
        )

        img_pil = self.get_image(path)
        if img_pil is None:
            return None

        img_chw = self.transform(img_pil)
        if type(img_chw) is list:
            img_chw = [img[self.channels] for img in img_chw]
        else:
            img_chw = img_chw[self.channels]

        if self.scale != 1:
            if type(img_chw) is list:
                # multi crop for DINO training
                img_chw = [c * self.scale for c in img_chw]
            else:
                # single view for linear probing
                img_chw *= self.scale

        return (
            img_chw,
            {
                "experiment_id": row.experiment_id,
                "channels": self.channels,
                "plate_id": row.plate_id,
                "well_id": row.well_id,
                "cell_type_id": row.cell_type_id,
                "site_id": row.site_id,
                "sirna_id": row.sirna_id,
            },
        )

    def __len__(self) -> int:
        return len(self.df)
