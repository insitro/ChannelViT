# Get the Camelyon17 dataset from wilds
import os

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from channelvit import transformations
from channelvit.data.s3dataset import S3Dataset

TEST_CENTER = 2
VAL_CENTER = 1


class Camelyon17(S3Dataset):
    def __init__(
        self,
        base_path: str,
        split: str,
        is_train: bool,
        transform_cfg: DictConfig,
        channels=[],
        scale=1,
    ):
        """
        Labeled splits: train, id_val (in distribution), val (OOD), test
        Unlabeled splits: train_unlabeled, val_unlabeled, test_unlabeled

        Input (x):
            96x96 image patches extracted from histopathology slides.

        Label (y):
            y is binary. It is 1 if the central 32x32 region contains any tumor tissue, and 0 otherwise.

        Metadata:
            Each patch is annotated with the ID of the hospital it came from (integer from 0 to 4)
            and the slide it came from (integer from 0 to 49).
        """
        super().__init__()

        self.base_path = base_path

        self.is_train = is_train
        self.transform = getattr(transformations, transform_cfg.name)(
            is_train=is_train, **transform_cfg.args
        )

        if split == "train":
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["split"] == 0]
            self.df = self.df[self.df["center"] != TEST_CENTER]
            self.df = self.df[self.df["center"] != VAL_CENTER]
        elif split == "id_val":
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["split"] == 1]
            self.df = self.df[self.df["center"] != TEST_CENTER]
            self.df = self.df[self.df["center"] != VAL_CENTER]
        elif split == "test":
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["center"] == TEST_CENTER]
        elif split == "val":
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["center"] == VAL_CENTER]
        else:
            raise ValueError(f"Unknown split {split}")

        self.channels = np.array([int(c) for c in channels])
        self.scale = scale

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(
            self.base_path,
            (
                f"patches/patient_{row.patient:03}_node_{row.node}/"
                f"patch_patient_{row.patient:03}_node_{row.node}_x_{row.x_coord}_y_{row.y_coord}.png"
            ),
        )

        try:
            img_pil = self.get_image(path)

        except Exception as e:
            print(path)
            raise e

        if img_pil is None:
            return None

        img_chw = self.transform(img_pil)
        if self.channels is not None:
            if type(img_chw) is list:
                img_chw = [img[self.channels, :, :] for img in img_chw]
            else:
                img_chw = img_chw[self.channels]

        channels = self.channels

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
                "tumor": row.tumor,  # 0 or 1 for labeled,  -1 for unlabeled
                "hospital_id": row.center,
                "slide_id": row.slide,
                "node_id": row.node,
                "patient_id": row.patient,
                "channels": channels,
            },
        )

    def __len__(self) -> int:
        return len(self.df)
