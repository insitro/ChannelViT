from typing import List, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from channelvit import transformations
from channelvit.data.s3dataset import S3Dataset


def load_meta_data():
    PLATE_TO_ID = {"BR00116991": 0, "BR00116993": 1, "BR00117000": 2}
    FIELD_TO_ID = dict(zip([str(i) for i in range(1, 10)], range(9)))
    WELL_TO_ID = {}
    for i in range(16):
        for j in range(1, 25):
            well_loc = f"{chr(ord('A') + i)}{j:02d}"
            WELL_TO_ID[well_loc] = len(WELL_TO_ID)

    WELL_TO_LBL = {}
    # map the well location to the perturbation label
    # Note that the platemaps are different for different perturbations
    PLATE_MAP = {
        "compound": "s3://insitro-user/yujia/cellpainting-gallery/platemap_and_metadata/JUMP-Target-1_compound_platemap.tsv",
        "crispr": "s3://insitro-user/yujia/cellpainting-gallery/platemap_and_metadata/JUMP-Target-1_crispr_platemap.tsv",
        # "orf": "s3://insitro-user/yujia/cellpainting-gallery/platemap_and_metadata/JUMP-Target-1_orf_platemap.tsv",
    }
    META_DATA = {
        "compound": "s3://insitro-user/yujia/cellpainting-gallery/platemap_and_metadata/JUMP-Target-1_compound_metadata.tsv",
        "crispr": "s3://insitro-user/yujia/cellpainting-gallery/platemap_and_metadata/JUMP-Target-1_crispr_metadata.tsv",
        # "orf": "s3://insitro-user/yujia/cellpainting-gallery/platemap_and_metadata/JUMP-Target-1_orf_metadata.tsv",
    }
    for perturbation in PLATE_MAP.keys():
        df_platemap = pd.read_parquet(PLATE_MAP[perturbation])
        df_metadata = pd.read_parquet(META_DATA[perturbation])
        df = df_metadata.merge(df_platemap, how="inner", on="broad_sample")

        if perturbation == "compound":
            target_name = "target"
        else:
            target_name = "gene"

        codes, uniques = pd.factorize(df[target_name])
        codes += 1  # set none (neg control) to id 0
        assert min(codes) == 0
        print(f"{target_name} has {len(uniques)} unique values")
        WELL_TO_LBL[perturbation] = dict(zip(df["well_position"], codes))

    return PLATE_TO_ID, FIELD_TO_ID, WELL_TO_ID, WELL_TO_LBL


class JUMPCP(S3Dataset):
    """JUMPCP dataset"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self,
        cyto_mask_path_list: ListConfig[str],
        split: str,  # train, valid or test
        is_train: bool,
        transform_cfg: DictConfig,
        perturbation_list: ListConfig[str],
        channels: Union[List[int], None],
        upsample: int = 1,
        channel_mask: bool = False,
        scale: float = 1,
    ) -> None:
        """Initialize the dataset."""
        super().__init__()

        # read the cyto mask df
        df = pd.concat(
            [pd.read_parquet(path) for path in cyto_mask_path_list], ignore_index=True
        )
        df = self.get_split(df, split)

        if upsample != 1:
            # upsample the dataset to increase num of batches per epoch --> match
            # optimization statistics  with imagenet
            print(f"Upsampling each epoch by {upsample}")
            print(f"Original size {len(df)}")
            df = pd.concat([df for _ in range(int(upsample))], ignore_index=True)
            print(f"After upsample size {len(df)}")

        self.data_path = list(df["path"])
        self.data_id = list(df["ID"])
        self.well_loc = list(df["well_loc"])

        assert len(perturbation_list) == 1
        self.perturbation_type = perturbation_list[0]

        if type(channels[0]) is str:
            # channel is separated by hyphen
            self.channels = torch.tensor([int(c) for c in channels[0].split('-')])
        else:
            self.channels = torch.tensor([c for c in channels])

        self.scale = scale  # scale the input to compensate for input channel masking

        self.transform = getattr(transformations, transform_cfg.name)(
            is_train,
            **transform_cfg.args,
            normalization_mean=transform_cfg.normalization.mean,
            normalization_std=transform_cfg.normalization.std,
        )

        self.plate2id, self.field2id, self.well2id, self.well2lbl = load_meta_data()

        self.channel_mask = channel_mask


    def get_split(self, df, split_name, seed=0):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(0.6 * m)
        validate_end = int(0.2 * m) + train_end

        if split_name == "train":
            return df.iloc[perm[:train_end]]
        elif split_name == "valid":
            return df.iloc[perm[train_end:validate_end]]
        elif split_name == "test":
            return df.iloc[perm[validate_end:]]
        elif "fluro_bright" in split_name:
            ratio = float(split_name[len("fluro_bright")+1:])
            return df.iloc[perm[int(train_end*ratio):train_end]]
        elif "fluro" in split_name:
            ratio = float(split_name[len("fluro")+1:])
            return df.iloc[perm[:int(train_end*ratio)]]
        else:
            raise ValueError("Unknown split")

    def __getitem__(self, index):
        if self.well_loc[index] not in self.well2lbl[self.perturbation_type]:
            # this well is not labeled
            return None

        img_chw = self.get_image(self.data_path[index])
        if img_chw is None:
            return None

        img_hwc = img_chw.transpose(1, 2, 0)
        img_chw = self.transform(img_hwc)

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

        return (
            img_chw,
            {
                # "ID": self.data_id[index],
                # "well_loc": self.well_loc[index],
                "channels": channels,
                "label": self.well2lbl[self.perturbation_type][self.well_loc[index]],
                "well_id": self.well2id[self.well_loc[index]],
                "plate_id": self.plate2id[self.data_id[index].split("_")[0]],
                "field_id": self.field2id[self.data_id[index].split("_")[2]],
            },
        )

    def __len__(self) -> int:
        return len(self.data_path)
