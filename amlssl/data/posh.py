from typing import List, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from amlssl import transformations
from amlssl.data.s3dataset import S3Dataset


class POSH(S3Dataset):
    """JUMPCP dataset"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self,
        cyto_mask_path_list: ListConfig[str],
        split: str,  # train, valid or test
        is_train: bool,
        transform_cfg: DictConfig,
        channels: Union[List[int], None],
        upsample: int = 1,
        sample_channels: int = -1,
        scale: float = 1,
    ) -> None:
        """Initialize the dataset."""
        super().__init__()

        self.base_path = "s3://insitro-research-2023-cellpaint-posh/Supp_Data_4/"

        # read the cyto mask df
        df = pd.concat(
            [pd.read_parquet(path) for path in cyto_mask_path_list], ignore_index=True
        )
        df = self.get_split(df, split)

        genes = sorted(list(df['gene_id'].unique()))
        self.gene2id = dict(zip(genes, range(len(genes))))

        if upsample != 1:
            # upsample the dataset to increase num of batches per epoch --> match
            # optimization statistics  with imagenet
            print(f"Upsampling each epoch by {upsample}")
            df = pd.concat([df for _ in range(int(upsample))], ignore_index=True)

        self.data_path = list(df["path"])
        self.data_id = list(df["ID"])
        self.genes = list(df["gene_id"])

        if type(channels[0]) is str:
            # channel is separated by hyphen
            self.channels = torch.tensor([int(c) for c in channels[0].split('-')])
        else:
            self.channels = torch.tensor([c for c in channels])

        normalization_mean = [transform_cfg.normalization.mean[int(c)] for c in channels]
        normalization_std = [transform_cfg.normalization.std[int(c)] for c in channels]

        self.transform = getattr(transformations, transform_cfg.name)(
            is_train,
            **transform_cfg.args,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )

        self.scale = scale  # scale the input to compensate for input channel masking


        #self.plate2id, self.field2id, self.well2id, self.well2lbl = load_meta_data()
        self.sample_channels = sample_channels

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

    def __getitem__(self, index):

        img_hwc = self.get_image(self.base_path + self.data_path[index])
        if img_hwc is None:
            return None

        if self.channels is not None:
            # only load the selected channels
            # because we don't have color jitter in our data augmentation
            # NOTE: converting channels to numpy before slicing
            # as slicing numpy array with 1-length torch tensor squeezes the dimension
            img_hwc = img_hwc[:,:,self.channels.numpy()]

        img_chw = self.transform(img_hwc)

        if self.sample_channels != -1:
            # sample a subset of the channels
            channel_indices = torch.randperm(len(self.channels))[:self.sample_channels]
            channels = channels[channel_indices]

            if type(img_chw) is list:
                # multi crop for DINO training
                img_chw = [c[channel_indices,:,:] for c in img_chw]
            else:
                # single view for linear probing
                img_chw = img_chw[channel_indices,:,:]
        else:
            channels = self.channels

        channels = np.array(channels)

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
                "channels": channels,
                "gene_id": self.gene2id[self.genes[index]],
            },
        )

    def __len__(self) -> int:
        return len(self.data_path)
