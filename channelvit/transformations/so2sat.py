import random
from typing import Union

import numpy as np
import torch


class So2SatAugmentation(object):
    def __init__(
        self,
        is_train: bool,
        normalization_mean: list[float] = [0.4914, 0.4822, 0.4465],
        normalization_std: list[float] = [0.2023, 0.1994, 0.2010],
        channel_mask=[],
    ):
        self.mean = np.array([m for m in normalization_mean])[:, np.newaxis, np.newaxis]
        self.std = np.array([m for m in normalization_std])[:, np.newaxis, np.newaxis]

        self.is_train = is_train
        self.channel_mask = list(channel_mask)

    def __call__(self, image) -> Union[list[torch.Tensor], torch.Tensor]:
        """
        Take a PIL image, generate its data augmented version
        """
        img = (image - self.mean) / self.std

        if self.is_train:
            # rotation
            r = random.randint(0, 3)
            img = np.rot90(img, r, (1, 2))

            # flip
            f = random.randint(0, 1)
            if f == 1:
                img = np.flip(img, 1)

            # flip
            f = random.randint(0, 1)
            if f == 1:
                img = np.flip(img, 2)

        if len(self.channel_mask) == 0:
            # do not mask channels
            return img
        else:
            # mask out the channels
            # NOTE: this channel mask index is relative / not absolute.
            # For instance, in JUMPCP where we have 8 channels.
            # If the data loader only sends over 3-channel images with channel 5, 6, 7.
            # The channel mask should be [0] if we want to mask out 5.
            img[self.channel_mask, :, :] = 0

            return img
