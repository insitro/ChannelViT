from typing import Union

import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BigEarthMMAugmentation(object):
    def __init__(
        self,
        is_train: bool,
        normalization_mean: list[float] = [0.4914, 0.4822, 0.4465],
        normalization_std: list[float] = [0.2023, 0.1994, 0.2010],
        channel_mask=[],
    ):
        self.mean = np.array([m for m in normalization_mean])[:,np.newaxis, np.newaxis]
        self.std = np.array([m for m in normalization_std])[:,np.newaxis, np.newaxis]

        self.is_train = is_train
        self.channel_mask = list(channel_mask)

        flip_rotate = A.OneOf(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(90),
                A.Rotate(180),
                A.Rotate(270),
            ]
        )

        self.transform = A.Compose([
            flip_rotate,
            A.Defocus(radius=(1, 3)),
            A.RandomBrightnessContrast(p=0.8),
        ])

        self.to_tensor = ToTensorV2()


    def __call__(self, image) -> Union[list[torch.Tensor], torch.Tensor]:
        """
        Take a PIL image, generate its data augmented version
        """
        img = (image - self.mean) / self.std
        img = img.clip(-5, 5)
        img = (img  * 25.6 + 128).astype(np.uint8)
        img_hwc = img.transpose(1, 2, 0)

        if self.is_train:
            img_hwc = self.transform(image=img_hwc)['image']

        # convert back to chw
        img = self.to_tensor(image=(img_hwc - 128.0) / 25.6)['image']

        if len(self.channel_mask) == 0:
            # do not mask channels
            return img
        else:
            # mask out the channels
            # NOTE: this channel mask index is relative / not absolute.
            # For instance, in JUMPCP where we have 8 channels.
            # If the data loader only sends over 3-channel images with channel 5, 6, 7.
            # The channel mask should be [0] if we want to mask out 5.
            # TODO: YB: Srini what's your suggestion for this?
            img[self.channel_mask,:,:] = 0

            return img


