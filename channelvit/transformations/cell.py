from typing import Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.aug = A.GaussianBlur(sigma_limit=(self.radius_min, self.radius_max))

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return self.aug(images=[img])[0]


def RandomPadCrop(size):
    """
    Crops image to range of `scale` inputs and resize to `size`
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_width=256,
                min_height=256,
                position="random",
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.RandomCrop(width=size, height=size),
        ]
    )


def RandomPadAndCropCenter(size):
    """
    Crops image to range of `scale` inputs and resize to `size`
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_width=320,
                min_height=320,
                position="random",
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.CenterCrop(width=size, height=size),
            # A.ChannelDropout(p=0.2, channel_drop_range=(1, 3)),
        ]
    )


class CellAugmentation(object):
    def __init__(
        self,
        is_train: bool,
        global_resize: int = 224,
        normalization_mean: list[float] = [0.4914, 0.4822, 0.4465],
        normalization_std: list[float] = [0.2023, 0.1994, 0.2010],
        brightness: bool = False,
        use_coarse_dropout: bool = True,
        channel_mask=[],
    ):
        """
        MulticropAugmentation strategy, as developed by M. Caron
        https://arxiv.org/pdf/2006.09882.pdf.
        ASSUMES images are from the distribution N(0,I).
        global_crops_scale: List[float]
            List of (a, b) that defines the scale, sampled uniformly, at which
            to crop the image for the global crop. For instance, (.8, 1.0) will mean that each
            global crop will shrink the original image to be x ~ Uniform([.8, 1.])
            % of the original size.
        local_crops_scale: List[float]
            List of (a, b) that defines the scale, sampled uniformly, at which
            to crop the image for the local crop. For instance, (.6, .8) will mean that each
            local crop will shrink the original image to be x ~ Uniform([.6, .8])
            % of the original size.
        n_local_crops_per_image : int
            number of of local crops per image in the original pair.
            n_local_crops_per_image==0 implies just a single pair of
            reference images (global crops only), whereas n_local_crops_per_image>0
            (as in DINO) implies applying a local crop to each image n_local_crops_per_image
            times.
        global_resize: int
            After cropping image to be of global_crops_scale size of the original size,
            will resize to this value. 224 by default.
        local_resize: int
            After cropping image to be of local_crops_scale size of the original size,
            will resize to this value. 96 by default.
        """
        flip_rotate = A.OneOf(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(90),
                A.Rotate(180),
                A.Rotate(270),
            ]
        )

        if brightness:
            print("Apply brightness change after flip and rotate")
            flip_rotate = A.Compose([flip_rotate, A.RandomBrightness()])

        mean_div_255 = [m / 255.0 for m in normalization_mean]
        std_div_255 = [s / 255.0 for s in normalization_std]
        normalize = A.Compose([A.Normalize(mean_div_255, std_div_255), ToTensorV2()])

        self.is_train = is_train
        self.normalize = normalize

        # global crop
        if use_coarse_dropout:
            coarse_dropout = A.CoarseDropout(max_holes=10, max_height=10, max_width=10)
        else:
            coarse_dropout = A.NoOp()

        self.global_transform1 = A.Compose(
            [
                RandomPadCrop(global_resize),
                flip_rotate,
                A.Defocus(radius=(1, 3)),
                coarse_dropout,
                normalize,
            ]
        )

        self.channel_mask = list(channel_mask)

    def __call__(self, image) -> Union[list[torch.Tensor], torch.Tensor]:
        """
        Take a PIL image, generate its data augmented version
        """
        img = np.asarray(image)
        if self.is_train:
            img = self.global_transform1(image=img)["image"]
        else:
            img = self.normalize(image=img)["image"]

        if len(self.channel_mask) == 0:
            # do not mask channels
            return img
        else:
            # mask out the channels
            # NOTE: this channel mask index is relative / not absolute.
            # For instance, in JUMPCP where we have 8 channels.
            # If the data loader only sends over 3-channel images with channel 5, 6, 7.
            # The channel mask should be [0] if we want to mask out 5.
            img[self.channel_mask, :, :] = torch.zeros_like(
                img[self.channel_mask, :, :]
            )

            return img
