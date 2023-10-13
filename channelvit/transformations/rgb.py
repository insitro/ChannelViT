import torch
from PIL import Image
from torchvision import transforms


class RGBAugmentation:
    """Transformations for standard RBG images"""

    def __init__(
        self,
        is_train: bool = True,
        global_resize: int = 224,
        flip_prob: float = 0.5,
        crop_scale=[0.08, 1.0],
        color_jitter: tuple[float, float, float, float] = (0.4, 0.4, 0.4, 0.1),
        color_jitter_prob: float = 0.8,
        normalize_mean: list[float] = [0.485, 0.456, 0.406],
        normalize_std: list[float] = [0.229, 0.224, 0.225],
        center_crop=None,
        channel_mask=[],
    ):
        crop_scale = tuple(crop_scale)
        if is_train:
            self.transform = transforms.Compose(
                [
                    # transforms.CenterCrop(global_resize),
                    transforms.RandomResizedCrop(
                        (global_resize, global_resize),
                        scale=crop_scale,
                        interpolation=Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=flip_prob),
                    transforms.RandomApply(
                        [transforms.ColorJitter(*color_jitter)], p=color_jitter_prob
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(normalize_mean, normalize_std),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (global_resize, global_resize), interpolation=Image.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(normalize_mean, normalize_std),
                ]
            )

        self.channel_mask = list(channel_mask)
        self.masked_channel_mean = torch.tensor(
            [normalize_mean[c] for c in channel_mask]
        ).view(-1, 1, 1)

    def __call__(self, img) -> torch.Tensor:
        img = self.transform(img)
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
