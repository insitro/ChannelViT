# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Note:
# We adapted the DINO code from https://github.com/facebookresearch/dino

import random

from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L36
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L57
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class CamelyonAugmentationDino(object):
    """For training the DINO embeddings on Camelyon17, we mostly follow the
    original DINO implementaiton (linked below) with two changes:
        1. We use 96 x 96 for the global crop size
        2. We use 32 x 32 for the local crop size
    """

    def __init__(
        self, global_crops_scale, local_crops_scale, local_crops_number, is_train
    ):

        self.is_train = is_train

        assert len(global_crops_scale) == 2
        assert len(local_crops_scale) == 2
        global_crops_scale = tuple(global_crops_scale)
        local_crops_scale = tuple(local_crops_scale)

        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    32, scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

        self.normalize = transforms.Compose([transforms.CenterCrop(96), normalize])

    def __call__(self, image):
        """Given a PIL image, we apply the global and local transformations to obtain
        different views of the image.
        """
        if self.is_train:
            crops = []

            crops.append(self.global_transfo1(image))
            crops.append(self.global_transfo2(image))
            for _ in range(self.local_crops_number):
                crops.append(self.local_transfo(image))
            return crops

        else:
            return self.normalize(image)
