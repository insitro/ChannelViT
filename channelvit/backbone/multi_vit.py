# Copyright (c) Insitro, Inc. and its affiliates.
#
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
from functools import partial

import torch
import torch.nn as nn

from channelvit.backbone.vit import VisionTransformer


class MultiChannelVisionTransformer(nn.Module):
    """Multi-Channel Vision Transformer for Epileptic Seizure Prediction
    """

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        aggregation="mean",
        input_drop=0.0,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # learn a separte vit for each channel
        self.vits = nn.ModuleList(
            [
                VisionTransformer(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=1,
                    num_classes=num_classes,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,
                    input_drop=0.0,
                    **kwargs,
                )
                for _ in range(in_chans)
            ]
        )

        # aggregation mlp
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

        self.in_chans = in_chans
        self.aggregation = aggregation

    def forward(self, x, extra_tokens={}):
        # assume all images in the same batch has the same input channels
        cur_channels = extra_tokens["channels"][0]

        # x shape: B, Cin, H, W

        # Here we look at each individual vit
        # Compute their output classifier token
        # And aggregate them together for mlp
        out = []
        for i, c in enumerate(cur_channels):
            out.append(self.vits[c](x[:, i : i + 1, :, :]))

        if self.aggregation == "mean":
            out = torch.stack(out, dim=0).mean(dim=0)  # batch x embed_dim
        elif self.aggregation == "max":
            out = torch.stack(out, dim=0).max(dim=0)[0]  # batch x embed_dim
        else:
            raise ValueError("Unknown aggregation name")

        out = self.mlp(out)

        return out


def multivit_tiny(patch_size=16, **kwargs):
    model = MultiChannelVisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def multivit_small(patch_size=16, **kwargs):
    model = MultiChannelVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def multivit_base(patch_size=16, **kwargs):
    model = MultiChannelVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
