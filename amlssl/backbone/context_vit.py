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
import math
from functools import partial
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

from amlssl.backbone.vision_transformer import Block, PatchEmbed
from amlssl.utils import trunc_normal_


class MeanPooling(nn.Module):
    """
    Wrapping torch.mean as a nn Module.
    Apply mean pooling over the given dimension.
    """

    def __init__(self, dim: List[int] = []):
        super().__init__()
        self.dim = tuple(dim)

    def forward(self, input):
        return torch.mean(input, dim=self.dim)


class ContextInferenceNetwork(nn.Module):
    """Inference the context token."""

    def __init__(
        self,
        embed_dim: int,
        context_keys: List[str] = [],
        context_values: List[int] = [],
        amortization: str = "mean_linear",
    ):
        """
            embed_dim: dim of the hidden embeddings of ViT

            context_keys: list of strings, where each string indicate the name of the
            covariate keys

            context_values: list of integers, where each integer indicate the total number
            of different values that the plate can take. Note that this is used for
            non-amortized oracle where each covariate value has a unique learnable token
            embeddings.

            amortization: the amortization method
                mean_linear: apply mean pooling + linear on the patch embeddings
                mean: apply mean pooling on the patch embeddings
                dictionary: no amortization, an unique plate token embedding is learned for
                each plate value
        """
        super().__init__()
        self.model = nn.ModuleDict({})
        self.num_extra_tokens = len(context_keys)

        self.embed_dim = embed_dim

        self.detach = "_detach" in amortization
        if self.detach:
            print("Detaching the embeddings before sending to the inference network")
        amortization = amortization.replace("_detach", "")

        self.amortization = amortization

        if amortization == "dictionary":
            # This is context vit with oracle token,
            # where we learn a dictionary for each covariate value.
            # This method cannot generalize to covariates with unseen values.
            # In the case of Camelyon17, we will be learning three different embedding
            # vectors for the three training hospitals. The model cannot be directly
            # applied to new unseen hospitals as it hasn't learned the corresponding
            # hospital embeddings.
            assert len(context_keys) == len(context_values)
            for k, cnts in zip(context_keys, context_values):
                print(
                    f"Initializing plate token dictionary "
                    f"for {k} with {cnts} different values."
                )
                self.model[k] = nn.Embedding(cnts, embed_dim)
                trunc_normal_(self.model[k].weight, std=0.02)

        else:
            # This is context vit with amortized inference.
            # In this repo, we consider two different models.
            for k in context_keys:
                print(f"Initalizing amortization network for {k}")
                if amortization == "mean_linear":
                    # apply mean pooling followed by a linear layer.
                    self.model[k] = nn.Sequential(
                        MeanPooling(dim=[0, 1]),  # pooling over the batch and seq dim
                        nn.Linear(embed_dim, embed_dim),
                    )

                elif amortization == "mean":
                    # simply apply mean pooling over all patches with the same covariate
                    # value.
                    self.model[k] = MeanPooling(dim=[0, 1])

                else:
                    raise ValueError(f"Unsupported amortization method {amortization}")

    def forward(self, images, patches, covariates, batch_first=True):
        """Add plate tokens to the patch sequence
        Input:
            images: the original image: batch size x nc x width x height
            patches: a patch sequence with dim: batch size x seq len x embed dim

            covariates: a dictionary of covariates for the input patches. The key is a
            string indicating the covariate name. The value is a 1D tensor object with
            dimension "batch_size"
        """
        for k in self.model.keys():
            v = covariates[k]
            # k: covariate name
            # v: id / value for the corresponding covariate
            token = self.compute_amortized_token(
                images, patches, k, v, batch_first
            )  # B, 1, embed_dim
            if batch_first:
                patches = torch.cat((token, patches), dim=1)
            else:
                patches = torch.cat((token, patches), dim=0)

        return patches

    def compute_amortized_token(
        self, images, patches, covariate_key, covariate_value, batch_first=True
    ):
        """Compute plate token for the covariate_key given the patches and the covariate
        value.
        Input:
            images: the original image: batch size x nc x w x h
            patches: a patch sequence with dim: batch size x seq len x embed dim

            covariate_key: a string indicating the covariate name.

            covariate_value: A 1D tensor object with dimension "batch_size". The value
            of the covariate for each example.
        Output:
            token: B x 1 x embed_dim. Plate token.
        """
        if self.amortization == "dictionary":
            token = self.model[covariate_key](covariate_value).unsqueeze(
                1
            )  # B, 1, embed_dim

        else:
            x = patches  # use patch-level information to generate the tokens
            x = x.contiguous()
            v = covariate_value.contiguous()

            try:
                # gather patches and covariate values from all gpus
                x_list = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
                v_list = [torch.zeros_like(v) for _ in range(dist.get_world_size())]
                dist.all_gather(x_list, x)
                dist.all_gather(v_list, v)
                v_all = torch.cat(v_list)
                if batch_first:
                    x_all = torch.cat(x_list)
                else:
                    x_all = torch.cat(x_list, dim=1)
            except Exception:
                x_all = x
                v_all = v

            if self.detach:
                x_all = x_all.detach()
                v_all = v_all.detach()

            # group patches that have the same covariate value
            unique, inverse = torch.unique(v_all, return_inverse=True)

            # placeholder for the output token
            token = torch.zeros(
                (len(covariate_value), self.embed_dim),
                dtype=patches.dtype,
                device=patches.device,
            )

            # iterate over all unique covariate values
            for idx, u in enumerate(unique):
                # covariate_value == u finds all examples that have covariate value u
                # inverse == idx finds all patches with covariate value u
                # average pooling over the sequence dimension and the batch dimension

                if batch_first:
                    token[covariate_value == u] = self.model[covariate_key](
                        x_all[inverse == idx]
                    ).to(patches.dtype)
                else:
                    token[covariate_value == u] = self.model[covariate_key](
                        x_all[:, inverse == idx]
                    ).to(patches.dtype)

            if batch_first:
                token = token.unsqueeze(1)  # batch_size, 1, hidden
            else:
                token = token.unsqueeze(0)  # 1, batch_size, hidden

        return token


class ContextVisionTransformer(nn.Module):
    """Vision Transformer with Context Conditioning"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
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
        context_keys: List[str] = [],
        context_values: List[int] = [],
        amortization: str = "linear",
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = self.out_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # whether to enable layerwise amortization
        self.layerwise_amortization = "_layer" in amortization
        amortization = amortization.replace("_layer", "")

        # initialize the context inference network that is going to be applied at the
        # input patch sequence
        self.cin = ContextInferenceNetwork(
            embed_dim=embed_dim,
            context_keys=context_keys,
            context_values=context_values,
            amortization=amortization,
        )

        self.num_extra_tokens = 1 + self.cin.num_extra_tokens  # cls token

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if self.layerwise_amortization:
            print("layer wise amortization")
            # If layerwise context conditioning is employed, initialize a list of context
            # inference networks for each transformer block
            self.cin_blocks = nn.ModuleList(
                [
                    ContextInferenceNetwork(
                        embed_dim=embed_dim,
                        context_keys=context_keys,
                        context_values=context_values,
                        amortization=amortization,
                    )
                    for _ in range(depth)
                ]
            )

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        # number of auxilary dimensions before the patches
        if not hasattr(self, "num_extra_tokens"):
            # backward compatibility
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.num_extra_tokens

        npatch = x.shape[1] - num_extra_tokens
        N = self.pos_embed.shape[1] - num_extra_tokens

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]

        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x, extra_tokens):
        images = x
        B, nc, w, h = images.shape
        x = self.patch_embed(images)  # patch linear embedding

        # use the context inference network to infer the context token and add it to the
        # input patch sequence
        x = self.cin(images, x, extra_tokens)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, extra_tokens={}):
        x = self.prepare_tokens(x, extra_tokens)
        for layer_id, blk in enumerate(self.blocks):
            x = blk(x)

            if self.layerwise_amortization:
                # use the context inference network to update the context token after each layer
                cls_token = x[:, 0:1]
                patches = self.cin_blocks[layer_id](
                    images=None, patches=x[:, 2:], covariates=extra_tokens
                )
                x = torch.cat((cls_token, patches), dim=1)

        x = self.norm(x)
        return x[:, 0]

        x = self.prepare_tokens(x, extra_tokens)
        return x[:, 1, :]


def contextvit_tiny(patch_size=16, **kwargs):
    model = ContextVisionTransformer(
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


def contextvit_small(patch_size=16, **kwargs):
    model = ContextVisionTransformer(
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


def contextvit_base(patch_size=16, **kwargs):
    model = ContextVisionTransformer(
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
