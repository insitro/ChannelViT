# Note samplev2 channel vit is only compatible with Supervised learning
import math
from functools import partial
from typing import List

import random
import torch
import torch.distributed as dist
import torch.nn as nn

from amlssl.backbone.vision_transformer import Block
from amlssl.utils import trunc_normal_



class HyperNetwork(nn.Module):
    def __init__(self, embedding_dim, conv_output_channels, conv_kernel_size):
        super(HyperNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.conv_output_channels = conv_output_channels
        self.conv_kernel_size = conv_kernel_size
        self.weight_size = conv_output_channels * 1 * (conv_kernel_size * conv_kernel_size)
        self.bias_size = conv_output_channels

        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, self.weight_size + self.bias_size)

    def forward(self, embedding_vector):
        x = torch.relu(self.fc1(embedding_vector))
        x = torch.tanh(self.fc2(x))
        weights = x[: self.weight_size].view(
            -1, self.conv_output_channels, 1, self.conv_kernel_size, self.conv_kernel_size
        )
        bias = x[self.weight_size:].view(self.conv_output_channels)

        return weights, bias


class HyperConvNet(nn.Module):
    def __init__(self, conv_output_channels, conv_kernel_size):
        super(HyperConvNet, self).__init__()
        self.conv_kernel_size = conv_kernel_size

    def forward(self, x, hyper_weights, hyper_bias=None):
        x = nn.functional.conv2d(
            x, hyper_weights[0], bias=hyper_bias, stride=self.conv_kernel_size
        )
        return x


class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True
    ):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hyper_network = HyperNetwork(embed_dim, embed_dim, patch_size)

        self.projs = nn.ModuleList(
            [
                HyperConvNet(embed_dim, patch_size)
                for _ in range(in_chans)
            ]
        )

        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        self.enable_sample = enable_sample

        trunc_normal_(self.channel_embed.weight, std=0.02)

    def forward(self, x, extra_tokens={}):
        # # assume all images in the same batch has the same input channels
        # cur_channels = extra_tokens["channels"][0]
        # embedding lookup
        cur_channel_embed = self.channel_embed(extra_tokens['channels']) # B, Cin, embed_dim=Cout
        cur_channel_embed = cur_channel_embed.permute(0, 2, 1)  # B Cout Cin

        B, Cin, H, W = x.shape
        # Note: The current number of channels (Cin) can be smaller or equal to in_chans

        channels = list(range(Cin))
        if self.training and self.enable_sample:
            # Per batch channel sampling
            # Note this may be slow
            # Randomly sample the number of channels for this batch
            Cin_new = random.randint(1, Cin)

            # Randomly sample the selected channels
            channels = random.sample(range(Cin), k=Cin_new)
            Cin = Cin_new
            x = x[:, channels, :, :]

        # shared projection layer across channels
        out = []
        for i, c in enumerate(channels):
            # TODO: Find a better way to do this
            # Note that here we change the indices from x[:,c,:,:] to x[:,i,:,:].
            # It should make no difference as long as the input data's channels don't
            # get shuffled.
            channel_conv = self.projs[c]
            hyper_weights, hyper_bias = self.hyper_network(
                self.channel_embed(torch.tensor(c).long().cuda())
            )
            out.append(channel_conv(x[:, i, :, :].unsqueeze(1), hyper_weights, hyper_bias))
            # B Cout H W

        x = torch.stack(out, dim=2)  # B Cout Cin H W

        # preparing the output sequence
        x = x.flatten(2)  # B Cout CinHW
        x = x.transpose(1, 2)  # B CinHW Cout

        return x, Cin


class ChannelVisionTransformer(nn.Module):
    """Channel Vision Transformer"""

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
        enable_sample=True,
        **kwargs,
    ):
        super().__init__()
        print("Warning!!!\n"
              "Samplev2 channel vit randomly sample channels for each batch.\n"
              "It is only compatible with Supervised learning\n"
              "Doesn't work with DINO or Linear Prob")

        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.in_chans = in_chans

        self.patch_embed = PatchEmbedPerChannel(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            enable_sample=enable_sample
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_extra_tokens = 1  # cls token

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, num_patches // self.in_chans + self.num_extra_tokens, embed_dim
            )
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

    def interpolate_pos_encoding(self, x, w, h, c):
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
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, 1, -1, dim)

        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(1, c, -1, dim).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x, extra_tokens):
        B, nc, w, h = x.shape
        x, nc = self.patch_embed(x, extra_tokens)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h, nc)

        return self.pos_drop(x)

    def forward(self, x, extra_tokens={}):
        x = self.prepare_tokens(x, extra_tokens)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x, extra_tokens={}):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, extra_tokens, n=1):
        x = self.prepare_tokens(x, extra_tokens)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def samplev3hyperchannelvit_tiny(patch_size=16, **kwargs):
    model = ChannelVisionTransformer(
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


def samplev3hyperchannelvit_small(patch_size=16, **kwargs):
    model = ChannelVisionTransformer(
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


def samplev3hyperchannelvit_base(patch_size=16, **kwargs):
    model = ChannelVisionTransformer(
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
