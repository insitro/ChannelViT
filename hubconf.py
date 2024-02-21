dependencies = ["torch"]

from torch import hub
from channelvit.backbone.channel_vit import channelvit_small
from channelvit.backbone.hcs_channel_vit import hcs_channelvit_small

def imagenet_channelvit_small_p16_DINO(pretrained=True, *args, **kwargs):
    """
    Pretrained ChannelViT-Small model (patch size = 16) trained on ImageNet using DINO
    """
    model = channelvit_small(patch_size=16, in_chans=3, *args, **kwargs)
    if pretrained:
        model.load_state_dict(
            hub.load_state_dict_from_url(
                "https://github.com/insitro/ChannelViT/releases/download/v1.0.0/imagenet_channelvit_small_p16_DINO.pth",
                progress=True
            )
        )
    # Set the model to evaluation mode
    model.eval()
    return model

def imagenet_channelvit_small_p16_with_hcs_supervised(pretrained=True, *args, **kwargs):
    """
    Pretrained Supervised ChannelViT-Small model (patch size = 16) trained on ImageNet
    """
    model = hcs_channelvit_small(patch_size=16, in_chans=3, *args, **kwargs)
    if pretrained:
        model.load_state_dict(
            hub.load_state_dict_from_url(
                "https://github.com/insitro/ChannelViT/releases/download/v1.0.0/imagenet_channelvit_small_p16_with_hcs_supervised.pth",
                progress=True
            )
        )
    # Set the model to evaluation mode
    model.eval()
    return model


def cpjump_cellpaint_channelvit_small_p8_with_hcs_supervised(pretrained=True, *args, **kwargs):
    """
    Pretrained Supervised ChannelViT-Small model (patch size = 8) trained on
    CellPainting channels from JUMP-CP (subset)
    """
    model = hcs_channelvit_small(patch_size=8, in_chans=5, *args, **kwargs)
    if pretrained:
        model.load_state_dict(
            hub.load_state_dict_from_url(
                "https://github.com/insitro/ChannelViT/releases/download/v1.0.0/cpjump_cellpaint_channelvit_small_p8_with_hcs_supervised.pth",
                progress=True
            )
        )
    # Set the model to evaluation mode
    model.eval()
    return model

def cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised(pretrained=True, *args, **kwargs):
    """
    Pretrained Supervised ChannelViT-Small model (patch size = 8) trained on
    CellPainting + Brightfield channels from JUMP-CP (subset)
    """
    model = hcs_channelvit_small(patch_size=8, in_chans=8, *args, **kwargs)
    if pretrained:
        model.load_state_dict(
            hub.load_state_dict_from_url(
                "https://github.com/insitro/ChannelViT/releases/download/v1.0.0/cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised.pth",
                progress=True
            )
        )
    # Set the model to evaluation mode
    model.eval()
    return model

def so2sat_channelvit_small_p8_with_hcs_random_split_supervised(pretrained=True, *args, **kwargs):
    """
    Pretrained Supervised ChannelViT-Small model (patch size = 8) trained on
    all channels from So2Sat dataset (random split)
    """
    model = hcs_channelvit_small(patch_size=8, in_chans=18, *args, **kwargs)
    if pretrained:
        model.load_state_dict(
            hub.load_state_dict_from_url(
                # TODO: replace with github release link
                "https://github.com/insitro/ChannelViT/releases/download/v1.0.0/so2sat_channelvit_small_p8_with_hcs_random_split_supervised.pth",
                progress=True
            )
        )
    # Set the model to evaluation mode
    model.eval()
    return model

def so2sat_channelvit_small_p8_with_hcs_hard_split_supervised(pretrained=True, *args, **kwargs):
    """
    Pretrained Supervised ChannelViT-Small model (patch size = 8) trained on
    all channels from So2Sat dataset (hard split)
    """
    model = hcs_channelvit_small(patch_size=8, in_chans=18, *args, **kwargs)
    if pretrained:
        model.load_state_dict(
            hub.load_state_dict_from_url(
                "https://github.com/insitro/ChannelViT/releases/download/v1.0.0/so2sat_channelvit_small_p8_with_hcs_hard_split_supervised.pth",
                progress=True
            )
        )
    # Set the model to evaluation mode
    model.eval()
    return model

def camelyon_channelvit_small_p8_with_hcs_supervised(pretrained=True, *args, **kwargs):
    """
    Pretrained Supervised ChannelViT-Small model (patch size = 8) trained on
    all channels from WILDS Camelyon17 dataset
    """
    model = hcs_channelvit_small(patch_size=8, in_chans=3, *args, **kwargs)
    if pretrained:
        model.load_state_dict(
            hub.load_state_dict_from_url(
                "https://github.com/insitro/ChannelViT/releases/download/v1.0.0/camelyon_channelvit_small_p8_with_hcs_supervised.pth",
                progress=True
            )
        )
    # Set the model to evaluation mode
    model.eval()
    return model