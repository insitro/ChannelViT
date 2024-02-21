dependencies = ["torch"]

from torch import hub
from channelvit.backbone.channel_vit import channelvit_small

def imagenet_channelvit_small_p16_DINO(pretrained=True, *args, **kwargs):
    """
    """
    # Call the model, load pretrained weights
    model = channelvit_small(patch_size=16, in_chans=3, *args, **kwargs)
    model.load_state_dict(
        hub.load_state_dict_from_url(
            "https://data.aws.insitro.com/account/insitro-root/s3/insitro-user/srinivasan/checkpoints/channelvit_small_imagenet_bs_256_fp32/epoch=99-step=500400.ckpt",
            progress=True
        )
    )
    # Set the model to evaluation mode
    model.eval()

    return model