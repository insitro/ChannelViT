# Channel Vision Transformer: An Image Is Worth C x 16 x 16 Words

Vision Transformer sets the benchmark for image representation. However, unique challenges arise in certain imaging fields such as microscopy and satellite imaging:

1. Unlike RGB images, images in these domains often contain multiple channels, each carrying semantically distinct and independent information.
2. Not all input channels may be available at test time, necessitating a model that performs robustly under these conditions.

In response to these challenges, we introduce ChannelViT and Hierarchical Channel Sampling.
1. ChannelViT constructs patch tokens independently from each input channel and employs a learnable channel embedding to encode channel-specific information. This modification enables ChannelViT to perform *cross-channel* and *cross-position* reasoning, a critical feature for multi-channel imaging.
2. Hierarchical Channel Sampling (HCS) employs a two-step sampling procedure to simulate test time channel unavailability during training. Unlike channel dropout, where each channel is dropped independently and biases a certain number of selected channels, the two-stage sampling procedure ensures HCS covers channel combinations with varying numbers of channels *uniformly*. This results in a consistent and significant improvement in robustness.

<figure>
  <p align="center">
  <img src="assets/channelvit.jpg" width=90% align="center" alt="my alt text"/>
  </p>
  <figcaption width=80%><em>
  Illustration of ChannelViT. The input for ChannelViT is a cell image from JUMP-CP, which comprises five fluorescence channels (colored differently) and three brightfield channels (colored in B&W). ChannelViT generates patch tokens for each individual channel, utilizing a learnable channel embedding </em><b>chn</b><em> to preserve channel-specific information. The positional embeddings </em><b>pos</b><em> and the linear projection </em><b>W</b><em> are shared across all channels.
  </em></figcaption>
</figure>
<br/>
<br/>

Should you have any questions or require further assistance, please do not hesitate to create an issue. We are here to provide support. 🤗


## Environment setup
This project is developed based on [PyTorch 2.0](https://pytorch.org) and [PyTorch-Lightning
2.0.1](https://www.pytorchlightning.ai/index.html).
We use [conda](https://docs.conda.io/en/latest/) to manage the Python environment. You
can setup the enviroment by running
```bash
git clone git@github.com:insitro/ChannelViT.git
cd ChannelViT
conda env create -f environment.yml
conda activate channelvit 
```
You can then install contextvit through pip.
```bash
pip install git+https://github.com/insitro/ChannelViT.git
```

## An example on JUMP-CP
This section provides an example of our training and evaluation pipelines using JUMP-CP. The preprocessed JUMP-CP data utilized here was released in our previous work, [insitro/ContextViT](https://github.com/insitro/ContextViT).


#### ViT-S/16 w/o HCS
Let's start with the most straightforward scenario: training the ViT-S/16 model without HCS. We employ [hydra](https://hydra.cc/) for managing our experiment configuration. The script provided below will load its corresponding main configuration file, `amlssl/config/main_supervised.yaml`, along with any command line overrides. It trains the ViT-S/16 model to minimize the cross-entropy loss on the JUMP-CP training data over the course of 100 epochs. The process requires a single GPU and operates with a batch size of 32.
```bash
python amlssl/main/main_supervised.py \
trainer.devices=1 \
trainer.max_epochs=100 \
meta_arch/backbone=vit_small \
meta_arch.backbone.args.in_chans=8 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.loader.batch_size=32 \
transformations@train_transformations=cell \
transformations@val_transformations=cell
```
Given that each cell image in JUMP-CP contains 8 channels, we override the input channels to 8. Throughout the training, we save the snapshots in the `./snapshots/` directory. You can alter this path by overriding the value of `trainer.default_root_dir`. 

#### ViT-S/8 w/ HCS
#### ChannelViT-S/16 w/o HCS
#### ChannelViT-S/8 w/ HCS


## Citation

If our work contributes to your research, we would greatly appreciate a citation.

```
@article{bao2023channel,
  title={Channel Vision Transformers: An Image Is Worth C x 16 x 16 Words},
  author={Bao, Yujia and Sivanandan, Srinivasan and Karaletsos, Theofanis},
  journal={arXiv preprint arXiv:2309.16108},
  year={2023}
}
```
