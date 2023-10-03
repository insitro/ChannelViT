ChannelViT

Vision Transformer delivers state-of-the-art image representation. However, in certain
imaging fields, such as microscopy and statellite imaging, there exist unique
challenges:
1. Unlike RGB images, the images in these domains often contain multiple channels, each
   carrying sematically distinct and independent information.
2. The input channels may not all be available at test time, requiring the model to
   perform robustly.

In this work, we introduce ChannelViT and Hierarchical Channel Sampling (HCS).
1. ChannelViT constructs patch tokens independently from each input channel and utilize a
learnable channel embedding to encode channel-specific infromation. This modification
allows ChannelViT to perform cross-channel and cross-position reasoning, which has shown
to be crucial for multi-channel imaging.
2. HCS uses a two-step sampling procedure to mimic the test time channel inavailabilites
   during training. Unlike channel dropout where each channel is dropped independently,
   which biases certan number of selected channels, the two stage sampling procedure ensures HCS cover channel combinations with different number of channels uniformly. This result in a consistent and signficant improvement in robustness.

If you have any questions or need further assistance, please don't hesitate to create an
issue. We are here to provide support and guidance. 🤗

<figure>
  <p align="center">
  <img src="assets/channelvit.jpg" width=80% align="center" alt="my alt text"/>
  </p>
  <figcaption width=80%><em>
  Illustration of ChannelViT. The input for ChannelViT is a cell image from JUMP-CP, which comprises five fluorescence channels (colored differently) and three brightfield channels (colored in B&W). ChannelViT generates patch tokens for each individual channel, utilizing a learnable channel embedding chn to preserve channel-specific information. The positional embeddings pos and the linear projection $W$ are shared across all channels.
  </em></figcaption>
</figure>
<br/>
<br/>


## Get started
### Environment
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
