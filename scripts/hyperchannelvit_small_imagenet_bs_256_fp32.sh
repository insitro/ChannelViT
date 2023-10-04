NICKNAME="hyperchannelvit_small_imagenet_bs_256_fp32"

python amlssl/main/main_dino.py \
wandb.project="amlssl-channelvit" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/srinivasan/checkpoints/${NICKNAME}" \
meta_arch/backbone=hyperchannelvit_small \
meta_arch.backbone.args.in_chans=3 \
data=imagenet \
data.imagenet.args.channels=[0,1,2] \
data.imagenet.loader.num_workers=32 \
data.imagenet.loader.batch_size=32 \
data.imagenet.loader.drop_last=True \
transformations=imagenet_dino


# This version uses the last 4 layer's CLS token for prediction
# https://wandb.aws.insitro.com/ml_team/amlssl-linear-prob/runs/lxi2qgn3?workspace=user-yujia
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-4-last-blocks" \
data@train_data=imagenet \
data@val_data_dict=[imagenet_val] \
train_data.imagenet.loader.num_workers=64 \
train_data.imagenet.loader.batch_size=32 \
transformations=rgb \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch.target="ID" \
meta_arch.num_classes=1000 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/srinivasan/checkpoints/${NICKNAME}/epoch\\=99"  # A unique ckpt prefix is enough. Doesn't need to be the full path


python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-r-4-last-blocks" \
data@train_data=imagenet \
data@val_data_dict=[imagenet_val] \
train_data.imagenet.loader.num_workers=64 \
train_data.imagenet.loader.batch_size=32 \
train_data.imagenet.args.channels=[0] \
val_data_dict.imagenet_val.args.channels=[0] \
transformations=rgb \
transformations.args.color_jitter_prob=0 \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch.target="ID" \
meta_arch.num_classes=1000 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/srinivasan/checkpoints/${NICKNAME}/epoch\\=99"


python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-g-4-last-blocks" \
data@train_data=imagenet \
data@val_data_dict=[imagenet_val] \
train_data.imagenet.loader.num_workers=64 \
train_data.imagenet.loader.batch_size=32 \
train_data.imagenet.args.channels=[1] \
val_data_dict.imagenet_val.args.channels=[1] \
transformations=rgb \
transformations.args.color_jitter_prob=0 \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch.target="ID" \
meta_arch.num_classes=1000 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/srinivasan/checkpoints/${NICKNAME}/epoch\\=99"


python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-b-4-last-blocks" \
data@train_data=imagenet \
data@val_data_dict=[imagenet_val] \
train_data.imagenet.loader.num_workers=64 \
train_data.imagenet.loader.batch_size=32 \
train_data.imagenet.args.channels=[2] \
val_data_dict.imagenet_val.args.channels=[2] \
transformations=rgb \
transformations.args.color_jitter_prob=0 \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch.target="ID" \
meta_arch.num_classes=1000 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/srinivasan/checkpoints/${NICKNAME}/epoch\\=99"
