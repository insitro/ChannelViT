NICKNAME="vit_small_jumpcp_bs_256_fp32"

python amlssl/main/main_dino.py \
wandb.project="amlssl-channelvit" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=vit_small \
meta_arch.backbone.args.in_chans=8 \
data=jumpcp \
data.jumpcp.args.upsample=1 \
data.jumpcp.args.split=train \
data.jumpcp.args.channels=[0,1,2,3,4,5,6,7] \
data.jumpcp.loader.num_workers=64 \
data.jumpcp.loader.batch_size=32 \
data.jumpcp.loader.drop_last=True \
transformations=cell_dino


# Linear probing with a linear classifier using the last 4 blocks
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
transformations=cell \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using the last 4 blocks
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
transformations=cell \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using the last 4 blocks
# Train and eval only on fluro channels
# Note that the model still receive inputs with 8 channels
# That's why in the train_data and val_data_dict, we have all 8 channels.
# It's just that we mask out the 3 brightfield channels with their mean.
# We control this by setting channel mask to [5,6,7] in the transformations args.
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-fluro-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2,3,4,5,6,7] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4,5,6,7] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4,5,6,7] \
transformations=cell \
transformations.args.channel_mask=[5,6,7] \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
