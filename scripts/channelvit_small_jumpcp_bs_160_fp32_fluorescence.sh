NICKNAME="channelvit_small_jumpcp_bs_160_fp32_fluorescence"

python amlssl/main/main_dino.py \
wandb.project="amlssl-channelvit" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/srinivasan/checkpoints/${NICKNAME}" \
meta_arch/backbone=channelvit_small \
meta_arch.backbone.args.in_chans=5 \
data=jumpcp \
data.jumpcp.args.split=train \
data.jumpcp.args.channels=[0,1,2,3,4] \
data.jumpcp.loader.num_workers=32 \
data.jumpcp.loader.batch_size=20 \
data.jumpcp.loader.drop_last=True \
transformations=cell_dino


# MLP probing using the last 4 blocks
# Train and eval only on fluro channels
# Note that the model only receive inputs with 5 channels
# That's why in the train_data and val_data_dict, we only have 5 channels.
# For the channel_mask in the trasnforamtions, we don't do anything here.
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-fluro-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[] \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/srinivasan/checkpoints/${NICKNAME}/epoch\\=99"
