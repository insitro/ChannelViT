NICKNAME="channelvit_small_fluro_jumpcp_sample_3_bs_256_fp32"

python amlssl/main/main_dino.py \
wandb.project="amlssl-channelvit" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplechannelvit_small \
meta_arch.backbone.args.in_chans=5 \
data=jumpcp \
data.jumpcp.args.split=train \
data.jumpcp.args.sample_channels=3  \
data.jumpcp.args.channels=[0,1,2,3,4] \
data.jumpcp.loader.num_workers=32 \
data.jumpcp.loader.batch_size=32 \
data.jumpcp.loader.drop_last=True \
transformations=cell_dino
