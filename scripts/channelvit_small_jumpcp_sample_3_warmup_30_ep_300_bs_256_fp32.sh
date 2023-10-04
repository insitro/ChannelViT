NICKNAME="channelvit_small_jumpcp_sample_3_warmup_30_ep_300_bs_256_fp32"

python amlssl/main/main_dino.py \
wandb.project="amlssl-channelvit" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=300 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplechannelvit_small \
meta_arch.backbone.args.in_chans=8 \
meta_arch.warmup_epochs=30 \
data=jumpcp \
data.jumpcp.args.split=train \
data.jumpcp.args.sample_channels=3  \
data.jumpcp.loader.num_workers=32 \
data.jumpcp.loader.batch_size=32 \
data.jumpcp.loader.drop_last=True \
transformations=cell_dino
