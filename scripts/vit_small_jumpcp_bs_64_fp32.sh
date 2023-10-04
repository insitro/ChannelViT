NICKNAME="vit_small_jumpcp_bs_64_fp32"

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
data.jumpcp.loader.batch_size=8 \
data.jumpcp.loader.drop_last=True \
transformations=cell_dino
