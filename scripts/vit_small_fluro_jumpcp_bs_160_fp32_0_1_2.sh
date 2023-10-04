NICKNAME="vit_small_fluro_0_1_2_jumpcp_bs_160_fp32"

python amlssl/main/main_dino.py \
wandb.project="amlssl-channelvit" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/srinivasan/checkpoints/${NICKNAME}" \
meta_arch/backbone=vit_small \
meta_arch.backbone.args.in_chans=3 \
data=jumpcp \
data.jumpcp.args.upsample=1 \
data.jumpcp.args.split=train \
data.jumpcp.args.channels=[0,1,2] \
data.jumpcp.loader.num_workers=64 \
data.jumpcp.loader.batch_size=20 \
data.jumpcp.loader.drop_last=True \
transformations=cell_dino
