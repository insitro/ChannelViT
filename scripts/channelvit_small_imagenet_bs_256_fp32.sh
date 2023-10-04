NICKNAME="channelvit_small_imagenet_bs_256_fp32"

python amlssl/main/main_dino.py \
wandb.project="amlssl-channelvit" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=channelvit_small \
meta_arch.backbone.args.in_chans=3 \
data=imagenet \
data.imagenet.args.channels=[0,1,2] \
data.imagenet.loader.num_workers=32 \
data.imagenet.loader.batch_size=32 \
data.imagenet.loader.drop_last=True \
transformations=imagenet_dino

