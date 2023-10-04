# red is trained under insitro-user/yujia
# green and blue checkpoints are stored under insitro-user/srinivasan

for CH in 0 1 2
do

    NICKNAME="vit_small_ch_${CH}_imagenet_bs_256_fp32"

    python amlssl/main/main_dino.py \
    wandb.project="amlssl-channelvit" \
    nickname="${NICKNAME}" \
    trainer.devices=8 \
    trainer.precision=32 \
    trainer.max_epochs=100 \
    trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
    meta_arch/backbone=vit_small \
    meta_arch.backbone.args.in_chans=1 \
    data=imagenet \
    data.imagenet.args.channels=[${CH}] \
    data.imagenet.loader.num_workers=32 \
    data.imagenet.loader.batch_size=32 \
    data.imagenet.loader.drop_last=True \
    transformations=imagenet_dino

    #
    #
    python amlssl/main/main_linear_prob.py \
    wandb.project="amlssl-linear-prob" \
    nickname="${NICKNAME}-ch-${CH}-4-last-blocks" \
    data@train_data=imagenet \
    data@val_data_dict=[imagenet_val] \
    train_data.imagenet.loader.num_workers=64 \
    train_data.imagenet.loader.batch_size=32 \
    train_data.imagenet.args.channels=[${CH}] \
    val_data_dict.imagenet_val.args.channels=[${CH}] \
    transformations=rgb \
    transformations.args.color_jitter_prob=0 \
    trainer.devices=8 \
    trainer.max_epochs=100 \
    meta_arch.target="ID" \
    meta_arch.num_classes=1000 \
    meta_arch.checkpoint="s3://insitro-user/srinivasan/checkpoints/${NICKNAME}/epoch\\=99"
    #meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

done
