for CH in 0 1 2
do
    NICKNAME="supervised_vit_small_ch_${CH}_imagenet_bs_256_fp32"

    python amlssl/main/main_supervised.py \
    wandb.project="amlssl-supervised" \
    nickname="${NICKNAME}" \
    trainer.devices=8 \
    trainer.precision=32 \
    trainer.max_epochs=100 \
    trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
    meta_arch/backbone=vit_small \
    meta_arch.target='ID' \
    meta_arch.num_classes=1000 \
    meta_arch.backbone.args.in_chans=1 \
    data@train_data=imagenet \
    data@val_data_dict=[imagenet_val] \
    train_data.imagenet.loader.num_workers=32 \
    train_data.imagenet.loader.batch_size=32 \
    train_data.imagenet.loader.drop_last=True \
    train_data.imagenet.args.channels=[${CH}] \
    val_data_dict.imagenet_val.loader.num_workers=32 \
    val_data_dict.imagenet_val.loader.batch_size=32 \
    val_data_dict.imagenet_val.loader.drop_last=True \
    val_data_dict.imagenet_val.args.channels=[${CH}] \
    transformations@train_transformations=rgb \
    transformations@val_transformations=rgb

done
