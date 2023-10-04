for CH in 0 1 2 3 4
do
    NICKNAME="vit_small_ch_${CH}_jumpcp_bs_160_fp_32"

    python amlssl/main/main_dino.py \
    wandb.project="amlssl-channelvit" \
    nickname="${NICKNAME}" \
    trainer.devices=8 \
    trainer.precision=32 \
    trainer.max_epochs=100 \
    trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
    meta_arch/backbone=vit_small \
    meta_arch.backbone.args.in_chans=1 \
    data=jumpcp \
    data.jumpcp.args.upsample=1 \
    data.jumpcp.args.split=train \
    data.jumpcp.args.channels=[${CH}] \
    data.jumpcp.loader.num_workers=64 \
    data.jumpcp.loader.batch_size=20 \
    data.jumpcp.loader.drop_last=True \
    transformations=cell_dino

    python amlssl/main/main_linear_prob.py \
    wandb.project="amlssl-linear-prob" \
    nickname="${NICKNAME}-ch-${CH}-mlp-4-last-blocks" \
    data@train_data=jumpcp \
    data@val_data_dict=[jumpcp_val,jumpcp_test] \
    train_data.jumpcp.loader.batch_size=32 \
    train_data.jumpcp.args.channels=[${CH}] \
    val_data_dict.jumpcp_val.args.channels=[${CH}] \
    val_data_dict.jumpcp_test.args.channels=[${CH}] \
    transformations=cell \
    transformations.args.channel_mask=[] \
    trainer.devices=8 \
    trainer.max_epochs=100 \
    meta_arch=mlp_prob \
    meta_arch.target="label" \
    meta_arch.num_classes=161 \
    meta_arch.n_last_blocks=4 \
    meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
done
