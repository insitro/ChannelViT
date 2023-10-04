NICKNAME="supervised_v2vit_small_fluro_jumpcp_bs_256_fp32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v2vit_small \
meta_arch.backbone.args.in_chans=5 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.num_workers=32 \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.loader.drop_last=True \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.loader.num_workers=32 \
val_data_dict.jumpcp_val.loader.batch_size=32 \
val_data_dict.jumpcp_val.loader.drop_last=False \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=32 \
val_data_dict.jumpcp_test.loader.drop_last=False \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations@train_transformations=cell \
transformations@val_transformations=cell

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-0-1-2-3-4" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v2vit_small \
meta_arch.backbone.args.in_chans=5 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_test] \
train_data.jumpcp.loader.num_workers=32 \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.loader.drop_last=True \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=32 \
val_data_dict.jumpcp_test.loader.drop_last=True \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations@val_transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-1-2-3-4" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v2vit_small \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=jumpcp \
train_data.jumpcp.loader.num_workers=32 \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.loader.drop_last=True \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
data@val_data_dict=[jumpcp_test] \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=32 \
val_data_dict.jumpcp_test.loader.drop_last=True \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.scale=1.25 \
transformations@val_transformations=cell \
val_transformations.args.channel_mask=[0] \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-0-2-3-4" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v2vit_small \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=jumpcp \
train_data.jumpcp.loader.num_workers=32 \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.loader.drop_last=True \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
data@val_data_dict=[jumpcp_test] \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=32 \
val_data_dict.jumpcp_test.loader.drop_last=True \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.scale=1.25 \
transformations@val_transformations=cell \
val_transformations.args.channel_mask=[1] \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-0-1-3-4" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v2vit_small \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=jumpcp \
train_data.jumpcp.loader.num_workers=32 \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.loader.drop_last=True \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
data@val_data_dict=[jumpcp_test] \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=32 \
val_data_dict.jumpcp_test.loader.drop_last=True \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.scale=1.25 \
transformations@val_transformations=cell \
val_transformations.args.channel_mask=[2] \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-0-1-2-4" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v2vit_small \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=jumpcp \
train_data.jumpcp.loader.num_workers=32 \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.loader.drop_last=True \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
data@val_data_dict=[jumpcp_test] \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=32 \
val_data_dict.jumpcp_test.loader.drop_last=True \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.scale=1.25 \
transformations@val_transformations=cell \
val_transformations.args.channel_mask=[3] \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-0-1-2-3" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v2vit_small \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=jumpcp \
train_data.jumpcp.loader.num_workers=32 \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.loader.drop_last=True \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
data@val_data_dict=[jumpcp_test] \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=32 \
val_data_dict.jumpcp_test.loader.drop_last=True \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.scale=1.25 \
transformations@val_transformations=cell \
val_transformations.args.channel_mask=[4] \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"



IFS='~'
for CH in "0-1-2~3-4" "0-1-3~2-4" "0-1-4~2-3" "0-2-3~1-4" "0-2-4~1-3" "0-3-4~1-2" "1-2-3~0-4" "1-2-4~0-3" "1-3-4~0-2" "2-3-4~0-1"
do
    read -a CH <<< ${CH}
    python amlssl/main/main_supervised.py \
    wandb.project="amlssl-supervised" \
    nickname="${NICKNAME}-evaluation-ch-${CH[0]}" \
    trainer.devices=8 \
    trainer.precision=32 \
    trainer.max_epochs=100 \
    trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
    meta_arch/backbone=samplev2channelvit_small \
    meta_arch.target='label' \
    meta_arch.num_classes=161 \
    data@train_data=jumpcp \
    train_data.jumpcp.loader.num_workers=32 \
    train_data.jumpcp.loader.batch_size=32 \
    train_data.jumpcp.loader.drop_last=True \
    train_data.jumpcp.args.channels=[0,1,2,3,4] \
    data@val_data_dict=[jumpcp_test] \
    val_data_dict.jumpcp_test.loader.num_workers=32 \
    val_data_dict.jumpcp_test.loader.batch_size=32 \
    val_data_dict.jumpcp_test.loader.drop_last=True \
    val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
    val_data_dict.jumpcp_test.args.scale=1.67 \
    transformations@val_transformations=cell \
    val_transformations.args.channel_mask=[${CH[1]//-/,}] \
    checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
done

for CH in "0-1~2-3-4" "0-2~1-3-4" "0-3~1-2-4" "0-4~1-2-3" "1-2~0-3-4" "1-3~0-2-4" "1-4~0-2-3" "2-3~0-1-4" "2-4~0-1-3" "3-4~0-1-2"
do
    read -a CH <<< ${CH}
    python amlssl/main/main_supervised.py \
    wandb.project="amlssl-supervised" \
    nickname="${NICKNAME}-evaluation-ch-${CH[0]}" \
    trainer.devices=8 \
    trainer.precision=32 \
    trainer.max_epochs=100 \
    trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
    meta_arch/backbone=samplev2channelvit_small \
    meta_arch.target='label' \
    meta_arch.num_classes=161 \
    data@train_data=jumpcp \
    train_data.jumpcp.loader.num_workers=32 \
    train_data.jumpcp.loader.batch_size=32 \
    train_data.jumpcp.loader.drop_last=True \
    train_data.jumpcp.args.channels=[0,1,2,3,4] \
    data@val_data_dict=[jumpcp_test] \
    val_data_dict.jumpcp_test.loader.num_workers=32 \
    val_data_dict.jumpcp_test.loader.batch_size=32 \
    val_data_dict.jumpcp_test.loader.drop_last=True \
    val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
    val_data_dict.jumpcp_test.args.scale=2.5 \
    transformations@val_transformations=cell \
    val_transformations.args.channel_mask=[${CH[1]//-/,}] \
    checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
done

for CH in "0~1-2-3-4" "1~0-2-3-4" "2~0-1-3-4" "3~0-1-2-4" "4~0-1-2-3"
do
    read -a CH <<< ${CH}
    python amlssl/main/main_supervised.py \
    wandb.project="amlssl-supervised" \
    nickname="${NICKNAME}-evaluation-ch-${CH[0]}" \
    trainer.devices=8 \
    trainer.precision=32 \
    trainer.max_epochs=100 \
    trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
    meta_arch/backbone=samplev2channelvit_small \
    meta_arch.target='label' \
    meta_arch.num_classes=161 \
    data@train_data=jumpcp \
    train_data.jumpcp.loader.num_workers=32 \
    train_data.jumpcp.loader.batch_size=32 \
    train_data.jumpcp.loader.drop_last=True \
    train_data.jumpcp.args.channels=[0,1,2,3,4] \
    data@val_data_dict=[jumpcp_test] \
    val_data_dict.jumpcp_test.loader.num_workers=32 \
    val_data_dict.jumpcp_test.loader.batch_size=32 \
    val_data_dict.jumpcp_test.loader.drop_last=True \
    val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
    val_data_dict.jumpcp_test.args.scale=5 \
    transformations@val_transformations=cell \
    val_transformations.args.channel_mask=[${CH[1]//-/,}] \
    checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
done
