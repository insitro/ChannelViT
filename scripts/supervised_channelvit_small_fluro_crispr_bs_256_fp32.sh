NICKNAME="supervised_channelvit_small_fluro_crispr_bs_256_fp32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=channelvit_small \
meta_arch.backbone.args.in_chans=5 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=crispr \
data@val_data_dict=[crispr_val,crispr_test] \
train_data.crispr.loader.num_workers=32 \
train_data.crispr.loader.batch_size=32 \
train_data.crispr.loader.drop_last=True \
train_data.crispr.args.channels=[0,1,2,3,4] \
val_data_dict.crispr_val.loader.num_workers=32 \
val_data_dict.crispr_val.loader.batch_size=32 \
val_data_dict.crispr_val.loader.drop_last=False \
val_data_dict.crispr_val.args.channels=[0,1,2,3,4] \
val_data_dict.crispr_test.loader.num_workers=32 \
val_data_dict.crispr_test.loader.batch_size=32 \
val_data_dict.crispr_test.loader.drop_last=False \
val_data_dict.crispr_test.args.channels=[0,1,2,3,4] \
transformations@train_transformations=cell \
transformations@val_transformations=cell

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-0-1-2-3-4" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev2channelvit_small \
meta_arch.backbone.args.in_chans=5 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=crispr \
data@val_data_dict=[crispr_test] \
train_data.crispr.loader.num_workers=32 \
train_data.crispr.loader.batch_size=32 \
train_data.crispr.loader.drop_last=True \
train_data.crispr.args.channels=[0,1,2,3,4] \
val_data_dict.crispr_test.loader.num_workers=32 \
val_data_dict.crispr_test.loader.batch_size=32 \
val_data_dict.crispr_test.loader.drop_last=True \
val_data_dict.crispr_test.args.channels=[0,1,2,3,4] \
transformations@val_transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-1-2-3-4" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev2channelvit_small \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=crispr \
train_data.crispr.loader.num_workers=32 \
train_data.crispr.loader.batch_size=32 \
train_data.crispr.loader.drop_last=True \
train_data.crispr.args.channels=[0,1,2,3,4] \
data@val_data_dict=[crispr_test] \
val_data_dict.crispr_test.loader.num_workers=32 \
val_data_dict.crispr_test.loader.batch_size=32 \
val_data_dict.crispr_test.loader.drop_last=True \
val_data_dict.crispr_test.args.channels=[1,2,3,4] \
transformations@val_transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-0-2-3-4" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev2channelvit_small \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=crispr \
train_data.crispr.loader.num_workers=32 \
train_data.crispr.loader.batch_size=32 \
train_data.crispr.loader.drop_last=True \
train_data.crispr.args.channels=[0,1,2,3,4] \
data@val_data_dict=[crispr_test] \
val_data_dict.crispr_test.loader.num_workers=32 \
val_data_dict.crispr_test.loader.batch_size=32 \
val_data_dict.crispr_test.loader.drop_last=True \
val_data_dict.crispr_test.args.channels=[0,2,3,4] \
transformations@val_transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-0-1-3-4" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev2channelvit_small \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=crispr \
train_data.crispr.loader.num_workers=32 \
train_data.crispr.loader.batch_size=32 \
train_data.crispr.loader.drop_last=True \
train_data.crispr.args.channels=[0,1,2,3,4] \
data@val_data_dict=[crispr_test] \
val_data_dict.crispr_test.loader.num_workers=32 \
val_data_dict.crispr_test.loader.batch_size=32 \
val_data_dict.crispr_test.loader.drop_last=True \
val_data_dict.crispr_test.args.channels=[0,1,3,4] \
transformations@val_transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-0-1-2-4" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev2channelvit_small \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=crispr \
train_data.crispr.loader.num_workers=32 \
train_data.crispr.loader.batch_size=32 \
train_data.crispr.loader.drop_last=True \
train_data.crispr.args.channels=[0,1,2,3,4] \
data@val_data_dict=[crispr_test] \
val_data_dict.crispr_test.loader.num_workers=32 \
val_data_dict.crispr_test.loader.batch_size=32 \
val_data_dict.crispr_test.loader.drop_last=True \
val_data_dict.crispr_test.args.channels=[0,1,2,4] \
transformations@val_transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-0-1-2-3" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev2channelvit_small \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=crispr \
train_data.crispr.loader.num_workers=32 \
train_data.crispr.loader.batch_size=32 \
train_data.crispr.loader.drop_last=True \
train_data.crispr.args.channels=[0,1,2,3,4] \
data@val_data_dict=[crispr_test] \
val_data_dict.crispr_test.loader.num_workers=32 \
val_data_dict.crispr_test.loader.batch_size=32 \
val_data_dict.crispr_test.loader.drop_last=True \
val_data_dict.crispr_test.args.channels=[0,1,2,3] \
transformations@val_transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"



for CH in 0-1-2 0-1-3 0-1-4 0-2-3 0-2-4 0-3-4 1-2-3 1-2-4 1-3-4 2-3-4
do
    python amlssl/main/main_supervised.py \
    wandb.project="amlssl-supervised" \
    nickname="${NICKNAME}-evaluation-ch-${CH}" \
    trainer.devices=8 \
    trainer.precision=32 \
    trainer.max_epochs=100 \
    trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
    meta_arch/backbone=samplev2channelvit_small \
    meta_arch.target='label' \
    meta_arch.num_classes=161 \
    data@train_data=crispr \
    train_data.crispr.loader.num_workers=32 \
    train_data.crispr.loader.batch_size=32 \
    train_data.crispr.loader.drop_last=True \
    train_data.crispr.args.channels=[0,1,2,3,4] \
    data@val_data_dict=[crispr_test] \
    val_data_dict.crispr_test.loader.num_workers=32 \
    val_data_dict.crispr_test.loader.batch_size=32 \
    val_data_dict.crispr_test.loader.drop_last=True \
    val_data_dict.crispr_test.args.channels=[${CH}] \
    transformations@val_transformations=cell \
    checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
done

for CH in 0-1 0-2 0-3 0-4 1-2 1-3 1-4 2-3 2-4 3-4
do
    python amlssl/main/main_supervised.py \
    wandb.project="amlssl-supervised" \
    nickname="${NICKNAME}-evaluation-ch-${CH}" \
    trainer.devices=8 \
    trainer.precision=32 \
    trainer.max_epochs=100 \
    trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
    meta_arch/backbone=samplev2channelvit_small \
    meta_arch.target='label' \
    meta_arch.num_classes=161 \
    data@train_data=crispr \
    train_data.crispr.loader.num_workers=32 \
    train_data.crispr.loader.batch_size=32 \
    train_data.crispr.loader.drop_last=True \
    train_data.crispr.args.channels=[0,1,2,3,4] \
    data@val_data_dict=[crispr_test] \
    val_data_dict.crispr_test.loader.num_workers=32 \
    val_data_dict.crispr_test.loader.batch_size=32 \
    val_data_dict.crispr_test.loader.drop_last=True \
    val_data_dict.crispr_test.args.channels=[${CH}] \
    transformations@val_transformations=cell \
    checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
done

for CH in 0 1 2 3 4
do
    python amlssl/main/main_supervised.py \
    wandb.project="amlssl-supervised" \
    nickname="${NICKNAME}-evaluation-ch-${CH}" \
    trainer.devices=8 \
    trainer.precision=32 \
    trainer.max_epochs=100 \
    trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
    meta_arch/backbone=samplev2channelvit_small \
    meta_arch.target='label' \
    meta_arch.num_classes=161 \
    data@train_data=crispr \
    train_data.crispr.loader.num_workers=32 \
    train_data.crispr.loader.batch_size=32 \
    train_data.crispr.loader.drop_last=True \
    train_data.crispr.args.channels=[0,1,2,3,4] \
    data@val_data_dict=[crispr_test] \
    val_data_dict.crispr_test.loader.num_workers=32 \
    val_data_dict.crispr_test.loader.batch_size=32 \
    val_data_dict.crispr_test.loader.drop_last=True \
    val_data_dict.crispr_test.args.channels=[${CH}] \
    transformations@val_transformations=cell \
    checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
done
