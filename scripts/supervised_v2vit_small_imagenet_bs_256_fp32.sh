NICKNAME="supervised_v2vit_small_imagenet_bs_256_fp32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v2vit_small \
meta_arch.target='ID' \
meta_arch.num_classes=1000 \
data@train_data=imagenet \
data@val_data_dict=[imagenet_val] \
train_data.imagenet.loader.num_workers=32 \
train_data.imagenet.loader.batch_size=32 \
train_data.imagenet.loader.drop_last=True \
val_data_dict.imagenet_val.loader.num_workers=32 \
val_data_dict.imagenet_val.loader.batch_size=32 \
val_data_dict.imagenet_val.loader.drop_last=True \
transformations@train_transformations=rgb \
transformations@val_transformations=rgb


# When checkpoint path is provided, we only run inference
python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-rgb" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=vit_small \
meta_arch.target='ID' \
meta_arch.num_classes=1000 \
data@val_data_dict=[imagenet_val] \
val_data_dict.imagenet_val.args.channels=[0,1,2] \
val_data_dict.imagenet_val.loader.num_workers=32 \
val_data_dict.imagenet_val.loader.batch_size=32 \
val_data_dict.imagenet_val.loader.drop_last=True \
val_data_dict.imagenet_val.loader.shuffle=False \
transformations@val_transformations=rgb \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# Red only
python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-r" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=vit_small \
meta_arch.target='ID' \
meta_arch.num_classes=1000 \
data@val_data_dict=[imagenet_val] \
val_data_dict.imagenet_val.args.channels=[0,1,2] \
val_data_dict.imagenet_val.loader.num_workers=32 \
val_data_dict.imagenet_val.loader.batch_size=32 \
val_data_dict.imagenet_val.loader.drop_last=True \
val_data_dict.imagenet_val.loader.shuffle=False \
transformations@val_transformations=rgb \
val_data_dict.imagenet_val.args.scale=3 \
val_transformations.args.channel_mask=[1,2] \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# green only
python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-g" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=vit_small \
meta_arch.target='ID' \
meta_arch.num_classes=1000 \
data@val_data_dict=[imagenet_val] \
val_data_dict.imagenet_val.args.channels=[0,1,2] \
val_data_dict.imagenet_val.loader.num_workers=32 \
val_data_dict.imagenet_val.loader.batch_size=32 \
val_data_dict.imagenet_val.loader.drop_last=True \
val_data_dict.imagenet_val.loader.shuffle=False \
transformations@val_transformations=rgb \
val_data_dict.imagenet_val.args.scale=3 \
val_transformations.args.channel_mask=[0,2] \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# blue only
python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-ch-b" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=vit_small \
meta_arch.target='ID' \
meta_arch.num_classes=1000 \
data@val_data_dict=[imagenet_val] \
val_data_dict.imagenet_val.args.channels=[0,1,2] \
val_data_dict.imagenet_val.loader.num_workers=32 \
val_data_dict.imagenet_val.loader.batch_size=32 \
val_data_dict.imagenet_val.loader.drop_last=True \
val_data_dict.imagenet_val.loader.shuffle=False \
transformations@val_transformations=rgb \
val_data_dict.imagenet_val.args.scale=3 \
val_transformations.args.channel_mask=[0,1] \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
