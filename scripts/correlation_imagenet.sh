NICKNAME="correlation_imagenet"

python amlssl/main/main_correlation.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
data@train_data=imagenet \
data@val_data_dict=[imagenet_val] \
train_data.imagenet.loader.num_workers=32 \
train_data.imagenet.loader.batch_size=32 \
train_data.imagenet.loader.drop_last=True \
transformations@train_transformations=rgb \
transformations@val_transformations=rgb

