NICKNAME="correlation_camelyon"

python amlssl/main/main_correlation.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
data@train_data=camelyon_train \
data@val_data_dict=[camelyon_test] \
transformations@train_transformations=rgb \
transformations@val_transformations=rgb

