NICKNAME="correlation_jumpcp"

python amlssl/main/main_correlation.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_test] \
transformations@train_transformations=cell \
transformations@val_transformations=cell
val_transformations.normalization.mean=[0,0,0,0,0,0,0,0] \
val_transformations.normalization.std=[1,1,1,1,1,1,1,1]

