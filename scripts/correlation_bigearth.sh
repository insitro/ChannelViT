NICKNAME="correlation_bigearth"

python amlssl/main/main_correlation.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
data@train_data=bigearth \
data@val_data_dict=[bigearth_test] \
train_data.bigearth.loader.num_workers=32 \
train_data.bigearth.loader.batch_size=32 \
train_data.bigearth.loader.drop_last=True \
transformations@train_transformations=bigearth \
transformations@val_transformations=bigearth

