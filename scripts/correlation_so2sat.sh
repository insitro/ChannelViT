NICKNAME="correlation_so2sat"

python amlssl/main/main_correlation.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
data@train_data=so2sat_hard \
data@val_data_dict=[so2sat_hard] \
train_data.so2sat_hard.loader.num_workers=32 \
train_data.so2sat_hard.loader.batch_size=32 \
train_data.so2sat_hard.loader.drop_last=True \
transformations@train_transformations=so2sat_hard \
transformations@val_transformations=so2sat_hard

