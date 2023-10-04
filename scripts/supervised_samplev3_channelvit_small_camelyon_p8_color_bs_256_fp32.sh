NICKNAME="supervised_samplev3_channelvit_small_camelyon_p8_color_bs_256_fp32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev3channelvit_small \
meta_arch.target='tumor' \
meta_arch.backbone.args.in_chans=3 \
meta_arch.backbone.args.patch_size=8 \
meta_arch.num_classes=2 \
data@train_data=camelyon_train \
data@val_data_dict=[camelyon_id_val] \
train_data.camelyon_train.loader.num_workers=64 \
train_data.camelyon_train.loader.batch_size=32 \
train_data.camelyon_train.loader.drop_last=True \
train_data.camelyon_train.args.channels=[0,1,2] \
val_data_dict.camelyon_id_val.loader.num_workers=64 \
val_data_dict.camelyon_id_val.loader.batch_size=32 \
val_data_dict.camelyon_id_val.loader.drop_last=False \
val_data_dict.camelyon_id_val.args.channels=[0,1,2] \
transformations@train_transformations=camelyon_rgb \
train_transformations.args.color_jitter_prob=0.8 \
val_transformations.args.color_jitter_prob=0.8 \
transformations@val_transformations=camelyon_rgb

python amlssl/main/main_supervised_evalall.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-id-val" \
transformations=camelyon_rgb \
transformation_mask=False \
channels=[0,1,2] \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v3vit_small \
meta_arch.backbone.args.in_chans=3 \
meta_arch.target='tumor' \
meta_arch.num_classes=2 \
data@val_data=[camelyon_id_val] \
val_data.camelyon_id_val.loader.num_workers=32 \
val_data.camelyon_id_val.loader.batch_size=32 \
val_data.camelyon_id_val.loader.drop_last=False \
val_data.camelyon_id_val.args.channels=[0,1,2] \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_supervised_evalall.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-test" \
transformations=camelyon_rgb \
transformation_mask=False \
channels=[0,1,2] \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v3vit_small \
meta_arch.backbone.args.in_chans=3 \
meta_arch.target='tumor' \
meta_arch.num_classes=2 \
data@val_data=[camelyon_test] \
val_data.camelyon_test.loader.num_workers=32 \
val_data.camelyon_test.loader.batch_size=32 \
val_data.camelyon_test.loader.drop_last=False \
val_data.camelyon_test.args.channels=[0,1,2] \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

python amlssl/main/main_visualize.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev3channelvit_small \
meta_arch.target='tumor' \
meta_arch.backbone.args.in_chans=3 \
meta_arch.backbone.args.patch_size=8 \
meta_arch.num_classes=2 \
data@train_data=camelyon_train \
data@val_data_dict=[camelyon_id_val] \
train_data.camelyon_train.loader.num_workers=64 \
train_data.camelyon_train.loader.batch_size=32 \
train_data.camelyon_train.loader.drop_last=True \
train_data.camelyon_train.args.channels=[0,1,2] \
val_data_dict.camelyon_id_val.loader.num_workers=64 \
val_data_dict.camelyon_id_val.loader.batch_size=32 \
val_data_dict.camelyon_id_val.loader.drop_last=False \
val_data_dict.camelyon_id_val.args.channels=[0,1,2] \
transformations@train_transformations=camelyon_rgb \
train_transformations.args.color_jitter_prob=0.8 \
val_transformations.args.color_jitter_prob=0.8 \
transformations@val_transformations=camelyon_rgb \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
