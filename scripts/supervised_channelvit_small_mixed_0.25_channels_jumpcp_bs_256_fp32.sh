NICKNAME="supervised_channelvit_small_mixed_0.25_channels_jumpcp_bs_256_fp32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
trainer.accumulate_grad_batches=1 \
meta_arch/backbone=channelvit_small \
meta_arch.backbone.args.in_chans=8 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=[jumpcp_fluro,jumpcp_fluro_bright] \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp_fluro.loader.num_workers=32 \
train_data.jumpcp_fluro.loader.batch_size=16 \
train_data.jumpcp_fluro.loader.drop_last=True \
train_data.jumpcp_fluro.args.channels=[0,1,2,3,4] \
train_data.jumpcp_fluro.args.split="fluro_0.25" \
train_data.jumpcp_fluro.args.upsample=1 \
train_data.jumpcp_fluro_bright.loader.num_workers=32 \
train_data.jumpcp_fluro_bright.loader.batch_size=16 \
train_data.jumpcp_fluro_bright.loader.drop_last=True \
train_data.jumpcp_fluro_bright.args.channels=[0,1,2,3,4,5,6,7] \
train_data.jumpcp_fluro_bright.args.split="fluro_bright_0.25" \
train_data.jumpcp_fluro_bright.args.upsample=1 \
val_data_dict.jumpcp_val.loader.num_workers=32 \
val_data_dict.jumpcp_val.loader.batch_size=32 \
val_data_dict.jumpcp_val.loader.drop_last=False \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4,5,6,7] \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=32 \
val_data_dict.jumpcp_test.loader.drop_last=False \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4,5,6,7] \
transformations@train_transformations=cell \
transformations@val_transformations=cell


python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-fluro" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
trainer.accumulate_grad_batches=1 \
meta_arch/backbone=samplev3channelvit_small \
meta_arch.backbone.args.in_chans=8 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=[jumpcp_fluro_bright] \
data@val_data_dict=[jumpcp_test] \
train_data.jumpcp_fluro_bright.loader.num_workers=32 \
train_data.jumpcp_fluro_bright.loader.batch_size=16 \
train_data.jumpcp_fluro_bright.loader.drop_last=True \
train_data.jumpcp_fluro_bright.args.channels=[0,1,2,3,4,5,6,7] \
train_data.jumpcp_fluro_bright.args.split="fluro_bright_0.5" \
train_data.jumpcp_fluro_bright.args.upsample=1 \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=32 \
val_data_dict.jumpcp_test.loader.drop_last=False \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations@train_transformations=cell \
transformations@val_transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}-evaluation-all" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
trainer.accumulate_grad_batches=1 \
meta_arch/backbone=samplev3channelvit_small \
meta_arch.backbone.args.in_chans=8 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=[jumpcp_fluro] \
data@val_data_dict=[jumpcp_test] \
train_data.jumpcp_fluro.loader.num_workers=32 \
train_data.jumpcp_fluro.loader.batch_size=16 \
train_data.jumpcp_fluro.loader.drop_last=True \
train_data.jumpcp_fluro.args.channels=[0,1,2,3,4,5,6,7] \
train_data.jumpcp_fluro.args.split="fluro_0.5" \
train_data.jumpcp_fluro.args.upsample=1 \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=32 \
val_data_dict.jumpcp_test.loader.drop_last=False \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4,5,6,7] \
transformations@train_transformations=cell \
transformations@val_transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
