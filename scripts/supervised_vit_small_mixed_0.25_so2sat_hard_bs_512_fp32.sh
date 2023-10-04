NICKNAME="supervised_vit_small_mixed_0.25_so2sat_hard_bs_512_fp32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
trainer.accumulate_grad_batches=1 \
meta_arch/backbone=vit_small \
meta_arch.backbone.args.in_chans=18 \
meta_arch.backbone.args.patch_size=8 \
meta_arch.target='label' \
meta_arch.data_ratio=0.25 \
meta_arch.num_classes=17 \
data@train_data=[so2sat_hard_sentinel_1,so2sat_hard] \
data@val_data_dict=[so2sat_hard_test_sentinel_1,so2sat_hard_test_sentinel_2,so2sat_hard_test] \
train_data.so2sat_hard_sentinel_1.loader.num_workers=32 \
train_data.so2sat_hard_sentinel_1.loader.batch_size=64 \
train_data.so2sat_hard_sentinel_1.loader.drop_last=True \
train_data.so2sat_hard_sentinel_1.args.upsample=1 \
train_data.so2sat_hard_sentinel_1.args.scale=2.25 \
train_data.so2sat_hard_sentinel_1.args.channel_mask=True \
train_data.so2sat_hard.loader.num_workers=32 \
train_data.so2sat_hard.loader.batch_size=64 \
train_data.so2sat_hard.loader.drop_last=True \
train_data.so2sat_hard.args.split=0.25 \
val_data_dict.so2sat_hard_test_sentinel_1.loader.num_workers=32 \
val_data_dict.so2sat_hard_test_sentinel_1.loader.batch_size=32 \
val_data_dict.so2sat_hard_test_sentinel_1.loader.drop_last=False \
val_data_dict.so2sat_hard_test_sentinel_1.args.channel_mask=True \
val_data_dict.so2sat_hard_test_sentinel_1.args.scale=2.25 \
val_data_dict.so2sat_hard_test_sentinel_2.loader.num_workers=32 \
val_data_dict.so2sat_hard_test_sentinel_2.loader.batch_size=32 \
val_data_dict.so2sat_hard_test_sentinel_2.loader.drop_last=False \
val_data_dict.so2sat_hard_test_sentinel_2.args.channel_mask=True \
val_data_dict.so2sat_hard_test_sentinel_2.args.scale=1.8 \
val_data_dict.so2sat_hard_test.loader.num_workers=32 \
val_data_dict.so2sat_hard_test.loader.batch_size=32 \
val_data_dict.so2sat_hard_test.loader.drop_last=False \
transformations@train_transformations=so2sat_hard \
transformations@val_transformations=so2sat_hard


NICKNAME="supervised_vit_small_mixed_0.50_so2sat_hard_bs_512_fp32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
trainer.accumulate_grad_batches=1 \
meta_arch/backbone=vit_small \
meta_arch.backbone.args.in_chans=18 \
meta_arch.backbone.args.patch_size=8 \
meta_arch.target='label' \
meta_arch.data_ratio=0.50 \
meta_arch.num_classes=17 \
data@train_data=[so2sat_hard_sentinel_1,so2sat_hard] \
data@val_data_dict=[so2sat_hard_test_sentinel_1,so2sat_hard_test_sentinel_2,so2sat_hard_test] \
train_data.so2sat_hard_sentinel_1.loader.num_workers=32 \
train_data.so2sat_hard_sentinel_1.loader.batch_size=64 \
train_data.so2sat_hard_sentinel_1.loader.drop_last=True \
train_data.so2sat_hard_sentinel_1.args.upsample=1 \
train_data.so2sat_hard_sentinel_1.args.scale=2.25 \
train_data.so2sat_hard_sentinel_1.args.channel_mask=True \
train_data.so2sat_hard.loader.num_workers=32 \
train_data.so2sat_hard.loader.batch_size=64 \
train_data.so2sat_hard.loader.drop_last=True \
train_data.so2sat_hard.args.split=0.50 \
val_data_dict.so2sat_hard_test_sentinel_1.loader.num_workers=32 \
val_data_dict.so2sat_hard_test_sentinel_1.loader.batch_size=32 \
val_data_dict.so2sat_hard_test_sentinel_1.loader.drop_last=False \
val_data_dict.so2sat_hard_test_sentinel_1.args.channel_mask=True \
val_data_dict.so2sat_hard_test_sentinel_1.args.scale=2.25 \
val_data_dict.so2sat_hard_test_sentinel_2.loader.num_workers=32 \
val_data_dict.so2sat_hard_test_sentinel_2.loader.batch_size=32 \
val_data_dict.so2sat_hard_test_sentinel_2.loader.drop_last=False \
val_data_dict.so2sat_hard_test_sentinel_2.args.channel_mask=True \
val_data_dict.so2sat_hard_test_sentinel_2.args.scale=1.8 \
val_data_dict.so2sat_hard_test.loader.num_workers=32 \
val_data_dict.so2sat_hard_test.loader.batch_size=32 \
val_data_dict.so2sat_hard_test.loader.drop_last=False \
transformations@train_transformations=so2sat_hard \
transformations@val_transformations=so2sat_hard


NICKNAME="supervised_vit_small_mixed_0.75_so2sat_hard_bs_512_fp32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
trainer.accumulate_grad_batches=1 \
meta_arch/backbone=vit_small \
meta_arch.backbone.args.in_chans=18 \
meta_arch.backbone.args.patch_size=8 \
meta_arch.target='label' \
meta_arch.data_ratio=0.75 \
meta_arch.num_classes=17 \
data@train_data=[so2sat_hard_sentinel_1,so2sat_hard] \
data@val_data_dict=[so2sat_hard_test_sentinel_1,so2sat_hard_test_sentinel_2,so2sat_hard_test] \
train_data.so2sat_hard_sentinel_1.loader.num_workers=32 \
train_data.so2sat_hard_sentinel_1.loader.batch_size=64 \
train_data.so2sat_hard_sentinel_1.loader.drop_last=True \
train_data.so2sat_hard_sentinel_1.args.upsample=1 \
train_data.so2sat_hard_sentinel_1.args.scale=2.25 \
train_data.so2sat_hard_sentinel_1.args.channel_mask=True \
train_data.so2sat_hard.loader.num_workers=32 \
train_data.so2sat_hard.loader.batch_size=64 \
train_data.so2sat_hard.loader.drop_last=True \
train_data.so2sat_hard.args.split=0.75 \
val_data_dict.so2sat_hard_test_sentinel_1.loader.num_workers=32 \
val_data_dict.so2sat_hard_test_sentinel_1.loader.batch_size=32 \
val_data_dict.so2sat_hard_test_sentinel_1.loader.drop_last=False \
val_data_dict.so2sat_hard_test_sentinel_1.args.channel_mask=True \
val_data_dict.so2sat_hard_test_sentinel_1.args.scale=2.25 \
val_data_dict.so2sat_hard_test_sentinel_2.loader.num_workers=32 \
val_data_dict.so2sat_hard_test_sentinel_2.loader.batch_size=32 \
val_data_dict.so2sat_hard_test_sentinel_2.loader.drop_last=False \
val_data_dict.so2sat_hard_test_sentinel_2.args.channel_mask=True \
val_data_dict.so2sat_hard_test_sentinel_2.args.scale=1.8 \
val_data_dict.so2sat_hard_test.loader.num_workers=32 \
val_data_dict.so2sat_hard_test.loader.batch_size=32 \
val_data_dict.so2sat_hard_test.loader.drop_last=False \
transformations@train_transformations=so2sat_hard \
transformations@val_transformations=so2sat_hard
