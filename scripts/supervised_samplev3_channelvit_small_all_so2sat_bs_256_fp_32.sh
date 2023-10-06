NICKNAME="supervised_samplev3_channelvit_small_all_so2sat_bs_256_fp_32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev3channelvit_small \
meta_arch.backbone.args.in_chans=18 \
meta_arch.backbone.args.patch_size=8 \
meta_arch.target='label' \
meta_arch.num_classes=17 \
data@train_data=so2sat \
data@val_data_dict=[so2sat_test] \
train_data.so2sat.loader.num_workers=64 \
train_data.so2sat.loader.batch_size=64 \
train_data.so2sat.loader.drop_last=True \
val_data_dict.so2sat_test.loader.num_workers=64 \
val_data_dict.so2sat_test.loader.batch_size=64 \
val_data_dict.so2sat_test.loader.drop_last=False \
transformations@train_transformations=so2sat \
transformations@val_transformations=so2sat


#python amlssl/main/main_supervised_evalall.py \
#wandb.project="amlssl-supervised" \
#nickname="${NICKNAME}" \
#trainer.devices=8 \
#trainer.precision=32 \
#trainer.max_epochs=100 \
#trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
#meta_arch/backbone=v3vit_small \
#meta_arch.backbone.args.in_chans=8 \
#meta_arch.target='label' \
#meta_arch.num_classes=161 \
#transformation_mask=True \
#data@val_data=jumpcp_test \
#val_data.jumpcp_test.loader.num_workers=32 \
#val_data.jumpcp_test.loader.batch_size=32 \
#val_data.jumpcp_test.loader.drop_last=False \
#val_data.jumpcp_test.args.channels=[0,1,2,3,4,5,6,7] \
#val_data.jumpcp_test.args.scale=1 \
#transformations=cell \
#checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"