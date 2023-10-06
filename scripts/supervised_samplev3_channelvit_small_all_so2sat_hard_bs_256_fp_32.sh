NICKNAME="supervised_samplev3_channelvit_small_all_so2sat_hard_bs_256_fp_32"

##python amlssl/main/main_supervised.py \
##wandb.project="amlssl-supervised" \
##nickname="${NICKNAME}" \
##trainer.devices=8 \
##trainer.precision=32 \
##trainer.max_epochs=100 \
##trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
##meta_arch/backbone=samplev3channelvit_small \
##meta_arch.backbone.args.in_chans=18 \
##meta_arch.backbone.args.patch_size=8 \
##meta_arch.target='label' \
##meta_arch.num_classes=17 \
##data@train_data=so2sat_hard \
##data@val_data_dict=[so2sat_hard_test] \
##train_data.so2sat_hard.loader.num_workers=64 \
##train_data.so2sat_hard.loader.batch_size=64 \
##train_data.so2sat_hard.loader.drop_last=True \
##val_data_dict.so2sat_hard_test.loader.num_workers=64 \
##val_data_dict.so2sat_hard_test.loader.batch_size=64 \
##val_data_dict.so2sat_hard_test.loader.drop_last=False \
##transformations@train_transformations=so2sat_hard \
##transformations@val_transformations=so2sat_hard


#python amlssl/main/main_supervised.py \
#wandb.project="amlssl-supervised" \
#nickname="${NICKNAME}-evaluation-sentinel-1" \
#trainer.devices=8 \
#trainer.precision=32 \
#trainer.max_epochs=100 \
#trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
#meta_arch/backbone=vit_small \
#meta_arch.backbone.args.in_chans=18 \
#meta_arch.backbone.args.patch_size=8 \
#meta_arch.target='label' \
#meta_arch.num_classes=17 \
#data@train_data=so2sat_hard \
#data@val_data_dict=[so2sat_hard_test] \
#train_data.so2sat_hard.loader.num_workers=64 \
#train_data.so2sat_hard.loader.batch_size=64 \
#train_data.so2sat_hard.loader.drop_last=True \
#val_data_dict.so2sat_hard_test.loader.num_workers=64 \
#val_data_dict.so2sat_hard_test.loader.batch_size=64 \
#val_data_dict.so2sat_hard_test.loader.drop_last=False \
#val_data_dict.so2sat_hard_test.args.channels=[0,1,2,3,4,5,6,7] \
#transformations@train_transformations=so2sat_hard \
#transformations@val_transformations=so2sat_hard \
#checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
##checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=10"


#python amlssl/main/main_supervised.py \
#wandb.project="amlssl-supervised" \
#nickname="${NICKNAME}-evaluation-sentinel-2" \
#trainer.devices=8 \
#trainer.precision=32 \
#trainer.max_epochs=100 \
#trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
#meta_arch/backbone=vit_small \
#meta_arch.backbone.args.in_chans=18 \
#meta_arch.backbone.args.patch_size=8 \
#meta_arch.target='label' \
#meta_arch.num_classes=17 \
#data@train_data=so2sat_hard \
#data@val_data_dict=[so2sat_hard_test] \
#train_data.so2sat_hard.loader.num_workers=64 \
#train_data.so2sat_hard.loader.batch_size=64 \
#train_data.so2sat_hard.loader.drop_last=True \
#val_data_dict.so2sat_hard_test.loader.num_workers=64 \
#val_data_dict.so2sat_hard_test.loader.batch_size=64 \
#val_data_dict.so2sat_hard_test.loader.drop_last=False \
#val_data_dict.so2sat_hard_test.args.channels=[8,9,10,11,12,13,14,15,16,17] \
#transformations@train_transformations=so2sat_hard \
#transformations@val_transformations=so2sat_hard \
#checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
##checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=10"
##checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
##
#
python amlssl/main/main_visualize.py \
trainer.accumulate_grad_batches=32 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev3channelvit_small \
meta_arch.backbone.args.in_chans=8 \
meta_arch.backbone.args.patch_size=8 \
meta_arch.patch_size=8 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.num_workers=32 \
train_data.jumpcp.loader.batch_size=1 \
train_data.jumpcp.loader.drop_last=True \
train_data.jumpcp.args.channels=[0,1,2,3,4,5,6,7] \
val_data_dict.jumpcp_val.loader.num_workers=32 \
val_data_dict.jumpcp_val.loader.batch_size=8 \
val_data_dict.jumpcp_val.loader.drop_last=False \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4,5,6,7] \
val_data_dict.jumpcp_test.loader.num_workers=32 \
val_data_dict.jumpcp_test.loader.batch_size=8 \
val_data_dict.jumpcp_test.loader.drop_last=False \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4,5,6,7] \
transformations@train_transformations=cell \
transformations@val_transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"