NICKNAME="supervised_weighted_v3vit_small_all_bigearthmm43_bs_256_fp32"

python amlssl/main/main_supervised_multiclass.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v3vit_small \
meta_arch.backbone.args.in_chans=14 \
meta_arch.backbone.args.patch_size=10 \
meta_arch.weighted_loss=4 \
meta_arch.target='label_original' \
meta_arch.num_classes=43 \
data@train_data=bigearthmm \
data@val_data_dict=[bigearthmm_val,bigearthmm_test] \
train_data.bigearthmm.loader.num_workers=32 \
train_data.bigearthmm.loader.batch_size=32 \
train_data.bigearthmm.loader.drop_last=True \
train_data.bigearthmm.args.channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
val_data_dict.bigearthmm_val.loader.num_workers=32 \
val_data_dict.bigearthmm_val.loader.batch_size=32 \
val_data_dict.bigearthmm_val.loader.drop_last=False \
val_data_dict.bigearthmm_val.args.channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
val_data_dict.bigearthmm_test.loader.num_workers=32 \
val_data_dict.bigearthmm_test.loader.batch_size=32 \
val_data_dict.bigearthmm_test.loader.drop_last=False \
val_data_dict.bigearthmm_test.args.channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] \
transformations@train_transformations=bigearth_mm \
transformations@val_transformations=bigearth_mm


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
