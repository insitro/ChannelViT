NICKNAME="supervised_v3vit_base_p_6_all_bigearth_bs_256_fp32"
sleep 1h

python amlssl/main/main_supervised_multiclass.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v3vit_base \
meta_arch.backbone.args.in_chans=12 \
meta_arch.backbone.args.patch_size=6 \
meta_arch.target='label' \
meta_arch.num_classes=19 \
data@train_data=bigearth \
data@val_data_dict=[bigearth_val,bigearth_test] \
train_data.bigearth.loader.num_workers=32 \
train_data.bigearth.loader.batch_size=32 \
train_data.bigearth.loader.drop_last=True \
train_data.bigearth.args.channels=[0,1,2,3,4,5,6,7,8,9,10,11] \
val_data_dict.bigearth_val.loader.num_workers=32 \
val_data_dict.bigearth_val.loader.batch_size=32 \
val_data_dict.bigearth_val.loader.drop_last=False \
val_data_dict.bigearth_val.args.channels=[0,1,2,3,4,5,6,7,8,9,10,11] \
val_data_dict.bigearth_test.loader.num_workers=32 \
val_data_dict.bigearth_test.loader.batch_size=32 \
val_data_dict.bigearth_test.loader.drop_last=False \
val_data_dict.bigearth_test.args.channels=[0,1,2,3,4,5,6,7,8,9,10,11] \
transformations@train_transformations=bigearth \
transformations@val_transformations=bigearth


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
