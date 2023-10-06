NICKNAME="supervised_weighted_samplev3_channelvit_small_all_bigearth_bs_256_fp32"

#python amlssl/main/main_supervised_multiclass.py \
#wandb.project="amlssl-supervised" \
#nickname="${NICKNAME}" \
#trainer.devices=8 \
#trainer.precision=32 \
#trainer.max_epochs=100 \
#trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
#meta_arch/backbone=samplev3channelvit_small \
#trainer.accumulate_grad_batches=2 \
#meta_arch.backbone.args.in_chans=12 \
#meta_arch.backbone.args.patch_size=10 \
#meta_arch.target='label' \
#meta_arch.num_classes=19 \
#meta_arch.weighted_loss=4 \
#data@train_data=bigearth \
#data@val_data_dict=[bigearth_val,bigearth_test] \
#train_data.bigearth.loader.num_workers=32 \
#train_data.bigearth.loader.batch_size=16 \
#train_data.bigearth.loader.drop_last=True \
#train_data.bigearth.args.channels=[0,1,2,3,4,5,6,7,8,9,10,11] \
#val_data_dict.bigearth_val.loader.num_workers=32 \
#val_data_dict.bigearth_val.loader.batch_size=16 \
#val_data_dict.bigearth_val.loader.drop_last=False \
#val_data_dict.bigearth_val.args.channels=[0,1,2,3,4,5,6,7,8,9,10,11] \
#val_data_dict.bigearth_test.loader.num_workers=32 \
#val_data_dict.bigearth_test.loader.batch_size=16 \
#val_data_dict.bigearth_test.loader.drop_last=False \
#val_data_dict.bigearth_test.args.channels=[0,1,2,3,4,5,6,7,8,9,10,11] \
#transformations@train_transformations=bigearth \
#transformations@val_transformations=bigearth


python amlssl/main/main_supervised_multiclass_evalall.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=samplev3channelvit_small \
meta_arch.backbone.args.in_chans=12 \
meta_arch.target='label' \
meta_arch.num_classes=19 \
transformation_mask=False \
channels=[0,1,2,3,4,5,6,7,8,9,10,11] \
data@val_data=bigearth_test \
val_data.bigearth_test.loader.num_workers=32 \
val_data.bigearth_test.loader.batch_size=32 \
val_data.bigearth_test.loader.drop_last=False \
val_data.bigearth_test.args.channels=[0,1,2,3,4,5,6,7,8,9,10,11] \
transformations=bigearth \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=22"