NICKNAME="supervised_samplev5_0.3_channelvit_small_all_jumpcp_bs_256_fp32_correct"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
trainer.accumulate_grad_batches=2 \
meta_arch/backbone=samplev5channelvit_small \
meta_arch.backbone.args.in_chans=8 \
meta_arch.backbone.args.input_drop=0.3 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.num_workers=32 \
train_data.jumpcp.loader.batch_size=16 \
train_data.jumpcp.loader.drop_last=True \
train_data.jumpcp.args.channels=[0,1,2,3,4,5,6,7] \
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


python amlssl/main/main_supervised_evalall.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
trainer.accumulate_grad_batches=2 \
meta_arch/backbone=samplev3channelvit_small \
meta_arch.backbone.args.in_chans=8 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@val_data=jumpcp_test \
val_data.jumpcp_test.loader.num_workers=32 \
val_data.jumpcp_test.loader.batch_size=32 \
val_data.jumpcp_test.loader.drop_last=False \
val_data.jumpcp_test.args.channels=[0,1,2,3,4,5,6,7] \
transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


#python amlssl/main/main_supervised.py \
#wandb.project="amlssl-supervised" \
#nickname="${NICKNAME}-evaluation-fluro" \
#trainer.devices=8 \
#trainer.precision=32 \
#trainer.max_epochs=100 \
#trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
#trainer.accumulate_grad_batches=2 \
#meta_arch/backbone=samplev3channelvit_small \
#meta_arch.backbone.args.in_chans=8 \
#meta_arch.target='label' \
#meta_arch.num_classes=161 \
#data@train_data=jumpcp \
#data@val_data_dict=[jumpcp_test] \
#train_data.jumpcp.loader.num_workers=32 \
#train_data.jumpcp.loader.batch_size=16 \
#train_data.jumpcp.loader.drop_last=True \
#train_data.jumpcp.args.channels=[0,1,2,3,4,5,6,7] \
#val_data_dict.jumpcp_test.loader.num_workers=32 \
#val_data_dict.jumpcp_test.loader.batch_size=32 \
#val_data_dict.jumpcp_test.loader.drop_last=False \
#val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
#transformations@train_transformations=cell \
#transformations@val_transformations=cell \
#checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
