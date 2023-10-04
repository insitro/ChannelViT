NICKNAME="supervised_v3vit_small_all_crispr_bs_256_fp32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
trainer.accumulate_grad_batches=1 \
meta_arch/backbone=v3vit_small \
meta_arch.backbone.args.in_chans=8 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
data@train_data=crispr \
data@val_data_dict=[crispr_val,crispr_test] \
train_data.crispr.loader.num_workers=32 \
train_data.crispr.loader.batch_size=32 \
train_data.crispr.loader.drop_last=True \
train_data.crispr.args.channels=[0,1,2,3,4,5,6,7] \
val_data_dict.crispr_val.loader.num_workers=32 \
val_data_dict.crispr_val.loader.batch_size=32 \
val_data_dict.crispr_val.loader.drop_last=False \
val_data_dict.crispr_val.args.channels=[0,1,2,3,4,5,6,7] \
val_data_dict.crispr_test.loader.num_workers=32 \
val_data_dict.crispr_test.loader.batch_size=32 \
val_data_dict.crispr_test.loader.drop_last=False \
val_data_dict.crispr_test.args.channels=[0,1,2,3,4,5,6,7] \
transformations@train_transformations=cell \
transformations@val_transformations=cell

python amlssl/main/main_supervised_evalall.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v3vit_small \
meta_arch.backbone.args.in_chans=8 \
meta_arch.target='label' \
meta_arch.num_classes=161 \
transformation_mask=True \
data@val_data=crispr_test \
val_data.crispr_test.loader.num_workers=32 \
val_data.crispr_test.loader.batch_size=32 \
val_data.crispr_test.loader.drop_last=False \
val_data.crispr_test.args.channels=[0,1,2,3,4,5,6,7] \
val_data.crispr_test.args.scale=1 \
transformations=cell \
checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
