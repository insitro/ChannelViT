NICKNAME="supervised_v3vit_small_posh_bs_256_fp32.sh"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=v3vit_small \
meta_arch.target='gene_id' \
meta_arch.backbone.args.in_chans=5 \
meta_arch.backbone.args.input_drop=0 \
meta_arch.num_classes=302 \
data@train_data=cp_posh_300 \
data@val_data_dict=[cp_posh_300_val,cp_posh_300_test] \
train_data.cp_posh_300.loader.num_workers=64 \
train_data.cp_posh_300.loader.batch_size=32 \
train_data.cp_posh_300.loader.drop_last=True \
train_data.cp_posh_300.args.channels=[0,1,2,3,4] \
val_data_dict.cp_posh_300_val.loader.num_workers=64 \
val_data_dict.cp_posh_300_val.loader.batch_size=32 \
val_data_dict.cp_posh_300_val.loader.drop_last=False \
val_data_dict.cp_posh_300_val.args.channels=[0,1,2,3,4] \
val_data_dict.cp_posh_300_test.loader.num_workers=64 \
val_data_dict.cp_posh_300_test.loader.batch_size=32 \
val_data_dict.cp_posh_300_test.loader.drop_last=False \
val_data_dict.cp_posh_300_test.args.channels=[0,1,2,3,4] \
transformations@train_transformations=cell_posh \
transformations@val_transformations=cell_posh

