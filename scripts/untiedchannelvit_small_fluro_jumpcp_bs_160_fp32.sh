NICKNAME="untiedchannelvit_small_fluro_jumpcp_bs_160_fp32"

python amlssl/main/main_dino.py \
wandb.project="amlssl-channelvit" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=untiedchannelvit_small \
meta_arch.backbone.args.in_chans=5 \
data=jumpcp \
data.jumpcp.args.split=train \
data.jumpcp.args.channels=[0,1,2,3,4] \
data.jumpcp.loader.num_workers=32 \
data.jumpcp.loader.batch_size=20 \
data.jumpcp.loader.drop_last=True \
transformations=cell_dino


# MLP probing using channel 0, 1, 2, 3, 4
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-fluro-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using channel 0, 1, 2
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-0-1-2-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2] \
val_data_dict.jumpcp_val.args.channels=[0,1,2] \
val_data_dict.jumpcp_test.args.channels=[0,1,2] \
transformations=cell \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using channel 2, 3, 4
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-2-3-4-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[2,3,4] \
val_data_dict.jumpcp_val.args.channels=[2,3,4] \
val_data_dict.jumpcp_test.args.channels=[2,3,4] \
transformations=cell \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using channel 1, 2, 3, 4
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-1-2-3-4-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[1,2,3,4] \
transformations=cell \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

# MLP probing using channel 0, 2, 3, 4
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-0-2-3-4-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,2,3,4] \
transformations=cell \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using channel 0, 1, 3, 4
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-0-1-3-4-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,3,4] \
transformations=cell \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using channel 0, 1, 2, 4
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-0-1-2-4-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,4] \
transformations=cell \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"

# MLP probing using channel 0, 1, 2, 3
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-0-1-2-3-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3] \
transformations=cell \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
