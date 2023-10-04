NICKNAME="vit_small_fluro_jumpcp_bs_160_fp32"

python amlssl/main/main_dino.py \
wandb.project="amlssl-channelvit" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=vit_small \
meta_arch.backbone.args.in_chans=5 \
data=jumpcp \
data.jumpcp.args.upsample=1 \
data.jumpcp.args.split=train \
data.jumpcp.args.channels=[0,1,2,3,4] \
data.jumpcp.loader.num_workers=64 \
data.jumpcp.loader.batch_size=20 \
data.jumpcp.loader.drop_last=True \
transformations=cell_dino


# MLP probing using the last 4 blocks
# Train and eval only on fluro channels
# Note that the model only receive inputs with 5 channels
# That's why in the train_data and val_data_dict, we only have 5 channels.
# For the channel_mask in the trasnforamtions, we don't do anything here.
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
transformations.args.channel_mask=[] \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using the last 4 blocks
# Train and eval only on the first channel.
# Note that the model still receive inputs with 5 channels
# That's why in the train_data and val_data_dict, we still have 5 channels.
# For the channel_mask in the trasnforamtions, we mask out the other four channels.
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-0-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[1,2,3,4] \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using channel 1
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-0-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[0,2,3,4] \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using channel 2
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-0-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[0,1,3,4] \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using channel 3
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-0-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[0,1,2,4] \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"


# MLP probing using channel 4
python amlssl/main/main_linear_prob.py \
wandb.project="amlssl-linear-prob" \
nickname="${NICKNAME}-ch-0-mlp-4-last-blocks" \
data@train_data=jumpcp \
data@val_data_dict=[jumpcp_val,jumpcp_test] \
train_data.jumpcp.loader.batch_size=32 \
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[0,1,2,3] \
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
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[0] \
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
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[1] \
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
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[2] \
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
train_data.jumpcp.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[3] \
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
val_data_dict.jumpcp_val.args.channels=[0,1,2,3,4] \
val_data_dict.jumpcp_test.args.channels=[0,1,2,3,4] \
transformations=cell \
transformations.args.channel_mask=[4] \
trainer.devices=8 \
trainer.max_epochs=100 \
meta_arch=mlp_prob \
meta_arch.target="label" \
meta_arch.num_classes=161 \
meta_arch.n_last_blocks=4 \
meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99"
