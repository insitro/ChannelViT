NICKNAME="supervised_vit_base_p_8_als_bs_256_fp32"

python amlssl/main/main_supervised.py \
wandb.project="amlssl-supervised" \
nickname="${NICKNAME}" \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="s3://insitro-user/yujia/checkpoints/${NICKNAME}" \
meta_arch/backbone=vit_base \
meta_arch.target='cell_line' \
meta_arch.backbone.args.in_chans=4 \
meta_arch.backbone.args.input_drop=0 \
meta_arch.backbone.args.patch_size=8 \
meta_arch.num_classes=29 \
data@train_data=als \
data@val_data_dict=[als_val] \
train_data.als.loader.num_workers=64 \
train_data.als.loader.batch_size=32 \
train_data.als.loader.drop_last=True \
train_data.als.args.channels=[0,1,2,3] \
val_data_dict.als_val.loader.num_workers=64 \
val_data_dict.als_val.loader.batch_size=32 \
val_data_dict.als_val.loader.drop_last=False \
val_data_dict.als_val.args.channels=[0,1,2,3] \
transformations@train_transformations=cell_als \
transformations@val_transformations=cell_als

## in distribution analysis

#for DATA in "C9ORF72_Dins390" "C9ORF72_Dins532" "C9ORF72_Dins604" "C9ORF72_Dins605" \
#    "kiTARDBP-G295Shet_Dins022" "kiTARDBP-G295Shet_Dins023" "kiTARDBP-G295Shet_Dins032" \
#    "kiTARDBP-M337Vhet_Dins022" "kiTARDBP-M337Vhet_Dins023" "kiTARDBP-M337Vhet_Dins025" \
#    "kiTARDBP-M337Vhet_Dins032" "kiVCP_Dins022" "kiVCP_Dins023" "kiVCP_Dins032" "kiVCP_Dins033"
#do
#    python amlssl/main/main_supervised_prob.py \
#        wandb.project="amlssl-als-evaluation" \
#        task_name="in_distribution_${DATA}" \
#        nickname="${NICKNAME}-linear" \
#        data@train_data=als \
#        data@val_data_dict=[als_val] \
#        train_data.als.loader.num_workers=64 \
#        train_data.als.loader.batch_size=32 \
#        train_data.als.loader.drop_last=True \
#        train_data.als.args.path=["s3://insitro-user/yujia/ALS/tmp/${DATA}_test.pq"] \
#        val_data_dict.als_val.args.path=["s3://insitro-user/yujia/ALS/tmp/${DATA}_test.pq"] \
#        val_data_dict.als_val.args.split="val" \
#        transformations=cell_als \
#        meta_arch.target='disease_state' \
#        meta_arch.num_classes=2 \
#        meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99" \
#        meta_arch.use_mlp=False \
#        meta_arch.n_last_blocks=1 \
#        trainer.devices=8 \
#        trainer.max_epochs=30

#    python amlssl/main/main_supervised_prob.py \
#        wandb.project="amlssl-als-evaluation" \
#        task_name="in_distribution_${DATA}" \
#        nickname="${NICKNAME}-4-mlp" \
#        data@train_data=als \
#        data@val_data_dict=[als_val] \
#        train_data.als.loader.num_workers=64 \
#        train_data.als.loader.batch_size=32 \
#        train_data.als.loader.drop_last=True \
#        train_data.als.args.path=["s3://insitro-user/yujia/ALS/tmp/${DATA}_test.pq"] \
#        val_data_dict.als_val.args.path=["s3://insitro-user/yujia/ALS/tmp/${DATA}_test.pq"] \
#        val_data_dict.als_val.args.split="val" \
#        transformations=cell_als \
#        meta_arch.target='disease_state' \
#        meta_arch.num_classes=2 \
#        meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99" \
#        meta_arch.use_mlp=True \
#        meta_arch.n_last_blocks=4 \
#        trainer.devices=8 \
#        trainer.max_epochs=30


#    python amlssl/main/main_supervised_prob.py \
#        wandb.project="amlssl-als-evaluation" \
#        task_name="in_distribution_${DATA}" \
#        nickname="${NICKNAME}-4-linear" \
#        data@train_data=als \
#        data@val_data_dict=[als_val] \
#        train_data.als.loader.num_workers=64 \
#        train_data.als.loader.batch_size=32 \
#        train_data.als.loader.drop_last=True \
#        train_data.als.args.path=["s3://insitro-user/yujia/ALS/tmp/${DATA}_test.pq"] \
#        val_data_dict.als_val.args.path=["s3://insitro-user/yujia/ALS/tmp/${DATA}_test.pq"] \
#        val_data_dict.als_val.args.split="val" \
#        transformations=cell_als \
#        meta_arch.target='disease_state' \
#        meta_arch.num_classes=2 \
#        meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99" \
#        meta_arch.use_mlp=False \
#        meta_arch.n_last_blocks=4 \
#        trainer.devices=8 \
#        trainer.max_epochs=30

#    python amlssl/main/main_supervised_prob.py \
#        wandb.project="amlssl-als-evaluation" \
#        task_name="in_distribution_${DATA}" \
#        nickname="${NICKNAME}-1-mlp" \
#        data@train_data=als \
#        data@val_data_dict=[als_val] \
#        train_data.als.loader.num_workers=64 \
#        train_data.als.loader.batch_size=32 \
#        train_data.als.loader.drop_last=True \
#        train_data.als.args.path=["s3://insitro-user/yujia/ALS/tmp/${DATA}_test.pq"] \
#        val_data_dict.als_val.args.path=["s3://insitro-user/yujia/ALS/tmp/${DATA}_test.pq"] \
#        val_data_dict.als_val.args.split="val" \
#        transformations=cell_als \
#        meta_arch.target='disease_state' \
#        meta_arch.num_classes=2 \
#        meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99" \
#        meta_arch.use_mlp=True \
#        meta_arch.n_last_blocks=1 \
#        trainer.devices=8 \
#        trainer.max_epochs=30
#done


#for n_blocks in 1 4
for n_blocks in 4
do
    #for mlp in True False
    for mlp in True
    do
        #for covariate in null donor_id
        for covariate in donor_id
        do
            # C9ORF72
            python amlssl/main/main_supervised_prob.py \
                wandb.project="amlssl-als-evaluation" \
                task_name="SingleClassifier-C9ORF72" \
                nickname="${NICKNAME}-blocks-${n_blocks}-mlp-${mlp}-covariate-${covariate}" \
                data@train_data=als \
                data@val_data_dict=[als_val_0,als_val_1,als_val_2,als_val_3] \
                train_data.als.loader.num_workers=64 \
                train_data.als.loader.batch_size=32 \
                train_data.als.loader.drop_last=True \
                train_data.als.args.path=["s3://insitro-user/yujia/ALS/tmp/C9ORF72_train.pq"] \
                train_data.als.args.split='all' \
                val_data_dict.als_val_0.args.path=["s3://insitro-user/yujia/ALS/tmp/C9ORF72_Dins390_test.pq"] \
                val_data_dict.als_val_1.args.path=["s3://insitro-user/yujia/ALS/tmp/C9ORF72_Dins532_test.pq"] \
                val_data_dict.als_val_2.args.path=["s3://insitro-user/yujia/ALS/tmp/C9ORF72_Dins604_test.pq"] \
                val_data_dict.als_val_3.args.path=["s3://insitro-user/yujia/ALS/tmp/C9ORF72_Dins605_test.pq"] \
                transformations=cell_als \
                meta_arch.target='disease_state' \
                meta_arch.covariate.name=${covariate} \
                meta_arch.num_classes=2 \
                meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99" \
                meta_arch.use_mlp=${mlp} \
                meta_arch.n_last_blocks=${n_blocks} \
                trainer.devices=8 \
                trainer.max_epochs=30

            # kiTARDBP-G295Shet
            python amlssl/main/main_supervised_prob.py \
                wandb.project="amlssl-als-evaluation" \
                task_name="SingleClassifier-kiTARDBP-G295Shet" \
                nickname="${NICKNAME}-blocks-${n_blocks}-mlp-${mlp}-covariate-${covariate}" \
                data@train_data=als \
                data@val_data_dict=[als_val_0,als_val_1,als_val_2] \
                train_data.als.loader.num_workers=64 \
                train_data.als.loader.batch_size=32 \
                train_data.als.loader.drop_last=True \
                train_data.als.args.path=["s3://insitro-user/yujia/ALS/tmp/kiTARDBP-G295Shet_train.pq"] \
                train_data.als.args.split='all' \
                val_data_dict.als_val_0.args.path=["s3://insitro-user/yujia/ALS/tmp/kiTARDBP-G295Shet_Dins022_test.pq"] \
                val_data_dict.als_val_1.args.path=["s3://insitro-user/yujia/ALS/tmp/kiTARDBP-G295Shet_Dins023_test.pq"] \
                val_data_dict.als_val_2.args.path=["s3://insitro-user/yujia/ALS/tmp/kiTARDBP-G295Shet_Dins032_test.pq"] \
                transformations=cell_als \
                meta_arch.target='disease_state' \
                meta_arch.num_classes=2 \
                meta_arch.covariate.name=${covariate} \
                meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99" \
                meta_arch.use_mlp=${mlp} \
                meta_arch.n_last_blocks=${n_blocks} \
                trainer.devices=8 \
                trainer.max_epochs=30

            # kiTARDBP-M337Vhet
            python amlssl/main/main_supervised_prob.py \
                wandb.project="amlssl-als-evaluation" \
                task_name="SingleClassifier-kiTARDBP-M337Vhet" \
                nickname="${NICKNAME}-blocks-${n_blocks}-mlp-${mlp}-covariate-${covariate}" \
                data@train_data=als \
                data@val_data_dict=[als_val_0,als_val_1,als_val_2,als_val_3] \
                train_data.als.loader.num_workers=64 \
                train_data.als.loader.batch_size=32 \
                train_data.als.loader.drop_last=True \
                train_data.als.args.path=["s3://insitro-user/yujia/ALS/tmp/kiTARDBP-M337Vhet_train.pq"] \
                train_data.als.args.split='all' \
                val_data_dict.als_val_0.args.path=["s3://insitro-user/yujia/ALS/tmp/kiTARDBP-M337Vhet_Dins022_test.pq"] \
                val_data_dict.als_val_1.args.path=["s3://insitro-user/yujia/ALS/tmp/kiTARDBP-M337Vhet_Dins023_test.pq"] \
                val_data_dict.als_val_2.args.path=["s3://insitro-user/yujia/ALS/tmp/kiTARDBP-M337Vhet_Dins025_test.pq"] \
                val_data_dict.als_val_3.args.path=["s3://insitro-user/yujia/ALS/tmp/kiTARDBP-M337Vhet_Dins032_test.pq"] \
                transformations=cell_als \
                meta_arch.target='disease_state' \
                meta_arch.num_classes=2 \
                meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99" \
                meta_arch.use_mlp=${mlp} \
                meta_arch.n_last_blocks=${n_blocks} \
                meta_arch.covariate.name=${covariate} \
                trainer.devices=8 \
                trainer.max_epochs=30

            python amlssl/main/main_supervised_prob.py \
                wandb.project="amlssl-als-evaluation" \
                task_name="SingleClassifier-kiVCP" \
                nickname="${NICKNAME}-blocks-${n_blocks}-mlp-${mlp}-covariate-${covariate}" \
                data@train_data=als \
                data@val_data_dict=[als_val_0,als_val_1,als_val_2,als_val_3] \
                train_data.als.loader.num_workers=64 \
                train_data.als.loader.batch_size=32 \
                train_data.als.loader.drop_last=True \
                train_data.als.args.path=["s3://insitro-user/yujia/ALS/tmp/kiVCP_train.pq"] \
                train_data.als.args.split='all' \
                val_data_dict.als_val_0.args.path=["s3://insitro-user/yujia/ALS/tmp/kiVCP_Dins022_test.pq"] \
                val_data_dict.als_val_1.args.path=["s3://insitro-user/yujia/ALS/tmp/kiVCP_Dins023_test.pq"] \
                val_data_dict.als_val_2.args.path=["s3://insitro-user/yujia/ALS/tmp/kiVCP_Dins032_test.pq"] \
                val_data_dict.als_val_3.args.path=["s3://insitro-user/yujia/ALS/tmp/kiVCP_Dins033_test.pq"] \
                transformations=cell_als \
                meta_arch.target='disease_state' \
                meta_arch.num_classes=2 \
                meta_arch.checkpoint="s3://insitro-user/yujia/checkpoints/${NICKNAME}/epoch\\=99" \
                meta_arch.use_mlp=${mlp} \
                meta_arch.n_last_blocks=${n_blocks} \
                meta_arch.covariate.name=${covariate} \
                trainer.devices=8 \
                trainer.max_epochs=30
        done

    done
done
