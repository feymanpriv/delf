python train.py \
    --debug=False \
    --max_iters=1500000 \
    --initial_lr=0.01 \
    --Warmup_step=80000 \
    --train_file_pattern=data/train* \
    --validation_file_pattern=data/validation* \
    --dataset_version=gld_v2_clean \
    --batch_size=64 \
    --logdir=output/
