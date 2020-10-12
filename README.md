# Install DELF(DELG) 
```
git clone https://github.com/feymanpriv/delf.git

cd training
bash install.sh
```

# Prepare Dataset(gldv2-clean)
refer [`training/DELG.md`]

# Training
```
python train.py \
    --train_file_pattern=gldv2_dataset/tfrecord/train* \
    --validation_file_pattern=gldv2_dataset/tfrecord/validation* \
    --imagenet_checkpoint=/path/to/pretrained_model \
    --dataset_version=gld_v2_clean \
    --batch_size=256 \
    --logdir=output/
    
```

# Visualize
```
tensorboard --logdir ./output/train_logs/ --port 8198 --host /your/ip/address
```
