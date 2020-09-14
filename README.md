# Install DELF(DELG) 
```
git clone https://github.com/tensorflow/models.git

cd models/research/delf/ && rm -rf ./delf
git clone https://github.com/feymanpriv/delf.git

cd models/research/delf/delf/python/training
bash install_delf.sh
```

# Download Dataset(gldv2-clean)
```
http://yq01-sys-hic-k8s-v100-box-a223-0091.yq01.baidu.com:8903/models/research/delf/delf/python/training/gldv2_dataset/tfrecord/
```

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

#Visualize
```
tensorboard --logdir ./output/train_logs/ --port 8198 --host yq01-sys-hic-k8s-v100-box-a223-0091.yq01.baidu.com
```
