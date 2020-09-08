#! /bin/bash
python build_image_dataset.py \
  --train_csv_path=gldv2_dataset/train/train.csv \
  --train_clean_csv_path=gldv2_dataset/train/train_clean.csv \
  --train_directory=gldv2_dataset/train/*/*/*/ \
  --output_directory=gldv2_dataset/tfrecord/ \
  --num_shards=128 \
  --generate_train_validation_splits \
  --validation_split_size=0.2
