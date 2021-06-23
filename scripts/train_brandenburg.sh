#!/bin/bash

BRANDENBURG_DIR=./brandenburg_gate

python train.py \
    --root_dir $BRANDENBURG_DIR --dataset_name phototourism \
    --img_downscale 2 --use_cache --N_importance 64 --N_samples 64 \
    --encode_a --encode_t --beta_min 0.03 --N_vocab 1500 \
    --num_epochs 20 --batch_size 1024 \
    --optimizer adam --lr 5e-4 --lr_scheduler cosine \
    --exp_name brandenburg_scale8_nerfw
