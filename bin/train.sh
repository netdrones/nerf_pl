#!/bin/bash

WORKSPACE_DIR=$1
EXP_NAME=$2
DOWNSCALE=$3

NUM_GPUS="$(nvidia-smi --query_gpu=name --format=csv,nohearder | wc -l)"

if [ ! -d "$WORKSPACE_DIR/dense" ]; then sh +x bin/run_colmap.sh $WORKSPACE_DIR; fi

python generate_splits.py $WORKSPACE_DIR/dense/images $WORKSPACE_DIR/$WORKSPACE_DIR.tsv $WORKSPACE_DIR $WORKSPACE_DIR/database.db
if [ ! -d "$WORKSPACE_DIR/cache" ]; then \
  python prepare_phototourism.py --root_dir $WORKSPACE_DIR --img_downscale $DOWNSCALE; \
fi

python train.py \
    --root_dir $WORKSPACE_DIR --dataset_name phototourism \
    --img_downscale $DOWNSCALE --use_cache --N_importance 64 --N_samples 64 \
    --encode_a --encode_t --beta_min 0.03 --N_vocab 1500 \
    --num_epochs 20 --num_gpus $NUM_GPUS --batch_size 1024 \
    --optimizer adam --lr 5e-4 --lr_scheduler cosine \
    --exp_name $EXP_NAME
