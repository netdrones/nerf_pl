#!/bin/bash

WORKSPACE_PATH=$1

colmap feature_extractor --database_path $WORKSPACE_PATH/database.db \
  --image_path $WORKSPACE_PATH/images \

colmap exhaustive_matcher --database_path $WORKSPACE_PATH/database.db \

mkdir -p $WORKSPACE_PATH/sparse

colmap mapper \
  --database_path $WORKSPACE_PATH/database.db \
  --image_path $WORKSPACE_PATH/images \
  --output_path $WORKSPACE_PATH/sparse

mkdir -p $WORKSPACE_PATH/dense

colmap image_undistorter \
  --image_path $WORKSPACE_PATH/images \
  --input_path $WORKSPACE_PATH/sparse/0 \
  --output_path $WORKSPACE_PATH/dense \
  --output_type COLMAP \
  --max_image_size 2000

colmap patch_match_stereo \
  --workspace_path $WORKSPACE_PATH/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true
