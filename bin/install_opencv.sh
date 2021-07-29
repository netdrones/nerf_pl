#!/bin/bash

sudo apt update
sudo apt install g++-5 --assume-yes
export CMAKE_ARGS='-DWITH_CUDA=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON -DCUDA_ARCH_BIN=7.0 -DWITH_QT=OFF -DWITH_OPENGL=OFF -DWITH_GSTREAMER=OFF -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include -DCUDNN_LIBRARY=/usr/local/cuda/lib64'
export ENABLE_CONTRIB=1

git clone --recursive git@github.com:netdrones/opencv-python ~/ws/git/opencv-python
cd ~/ws/git/opencv-python
mkdir build && cd build
CC=/usr/bin/gcc-5 CXX=/usr/bin/g++-5 ~/miniconda3/envs/nerf_pl/bin/pip wheel .. --verbose
~/miniconda3/envs/nerf_pl/bin/pip install *.whl
