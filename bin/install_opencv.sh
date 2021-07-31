#!/bin/bash

# Dependencies
sudo apt update
sudo apt install g++-6 --assume-yes

# NOTE: old pip versions will cause the build to segfault
pip install --upgrade pip

# Build flags
export CMAKE_ARGS='-DWITH_CUDA=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF -DCUDA_ARCH_BIN=7.0 -DWITH_QT=OFF -DWITH_OPENGL=OFF -DWITH_GSTREAMER=OFF -DOPENCV_ENABLE_NONFREE=ON'
export ENABLE_CONTRIB=1

# Build opencv-python
git clone --recursive git@github.com:netdrones/opencv-python ~/ws/git/opencv-python
cd ~/ws/git/opencv-python
mkdir build && cd build
CC=/usr/bin/gcc-6 CXX=/usr/bin/g++-6 pip wheel .. --verbose
