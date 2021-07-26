#!/bin/bash

# Remove CUDA artifacts
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit --assume-yes
sudo apt remove --autoremove nvidia-* --assume-yes
sudo apt purge libcudnn* --assume-yes
sudo apt-get purge nvidia* --assume-yes
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

# Install CUDA 11.0

sudo apt update
sudo add-apt-repository ppa:graphics-drivers --assume-yes
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

sudo apt update
sudo apt install cuda-11-0 --assume-yes

# Install cuDNN
gsutil -m cp gs://lucas.netdron.es/cudnn-11.2-linux-x64-v8.1.1.33.tgz .
tar -xzvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
rm -r cuda *.tgz

# Install NVIDIA CUDA Toolkit
sudo apt-get install nvidia-cuda-toolkit --assume-yes

# Reboot
sudo reboot now
