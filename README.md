# Nerf-W

## :computer: Installation

### Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=10.2** (tested with 1 RTX2080Ti, 2 Tesla P4, 2 Tesla T4, 2 Tesla V100)
* We recommend using 2 NVIDIA Tesla V100 GPUs for a typical job.

To install CUDA, use `sh +x bin/install_cuda110.sh`

### Software

* Clone this repo by `git@github.com:netdrones/nerf_pl`, and checkout the `nerfw` branch
* Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

* Install dependencies with `make install`

## :key: Training

### Canonical Datasets

* PixPole Historical House: `make train-house`
* Playground: `make train-playground`
* Brandenburg Gate: `make train-brandenburg`

### New Datasets

For images in `<my-dataset>`, run the following command:

```bash
sh +x bin/train.sh -i </path/to/my-dataset> -c
```

### :warning: Image Filtering

The script `bin/train.sh` currently supports three flags.

* The `-i` flag's argument should be the directory
containing the images (or nested image subfolders) from which the 3D model is to be reconstructed.
* The `-c` flag when invoked will perform automated dataset cleaning to filter out any excessively blurry images, images that
are unrelated to the main scene, or redundant images
* The `-o` flag is an extra layer of customization on top of the `-c` flag, which allows a "maximum overlap threshold" to be set.
Successive images that exceed this percentage of overlap will be thrown out. The default value is 0.98.

Note that if you're not using an Android Pixel phone to take the images, the `-c` flag is currently not supported. However, iPhone
support will shortly be available.

Also note that OpenCV uses CUDA for GPU acceleration, and we're using a custom-built version of the package to enable this. On architectures besides Ubuntu 18.04 with a Tesla V100 GPU, this is not tested. If the `pip` installation isn't working, you'll need to build OpenCV from source;  use `make build-opencv`.
