# Nerf-W

Unofficial implementation of [NeRF-W](https://nerf-w.github.io/) (NeRF in the wild) using pytorch ([pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)). I try to reproduce (some of) the results on the lego dataset (Section D). Training on [Phototourism real images](https://github.com/ubc-vision/image-matching-benchmark) (as the main content of the paper) has also passed. Please read the following sections for the results.

The code is largely based on NeRF implementation (see master or dev branch), the main difference is the model structure and the rendering process, which can be found in the two files under `models/`.

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

#### :warning: IMPORTANT: Automatic Dataset Cleaning

In the case that the images in `<my-dataset>` were taken with the PixPole Android camera setup,
these will be automatically filtered. Currently other camera software architectures are not supported (but likely
iPhone will be added soon!).

For smaller datasets, this can be automatically handled by the training script. But for larger datasets, this might become
computationally far too inefficient. In this case, we recommend separating the data cleaning into its own separate process and caching to GCP in the following manner.

First, create a GCP instance with a high amount of CPUs and RAM, but no GPU. The dataset cleaning doesn't run on the GPU but can take a long time, so we don't want to get billed for the GPUs while we're not using them for hours on end.

To run the Python script on its own, use the following command:

```bash
python image_utils.py <input-dir> <output-dir>
```

This will filter images in `<input-dir>`, and place the cleaned results into `output-dir`. Then, upload the cleaned images to GCP, and on a machine with a GPU, run the training script again, but this time, don't use the `-c` flag!
