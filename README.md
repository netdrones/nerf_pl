# Nerf-W

Unofficial implementation of [NeRF-W](https://nerf-w.github.io/) (NeRF in the wild) using pytorch ([pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)). I try to reproduce (some of) the results on the lego dataset (Section D). Training on [Phototourism real images](https://github.com/ubc-vision/image-matching-benchmark) (as the main content of the paper) has also passed. Please read the following sections for the results.

The code is largely based on NeRF implementation (see master or dev branch), the main difference is the model structure and the rendering process, which can be found in the two files under `models/`.

## :computer: Installation

### Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=10.2** (tested with 1 RTX2080Ti, 2 Tesla P4, 2 Tesla T4, 2 Tesla V100)

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
sh +x bin/train.sh </path/to/my-dataset>
```

In the case that the images in `<my-dataset>` were taken with the PixPole Android camera setup,
these will be automatically filtered. Currently other camera software architectures are not supported (but likely
iPhone will be added soon!).
