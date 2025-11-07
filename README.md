<div align="center">
<h1>🐑 SHeaP 🐑</h1>
<h2>Self-Supervised Head Geometry Predictor Learned via 2D Gaussians</h2>

<a href="https://nlml.github.io/sheap" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page">
</a>
<a href="https://arxiv.org/abs/2504.12292"><img src="https://img.shields.io/badge/arXiv-2504.12292-b31b1b" alt="arXiv"></a>
<a href="https://www.youtube.com/watch?v=vhXsZJWCBMA"><img src="https://img.shields.io/badge/YouTube-Video-red" alt="YouTube"></a>

**Liam Schoneveld, Zhe Chen, Davide Davoli, Jiapeng Tang, Saimon Terazawa, Ko Nishino, Matthias Nießner**

<img src="teaser.jpg" alt="SHeaP Teaser" width="100%">

</div>

## Overview

SHeaP learns to predict head geometry (FLAME parameters) from a single image, by predicting and rendering 2D Gaussians.

This repository contains code and models for the **FLAME parameter inference only**.

## Example usage

After setting up, run `python demo.py`.

Or, here is a minimal example script:

```python
import torch, torchvision.io as io
from sheap import load_sheap_model
# Available model variants:
# sheap_model = load_sheap_model(model_type="paper")
sheap_model = load_sheap_model(model_type="expressive")
impath = "example_images/00000200.jpg"
# input: (N, 3, 224, 224) | pixel values in [0, 1]
image_tensor = io.decode_image(impath).float() / 255
# flame_params_dict contains predicted FLAME parameters
flame_params_dict = sheap_model(image_tensor[None])
```

**Note: `model_type`** can be one of 2 values:

- **`"paper"`**: used for paper results; gets best performance on NoW.
- **`"expressive"`**: perhaps better for real-world use; it was trained for longer with less regularisation and tends to be more expressive.

## Setup

### Step 1: Install dependencies

We just require `torch>=2.0.0` and a few other dependencies.

Just install the latest `torch` in a new venv, then `pip install .`

Or, if you use [`uv`](https://docs.astral.sh/uv/), you can just run `uv sync`.

### Step 2: Download and convert FLAME

Only needed if you want to predict FLAME vertices or render a mesh.

Download [FLAME2020](https://flame.is.tue.mpg.de/).

Put it in the `FLAME2020/` dir. We only need gerneric_model.pkl. Your `FLAME2020/` directory should look like this:

```bash
FLAME2020/
├── eyelids.pt
├── flame_landmark_idxs_barys.pt
└── generic_model.pkl
```

Now convert FLAME to our format:

```bash
python convert_flame.py
```

## Reproduce paper results on NoW dataset

To reproduce the validation results from the paper (median=0.93mm):

First, update submodules:

```bash
git submodule update --init --recursive
```

Then build the NoW Evaluation docker image:

```bash
docker build -t noweval now/now_evaluation
```

Then predict FLAME meshes for all images in NoW using SHeaP:

```
cd now/
python now.py --now-dataset-root /path/to/NoW_Evaluation/dataset
```

Upon finishing, the above command will print a command like the following:

```
chmod 777 -R /home/user/sheap/now/now_eval_outputs/now_preds && docker run --ipc host --gpus all -it --rm -v /data/NoW_Evaluation/dataset:/dataset -v /home/user/sheap/now/now_eval_outputs/now_preds:/preds noweval
```

Run that command. This will run NoW evaluation on the FLAME meshes we just predicted.

Finally, the results will be placed in `/home/user/sheap/now/now_eval_outputs/now_preds` (or equivalent). The mean and median are already calculated:

```bash
➜ cat /home/user/sheap/now/now_eval_outputs/now_preds/results/RECON_computed_distances.npy.meanmedian
0.9327719333872148  # result in the paper
1.1568168246248534
```
