# 🐑 SHeaP 🐑
Code and models for inferencing [SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians](https://nlml.github.io/sheap).

## Example usage

After setting up, run `python demo.py`.

Here is a minimal example script:

```python
import torch, torchvision.io as io
from sheap import load_sheap_model
# Available model variants:
# sheap_model = load_sheap_model(model_type="paper")
sheap_model = load_sheap_model(model_type="expressive")
impath = "example_images/00000200.png"
image_tensor = io.decode_image(impath).float() / 255
flame_params_dict = sheap_model(image_tensor[None])
```

**Note: `model_type`** can be one of 2 values:

- **`"paper"`**: used for paper results; gets best performance on NoW.
- **`"expressive"`**: perhaps better for real-world use; less regularisation, more expressive.

## Setup

### Step 1: Install dependencies

We just require `torch>=2.0.0` and a few other dependencies.

Just install the latest `torch` in a new venv, then `pip install .`

Or, if you use [`uv`](https://docs.astral.sh/uv/), you can just run `uv sync`.

### Step 2: Download and convert FLAME

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

### Done!

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
