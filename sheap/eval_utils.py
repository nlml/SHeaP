from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud
import trimesh
from PIL import Image
from torchvision.transforms import Normalize
from tqdm import tqdm


def _preproc_im_default(p):
    return Image.open(p)


class ImsDataset(tud.Dataset):
    def __init__(self, image_paths, img_wh, load_and_preproc_im=_preproc_im_default):
        self.image_paths = image_paths
        self.img_wh = img_wh
        self.load_and_preproc_im = load_and_preproc_im
        if self.load_and_preproc_im is None:
            self.load_and_preproc_im = _preproc_im_default

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        impath = self.image_paths[idx]
        pil_im = self.load_and_preproc_im(impath)
        im = pil_im.convert("RGB").resize(self.img_wh)
        im = np.array(im).astype("float64") / 255.0
        im = torch.from_numpy(im).permute(2, 0, 1).float()
        return im


@torch.no_grad()
def inference_images_list(
    model,
    device,
    image_paths,
    custom_pil_im_load_fn=None,
    img_wh=(224, 224),
    batch_size=4,
    num_workers=4,
    verbose=False,
):
    model = model.to(device)
    ds = ImsDataset(image_paths, img_wh=img_wh, load_and_preproc_im=custom_pil_im_load_fn)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    all_outs = {}
    for images in tqdm(dl, desc="Inferencing images through ViT model"):
        images = images.to(device)
        batch_size = images.shape[0]
        model_outs = model(images)
        for k in model_outs:
            if not isinstance(model_outs[k], torch.Tensor):
                continue
            if k not in all_outs:
                all_outs[k] = []
            all_outs[k].append(model_outs[k].detach().cpu())

    if verbose:
        print("Concatenated output shapes:")
    for k in all_outs:
        all_outs[k] = torch.cat(all_outs[k], dim=0)
        if verbose:
            print(" --", k, all_outs[k].shape)
    return all_outs


def invert_4x4_cam_matrix(inp_cam):
    rr = inp_cam[:3, :3].T
    tt = rr @ -inp_cam[:3, 3]
    inv_cam = torch.eye(4, device=inp_cam.device, dtype=inp_cam.dtype)
    inv_cam[:3, :3] = rr
    inv_cam[:3, 3] = tt
    return inv_cam


def save_obj(outpath, verts, faces):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(outpath)


def save_result(
    flame_faces,
    base_dir,
    verts_with_zero_exprn,
    lmks7_3d,
    preds_outdir,
    input_im_path,
    verbose=False,
):
    # MICA scaled up by 1000, so let's try it too:
    pred_verts = verts_with_zero_exprn * 1000.0
    pred_lmks7_3d = lmks7_3d.numpy() * 1000.0

    outpath_jpg = preds_outdir / Path(input_im_path).relative_to(base_dir)
    outpath_obj = outpath_jpg.with_suffix(".obj")

    outpath_obj.parent.mkdir(parents=True, exist_ok=True)

    save_obj(outpath_obj, verts=pred_verts, faces=flame_faces)
    if verbose:
        print(f"Saved {outpath_obj}")

    outpath_lmk_npy = outpath_obj.with_suffix(".npy")
    np.save(outpath_lmk_npy, pred_lmks7_3d)
    if verbose:
        print(f"Saved {outpath_lmk_npy}")

    assert outpath_obj.exists()
    assert outpath_lmk_npy.exists()


def add_pct_to_bbox(top, left, bottom, right, im_np_array, pct=0.2):
    if isinstance(im_np_array, Image.Image):
        im_np_array = np.array(im_np_array)
    h, w, _ = im_np_array.shape

    box_height = bottom - top
    top = max(0, top - int(box_height * pct * 0.5))
    bottom = top + int(box_height * (1 + pct))
    bottom = min(h, bottom)

    box_width = right - left
    left = max(0, left - int(box_width * pct * 0.5))
    right = left + int(box_width * (1 + pct))
    right = min(w, right)

    return top, left, bottom, right


def resize_to_max_size(im, max_size=512, pad_smaller=True):
    was_np = False
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
        was_np = True
    w, h = im.size
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    im = im.resize((new_w, new_h))
    if pad_smaller:
        new_im = Image.new("RGB", (max_size, max_size))
        new_im.paste(im, ((max_size - new_w) // 2, (max_size - new_h) // 2))
        im = new_im
    if was_np:
        return np.array(im)
    return im
