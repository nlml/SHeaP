import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sheap import load_sheap_model
from sheap.eval_utils import (
    add_pct_to_bbox,
    inference_images_list,
    resize_to_max_size,
    save_result,
)
from sheap.landmark_utils import vertices_to_7_lmks
from sheap.tiny_flame import TinyFlame


def path_to_detected_face_path(p, now_dataset_root):
    p = Path(p)
    now_dataset_root = Path(now_dataset_root)
    r = now_dataset_root / "NoW_Dataset/final_release_version/iphone_pictures"
    b = now_dataset_root / "NoW_Dataset/final_release_version/detected_face"
    a = p.relative_to(r)
    return (b / a).with_suffix(".npy")


def path_to_detected_face(p, now_dataset_root):
    det = np.load(
        path_to_detected_face_path(p, now_dataset_root),
        allow_pickle=True,
        encoding="latin1",
    ).item()
    top = det["top"]
    left = det["left"]
    bottom = det["bottom"]
    right = det["right"]
    return [int(i) for i in [top, left, bottom, right]]


def preproc_im(p, now_dataset_root, zoom_out=0.8):
    im = Image.open(p)
    top, left, bottom, right = path_to_detected_face(p, now_dataset_root)
    top, left, bottom, right = add_pct_to_bbox(top, left, bottom, right, im, pct=zoom_out)
    im = im.crop((left, top, right, bottom))
    im = resize_to_max_size(im, max_size=512)
    return im


@torch.no_grad()
def get_zero_exp_flame_and_7_lmks(
    flame, pred_shape, face_alignment_lmk_faces_idx, face_alignment_lmk_bary_coords
):
    batch_size = pred_shape.shape[0]
    device = pred_shape.device
    neutral_vertices = flame(
        shape=pred_shape,
        expression=torch.zeros([batch_size, 100], device=device),
        pose=torch.eye(3, device=device)[None][None].repeat(batch_size, 5, 1, 1),
        translation=torch.zeros([batch_size, 3], device=device),
        eyelids=torch.zeros([batch_size, 2], device=device),
    )
    lmks7_3d, _ = vertices_to_7_lmks(
        neutral_vertices,
        flame.faces,
        face_alignment_lmk_faces_idx,
        face_alignment_lmk_bary_coords,
    )
    return neutral_vertices, lmks7_3d


def main(
    now_dataset_root,
    outpath,
    sheap_model_path,
    zoom_out,
    use_test_set=False,
    check=False,
    flame_dir="../FLAME2020",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: running on CPU, this will be slow!")
    face_alignment_lmk_faces_idx, face_alignment_lmk_bary_coords = torch.load(
        "../FLAME2020/flame_landmark_idxs_barys.pt"
    )
    flame_dir = Path(flame_dir)
    flame = TinyFlame(flame_dir / "generic_model.pt", eyelids_ckpt=flame_dir / "eyelids.pt")
    now_dataset_root = Path(now_dataset_root)
    base_dir = now_dataset_root / "NoW_Dataset" / "final_release_version" / "iphone_pictures"
    txtfile = "imagepathstest.txt" if use_test_set else "imagepathsvalidation.txt"
    with open(now_dataset_root / txtfile) as f:
        validation_image_paths = f.readlines()
    all_ims = [base_dir / Path(p.strip()) for p in validation_image_paths]

    model = load_sheap_model(model_type="paper", models_dir=sheap_model_path).to(device)

    preds_outdir = Path(outpath) / "now_preds"
    if preds_outdir.exists():
        raise ValueError(f"{preds_outdir} already exists")
    else:
        preds_outdir.mkdir(parents=True)

    preproc_im_fn = partial(preproc_im, now_dataset_root=now_dataset_root, zoom_out=zoom_out)
    outs = inference_images_list(model, device, all_ims, preproc_im_fn)
    all_verts, all_lmks7_3d = get_zero_exp_flame_and_7_lmks(
        flame,
        outs["shape_from_facenet"],
        face_alignment_lmk_faces_idx,
        face_alignment_lmk_bary_coords,
    )
    all_verts = all_verts.cpu()
    all_lmks7_3d = all_lmks7_3d.cpu()

    print("Saving results...")
    for i in tqdm(range(all_verts.shape[0])):
        save_result(
            flame.faces.cpu(),
            base_dir,
            all_verts[i],
            all_lmks7_3d[i],
            preds_outdir,
            all_ims[i],
        )
    print("Saved to:")
    print(preds_outdir)
    print("Now run:")
    cmd = (
        f"chmod 777 -R {Path(preds_outdir).absolute()} && docker run --ipc host --gpus all -it "
        f"--rm -v {now_dataset_root}:/dataset -v {Path(preds_outdir).absolute()}:/preds noweval"
    )
    print(cmd)
    if check:
        for im in all_ims:
            outpath_jpg = preds_outdir / Path(im).relative_to(base_dir)
            outpath_obj = outpath_jpg.with_suffix(".obj")
            outpath_npy = outpath_jpg.with_suffix(".npy")
            # Check obj is larger than 20mb
            assert outpath_obj.exists(), outpath_obj
            assert outpath_obj.stat().st_size < 20 * 1024 * 1024, outpath_obj
            assert outpath_npy.exists(), outpath_npy
            assert np.load(outpath_npy).shape == (7, 3), outpath_npy
        print("Check passed")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sheap-model-path",
        type=str,
        default="../models",
    )
    parser.add_argument(
        "--outpath",
        type=str,
        default="./now_eval_outputs",
    )
    parser.add_argument(
        "--now-dataset-root",
        type=str,
        default="/data22tb0/NoW_Evaluation/dataset",
    )
    parser.add_argument("--zoom_out", type=float, default=0.8)
    parser.add_argument("--mul_shape_by_zero", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.test:
        args.check = True
    return args


if __name__ == "__main__":
    args = get_args()
    main(
        args.now_dataset_root,
        args.outpath,
        args.sheap_model_path,
        args.zoom_out,
        use_test_set=args.test,
        check=args.check,
    )
