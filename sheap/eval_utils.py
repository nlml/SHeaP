from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as tud
import trimesh
from PIL import Image
from tqdm import tqdm


def _preproc_im_default(p: Union[str, Path]) -> Image.Image:
    """Default image preprocessing function that loads an image from a path.

    Args:
        p: Path to the image file.

    Returns:
        PIL Image object.
    """
    return Image.open(p)


class ImsDataset(tud.Dataset):
    """Dataset for loading and preprocessing images.

    Args:
        image_paths: List of paths to image files.
        img_wh: Tuple of (width, height) to resize images to.
        load_and_preproc_im: Optional custom function to load and preprocess images.
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        img_wh: Tuple[int, int],
        load_and_preproc_im: Optional[
            Callable[[Union[str, Path]], Image.Image]
        ] = _preproc_im_default,
    ) -> None:
        self.image_paths = image_paths
        self.img_wh = img_wh
        self.load_and_preproc_im = load_and_preproc_im
        if self.load_and_preproc_im is None:
            self.load_and_preproc_im = _preproc_im_default

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and preprocess an image at the given index.

        Args:
            idx: Index of the image to load.

        Returns:
            Preprocessed image tensor of shape (3, H, W) with values in [0, 1].
        """
        impath = self.image_paths[idx]
        pil_im = self.load_and_preproc_im(impath)
        im = pil_im.convert("RGB").resize(self.img_wh)
        im = np.array(im).astype("float64") / 255.0
        im = torch.from_numpy(im).permute(2, 0, 1).float()
        return im


@torch.no_grad()
def inference_images_list(
    model: torch.nn.Module,
    device: torch.device,
    image_paths: List[Union[str, Path]],
    custom_pil_im_load_fn: Optional[Callable[[Union[str, Path]], Image.Image]] = None,
    img_wh: Tuple[int, int] = (224, 224),
    batch_size: int = 4,
    num_workers: int = 4,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run inference on a list of images using a model.

    Args:
        model: PyTorch model to use for inference.
        device: Device to run inference on.
        image_paths: List of paths to image files.
        custom_pil_im_load_fn: Optional custom function to load and preprocess images.
        img_wh: Tuple of (width, height) to resize images to. Default is (224, 224).
        batch_size: Batch size for inference. Default is 4.
        num_workers: Number of workers for data loading. Default is 4.
        verbose: Whether to print output shapes. Default is False.

    Returns:
        Dictionary mapping output keys to concatenated tensors across all batches.
    """
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


def invert_4x4_cam_matrix(inp_cam: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 camera transformation matrix.

    Args:
        inp_cam: 4x4 camera transformation matrix.

    Returns:
        Inverted 4x4 camera transformation matrix.
    """
    rr = inp_cam[:3, :3].T
    tt = rr @ -inp_cam[:3, 3]
    inv_cam = torch.eye(4, device=inp_cam.device, dtype=inp_cam.dtype)
    inv_cam[:3, :3] = rr
    inv_cam[:3, 3] = tt
    return inv_cam


def save_obj(outpath: Union[str, Path], verts: np.ndarray, faces: np.ndarray) -> None:
    """Save vertices and faces as an OBJ file using trimesh.

    Args:
        outpath: Path where the OBJ file will be saved.
        verts: Vertex array of shape (N, 3).
        faces: Face array of shape (M, 3) containing vertex indices.
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(outpath)


def save_result(
    flame_faces: np.ndarray,
    base_dir: Union[str, Path],
    verts_with_zero_exprn: np.ndarray,
    lmks7_3d: torch.Tensor,
    preds_outdir: Path,
    input_im_path: Union[str, Path],
    verbose: bool = False,
) -> None:
    """Save FLAME model prediction results to disk.

    Saves the predicted mesh as an OBJ file and 3D landmarks as a numpy file.
    Vertices and landmarks are scaled by 1000 to match MICA format.

    Args:
        flame_faces: FLAME model face indices.
        base_dir: Base directory for computing relative paths.
        verts_with_zero_exprn: Predicted vertices with zero expression.
        lmks7_3d: 3D landmarks tensor.
        preds_outdir: Output directory for predictions.
        input_im_path: Path to the input image.
        verbose: Whether to print save confirmation messages. Default is False.
    """
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


def add_pct_to_bbox(
    top: int,
    left: int,
    bottom: int,
    right: int,
    im_np_array: Union[np.ndarray, Image.Image],
    pct: float = 0.2,
) -> Tuple[int, int, int, int]:
    """Expand a bounding box by a percentage while staying within image bounds.

    Args:
        top: Top coordinate of the bounding box.
        left: Left coordinate of the bounding box.
        bottom: Bottom coordinate of the bounding box.
        right: Right coordinate of the bounding box.
        im_np_array: Image as numpy array or PIL Image.
        pct: Percentage to expand the bounding box by. Default is 0.2 (20%).

    Returns:
        Tuple of (top, left, bottom, right) coordinates of the expanded bounding box.
    """
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


def resize_to_max_size(
    im: Union[np.ndarray, Image.Image], max_size: int = 512, pad_smaller: bool = True
) -> Union[np.ndarray, Image.Image]:
    """Resize an image to fit within a maximum size, optionally padding to square.

    Args:
        im: Input image as numpy array or PIL Image.
        max_size: Maximum size for the longest dimension. Default is 512.
        pad_smaller: Whether to pad the smaller dimension to create a square image.
            Default is True.

    Returns:
        Resized (and optionally padded) image in the same format as input.
    """
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
