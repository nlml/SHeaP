import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sheap import inference_images_list, load_sheap_model, render_mesh
from sheap.tiny_flame import TinyFlame, pose_components_to_rotmats

os.environ["PYOPENGL_PLATFORM"] = "egl"


if __name__ == "__main__":
    # Load SHeaP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sheap_model = load_sheap_model(model_type="expressive").to(device)

    # Inference on example images
    folder_containing_images = Path("example_images/")
    image_paths = list(sorted(folder_containing_images.glob("*.jpg")))
    with torch.no_grad():
        predictions = inference_images_list(
            model=sheap_model,
            device=device,
            image_paths=image_paths,
        )

    # Load and infer FLAME with our predicted parameters
    flame_dir = Path("FLAME2020/")
    flame = TinyFlame(flame_dir / "generic_model.pt", eyelids_ckpt=flame_dir / "eyelids.pt")
    verts = flame(
        shape=predictions["shape_from_facenet"],
        expression=predictions["expr"],
        pose=pose_components_to_rotmats(predictions),
        eyelids=predictions["eyelids"],
        translation=predictions["cam_trans"],
    )

    # Render the FLAME mesh for each input image
    c2w = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.float32
    )
    for i_frame in range(verts.shape[0]):
        color, depth = render_mesh(verts=verts[i_frame], faces=flame.faces, c2w=c2w)
        outpath = image_paths[i_frame].with_name(f"{image_paths[i_frame].name}_rendered.png")
        if outpath.exists():
            outpath.unlink()
        original = Image.open(image_paths[i_frame]).convert("RGB").resize((512, 512))
        combined = Image.new("RGB", (512 * 3, 512))
        mask = (depth > 0).astype(np.float32)[..., None]
        blended = (np.array(color) * mask + np.array(original) * (1 - mask)).astype(np.uint8)
        combined.paste(original, (0, 0))
        combined.paste(Image.fromarray(color), (512, 0))
        combined.paste(Image.fromarray(blended), (512 * 2, 0))
        combined.save(outpath)
