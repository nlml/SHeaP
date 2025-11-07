from typing import Tuple, Union

import numpy as np
import pyrender
import torch
import trimesh


def render_mesh(
    verts: Union[np.ndarray, torch.Tensor],
    faces: Union[np.ndarray, torch.Tensor],
    c2w: Union[np.ndarray, torch.Tensor],
    img_width: int = 512,
    img_height: int = 512,
    fov_degrees: Union[float, int] = 14.2539,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a mesh using pyrender with a perspective camera defined by FOV.

    Args:
        verts: Mesh vertex positions of shape (N, 3).
        faces: Triangle vertex indices of shape (F, 3).
        c2w: Camera-to-world transform matrix (extrinsics) of shape (4, 4).
        img_width: Rendered image width in pixels. Default is 512.
        img_height: Rendered image height in pixels. Default is 512.
        fov_degrees: Vertical field of view in degrees. Default is 14.2539.

    Returns:
        Tuple containing:
            - color: RGB image from the render of shape (H, W, 3) as uint8.
            - depth: Depth map from the render of shape (H, W) as float32.
    """
    if isinstance(c2w, torch.Tensor):
        c2w = c2w.detach().cpu().numpy()
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if not isinstance(fov_degrees, (float, int)):
        fov_degrees = float(fov_degrees)

    # Convert degrees to radians
    yfov = np.deg2rad(fov_degrees)

    # Create trimesh mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Convert to pyrender mesh
    render_mesh = pyrender.Mesh.from_trimesh(mesh)

    # Create scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(render_mesh)

    # Add directional light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=c2w)

    # Perspective camera
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=img_width / img_height)

    # pyrender expects camera-to-world
    scene.add(camera, pose=c2w)

    # Offscreen render
    renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
    color, depth = renderer.render(scene)

    return color, depth
