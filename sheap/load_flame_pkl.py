import os
import pickle
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from torch import Tensor


def load_pkl_format_flame_model(path: Union[str, os.PathLike, Path]) -> Dict[str, Tensor]:
    """Load a FLAME model from a pickle file format.

    Loads FLAME model parameters including faces, kinematic tree, joint regressor,
    shape directions, joints, weights, pose directions, and vertex template.

    Args:
        path: Path to the FLAME model pickle file.

    Returns:
        Dictionary containing FLAME model parameters as tensors.
    """
    flame_params: Dict[str, Tensor] = {}
    with open(path, "rb") as f:
        flame_data = pickle.load(f, encoding="latin1")
    flame_params["faces"] = torch.from_numpy(flame_data["f"].astype("int64"))
    kintree = torch.from_numpy(flame_data["kintree_table"].astype("int64"))
    kintree[kintree > 100] = -1
    flame_params["kintree"] = kintree
    flame_params["J_regressor"] = torch.from_numpy(
        flame_data["J_regressor"].toarray().astype("float32")
    )
    for thing in ["shapedirs", "J", "weights", "posedirs", "v_template"]:
        flame_params[thing] = torch.from_numpy(np.array(flame_data[thing]).astype("float32"))
    return flame_params
