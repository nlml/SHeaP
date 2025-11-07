import urllib.request
from pathlib import Path
from typing import Dict, Literal

import torch

# Map model types to filenames and (optional) download URLs
MODEL_INFO: Dict[str, Dict[str, str]] = {
    "paper": {
        "filename": "model_paper.pt",
        "url": "https://github.com/nlml/sheap/releases/download/v1.0.0/model_paper.pt",
    },
    "expressive": {
        "filename": "model_expressive.pt",
        "url": "https://github.com/nlml/sheap/releases/download/v1.0.0/model_expressive.pt",
    },
}


def ensure_model_downloaded(
    model_type: Literal["paper", "expressive"] = "paper", models_dir: Path = Path("./models")
) -> None:
    """Ensure the requested model is present locally, downloading if needed.

    Args:
        model_type: Which model variant to use. Valid options are "paper" or "expressive".
            Default is "paper".
        models_dir: Directory where models are stored. Default is "./models".

    Raises:
        ValueError: If model_type is not recognized.
        FileNotFoundError: If model file is not found and no download URL is configured.
    """
    if model_type not in MODEL_INFO:
        valid = ", ".join(MODEL_INFO.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Valid options: {valid}")

    models_dir = Path(models_dir)
    filename = MODEL_INFO[model_type]["filename"]
    url = MODEL_INFO[model_type]["url"]
    model_path = models_dir / filename

    if model_path.exists():
        return

    # If we don't have a URL
    if not url:
        raise FileNotFoundError(
            f"Model file '{model_path}' not found and no download URL is configured for "
            f"model_type='{model_type}'. Place the file manually or update MODEL_INFO with a valid URL."
        )

    print(f"Downloading '{model_type}' model to {model_path}...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, model_path)


def load_sheap_model(
    model_type: Literal["paper", "expressive"] = "paper", models_dir: Path = Path("./models")
) -> torch.jit.ScriptModule:
    """Load the SHeaP model as a PyTorch JIT trace.

    The function will download the model if it is not present locally (if a URL is
    configured for the selected model_type).

    Args:
        model_type: Which model variant to load. Valid options are "paper" or "expressive".
            Default is "paper" for backward compatibility.
        models_dir: Directory where models are stored. Default is "./models".

    Returns:
        The loaded SHeaP model as a PyTorch JIT ScriptModule.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type not in MODEL_INFO:
        valid = ", ".join(MODEL_INFO.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Valid options: {valid}")

    models_dir = Path(models_dir)
    ensure_model_downloaded(model_type=model_type, models_dir=models_dir)
    filename = MODEL_INFO[model_type]["filename"]
    sheap_model = torch.jit.load(models_dir / filename)
    return sheap_model
