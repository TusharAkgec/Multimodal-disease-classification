from __future__ import annotations

from typing import Tuple

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_img_size() -> int:
    """
    Read IMG_SIZE from config.py so GUI preprocessing matches training.
    """
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        import config  # type: ignore

        return int(getattr(config, "IMG_SIZE", 224))
    except Exception:
        return 224


_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((_get_img_size(), _get_img_size())),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """
    Validation preprocessing: Resize(IMG_SIZE x IMG_SIZE) + ImageNet normalization.
    Returns a float32 tensor of shape [3, IMG_SIZE, IMG_SIZE].
    """
    img = Image.open(image_path).convert("RGB")
    t = _VAL_TRANSFORM(img).to(dtype=torch.float32)
    if t.dim() != 3 or t.shape[0] != 3:
        raise RuntimeError(f"Unexpected image tensor shape: {tuple(t.shape)} (expected (3,H,W))")
    return t


def build_metadata(age: float, gender: str, view: str) -> torch.Tensor:
    """
    EXACT metadata vector format:
      [age_norm, gender, AP, PA, Lateral]
    where age_norm = age/100, Male=1, Female=0, and view is one-hot.
    """
    age_f = float(age)
    age_norm = age_f / 100.0

    g = str(gender).strip().lower()
    if g in {"m", "male"}:
        gender_val = 1.0
    elif g in {"f", "female"}:
        gender_val = 0.0
    else:
        raise ValueError("Gender must be Male or Female.")

    v = str(view).strip().lower()
    ap = 1.0 if v == "ap" else 0.0
    pa = 1.0 if v == "pa" else 0.0
    lateral = 1.0 if v == "lateral" else 0.0
    if (ap + pa + lateral) != 1.0:
        raise ValueError("View must be one of: AP, PA, Lateral.")

    meta = torch.tensor([age_norm, gender_val, ap, pa, lateral], dtype=torch.float32)
    return meta


def prepare_inputs(
    image_path: str, age: float, gender: str, view: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_tensor = load_and_preprocess_image(image_path)
    metadata_tensor = build_metadata(age=age, gender=gender, view=view)
    return image_tensor, metadata_tensor

