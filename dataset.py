"""
NIH chest X-ray dataset with metadata.
Returns: image_tensor, metadata_tensor [age_norm, gender, AP, PA, Lateral], label (class index).
Uses OpenCV + Albumentations. No vertical flip anywhere.
"""
import os
import config
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization for pretrained backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(size: int = config.IMG_SIZE):
    """Train: Resize, HorizontalFlip only, RandomBrightnessContrast, Normalize, ToTensorV2. No vertical flip."""
    return A.Compose([
    A.Resize(height=size, width=size),

    A.HorizontalFlip(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.02,
        scale_limit=0.05,
        rotate_limit=10,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),

    A.RandomBrightnessContrast(p=0.3),

    A.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        max_pixel_value=255.0
    ),

    ToTensorV2(),
])


def get_val_transforms(size: int = config.IMG_SIZE):
    """Validation: Resize, Normalize, ToTensorV2."""
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def _infer_csv_columns(df: pd.DataFrame) -> dict:
    """Infer column names for filename, age, gender, view, label (handles CSV quirks)."""
    cols = df.columns.str.strip().tolist()
    out = {}
    for c in cols:
        c_lower = c.lower()
        if "filename" in c_lower or c == "filename":
            out["filename"] = c
        if "age" in c_lower and "patient" in c_lower:
            out["age"] = c
        if "gender" in c_lower and "patient" in c_lower:
            out["gender"] = c
        if c_lower == "view":
            out["view"] = c
        if c_lower == "label":
            out["label"] = c
    if "label" not in out and len(cols) > 0:
        out["label"] = cols[-1]
    if "age" not in out:
        for c in cols:
            if "age" in c.lower():
                out["age"] = c
                break
    if "gender" not in out:
        for c in cols:
            if "gender" in c.lower():
                out["gender"] = c
                break
    if "view" not in out:
        for c in cols:
            if c.lower() == "view":
                out["view"] = c
                break
    if "filename" not in out:
        out["filename"] = cols[0]
    return out


def build_metadata_vector(age: float, gender_val, view: str) -> np.ndarray:
    """
    Metadata tensor format EXACTLY: [age_normalized, gender, AP, PA, Lateral].
    - age_normalized = age / 100.0
    - gender: numeric (M=1, F=0 if string)
    - AP, PA, Lateral: one-hot from view
    """
    age_norm = float(age) / 100.0
    if isinstance(gender_val, (int, float)):
        g = float(gender_val)
    else:
        g = 1.0 if str(gender_val).strip().upper() == "M" else 0.0
    v = str(view).strip().upper()
    AP = 1.0 if v == "AP" else 0.0
    PA = 1.0 if v == "PA" else 0.0
    Lateral = 1.0 if v == "LATERAL" else 0.0
    return np.array([age_norm, g, AP, PA, Lateral], dtype=np.float32)


class NIHChestDataset(Dataset):
    """
    Dataset over data/images/ and data/nih_metadata_prepped.csv.
    Returns: image_tensor, metadata_tensor, label (class index).
    """

    def __init__(self, image_dir: str, csv_path: str, dataframe: pd.DataFrame,
                 class_to_idx: dict, transform=None):
        self.image_dir = Path(image_dir)
        self.csv_path = csv_path
        self.df = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self._col = _infer_csv_columns(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = row[self._col["filename"]]
        if isinstance(fn, bytes):
            fn = fn.decode("utf-8")
        path = self.image_dir / fn
        if not path.suffix and not str(fn).endswith(".png"):
            path = self.image_dir / f"{fn}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Missing image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        age = row[self._col["age"]]
        gender = row[self._col["gender"]]
        view = row[self._col["view"]]
        metadata = build_metadata_vector(age, gender, view)

        label_str = row[self._col["label"]]
        if isinstance(label_str, bytes):
            label_str = label_str.decode("utf-8")
        label = self.class_to_idx.get(str(label_str).strip(), 0)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"].float()
            image = torch.clamp(image, -5, 5)

        meta_tensor = torch.from_numpy(metadata)
        return image, meta_tensor, label


def load_metadata_and_labels(csv_path: str, image_dir: str):
    """
    Load CSV and filter rows whose image exists in image_dir.
    Returns: (dataframe, class_to_idx).
    """
    df = pd.read_csv(csv_path)
    col = _infer_csv_columns(df)
    image_dir = Path(image_dir)
    filenames = set()
    if image_dir.exists():
        for p in image_dir.iterdir():
            if p.is_file():
                filenames.add(p.name)
    fn_col = col["filename"]
    def exists(name):
        s = str(name).strip()
        return s in filenames or (s + ".png" in filenames) if not s.endswith(".png") else s in filenames
    if filenames:
        df = df[df[fn_col].astype(str).apply(exists)]
    label_col = col["label"]
    labels = df[label_col].dropna().astype(str).str.strip().unique()
    class_to_idx = {c: i for i, c in enumerate(sorted(labels))}
    return df, class_to_idx, label_col
