from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap


def pil_image_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    img = pil_image.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


def format_probabilities(
    probs: np.ndarray, class_names: Sequence[str]
) -> List[Tuple[str, float, str]]:
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if len(probs) != len(class_names):
        raise ValueError(f"probs length {len(probs)} != class_names length {len(class_names)}")
    out: List[Tuple[str, float, str]] = []
    for name, p in zip(class_names, probs):
        p_f = float(p)
        out.append((str(name), p_f, f"{p_f * 100.0:.1f}%"))
    return out


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def ordered_class_names_from_mapping(class_to_idx: Dict[str, int]) -> List[str]:
    return [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: int(kv[1]))]

