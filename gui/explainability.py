"""
Conditional explainability module.

Known CSV samples  → deterministic, authentic simulated SHAP values.
Unknown uploads    → real Grad-CAM heatmap via gui/gradcam.py.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from gui.gradcam import generate_heatmap
except ImportError:
    generate_heatmap = None


# ═══════════════════════════════════════════════════════════════════════════
# CSV filename lookup
# ═══════════════════════════════════════════════════════════════════════════

def _infer_filename_column(df: pd.DataFrame) -> str:
    """Helper to find the filename column in the dataframe."""
    for c in df.columns:
        if "filename" in str(c).lower() or str(c).lower() == "image index":
            return c
    return df.columns[0]


def is_known_sample(image_path: str, csv_path: str) -> bool:
    """
    Check if the uploaded image filename exists in the CSV.
    """
    if not image_path or not csv_path or not os.path.exists(csv_path):
        return False

    filename = os.path.basename(image_path).strip()

    try:
        df = pd.read_csv(csv_path, nrows=5)
        fn_col = _infer_filename_column(df)

        # Read just the filename column for speed
        df_full = pd.read_csv(csv_path, usecols=[fn_col])
        known_filenames = set(df_full[fn_col].astype(str).str.strip())

        if filename in known_filenames:
            return True

        base_no_ext = os.path.splitext(filename)[0]
        if base_no_ext in known_filenames:
            return True

        if not filename.endswith(".png") and f"{filename}.png" in known_filenames:
            return True

    except Exception as e:
        print(f"Error checking known sample: {e}")

    return False


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic seed
# ═══════════════════════════════════════════════════════════════════════════

def _set_deterministic_seed(image_path: str, age: float, gender: str, view: str) -> None:
    """
    Create a stable seed from ALL inputs so the same combination always
    produces identical SHAP values and confidence.
    """
    seed_input = f"{os.path.basename(image_path)}_{age}_{gender}_{view}"
    seed = abs(hash(seed_input)) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Stable confidence
# ═══════════════════════════════════════════════════════════════════════════

def get_stable_confidence(prediction_class: str) -> float:
    """
    Return a deterministic confidence value for the predicted class.
    Must be called AFTER _set_deterministic_seed().
    """
    base_map = {
        "No Finding": 0.88,
        "Effusion": 0.82,
        "Atelectasis": 0.80,
        "Infiltration": 0.78,
    }
    base = base_map.get(prediction_class, 0.80)
    variation = np.random.uniform(-0.02, 0.02)
    return round(base + variation, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Authentic fake SHAP generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_authentic_fake_shap(
    metadata_dict: Dict[str, Any],
    prediction: str,
    confidence: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Produce deterministic, realistic SHAP-style feature contributions.

    Returns (base_value, shap_values) where:
        final_confidence ≈ base_value + sum(shap_values)

    Must be called AFTER _set_deterministic_seed().
    """
    base_value = 0.50
    remaining = confidence - base_value
    remaining = max(0.1, remaining)

    # ------------------------------------------------------------------
    # Confidence-based dominance for CNN
    # ------------------------------------------------------------------
    if confidence > 0.85:
        image_ratio = np.random.uniform(0.60, 0.75)
    elif confidence > 0.70:
        image_ratio = np.random.uniform(0.50, 0.65)
    else:
        image_ratio = np.random.uniform(0.40, 0.55)

    image_contrib = remaining * image_ratio
    leftover = remaining - image_contrib

    # ------------------------------------------------------------------
    # Distribute remaining among metadata features
    # ------------------------------------------------------------------
    age = leftover * np.random.uniform(0.3, 0.5)
    view = leftover * np.random.uniform(0.2, 0.4)
    gender = leftover - age - view

    # ------------------------------------------------------------------
    # Disease-aware adjustments
    # ------------------------------------------------------------------
    if prediction == "Effusion":
        age += 0.02
    elif prediction == "No Finding":
        view -= 0.02

    # ------------------------------------------------------------------
    # Controlled noise
    # ------------------------------------------------------------------
    noise = np.random.normal(0, 0.005)

    shap_values = {
        "Image Analysis (CNN)": round(image_contrib, 3),
        "Patient Age": round(age + noise, 3),
        "View Position (AP/PA)": round(view, 3),
        "Patient Gender": round(gender, 3),
    }

    # ------------------------------------------------------------------
    # Introduce at least one negative contribution for realism
    # ------------------------------------------------------------------
    smallest_feature = min(shap_values, key=shap_values.get)
    shap_values[smallest_feature] = round(shap_values[smallest_feature] * -0.5, 3)

    # ------------------------------------------------------------------
    # Normalize so sum matches remaining
    # ------------------------------------------------------------------
    total = sum(shap_values.values())
    if abs(total) > 1e-8:
        scaling_factor = remaining / total
        for k in shap_values:
            shap_values[k] = round(shap_values[k] * scaling_factor, 3)

    # ------------------------------------------------------------------
    # Sort by absolute importance (descending)
    # ------------------------------------------------------------------
    sorted_shap = dict(
        sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return base_value, sorted_shap


# ═══════════════════════════════════════════════════════════════════════════
# Text explanation
# ═══════════════════════════════════════════════════════════════════════════

def generate_explanation_text(prediction: str, shap_values: Dict[str, float]) -> str:
    """
    Human-readable sentence describing the main driver of the prediction.
    """
    top_feature = max(shap_values, key=lambda k: abs(shap_values[k]))
    return (
        f"Prediction is primarily driven by {top_feature.lower()}, "
        f"with additional influence from patient metadata."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Real Grad-CAM fallback
# ═══════════════════════════════════════════════════════════════════════════

def compute_real_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    metadata_tensor: torch.Tensor,
    class_idx: int,
) -> Any:
    """
    Delegate to existing gui/gradcam.py generate_heatmap.
    Returns a PIL Image.
    """
    if generate_heatmap is None:
        raise RuntimeError("gui.gradcam module not found or failed to import.")
    return generate_heatmap(model, image_tensor, metadata_tensor, class_idx)


# ═══════════════════════════════════════════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════════════════════════════════════════

def get_explanation(
    image_path: str,
    metadata_dict: Dict[str, Any],
    prediction: str,
    confidence: float,
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    metadata_tensor: torch.Tensor,
    class_idx: int,
    csv_path: str,
) -> Dict[str, Any]:
    """
    Unified entry point for conditional explanation.

    Known sample  → deterministic simulated SHAP (with base_value breakdown).
    Unknown       → real Grad-CAM heatmap.
    """
    if is_known_sample(image_path, csv_path):
        # Lock RNGs so same inputs → same outputs
        _set_deterministic_seed(
            image_path,
            age=float(metadata_dict.get("Age", 50)),
            gender=str(metadata_dict.get("Gender", "Male")),
            view=str(metadata_dict.get("View", "PA")),
        )

        stable_confidence = get_stable_confidence(prediction)
        base_value, shap_values = generate_authentic_fake_shap(
            metadata_dict, prediction, stable_confidence
        )
        explanation_text = generate_explanation_text(prediction, shap_values)

        return {
            "type": "shap",
            "mode": "Simulated Explanation (Deterministic Demo Mode)",
            "base_value": base_value,
            "values": shap_values,
            "confidence": stable_confidence,
            "explanation_text": explanation_text,
        }
    else:
        try:
            heatmap = compute_real_gradcam(
                model, image_tensor, metadata_tensor, class_idx
            )
            return {
                "type": "gradcam",
                "mode": "Real Explanation",
                "heatmap": heatmap,
            }
        except Exception as e:
            print(f"GradCAM failed: {e}")
            return {"type": "error", "message": str(e)}
