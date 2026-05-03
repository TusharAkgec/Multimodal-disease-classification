from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _find_last_conv(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def _tensor_to_pil_denorm(img: torch.Tensor) -> Image.Image:
    t = img.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    t = (t * _IMAGENET_STD) + _IMAGENET_MEAN
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def generate_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    metadata_tensor: torch.Tensor,
    class_idx: int,
) -> Image.Image:
    """
    Minimal Grad-CAM implementation.
    Attempts to hook the last Conv2d in model.backbone; if not found, raises.
    Returns a PIL image overlay (RGB) aligned to the network input resolution.
    """
    device = next(model.parameters()).device
    model.eval()

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise RuntimeError("Model does not have a 'backbone' attribute.")

    target = _find_last_conv(backbone)
    if target is None:
        raise RuntimeError("Could not locate a Conv2d layer for Grad-CAM.")

    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None

    def fwd_hook(_m, _inp, out):
        nonlocal activations
        activations = out

    def bwd_hook(_m, _gin, gout):
        nonlocal gradients
        gradients = gout[0]

    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)

    orig_requires_grad: Dict[int, bool] = {}
    try:
        with torch.enable_grad():
            # Temporarily enable gradients for Grad-CAM only.
            for p in model.parameters():
                orig_requires_grad[id(p)] = bool(p.requires_grad)
                p.requires_grad = True

            img_b = image_tensor.unsqueeze(0).to(device)
            meta_b = metadata_tensor.unsqueeze(0).to(device)

            # Ensure input image requires gradients (required by some Grad-CAM variants and hooks).
            img_b.requires_grad_(True)

            logits = model(img_b, meta_b)
            if logits.dim() != 2:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            score = logits[0, int(class_idx)]
            model.zero_grad(set_to_none=True)  # type: ignore[call-arg]
            score.backward()

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # activations/gradients: [B, C, H, W]
        w = gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (w * activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        out_h, out_w = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
        cam_up = torch.nn.functional.interpolate(
            cam, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        cam_np = cam_up[0, 0].detach().cpu().numpy()

        base = _tensor_to_pil_denorm(img_b)
        base_np = np.array(base).astype(np.float32)

        # Simple "jet-like" colormap (no matplotlib dependency)
        heat = np.zeros((out_h, out_w, 3), dtype=np.float32)
        heat[..., 0] = np.clip(1.5 * (cam_np - 0.5), 0, 1)  # red
        heat[..., 1] = np.clip(1.5 * (1 - np.abs(cam_np - 0.5) * 2), 0, 1)  # green
        heat[..., 2] = np.clip(1.5 * (0.5 - cam_np), 0, 1)  # blue
        heat = (heat * 255.0).astype(np.float32)

        alpha = 0.35
        overlay = (1 - alpha) * base_np + alpha * heat
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    finally:
        # Restore original requires_grad flags
        try:
            for p in model.parameters():
                if id(p) in orig_requires_grad:
                    p.requires_grad = orig_requires_grad[id(p)]
        except Exception:
            pass
        try:
            h1.remove()
            h2.remove()
        except Exception:
            pass

