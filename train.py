"""
Full training pipeline for SpatialGated multimodal model.
Run: python train.py
"""
import os
import sys
import inspect
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

import config
from dataset import (
    NIHChestDataset,
    load_metadata_and_labels,
    get_train_transforms,
    get_val_transforms,
)

# Set by main() so workers never run device/cudnn code
device = None

# -----------------------------------------------------------------------------
# Safety checks (before training)
# -----------------------------------------------------------------------------
def run_safety_checks(create_backbone_fn, predict_with_tta_fn):
    """Verify backbone 4D, IMG_SIZE==448, predict_with_tta has no """
    errors = []
    if config.IMG_SIZE < 128:
        errors.append(f"IMG_SIZE too small: {config.IMG_SIZE}")
    source = inspect.getsource(predict_with_tta_fn)
    if "dims=[2]" in source:
        errors.append("predict_with_tta must NOT contain  (no vertical flip)")
    if "dims=[3]" not in source:
        errors.append("predict_with_tta must use dims=[3] for horizontal flip only")
    # Backbone 4D check
    backbone = create_backbone_fn("convnext_base")
    t = torch.randn(1, 3, 448, 448)
    out = backbone.forward_features(t)
    if out.dim() != 4:
        errors.append(f"backbone.forward_features must return 4D tensor, got dim={out.dim()}, shape={out.shape}")
    if errors:
        raise RuntimeError("Safety checks failed:\n" + "\n".join(f"  - {e}" for e in errors))
    print("Safety checks passed: backbone 4D, IMG_SIZE=224, no dims in TTA.")


# -----------------------------------------------------------------------------
# Class weights
# -----------------------------------------------------------------------------
def compute_class_weights(labels: np.ndarray, num_classes: int):
    """Inverse frequency class weights for imbalanced data."""
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


# -----------------------------------------------------------------------------
# Train / validation steps
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
    for batch_idx, (images, metadata, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images, metadata)
            loss = criterion(logits, labels)

            if torch.isnan(loss):
                print("Skipping NaN batch")
                optimizer.zero_grad(set_to_none=True)
                continue

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)

        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    epoch_loss = running_loss / len(loader) if len(loader) else 0.0
    return epoch_loss


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = [] 
    with torch.no_grad():
        for images, metadata, labels in loader:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images, metadata)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print("\nValidation Metrics:")
    print(classification_report(all_labels, all_preds))
    model.train()
    return acc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = (config.IMG_SIZE > 0)
    # ===== RTX GPU SPEED OPTIMIZATION (TF32 ENABLE) =====
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.set_num_threads(min(8, os.cpu_count() or 8))

    import model as model_module
    from model import (
        create_backbone,
        SpatialGatedModel,
        create_optimizer,
        predict_with_tta,
    )

    print("RUNNING TRAIN FILE:", os.path.abspath(__file__))
    print("MODEL MODULE PATH:", inspect.getfile(model_module))
    print("Using device:", device)
    print("Config: IMG_SIZE={}, BATCH_SIZE={}, NUM_EPOCHS={}, DEVICE={}".format(
        config.IMG_SIZE, config.BATCH_SIZE, config.NUM_EPOCHS, config.DEVICE))
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(inspect.getfile(predict_with_tta))
    run_safety_checks(create_backbone, predict_with_tta)

    # A. Load CSV and filter by existing images
    df_full, class_to_idx, label_col = load_metadata_and_labels(config.CSV_PATH, config.IMAGE_DIR)

    # -------------------------------
    # Patient-balanced sampling
    # -------------------------------

    target_patients_per_class = 2000
    patient_col = "Patient ID"

    # determine each patient's class
    patient_labels = df_full.groupby(patient_col)[label_col].agg(lambda x: x.mode()[0]).reset_index()

    selected_patients = []

    for cls in class_to_idx.keys():

        class_patients = patient_labels[patient_labels[label_col] == cls][patient_col]

        sampled = np.random.choice(
            class_patients,
            size=min(target_patients_per_class, len(class_patients)),
            replace=False
        )

        selected_patients.extend(sampled)

    # keep all images belonging to those patients
    df_full = df_full[df_full[patient_col].isin(selected_patients)]

    print("Balanced dataset size:", len(df_full))
    print("Unique patients:", df_full[patient_col].nunique())
    print("Class distribution:")
    print(df_full[label_col].value_counts())

    num_classes = len(class_to_idx)
    if num_classes == 0:
        raise RuntimeError("No classes found. Check CSV and image dir.")
    print(f"Classes: {num_classes}, samples: {len(df_full)}")

    # B. Train/validation split (stratified by label when possible)
    y = df_full[label_col].astype(str).str.strip().map(class_to_idx)
    y = y.fillna(0).astype(int)
    try:
        train_df, val_df = train_test_split(
            df_full, test_size=0.2, stratify=y, random_state=42, shuffle=True
        )
    except ValueError:
        train_df, val_df = train_test_split(
            df_full, test_size=0.2, random_state=42, shuffle=True
        )

    # Class weights from training set
    train_labels = train_df[label_col].astype(str).str.strip().map(class_to_idx).values
    class_weights = compute_class_weights(train_labels, num_classes)

    # C. Datasets
    train_ds = NIHChestDataset(
        config.IMAGE_DIR,
        config.CSV_PATH,
        train_df,
        class_to_idx,
        transform=get_train_transforms(config.IMG_SIZE),
    )
    val_ds = NIHChestDataset(
        config.IMAGE_DIR,
        config.CSV_PATH,
        val_df,
        class_to_idx,
        transform=get_val_transforms(config.IMG_SIZE),
    )

    # D. Dataloaders
    _nw = min(10, os.cpu_count() or 4)
    _pin = (device.type == "cuda")
    _persistent = _nw > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=_nw,
        pin_memory=_pin,
        persistent_workers=_persistent,
        prefetch_factor=4 if _nw > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=_nw,
        pin_memory=_pin,
        persistent_workers=_persistent,
        prefetch_factor=4 if _nw > 0 else None,
    )

    # E. Model
    backbone = create_backbone("convnext_base")
    model = SpatialGatedModel(
        backbone=backbone,
        meta_features=5,
        num_classes=num_classes,
    )
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    # G. Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)

    # H. Optimizer (using create_optimizer from model.py)
    optimizer = create_optimizer(model, backbone_lr=config.LR_BACKBONE)

    # I. Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Checkpoints dir and resume state
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "last_checkpoint.pth")
    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(last_checkpoint_path):
        print("🔄 Resuming from checkpoint...")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        print(f"Resumed at epoch {start_epoch}")

        for p in model.backbone.parameters():
            p.requires_grad = (start_epoch >= config.FREEZE_BACKBONE_EPOCHS)

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # Freeze backbone first 5 epochs, unfreeze afterward (0-based epoch)
        for p in model.backbone.parameters():
            p.requires_grad = (epoch >= config.FREEZE_BACKBONE_EPOCHS)

        print(f"\n🚀 Starting Epoch {epoch+1}/{config.NUM_EPOCHS}")
        epoch_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )

        scheduler.step()

        val_acc = evaluate(model, val_loader, device)

        print("\n" + "=" * 65)
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("=" * 65)

        # Save best model (only if improved)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print("🔥 New BEST model saved!")

        # Save last checkpoint (always, for resume)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_acc,
        }
        torch.save(checkpoint, last_checkpoint_path)

    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
