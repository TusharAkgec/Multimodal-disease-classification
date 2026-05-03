"""
Training configuration for SpatialGated multimodal model.
"""
import os
import torch

# -----------------------------------------------------------------------------
# Image & data
# -----------------------------------------------------------------------------
IMG_SIZE = 224
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "nih_metadata_prepped.csv")

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 60
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 10
USE_AMP = True  # Can be set False by setup_and_train.py on AMP error

# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# -----------------------------------------------------------------------------
# Training schedule
# -----------------------------------------------------------------------------
FREEZE_BACKBONE_EPOCHS = 5   # Freeze backbone for first 5 epochs
TTA_EPOCHS_LAST = 10         # Enable TTA during last 10 epochs
GRADIENT_CLIP = 2.0
LABEL_SMOOTHING = 0.1
