## PyQt5 inference GUI

This folder adds a desktop GUI for running inference with the existing project model.

### Install

```bash
python -m pip install -r gui/requirements.txt
```

### Run

```bash
python gui/app.py
```

### Required files

- **Checkpoint**: `checkpoints/best_model.pth` (project root)
- **CSV for class order (recommended)**: `data/nih_metadata_prepped.csv`

On first run, the GUI will create **`gui/class_to_idx.json`** as the canonical class order:
- If `gui/class_to_idx.json` already exists, it is used directly.
- Else, if `data/nih_metadata_prepped.csv` exists, the GUI infers the label column (same logic as `dataset._infer_csv_columns`) and builds a sorted unique label list.
- Else, it falls back to `["Infiltration","No Finding","Effusion","Atelectasis"]` and prints a warning.

### Notes

- Preprocessing matches validation: **Resize to 448×448 + ImageNet normalization**, no augmentations.
- Metadata vector is EXACT: **[age/100, Male(1)/Female(0), AP, PA, Lateral]**.
- Model loads once on app startup; inference uses `torch.no_grad()` and AMP autocast on CUDA.

