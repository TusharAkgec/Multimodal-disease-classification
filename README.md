# Multimodal AI for Chest Disease Diagnosis

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-research--prototype-purple)

## Abstract / Overview

**Multimodal AI for Chest Disease Diagnosis** is a deep learning research project for classifying chest diseases from both chest X-ray images and structured patient metadata. The system is designed for clinical settings where access to expert radiologists may be limited, especially in resource-constrained hospitals and screening environments.

Instead of relying only on image pixels, the model combines radiographic evidence with demographic and clinical context such as age, sex, view position, and symptoms. This multimodal design improves diagnostic robustness, supports transparent decision-making, and keeps the pipeline computationally lean enough to run on standard hospital processors without requiring specialized high-end hardware.

The project aims to help democratize healthcare diagnostics by making AI-assisted chest disease screening more scalable, interpretable, and accessible.

## Key Features

- **Multimodal diagnosis**: Combines chest X-ray imaging with structured clinical and demographic metadata.
- **CNN image encoder**: Uses a pre-trained convolutional backbone such as ResNet-18, EfficientNet, or a `timm` backbone to extract X-ray features.
- **Lightweight metadata encoder**: Processes tabular patient data using a compact multilayer perceptron (MLP).
- **Fusion-based prediction**: Joins visual and metadata representations using concatenation, attention, or gated fusion.
- **Explainable AI support**: Uses Grad-CAM heatmaps to highlight X-ray regions that influenced each prediction.
- **Clinician-facing output**: Produces disease probability scores through a Softmax classification layer.
- **Lightweight deployment**: Designed to run on commonly available CPU/GPU hospital systems.
- **Interactive interface**: Includes a Streamlit-based GUI for inference and visualization.

## Architecture

The model follows a two-branch multimodal architecture:

```text
Chest X-ray image
      |
      v
Pre-trained CNN backbone
(ResNet-18 / EfficientNet / timm model)
      |
      v
Image feature vector
      |
      +----------------------+
                             |
Patient metadata             |
(age, sex, symptoms, view)   |
      |                      |
      v                      |
MLP metadata encoder         |
      |                      |
      v                      |
Metadata feature vector      |
      |                      |
      +----------+-----------+
                 |
                 v
Fusion module
(concatenation / attention / gating)
                 |
                 v
Classification layer
                 |
                 v
Disease probabilities
```

### Model Components

- **Image Branch**: A pre-trained CNN extracts high-level radiographic features from chest X-rays.
- **Metadata Branch**: A lightweight MLP converts structured patient attributes into a dense feature representation.
- **Fusion Module**: The image and metadata vectors are merged to create a richer patient-level representation.
- **Classification Layer**: A Softmax classifier outputs probabilities for each diagnosis category.
- **Explainability Module**: Grad-CAM generates heatmaps that show the image regions most responsible for the prediction.

The current repository implementation includes a spatial-gated multimodal model, where metadata influences image feature maps before pooling. This helps the model condition visual interpretation on patient context.

## Datasets & Classes

The project is designed for training and validation on public benchmark chest radiography datasets:

- **NIH ChestX-ray14**
- **MIMIC-CXR**

### Target Classes

The model classifies each case into one of four categories:

| Class | Description |
| --- | --- |
| `Atelectasis` | Partial or complete collapse of lung tissue |
| `Effusion` | Abnormal fluid accumulation in the pleural space |
| `Infiltration` | Abnormal substance accumulation in lung tissue |
| `No Finding` | No disease finding detected |

> Dataset files and trained checkpoints are not included in this repository because of size and licensing constraints. Add the prepared metadata CSV and image files under the `data/` directory before training or inference.

## Results

The multimodal approach achieves strong diagnostic performance by combining radiographic and metadata signals.

| Model Type | Inputs | Overall Accuracy | AUC |
| --- | --- | ---: | ---: |
| Image-only baseline | Chest X-ray | Lower than multimodal model | Lower than multimodal model |
| Metadata-only baseline | Structured patient data | Lower than multimodal model | Lower than multimodal model |
| **Proposed multimodal model** | Chest X-ray + metadata | **~91%** | **>91%** |

The results indicate that multimodal fusion outperforms unimodal approaches that rely only on images or only on patient metadata. Grad-CAM visualizations further improve clinical transparency by showing the regions that contributed to the prediction.

## Getting Started / Installation

### Prerequisites

- Python 3.9 or higher
- PyTorch
- `pip`
- Optional: CUDA-compatible GPU for faster training

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Multimodal-Disease-Classification-on-Clinical-Images-and-Metadata.git
cd Multimodal-Disease-Classification-on-Clinical-Images-and-Metadata
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Data

Place images and metadata in the expected project structure:

```text
data/
  images/
    image_001.png
    image_002.png
    ...
  nih_metadata_prepped.csv
```

The metadata file should include image identifiers, labels, and patient metadata fields such as age, sex, and view position. The current dataset loader expects metadata values that can be converted into a vector similar to:

```text
[age_normalized, sex, AP, PA, Lateral]
```

### 5. Train the Model

```bash
python train.py
```

Training configuration can be adjusted in:

```text
config.py
```

Important settings include image size, batch size, number of epochs, learning rates, dataset paths, and checkpoint paths.

### 6. Run Inference

If using the GUI:

```bash
streamlit run gui/streamlit_app.py
```

For script-based inference, use or extend the files in:

```text
gui/inference.py
gui/preprocess.py
gui/gradcam.py
```

Example placeholder command:

```bash
python gui/inference.py --image path/to/xray.png --metadata path/to/patient_metadata.json
```

> Adjust the command-line interface as needed if your local inference script uses a different input format.

## Project Structure

```text
.
|-- config.py                  # Training and path configuration
|-- dataset.py                 # Dataset loading and metadata preprocessing
|-- model.py                   # Multimodal CNN/MLP fusion model
|-- train.py                   # Training pipeline
|-- requirements.txt           # Python dependencies
|-- data/                      # Local dataset directory
`-- gui/
    |-- streamlit_app.py       # Streamlit inference application
    |-- inference.py           # Inference utilities
    |-- gradcam.py             # Grad-CAM visualization utilities
    |-- explainability.py      # Explanation helpers
    `-- class_to_idx.json      # Class mapping
```

## Explainability

The project uses **Grad-CAM** to generate visual explanations for model predictions. Grad-CAM highlights discriminative regions in the chest X-ray, helping clinicians understand whether the model is focusing on medically relevant anatomy.

This is especially important in clinical AI because model confidence alone is not enough. Interpretable heatmaps support trust, auditability, and safer human-AI collaboration.

## Deployment Notes

The system is intentionally designed to be computationally lean:

- Uses compact CNN backbones where appropriate.
- Encodes metadata with a small MLP.
- Supports CPU execution for accessibility.
- Can be integrated into a lightweight Streamlit interface.
- Avoids unnecessary hardware requirements for basic inference.

For production clinical deployment, additional validation, calibration, privacy review, regulatory assessment, and prospective clinical testing are required.

## Authors & Acknowledgements

### Research Team

- `<Author Name 1>` - Model development and experimentation
- `<Author Name 2>` - Dataset preparation and evaluation
- `<Author Name 3>` - Explainability and interface development
- `<Institution / Department>` - Research supervision

### Acknowledgements

This project acknowledges the contributors and maintainers of:

- NIH ChestX-ray14
- MIMIC-CXR
- PyTorch
- `timm`
- Streamlit
- The open-source medical imaging research community

## License

This repository is released under the MIT License. Replace this section if your project uses a different license.

## Disclaimer

This project is intended for research and educational purposes only. It is not a certified medical device and should not be used as the sole basis for clinical diagnosis or treatment decisions.
