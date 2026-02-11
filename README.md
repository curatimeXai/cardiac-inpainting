# Cardiac Inpainting: Cardiomegaly to Healthy Heart Transformation

Transform chest X-rays with **cardiomegaly** (enlarged heart) into realistic X-rays showing a **healthy heart** using Stable Diffusion inpainting with LoRA fine-tuning.

## Overview

This project implements a **counterfactual medical image generation** system that transforms chest X-ray images with cardiomegaly into anatomically plausible healthy versions. The transformation is minimal: only the heart region changes while preserving all other anatomical structures.

**Important**: This is a research project and should not be used for clinical diagnosis or treatment decisions.

## Pipeline Components

1. **Segmenter**: CheXMask HybridGNet for cardiac and lung segmentation
2. **Inpainter**: Stable Diffusion Inpainting model fine-tuned with LoRA on healthy chest X-rays
3. **Classifier**: DenseNet121 binary classifier (Cardiomegaly vs Healthy)
4. **Validator**: Anatomical validation using Cardiothoracic Ratio (CTR) measurements

```
Input X-ray (Cardiomegaly)
         │
         ▼
┌─────────────────┐
│   Segmenter     │  CheXMask HybridGNet
│  (Heart Mask)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Inpainter     │  Stable Diffusion + LoRA
│  (Generate)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validation    │  CTR Check + Classifier
│                 │
└────────┬────────┘
         │
         ▼
Output X-ray (Healthy)
```

## Key Concept: Cardiothoracic Ratio (CTR)

The **Cardiothoracic Ratio (CTR)** is a clinical measurement used to assess heart size:

```
CTR = Heart Width / Chest Width
```

- **CTR < 0.5**: Healthy heart (normal size)
- **CTR >= 0.5**: Cardiomegaly (enlarged heart)

The system transforms images with CTR >= 0.5 to images with CTR < 0.5, validated through automatic CTR calculation using segmented heart and lung regions.

## Training Strategy

The inpainting model is trained **exclusively on healthy chest X-rays** with dilated masks:

1. Take a healthy chest X-ray
2. Generate the heart mask using CheXMask segmentation
3. Dilate the mask to simulate a larger (cardiomegaly-sized) region
4. Train the model to reconstruct the original healthy heart within this enlarged mask

This approach teaches the model to fill oversized masks with correctly-sized healthy hearts, without requiring paired cardiomegaly-to-healthy data.

## Dataset

This project uses the [NIH ChestX-ray14 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC).

### Dataset Filtering

From the full NIH dataset (112,120 images), we filtered:

| Stage | Count | Notes |
|-------|-------|-------|
| Total NIH images | 112,120 | Full ChestX-ray14 dataset |
| After PA view filter | ~67,000 | Removed lateral/AP views |
| Cardiomegaly labeled | ~2,772 | Images with "Cardiomegaly" finding |
| No Finding (healthy) | ~60,361 | Images labeled "No Finding" |
| Final balanced set | 1,252 | 626 per class |

### Data Split

| Split | Images | Purpose |
|-------|--------|---------|
| Training | 500 (healthy only) | Train inpainting model |
| Validation | 126 (healthy only) | Monitor training |
| Test | 626 (cardiomegaly) | Evaluate transformation |

## Model Configuration

### LoRA Settings

```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - "to_q"
    - "to_v"
    - "to_k"
    - "to_out.0"
```

### Validation Thresholds

```yaml
validation:
  min_ctr: 0.35
  max_ctr: 0.50
  min_healthy_confidence: 0.80
```

## Project Structure

```
cardiac-inpainting/
├── configs/
│   ├── default.yaml
│   ├── training.yaml
│   └── inference.yaml
├── data/
│   ├── raw/
│   │   ├── cardiomegaly/
│   │   └── healthy/
│   ├── masks/
│   └── processed/
├── models/
│   ├── CheXmask-Database/
│   ├── classifier/
│   └── inpainting/
├── outputs/
├── scripts/
│   ├── prepare_data.py
│   ├── generate_masks.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── inference/
│   └── validation/
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU with 8GB+ VRAM
- ~50GB storage for dataset and models

### Installation

```bash
git clone https://github.com/your-repo/cardiac-inpainting.git
cd cardiac-inpainting

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

pip install -r requirements.txt
pip install -e .
```

### Data Preparation

1. **Download NIH ChestX-ray14 Dataset** from [NIH website](https://nihcc.app.box.com/v/ChestXray-NIHCC)

2. **Filter the dataset**
   ```bash
   python scripts/prepare_data.py \
       --source-dir /path/to/nih-chest-xrays \
       --output-dir data/raw \
       --view-position PA \
       --conditions "No Finding" "Cardiomegaly"
   ```

3. **Generate heart masks**
   ```bash
   python scripts/generate_masks.py \
       --input-dir data/raw \
       --output-dir data/masks
   ```

### Model Setup

1. **CheXMask**: Download from [PhysioNet](https://physionet.org/content/chexmask-cxr-segmentation-data/0.4/) and place in `models/CheXmask-Database/`

2. **Classifier**: Place trained DenseNet121 weights in `models/classifier/dataset_a_classifier.pt`

3. **Stable Diffusion**: Downloaded automatically from HuggingFace on first run

### Training

```bash
python scripts/train.py \
    --config configs/training.yaml \
    --data-dir data/processed \
    --output-dir models/inpainting \
    --epochs 100
```

### Inference

**Single image:**
```bash
python scripts/inference.py \
    --input /path/to/cardiomegaly_xray.png \
    --output /path/to/output.png \
    --checkpoint models/inpainting/final_model
```

**Batch processing:**
```bash
python scripts/inference.py \
    --input data/raw/cardiomegaly/ \
    --output outputs/generated/ \
    --checkpoint models/inpainting/final_model \
    --save-comparison
```

### Classifier Calibration (Recommended)

The classifier supports temperature scaling for better calibrated confidence scores:

```bash
python scripts/calibrate_classifier_temperature.py --device cuda
```

The calibration is saved to `outputs/classifier/dataset_a_calibration.json` and loaded automatically during inference.

## Results

Evaluation on 249 cardiomegaly test images:

| Metric | Value |
|--------|-------|
| Success Rate | 75.1% (187/249) |
| Mean CTR (input) | 0.54 |
| Mean CTR (output) | 0.45 |
| Mean SSIM (outside mask) | 0.92 |
| Mean Healthy Confidence | 95% |

## Limitations

- This is a research project, not a clinical tool
- Success rate drops for severe cardiomegaly (CTR > 0.60)
- Models trained for too many epochs may produce artifacts (dark lines across heart region)
- 25% of images fail validation and cannot be transformed

## License

MIT License

## References

- Wang et al. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database. CVPR.
- Gaggion et al. (2023). CheXMask Database. PhysioNet.
- Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv.
