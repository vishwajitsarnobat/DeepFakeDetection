# LAA-Net: Localized Artifact Attention Network for Deepfake Detection (Implementation)

This repository contains an implementation and evaluation scripts for the LAA-Net architecture, designed for robust and generalizable deepfake detection, as presented in the paper:

**LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection**  
*Dat Nguyen, Nesryne Mejri, Inder Pal Singh, Polina Kuleshova, Marcella Astrid, Anis Kacem, Enjie Ghorbel, Djamila Aouada*  
*CVPR 2024*  
[[Paper](https://arxiv.org/pdf/2401.13856.pdf)]

---

## Table of Contents

- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
  - [Explicit Attention via Multi-Task Learning](#explicit-attention-via-multi-task-learning)
  - [Enhanced Feature Pyramid Network (E-FPN)](#enhanced-feature-pyramid-network-e-fpn)
  - [Pseudo-Fake Generation (SBI)](#pseudo-fake-generation-sbi)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
  - [1. Download Dataset](#1-download-dataset)
  - [2. Extract Frames](#2-extract-frames)
  - [3. Download Landmark Predictor](#3-download-landmark-predictor)
  - [4. Extract Landmarks & Generate Metadata](#4-extract-landmarks--generate-metadata)
- [Pre-trained Model](#pre-trained-model)
- [Evaluation](#evaluation)
  - [Running Evaluation](#running-evaluation)
  - [Limiting Evaluation Data](#limiting-evaluation-data)
  - [Understanding Results](#understanding-results)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Introduction

Deepfake detection faces challenges with high-quality fakes containing only subtle, localized artifacts. Standard classifiers often struggle with generalization to unseen manipulation types. LAA-Net addresses this by introducing an explicit attention mechanism combined with a novel feature pyramid network (E-FPN) to focus on artifact-prone regions and effectively utilize multi-scale features. This implementation allows for evaluating pre-trained LAA-Net models on various datasets, particularly FaceForensics++.

## Core Concepts

### Explicit Attention via Multi-Task Learning

Instead of relying only on implicit attention from a standard classifier, LAA-Net employs a multi-task framework:

1. **Classification Branch:** Predicts if the input is Real or Fake.
2. **Heatmap Regression Branch:** Explicitly localizes "vulnerable points" likely to contain blending artifacts by predicting a heatmap.
3. **Self-Consistency Branch:** Predicts the consistency between a random vulnerable point and other pixels within the manipulated region.

This multi-task approach forces the network to learn and attend to localized artifact regions.

### Enhanced Feature Pyramid Network (E-FPN)

Standard FPNs aggregate features across scales. E-FPN improves this by:

1. Processing higher-level features (convolution + transpose convolution) and upsampling them (`E(l)`).
2. Using these processed features to *filter* the corresponding lower-level features (`F(l)`): `(1 - sigmoid(E(l))) * F(l)`.
3. Concatenating the filtered lower-level features with the processed higher-level features `E(l)`.

This mechanism aims to suppress redundant information while emphasizing unique low-level details crucial for detecting subtle artifacts.

### Pseudo-Fake Generation (SBI)

The LAA-Net paper primarily uses **Self-Blended Images (SBI)** for training. This involves generating pseudo-fakes *on-the-fly* during training using only real images.
- A blending mask (`M`) is generated.
- "Vulnerable points" (`P`) are identified from the mask (pixels with balanced foreground/background contribution).
- Ground truth for the Heatmap and Self-Consistency branches are derived from `M` and `P`.

**(Note:** While this implementation focuses on evaluation, understanding the SBI training process is key to the model's generalization capabilities.)

---

## Environment Setup

We recommend setting up a dedicated Python environment (e.g., using Conda or venv).

**Required Libraries:**

- Python >= 3.8  
- PyTorch >= 1.8.0 (Ensure CUDA compatibility if using GPU)  
- Torchvision >= 0.9.0  
- NumPy  
- OpenCV (cv2) (`opencv-python`)  
- dlib (For landmark extraction)  
- imutils  
- simplejson  
- PyYAML (`pip install pyyaml`)  
- python-box (`pip install python-box`)  
- tqdm  
- scikit-learn (`sklearn`)  
- matplotlib  
- (Optional but recommended for training/full reproduction) tensorboardX, imgaug, scikit-image, albumentations  

**Installation Example (using pip):**

```bash
# Create and activate environment (example using venv)
# python3 -m venv .venv
# source .venv/bin/activate

# Install core libraries (adjust torch version based on your CUDA setup)
pip install torch torchvision torchaudio 
pip install numpy opencv-python dlib imutils simplejson pyyaml python-box tqdm scikit-learn matplotlib
```

---

## Data Preparation

This guide assumes you are using the FaceForensics++ (FF++) dataset. Steps for other datasets might vary.

### 1. Download Dataset

Download the FaceForensics++ dataset, including the original_sequences, and manipulated videos (Deepfakes, FaceSwap, etc.) at your desired compression level (e.g., c40). Organize them according to a structure like the one shown below (or adapt paths in config files).

Example base path: `/path/to/your/datasets/FaceForensics++/`

### 2. Extract Frames

Extract frames from the downloaded videos. Standard tools like ffmpeg can be used. Ensure frames are saved in a structured way, preserving the split (train/val/test), manipulation type (Original/Deepfakes/...), and video ID.

**Target Structure (Example for c40):**

```
/path/to/your/datasets/FaceForensics++/processed/  # This will be your ROOT in configs
└── test/
    └── frames/
        ├── Deepfakes/
        │   ├── 000/
        │   │   ├── 0000.png
        │   │   └── ...
        │   ├── 001/
        │   └── ...
        ├── FaceSwap/
        │   ├── 000/
        │   └── ...
        └── Original/
            ├── 000/
            └── ...
```

*(Note: The `images_crop.py` mentioned in the original README likely performs frame extraction and face cropping. If you only need evaluation, ensure frames are extracted correctly into the structure above.)*

### 3. Download Landmark Predictor

Download the 81-point facial landmark predictor model file compatible with dlib.

Download: `shape_predictor_81_face_landmarks.dat` (Search online or use link from original README if available)

Place it in the `pretrained/` directory within this project. Create the directory if it doesn't exist.

### 4. Extract Landmarks & Generate Metadata

Use the provided script to detect landmarks and generate a metadata file. While our evaluation script uses directory scanning, generating this is good practice.

```bash
python package_utils/geo_landmarks_extraction.py \
    --config configs/data_preprocessing_c40.yaml \
    --extract_landmark
```

Verify `configs/data_preprocessing_c40.yaml`:

- Set `ROOT` to your processed data path (e.g., `/path/to/your/datasets/FaceForensics++/processed/`).
- Set `DATA_TYPE` to frames.
- Set `FAKETYPE` to include Original and all fake types present (e.g., `[Deepfakes, FaceSwap, Original]`).
- Ensure `facial_lm_pretrained` points to `pretrained/shape_predictor_81_face_landmarks.dat`.

This creates a JSON file (e.g., `processed_data/c40/test_FaceForensics_Original_Deepfakes_FaceSwap_lm81.json`).

---

## Pre-trained Model

Download the pre-trained LAA-Net model weights (e.g., the SBI variant) from the source provided in the original paper/repository.

Place the `.pth` file in the `pretrained/` directory.

Update `TEST.pretrained` in your configuration file (`configs/efn4_fpn_sbi_adv.yaml` or your copy) to the correct filename. Example:

```yaml
TEST:
  # ...
  pretrained: pretrained/PoseEfficientNet_EFN_hm100_EFPN_NoBasedCLS_Focal_C3_256Cstency100_8SBI_SAM(Adam)_ADV_Era1_OutSigmoid_1e7_boost500_UnFZ_model_best.pth # <- UPDATE THIS
  # ...
```

---

## Evaluation

The `scripts/test.py` script evaluates the model's performance.

### Running Evaluation

Execute the script pointing to your config file:

```bash
python3 scripts/test.py -c configs/efn4_fpn_sbi_adv.yaml
```

**Key Configuration** (`configs/efn4_fpn_sbi_adv.yaml`):

- `DATASET.DATA.TEST.ROOT`: Path to the processed directory (parent of `test/frames/`).
- `DATASET.DATA.TEST.FAKETYPE`: List exactly matching directory names under `test/frames/`.
- `DATASET.DATA.TEST.FROM_FILE`: Must be False.
- `TEST.pretrained`: Path to the downloaded `.pth` model file.
- `TEST.batch_size`: Adjust based on GPU memory. Start low (e.g., 4, 2, or 1) if you encounter CUDA Out-of-Memory errors.

### Limiting Evaluation Data

Use the `--limit_percent` argument for faster runs on a subset of the data:

```bash
# Evaluate on 10% of the test set (randomly sampled)
python3 scripts/test.py -c configs/efn4_fpn_sbi_adv.yaml --limit_percent 10.0

# Evaluate on 1% of the test set
python3 scripts/test.py -c configs/efn4_fpn_sbi_adv.yaml --limit_percent 1.0
```

### Understanding Results

The script outputs:

- **Setup & Progress Info:** Configuration, seeding, data loading, evaluation progress.
- **Optimal Threshold:** The threshold maximizing F1-score on the evaluated subset.
- **Final Metrics:** Accuracy, AUC, AP, Recall, F1-score calculated using the optimal threshold.

**Saved Files** (in `test_results/` directory):

- `*_preds[_limXX].npy`: Raw prediction scores.
- `*_labels[_limXX].npy`: Ground truth labels.
- `*_CM_optimal[_limXX].png`: Confusion matrix plot (optimal threshold).
- `*_CM_optimal_norm[_limXX].png`: Normalized confusion matrix plot.

Metrics reported at the optimal threshold provide a better view of the model's performance than using a fixed 0.5 threshold, especially if the output scores aren't centered around 0.5.

---

## Contact

Refer to the contact information in the original LAA-Net repository or paper for research-related questions.

---

## Acknowledgements

This implementation builds upon the LAA-Net paper and potentially utilizes code adapted from:

- mmengine  
- BI (Face-Xray)  
- SBI

---

## Citation

If you use this work or the LAA-Net model, please cite the original paper:

```bibtex
@InProceedings{Nguyen_2024_CVPR,
    author    = {Nguyen, Dat and Mejri, Nesryne and Singh, Inder Pal and Kuleshova, Polina and Astrid, Marcella and Kacem, Anis and Ghorbel, Enjie and Aouada, Djamila},
    title     = {LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17395-17405}
}
```

