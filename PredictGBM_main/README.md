# PredictGBM

[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A unified framework for **glioblastoma (GBM) MRI preprocessing and tumor growth model benchmarking**. PredictGBM provides an end-to-end pipeline from raw MRI data to atlas-registered, skull-stripped, tumor-segmented outputs ready for downstream growth modeling and recurrence prediction.

> This repository is adapted from [BrainLesion/PredictGBM](https://github.com/BrainLesion/PredictGBM), with modifications including **nnU-Net v2** for tumor segmentation and **custom atlas template** support.

## Features

- **Multi-modal MRI preprocessing**: Handles T1, T1CE, T2, and FLAIR modalities
- **Atlas co-registration**: ANTsPy-based registration to standard brain atlas (SRI-24 / MNI152 / custom template) via [brainles-preprocessing](https://github.com/BrainLesion/preprocessing)
- **Skull stripping**: HD-BET integration through brainles-preprocessing
- **Tumor segmentation**: nnU-Net v2 replacing the original BraTS toolkit for improved segmentation accuracy
- **Tissue segmentation**: Atlas-based GM/WM/CSF segmentation via ANTs SyN registration
- **Longitudinal registration**: SyN registration between pre-operative and follow-up exams for recurrence tracking
- **Visualization**: Multi-slice axial/coronal PDF reports with tumor overlay

## Pipeline Overview

```
Raw MRI (DICOM/NIfTI)
    │
    ▼
┌─────────────────────────────┐
│  1. Normalization            │  Percentile normalization (0.1–99.9%)
│  2. Skull Stripping          │  HD-BET (optional)
│  3. Co-registration          │  ANTsPy affine → atlas space
│     ├─ Modality alignment    │  T1/T2/FLAIR → T1CE space
│     └─ Atlas registration    │  T1CE → standard atlas
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  4. Tumor Segmentation       │  nnU-Net v2 (3D full-res)
│     ├─ Enhancing tumor       │  Label 3
│     ├─ Peritumoral edema     │  Label 2
│     └─ Necrotic core         │  Label 1
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  5. Tissue Segmentation      │  Atlas-based ANTs SyN
│     ├─ Gray matter (GM)      │
│     ├─ White matter (WM)     │
│     └─ CSF                   │
└─────────────────────────────┘
    │
    ▼
  Registered & segmented outputs
```

## Project Structure

```
PredictGBM_main/
├── predict_gbm/
│   ├── __init__.py                  # Package entry (exports NiftiProcessor)
│   ├── base.py                      # BasePipe abstract class
│   ├── pipeline.py                  # NiftiProcessor — main pipeline orchestrator
│   ├── preprocessing/
│   │   ├── preprocess.py            # BasePreprocessor, NiftiPreprocessor, RegisterRecurrencePipe
│   │   ├── norm_ss_coregistration.py # Normalization, skull stripping, co-registration
│   │   ├── tumor_segmentation.py    # nnU-Net v2 tumor segmentation
│   │   └── tissue_segmentation.py   # Atlas-based tissue segmentation
│   ├── utils/
│   │   ├── constants.py             # Path schemas, atlas paths, label definitions
│   │   ├── utils.py                 # I/O helpers, symlink, PDF merge
│   │   ├── visualization.py         # Multi-slice grid plotting
│   │   └── parsing.py               # PatientDataset JSON parser
│   └── data/
│       ├── mni152_atlas/            # MNI-152 atlas (T1, tissues, probability maps)
│       ├── sri24_atlas/             # SRI-24 atlas
│       ├── datasets/                # Dataset JSON configs
│       └── models/                  # Growth model archives
├── templates/                       # Custom atlas templates (e.g., T1_brain.nii)
├── pyproject.toml                   # Poetry build config & dependencies
├── LICENCE                          # Apache 2.0
└── CONTRIBUTING.md
```

## Registration Transformation Matrices

During preprocessing, ANTsPy generates `.mat` affine transformation files saved under `skull_stripped/`:

| File | Description | Transform Direction |
|------|-------------|---------------------|
| `1_M_co_t1ce__{modality}.mat` | Modality co-registration | T1/T2/FLAIR → T1CE space |
| `2_M_atlas__t1ce.mat` | Atlas registration | T1CE → atlas (standard) space |
| `3_M_atlas_corrected__t1ce__{modality}.mat` | Combined transform (1 + 2) | T1/T2/FLAIR → atlas space (one step) |

These matrices can be reused to transform new images into the registered space:

```python
import ants

template = ants.image_read("templates/T1_brain.nii")
new_img = ants.image_read("new_image.nii.gz")

# Transform to atlas space using saved matrix
warped = ants.apply_transforms(
    fixed=template,
    moving=new_img,
    transformlist=["skull_stripped/t1ce/2_M_atlas__t1ce.mat"],
    interpolator="linear",  # use "nearestNeighbor" for segmentation masks
)
```

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-compatible GPU (recommended for nnU-Net inference)
- [Poetry](https://python-poetry.org/) (for dependency management)

### Install via Poetry

```bash
git clone https://github.com/lyf7374/NestedTumorContour/PredictGBM.git
cd PredictGBM
poetry install
```

### Install via pip

```bash
pip install -e .
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| [brainles-preprocessing](https://github.com/BrainLesion/preprocessing) | Normalization, skull stripping, ANTs co-registration |
| [ANTsPy](https://github.com/ANTsX/ANTsPy) | Image registration & spatial transforms |
| [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) | Tumor segmentation |
| [brats](https://github.com/BrainLesion/BraTS) | Data handling utilities |
| [nibabel](https://nipy.org/nibabel/) | NIfTI I/O |
| [loguru](https://github.com/Delgan/loguru) | Logging |

## Usage

### Quick Start — Single Patient (NIfTI)

```python
from pathlib import Path
from predict_gbm import NiftiProcessor

processor = NiftiProcessor(
    patient_id="Patient-001",
    model_id="test_model",
    t1_preop_file=Path("data/Patient-001/t1.nii.gz"),
    t1c_preop_file=Path("data/Patient-001/t1ce.nii.gz"),
    t2_preop_file=Path("data/Patient-001/t2.nii.gz"),
    flair_preop_file=Path("data/Patient-001/flair.nii.gz"),
    template_file=Path("templates/T1_brain.nii"),
    outdir=Path("output/Patient-001"),
    is_coregistered=True,     # perform co-registration
    is_skull_stripped=False,   # skip skull stripping
    cuda_device="0",
)

processor.run()
```

### Output Directory Structure

After running the pipeline, outputs are organized as:

```
output/Patient-001/predict_gbm_2/ses-preop/
├── nifti_conversion/          # Symlinked/converted NIfTI inputs
│   ├── t1.nii.gz
│   ├── t1ce.nii.gz
│   ├── t2.nii.gz
│   └── flair.nii.gz
├── skull_stripped/             # Registered & normalized outputs
│   ├── t1_bet_normalized.nii.gz
│   ├── t1ce_bet_normalized.nii.gz
│   ├── t2_bet_normalized.nii.gz
│   ├── flair_bet_normalized.nii.gz
│   ├── brainles.log
│   ├── flair/                 # Transformation matrices per modality
│   │   ├── 1_M_co_t1ce__flair.mat
│   │   ├── 2_M_atlas__t1ce.mat
│   │   └── 3_M_atlas_corrected__t1ce__flair.mat
│   ├── t1/
│   ├── t1ce/
│   └── t2/
├── tumor_segmentation/        # nnU-Net outputs
│   ├── tumor_seg.nii.gz
│   ├── enhancing_non_enhancing_tumor.nii.gz
│   └── peritumoral_edema.nii.gz
└── tissue_segmentation/       # Atlas-based tissue maps
    ├── tissue_seg.nii.gz
    ├── csf_pbmap.nii.gz
    ├── gm_pbmap.nii.gz
    └── wm_pbmap.nii.gz
```

### Using a Dataset

```python
from predict_gbm.utils.parsing import PatientDataset

dataset = PatientDataset(dataset_id="my_dataset")
dataset.load("predict_gbm/data/datasets/predict_gbm.json")

for patient in dataset:
    print(patient["patient_id"])
    for exam in patient.get("exams", []):
        print(f"  {exam['timepoint']}: {exam.get('t1c')}")
```

### Standalone Components

Individual preprocessing steps can be used independently:

```python
from predict_gbm.preprocessing import (
    norm_ss_coregister,
    run_brats,
    run_tissue_seg_registration,
)

# Step 1: Normalization + skull stripping + co-registration
norm_ss_coregister(
    t1_file="t1.nii.gz",
    t1ce_file="t1ce.nii.gz",
    t2_file="t2.nii.gz",
    flair_file="flair.nii.gz",
    template_file="templates/T1_brain.nii",
    outdir="output/",
    skull_strip=True,
)

# Step 2: Tumor segmentation via nnU-Net v2
run_brats(
    t1_file="output/skull_stripped/t1_bet_normalized.nii.gz",
    t1ce_file="output/skull_stripped/t1ce_bet_normalized.nii.gz",
    t2_file="output/skull_stripped/t2_bet_normalized.nii.gz",
    flair_file="output/skull_stripped/flair_bet_normalized.nii.gz",
    outdir="output/",
)

# Step 3: Tissue segmentation
run_tissue_seg_registration(
    t1_file="output/skull_stripped/t1ce_bet_normalized.nii.gz",
    outdir="output/",
)
```

## Data

### Atlases

Two brain atlases are included:

- **MNI-152** (`predict_gbm/data/mni152_atlas/`): T1 template, tissue segmentation, tissue probability maps (GM/WM/CSF)
- **SRI-24** (`predict_gbm/data/sri24_atlas/`): T1 stripped, tissue segmentation, tissue probability maps

### Preprocessed Dataset

Preprocessed GBM data is available on [HuggingFace](https://huggingface.co/datasets/LZimmer/PREDICT-GBM).

### Growth Models

Dockerized growth models are available on [HuggingFace](https://huggingface.co/LZimmer/PREDICT-GBM-Models). Place `.tar` files in `predict_gbm/data/models/` to use them.

## Modifications from Original

This fork includes the following changes from [BrainLesion/PredictGBM](https://github.com/BrainLesion/PredictGBM):

| Change | Description |
|--------|-------------|
| Tumor segmentation | Replaced BraTS toolkit with **nnU-Net v2** (`nnUNetPredictor`) |
| Atlas template | Added support for **custom atlas templates** (`template_file` parameter) |
| Pipeline simplification | Streamlined for single-timepoint preprocessing use cases |

## License

This project is licensed under the [Apache License 2.0](LICENCE).

Original work copyright (c) Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany.

## Citation

If you use PredictGBM in your research, please cite:

```bibtex
@software{predictgbm,
  title={PredictGBM: A Framework for Glioblastoma MRI Preprocessing and Growth Model Benchmarking},
  author={Zimmer, Lucas and contributors},
  url={https://github.com/BrainLesion/PredictGBM},
  license={Apache-2.0}
}
```

## Acknowledgments

- [BrainLesion](https://github.com/BrainLesion) for the original PredictGBM framework and brainles-preprocessing
- [ANTsPy](https://github.com/ANTsX/ANTsPy) for image registration
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for tumor segmentation
- [HD-BET](https://github.com/MIC-DKFZ/HD-BET) for brain extraction
