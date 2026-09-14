# NestedTumorContour

Code for learning pairwise tumor-contour rankings and generating nested contour hierarchies from multimodal brain MRI.

- [Ranking Model](Ranking%20Model/): contour comparator, pairwise training, and validation-based checkpoint selection.
- [Nested Inference](Nested%20Inference/): ranking-guided contour shrinkage, multiple initializations, and aggregation into voxelwise ordinal maps.
- [PredictGBM preprocessing](PredictGBM_main/README.md): MRI registration, skull stripping, and tumor/tissue segmentation. Existing contour utilities are in [Preprocess](Preprocess/).

## Setup

Use Python 3.10 or newer. A CUDA GPU is recommended for training and inference.

```bash
git clone https://github.com/lyf7374/NestedTumorContour.git
cd NestedTumorContour
pip install torch numpy scipy h5py nibabel tqdm
```

The dependencies above cover ranking and inference. Preprocessing has its own installation instructions in [PredictGBM_main](PredictGBM_main/README.md). Training data and model checkpoints are supplied separately.

## Ranking Model

The default comparator is a point Transformer with 6 layers, 8 attention heads, and a 128-dimensional embedding. Each contour contains 4096 ordered points with features `[X, Y, Z, FLAIR, T1, T1CE, T2]`. Coordinates are voxel coordinates; MRI channels use 1st–99th percentile normalization. The model learns pairwise preferences using binary cross-entropy on logits.

### Training input

Place prepared trajectory `.h5` files in one directory. Each file represents one patient and one contour initialization:

| HDF5 field | Contents |
| --- | --- |
| Root attribute `patient_id` | Stable patient identifier shared by all files from that patient |
| `img/flair_data`, `img/t1`, `img/t1gd`, `img/t2` | Aligned MRI arrays, each shaped `(X, Y, Z)` |
| `shrink` | Inner contour pool, shaped `(S, 4096, 3)` |
| `expand` | Outer contour pool, shaped `(S, 4096, 3)` |
| `points_full` | Ordered intermediate contours, shaped `(L, 4096, 3)`; larger indices have higher target rank |

The three contour pools may have different lengths. Training samples inner–outer, inner–intermediate, intermediate–outer, and ordered intermediate–intermediate pairs. Train/validation splitting groups files by patient, with 20% of patients reserved for validation by default.

```bash
python "Ranking Model/train.py" data/training_h5 \
  --from-scratch \
  --run-name contour_rank \
  --output-dir runs/ranking
```

To initialize from an existing checkpoint, replace `--from-scratch` with `--pretrained path/to/checkpoint.pth`. Use `--split-json path/to/split.json` to reuse a patient split. Checkpoints, the split, training settings, and metrics are saved under `runs/ranking/transformer/contour_rank/`.

Select a checkpoint using fixed validation pairs:

```bash
python "Ranking Model/evaluate.py" data/training_h5 \
  --ckpt-dir runs/ranking/transformer/contour_rank
```

This reuses the saved validation split and exports `transformer_ft_selected_by_fixed_val.pth`, together with validation metrics.

## Nested Inference

Inference uses the trained Transformer comparator to rank live contours and accept tissue-guided shrinkage proposals subject to a tumor-core boundary. Defaults are 5 live contours, 1000 removed contours per run, and 10 initializations. Supply a checkpoint trained with the default Transformer architecture shown above.

### Prepared HDF5 input

Place one `.h5` file per case in an input directory. The filename stem identifies the output case.

| HDF5 field | Contents |
| --- | --- |
| `img/flair_data`, `img/t1`, `img/t1gd`, `img/t2` | Aligned MRI arrays in `(X, Y, Z)` order |
| `img/affines/flair_data` | Shared `4 × 4` voxel-to-world affine |
| `seg_data` | Integer tumor segmentation labels |
| `wm_pbmap`, `gm_pbmap` | White/gray matter probabilities in `[0, 1]` |
| `brain_mask` | Nonempty brain mask |
| `expand` | Reference contours shaped `(S, 4096, 3)` on an ordered `64 × 64` radial grid; the last contour supplies reference directions |
| `center_xyz` | Optional contour center, shaped `(3,)`, in voxel coordinates |

All volumes must share the same shape and spatial alignment. Reference contour coordinates must be finite and inside the volume.

```bash
python "Nested Inference/run_h5.py" \
  --h5-dir data/inference_h5 \
  --ckpt runs/ranking/transformer/contour_rank/transformer_ft_selected_by_fixed_val.pth \
  --out-dir outputs/nested \
  --label-profile brats_124
```

`brats_124` means necrotic core = 1, edema = 2, and enhancing tumor = 4. Choose the profile matching the input labels; available profiles are listed by `--help`. Use `--case-id CASE_ID` to run one HDF5 case.

### Aligned NIfTI input

The NIfTI entry point constructs the reference contour and a temporary HDF5 input. Place these files in one patient directory:

| Input | Default filename |
| --- | --- |
| FLAIR | `flair_bet_normalized.nii.gz` |
| T1 | `t1_bet_normalized.nii.gz` |
| T1CE | `t1ce_bet_normalized.nii.gz` |
| T2 | `t2_bet_normalized.nii.gz` |
| Tumor segmentation | `tumor_seg.nii.gz` |
| White matter probability | `wm_pbmap.nii.gz` |
| Gray matter probability | `gm_pbmap.nii.gz` |
| Brain mask | `brain_mask.nii.gz` |

All inputs must be finite 3D images with matching shapes and affines. Individual paths can be supplied with `--flair`, `--t1`, `--t1ce`, `--t2`, `--seg`, `--wm`, `--gm`, and `--brain-mask`.

```bash
python "Nested Inference/run_nifti.py" \
  --patient-dir data/Patient-001 \
  --pid Patient-001 \
  --ckpt runs/ranking/transformer/contour_rank/transformer_ft_selected_by_fixed_val.pth \
  --out-dir outputs/nested \
  --label-profile brats_124
```

Add `--validate-only` to check the NIfTI inputs without running inference. Both entry points accept search options such as `--K`, `--target-removed`, `--multi-init-runs`, and `--device`.

### Outputs

Each case directory contains per-initialization contours, search histories, and run status. `aggregate_multi_init/` contains the mean and standard deviation of the voxelwise ordinal maps in `.npy` and `.nii.gz` formats. These values describe within-patient ordering; they are not calibrated infiltration probabilities.

## Preprocessing attribution

The preprocessing component is adapted from [BrainLesion/PredictGBM](https://github.com/BrainLesion/PredictGBM). Its documentation, acknowledgments, and [Apache 2.0 license](PredictGBM_main/LICENCE) are retained in `PredictGBM_main/`.
