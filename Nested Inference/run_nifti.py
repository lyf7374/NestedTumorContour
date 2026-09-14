"""Run contour search on aligned NIfTI inputs through a temporary HDF5 adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import tempfile

import h5py
import nibabel as nib
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "Nested Inference"

from .geometry import generate_initial_contours, zooms_from_affine
from . import run_h5


NII_DEFAULTS = {
    "flair": "flair_bet_normalized.nii.gz",
    "t1": "t1_bet_normalized.nii.gz",
    "t1ce": "t1ce_bet_normalized.nii.gz",
    "t2": "t2_bet_normalized.nii.gz",
    "seg": "tumor_seg.nii.gz",
    "wm": "wm_pbmap.nii.gz",
    "gm": "gm_pbmap.nii.gz",
    "brain_mask": "brain_mask.nii.gz",
}


def write_adapter(args, h5_path):
    arrays, input_paths, affines = {}, {}, {}
    for key, filename in NII_DEFAULTS.items():
        path = (getattr(args, key) or args.patient_dir / filename).expanduser().resolve()
        image = nib.load(str(path))
        data = image.get_fdata(dtype=np.float32)
        if data.ndim != 3 or not np.isfinite(data).all():
            raise ValueError(f"{key}: expected a finite 3D NIfTI image")
        arrays[key] = data
        affines[key] = np.asarray(image.affine, dtype=np.float32)
        input_paths[key] = str(path)
    shape = arrays["flair"].shape
    affine = affines["flair"]
    for key, data in arrays.items():
        if data.shape != shape or not np.allclose(affines[key], affine, atol=1e-3, rtol=0):
            raise ValueError(f"{key}: shape/affine differs from FLAIR; preprocess into one space first")
    if not np.allclose(arrays["seg"], np.rint(arrays["seg"]), atol=1e-4):
        raise ValueError("seg must contain integer labels (use nearest-neighbor resampling)")
    seg = np.rint(arrays["seg"]).astype(np.int32)
    brain = (arrays["brain_mask"] > 1e-3).astype(np.uint8)
    if not brain.any():
        raise ValueError("brain_mask is empty")
    core_labels, _ = run_h5.resolve_role_labels_compat(
        seg, role="constraint_core", label_profile=args.label_profile,
        label_spec=args.constraint_labels,
    )
    base_labels = run_h5._labels_for_role_compat(args.label_profile, "wt")
    if not np.any(np.isin(seg, core_labels) & (brain > 0)):
        raise ValueError("Tumor core is empty inside brain_mask; check --label-profile")
    for key in ("wm", "gm"):
        if arrays[key].min() < -1e-5 or arrays[key].max() > 1 + 1e-5:
            raise ValueError(f"{key}: tissue maps must be probabilities in [0,1]")
    zooms = zooms_from_affine(affine)
    # The 15-mm reference establishes directions only. Multi-init reconstructs
    # actual live contours from ED union core plus its separate 5-mm margin.
    initial, center, _ = generate_initial_contours(
        seg_xyz=seg, brain_mask_xyz=brain, zooms_xyz=zooms, K=1,
        ctv_margin_mm=15, init_shrink_mm=1, n_theta=64, n_phi=64,
        ray_step_mm=0.5, base_labels=base_labels, center_labels=core_labels,
    )
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["input_adapter"] = "aligned_nifti"
        images = handle.create_group("img")
        affs = images.create_group("affines")
        for h5_key, key in (("flair_data", "flair"), ("t1", "t1"), ("t1gd", "t1ce"), ("t2", "t2")):
            images.create_dataset(h5_key, data=arrays[key], compression="gzip", compression_opts=1)
            affs.create_dataset(h5_key, data=affine)
        for key, data in (("seg_data", seg), ("wm_pbmap", arrays["wm"] * brain),
                          ("gm_pbmap", arrays["gm"] * brain), ("brain_mask", brain)):
            handle.create_dataset(key, data=data, compression="gzip", compression_opts=1)
        handle.create_dataset("center_xyz", data=center)
        handle.create_dataset("expand", data=initial)
    return {
        "input_mode": "aligned_nifti", "input_paths": input_paths,
        "shape_xyz": list(shape), "affine": affine.tolist(),
        "center_xyz": center.tolist(), "constraint_labels": core_labels,
        "label_profile": args.label_profile,
        "reference_grid": [64, 64], "reference_margin_mm": 15,
        "reference_note": "Reference directions only; production live contours rebuilt by run_h5",
        "temporary_adapter_deleted_after_run": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Search options (e.g. --K, --target-removed, --multi-init-runs, --max-attempts) "
               "are forwarded to run_h5.py; run its --help for the full list.",
    )
    parser.add_argument("--patient-dir", type=Path, required=True)
    parser.add_argument("--pid", required=True, help="Non-identifying output case identifier")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label-profile", choices=list(run_h5.LABEL_PROFILE_DEFS_COMPAT), required=True)
    parser.add_argument("--constraint-labels", default="")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs and temporary adapter without inference")
    for key in NII_DEFAULTS:
        parser.add_argument("--" + key.replace("_", "-"), dest=key, type=Path)
    args, search_args = parser.parse_known_args(argv)
    if Path(args.pid).name != args.pid or args.pid in ("", ".", ".."):
        raise ValueError("--pid must be a single directory name")
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    # Validate all forwarded options before reading data or creating output.
    forwarded = ["--h5-dir", ".", "--ckpt", str(args.ckpt), "--out-dir", str(args.out_dir),
                 "--label-profile", args.label_profile, "--constraint-labels", args.constraint_labels,
                 "--case-id", args.pid] + search_args
    parsed_search = run_h5.parse_args(forwarded)
    case_dir = args.out_dir / args.pid
    if not args.validate_only and case_dir.exists() and any(case_dir.iterdir()) and not parsed_search.overwrite:
        raise FileExistsError(f"Output exists: {case_dir}; choose another --out-dir or use --overwrite")
    with tempfile.TemporaryDirectory(prefix="contourrank_nifti_") as temp_dir:
        provenance = write_adapter(args, Path(temp_dir) / f"{args.pid}.h5")
        if args.validate_only:
            print(json.dumps({"status": "validated", **provenance}, indent=2))
            return
        forwarded[1] = temp_dir
        try:
            run_h5.main(forwarded)
        finally:
            if case_dir.exists():
                provenance["checkpoint"] = str(args.ckpt.resolve())
                digest = hashlib.sha256()
                with args.ckpt.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                provenance["checkpoint_sha256"] = digest.hexdigest()
                (case_dir / "nifti_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
