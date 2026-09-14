"""
Portable multi-initialization contour search, derived from the original research runner.

This script mirrors the notebook logic, iterating over all .h5 files in a
dataset directory, running nested sampling, and saving removed shells and
history per patient.

Constraint variant:
- Enforces a hard inner boundary so candidate contours cannot shrink inside
  pre-op NCR+ET (core+enhancing) voxels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import concurrent.futures as futures
import multiprocessing as mp

import h5py
import numpy as np
import torch
from scipy.ndimage import center_of_mass, distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "Nested Inference"

from .transition_matrix import radial_shrink_points_transition_matrix_v3
from .ordinal import shells_to_prob_volume

_ranking_models = import_module("Ranking Model.models")
PointTransformerComparator = _ranking_models.PointTransformerComparator
load_pretrained = _ranking_models.load_pretrained

try:
    import nibabel as nib  # optional: only needed when exporting aggregated .nii.gz
except Exception:
    nib = None


# -------------------------
# Model and helper functions
# -------------------------
@torch.no_grad()
def load_comparator(ckpt, device, d_model=128, nhead=8, num_layers=6, dim_feedforward=256, dropout=0.2):
    comp = PointTransformerComparator(input_dim=7, d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout).to(device)
    load_pretrained(comp, ckpt, map_location=device, strict=True)
    comp.eval()
    return comp


def robust_normalize_np(u, p1=1, p99=99):
    vals = u[np.isfinite(u)]
    if vals.size == 0:
        return np.zeros_like(u, dtype=np.float32)
    lo, hi = np.percentile(vals, [p1, p99])
    if hi <= lo:
        return np.clip(u, 0, 1).astype(np.float32)
    return np.clip((u - lo) / (hi - lo), 0, 1).astype(np.float32)


def nearest_intensity_xyz(coords_xyz, img_vol_xyz4):
    idx = np.rint(coords_xyz).astype(np.int32)
    X, Y, Z = img_vol_xyz4.shape[1:]
    idx[:, 0] = np.clip(idx[:, 0], 0, X - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, Y - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, Z - 1)
    x, y, z = idx.T
    vals = img_vol_xyz4[:, x, y, z]
    return vals.T.astype(np.float32)


def coords_to_pc_xyz(coords_xyz, img_vol_xyz4):
    ints = nearest_intensity_xyz(coords_xyz, img_vol_xyz4)
    pc = np.concatenate([coords_xyz.astype(np.float32), ints], axis=1)
    return torch.from_numpy(pc)


def center_from_seg_xyz(seg_xyz):
    mask = np.isin(seg_xyz, [1, 4])
    if not np.any(mask):
        mask = np.isin(seg_xyz, [1, 2, 4])
        if not np.any(mask):
            mask = (seg_xyz != 0)
    cx, cy, cz = center_of_mass(mask.astype(np.float32))
    return np.array([cx, cy, cz], dtype=np.float32)


def zooms_from_affine(aff_4x4):
    A = np.asarray(aff_4x4, dtype=np.float64)[:3, :3]
    return np.sqrt((A * A).sum(axis=0)) + 1e-12


def parse_constraint_labels(spec: str) -> List[int]:
    out: List[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def core_et_mask_from_seg(seg_xyz: np.ndarray, constraint_labels: Optional[List[int]] = None) -> Tuple[np.ndarray, List[int]]:
    labels_present = sorted(int(x) for x in np.unique(seg_xyz) if int(x) != 0)
    if constraint_labels:
        labels_used = [int(x) for x in constraint_labels]
    else:
        # Robust NCR+ET mapping across common label conventions.
        et = 4 if 4 in labels_present else (3 if 3 in labels_present else None)
        labels_used = [1] + ([int(et)] if et is not None else [])
    if not labels_used:
        return np.zeros_like(seg_xyz, dtype=bool), []
    return np.isin(seg_xyz, labels_used), labels_used


LABEL_PROFILE_DEFS_COMPAT: Dict[str, Dict[str, List[int]]] = {
    "brats_124": {"ncr": [1], "et": [4], "ed": [2]},
    "rhuh_preop_413": {"ncr": [4], "et": [1], "ed": [3]},
    "rhuh_recurrence_132": {"ncr": [1], "et": [3], "ed": [2]},
    "rec_native_341": {"ncr": [3], "et": [4], "ed": [1]},
    "contoursrec_h5_123": {"ncr": [2], "et": [1], "ed": [3]},
}

LABEL_PROFILE_ALIASES_COMPAT = {
    "auto": "auto",
    "brats": "brats_124",
    "brats_124": "brats_124",
    "124": "brats_124",
    "rhuh_preop": "rhuh_preop_413",
    "rhuh_pre": "rhuh_preop_413",
    "rhuh_preop_413": "rhuh_preop_413",
    "rhuh_recurrence": "rhuh_recurrence_132",
    "rhuh_rec": "rhuh_recurrence_132",
    "rhuh_recurrence_132": "rhuh_recurrence_132",
    "rec": "rec_native_341",
    "rec_native": "rec_native_341",
    "rec_native_341": "rec_native_341",
    "rec_341": "rec_native_341",
    "contoursrec": "contoursrec_h5_123",
    "contoursrec_h5": "contoursrec_h5_123",
    "contoursrec_h5_123": "contoursrec_h5_123",
    "rec_h5": "contoursrec_h5_123",
}


def _canonicalize_label_profile_compat(label_profile: str) -> str:
    key = str(label_profile).strip().lower()
    if key not in LABEL_PROFILE_ALIASES_COMPAT:
        allowed = ", ".join(sorted(k for k in LABEL_PROFILE_ALIASES_COMPAT if k != "auto"))
        raise ValueError(f"Unsupported --label-profile={label_profile!r}. Allowed aliases: {allowed}")
    return LABEL_PROFILE_ALIASES_COMPAT[key]


def _context_hint_blob_compat(path_hints: Optional[Sequence[str]], pid: Optional[str]) -> str:
    parts: List[str] = []
    if path_hints:
        parts.extend(str(x) for x in path_hints if str(x).strip())
    if pid:
        parts.append(str(pid))
    return " | ".join(parts).lower()


def _resolve_label_profile_local(
    seg_xyz: np.ndarray,
    label_profile: str = "auto",
    *,
    path_hints: Optional[Sequence[str]] = None,
    pid: Optional[str] = None,
) -> str:
    profile = _canonicalize_label_profile_compat(label_profile)
    if profile != "auto":
        return profile

    labels_present = {int(x) for x in np.unique(seg_xyz) if int(x) != 0}
    if not labels_present:
        raise ValueError("Cannot infer label profile from an all-zero segmentation.")

    ctx = _context_hint_blob_compat(path_hints, pid)

    if labels_present.issubset({1, 2, 4}):
        return "brats_124"
    if "processedrec_rhuhlabels" in ctx:
        return "brats_124"
    if "contoursrec" in ctx:
        return "contoursrec_h5_123"
    if "processedrec" in ctx:
        return "rec_native_341"
    if "processedrhuh" in ctx:
        if any(tok in ctx for tok in ("/2/", "\\2\\", "ses-followup", "followup", "recurrence")):
            return "rhuh_recurrence_132"
        return "rhuh_preop_413"
    if "contoursrhuh" in ctx:
        return "rhuh_preop_413"

    if pid:
        pid_s = str(pid)
        if pid_s.startswith("RHUH-"):
            if labels_present.issubset({1, 2, 3}) and 4 not in labels_present:
                return "rhuh_recurrence_132"
            return "rhuh_preop_413"
        if pid_s.startswith("Patient-"):
            if labels_present.issubset({1, 2, 3}) and 4 not in labels_present:
                return "contoursrec_h5_123"
            return "rec_native_341"

    if labels_present.issubset({1, 2, 3}) and 4 not in labels_present:
        return "rhuh_recurrence_132"

    raise ValueError(
        "Cannot infer label profile automatically for labels="
        f"{sorted(labels_present)}. Use --label-profile to specify one explicitly."
    )


def _labels_for_role_compat(profile: str, role: str) -> List[int]:
    prof = LABEL_PROFILE_DEFS_COMPAT[profile]
    role_key = str(role).strip().lower()
    if role_key == "ncr":
        labels = prof["ncr"]
    elif role_key == "et":
        labels = prof["et"]
    elif role_key == "ed":
        labels = prof["ed"]
    elif role_key in {"constraint_core", "tcet", "core_et", "ncr_et"}:
        labels = prof["ncr"] + prof["et"]
    elif role_key in {"wt", "whole_tumor"}:
        labels = prof["ncr"] + prof["et"] + prof["ed"]
    else:
        raise ValueError(f"Unsupported tumor role: {role}")
    return sorted({int(x) for x in labels})


def resolve_label_profile_compat(
    seg_xyz: np.ndarray,
    label_profile: str = "auto",
    *,
    path_hints: Optional[Sequence[str]] = None,
    pid: Optional[str] = None,
) -> str:
    return _resolve_label_profile_local(seg_xyz, label_profile=label_profile, path_hints=path_hints, pid=pid)


def resolve_role_labels_compat(
    seg_xyz: np.ndarray,
    *,
    role: str,
    label_spec: str = "",
    label_profile: str = "auto",
    path_hints: Optional[Sequence[str]] = None,
    pid: Optional[str] = None,
) -> Tuple[List[int], str]:
    labels_present = {int(x) for x in np.unique(seg_xyz) if int(x) != 0}
    if str(label_spec).strip():
        labels = [int(x) for x in parse_constraint_labels(label_spec) if int(x) in labels_present]
        return sorted({int(x) for x in labels}), _canonicalize_label_profile_compat(label_profile)

    resolved_profile = resolve_label_profile_compat(
        seg_xyz,
        label_profile=label_profile,
        path_hints=path_hints,
        pid=pid,
    )
    labels = [int(x) for x in _labels_for_role_compat(resolved_profile, role) if int(x) in labels_present]
    return sorted({int(x) for x in labels}), resolved_profile


def resolve_constraint_labels_compat(
    seg_xyz: np.ndarray,
    label_spec: str,
    *,
    label_profile: str = "auto",
    path_hints: Optional[Sequence[str]] = None,
    pid: Optional[str] = None,
) -> List[int]:
    labels, _ = resolve_role_labels_compat(
        seg_xyz, role="constraint_core", label_spec=label_spec,
        label_profile=label_profile, path_hints=path_hints, pid=pid,
    )
    return labels

def _nearest_mask_voxel_compat(mask_xyz: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask_xyz > 0)
    if pts.shape[0] == 0:
        raise RuntimeError("Mask is empty; cannot find nearest voxel.")
    d2 = np.sum((pts.astype(np.float32) - query_xyz[None, :]) ** 2, axis=1)
    return pts[int(np.argmin(d2))].astype(np.float32)


def jitter_center_xyz_within_mask_compat(
    mask_xyz: np.ndarray,
    anchor_xyz: np.ndarray,
    zooms_xyz: np.ndarray,
    rng: np.random.Generator,
    max_jitter_mm: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    mask = mask_xyz > 0
    if not np.any(mask):
        raise RuntimeError("Mask is empty; cannot sample jittered center.")

    anchor = np.asarray(anchor_xyz, dtype=np.float32)
    zooms = np.asarray(zooms_xyz, dtype=np.float32)
    anchor_in_mask = _nearest_mask_voxel_compat(mask, anchor).astype(np.float32)

    if float(max_jitter_mm) <= 0.0:
        shift_mm = float(np.linalg.norm((anchor_in_mask - anchor) * zooms))
        return anchor_in_mask, {
            "anchor_center_xyz": anchor.tolist(),
            "anchor_center_in_mask_xyz": anchor_in_mask.tolist(),
            "selected_center_xyz": anchor_in_mask.tolist(),
            "jitter_limit_mm": float(max_jitter_mm),
            "actual_shift_from_anchor_mm": shift_mm,
            "sampled_from_n_candidates": 1,
        }

    pts = np.argwhere(mask).astype(np.float32)
    d_mm = np.linalg.norm((pts - anchor_in_mask[None, :]) * zooms[None, :], axis=1)
    keep = d_mm <= (float(max_jitter_mm) + 1e-6)
    if not np.any(keep):
        keep = np.zeros_like(d_mm, dtype=bool)
        keep[int(np.argmin(d_mm))] = True

    cand = pts[keep]
    cand_d_mm = d_mm[keep]
    sigma_mm = max(float(max_jitter_mm) * 0.5, 1e-3)
    weights = np.exp(-0.5 * (cand_d_mm / sigma_mm) ** 2).astype(np.float64)
    wsum = float(np.sum(weights))
    if not np.isfinite(wsum) or wsum <= 0.0:
        weights = np.ones((cand.shape[0],), dtype=np.float64) / float(cand.shape[0])
    else:
        weights = weights / wsum

    idx = int(rng.choice(cand.shape[0], p=weights))
    chosen = cand[idx].astype(np.float32)
    shift_mm = float(np.linalg.norm((chosen - anchor) * zooms))
    return chosen, {
        "anchor_center_xyz": anchor.tolist(),
        "anchor_center_in_mask_xyz": anchor_in_mask.tolist(),
        "selected_center_xyz": chosen.tolist(),
        "jitter_limit_mm": float(max_jitter_mm),
        "actual_shift_from_anchor_mm": shift_mm,
        "sampled_from_n_candidates": int(cand.shape[0]),
    }


def build_dirs_vox_per_mm_from_reference(
    ref_coords_xyz: np.ndarray,
    center_xyz: np.ndarray,
    zooms_xyz: np.ndarray,
) -> np.ndarray:
    delta_vox = ref_coords_xyz.astype(np.float32) - center_xyz.astype(np.float32)[None, :]
    delta_mm = delta_vox * zooms_xyz.astype(np.float32)[None, :]
    r_mm = np.linalg.norm(delta_mm, axis=1)
    good = r_mm > 1e-6
    if not np.any(good):
        raise RuntimeError("Cannot build direction basis: all reference radii are ~0.")
    dirs_vox_per_mm = np.zeros_like(delta_vox, dtype=np.float32)
    dirs_vox_per_mm[good] = delta_vox[good] / r_mm[good, None]
    # Fill any degenerate direction with the first valid one.
    first_valid = dirs_vox_per_mm[np.flatnonzero(good)[0]]
    dirs_vox_per_mm[~good] = first_valid[None, :]
    return dirs_vox_per_mm


def build_directions_equirect(n_theta: int = 256, n_phi: int = 128) -> np.ndarray:
    theta_edges = np.linspace(0, 2 * np.pi, int(n_theta) + 1)
    phi_edges = np.linspace(0, np.pi, int(n_phi) + 1)
    theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_c = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    dirs = []
    for th in theta_c:
        for ph in phi_c:
            x = np.sin(ph) * np.cos(th)
            y = np.sin(ph) * np.sin(th)
            z = np.cos(ph)
            v = np.array([x, y, z], dtype=np.float32)
            v /= np.linalg.norm(v) + 1e-8
            dirs.append(v)
    return np.stack(dirs, axis=0)


def ray_rmax_mm(center_xyz: np.ndarray, dir_vox_per_mm: np.ndarray, shape_xyz: Tuple[int, int, int]) -> float:
    c = center_xyz.astype(np.float64)
    d = dir_vox_per_mm.astype(np.float64)
    tmax = []
    for ax in range(3):
        if abs(d[ax]) < 1e-12:
            continue
        if d[ax] > 0:
            t = (shape_xyz[ax] - 1 - c[ax]) / d[ax]
        else:
            t = (0 - c[ax]) / d[ax]
        if t > 0:
            tmax.append(t)
    if not tmax:
        return 0.0
    return float(max(0.0, min(tmax)))


def contour_radii_mm_from_mask(
    mask_xyz: np.ndarray,
    center_xyz: np.ndarray,
    dirs_xyz: np.ndarray,
    zooms_xyz: np.ndarray,
    dr_mm: float = 0.5,
) -> np.ndarray:
    mask_f = mask_xyz.astype(np.float32, copy=False)
    shape = mask_xyz.shape
    radii = np.zeros((dirs_xyz.shape[0],), dtype=np.float32)

    for i, d in enumerate(dirs_xyz):
        d_vox = d / (zooms_xyz + 1e-12)
        rmax = ray_rmax_mm(center_xyz, d_vox, shape)
        if rmax <= 0:
            continue
        rs = np.arange(0.0, rmax + dr_mm, dr_mm, dtype=np.float32)
        pts = center_xyz[None, :] + rs[:, None] * d_vox[None, :]
        vals = map_coordinates(
            mask_f,
            [pts[:, 0], pts[:, 1], pts[:, 2]],
            order=1,
            mode="constant",
            cval=0.0,
        )
        inside = vals >= 0.5
        if not np.any(inside):
            continue
        k = int(np.max(np.where(inside)[0]))
        radii[i] = float(rs[k])
    return radii


def contour_radii_mm_from_mask_with_dirs(
    mask_xyz: np.ndarray,
    center_xyz: np.ndarray,
    dirs_vox_per_mm: np.ndarray,
    dr_mm: float = 0.5,
) -> np.ndarray:
    mask_f = mask_xyz.astype(np.float32, copy=False)
    shape = mask_xyz.shape
    radii = np.zeros((dirs_vox_per_mm.shape[0],), dtype=np.float32)
    for i, d_vox in enumerate(dirs_vox_per_mm):
        rmax = ray_rmax_mm(center_xyz, d_vox, shape)
        if rmax <= 0:
            continue
        rs = np.arange(0.0, rmax + dr_mm, dr_mm, dtype=np.float32)
        pts = center_xyz[None, :] + rs[:, None] * d_vox[None, :]
        vals = map_coordinates(
            mask_f,
            [pts[:, 0], pts[:, 1], pts[:, 2]],
            order=1,
            mode="constant",
            cval=0.0,
        )
        inside = vals >= 0.5
        if not np.any(inside):
            continue
        k = int(np.max(np.where(inside)[0]))
        radii[i] = float(rs[k])
    return radii


def enforce_min_radii_constraint_dense_floor(
    coords_xyz: np.ndarray,
    center_xyz: np.ndarray,
    zooms_xyz: np.ndarray,
    floor_dirs_xyz: np.ndarray,
    floor_radii_mm: np.ndarray,
    floor_tree: cKDTree,
    shape_xyz: Tuple[int, int, int],
    floor_k: int = 3,
    floor_margin_mm: float = 0.0,
) -> Tuple[np.ndarray, int]:
    pts = coords_xyz.astype(np.float32)
    center = center_xyz.astype(np.float32)
    zooms = zooms_xyz.astype(np.float32)

    delta_vox = pts - center[None, :]
    delta_mm = delta_vox * zooms[None, :]
    r_curr_mm = np.linalg.norm(delta_mm, axis=1)
    good = r_curr_mm > 1e-6

    dirs_xyz = np.zeros_like(delta_mm, dtype=np.float32)
    dirs_xyz[good] = delta_mm[good] / r_curr_mm[good, None]
    if np.any(~good):
        dirs_xyz[~good] = floor_dirs_xyz[0][None, :]

    k = max(1, int(floor_k))
    k = min(k, int(floor_dirs_xyz.shape[0]))
    _, idx = floor_tree.query(dirs_xyz.astype(np.float64), k=k)
    if k == 1:
        floor_req = floor_radii_mm[np.asarray(idx).reshape(-1)]
    else:
        idx = np.asarray(idx)
        floor_req = np.max(floor_radii_mm[idx], axis=1)
    floor_req = floor_req.astype(np.float32) + float(floor_margin_mm)

    r_new = np.maximum(r_curr_mm, floor_req)
    n_clamped = int(np.sum(r_new > (r_curr_mm + 1e-6)))

    dirs_vox_per_mm = dirs_xyz / (zooms[None, :] + 1e-12)
    out = center[None, :] + r_new[:, None] * dirs_vox_per_mm
    out[:, 0] = np.clip(out[:, 0], 0, shape_xyz[0] - 1)
    out[:, 1] = np.clip(out[:, 1], 0, shape_xyz[1] - 1)
    out[:, 2] = np.clip(out[:, 2], 0, shape_xyz[2] - 1)
    return out.astype(np.float32), n_clamped


def enforce_min_radii_constraint(
    coords_xyz: np.ndarray,
    center_xyz: np.ndarray,
    dirs_vox_per_mm: np.ndarray,
    min_radii_mm: np.ndarray,
    shape_xyz: Tuple[int, int, int],
) -> Tuple[np.ndarray, int]:
    delta = coords_xyz.astype(np.float32) - center_xyz.astype(np.float32)[None, :]
    denom = np.sum(dirs_vox_per_mm * dirs_vox_per_mm, axis=1) + 1e-12
    r_proj = np.sum(delta * dirs_vox_per_mm, axis=1) / denom
    r_new = np.maximum(r_proj, min_radii_mm.astype(np.float32))
    n_clamped = int(np.sum(r_new > (r_proj + 1e-6)))
    out = center_xyz.astype(np.float32)[None, :] + r_new[:, None] * dirs_vox_per_mm
    out[:, 0] = np.clip(out[:, 0], 0, shape_xyz[0] - 1)
    out[:, 1] = np.clip(out[:, 1], 0, shape_xyz[1] - 1)
    out[:, 2] = np.clip(out[:, 2], 0, shape_xyz[2] - 1)
    return out.astype(np.float32), n_clamped


def build_margin_mask_from_seed(
    seed_mask_xyz: np.ndarray,
    brain_mask_xyz: np.ndarray,
    zooms_xyz: np.ndarray,
    margin_mm: float,
) -> np.ndarray:
    seed = seed_mask_xyz.astype(bool, copy=False)
    brain = brain_mask_xyz.astype(bool, copy=False)
    if float(margin_mm) <= 0.0:
        out = seed.copy()
    else:
        dist = distance_transform_edt(~seed, sampling=np.asarray(zooms_xyz, dtype=np.float64))
        out = dist <= float(margin_mm)
    out |= seed
    out &= brain
    return out


def _sample_smooth_direction_field(
    dirs_xyz: np.ndarray,
    rng: np.random.Generator,
    nn_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    r = dirs_xyz.shape[0]
    field = np.zeros((r,), dtype=np.float32)
    n_terms = int(rng.integers(3, 7))
    for _ in range(n_terms):
        u = rng.normal(size=3).astype(np.float32)
        u /= float(np.linalg.norm(u) + 1e-8)
        v = rng.normal(size=3).astype(np.float32)
        v /= float(np.linalg.norm(v) + 1e-8)
        d1 = dirs_xyz @ u
        d2 = dirs_xyz @ v
        mode = int(rng.integers(0, 4))
        if mode == 0:
            comp = d1
        elif mode == 1:
            comp = d1 * d1 - (1.0 / 3.0)
        elif mode == 2:
            comp = d1 * d2
        else:
            comp = np.sin(np.pi * d1) * np.cos(np.pi * d2)
        field += float(rng.uniform(-1.0, 1.0)) * comp.astype(np.float32)
    field = field - float(field.mean())
    field = field / float(field.std() + 1e-6)
    if nn_idx is not None:
        field = field[nn_idx].mean(axis=1).astype(np.float32)
        field = field - float(field.mean())
        field = field / float(field.std() + 1e-6)
    return field.astype(np.float32)


def _resolve_multi_init_seed_mask_from_seg(
    seg_xyz: np.ndarray,
    core_mask_xyz: np.ndarray,
    brain_mask_xyz: np.ndarray,
    *,
    outer_source: str,
    outer_labels_spec: str,
    label_profile: str,
    path_hints: Optional[Sequence[str]] = None,
    pid: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    resolved_profile = str(label_profile or "auto")
    outer_labels_used: List[int] = []
    outer_mask_xyz = np.zeros_like(core_mask_xyz, dtype=np.uint8)
    fallback_reason = None

    if str(outer_source).strip().lower() == "constraint_core":
        outer_mask_xyz = core_mask_xyz.astype(np.uint8, copy=True)
    elif str(outer_source).strip().lower() == "ed":
        outer_labels_used, resolved_profile = resolve_role_labels_compat(
            seg_xyz.astype(np.int32),
            role="ed",
            label_spec=outer_labels_spec,
            label_profile=resolved_profile,
            path_hints=path_hints,
            pid=pid,
        )
        if outer_labels_used:
            outer_mask_xyz = np.logical_and(np.isin(seg_xyz, outer_labels_used), brain_mask_xyz > 0).astype(np.uint8)
        if not np.any(outer_mask_xyz):
            outer_mask_xyz = core_mask_xyz.astype(np.uint8, copy=True)
            fallback_reason = "ed_empty_fallback_to_constraint_core"
    else:
        raise ValueError(f"Unsupported --multi-init-outer-source={outer_source!r}")

    seed_mask_xyz = np.logical_or(outer_mask_xyz > 0, core_mask_xyz > 0).astype(np.uint8)
    seed_mask_xyz &= (brain_mask_xyz > 0).astype(np.uint8)
    meta = {
        "label_profile": str(resolved_profile),
        "outer_source": str(outer_source),
        "outer_labels_used": [int(x) for x in outer_labels_used],
        "outer_mask_voxels": int(np.sum(outer_mask_xyz > 0)),
        "seed_mask_voxels": int(np.sum(seed_mask_xyz > 0)),
        "constraint_mask_voxels": int(np.sum(core_mask_xyz > 0)),
        "outer_fallback": fallback_reason,
    }
    return seed_mask_xyz.astype(np.uint8), meta


def _select_run_center_xyz(
    core_mask_xyz: np.ndarray,
    base_center_xyz: np.ndarray,
    zooms_xyz: np.ndarray,
    *,
    run_seed: int,
    center_jitter_mm: float,
) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(int(run_seed) + 1009)
    center_xyz, center_meta = jitter_center_xyz_within_mask_compat(
        mask_xyz=core_mask_xyz.astype(np.uint8),
        anchor_xyz=np.asarray(base_center_xyz, dtype=np.float32),
        zooms_xyz=np.asarray(zooms_xyz, dtype=np.float32),
        rng=rng,
        max_jitter_mm=float(center_jitter_mm),
    )
    return center_xyz.astype(np.float32), center_meta


def build_diverse_initial_contours_from_seed(
    ref_coords_xyz: np.ndarray,
    center_xyz: np.ndarray,
    zooms_xyz: np.ndarray,
    core_mask_xyz: np.ndarray,
    seed_mask_xyz: np.ndarray,
    brain_mask_xyz: np.ndarray,
    shape_xyz: Tuple[int, int, int],
    k: int,
    margin_mm: float,
    run_seed: int,
    within_step_mm: float,
    shape_jitter_mm: float,
    brain_cap_margin_mm: float,
    ray_step_mm: float,
) -> Tuple[List[np.ndarray], Dict]:
    k = max(1, int(k))
    center = center_xyz.astype(np.float32)
    zooms = zooms_xyz.astype(np.float32)
    dirs_vox_per_mm = build_dirs_vox_per_mm_from_reference(ref_coords_xyz, center, zooms)
    dirs_xyz = dirs_vox_per_mm * zooms[None, :]
    dirs_xyz = dirs_xyz / (np.linalg.norm(dirs_xyz, axis=1, keepdims=True) + 1e-8)

    core_r_mm = contour_radii_mm_from_mask_with_dirs(
        core_mask_xyz.astype(np.uint8), center, dirs_vox_per_mm, dr_mm=float(ray_step_mm)
    )
    margin_mask = build_margin_mask_from_seed(seed_mask_xyz, brain_mask_xyz, zooms, margin_mm=float(margin_mm))
    base_r_mm = contour_radii_mm_from_mask_with_dirs(
        margin_mask.astype(np.uint8), center, dirs_vox_per_mm, dr_mm=float(ray_step_mm)
    )
    brain_r_mm = contour_radii_mm_from_mask_with_dirs(
        (brain_mask_xyz > 0).astype(np.uint8), center, dirs_vox_per_mm, dr_mm=float(ray_step_mm)
    )
    if float(np.max(base_r_mm)) <= 0.0:
        raise RuntimeError("Computed multi-init seed+margin initial shell has all-zero radii.")

    floor_r = core_r_mm + 0.2
    cap_r = np.maximum(brain_r_mm - float(brain_cap_margin_mm), floor_r + 1e-3)
    base_r_mm = np.minimum(np.maximum(base_r_mm, floor_r + 0.5), cap_r)

    tree = cKDTree(dirs_xyz.astype(np.float64))
    k_nb = min(24, int(dirs_xyz.shape[0]))
    _, nn_idx = tree.query(dirs_xyz.astype(np.float64), k=k_nb)
    if k_nb == 1:
        nn_idx = np.asarray(nn_idx).reshape(-1, 1)
    else:
        nn_idx = np.asarray(nn_idx)

    rng = np.random.default_rng(int(run_seed))
    offsets = (np.arange(k, dtype=np.float32) - 0.5 * float(k - 1)) * float(within_step_mm)
    contours: List[np.ndarray] = []
    contour_meta: List[Dict[str, float]] = []

    for j in range(k):
        field = _sample_smooth_direction_field(dirs_xyz, rng, nn_idx=nn_idx)
        amp = float(shape_jitter_mm) * float(0.6 + 0.8 * rng.random())
        scale = float(1.0 + rng.uniform(-0.08, 0.08))
        r_mm = (base_r_mm * scale) + float(offsets[j]) + amp * field
        r_mm = np.clip(r_mm, floor_r, cap_r)
        coords = center[None, :] + r_mm[:, None] * dirs_vox_per_mm
        coords[:, 0] = np.clip(coords[:, 0], 0, shape_xyz[0] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, shape_xyz[1] - 1)
        coords[:, 2] = np.clip(coords[:, 2], 0, shape_xyz[2] - 1)
        contours.append(coords.astype(np.float32))
        contour_meta.append(
            {
                "idx": int(j),
                "offset_mm": float(offsets[j]),
                "shape_amp_mm": float(amp),
                "global_scale": float(scale),
                "r_mean_mm": float(np.mean(r_mm)),
                "r_std_mm": float(np.std(r_mm)),
            }
        )

    meta = {
        "run_seed": int(run_seed),
        "k": int(k),
        "margin_mm": float(margin_mm),
        "within_step_mm": float(within_step_mm),
        "shape_jitter_mm": float(shape_jitter_mm),
        "brain_cap_margin_mm": float(brain_cap_margin_mm),
        "ray_step_mm": float(ray_step_mm),
        "base_r_mean_mm": float(np.mean(base_r_mm)),
        "base_r_std_mm": float(np.std(base_r_mm)),
        "floor_r_mean_mm": float(np.mean(floor_r)),
        "cap_r_mean_mm": float(np.mean(cap_r)),
        "contours": contour_meta,
    }
    return contours, meta


def aggregate_multi_init_risk_for_case(
    h5_path: Path,
    pid: str,
    run_dirs: List[Path],
    out_root: Path,
    smooth_sigma: float = 0.0,
    equal_volume: bool = False,
) -> None:
    with h5py.File(h5_path, "r") as hf:
        shape_xyz = tuple(int(x) for x in hf["seg_data"].shape)
        affine = np.eye(4, dtype=np.float32)
        if "img" in hf and "affines" in hf["img"] and "flair_data" in hf["img"]["affines"]:
            affine = np.asarray(hf["img"]["affines"]["flair_data"][()], dtype=np.float32)

    probs = []
    used_shells = []
    run_summaries = []
    for rd in run_dirs:
        sf = rd / f"{pid}_removed_shells.npy"
        if not sf.exists():
            raise RuntimeError(f"{pid}: missing run output {sf}; aggregation requires all requested runs")
        status_path = rd / f"{pid}_run_status.json"
        if not status_path.is_file():
            raise RuntimeError(f"{pid}: missing run status {status_path}; cannot verify completion")
        status = json.loads(status_path.read_text())
        if status.get("status") != "completed":
            raise RuntimeError(f"{pid}: run {rd.name} did not complete; refusing to aggregate partial output")
        shells = np.load(sf)
        if shells.shape[0] < 2:
            raise RuntimeError(f"{pid}: {rd.name} has fewer than two levels; ordinal aggregation is undefined")
        if shells.shape[0] != int(status["target_removed"]):
            raise RuntimeError(f"{pid}: {rd.name} shell count differs from requested hierarchy length")
        if np.allclose(shells, shells[:1], rtol=0, atol=1e-5):
            raise RuntimeError(f"{pid}: {rd.name} contains only identical surfaces; no hierarchy to aggregate")
        prob = shells_to_prob_volume(
            shells=shells,
            shape=shape_xyz,
            smooth_sigma=float(smooth_sigma),
            equal_volume=bool(equal_volume),
        )
        probs.append(np.clip(prob.astype(np.float32), 0.0, 1.0))
        used_shells.append(str(sf))
        run_summaries.append({"run": rd.name, "raw_levels": int(shells.shape[0]), **status})

    if not probs:
        raise RuntimeError(f"{pid}: no run outputs found for multi-init aggregation.")

    stack = np.stack(probs, axis=0).astype(np.float32)
    mean_prob = np.mean(stack, axis=0).astype(np.float32)
    std_prob = np.std(stack, axis=0).astype(np.float32)

    agg_dir = out_root / pid / "aggregate_multi_init"
    agg_dir.mkdir(parents=True, exist_ok=True)
    np.save(agg_dir / "risk_prob_mean.npy", mean_prob)
    np.save(agg_dir / "risk_prob_std.npy", std_prob)
    if nib is not None:
        nib.save(nib.Nifti1Image(mean_prob, affine=affine), str(agg_dir / "risk_prob_mean.nii.gz"))
        nib.save(nib.Nifti1Image(std_prob, affine=affine), str(agg_dir / "risk_prob_std.nii.gz"))

    meta = {
        "pid": pid,
        "n_runs_used": int(stack.shape[0]),
        "shape_xyz": list(shape_xyz),
        "smooth_sigma": float(smooth_sigma),
        "equal_volume": bool(equal_volume),
        "used_shell_files": used_shells,
        "runs": run_summaries,
        "interpretation": "Within-patient ordinal values, not calibrated infiltration probabilities",
        "rasterization": "Legacy (i+1)/T surface painting, max at collisions, EDT fill inside expanded bbox",
        "aggregation": "Arithmetic mean; population standard deviation (ddof=0)",
        "mean_prob_min": float(np.min(mean_prob)),
        "mean_prob_max": float(np.max(mean_prob)),
        "mean_prob_mean": float(np.mean(mean_prob)),
    }
    (agg_dir / "meta.json").write_text(json.dumps(meta, indent=2))


@torch.no_grad()
def pairwise_matrix(comp, pcs, device, chunk_size=None):
    """All pairwise comparisons with optional chunking."""
    K = len(pcs)
    P = torch.full((K, K), 0.5, dtype=torch.float32, device=device)
    if K <= 1:
        return P
    pairs = [(i, j) for i in range(K) for j in range(K) if i != j]
    if not pairs:
        return P
    mats = [pc if pc.device == device else pc.to(device, non_blocking=True) for pc in pcs]
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = len(pairs)
    chunk_size = max(1, int(chunk_size))
    for i0 in range(0, len(pairs), chunk_size):
        sub_pairs = pairs[i0 : i0 + chunk_size]
        A = torch.stack([mats[i] for i, _ in sub_pairs], dim=0)
        B = torch.stack([mats[j] for _, j in sub_pairs], dim=0)
        p, _ = comp(A, B)
        for (i, j), pij in zip(sub_pairs, p):
            P[i, j] = pij
    return P


def rank_scores_from_P(P: torch.Tensor):
    K = P.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=P.device)
    sums = (P * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    s = (sums / counts).detach().cpu().numpy()
    return s


class AdaptiveShrink:
    def __init__(self,
                 eps_init=0.25, eps_max=1.0,
                 steps_init=6, steps_max=16,
                 warmup_iters=50, cutoff_iter=120,
                 factor=1.4, retries=3):
        self.eps_init = float(eps_init)
        self.eps_max = float(eps_max)
        self.steps_init = int(steps_init)
        self.steps_max = int(steps_max)
        self.warmup_iters = int(warmup_iters)
        self.cutoff_iter = int(cutoff_iter)
        self.factor = float(factor)
        self.retries = int(retries)

    def propose(self, t, attempt=0):
        if t < self.warmup_iters:
            ratio = (t + 1) / max(1, self.warmup_iters)
            base_eps = self.eps_init + ratio * (0.7 * self.eps_max - self.eps_init)
            base_steps = self.steps_init + ratio * (0.7 * self.steps_max - self.steps_init)
        elif t < self.cutoff_iter:
            ratio = (t - self.warmup_iters) / max(1, self.cutoff_iter - self.warmup_iters)
            base_eps = (1 - ratio) * (0.6 * self.eps_max) + ratio * self.eps_init
            base_steps = (1 - ratio) * (0.6 * self.steps_max) + ratio * self.steps_init
        else:
            base_eps, base_steps = self.eps_init, self.steps_init

        eps = min(self.eps_max, base_eps * (self.factor ** attempt))
        steps = int(min(self.steps_max,
                        max(self.steps_init,
                            round(base_steps * (self.factor ** (attempt * 0.5))))))
        return float(eps), int(steps)


def hard_cleanup():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# -------------------------
# Case runner
# -------------------------
def validate_h5(h5_path: Path) -> None:
    """Validate the input arrays before comparator inference."""
    with h5py.File(h5_path, "r") as hf:
        required = ["seg_data", "expand", "wm_pbmap", "gm_pbmap", "brain_mask",
                    "img/flair_data", "img/t1", "img/t1gd", "img/t2",
                    "img/affines/flair_data"]
        missing = [key for key in required if key not in hf]
        if missing:
            raise ValueError(f"{h5_path.name}: missing HDF5 fields: {missing}")
        shape = hf["seg_data"].shape
        if len(shape) != 3:
            raise ValueError("seg_data must be a 3D array in XYZ voxel order")
        for key in ("seg_data", "wm_pbmap", "gm_pbmap", "brain_mask",
                    "img/flair_data", "img/t1", "img/t1gd", "img/t2"):
            arr = hf[key][()]
            if arr.shape != shape or not np.isfinite(arr).all():
                raise ValueError(f"{key}: non-finite data or shape differs from seg_data")
        if not np.any(hf["brain_mask"][()] > 0):
            raise ValueError("brain_mask is empty")
        seg = hf["seg_data"][()]
        if not np.allclose(seg, np.rint(seg), atol=1e-4):
            raise ValueError("seg_data must contain integer labels")
        for key in ("wm_pbmap", "gm_pbmap"):
            arr = hf[key][()]
            if arr.min() < -1e-5 or arr.max() > 1 + 1e-5:
                raise ValueError(f"{key}: expected probabilities in [0,1]")
        expand = hf["expand"]
        if expand.ndim != 3 or expand.shape[0] < 1 or expand.shape[2] != 3:
            raise ValueError("expand must have shape (S,N,3) with S>=1")
        n = expand.shape[1]
        if int(round(np.sqrt(n))) ** 2 != n or n < 4:
            raise ValueError("expand surface must have N=side*side points on an ordered radial grid")
        ref = expand[-1]
        if not np.isfinite(ref).all() or (ref < 0).any() or (ref > np.array(shape) - 1).any():
            raise ValueError("expand[-1] must contain finite in-bounds voxel XYZ coordinates")
        if "center_xyz" in hf:
            center = hf["center_xyz"][()]
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError("center_xyz must contain three finite voxel coordinates")
        affine = hf["img/affines/flair_data"][()]
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
            raise ValueError("FLAIR affine must be finite and nonsingular")
        for key in ("t1", "t1gd", "t2"):
            path = f"img/affines/{key}"
            if path in hf and not np.allclose(hf[path][()], affine, atol=1e-3, rtol=0):
                raise ValueError(f"{path}: affine differs from FLAIR")


def run_case(h5_path: Path, cfg: dict, comp, device) -> None:
    device = torch.device(device)
    pid = h5_path.stem
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as hf:
        flair = robust_normalize_np(hf["img"]["flair_data"][()], cfg["p1"], cfg["p99"])
        t1 = robust_normalize_np(hf["img"]["t1"][()], cfg["p1"], cfg["p99"])
        t1gd = robust_normalize_np(hf["img"]["t1gd"][()], cfg["p1"], cfg["p99"])
        t2 = robust_normalize_np(hf["img"]["t2"][()], cfg["p1"], cfg["p99"])
        img_vol = np.stack([flair, t1, t1gd, t2], axis=0).astype(np.float32)

        seg = hf["seg_data"][()].astype(np.int32)
        center_xyz = hf["center_xyz"][()].astype(np.float32) if "center_xyz" in hf else center_from_seg_xyz(seg)

        expand = hf["expand"][()]
        if expand.shape[0] == 0:
            raise RuntimeError(f"{pid}: no expand shells in this case")
        ref_init_coords = expand[-1].astype(np.float32)

        if "img" in hf and "affines" in hf["img"] and "flair_data" in hf["img"]["affines"]:
            flair_aff = hf["img"]["affines"]["flair_data"][()]
            zooms = zooms_from_affine(flair_aff)
        else:
            zooms = (1.0, 1.0, 1.0)

        wm = hf["wm_pbmap"][()].astype(np.float32)
        gm = hf["gm_pbmap"][()].astype(np.float32)
        brain_mask = (hf["brain_mask"][()] > 0).astype(np.uint8)
        bm = (brain_mask > 0).astype(np.float32)
        wm = wm * bm
        gm = gm * bm

    adaptor = AdaptiveShrink(
        eps_init=cfg["eps_init"], eps_max=cfg["eps_max"],
        steps_init=cfg["steps_init"], steps_max=cfg["steps_max"],
        warmup_iters=cfg["warmup_iters"], cutoff_iter=cfg["cutoff_iters"],
        factor=cfg["adapt_factor"], retries=cfg["adapt_retries"],
    )

    label_profile = resolve_label_profile_compat(
        seg.astype(np.int32),
        label_profile=str(cfg.get("label_profile", "auto")),
        path_hints=[str(h5_path), str(h5_path.parent)],
        pid=pid,
    )
    labels_used = resolve_constraint_labels_compat(
        seg.astype(np.int32),
        str(cfg.get("constraint_labels", "")).strip(),
        label_profile=label_profile,
        path_hints=[str(h5_path), str(h5_path.parent)],
        pid=pid,
    )
    core_mask_raw = np.isin(seg, labels_used)
    core_mask = np.logical_and(core_mask_raw, brain_mask > 0)
    constraint_active = bool(np.any(core_mask))
    if bool(cfg.get("constraint_require_nonempty", True)) and not constraint_active:
        raise RuntimeError(f"{pid}: NCR+ET constraint mask is empty (labels_used={labels_used}).")

    zooms_xyz = np.asarray(zooms, dtype=np.float32)
    center_xyz_run = np.asarray(center_xyz, dtype=np.float32)
    center_meta = None
    multi_init_enabled = bool(cfg.get("multi_init_enabled", False))
    if multi_init_enabled:
        center_xyz_run, center_meta = _select_run_center_xyz(
            core_mask.astype(np.uint8),
            center_xyz,
            zooms_xyz,
            run_seed=int(cfg.get("seed", 2025)),
            center_jitter_mm=float(cfg.get("multi_init_center_jitter_mm", 3.0)),
        )

    floor_dirs_xyz = build_directions_equirect(
        n_theta=int(cfg.get("constraint_floor_n_theta", 256)),
        n_phi=int(cfg.get("constraint_floor_n_phi", 128)),
    )
    floor_radii_mm = contour_radii_mm_from_mask(
        mask_xyz=core_mask.astype(np.uint8),
        center_xyz=center_xyz_run,
        dirs_xyz=floor_dirs_xyz,
        zooms_xyz=zooms_xyz,
        dr_mm=float(cfg.get("constraint_ray_step_mm", 0.5)),
    )
    floor_tree = cKDTree(floor_dirs_xyz.astype(np.float64))
    if constraint_active and float(np.max(floor_radii_mm)) <= 0.0:
        raise RuntimeError(f"{pid}: computed NCR+ET floor radii are all zero.")

    init_meta: Optional[Dict] = None
    if multi_init_enabled:
        run_idx = int(cfg.get("multi_init_run_idx", 0))
        seed_mask_xyz, seed_meta = _resolve_multi_init_seed_mask_from_seg(
            seg,
            core_mask.astype(np.uint8),
            brain_mask.astype(np.uint8),
            outer_source=str(cfg.get("multi_init_outer_source", "ed")),
            outer_labels_spec=str(cfg.get("multi_init_outer_labels", "")),
            label_profile=label_profile,
            path_hints=[str(h5_path), str(h5_path.parent)],
            pid=pid,
        )
        live_coords, init_meta = build_diverse_initial_contours_from_seed(
            ref_coords_xyz=ref_init_coords,
            center_xyz=center_xyz_run,
            zooms_xyz=zooms_xyz,
            core_mask_xyz=core_mask.astype(np.uint8),
            seed_mask_xyz=seed_mask_xyz.astype(np.uint8),
            brain_mask_xyz=brain_mask.astype(np.uint8),
            shape_xyz=seg.shape,
            k=int(cfg["K"]),
            margin_mm=float(cfg.get("multi_init_base_margin_mm", 5.0)),
            run_seed=int(cfg.get("seed", 2025)),
            within_step_mm=float(cfg.get("multi_init_within_step_mm", 1.0)),
            shape_jitter_mm=float(cfg.get("multi_init_shape_jitter_mm", 3.0)),
            brain_cap_margin_mm=float(cfg.get("multi_init_brain_cap_margin_mm", 0.5)),
            ray_step_mm=float(cfg.get("constraint_ray_step_mm", 0.5)),
        )
        init_meta.update(seed_meta)
        if center_meta is not None:
            init_meta.update(center_meta)
            init_meta["run_center_xyz"] = center_xyz_run.tolist()
        print(f"{pid}: multi-init enabled (run={run_idx}) -> generated {len(live_coords)} initial contours", flush=True)
    else:
        init_num = min(cfg["K"], expand.shape[0])
        live_coords = [expand[-init_num + i].astype(np.float32) for i in range(init_num)]

    if constraint_active:
        constrained = []
        for c in live_coords:
            c_new, _ = enforce_min_radii_constraint_dense_floor(
                c,
                center_xyz=center_xyz_run,
                zooms_xyz=zooms_xyz,
                floor_dirs_xyz=floor_dirs_xyz,
                floor_radii_mm=floor_radii_mm,
                floor_tree=floor_tree,
                shape_xyz=seg.shape,
                floor_k=int(cfg.get("constraint_floor_k", 3)),
                floor_margin_mm=float(cfg.get("constraint_floor_margin_mm", 0.0)),
            )
            constrained.append(c_new)
        live_coords = constrained

    if multi_init_enabled and init_meta is not None:
        np.save(out_dir / f"{pid}_initial_contours.npy", np.stack(live_coords, axis=0).astype(np.float32))
        (out_dir / f"{pid}_multi_init_meta.json").write_text(json.dumps(init_meta, indent=2))

    live_pcs_dev = [coords_to_pc_xyz(c, img_vol).to(device, non_blocking=True) for c in live_coords]
    cand_score_batch = max(1, int(cfg.get("cand_score_batch", cfg.get("cand", 1))))
    gpu_cleanup_interval = max(0, int(cfg.get("gpu_cleanup_interval", 0)))
    history = []
    removed_coords = []
    remaining_target = int(cfg.get("target_removed", 0))
    if remaining_target <= 0:
        raise ValueError("CONFIG['target_removed'] must be > 0")
    it = 0
    exhausted = False
    while len(removed_coords) < remaining_target:
        if it % 50 == 0:
            print(f"{pid}: current {it}", flush=True)
        P_live = pairwise_matrix(comp, live_pcs_dev, device, chunk_size=cfg["pair_chunk"])
        s_live = rank_scores_from_P(P_live)
        span_curr = float(s_live.max() - s_live.min())
        idx_w = int(np.argmin(s_live))
        worst_coords = live_coords.pop(idx_w)
        worst_pc_dev = live_pcs_dev.pop(idx_w)
        worst_rank = float(s_live[idx_w])
        removed_coords.append(worst_coords.copy())
        parent_pc_dev = worst_pc_dev
        del P_live, s_live
        best_prob = -1.0
        best_coords = None
        best_clamped = 0
        used_eps = None
        used_steps = None
        accepted_coords = []
        accepted_pvals = []
        accepted_clamped = []
        stationary_proposals = 0
        attempt = 0
        while True:
            eps_mm, steps_now = adaptor.propose(t=it, attempt=attempt)
            num_done = 0
            while num_done < int(cfg["cand"]):
                n_make = min(cand_score_batch, int(cfg["cand"]) - num_done)
                cand_coords_batch: List[np.ndarray] = []
                cand_clamped_batch: List[int] = []
                cand_pc_batch: List[torch.Tensor] = []
                for b in range(n_make):
                    cand_seed = int(cfg["seed"] + it * 7919 + attempt * 101 + num_done + b)
                    traj = radial_shrink_points_transition_matrix_v3(
                        worst_coords,
                        wm=wm, gm=gm,
                        brain_mask=brain_mask,
                        seg=seg,
                        zooms=zooms,
                        center_xyz=center_xyz_run,
                        n_steps=steps_now,
                        eps_phys=cfg.get("eps_phys", eps_mm),
                        eps_floor=cfg.get("eps_floor", 0.05),
                        step_fracs=cfg.get("step_fracs", (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)),
                        R=cfg.get("R", 3.0),
                        perm_floor=cfg.get("perm_floor", 0.08),
                        stay_floor=cfg.get("stay_floor", 1e-4),
                        lambda0=cfg.get("lambda0", 0.50),
                        lambda_min=cfg.get("lambda_min", 0.15),
                        lambda_max=cfg.get("lambda_max", 0.90),
                        beta_aff=cfg.get("beta_aff", 8.0),
                        move_k=cfg.get("move_k", 256),
                        act_sigma=cfg.get("act_sigma", 2.0),
                        kappa_lag=cfg.get("kappa_lag", 3.0),
                        lag_no_stay_mm=cfg.get("lag_no_stay_mm", 0.25),
                        cap_mm=cfg.get("cap_mm", 0.20),
                        gamma_cap=cfg.get("gamma_cap", 8.0),
                        project_after=True,
                        project_iters=cfg.get("project_iters", 1),
                        motion_close_iter=cfg.get("motion_close_iter", 2),
                        motion_dil_iter=cfg.get("motion_dil_iter", 1),
                        dr_probe=cfg.get("dr_probe", 0.5),
                        cap_margin_mm=cfg.get("cap_margin_mm", 0.2),
                        rng=np.random.default_rng(cand_seed),
                        debug=cfg.get("tm_debug", False),
                    )
                    cand_coords = traj[-1].astype(np.float32)
                    del traj
                    n_clamped = 0
                    if constraint_active:
                        cand_coords, n_clamped = enforce_min_radii_constraint_dense_floor(
                            cand_coords,
                            center_xyz=center_xyz_run,
                            zooms_xyz=zooms_xyz,
                            floor_dirs_xyz=floor_dirs_xyz,
                            floor_radii_mm=floor_radii_mm,
                            floor_tree=floor_tree,
                            shape_xyz=seg.shape,
                            floor_k=int(cfg.get("constraint_floor_k", 3)),
                            floor_margin_mm=float(cfg.get("constraint_floor_margin_mm", 0.0)),
                        )
                    cand_coords_batch.append(cand_coords)
                    cand_clamped_batch.append(int(n_clamped))
                    cand_pc_batch.append(coords_to_pc_xyz(cand_coords, img_vol))
                cand_stack_dev = torch.stack(cand_pc_batch, dim=0).to(device, non_blocking=True)
                parent_batch_dev = parent_pc_dev.unsqueeze(0).expand(cand_stack_dev.shape[0], -1, -1)
                with torch.inference_mode():
                    p_vals, _ = comp(cand_stack_dev, parent_batch_dev)
                p_vals_np = p_vals.detach().cpu().numpy().astype(np.float32)
                for b, p_val in enumerate(p_vals_np):
                    p_val = float(p_val)
                    cand_coords = cand_coords_batch[b]
                    n_clamped = cand_clamped_batch[b]
                    if p_val > best_prob:
                        best_prob = p_val
                        best_coords = cand_coords.copy()
                        best_clamped = int(n_clamped)
                    moved = not np.allclose(cand_coords, worst_coords, rtol=0, atol=1e-5)
                    stationary_proposals += int(not moved)
                    if p_val > 0.5 and (moved or bool(cfg.get("allow_unranked_fallback", False))):
                        accepted_coords.append(cand_coords.copy())
                        accepted_pvals.append(p_val)
                        accepted_clamped.append(int(n_clamped))
                num_done += n_make
                del cand_stack_dev, parent_batch_dev, p_vals, p_vals_np, cand_pc_batch
            if accepted_coords:
                used_eps, used_steps = eps_mm, steps_now
                break
            attempt += 1
            if attempt >= int(cfg.get("max_attempts", 20)):
                used_eps, used_steps = eps_mm, steps_now
                break
        del worst_pc_dev
        if accepted_coords:
            if len(accepted_coords) == 1:
                replace_coords = accepted_coords[0]
                chosen_p = float(accepted_pvals[0])
                chosen_clamped = int(accepted_clamped[0])
            else:
                best_idx_pool = int(np.argmax(accepted_pvals))
                replace_coords = accepted_coords[best_idx_pool]
                chosen_p = float(accepted_pvals[best_idx_pool])
                chosen_clamped = int(accepted_clamped[best_idx_pool])
        elif not bool(cfg.get("allow_unranked_fallback", False)):
            history.append({
                "iter": int(it), "span": span_curr, "worst_rank": worst_rank,
                "accepted": 0, "last_p": float(best_prob),
                "status": "stopped_no_admissible_replacement", "proposal_cycles": int(attempt),
                "stationary_proposals": stationary_proposals,
                "effective_eps_phys": float(cfg.get("eps_phys", used_eps)),
                "eps": used_eps, "steps": used_steps,
                "constraint_active": constraint_active,
                "constraint_labels_used": [int(x) for x in labels_used],
            })
            exhausted = True
            break
        else:
            replace_coords = best_coords if best_coords is not None else worst_coords.copy()
            chosen_p = best_prob
            chosen_clamped = int(best_clamped if best_coords is not None else 0)
        if constraint_active:
            replace_coords, chosen_clamped_refine = enforce_min_radii_constraint_dense_floor(
                replace_coords,
                center_xyz=center_xyz_run,
                zooms_xyz=zooms_xyz,
                floor_dirs_xyz=floor_dirs_xyz,
                floor_radii_mm=floor_radii_mm,
                floor_tree=floor_tree,
                shape_xyz=seg.shape,
                floor_k=int(cfg.get("constraint_floor_k", 3)),
                floor_margin_mm=float(cfg.get("constraint_floor_margin_mm", 0.0)),
            )
            chosen_clamped = int(max(chosen_clamped, chosen_clamped_refine))
        live_coords.insert(idx_w, replace_coords)
        live_pcs_dev.insert(idx_w, coords_to_pc_xyz(replace_coords, img_vol).to(device, non_blocking=True))
        history.append({
            "iter": int(it),
            "span": float(span_curr),
            "worst_rank": float(worst_rank),
            "accepted": int(len(accepted_coords)),
            "status": "accepted" if accepted_coords else "legacy_unranked_fallback",
            "proposal_cycles": int(attempt + 1 if accepted_coords else attempt),
            "stationary_proposals": stationary_proposals,
            "effective_eps_phys": float(cfg.get("eps_phys", used_eps)),
            "last_p": (None if chosen_p is None else float(chosen_p)),
            "eps": used_eps,
            "steps": used_steps,
            "constraint_active": bool(constraint_active),
            "constraint_labels_used": [int(x) for x in labels_used],
            "constraint_clamped_points": int(chosen_clamped),
        })
        it += 1
        if gpu_cleanup_interval > 0 and (it % gpu_cleanup_interval == 0):
            hard_cleanup()
        if it % 10 == 0:
            print(
                f"{pid}: iter {it}/{remaining_target} | span={span_curr:.4f} | "
                f"worst={worst_rank:.4f} | accN={len(accepted_coords)} | last p={chosen_p}",
                flush=True,
            )
    if removed_coords:
        removed_arr = np.stack(removed_coords).astype(np.float32)
        np.save(out_dir / f"{pid}_removed_shells.npy", removed_arr)
    hist_path = out_dir / f"{pid}_nested_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    run_status = {
        "status": "stopped_no_admissible_replacement" if exhausted else "completed",
        "target_removed": remaining_target, "removed_shells": len(removed_coords),
        "accepted_replacements": sum(h["status"] == "accepted" for h in history),
        "legacy_fallbacks": sum(h["status"] == "legacy_unranked_fallback" for h in history),
        "allow_unranked_fallback": bool(cfg.get("allow_unranked_fallback", False)),
    }
    (out_dir / f"{pid}_run_status.json").write_text(json.dumps(run_status, indent=2))
    print(f"{pid}: saved history to {hist_path}", flush=True)
    if exhausted:
        raise RuntimeError(f"{pid}: no admissible replacement; partial outputs saved, aggregation stopped. "
                           "Legacy reproduction requires explicit --allow-unranked-fallback.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Multi-init contour search from aligned HDF5 inputs.")
    ap.add_argument("--h5-dir", type=Path, required=True, help="Directory containing prepared .h5 cases.")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--target-removed", type=int, default=1000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument(
        "--case-id",
        type=str,
        default="",
        help="Optional case id(s) to run, comma-separated (e.g. RHUH-0039 or Patient-042).",
    )
    ap.add_argument("--max-workers", type=int, default=1, help="Parallel workers (one GPU worker recommended unless you have multiple GPUs).")
    ap.add_argument("--multi-init-runs", type=int, default=10, help="Run multiple diverse initializations per patient.")
    ap.add_argument(
        "--multi-init-parallel",
        type=int,
        default=0,
        help="Parallel workers for multi-init runs of one patient (0 => auto).",
    )
    ap.add_argument("--multi-init-base-margin-mm", type=float, default=5.0, help="Initial contour base mask margin in mm (applied to the seed region).")
    ap.add_argument(
        "--multi-init-center-jitter-mm",
        type=float,
        default=3.0,
        help="Per-run center perturbation radius (mm) sampled inside NCR+ET.",
    )
    ap.add_argument(
        "--multi-init-outer-source",
        type=str,
        default="ed",
        choices=["ed", "constraint_core"],
        help="Seed region for multi-init outer contour generation.",
    )
    ap.add_argument(
        "--multi-init-outer-labels",
        type=str,
        default="",
        help="Optional explicit labels for multi-init outer seed. Empty = resolve from --label-profile.",
    )
    ap.add_argument("--multi-init-within-step-mm", type=float, default=1.0, help="Within-run radial offset spacing (mm) between K initials.")
    ap.add_argument("--multi-init-shape-jitter-mm", type=float, default=3.0, help="Amplitude of shape perturbation (mm).")
    ap.add_argument("--multi-init-brain-cap-margin-mm", type=float, default=0.5, help="Safety margin (mm) from brain boundary for initials.")
    ap.add_argument("--multi-init-risk-smooth-sigma", type=float, default=0.0, help="Smoothing sigma for aggregated multi-init risk field.")
    ap.add_argument("--multi-init-risk-equal-volume", action="store_true", help="Use equal-volume remap when aggregating multi-init risk.")
    ap.add_argument(
        "--cand-score-batch",
        type=int,
        default=8,
        help="Batch size for GPU comparator scoring of candidate contours per attempt.",
    )
    ap.add_argument(
        "--gpu-cleanup-interval",
        type=int,
        default=0,
        help="Run hard GPU cleanup every N iterations (0 disables periodic cleanup).",
    )
    ap.add_argument(
        "--constraint-labels",
        type=str,
        default="",
        help="Optional fixed labels for hard inner boundary, e.g. '1,4'. Empty = auto NCR+ET mapping.",
    )
    ap.add_argument(
        "--label-profile",
        type=str,
        required=True, choices=list(LABEL_PROFILE_DEFS_COMPAT),
        help=(
            "Tumor label semantics: auto, brats_124, rhuh_preop_413, "
            "rhuh_recurrence_132, rec_native_341, contoursrec_h5_123."
        ),
    )
    ap.add_argument("--constraint-ray-step-mm", type=float, default=0.5)
    ap.add_argument("--constraint-floor-n-theta", type=int, default=256)
    ap.add_argument("--constraint-floor-n-phi", type=int, default=128)
    ap.add_argument("--constraint-floor-k", type=int, default=3)
    ap.add_argument("--constraint-floor-margin-mm", type=float, default=0.0)
    ap.add_argument(
        "--allow-empty-constraint",
        action="store_true",
        help="Allow running without hard floor when NCR+ET mask is empty.",
    )
    ap.add_argument("--cand", type=int, default=5, help="Proposals per retry cycle (original: 5).")
    ap.add_argument("--max-attempts", type=int, default=10, help="Maximum proposal cycles (original: 10).")
    ap.add_argument("--allow-unranked-fallback", action="store_true",
                    help="Legacy reproduction: allow unchanged proposals, and best P<=0.5 fallback after exhaustion.")
    ap.add_argument("--overwrite", action="store_true", help="Replace outputs for these cases deliberately.")
    return ap.parse_args(argv)


# top-level worker for multiprocessing (must be picklable)
def _worker_entry(args):
    h5_path_str, cfg, ckpt_path, device_str = args
    device = torch.device(device_str)
    comp_local = load_comparator(
        ckpt_path, device,
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    )
    run_case(Path(h5_path_str), cfg, comp_local, device)
    hard_cleanup()


def main(argv=None):
    args = parse_args(argv)
    if args.K < 2 or args.target_removed < 1 or args.cand < 1 or args.max_attempts < 1:
        raise ValueError("K>=2, target-removed>=1, cand>=1 and max-attempts>=1 are required")
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if args.multi_init_runs <= 0:
        raise ValueError("--multi-init-runs must be > 0")
    if args.multi_init_parallel < 0:
        raise ValueError("--multi-init-parallel must be >= 0")
    if args.multi_init_base_margin_mm < 0:
        raise ValueError("--multi-init-base-margin-mm must be >= 0")
    if args.multi_init_center_jitter_mm < 0:
        raise ValueError("--multi-init-center-jitter-mm must be >= 0")
    if args.multi_init_within_step_mm < 0:
        raise ValueError("--multi-init-within-step-mm must be >= 0")
    if args.multi_init_shape_jitter_mm < 0:
        raise ValueError("--multi-init-shape-jitter-mm must be >= 0")
    if args.multi_init_brain_cap_margin_mm < 0:
        raise ValueError("--multi-init-brain-cap-margin-mm must be >= 0")
    if args.multi_init_risk_smooth_sigma < 0:
        raise ValueError("--multi-init-risk-smooth-sigma must be >= 0")
    if args.cand_score_batch <= 0:
        raise ValueError("--cand-score-batch must be > 0")
    if args.gpu_cleanup_interval < 0:
        raise ValueError("--gpu-cleanup-interval must be >= 0")
    if args.constraint_ray_step_mm <= 0:
        raise ValueError("--constraint-ray-step-mm must be > 0")
    if args.constraint_floor_n_theta <= 0 or args.constraint_floor_n_phi <= 0:
        raise ValueError("--constraint-floor-n-theta and --constraint-floor-n-phi must be > 0")
    if args.constraint_floor_k <= 0:
        raise ValueError("--constraint-floor-k must be > 0")
    if args.constraint_floor_margin_mm < 0:
        raise ValueError("--constraint-floor-margin-mm must be >= 0")
    cfg = {
        "out_dir": str(args.out_dir),
        "device": args.device,
        # model
        "d_model": 128, "nhead": 8, "num_layers": 6, "dim_feedforward": 256, "dropout": 0.2,
        # nested sampling
        "K": args.K,
        "target_removed": args.target_removed,
        "cand": args.cand,
        "max_attempts": args.max_attempts,
        "allow_unranked_fallback": args.allow_unranked_fallback,
        "eps_rank": 0.01,
        # adaptive schedule
        "eps_init": 0.25, "eps_max": 1.0,
        "steps_init": 6, "steps_max": 16,
        "warmup_iters": 50, "cutoff_iters": 120,
        "adapt_factor": 1.4, "adapt_retries": 3,
        # shrink kernel params (unused but kept)
        "shrink_eps_min": 0.03,
        "shrink_backtrack": 0.5,
        "shrink_alpha": 1.0,
        "shrink_theta_max": 60.0,
        "shrink_move_k": -1,
        # transition params
        "eps_phys": 0.1,
        "eps_phys_min": 0.02,
        "backtrack": 0.5,
        "D_w": 1.0,
        "R": 3.0,
        "alpha": 2.0,
        "theta_max_deg": 85.0,
        "move_k": 256,
        "w1": 0.35,
        "w2": 0.15,
        "lambda_max": 0.9,
        "beta_aff": 8.0,
        "eps_floor": 0.05,
        "perm_floor": 0.08,
        "stay_floor": 1e-4,
        "lambda0": 0.50,
        "lambda_min": 0.15,
        "act_sigma": 2.0,
        "kappa_lag": 3.0,
        "lag_no_stay_mm": 0.25,
        "cap_mm": 0.20,
        "cap_margin_mm": 0.2,
        "motion_close_iter": 2,
        "motion_dil_iter": 1,
        "dr_probe": 0.5,
        "gamma_cap": 8.0,
        "step_fracs": (0.0, 0.05, 0.10, 0.25, 0.50, 1.0),
        "project_iters": 1,
        "tm_debug": False,
        # misc
        "p1": 1, "p99": 99,
        "pair_chunk": 64,
        "cand_score_batch": int(args.cand_score_batch),
        "gpu_cleanup_interval": int(args.gpu_cleanup_interval),
        "seed": args.seed,
        "constraint_labels": args.constraint_labels,
        "label_profile": args.label_profile,
        "constraint_ray_step_mm": float(args.constraint_ray_step_mm),
        "constraint_floor_n_theta": int(args.constraint_floor_n_theta),
        "constraint_floor_n_phi": int(args.constraint_floor_n_phi),
        "constraint_floor_k": int(args.constraint_floor_k),
        "constraint_floor_margin_mm": float(args.constraint_floor_margin_mm),
        "constraint_require_nonempty": bool(not args.allow_empty_constraint),
        "multi_init_enabled": False,
        "multi_init_run_idx": 0,
        "multi_init_base_margin_mm": float(args.multi_init_base_margin_mm),
        "multi_init_center_jitter_mm": float(args.multi_init_center_jitter_mm),
        "multi_init_outer_source": str(args.multi_init_outer_source),
        "multi_init_outer_labels": str(args.multi_init_outer_labels),
        "multi_init_within_step_mm": float(args.multi_init_within_step_mm),
        "multi_init_shape_jitter_mm": float(args.multi_init_shape_jitter_mm),
        "multi_init_brain_cap_margin_mm": float(args.multi_init_brain_cap_margin_mm),
    }


    # h5_paths = sorted(p for p in Path(args.h5_dir).glob("RHUH-*.h5") if p.is_file())
    h5_dir = Path(args.h5_dir)
    h5_paths = sorted(h5_dir.glob("*.h5"))
    h5_paths = [p for p in h5_paths if p.is_file()]
    if str(args.case_id).strip():
        keep = {x.strip() for x in str(args.case_id).split(",") if x.strip()}
        h5_paths = [p for p in h5_paths if p.stem in keep]
    if not h5_paths:
        raise FileNotFoundError(f"No h5 files found in {args.h5_dir}")
    for hp in h5_paths:
        validate_h5(hp)
    checkpoint_digest = hashlib.sha256()
    with args.ckpt.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_digest.update(block)
    checkpoint_sha256 = checkpoint_digest.hexdigest()
    print(f"Found {len(h5_paths)} cases in {args.h5_dir}", flush=True)

    max_workers = max(1, int(args.max_workers))
    # Use varied-core initialization, including --multi-init-runs 1.
    if int(args.multi_init_runs) >= 1:
        base_out = Path(cfg["out_dir"])
        run_workers = int(args.multi_init_parallel) if int(args.multi_init_parallel) > 0 else min(max_workers, int(args.multi_init_runs))
        run_workers = max(1, run_workers)
        print(
            f"Multi-init mode: runs={int(args.multi_init_runs)}, per-patient parallel workers={run_workers}, "
            f"base_margin_mm={float(args.multi_init_base_margin_mm)}",
            flush=True,
        )
        for hp in h5_paths:
            pid = hp.stem
            print(f"=== Multi-init constraint shrink: {pid} ===", flush=True)
            case_dir = base_out / pid
            if case_dir.exists() and any(case_dir.iterdir()):
                if not args.overwrite:
                    raise FileExistsError(f"Output exists: {case_dir}. Choose another --out-dir or use --overwrite.")
                # Explicit replacement must remove old aggregates too: a failed
                # rerun must never leave a prior successful map looking current.
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "inference_config.json").write_text(json.dumps({
                **cfg, "checkpoint": str(args.ckpt.resolve()), "input_h5": str(hp.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "multi_init_runs": args.multi_init_runs,
                "aggregation": {"smooth_sigma": args.multi_init_risk_smooth_sigma,
                                "equal_volume": args.multi_init_risk_equal_volume},
            }, indent=2))
            run_dirs: List[Path] = []
            job_args = []
            for rid in range(int(args.multi_init_runs)):
                cfg_i = dict(cfg)
                cfg_i["multi_init_enabled"] = True
                cfg_i["multi_init_run_idx"] = int(rid)
                cfg_i["seed"] = int(cfg["seed"] + rid * 100003)
                run_dir = base_out / pid / f"run_{rid:02d}"
                cfg_i["out_dir"] = str(run_dir)
                run_dirs.append(run_dir)
                job_args.append((str(hp), cfg_i, str(args.ckpt), cfg["device"]))

            if run_workers == 1:
                device = torch.device(cfg["device"])
                comp = load_comparator(
                    args.ckpt, device,
                    d_model=cfg["d_model"], nhead=cfg["nhead"],
                    num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
                    dropout=cfg["dropout"],
                )
                for _, cfg_i, _, _ in job_args:
                    run_case(hp, cfg_i, comp, device)
                    hard_cleanup()
            else:
                ctx = mp.get_context("spawn")
                with futures.ProcessPoolExecutor(max_workers=run_workers, mp_context=ctx) as ex:
                    list(ex.map(_worker_entry, job_args))

            aggregate_multi_init_risk_for_case(
                h5_path=hp,
                pid=pid,
                run_dirs=run_dirs,
                out_root=base_out,
                smooth_sigma=float(args.multi_init_risk_smooth_sigma),
                equal_volume=bool(args.multi_init_risk_equal_volume),
            )
            print(f"{pid}: multi-init aggregate risk saved under {base_out / pid / 'aggregate_multi_init'}", flush=True)
        return


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
