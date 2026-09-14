"""Reference-shell geometry for aligned NIfTI inputs."""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt, map_coordinates

def center_from_seg_xyz(seg_xyz, labels=(1, 4)):
    mask = np.isin(seg_xyz, list(labels))
    if not np.any(mask):
        mask = np.isin(seg_xyz, [1, 2, 4])
        if not np.any(mask):
            mask = seg_xyz != 0
    cx, cy, cz = center_of_mass(mask.astype(np.float32))
    return np.array([cx, cy, cz], dtype=np.float32)


def zooms_from_affine(aff_4x4):
    A = np.asarray(aff_4x4, dtype=np.float64)[:3, :3]
    return np.sqrt((A * A).sum(axis=0)) + 1e-12


def build_directions_equirect(n_theta=64, n_phi=64):
    theta_edges = np.linspace(0, 2 * np.pi, n_theta + 1)
    phi_edges = np.linspace(0, np.pi, n_phi + 1)
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
            radii[i] = 0.0
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
            radii[i] = 0.0
            continue

        k = int(np.max(np.where(inside)[0]))
        if k < len(rs) - 1 and inside[k] and (not inside[k + 1]):
            v1, v2 = float(vals[k]), float(vals[k + 1])
            r1, r2 = float(rs[k]), float(rs[k + 1])
            if abs(v2 - v1) > 1e-8:
                r_star = r1 + (0.5 - v1) * (r2 - r1) / (v2 - v1)
            else:
                r_star = r1
        else:
            r_star = float(rs[k])
        radii[i] = max(0.0, float(r_star))

    return radii


def nearest_mask_voxel(mask_xyz: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask_xyz > 0)
    if pts.shape[0] == 0:
        raise RuntimeError("Mask is empty; cannot find nearest voxel.")
    d2 = np.sum((pts.astype(np.float32) - query_xyz[None, :]) ** 2, axis=1)
    return pts[int(np.argmin(d2))].astype(np.float32)


def build_ctv_mask(
    seg_xyz: np.ndarray,
    brain_mask_xyz: np.ndarray,
    zooms_xyz: np.ndarray,
    base_labels: Sequence[int],
    margin_mm: float,
) -> np.ndarray:
    brain = brain_mask_xyz > 0
    base_raw = np.isin(seg_xyz, list(base_labels))
    if not np.any(base_raw):
        raise ValueError(f"No voxels found for base labels: {list(base_labels)}")

    # Compute CTV expansion from the original tumor labels, then hard-clip to brain.
    dist = distance_transform_edt(~base_raw, sampling=zooms_xyz.astype(np.float64))
    ctv = (dist <= float(margin_mm)) & brain

    # Ensure the clipped tumor core remains included after clipping.
    base_in_brain = base_raw & brain
    ctv |= base_in_brain
    ctv &= brain

    if not np.any(ctv):
        raise RuntimeError("CTV mask is empty after clipping to brain mask.")
    return ctv.astype(np.uint8)


def generate_initial_contours(
    seg_xyz: np.ndarray,
    brain_mask_xyz: np.ndarray,
    zooms_xyz: np.ndarray,
    K: int,
    ctv_margin_mm: float,
    init_shrink_mm: float,
    n_theta: int,
    n_phi: int,
    ray_step_mm: float,
    base_labels: Sequence[int],
    center_labels: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctv15 = build_ctv_mask(seg_xyz, brain_mask_xyz, zooms_xyz, base_labels, ctv_margin_mm)
    if np.any((ctv15 > 0) & (brain_mask_xyz == 0)):
        raise RuntimeError("CTV15 contains voxels outside brain_mask.")
    center = center_from_seg_xyz(seg_xyz, labels=center_labels)
    center_idx = np.rint(center).astype(np.int32)
    center_idx[0] = np.clip(center_idx[0], 0, seg_xyz.shape[0] - 1)
    center_idx[1] = np.clip(center_idx[1], 0, seg_xyz.shape[1] - 1)
    center_idx[2] = np.clip(center_idx[2], 0, seg_xyz.shape[2] - 1)
    if brain_mask_xyz[tuple(center_idx)] == 0:
        center = nearest_mask_voxel(brain_mask_xyz, center)
        center_idx = np.rint(center).astype(np.int32)
        center_idx[0] = np.clip(center_idx[0], 0, seg_xyz.shape[0] - 1)
        center_idx[1] = np.clip(center_idx[1], 0, seg_xyz.shape[1] - 1)
        center_idx[2] = np.clip(center_idx[2], 0, seg_xyz.shape[2] - 1)
    if ctv15[tuple(center_idx)] == 0:
        center = nearest_mask_voxel(ctv15, center)

    dirs_xyz = build_directions_equirect(n_theta=n_theta, n_phi=n_phi)
    ctv_radii_mm = contour_radii_mm_from_mask(
        ctv15, center, dirs_xyz, zooms_xyz=zooms_xyz, dr_mm=ray_step_mm
    )
    brain_radii_mm = contour_radii_mm_from_mask(
        (brain_mask_xyz > 0).astype(np.uint8), center, dirs_xyz, zooms_xyz=zooms_xyz, dr_mm=ray_step_mm
    )
    ctv_radii_mm = np.minimum(ctv_radii_mm, brain_radii_mm)
    if np.max(ctv_radii_mm) <= 0:
        raise RuntimeError("Failed to build contour from CTV15 mask (all radii are zero).")

    d_vox_per_mm = dirs_xyz / (zooms_xyz[None, :] + 1e-12)
    contours = []
    for i in range(K):
        radii_i = np.maximum(ctv_radii_mm - float(i) * float(init_shrink_mm), 0.0)
        radii_i = np.minimum(radii_i, brain_radii_mm)
        pts = center[None, :] + radii_i[:, None] * d_vox_per_mm
        pts[:, 0] = np.clip(pts[:, 0], 0, seg_xyz.shape[0] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, seg_xyz.shape[1] - 1)
        pts[:, 2] = np.clip(pts[:, 2], 0, seg_xyz.shape[2] - 1)
        contours.append(pts.astype(np.float32))

    return np.stack(contours, axis=0), center.astype(np.float32), ctv15.astype(np.uint8)
