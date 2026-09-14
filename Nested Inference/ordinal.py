"""Original shell-to-volume ordinal rasterization; legacy names retained for compatibility."""
from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

def shells_to_prob_volume(
    shells: np.ndarray,
    shape: Tuple[int, int, int],
    smooth_sigma: float = 1.0,
    equal_volume: bool = False,
) -> np.ndarray:
    """
    Legacy ordinal rasterization inside the surface bounding box (not a probability).
    shell i receives probability (i+1)/N; voxels touched by multiple shells keep max prob.
    Interior voxels take the probability of the nearest shell voxel (EDT fill).
    """
    shells = np.asarray(shells)
    if shells.ndim != 3 or shells.shape[0] < 1 or shells.shape[2] != 3 or not np.isfinite(shells).all():
        raise ValueError("shells must be finite with shape (T,N,3), T>=1")
    if len(shape) != 3 or any(int(s) <= 0 for s in shape):
        raise ValueError("shape must contain three positive dimensions")
    prob = np.zeros(shape, dtype=np.float32)
    n_shells = shells.shape[0]
    all_ijk = []
    # optional equal-volume mapping: assign probabilities after painting based on voxel area per shell
    shell_weights = None
    if equal_volume:
        shell_weights = np.zeros(n_shells, dtype=np.float64)

    for i, coords in enumerate(shells):
        p = float(i + 1) / float(n_shells)
        ijk = np.round(coords).astype(np.int64)
        x, y, z = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        valid = (
            (0 <= x) & (x < shape[0]) &
            (0 <= y) & (y < shape[1]) &
            (0 <= z) & (z < shape[2])
        )
        if not np.any(valid):
            continue
        xv, yv, zv = x[valid], y[valid], z[valid]
        all_ijk.append(np.stack([xv, yv, zv], axis=1))
        current = prob[xv, yv, zv]
        if equal_volume:
            shell_weights[i] = len(xv)
        prob[xv, yv, zv] = np.maximum(current, p)

    if all_ijk:
        all_ijk = np.concatenate(all_ijk, axis=0)
        mins = np.maximum(all_ijk.min(axis=0) - 2, 0)
        maxs = np.minimum(all_ijk.max(axis=0) + 3, np.array(shape))
        sx = slice(int(mins[0]), int(maxs[0]))
        sy = slice(int(mins[1]), int(maxs[1]))
        sz = slice(int(mins[2]), int(maxs[2]))
        shell_mask = prob[sx, sy, sz] > 0
    else:
        shell_mask = None

    if shell_mask is not None and np.any(shell_mask):
        from scipy.ndimage import distance_transform_edt

        sub = prob[sx, sy, sz]
        distances, indices = distance_transform_edt(
            ~shell_mask, return_distances=True, return_indices=True
        )
        filled = sub.copy()
        fill_mask = ~shell_mask
        if np.any(fill_mask):
            idx0 = indices[0][fill_mask]
            idx1 = indices[1][fill_mask]
            idx2 = indices[2][fill_mask]
            filled[fill_mask] = sub[idx0, idx1, idx2]
        if smooth_sigma and smooth_sigma > 0:
            filled = gaussian_filter(filled, sigma=float(smooth_sigma))
        prob[sx, sy, sz] = filled

    # remap to equal-volume percentiles if requested
    if equal_volume and np.any(prob > 0):
        flat = prob.ravel()
        mask = flat > 0
        ranks = np.argsort(flat[mask])
        frac = np.linspace(1.0 / ranks.size, 1.0, ranks.size, dtype=np.float64)
        mapped = flat[mask].copy()
        mapped[ranks] = frac
        flat = flat.copy()
        flat[mask] = mapped
        prob = flat.reshape(shape)
    return prob
