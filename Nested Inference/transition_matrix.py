"""
TransitionMatrixNew v3: radial contour evolution with motion-mask feasibility,
WM/GM-aware transition matrix mixing, lag catch-up, and spike suppression.
Supports both shrink and expand modes.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    map_coordinates,
    binary_fill_holes,
    binary_closing,
    binary_dilation,
    generate_binary_structure,
    distance_transform_edt,
)

__all__ = [
    "_interp_trilinear",
    "_interps_trilinear",
    "make_motion_mask",
    "project_point_to_mask_fast",
    "init_radial_shell",
    "precompute_prefix_ray_caps",
    "neighbor_mean_4",
    "activation_field",
    "local_mix_probs_4nbr",
    "radial_shrink_points_transition_matrix_v3",
    "radial_expand_points_transition_matrix_v3",
    "radial_shrink_points_transition_matrix",
]


# =========================================================
# Trilinear interpolation helpers (voxel space)
# =========================================================
def _interp_trilinear(vol, xyz):
    """vol: (X,Y,Z), xyz: (3,) voxel-float -> scalar"""
    x, y, z = xyz
    return float(
        map_coordinates(vol, [[x], [y], [z]], order=1, mode="nearest").ravel()[0]
    )


def _interps_trilinear(vol, xyz_batch):
    """vol: (X,Y,Z), xyz_batch: (N,3) voxel-float -> (N,)"""
    xyz_batch = np.asarray(xyz_batch, dtype=np.float64)
    xs, ys, zs = xyz_batch[:, 0], xyz_batch[:, 1], xyz_batch[:, 2]
    return map_coordinates(vol, [xs, ys, zs], order=1, mode="nearest")


# =========================================================
# Motion mask (hard feasibility set) construction
# =========================================================
def make_motion_mask(brain_mask, seg=None, close_iter=2, dil_iter=1):
    """
    Build a hard feasibility mask for radial shrink.
    - Includes intracranial region and (optionally) tumor seg.
    - closing + fill_holes reduce thin gaps that break radial traversal.
    """
    m = (brain_mask > 0)
    if seg is not None:
        m = m | (seg > 0)

    st = generate_binary_structure(3, 2)  # 26-connectivity
    if close_iter and close_iter > 0:
        m = binary_closing(m, structure=st, iterations=int(close_iter))
    m = binary_fill_holes(m)

    if dil_iter and dil_iter > 0:
        m = binary_dilation(m, structure=st, iterations=int(dil_iter))

    return m.astype(np.uint8)


def project_point_to_mask_fast(point_vox, mask_bool, max_radius=40):
    """
    Project point_vox to the nearest True voxel in mask_bool via local search.
    If already inside, return the input coordinates.
    """
    p = np.asarray(point_vox, dtype=np.float64)
    X, Y, Z = mask_bool.shape
    c = np.round(p).astype(int)
    c[0] = np.clip(c[0], 0, X - 1)
    c[1] = np.clip(c[1], 0, Y - 1)
    c[2] = np.clip(c[2], 0, Z - 1)

    if mask_bool[c[0], c[1], c[2]]:
        return p

    best = None
    for r in range(1, int(max_radius) + 1):
        x0, x1 = max(c[0] - r, 0), min(c[0] + r, X - 1)
        y0, y1 = max(c[1] - r, 0), min(c[1] + r, Y - 1)
        z0, z1 = max(c[2] - r, 0), min(c[2] + r, Z - 1)
        sub = mask_bool[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1]
        if not np.any(sub):
            continue
        idx = np.argwhere(sub)
        idx[:, 0] += x0
        idx[:, 1] += y0
        idx[:, 2] += z0
        d2 = np.sum((idx - c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d2))
        best = idx[k]
        break

    if best is None:
        return p
    return best.astype(np.float64)


# =========================================================
# Radial shell init
# =========================================================
def init_radial_shell(points_xyz_float, zooms, center_xyz):
    """
    Fixed direction unit vectors (in physical space) and radii from a center.
    """
    pts_vox = np.asarray(points_xyz_float, dtype=np.float64)
    zooms = np.asarray(zooms, dtype=np.float64)

    center_vox = np.asarray(center_xyz, dtype=np.float64)
    center_phys = center_vox * zooms

    pts_phys = pts_vox * zooms
    vec = pts_phys - center_phys[None, :]
    r = np.linalg.norm(vec, axis=1)
    dirs = vec / (r[:, None] + 1e-12)

    bad = r < 1e-6
    if np.any(bad):
        dirs[bad] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        r[bad] = 0.0

    return center_vox, center_phys, dirs, r


# =========================================================
# Precompute prefix ray-caps for radial feasibility
# =========================================================
def precompute_prefix_ray_caps(
    center_phys,
    dirs_phys,
    zooms,
    motion_mask,
    r_max_phys,
    dr_probe=0.5,
    margin_mm=0.2,
    chunk=256,
):
    """
    For each ray compute r_cap: largest radius such that [0, r_cap] stays inside
    motion_mask. Guarantees feasible inward moves by clamping to prefix caps.
    """
    zooms = np.asarray(zooms, dtype=np.float64)
    mask_f = motion_mask.astype(np.float32, copy=False)

    radii = np.arange(0.0, float(r_max_phys) + 1e-6, float(dr_probe), dtype=np.float64)
    M = len(radii)
    N = dirs_phys.shape[0]

    r_cap = np.zeros(N, dtype=np.float64)

    for i0 in range(0, N, int(chunk)):
        i1 = min(i0 + int(chunk), N)
        dirs = dirs_phys[i0:i1]
        B = dirs.shape[0]

        pts_phys = center_phys[None, None, :] + dirs[:, None, :] * radii[None, :, None]
        pts_vox = pts_phys / zooms[None, None, :]
        v = _interps_trilinear(mask_f, pts_vox.reshape(-1, 3)).reshape(B, M)
        inside = v > 0.5

        for b in range(B):
            if not inside[b, 0]:
                cap = 0.0
            else:
                idx_false = np.where(~inside[b])[0]
                if idx_false.size == 0:
                    cap = radii[-1]
                else:
                    ff = int(idx_false[0])
                    cap = radii[ff - 1] if ff > 0 else 0.0

            cap = max(0.0, cap - float(margin_mm))
            r_cap[i0 + b] = cap

    return r_cap


# =========================================================
# Grid utilities: smoothness, activation, local mixing
# =========================================================
def neighbor_mean_4(grid2d):
    """4-neighbor mean with boundary-aware normalization."""
    g = np.asarray(grid2d, dtype=np.float64)
    out = np.zeros_like(g, dtype=np.float64)
    cnt = np.zeros_like(g, dtype=np.float64)

    out[1:] += g[:-1]
    cnt[1:] += 1
    out[:-1] += g[1:]
    cnt[:-1] += 1
    out[:, 1:] += g[:, :-1]
    cnt[:, 1:] += 1
    out[:, :-1] += g[:, 1:]
    cnt[:, :-1] += 1

    out /= (cnt + 1e-12)
    return out


def activation_field(side, anchors, sigma=2.0, s_min=0.25):
    """
    anchor -> smooth activation field s(i) in [s_min, 1].
    Used to soften move candidates around sampled anchors.
    """
    if anchors is None:
        return np.ones((side, side), dtype=np.float64)

    a = np.zeros((side, side), dtype=bool)
    a.flat[np.asarray(anchors, dtype=int)] = True
    dist = distance_transform_edt(~a).astype(np.float64)
    s = np.exp(-(dist * dist) / (2.0 * float(sigma) * float(sigma)))
    s = float(s_min) + (1.0 - float(s_min)) * s
    return s


def local_mix_probs_4nbr(P0, wm_grid, gm_grid, beta_aff=8.0):
    """
    4-neighbor tissue-aware mixing on transition matrix:
      P_avg(i) = normalize( P0(i) + Σ w_ij P0(j) )
    """
    side, _, K = P0.shape
    P_sum = P0.copy()
    W_sum = np.ones((side, side), dtype=np.float64)

    diff = np.abs(wm_grid[1:] - wm_grid[:-1]) + np.abs(gm_grid[1:] - gm_grid[:-1])
    w = np.exp(-float(beta_aff) * diff)
    P_sum[1:] += w[..., None] * P0[:-1]
    W_sum[1:] += w
    P_sum[:-1] += w[..., None] * P0[1:]
    W_sum[:-1] += w

    diff = np.abs(wm_grid[:, 1:] - wm_grid[:, :-1]) + np.abs(gm_grid[:, 1:] - gm_grid[:, :-1])
    w = np.exp(-float(beta_aff) * diff)
    P_sum[:, 1:] += w[..., None] * P0[:, :-1]
    W_sum[:, 1:] += w
    P_sum[:, :-1] += w[..., None] * P0[:, 1:]
    W_sum[:, :-1] += w

    P_avg = P_sum / (W_sum[..., None] + 1e-12)
    P_avg = P_avg / (P_avg.sum(axis=2, keepdims=True) + 1e-12)
    return P_avg


# =========================================================
# Final: radial contour evolution with transition-matrix guidance (v3)
# =========================================================
def radial_shrink_points_transition_matrix_v3(
    points_xyz_float,
    wm,
    gm,
    brain_mask,
    seg=None,
    zooms=(1.0, 1.0, 1.0),
    center_xyz=None,
    n_steps=50,
    eps_phys=0.2,
    eps_floor=0.05,
    step_fracs=(0.0, 0.05, 0.10, 0.25, 0.50, 1.0),
    R=3.0,
    perm_floor=0.08,
    stay_floor=1e-6,
    lambda0=0.50,
    lambda_min=0.15,
    lambda_max=0.90,
    beta_aff=8.0,
    move_k=256,
    act_sigma=2.0,
    kappa_lag=3.0,
    lag_no_stay_mm=0.25,
    cap_mm=0.20,
    gamma_cap=8.0,
    project_after=True,
    project_iters=1,
    motion_close_iter=2,
    motion_dil_iter=1,
    dr_probe=0.5,
    cap_margin_mm=0.2,
    rng=None,
    debug=False,
    mode="shrink",
):
    """
    Radial contour evolution that only updates radii.
    - hard feasibility: motion_mask (brain_mask ∪ seg) + prefix ray-cap
    - soft weights: WM/GM -> perm_mid
    - transition coupling: local probability mixing (not displacement diffusion)
    - spike control: lag catch-up + neighbor cap prior + mode-specific projection
    mode:
      - "shrink": inward evolution (legacy behavior)
      - "expand": outward evolution
    Returns traj list, each entry shape (N,3) voxel-float.
    """
    if rng is None:
        rng = np.random.default_rng(2025)

    pts0 = np.asarray(points_xyz_float, dtype=np.float64)
    N = pts0.shape[0]
    side = int(round(np.sqrt(N)))
    if side * side != N:
        raise ValueError(f"Need square grid, got N={N}")

    zooms = np.asarray(zooms, dtype=np.float64)

    if center_xyz is None:
        raise ValueError("center_xyz is required (tumor center) for radial scheme.")

    # 1) motion mask (hard feasibility domain)
    motion_mask = make_motion_mask(
        brain_mask, seg=seg, close_iter=motion_close_iter, dil_iter=motion_dil_iter
    )
    motion_bool = motion_mask > 0

    # 2) ensure center is inside mask
    center_xyz = np.asarray(center_xyz, dtype=np.float64)
    center_xyz = project_point_to_mask_fast(center_xyz, motion_bool, max_radius=40)

    # 3) radial init
    center_vox, center_phys, dirs_phys, r_phys = init_radial_shell(pts0, zooms, center_xyz)

    # 4) precompute prefix ray caps
    r_max = float(np.max(r_phys) + 20.0)
    r_cap = precompute_prefix_ray_caps(
        center_phys=center_phys,
        dirs_phys=dirs_phys,
        zooms=zooms,
        motion_mask=motion_mask,
        r_max_phys=r_max,
        dr_probe=dr_probe,
        margin_mm=cap_margin_mm,
        chunk=256,
    )

    # 5) clamp initial shell to caps
    r_phys = np.minimum(r_phys, r_cap)

    mode = str(mode).strip().lower()
    if mode not in ("shrink", "expand"):
        raise ValueError(f"Unknown mode: {mode} (expected 'shrink' or 'expand')")
    is_expand = mode == "expand"

    fracs = np.abs(np.asarray(step_fracs, dtype=np.float64))
    K = len(fracs)

    s_min = float(eps_floor) / (float(eps_phys) + 1e-12)
    s_min = np.clip(s_min, 0.01, 1.0)

    traj = []

    for t in range(int(n_steps) + 1):
        pts_phys = center_phys[None, :] + dirs_phys * r_phys[:, None]
        pts_vox = pts_phys / zooms[None, :]
        traj.append(pts_vox.copy())

        if t == n_steps:
            break

        anchors = None if move_k is None else rng.choice(N, size=min(int(move_k), N), replace=False)
        s_grid = activation_field(side, anchors, sigma=act_sigma, s_min=s_min)
        s = s_grid.reshape(-1)

        r_grid = r_phys.reshape(side, side)
        rbar_grid = neighbor_mean_4(r_grid)
        rbar = rbar_grid.reshape(-1)
        # In shrink, lag means "point protrudes outward vs neighborhood".
        # In expand, lag means "point lags inward vs neighborhood".
        lag = (rbar - r_phys) if is_expand else (r_phys - rbar)
        lag_pos = np.maximum(lag, 0.0)

        # dr < 0 for shrink, dr > 0 for expand.
        move_sign = 1.0 if is_expand else -1.0
        dr = move_sign * (float(eps_phys) * fracs[None, :]) * s[:, None]
        dr[:, 0] = 0.0

        r_cand = r_phys[:, None] + dr
        valid = (r_cand >= 0.0) & (r_cand <= r_cap[:, None] + 1e-12)

        cand_phys = center_phys[None, None, :] + dirs_phys[:, None, :] * r_cand[:, :, None]
        cand_vox = cand_phys / zooms[None, None, :]
        cand_flat = cand_vox.reshape(-1, 3)

        wm_c = _interps_trilinear(wm, cand_flat).astype(np.float64).reshape(N, K)
        gm_c = _interps_trilinear(gm, cand_flat).astype(np.float64).reshape(N, K)

        wm_p = wm_c[:, 0]
        gm_p = gm_c[:, 0]
        perm_p = wm_p + gm_p / (float(R) + 1e-12)
        perm_c = wm_c + gm_c / (float(R) + 1e-12)
        perm_mid = 0.5 * (perm_p[:, None] + perm_c)

        w = np.zeros((N, K), dtype=np.float64)
        w[:, 0] = float(stay_floor)

        step_mag = np.abs(dr) / (float(eps_phys) + 1e-12)
        w[:, 1:] = (perm_mid[:, 1:] + float(perm_floor)) * step_mag[:, 1:]

        over = np.maximum(r_cand - (rbar[:, None] + float(cap_mm)), 0.0)
        cap_prior = np.exp(-float(gamma_cap) * (over / (float(cap_mm) + 1e-12)) ** 2)
        w *= cap_prior

        lag_scale = lag_pos / (float(eps_phys) + 1e-12)
        exp_clip = 60.0  # exp(60) ~= 1e26; prevents overflow.
        expo_stay = np.clip(-float(kappa_lag) * lag_scale, -exp_clip, exp_clip)
        expo_move = np.clip(+float(kappa_lag) * lag_scale[:, None] * step_mag[:, 1:], -exp_clip, exp_clip)
        w[:, 0] *= np.exp(expo_stay)
        w[:, 1:] *= np.exp(expo_move)

        if lag_no_stay_mm is not None and float(lag_no_stay_mm) > 0:
            bad = lag_pos > float(lag_no_stay_mm)
            w[bad, 0] = 0.0

        w *= valid
        # Numerical guard: remove NaN/Inf and keep non-negative finite weights.
        w = np.nan_to_num(w, nan=0.0, posinf=1e30, neginf=0.0)
        w = np.clip(w, 0.0, 1e30)

        has_move = valid[:, 1:].any(axis=1)
        w[~has_move, :] = 0.0
        w[~has_move, 0] = 1.0

        # Avoid eager evaluation warning from np.where(w / S) when S==0.
        S = w.sum(axis=1, keepdims=True)
        P_base = np.zeros_like(w, dtype=np.float64)
        good_rows = np.isfinite(S[:, 0]) & (S[:, 0] > 0)
        if np.any(good_rows):
            P_base[good_rows] = w[good_rows] / (S[good_rows] + 1e-12)
        if np.any(~good_rows):
            P_base[~good_rows, 0] = 1.0

        P0 = P_base.reshape(side, side, K)
        wm_grid = wm_p.reshape(side, side)
        gm_grid = gm_p.reshape(side, side)
        P_avg = local_mix_probs_4nbr(P0, wm_grid, gm_grid, beta_aff=beta_aff)

        lam = np.clip(float(lambda_min) + float(lambda0) * s_grid, 0.0, float(lambda_max))
        P_final = (1.0 - lam[..., None]) * P0 + lam[..., None] * P_avg

        valid_grid = valid.reshape(side, side, K)
        P_final *= valid_grid
        # Stable renormalization with finite fallback-to-stay rows.
        P_final = np.nan_to_num(P_final, nan=0.0, posinf=0.0, neginf=0.0)
        P_flat = P_final.reshape(N, K)
        S_flat = P_flat.sum(axis=1, keepdims=True)
        good_pf = np.isfinite(S_flat[:, 0]) & (S_flat[:, 0] > 0)
        if np.any(good_pf):
            P_flat[good_pf] = P_flat[good_pf] / (S_flat[good_pf] + 1e-12)
        if np.any(~good_pf):
            P_flat[~good_pf, :] = 0.0
            P_flat[~good_pf, 0] = 1.0
        P_final = P_flat

        if debug:
            no_move = int(np.sum(~has_move))
            mean_stay = float(P_final[:, 0].mean())
            max_stay = float(P_final[:, 0].max())
            print(
                f"[t={t}] no-move: {no_move}/{N}  mean_stay:{mean_stay:.6f}  max_stay:{max_stay:.6f}"
            )

        u = rng.random(N)
        cdf = np.cumsum(P_final, axis=1)
        choice = (u[:, None] <= cdf).argmax(axis=1)
        dr_sel = dr[np.arange(N), choice]

        r_phys = np.maximum(r_phys + dr_sel, 0.0)
        r_phys = np.minimum(r_phys, r_cap)

        if project_after:
            r_grid = r_phys.reshape(side, side)
            for _ in range(int(project_iters)):
                rbar_grid = neighbor_mean_4(r_grid)
                if is_expand:
                    # Suppress inward pits while allowing outward growth.
                    r_grid = np.maximum(r_grid, rbar_grid - float(cap_mm))
                else:
                    # Suppress outward spikes while shrinking.
                    r_grid = np.minimum(r_grid, rbar_grid + float(cap_mm))
            r_phys = r_grid.reshape(-1)
            r_phys = np.minimum(r_phys, r_cap)

    return traj


def radial_expand_points_transition_matrix_v3(*args, **kwargs):
    """Explicit expand-mode entrypoint."""
    kwargs = dict(kwargs)
    kwargs["mode"] = "expand"
    return radial_shrink_points_transition_matrix_v3(*args, **kwargs)


# Backward compatibility alias
radial_shrink_points_transition_matrix = radial_shrink_points_transition_matrix_v3
