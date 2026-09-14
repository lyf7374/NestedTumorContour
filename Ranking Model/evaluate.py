"""Select a ranking checkpoint using fixed validation contour pairs."""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import tqdm

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "Ranking Model"

from .train import (
    build_grouped_split,
    build_model,
    h5_sort_key,
    load_pretrained,
    nearest_intensity_xyz,
    resolve_pair_counts,
    robust_normalize_np,
)


PAIR_TYPES = (
    "hard_shrink_expand",
    "core_infil_shrink_points_full",
    "infil_healthy_points_full_expand",
    "rank_points_full_points_full",
)


def stable_seed(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "little") % (2**32)


def sample_indices(rng, pool_size, n):
    if n <= 0:
        return []
    if pool_size <= 0:
        raise RuntimeError("Empty pool when sampling indices.")
    replace = pool_size < n
    return rng.choice(pool_size, size=n, replace=replace).astype(np.int64).tolist()


def load_img_vol(hf, percentile_norm):
    p_lo, p_hi = percentile_norm
    vols = []
    for key in ("flair_data", "t1", "t1gd", "t2"):
        vols.append(robust_normalize_np(hf["img"][key][()], p_lo, p_hi))
    return np.stack(vols, axis=0).astype(np.float32)


def coords_to_pc(coords_xyz, img_vol):
    ints = nearest_intensity_xyz(coords_xyz, img_vol)
    pc = np.concatenate([coords_xyz.astype(np.float32), ints], axis=1)
    return torch.from_numpy(pc)


def append_pair(g_a_list, g_b_list, y_list, type_list, left, right, y, pair_type, img_vol):
    g_a_list.append(coords_to_pc(left, img_vol))
    g_b_list.append(coords_to_pc(right, img_vol))
    y_list.append(float(y))
    type_list.append(pair_type)


def make_eval_pairs(data_dir, fname, pair_counts, percentile_norm, seed):
    rng = np.random.default_rng(stable_seed(seed, fname))
    path = os.path.join(data_dir, fname)
    with h5py.File(path, "r") as hf:
        img_vol = load_img_vol(hf, percentile_norm)
        shrink = hf["shrink"][()]
        expand = hf["expand"][()]
        pts_full = hf["points_full"][()]

    k_hard, k_core_infil, k_infil_healthy, k_rank = pair_counts
    g_a_list, g_b_list, y_list, type_list = [], [], [], []

    for i_s, i_w in zip(sample_indices(rng, len(shrink), k_hard), sample_indices(rng, len(expand), k_hard)):
        left, right, y = shrink[i_s], expand[i_w], 1.0
        if rng.random() < 0.5:
            left, right, y = right, left, 0.0
        append_pair(g_a_list, g_b_list, y_list, type_list, left, right, y, PAIR_TYPES[0], img_vol)

    for i_s, i_l in zip(sample_indices(rng, len(shrink), k_core_infil), sample_indices(rng, len(pts_full), k_core_infil)):
        left, right, y = shrink[i_s], pts_full[i_l], 1.0
        if rng.random() < 0.5:
            left, right, y = right, left, 0.0
        append_pair(g_a_list, g_b_list, y_list, type_list, left, right, y, PAIR_TYPES[1], img_vol)

    for i_l, i_w in zip(sample_indices(rng, len(pts_full), k_infil_healthy), sample_indices(rng, len(expand), k_infil_healthy)):
        left, right, y = pts_full[i_l], expand[i_w], 1.0
        if rng.random() < 0.5:
            left, right, y = right, left, 0.0
        append_pair(g_a_list, g_b_list, y_list, type_list, left, right, y, PAIR_TYPES[2], img_vol)

    rank_ids = sample_indices(rng, len(pts_full), 2 * k_rank)
    for k in range(k_rank):
        i1, i2 = rank_ids[2 * k], rank_ids[2 * k + 1]
        append_pair(
            g_a_list,
            g_b_list,
            y_list,
            type_list,
            pts_full[i1],
            pts_full[i2],
            1.0 if i1 > i2 else 0.0,
            PAIR_TYPES[3],
            img_vol,
        )

    g_a = torch.stack(g_a_list, dim=0)
    g_b = torch.stack(g_b_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.float32)
    return g_a, g_b, y, type_list


def discover_checkpoints(ckpt_dir, pattern, include_best):
    ckpt_dir = Path(ckpt_dir)
    paths = sorted(ckpt_dir.glob(pattern), key=lambda p: h5_sort_key(p.name))
    if include_best:
        best = ckpt_dir / "transformer_ft_best.pth"
        if best.exists() and best not in paths:
            paths.append(best)
    if not paths:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir} with pattern {pattern}")
    return paths


def load_train_config(ckpt_dir):
    path = Path(ckpt_dir) / "train_config.json"
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def infer_pair_counts(args, train_config):
    pc = train_config.get("pair_counts", {})
    if pc and args.K_per_type is None and all(x is None for x in (args.K_hard, args.K_core_infil, args.K_infil_healthy, args.K_rank)):
        return (
            int(pc["hard_shrink_expand"]),
            int(pc["core_infil_shrink_points_full"]),
            int(pc["infil_healthy_points_full_expand"]),
            int(pc["rank_points_full_points_full"]),
        )
    K_per_type = 16 if args.K_per_type is None else args.K_per_type
    return resolve_pair_counts(
        K_per_type,
        K_hard=args.K_hard,
        K_core_infil=args.K_core_infil,
        K_infil_healthy=args.K_infil_healthy,
        K_rank=args.K_rank,
    )


def make_model_kwargs(args, train_config):
    if train_config.get("model_kwargs"):
        return train_config["model_kwargs"]
    return {
        "input_dim": 7,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }


def init_stats():
    stats = {"loss_sum": 0.0, "correct": 0, "n": 0}
    for pair_type in PAIR_TYPES:
        stats[f"{pair_type}_loss_sum"] = 0.0
        stats[f"{pair_type}_correct"] = 0
        stats[f"{pair_type}_n"] = 0
    return stats


def update_stats(stats, losses, probs, y, type_list):
    preds = (probs >= 0.5).to(torch.float32)
    correct = (preds == y).to(torch.int64)
    losses_np = losses.detach().cpu().numpy()
    correct_np = correct.detach().cpu().numpy()
    y_n = int(y.numel())
    stats["loss_sum"] += float(losses.sum().item())
    stats["correct"] += int(correct.sum().item())
    stats["n"] += y_n
    for i, pair_type in enumerate(type_list):
        stats[f"{pair_type}_loss_sum"] += float(losses_np[i])
        stats[f"{pair_type}_correct"] += int(correct_np[i])
        stats[f"{pair_type}_n"] += 1


def finalize_stats(stats, checkpoint):
    checkpoint_path = Path(checkpoint)
    row = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "epoch": (int(re.search(r"epoch(\d+)", checkpoint_path.stem).group(1))
                  if re.search(r"epoch(\d+)", checkpoint_path.stem) else None),
        "val_loss": stats["loss_sum"] / max(1, stats["n"]),
        "val_acc": stats["correct"] / max(1, stats["n"]),
        "n_pairs": stats["n"],
    }
    for pair_type in PAIR_TYPES:
        n = stats[f"{pair_type}_n"]
        row[f"{pair_type}_loss"] = stats[f"{pair_type}_loss_sum"] / max(1, n)
        row[f"{pair_type}_acc"] = stats[f"{pair_type}_correct"] / max(1, n)
        row[f"{pair_type}_n"] = n
    return row


def evaluate_checkpoint(model, ckpt_path, val_files, args, m_kwargs, pair_counts, percentile_norm):
    model = build_model("transformer", m_kwargs).to(args.device)
    model = load_pretrained(model, str(ckpt_path), map_location=args.device, strict=True)
    model.eval()

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    use_amp = bool(args.amp) and str(args.device).startswith("cuda")
    stats = init_stats()
    with torch.no_grad():
        for fname in tqdm.tqdm(val_files, desc=f"Eval {Path(ckpt_path).name}"):
            g_a, g_b, y, type_list = make_eval_pairs(
                args.data_dir,
                fname,
                pair_counts=pair_counts,
                percentile_norm=percentile_norm,
                seed=args.eval_seed,
            )
            for start in range(0, y.numel(), args.pair_batch_size):
                end = min(start + args.pair_batch_size, y.numel())
                g_a_b = g_a[start:end].to(args.device, non_blocking=True)
                g_b_b = g_b[start:end].to(args.device, non_blocking=True)
                y_b = y[start:end].to(args.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    probs, logits = model(g_a_b, g_b_b)
                    losses = criterion(logits, y_b)
                update_stats(stats, losses, probs, y_b, type_list[start:end])
    return finalize_stats(stats, ckpt_path)


def write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BraTS transition finetune checkpoints on a fixed validation sample.")
    parser.add_argument("data_dir")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--split-json", help="Defaults to the training run's <ckpt-dir>/split.json.")
    parser.add_argument("--checkpoint-pattern", default="transformer_ft_epoch*.pth")
    parser.add_argument("--include-best", action="store_true")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision for evaluation.")
    parser.add_argument("--eval-seed", type=int, default=2027)
    parser.add_argument("--K-per-type", type=int)
    parser.add_argument("--K-hard", type=int)
    parser.add_argument("--K-core-infil", type=int)
    parser.add_argument("--K-infil-healthy", type=int)
    parser.add_argument("--K-rank", type=int)
    parser.add_argument("--percentile-low", type=int, help="Defaults to train_config.json, or 1")
    parser.add_argument("--percentile-high", type=int, help="Defaults to train_config.json, or 99")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--out-csv", help="Defaults to <ckpt-dir>/checkpoint_eval_val.csv")
    parser.add_argument("--export-best-name", default="transformer_ft_selected_by_fixed_val.pth")
    args = parser.parse_args()
    if args.pair_batch_size <= 0:
        parser.error("--pair-batch-size must be positive")

    ckpt_dir = Path(args.ckpt_dir)
    train_config = load_train_config(ckpt_dir)
    if train_config.get("model_type", "transformer") != "transformer":
        parser.error("This fixed-validation evaluator supports the Transformer recipe only")
    split_json = args.split_json or str(ckpt_dir / "split.json")
    train_files, val_files, split = build_grouped_split(args.data_dir, split_json=split_json)
    pair_counts = infer_pair_counts(args, train_config)
    m_kwargs = make_model_kwargs(args, train_config)
    saved_norm = train_config.get("percentile_norm", [1, 99])
    percentile_norm = (saved_norm[0] if args.percentile_low is None else args.percentile_low,
                       saved_norm[1] if args.percentile_high is None else args.percentile_high)
    if not 0 <= percentile_norm[0] < percentile_norm[1] <= 100:
        parser.error("Normalization percentiles must satisfy 0 <= low < high <= 100")
    checkpoints = discover_checkpoints(ckpt_dir, args.checkpoint_pattern, args.include_best)

    print(f"Validation files: {len(val_files)} from {split.get('n_val_cases')} cases")
    print(f"Pair counts per H5: {pair_counts}")
    print(f"Checkpoints: {len(checkpoints)}")

    rows = []
    for ckpt_path in checkpoints:
        row = evaluate_checkpoint(
            model=None,
            ckpt_path=ckpt_path,
            val_files=val_files,
            args=args,
            m_kwargs=m_kwargs,
            pair_counts=pair_counts,
            percentile_norm=percentile_norm,
        )
        rows.append(row)

    rows.sort(key=lambda r: (r["val_loss"], -r["val_acc"]))
    out_csv = Path(args.out_csv) if args.out_csv else ckpt_dir / "checkpoint_eval_val.csv"
    write_csv(out_csv, rows)

    best = rows[0]
    best_src = Path(best["checkpoint"])
    best_dst = ckpt_dir / args.export_best_name
    if best_src.resolve() != best_dst.resolve():
        shutil.copyfile(best_src, best_dst)
    summary = {
        "best_checkpoint": str(best_src),
        "exported_best": str(best_dst),
        "selection": "minimum fixed validation BCE loss; ties prefer higher accuracy",
        "best_metrics": best,
        "pair_counts": dict(zip(PAIR_TYPES, pair_counts)),
        "eval_seed": args.eval_seed,
        "csv": str(out_csv),
    }
    with open(ckpt_dir / "checkpoint_eval_val_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(ckpt_dir / "best_checkpoint.txt", "w") as f:
        f.write(str(best_src) + "\n")
        f.write(str(best_dst) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
