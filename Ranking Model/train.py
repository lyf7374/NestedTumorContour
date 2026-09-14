"""Train the pairwise contour-ranking model from patient-grouped HDF5 data."""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import tqdm
import h5py

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "Ranking Model"

from .models import build_model, load_pretrained

# =========================
# Utilities
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def robust_normalize_np(u, p1=1, p99=99):
    """Per-channel robust min–max to [0,1]."""
    vals = u[np.isfinite(u)]
    if vals.size == 0:
        return np.zeros_like(u, dtype=np.float32)
    lo, hi = np.percentile(vals, [p1, p99])
    if hi <= lo:
        return np.clip(u, 0, 1).astype(np.float32)
    return np.clip((u - lo) / (hi - lo), 0, 1).astype(np.float32)

def nearest_intensity_xyz(coords_xyz, img_vol_xyz4):
    """
    Nearest-neighbour sample 4 modalities on XYZ grid.
    coords_xyz: [N,3] float (x,y,z)
    img_vol_xyz4: [4, X, Y, Z]
    returns: [N,4]
    """
    idx = np.rint(coords_xyz).astype(np.int32)
    X, Y, Z = img_vol_xyz4.shape[1:]
    idx[:, 0] = np.clip(idx[:, 0], 0, X - 1)  # x
    idx[:, 1] = np.clip(idx[:, 1], 0, Y - 1)  # y
    idx[:, 2] = np.clip(idx[:, 2], 0, Z - 1)  # z
    x, y, z = idx.T
    vals = img_vol_xyz4[:, x, y, z]  # [4, N]
    return vals.T.astype(np.float32)

# =========================
# Finetune Dataset (XYZ)
# =========================

class MultiPairPerPatientDataset_Finetune(Dataset):
    """
    每个 .h5（一个 case 的一个随机中心）生成若干 pair：
      1) (shrink,    expand)      baseline y=1，50% 概率对调并翻转
      2) (shrink,    points_full) baseline y=1，50% 概率对调并翻转
      3) (points_full, expand)    baseline y=1，50% 概率对调并翻转
      4) (points_full, points_full) 用索引比较：左索引>右索引 → y=1，否则 0
    返回：
      g_a_pack: [Npair, 4096, 7], g_b_pack: [Npair, 4096, 7], y_pack: [Npair]
    """
    def __init__(self, h5_dir, K=1, file_list=None,
                 use_modalities=("flair_data","t1","t1gd","t2"),
                 percentile_norm=(1, 99),
                 pair_counts=None):
        super().__init__()
        self.h5_dir = h5_dir
        self.K = int(K)
        if pair_counts is None:
            pair_counts = (self.K, self.K, self.K, self.K)
        if len(pair_counts) != 4:
            raise ValueError("pair_counts must contain 4 integers.")
        self.pair_counts = tuple(int(x) for x in pair_counts)
        if min(self.pair_counts) < 0:
            raise ValueError("pair_counts values must be >= 0.")
        if sum(self.pair_counts) <= 0:
            raise ValueError("At least one pair count must be > 0.")
        self.use_modalities = tuple(use_modalities)
        self.p_lo, self.p_hi = percentile_norm

        if file_list is None:
            all_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
            def _key(x):
                base = os.path.splitext(x)[0]
                toks = [t for t in base.replace('-', '_').split('_') if t.isdigit()]
                return (0, int(toks[-1]), base) if toks else (1, 0, base)
            all_files.sort(key=_key)
            self.files = all_files
        else:
            try:
                base_list = file_list.dataset.files
                self.files = [base_list[i] for i in file_list.indices]
            except AttributeError:
                self.files = list(file_list)

    def __len__(self):
        return len(self.files)

    def _load_and_norm_modalities(self, hf):
        vols = []
        for key in self.use_modalities:
            v = hf['img'][key][()]  # (X,Y,Z)
            v = robust_normalize_np(v, self.p_lo, self.p_hi)
            vols.append(v)
        img_vol = np.stack(vols, axis=0).astype(np.float32)  # [4, X, Y, Z]
        return img_vol

    def _coords_to_pc_xyz(self, coords_xyz, img_vol_xyz4):
        ints = nearest_intensity_xyz(coords_xyz, img_vol_xyz4)  # [4096,4]
        pc = np.concatenate([coords_xyz.astype(np.float32), ints], axis=1)  # [4096,7]
        return torch.from_numpy(pc)

    def _safe_sample_indices(self, pool_size, n):
        if n <= 0:
            return []
        if pool_size <= 0:
            raise RuntimeError("Empty pool when sampling indices.")
        if pool_size >= n:
            return random.sample(range(pool_size), n)
        else:
            return [random.randrange(pool_size) for _ in range(n)]

    def __getitem__(self, idx):
        fname = self.files[idx]
        path  = os.path.join(self.h5_dir, fname)

        with h5py.File(path, "r") as hf:
            img_vol  = self._load_and_norm_modalities(hf)  # [4,X,Y,Z]
            shrink   = hf['shrink'][()]       # (Ns,4096,3)  ~ TC_s
            expand   = hf['expand'][()]       # (Nw,4096,3)  ~ WT_l
            pts_full = hf['points_full'][()]  # (Nl,4096,3)  ~ TC_l (索引严格递增)

            Ns, Nw, Nl = len(shrink), len(expand), len(pts_full)
            k_hard, k_core_infil, k_infil_healthy, k_rank = self.pair_counts

            idx_s_1 = self._safe_sample_indices(Ns, k_hard)     # 1) s vs e
            idx_w_1 = self._safe_sample_indices(Nw, k_hard)

            idx_s_2 = self._safe_sample_indices(Ns, k_core_infil)     # 2) s vs l
            idx_l_2 = self._safe_sample_indices(Nl, k_core_infil)

            idx_l_3 = self._safe_sample_indices(Nl, k_infil_healthy)     # 3) l vs e
            idx_w_3 = self._safe_sample_indices(Nw, k_infil_healthy)

            idx_l_4 = self._safe_sample_indices(Nl, 2 * k_rank) # 4) l vs l

            g_a_list, g_b_list, y_list = [], [], []

            # (1)
            for i_s, i_w in zip(idx_s_1, idx_w_1):
                c_left, c_right = shrink[i_s], expand[i_w]
                y = 1.0
                if random.random() < 0.5:
                    c_left, c_right = c_right, c_left; y = 0.0
                g_a_list.append(self._coords_to_pc_xyz(c_left,  img_vol))
                g_b_list.append(self._coords_to_pc_xyz(c_right, img_vol))
                y_list.append(y)

            # (2)
            for i_s, i_l in zip(idx_s_2, idx_l_2):
                c_left, c_right = shrink[i_s], pts_full[i_l]
                y = 1.0
                if random.random() < 0.5:
                    c_left, c_right = c_right, c_left; y = 0.0
                g_a_list.append(self._coords_to_pc_xyz(c_left,  img_vol))
                g_b_list.append(self._coords_to_pc_xyz(c_right, img_vol))
                y_list.append(y)

            # (3)
            for i_l, i_w in zip(idx_l_3, idx_w_3):
                c_left, c_right = pts_full[i_l], expand[i_w]
                y = 1.0
                if random.random() < 0.5:
                    c_left, c_right = c_right, c_left; y = 0.0
                g_a_list.append(self._coords_to_pc_xyz(c_left,  img_vol))
                g_b_list.append(self._coords_to_pc_xyz(c_right, img_vol))
                y_list.append(y)

            # (4) 索引严格递增 → 用索引比较决定 y
            for k in range(k_rank):
                i1 = idx_l_4[2*k]
                i2 = idx_l_4[2*k + 1]
                c_left, c_right = pts_full[i1], pts_full[i2]
                y = 1.0 if i1 > i2 else 0.0
                g_a_list.append(self._coords_to_pc_xyz(c_left,  img_vol))
                g_b_list.append(self._coords_to_pc_xyz(c_right, img_vol))
                y_list.append(y)

        g_a_pack = torch.stack(g_a_list, dim=0)              # [Npair,4096,7]
        g_b_pack = torch.stack(g_b_list, dim=0)              # [Npair,4096,7]
        y_pack   = torch.tensor(y_list, dtype=torch.float32) # [Npair]
        return g_a_pack, g_b_pack, y_pack

# =========================
# Models (same as pretrain)
# =========================

def freeze_backbone_head_only(model, model_type):
    """只训练分类头（以及 MLP/Transformer 之前的 pooling 不含参数），其余冻结。"""
    for p in model.parameters():
        p.requires_grad = False
    # 解冻分类头
    for p in model.classifier.parameters():
        p.requires_grad = True
    # 可选：对 transformer 版也解冻 input/pos MLP 以适配分布差异
    if model_type == "transformer":
        for p in model.input_mlp.parameters():
            p.requires_grad = True
        for p in model.pos_mlp.parameters():
            p.requires_grad = True

# =========================
# Finetune training
# =========================

def h5_sort_key(name):
    base = os.path.splitext(os.path.basename(name))[0]
    toks = [t for t in base.replace("-", "_").split("_") if t.isdigit()]
    return (0, int(toks[-1]), base) if toks else (1, 0, base)

def load_manifest_case_map(data_dir):
    manifest = Path(data_dir) / "manifest.jsonl"
    if not manifest.exists():
        return {}
    case_map = {}
    with manifest.open("r") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "case":
                continue
            out_name = os.path.basename(str(rec.get("out_path", "")))
            case_id = rec.get("case_id")
            if out_name and case_id:
                case_map[out_name] = str(case_id)
    return case_map

def read_h5_case_id(path):
    with h5py.File(path, "r") as hf:
        for key in ("patient_id", "case_id"):
            if key in hf.attrs:
                val = hf.attrs[key]
                return val.decode() if isinstance(val, bytes) else str(val)
        if "meta" in hf and "json" in hf["meta"].attrs:
            meta = json.loads(hf["meta"].attrs["json"])
            if "case" in meta and isinstance(meta["case"], dict) and meta["case"].get("case_id"):
                return str(meta["case"]["case_id"])
            for key in ("patient_id", "case_id"):
                if meta.get(key):
                    return str(meta[key])
    raise RuntimeError(f"Could not find case id in {path}")

def build_grouped_split(data_dir, val_fraction=0.2, seed=42, split_json=None):
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    if not all_files:
        raise RuntimeError(f"No .h5 files found under {data_dir}")
    all_files.sort(key=h5_sort_key)

    if split_json:
        with open(split_json, "r") as f:
            split = json.load(f)
        validate_split(data_dir, split)
        return split["train_files"], split["val_files"], split

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    manifest_case_map = load_manifest_case_map(data_dir)
    files_by_case = defaultdict(list)
    for fname in all_files:
        case_id = manifest_case_map.get(fname)
        if case_id is None:
            case_id = read_h5_case_id(os.path.join(data_dir, fname))
        files_by_case[case_id].append(fname)

    cases = sorted(files_by_case)
    if len(cases) < 2:
        raise ValueError("At least two patients are required for a disjoint train/validation split.")

    rng = random.Random(seed)
    rng.shuffle(cases)
    n_val = max(1, int(round(len(cases) * float(val_fraction))))
    n_val = min(n_val, max(1, len(cases) - 1))
    val_cases = set(cases[:n_val])
    train_cases = set(cases[n_val:])

    train_files = []
    val_files = []
    for case_id in sorted(train_cases):
        train_files.extend(sorted(files_by_case[case_id], key=h5_sort_key))
    for case_id in sorted(val_cases):
        val_files.extend(sorted(files_by_case[case_id], key=h5_sort_key))

    split = {
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "n_cases": len(cases),
        "n_train_cases": len(train_cases),
        "n_val_cases": len(val_cases),
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "train_cases": sorted(train_cases),
        "val_cases": sorted(val_cases),
        "train_files": train_files,
        "val_files": val_files,
    }
    return train_files, val_files, split

def validate_split(data_dir, split):
    """Reject missing files, patient leakage, and renamed/mismatched H5 manifests.

    Trajectory files use sequential data_N.h5 names, so checking only those
    names can silently apply it to different patients after regeneration.
    """
    actual_cases = {}
    files_seen = set()
    for partition in ("train", "val"):
        files = split[partition + "_files"]
        declared_cases = set(split[partition + "_cases"])
        if not files or not declared_cases:
            raise ValueError("Both split partitions must be nonempty.")
        actual_cases[partition] = set()
        for name in files:
            if Path(name).name != name or name in files_seen:
                raise ValueError("Split filenames must be unique basenames: %s" % name)
            files_seen.add(name)
            path = Path(data_dir) / name
            if not path.is_file():
                raise FileNotFoundError("Split references missing H5: %s" % path)
            case_id = read_h5_case_id(path)
            if case_id not in declared_cases:
                raise ValueError("Split case mismatch for %s: %s" % (name, case_id))
            actual_cases[partition].add(case_id)
        if actual_cases[partition] != declared_cases:
            raise ValueError("Some declared %s patients have no H5 files." % partition)
        if split.get("n_" + partition + "_files", len(files)) != len(files):
            raise ValueError("Inconsistent split file count for %s." % partition)
        if split.get("n_" + partition + "_cases", len(declared_cases)) != len(declared_cases):
            raise ValueError("Inconsistent split case count for %s." % partition)
    if actual_cases["train"] & actual_cases["val"]:
        raise ValueError("Train and validation patients overlap.")

def resolve_pair_counts(K_per_type, K_hard=None, K_core_infil=None, K_infil_healthy=None, K_rank=None):
    base = int(K_per_type)
    counts = (
        base if K_hard is None else int(K_hard),
        base if K_core_infil is None else int(K_core_infil),
        base if K_infil_healthy is None else int(K_infil_healthy),
        base if K_rank is None else int(K_rank),
    )
    if min(counts) < 0:
        raise ValueError("Pair counts must be >= 0.")
    if sum(counts) <= 0:
        raise ValueError("At least one pair count must be > 0.")
    return counts

def finetune(
    data_dir,
    pretrained_path,
    model_type,
    m_kwargs,
    epochs=20,
    batch_size=5,          # Each H5 is one trajectory, not one patient.
    lr=1e-5,
    weight_decay=1e-3,
    scheduler_type="cosine",
    step_size=5,
    gamma=0.5,
    T_max=20,
    device="cpu",
    seed=42,
    K_per_type=8,
    K_hard=None,
    K_core_infil=None,
    K_infil_healthy=None,
    K_rank=None,
    num_workers=4,
    pin_memory=True,
    train_head_only=False,
    percentile_norm=(1,99),
    val_fraction=0.2,
    split_json=None,
    output_dir="runs/ranking",
    run_name=None,
    strict_pretrained=True,
    amp=False
):
    if epochs <= 0 or batch_size <= 0 or T_max <= 0:
        raise ValueError("epochs, batch_size and T_max must be positive.")
    if not 0 <= percentile_norm[0] < percentile_norm[1] <= 100:
        raise ValueError("Normalization percentiles must satisfy 0 <= low < high <= 100.")
    set_seed(seed)

    train_list, val_list, split = build_grouped_split(
        data_dir=data_dir,
        val_fraction=val_fraction,
        seed=seed,
        split_json=split_json,
    )
    print(
        "Grouped split: "
        f"{split['n_train_cases']} train cases/{split['n_train_files']} files, "
        f"{split['n_val_cases']} val cases/{split['n_val_files']} files"
    )
    pair_counts = resolve_pair_counts(
        K_per_type=K_per_type,
        K_hard=K_hard,
        K_core_infil=K_core_infil,
        K_infil_healthy=K_infil_healthy,
        K_rank=K_rank,
    )
    print(
        "Pair counts per H5: "
        f"hard(shrink,expand)={pair_counts[0]}, "
        f"core_infil(shrink,points_full)={pair_counts[1]}, "
        f"infil_healthy(points_full,expand)={pair_counts[2]}, "
        f"rank(points_full,points_full)={pair_counts[3]}"
    )

    train_ds = MultiPairPerPatientDataset_Finetune(
        data_dir, K=K_per_type, file_list=train_list,
        percentile_norm=percentile_norm,
        pair_counts=pair_counts,
    )
    val_ds = MultiPairPerPatientDataset_Finetune(
        data_dir, K=K_per_type, file_list=val_list,
        percentile_norm=percentile_norm,
        pair_counts=pair_counts,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    # 模型 & 预训练
    model = build_model(model_type, m_kwargs).to(device)
    if pretrained_path is not None:
        model = load_pretrained(model, pretrained_path, map_location=device, strict=strict_pretrained)
    else:
        print("Training from random initialization.")
    if train_head_only:
        freeze_backbone_head_only(model, model_type)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=weight_decay)
    use_amp = bool(amp) and str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=step_size)
    else:
        scheduler = None

    if run_name is None:
        run_name = f"{Path(data_dir).name}_seed{seed}"
    ckpt_dir = os.path.join(output_dir, f"{model_type}", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2, sort_keys=True)
    with open(os.path.join(ckpt_dir, "train_config.json"), "w") as f:
        json.dump(
            {
                "data_dir": str(data_dir),
                "initialization": "pretrained" if pretrained_path is not None else "random",
                "seed": seed,
                "T_max": T_max,
                "step_size": step_size,
                "gamma": gamma,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "split_json": str(split_json) if split_json else None,
                "pretrained": str(pretrained_path) if pretrained_path is not None else None,
                "model_type": model_type,
                "model_kwargs": m_kwargs,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "scheduler": scheduler_type,
                "K_per_type": K_per_type,
                "pair_counts": {
                    "hard_shrink_expand": pair_counts[0],
                    "core_infil_shrink_points_full": pair_counts[1],
                    "infil_healthy_points_full_expand": pair_counts[2],
                    "rank_points_full_points_full": pair_counts[3],
                },
                "percentile_norm": list(percentile_norm),
                "train_head_only": train_head_only,
                "strict_pretrained": strict_pretrained,
                "amp": use_amp,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    metrics_path = os.path.join(ckpt_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        f.write(
            "epoch,train_loss,train_acc,val_loss,val_acc,lr,"
            "n_train_pairs,n_val_pairs,epoch_seconds,is_best,best_val_loss\n"
        )

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        # ---- train ----
        model.train()
        running_train, n_train_pairs = 0.0, 0
        correct_train = 0
        for gA_pack, gB_pack, y_pack in tqdm.tqdm(train_loader, desc=f"FT Train {epoch}/{epochs}"):
            N, fourK, P, D = gA_pack.shape
            g_a = gA_pack.view(N * fourK, P, D).to(device, non_blocking=True)
            g_b = gB_pack.view(N * fourK, P, D).to(device, non_blocking=True)
            y   = y_pack.view(N * fourK).to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                p, logits = model(g_a, g_b)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train += loss.item() * (N * fourK)
            n_train_pairs += (N * fourK)
            preds = (p.detach() >= 0.5).float()
            correct_train += (preds == y).sum().item()

        train_loss = running_train / max(1, n_train_pairs)
        train_acc  = correct_train / max(1, n_train_pairs)

        # ---- val ----
        model.eval()
        running_val, n_val_pairs = 0.0, 0
        correct_val = 0
        with torch.no_grad():
            for gA_pack, gB_pack, y_pack in tqdm.tqdm(val_loader, desc=f"FT Val   {epoch}/{epochs}"):
                N, fourK, P, D = gA_pack.shape
                g_a = gA_pack.view(N * fourK, P, D).to(device, non_blocking=True)
                g_b = gB_pack.view(N * fourK, P, D).to(device, non_blocking=True)
                y   = y_pack.view(N * fourK).to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    p, logits = model(g_a, g_b)
                    batch_loss = criterion(logits, y).item()
                running_val += batch_loss * (N * fourK)
                n_val_pairs += (N * fourK)
                preds = (p >= 0.5).float()
                correct_val += (preds == y).sum().item()

        val_loss = running_val / max(1, n_val_pairs)
        val_acc  = correct_val / max(1, n_val_pairs)

        print(f"[Epoch {epoch}/{epochs}] "
              f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.3f}")
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        epoch_seconds = time.time() - epoch_t0
        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.8f},{train_acc:.8f},"
                f"{val_loss:.8f},{val_acc:.8f},{optimizer.param_groups[0]['lr']:.8e},"
                f"{n_train_pairs},{n_val_pairs},{epoch_seconds:.3f},{int(is_best)},{best_val:.8f}\n"
            )

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # save last & best
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{model_type}_ft_epoch{epoch:03d}.pth"))
        if is_best:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{model_type}_ft_best.pth"))

    print("Finetuning complete. Checkpoints saved to:", ckpt_dir)

# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Train a contour comparator on patient-grouped HDF5 data"
    )
    parser.add_argument("data_dir", help="Training trajectory HDF5 directory")
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--pretrained", help="Initialization checkpoint (.pth)")
    initialization.add_argument("--from-scratch", action="store_true",
                                help="Train from random initialization.")
    parser.add_argument("--model-type", choices=["mlp","transformer"], default="transformer")

    # MLP
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)

    # Transformer
    parser.add_argument("--d-model",         type=int, default=128)
    parser.add_argument("--nhead",           type=int, default=8)
    parser.add_argument("--num-layers",      type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout",         type=float, default=0.2)

    # Training
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch-size",   type=int,   default=5, help="每步 H5 数；一个 H5 是一个随机中心")
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--scheduler",    choices=["step","cosine","plateau"], default="cosine")
    parser.add_argument("--step-size",    type=int,   default=5)
    parser.add_argument("--gamma",        type=float, default=0.5)
    parser.add_argument("--T-max",        type=int,   default=None, help="Cosine period; defaults to --epochs")
    parser.add_argument("--device",       default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--K-per-type",   type=int,   default=8,
                        help="默认每类 pair 采样 K 对；可被下面四个 K-* 参数分别覆盖")
    parser.add_argument("--K-hard", type=int,
                        help="shrink vs expand pair 数量。默认等于 --K-per-type")
    parser.add_argument("--K-core-infil", type=int,
                        help="shrink vs points_full pair 数量。默认等于 --K-per-type")
    parser.add_argument("--K-infil-healthy", type=int,
                        help="points_full vs expand pair 数量。默认等于 --K-per-type")
    parser.add_argument("--K-rank", type=int,
                        help="points_full vs points_full pair 数量。默认等于 --K-per-type")
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Validation fraction by case, not by H5 file.")
    parser.add_argument("--split-json",
                        help="Optional split.json from a previous run to reuse the exact case split.")
    parser.add_argument("--output-dir", default="runs/ranking")
    parser.add_argument("--run-name",
                        help="Checkpoint subdirectory name. Default: <data_dir basename>_seed<seed>.")
    parser.add_argument("--non-strict-pretrained", action="store_true",
                        help="Load checkpoint with strict=False.")
    parser.add_argument("--amp", action="store_true",
                        help="Use CUDA automatic mixed precision.")

    # Options
    parser.add_argument("--train-head-only", action="store_true",
                        help="只训练分类头（transformer 还会解冻 input_mlp/pos_mlp）")
    parser.add_argument("--percentile-low",  type=int, default=1)
    parser.add_argument("--percentile-high", type=int, default=99)

    args = parser.parse_args()

    # Pack model kwargs
    if args.model_type == "mlp":
        m_kwargs = {'input_dim': 7, 'hidden_dim': args.mlp_hidden_dim}
    else:
        m_kwargs = {
            'input_dim':        7,
            'd_model':          args.d_model,
            'nhead':            args.nhead,
            'num_layers':       args.num_layers,
            'dim_feedforward':  args.dim_feedforward,
            'dropout':          args.dropout
        }

    finetune(
        data_dir=args.data_dir,
        pretrained_path=args.pretrained,
        model_type=args.model_type,
        m_kwargs=m_kwargs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        T_max=(args.epochs if args.T_max is None else args.T_max),
        device=args.device,
        seed=args.seed,
        K_per_type=args.K_per_type,
        K_hard=args.K_hard,
        K_core_infil=args.K_core_infil,
        K_infil_healthy=args.K_infil_healthy,
        K_rank=args.K_rank,
        num_workers=args.num_workers,
        pin_memory=(not args.no_pin_memory),
        train_head_only=args.train_head_only,
        percentile_norm=(args.percentile_low, args.percentile_high),
        val_fraction=args.val_fraction,
        split_json=args.split_json,
        output_dir=args.output_dir,
        run_name=args.run_name,
        strict_pretrained=(not args.non_strict_pretrained),
        amp=args.amp,
    )




if __name__ == "__main__":
    main()
