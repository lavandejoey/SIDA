"""
Dataset structure
Relative paths has 2 patterns, SUBSET in {fake_frames, fake_videos, real_frames, real_videos}:

```text
<TASK>/<METHOD>/<SUBSET>/
<TASK>/<METHOD>/<S_METHOD>/<SUBSET>/
```

After that, the files are organized as:

```text
<v_name>/frame_<%06d>.jpg
<v_name>.mp4
```
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from tqdm import tqdm
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# VIDEOS_DIR = Path("/home/zliu/FakeParts2/datasets/FakeParts_V2")
# FRAMES_ROOT = Path("/home/zliu/FakeParts2/datasets/FakeParts_V2_Frame")
VIDEOS_DIR = Path("/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_videos_only")
FRAMES_ROOT = Path("/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only")
VID_EXTS: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov")
IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


class Subset(str, Enum):
    FAKE_VIDEOS = "fake_videos"
    REAL_VIDEOS = "real_videos"
    FAKE_FRAMES = "fake_frames"
    REAL_FRAMES = "real_frames"


@dataclass
class IndexEntry:
    root: Path  # Root path
    rel_path: Path  # Path to file
    task: str  # task name, metadata
    method: str  # method name, metadata
    subset: Subset  # help defining the label
    label: int  # 0=real, 1=fake for gt value
    mode: str  # 'video' or 'frame' data file type

    def __getitem__(self, item: str) -> object:
        return getattr(self, item)


def data_parse(file_path: Path, root: Path) -> IndexEntry:
    """
    Parse a file path into IndexEntry components.
    Expected structures:
      <root>/<Task>/<Method>/{real_videos|fake_videos|real_frames|fake_frames}/file
      <root>/<Task>/<Method>/<Method2>/{real_videos|fake_videos|real_frames|fake_frames}/file
      <root>/{0_real|1_fake}/<Task>/<Method>/{real_videos|fake_videos|real_frames|fake_frames}/file
      <root>/{0_real|1_fake}/<Task>/<Method>/<Method2>/{real_videos|fake_videos|real_frames|fake_frames}/file
    - file could be <v_name>.mp4 or <v_name>/frame_%06d.jpg
    - 0_real/1_fake is an optional flag that can be used for quick filtering,
    """
    parts = file_path.relative_to(root).parts
    if len(parts) < 4:        raise ValueError(f"Unexpected path structure: {file_path} (relative: {parts})")

    # Remove the filename; only directories remain
    dir_parts = list(parts[:-1])
    subset_values = {s.value for s in Subset}

    # 1) Find the subset by scanning from right to left (to allow extra dirs after subset, e.g., video_name/)
    subset_idx = -1
    for i in range(len(dir_parts) - 1, -1, -1):
        if dir_parts[i] in subset_values:
            subset_idx = i
            break
    if subset_idx == -1:
        raise ValueError(
            f"Could not locate subset in path: {file_path}. "
            f"Expected one of {sorted(subset_values)}."
        )

    subset = dir_parts[subset_idx]

    # 2) Everything before the subset contains: [optional flag]/Task/Method[/Method2/...]
    head = dir_parts[:subset_idx]

    # Optional leading flag
    flag = None
    if head and head[0] in {"0_real", "1_fake"}:
        flag = head.pop(0)

    # We now require at least <Task>/<Method>
    if len(head) < 2:
        raise ValueError(
            f"Cannot parse <Task>/<Method> from: {dir_parts}. "
            f"Got head={head}, subset={subset}."
        )

    task = head[0]
    method_parts = head[1:]
    method = "/".join(method_parts)

    # 3) Label & mode
    label_from_subset = 0 if subset.startswith("real_") else 1
    if flag is not None:
        label_from_flag = 0 if flag == "0_real" else 1
        if label_from_flag != label_from_subset:
            print(f"[warn] flag {flag} != subset {subset}; using subset-derived label.")

    mode = "video" if "videos" in subset else ("frame" if "frames" in subset else "unknown")

    return IndexEntry(
        root=Path(root),
        rel_path=file_path.relative_to(root),
        task=str(task),
        method=str(method),
        subset=Subset(subset),
        label=label_from_subset,
        mode=mode,
    )


def index_list(root_path: Path, file_exts: Tuple[str, ...]) -> List[IndexEntry]:
    entries: List[IndexEntry] = []
    root_path = Path(root_path)
    for dirpath, _, filenames in tqdm(os.walk(root_path), desc=f"Indexing {root_path}"):
        dirpath = Path(dirpath)
        for fn in filenames:
            if fn.lower().endswith(file_exts):
                p = dirpath / fn
                try:
                    entries.append(data_parse(p, root_path))
                except Exception as e:
                    tqdm.write(f"[warn] Skipping file {p}: {e}")
                    continue
    return entries


def index_dataframe(root_path: Path,
                    file_exts: Tuple[str, ...] = IMG_EXTS, ) -> pd.DataFrame:
    """
    Build a DataFrame with one row per media file (video or frame).
    Columns: ['task','method','subset','label','mode','rel_path','abs_path','root']
    """
    root_path = Path(root_path)
    all_entries: List[IndexEntry] = []
    all_entries.extend(index_list(root_path, file_exts=file_exts))

    if not all_entries:
        return pd.DataFrame(columns=[
            "task", "method", "subset", "label", "mode", "rel_path", "abs_path", "root"
        ])

    df = pd.DataFrame([asdict(e) for e in all_entries])
    df["abs_path"] = df["rel_path"].map(lambda p: str(root_path / p))
    # Cast types
    df["task"] = df["task"].astype("string")
    df["method"] = df["method"].astype("string")
    df["subset"] = df["subset"].astype("string")
    df["mode"] = df["mode"].astype("string")
    df["label"] = df["label"].astype(np.int32)
    df["root"] = df["root"].astype("string")
    df["rel_path"] = df["rel_path"].astype("string")
    df["abs_path"] = df["abs_path"].astype("string")
    return df


# ==================== Fixed column schema for model outputs ====================

# Required columns every model output should provide.
REQUIRED_COLS: Tuple[str, ...] = (
    "sample_id",  # unique id per row (string or int) you use to join with index
    "task",  # from dataset
    "method",  # from dataset
    "subset",  # real_videos/fake_videos/... from dataset
    "label",  # 0=real, 1=fake  (ground truth)
    "model",  # model name or identifier
    "mode",  # 'video' or 'frame'
    "score",  # real-valued score, higher => more likely fake, -1 indicating unavailable
    "pred",  # hard prediction in {0,1} produced by the model
)


def standardise_predictions(
        df_like: Union[pd.DataFrame, Sequence[Mapping[str, object]]],
        column_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    if not isinstance(df_like, pd.DataFrame):
        df = pd.DataFrame(df_like)
    else:
        df = df_like.copy()

    if column_map:
        df = df.rename(columns=dict(column_map))

    # Check required
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Provided columns: {list(df.columns)}")

    # Minimal dtype hygiene (no value transforms)
    # Leave 'pred' and 'label' as ints; 'score' as float.
    df["label"] = df["label"].astype(int)
    df["pred"] = df["pred"].astype(int)
    df["score"] = df["score"].astype(float)

    # Ensure string-ish cols are strings
    for c in ("sample_id", "task", "method", "subset", "model", "mode"):
        df[c] = df[c].astype("string")

    return df


# ========================= Metrics (overall & grouped) =========================

def _ensure_binary_labels(df: pd.DataFrame) -> None:
    uniq = set(pd.unique(df["label"]))
    if not uniq.issubset({0, 1}):
        raise ValueError(f"`label` must be binary in {{0,1}}, got: {uniq}")


def overall_accuracy(df: pd.DataFrame) -> float:
    _ensure_binary_labels(df)
    return float(accuracy_score(df["label"].to_numpy(), df["pred"].to_numpy()))


def accuracy_by(df: pd.DataFrame, by: Union[str, Sequence[str]]) -> pd.DataFrame:
    _ensure_binary_labels(df)
    if isinstance(by, str):
        by = [by]
    g = df.groupby(list(by), dropna=False)
    acc = g.apply(lambda x: accuracy_score(x["label"].to_numpy(), x["pred"].to_numpy()))
    return acc.reset_index(name="accuracy")


def roc_auc_overall(df: pd.DataFrame) -> Optional[float]:
    _ensure_binary_labels(df)
    y_true = df["label"].to_numpy()
    y_score = df["score"].to_numpy()
    # Need both classes present for AUC
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def roc_auc_by(df: pd.DataFrame, by: Union[str, Sequence[str]]) -> pd.DataFrame:
    _ensure_binary_labels(df)
    if isinstance(by, str):
        by = [by]
    rows = []
    for keys, x in df.groupby(list(by), dropna=False):
        y_true = x["label"].to_numpy()
        y_score = x["score"].to_numpy()
        if len(np.unique(y_true)) < 2:
            auc = None
        else:
            auc = float(roc_auc_score(y_true, y_score))
        key_vals = keys if isinstance(keys, tuple) else (keys,)
        rows.append(tuple(key_vals) + (auc,))
    cols = list(by) + ["roc_auc"]
    return pd.DataFrame(rows, columns=cols)


def tpr_at_fpr(df: pd.DataFrame, target_fpr: float = 1e-2) -> Optional[float]:
    """
    Compute TPR at the smallest threshold where FPR <= target_fpr.
    Returns None if ROC cannot be computed.
    """
    _ensure_binary_labels(df)
    y_true = df["label"].to_numpy()
    y_score = df["score"].to_numpy()
    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, thresh = roc_curve(y_true, y_score, pos_label=1)
    # Find all positions with FPR <= target; pick the *max* TPR among them
    mask = fpr <= target_fpr
    if not np.any(mask):
        # No point on ROC meets the target FPR; return best-effort (closest over)
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
        return float(tpr[idx])
    return float(np.max(tpr[mask]))


def tpr_at_fpr_by(df: pd.DataFrame, by: Union[str, Sequence[str]], target_fpr: float = 1e-2) -> pd.DataFrame:
    _ensure_binary_labels(df)
    if isinstance(by, str):
        by = [by]
    rows = []
    for keys, x in df.groupby(list(by), dropna=False):
        val = tpr_at_fpr(x, target_fpr=target_fpr)
        key_vals = keys if isinstance(keys, tuple) else (keys,)
        rows.append(tuple(key_vals) + (val,))
    cols = list(by) + [f"tpr@fpr={target_fpr:g}"]
    return pd.DataFrame(rows, columns=cols)


def _balance_by_group(df: pd.DataFrame, by: Union[str, Sequence[str]],
                      random_state: Optional[int] = None) -> pd.DataFrame:
    """
    For each group in `by`, undersample the real class (label==0) to match the number of fakes (label==1).
    If a group has no fakes, the group is returned unchanged (metrics for fake-only will be None).
    """
    if isinstance(by, str):
        by = [by]
    pieces = []
    for keys, g in df.groupby(list(by), dropna=False):
        n_fake = int((g["label"] == 1).sum())
        if n_fake == 0:
            pieces.append(g)  # nothing to balance here
            continue
        fake_part = g[g["label"] == 1]
        real_part = g[g["label"] == 0]
        if len(real_part) <= n_fake:
            # already <= balanced
            pieces.append(g)
        else:
            real_sampled = real_part.sample(n=n_fake, random_state=random_state, replace=False)
            pieces.append(pd.concat([fake_part, real_sampled], axis=0, ignore_index=True))
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, axis=0, ignore_index=True)


def accuracy_by_fake(df: pd.DataFrame, by: Union[str, Sequence[str]]) -> pd.DataFrame:
    """
    Accuracy computed **only on the fake class** per group (equivalent to recall/TPR for label==1).
    """
    if isinstance(by, str):
        by = [by]
    _ensure_binary_labels(df)
    rows = []
    for keys, x in df.groupby(list(by), dropna=False):
        x_fake = x[x["label"] == 1]
        if len(x_fake) == 0:
            acc = None
        else:
            acc = float(accuracy_score(x_fake["label"].to_numpy(), x_fake["pred"].to_numpy()))
        key_vals = keys if isinstance(keys, tuple) else (keys,)
        rows.append(tuple(key_vals) + (acc,))
    cols = list(by) + ["accuracy_fake_only"]
    return pd.DataFrame(rows, columns=cols)


# ============================== Convenience report ==============================
def quick_report(df: pd.DataFrame, *, balance_for_groups: bool = True, group_fake_only: bool = True,
                 random_state: Optional[int] = None) -> Mapping[str, object]:
    """
    Produce a small dictionary of core numbers. Suitable for logging.

    DF input need cols: ['sample_id', 'task', 'method', 'subset', 'label', 'model', 'mode', 'score', 'pred']
    "sample_id",  # unique id per row (string or int) you use to join with index
    "task",  # from dataset
    "method",  # from dataset
    "subset",  # real_videos/fake_videos/... from dataset
    "label",  # 0=real, 1=fake  (ground truth)
    "model",  # model name or identifier
    "mode",  # 'video' or 'frame'
    "score",  # real-valued score, higher => more likely fake
    "pred",  # hard prediction in {0,1} produced by the model

    Overall Accuracy
    Accuracy by Task
    Accuracy by Methods
    ROC-AUC by Task
    ROC-AUC by Methods
    TPR@FPR=1e-2 by Task
    TPR@FPR=1e-2 by Methods
    """
    # Prepare grouped DataFrames
    df_task = df
    df_method = df
    if balance_for_groups:
        df_task = _balance_by_group(df, by="task", random_state=random_state)
        df_method = _balance_by_group(df, by="method", random_state=random_state)
    # Grouped accuracy: fake-only (recall) if requested; else legacy accuracy
    acc_task = accuracy_by_fake(df_task, "task") if group_fake_only else accuracy_by(df_task, "task")
    acc_method = accuracy_by_fake(df_method, "method") if group_fake_only else accuracy_by(df_method, "method")
    # Grouped ROC-AUC / TPR: computed on the (optionally) balanced sets including both classes
    roc_task = roc_auc_by(df_task, "task")
    roc_method = roc_auc_by(df_method, "method")
    tpr_task = tpr_at_fpr_by(df_task, "task", target_fpr=1e-2)
    tpr_method = tpr_at_fpr_by(df_method, "method", target_fpr=1e-2)
    return {
        "overall": {
            "accuracy": overall_accuracy(df),
            "roc_auc": roc_auc_overall(df),
            "tpr@1e-2": tpr_at_fpr(df, target_fpr=1e-2),
        },
        "by_task": {
            "accuracy": acc_task,
            "roc_auc": roc_task,
            "tpr@1e-2": tpr_task,
        },
        "by_method": {
            "accuracy": acc_method,
            "roc_auc": roc_method,
            "tpr@1e-2": tpr_method,
        },
    }


__all__ = [
    # indexing
    "Subset", "IndexEntry", "data_parse", "index_list", "index_dataframe",
    # schema helpers
    "REQUIRED_COLS", "standardise_predictions",
    # metrics
    "overall_accuracy", "accuracy_by", "roc_auc_overall", "roc_auc_by",
    "tpr_at_fpr", "tpr_at_fpr_by", "quick_report",
]
