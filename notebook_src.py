# %% [markdown]
# # Evaluating Uncertainty Label Resolution Methods for Multi-Label Chest X-Ray Classification
#
# CheXpert's report labeler marks hedged findings as *uncertain* (-1). This notebook compares
# six ways of turning that label into a training target under one fixed protocol: one
# backbone (ImageNet DenseNet-121, fully fine-tuned), one patient-level split, one training
# recipe and one metric set. Only the uncertainty policy changes between runs.
#
# **Research questions**
# 1. Do U-Ones, U-Ignore or U-MultiClass beat the U-Zeros baseline? (E1, E2)
# 2. Do label smoothing (U-Ones + LSR) and MC Dropout help under the same protocol? (E3)
# 3. Which policy gains most where uncertain labels are common? (E4)
# 4. Which policy is best calibrated, and does MC entropy flag uncertain studies? (E5, E6)
# 5. What does each policy cost to train and run? (E7)
#
# **How to run**
# - `MODE = "train"` (Kaggle GPU): trains one policy and pushes its outputs to the run branch
#   `run/<policy>_s<seed>`. Each person sets `POLICY` (and `SEED`) in the next cell and runs
#   their own policies; runs never share a branch, so they can train at the same time.
# - `MODE = "results"` (Kaggle CPU): fetches every run branch and builds all tables and figures.
#
# The next cell is the only one to edit. Everything after it is the frozen protocol.

# %% [markdown]
# ## 0.1 Run settings (the only cell to edit)
#
# `POLICY` is one of `u_zeros`, `u_ones`, `u_ignore`, `u_multiclass`, `u_ones_lsr`,
# `u_zeros_mc` (plus the extra experiment `u_zeros_posw`). `SEED` is 42 unless your team has
# agreed on an extra seed layer (43, 44). `N_SEEDS` is read only in results mode. Leave
# `REPO_REF` alone: it selects the tagged code version whose split file every run uses.

# %%
import os

MODE = os.environ.get("MODE", "train")
POLICY = os.environ.get("POLICY", "u_zeros")
SEED = int(os.environ.get("SEED", 42))
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_SEEDS = os.environ.get("N_SEEDS", "1")
REPO_REF = os.environ.get("REPO_REF", "v1")

# %% [markdown]
# ## 0.2 Frozen protocol
#
# `CONFIG` never changes between runs; results mode asserts it is identical in every run it
# combines. Head, loss and dropout settings are per policy and live in the policy registry
# (section 2.2). Environment details such as the dataset mount path are deliberately *not* in
# `CONFIG`, because they may differ between Kaggle accounts without changing the experiment.

# %%
import base64
import hashlib
import json
import math
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import v2 as T

CONFIG = {
    "code_version": "v1",
    "pathologies": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
    "image_size": 320,
    "val_fraction": 0.10,
    "split_seed": 42,
    "seeds": (42, 43, 44),
    "batch_size": 32,
    "max_epochs": 10,
    "optimizer": "AdamW",
    "lr": 1e-4,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-5,
    "plateau_factor": 0.1,
    "plateau_patience": 1,
    "early_stop_patience": 3,
    "amp": True,
    "rotation_deg": 10,
    "crop_scale": (0.9, 1.0),
    "crop_ratio": (0.95, 1.05),
    "jitter": 0.1,
    "mc_dropout_p": 0.3,
    "mc_passes": 20,
    "lsr_range": (0.55, 0.85),
    "bootstrap_resamples": 1000,
    "ece_bins": 10,
    "num_workers": 4,
    "channels_last": True,
    "cudnn_benchmark": False,
    "log_every": 50,
    "grad_stats_every": 200,
    "qual_val_images": 600,
    "qual_min_uncertain": 50,
}
PATHOLOGIES = CONFIG["pathologies"]

assert MODE in ("train", "results"), f"MODE must be 'train' or 'results', got {MODE!r}"
assert SEED in CONFIG["seeds"], f"SEED must be one of {CONFIG['seeds']}, got {SEED}"
assert N_SEEDS in ("1", "2", "3", "auto"), f"N_SEEDS must be 1, 2, 3 or auto, got {N_SEEDS!r}"

# A smoke pass (SMOKE=1) checks that a run works end to end in minutes: a data subset, two
# epochs and a short monitoring interval. Its results go to runs_smoke/ and are never used.
# The environment variables only matter for local CPU checks; on Kaggle the defaults apply.
RUN_LIMITS = {
    "train_images": int(os.environ.get("SMOKE_TRAIN", 2000)) if SMOKE else None,
    "val_images": int(os.environ.get("SMOKE_VAL", 500)) if SMOKE else None,
    "qual_images": int(os.environ.get("SMOKE_QUAL", 600)) if SMOKE else None,
    "max_epochs": int(os.environ.get("SMOKE_EPOCHS", 2)) if SMOKE else CONFIG["max_epochs"],
    "batch_size": int(os.environ.get("SMOKE_BATCH", 32)) if SMOKE else CONFIG["batch_size"],
    "grad_stats_every": 5 if SMOKE else CONFIG["grad_stats_every"],
}


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) random number generators."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party code may draw from the global RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

print(f"MODE={MODE} POLICY={POLICY} SEED={SEED} SMOKE={SMOKE} N_SEEDS={N_SEEDS} REF={REPO_REF}")
print(f"python {platform.python_version()} on {platform.platform()}")
for name, module in [
    ("torch", torch),
    ("torchvision", torchvision),
    ("numpy", np),
    ("pandas", pd),
    ("opencv", cv2),
    ("matplotlib", matplotlib),
]:
    print(f"{name:12s} {module.__version__}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name()
    print(f"cuda {torch.version.cuda}, {torch.cuda.device_count()} x {gpu}")
else:
    print("cuda not available (CPU only)")

# %% [markdown]
# ## 0.3 Environment
#
# Detects Kaggle, reads the GitHub secrets, finds the dataset and locates the repository.
#
# On Kaggle the repository is cloned at `REPO_REF` into the working directory, so every run
# (by any teammate, at any time) reads the same committed split and qualitative-id files.
# Git commands that talk to GitHub receive the token as a per-command HTTP header, never in
# the URL or the git config, and every line of git output is scrubbed of it before printing.

# %%
IS_KAGGLE = Path("/kaggle/input").is_dir()
KAGGLE_WORKING = Path("/kaggle/working")


def get_secret(name: str) -> str | None:
    """Return a Kaggle Secret (on Kaggle) or an environment variable (elsewhere), else None."""
    if IS_KAGGLE:
        from kaggle_secrets import UserSecretsClient

        try:
            return UserSecretsClient().get_secret(name)
        except Exception:
            return None
    return os.environ.get(name) or None


GITHUB_PAT = get_secret("GITHUB_PAT")
GITHUB_REPO = get_secret("GITHUB_REPO")  # "owner/name"
GITHUB_USER = get_secret("GITHUB_USER")
RUN_BY = GITHUB_USER or os.environ.get("USER", "unknown")
_AUTH_HEADER = (
    "Authorization: Basic " + base64.b64encode(f"x-access-token:{GITHUB_PAT}".encode()).decode()
    if GITHUB_PAT
    else None
)
_SECRETS = [s for s in (GITHUB_PAT, _AUTH_HEADER) if s]


def scrub(text: str) -> str:
    """Replace every secret (raw token and its auth-header form) in ``text`` with ***."""
    for secret in _SECRETS:
        text = text.replace(secret, "***")
    return text


def run_git(*args: str, cwd: Path | None = None, auth: bool = False) -> str:
    """Run a git command and return its scrubbed output; raise with scrubbed output on failure.

    With ``auth=True`` the token is passed as a one-off ``http.extraHeader``, so it never lands
    in ``.git/config``, the remote URL or the process output.
    """
    cmd = ["git"]
    if auth and _AUTH_HEADER:
        cmd += ["-c", f"http.extraHeader={_AUTH_HEADER}"]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # fail instead of waiting for a password
    proc = subprocess.run(cmd + list(args), cwd=cwd, env=env, capture_output=True, text=True)
    out = scrub(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed (exit {proc.returncode}):\n{out}")
    return out


def clone_repo(ref: str, dest: Path) -> Path:
    """Shallow-clone ``GITHUB_REPO`` at tag or branch ``ref`` into ``dest`` (reused if present)."""
    assert GITHUB_REPO, "Kaggle Secret GITHUB_REPO (owner/name) is not set"
    if not (dest / ".git").is_dir():
        url = f"https://github.com/{GITHUB_REPO}.git"
        run_git("clone", "--depth", "1", "--branch", ref, url, str(dest), auth=True)
    head = run_git("rev-parse", "--short", "HEAD", cwd=dest).strip()
    print(f"repo {GITHUB_REPO} at {ref} ({head}) in {dest}")
    return dest


def find_data_root() -> Path:
    """Locate the folder holding CheXpert-small's ``train.csv`` and ``valid.csv``.

    ``DATA_ROOT`` wins when set; on Kaggle the attached inputs are searched otherwise.
    """
    if os.environ.get("DATA_ROOT"):
        candidates = [Path(os.environ["DATA_ROOT"]).resolve()]
    elif IS_KAGGLE:
        inputs = Path("/kaggle/input")
        found = inputs.glob("**/train.csv")
        candidates = sorted(p.parent for p in found if len(p.relative_to(inputs).parts) <= 5)
    else:
        candidates = []
    roots = [c for c in candidates if (c / "train.csv").is_file() and (c / "valid.csv").is_file()]
    assert len(roots) == 1, f"expected one dataset folder with train.csv and valid.csv, got {roots}"
    return roots[0]


if os.environ.get("REPO_DIR"):
    REPO_DIR = Path(os.environ["REPO_DIR"]).resolve()
elif IS_KAGGLE:
    REPO_DIR = clone_repo(REPO_REF, KAGGLE_WORKING / "repo")
else:
    REPO_DIR = Path.cwd()
REPO_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = find_data_root()
TABLES_DIR = REPO_DIR / "results" / "tables"
FIGURES_DIR = REPO_DIR / "results" / "figures"
for d in (TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"kaggle={IS_KAGGLE} run_by={RUN_BY} token={'set' if GITHUB_PAT else 'not set'}")
print(f"data root: {DATA_ROOT}")
print(f"repo dir:  {REPO_DIR}")

# %% [markdown]
# # Part 1: Data preparation
#
# ## 1.1 Load and filter
#
# Reads the official `train.csv` (training data) and `valid.csv` (our test set: 200 studies
# labelled by three board-certified radiologists, no uncertain class). The `Path` prefix that
# differs between mirrors is stripped, so `path` (e.g. `train/patient00001/study1/
# view1_frontal.jpg`) is the stable row id everywhere. Rows are filtered to frontal views, then
# to AP or PA position, with the count printed after each step.
#
# Raw label values are kept as read (1, 0, -1, or blank for *not mentioned*). Blanks become 0
# only when a label matrix is built (`label_matrix`), so the dataset statistics can still count
# not-mentioned entries separately. The data fingerprint (SHA-256 over both CSVs) is recorded
# with every run so results mode can prove all runs used the same data.

# %%
EXPECTED_RAW_ROWS = {"train": 223_414, "valid": 234}  # official CheXpert-small CSVs
_PREFIX = re.compile(r"^.*?(?=(?:train|valid)/patient\d+/)")
_PATH_FORMAT = r"^(train|valid)/patient\d+/study\d+/view\d+_\w+\.jpg$"


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split_csv(name: str) -> pd.DataFrame:
    """Load ``<name>.csv``, strip the path prefix and keep frontal AP/PA rows with parsed ids."""
    raw = pd.read_csv(DATA_ROOT / f"{name}.csv")
    expected = EXPECTED_RAW_ROWS[name]
    if IS_KAGGLE:
        assert len(raw) == expected, f"{name}.csv: {len(raw)} rows, official mirror has {expected}"
    elif len(raw) != expected:
        print(f"note: {name}.csv has {len(raw)} rows (official {expected}); not the real dataset")
    df = raw.assign(path=raw["Path"].str.replace(_PREFIX, "", regex=True))
    assert df["path"].str.match(_PATH_FORMAT).all(), f"unexpected Path format in {name}.csv"

    frontal = df[df["Frontal/Lateral"] == "Frontal"]
    kept = frontal[frontal["AP/PA"].isin(["AP", "PA"])]
    print(f"{name}.csv: {len(df):>7,} rows -> {len(frontal):>7,} frontal -> {len(kept):>7,} AP/PA")

    parts = kept["path"].str.split("/")
    out = pd.DataFrame(
        {
            "path": kept["path"],
            "patient_id": parts.str[1],
            "study_id": parts.str[1] + "/" + parts.str[2],
            "view": kept["AP/PA"],
        }
    )
    out[PATHOLOGIES] = kept[PATHOLOGIES]
    return out.reset_index(drop=True)


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n, 5) int8 labels: 1 positive, 0 negative or not mentioned, -1 uncertain."""
    return df[PATHOLOGIES].fillna(0).to_numpy(dtype=np.int8)


TRAIN_ALL = load_split_csv("train")
TEST = load_split_csv("valid")

train_values = set(np.unique(TRAIN_ALL[PATHOLOGIES].to_numpy()[~TRAIN_ALL[PATHOLOGIES].isna()]))
assert train_values <= {-1.0, 0.0, 1.0}, f"unexpected train label values {train_values}"
assert TEST[PATHOLOGIES].notna().all().all(), "test labels must have no blanks"
assert set(np.unique(TEST[PATHOLOGIES])) <= {0.0, 1.0}, "test labels must be 0 or 1"
assert not set(TRAIN_ALL["patient_id"]) & set(TEST["patient_id"]), "train/test patients overlap"
assert TRAIN_ALL["path"].is_unique and TEST["path"].is_unique

# A CSV-only copy (no image folders) is enough to build the split and statistics locally.
csv_only = not IS_KAGGLE and not (DATA_ROOT / "train").is_dir()
if csv_only:
    print("note: no image folders under the data root; image file check skipped")
check_rng = np.random.default_rng(0)
for frame in [] if csv_only else (TRAIN_ALL, TEST):
    sample = check_rng.choice(frame["path"].to_numpy(), size=min(20, len(frame)), replace=False)
    missing = [p for p in sample if not (DATA_ROOT / p).is_file()]
    assert not missing, f"image files missing under {DATA_ROOT}: {missing[:3]}"

DATA_FINGERPRINT = hashlib.sha256(
    (sha256_file(DATA_ROOT / "train.csv") + sha256_file(DATA_ROOT / "valid.csv")).encode()
).hexdigest()
print(f"test: {TEST['study_id'].nunique()} studies, {TEST['patient_id'].nunique()} patients")
print(f"data fingerprint: {DATA_FINGERPRINT}")

# %% [markdown]
# ## 1.2 Patient-level split
#
# The official training rows are split by patient into *analysis train* (90%) and *internal
# validation* (10%), so no patient appears in both. Validation keeps the original -1 labels.
# Patients are visited in a seeded random order and added to validation until it holds 10% of
# the images, which keeps the image fraction within one patient of the target.
#
# The split is created **once**, by the builder, outside Kaggle, and committed to
# `splits/split_seed42.csv` before the code version is tagged. Every run then loads that file
# from the tagged clone, so all teammates train on the identical split even when their runs
# overlap in time. A Kaggle run never creates a split: if the file is missing it stops.

# %%
SPLIT_PATH = REPO_DIR / "splits" / f"split_seed{CONFIG['split_seed']}.csv"


def make_split_csv(df: pd.DataFrame) -> str:
    """Deterministic patient-level split of ``df`` as CSV text (patient_id, split)."""
    patients = np.array(sorted(df["patient_id"].unique()))
    images = df["patient_id"].value_counts()
    order = np.random.default_rng(CONFIG["split_seed"]).permutation(len(patients))
    target = round(CONFIG["val_fraction"] * len(df))
    val, n_val = set(), 0
    for i in order:
        if n_val >= target:
            break
        val.add(patients[i])
        n_val += images[patients[i]]
    split = ["val" if p in val else "train" for p in patients]
    return pd.DataFrame({"patient_id": patients, "split": split}).to_csv(
        index=False, lineterminator="\n"
    )


expected_split = make_split_csv(TRAIN_ALL)
if SPLIT_PATH.is_file():
    if SPLIT_PATH.read_text() != expected_split:
        print("warning: regenerating the split gives a different file; using the committed one")
elif IS_KAGGLE:
    raise FileNotFoundError(
        f"{SPLIT_PATH.relative_to(REPO_DIR)} is missing at {REPO_REF}. The builder creates it once "
        "outside Kaggle and commits it before tagging; never create it in a run."
    )
else:
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(expected_split)
    print(f"created {SPLIT_PATH} (commit it before tagging the code version)")

split_table = pd.read_csv(SPLIT_PATH)
assert split_table["patient_id"].is_unique, "a patient is listed twice in the split file"
assert set(split_table["split"]) == {"train", "val"}
assert set(split_table["patient_id"]) == set(TRAIN_ALL["patient_id"]), (
    "split file does not cover exactly the patients in train.csv (different data?)"
)
SPLIT_SHA256 = sha256_file(SPLIT_PATH)

split_of = TRAIN_ALL["patient_id"].map(split_table.set_index("patient_id")["split"])
TRAIN = TRAIN_ALL[split_of == "train"].reset_index(drop=True)
VAL = TRAIN_ALL[split_of == "val"].reset_index(drop=True)
assert not set(TRAIN["patient_id"]) & set(VAL["patient_id"]), "patient in both train and val"
val_share = len(VAL) / len(TRAIN_ALL)
assert abs(val_share - CONFIG["val_fraction"]) <= 0.005, f"validation share {val_share:.2%}"

for name, frame in [("analysis train", TRAIN), ("internal val", VAL), ("test", TEST)]:
    print(f"{name:15s} {len(frame):>7,} images {frame['patient_id'].nunique():>6,} patients")
print(f"validation share {val_share:.2%}; split sha256 {SPLIT_SHA256}")

# %% [markdown]
# ## 1.3 Dataset statistics
#
# Label counts per pathology for the full (frontal AP/PA) training data, the analysis train and
# internal validation splits, and the test set, plus images and patients per split. The
# uncertain share in the analysis split is the prevalence used by experiment E4.
#
# Tables are written as CSV and LaTeX (`booktabs`), figures as vector PDF plus 300 dpi PNG at
# IEEE single-column (3.5 in) or double-column (7.16 in) width.

# %%
COLUMN_WIDTH_IN, PAGE_WIDTH_IN = 3.5, 7.16


def save_table(df: pd.DataFrame, name: str, float_format: str = "%.1f") -> None:
    """Write ``df`` to ``results/tables/<name>.csv`` and ``.tex`` (booktabs)."""
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    latex = df.to_latex(index=False, escape=True, float_format=lambda x: float_format % x)
    (TABLES_DIR / f"{name}.tex").write_text(latex)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Write ``fig`` to ``results/figures/<name>.pdf`` and a 300 dpi ``.png``."""
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")


def show_figure(fig: plt.Figure) -> None:
    """Display ``fig`` inline (skipped when running headless as a script), then close it."""
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def label_counts(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Positive / negative / uncertain / not-mentioned counts per pathology for one split."""
    rows = []
    for p in PATHOLOGIES:
        values = df[p]
        n_unc = int((values == -1).sum())
        rows.append(
            {
                "split": split,
                "pathology": p,
                "positive": int((values == 1).sum()),
                "negative": int((values == 0).sum()),
                "uncertain": n_unc,
                "not_mentioned": int(values.isna().sum()),
                "uncertain_pct": 100 * n_unc / len(df),
            }
        )
    return pd.DataFrame(rows)


SPLITS = {"full train": TRAIN_ALL, "analysis train": TRAIN, "internal val": VAL, "test": TEST}
dataset_stats = pd.concat([label_counts(df, name) for name, df in SPLITS.items()])
dataset_splits = pd.DataFrame(
    [
        {
            "split": name,
            "images": len(df),
            "studies": df["study_id"].nunique(),
            "patients": df["patient_id"].nunique(),
        }
        for name, df in SPLITS.items()
    ]
)
counted = dataset_stats[["positive", "negative", "uncertain", "not_mentioned"]].sum(axis=1)
assert (counted.to_numpy() == np.repeat([len(df) for df in SPLITS.values()], 5)).all()
assert (dataset_stats.query("split == 'test'")[["uncertain", "not_mentioned"]] == 0).all().all()

save_table(dataset_stats, "dataset_stats")
save_table(dataset_splits, "dataset_splits")
print(dataset_splits.to_string(index=False))
print(dataset_stats.to_string(index=False, float_format="%.1f"))

analysis_stats = dataset_stats[dataset_stats["split"] == "analysis train"]
UNCERTAIN_PREVALENCE = analysis_stats.set_index("pathology")["uncertain_pct"]
fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.8))
bars = ax.barh(PATHOLOGIES[::-1], UNCERTAIN_PREVALENCE[PATHOLOGIES[::-1]], color="0.35")
ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=7)
ax.set_xlabel("Uncertain labels (% of images)", fontsize=8)
ax.set_xlim(0, UNCERTAIN_PREVALENCE.max() * 1.25)
ax.tick_params(labelsize=7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save_figure(fig, "uncertain_prevalence")
show_figure(fig)

# %% [markdown]
# ## 1.4 Fixed qualitative image lists
#
# The qualitative section compares all policies on the *same* images, so the image list is
# fixed before any training: every test image, plus 600 internal-validation images that
# include at least 50 uncertain-labelled images per pathology (topped up with random images).
# Like the split, `qual/qual_ids.csv` is created once outside Kaggle, committed, and only read
# by runs.

# %%
QUAL_PATH = REPO_DIR / "qual" / "qual_ids.csv"


def make_qual_csv(val: pd.DataFrame, test: pd.DataFrame) -> str:
    """Qualitative image list as CSV text (path, split, group), sorted by path."""
    rng = np.random.default_rng(CONFIG["split_seed"])
    n_total, n_min = CONFIG["qual_val_images"], CONFIG["qual_min_uncertain"]
    group = {}
    for p in PATHOLOGIES:
        uncertain = val["path"][val[p] == -1]
        have = sum(1 for path in uncertain if path in group)
        pool = np.array(sorted(set(uncertain) - set(group)))
        assert have + len(pool) >= n_min, f"internal val has under {n_min} uncertain {p} images"
        for path in rng.choice(pool, size=max(n_min - have, 0), replace=False):
            group[path] = f"uncertain {p}"
    rest = np.array(sorted(set(val["path"]) - set(group)))
    for path in rng.choice(rest, size=n_total - len(group), replace=False):
        group[path] = "random"
    rows = [{"path": p, "split": "val", "group": g} for p, g in group.items()]
    rows += [{"path": p, "split": "test", "group": "all test"} for p in test["path"]]
    table = pd.DataFrame(rows).sort_values("path", ignore_index=True)
    return table.to_csv(index=False, lineterminator="\n")


expected_qual = make_qual_csv(VAL, TEST)
if QUAL_PATH.is_file():
    if QUAL_PATH.read_text() != expected_qual:
        print("warning: regenerating the qualitative list differs; using the committed one")
elif IS_KAGGLE:
    raise FileNotFoundError(
        f"{QUAL_PATH.relative_to(REPO_DIR)} is missing at {REPO_REF}. The builder creates it once "
        "outside Kaggle and commits it before tagging; never create it in a run."
    )
else:
    QUAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUAL_PATH.write_text(expected_qual)
    print(f"created {QUAL_PATH} (commit it before tagging the code version)")

QUAL_IDS = pd.read_csv(QUAL_PATH)
qual_val = VAL[VAL["path"].isin(QUAL_IDS.query("split == 'val'")["path"])]
assert len(qual_val) == CONFIG["qual_val_images"], "qualitative val images not all in VAL"
assert set(QUAL_IDS.query("split == 'test'")["path"]) == set(TEST["path"])
qual_uncertain = (qual_val[PATHOLOGIES] == -1).sum()
assert (qual_uncertain >= CONFIG["qual_min_uncertain"]).all(), qual_uncertain.to_dict()
print(f"qualitative list: {len(qual_val)} val + {len(TEST)} test images")
print("uncertain per pathology in the val subset:", qual_uncertain.to_dict())

# %% [markdown]
# ## 1.5 Dataset and loaders
#
# Images are read as grayscale and resized directly to 320 x 320 (no letterboxing), so a
# sample is a uint8 tensor of shape (1, 320, 320) with its raw labels in {-1, 0, 1}; each
# policy decides later what -1 means. Training augmentation runs on the CPU workers: rotation
# up to 10 degrees, a random resized crop keeping 90-100% of the area, and brightness and
# contrast jitter of 0.1. There is no horizontal flip (it would move the heart to the right).
# Conversion to float, the copy to three channels and ImageNet normalisation happen on the
# GPU (`to_model_input`), which keeps the CPU workers and the host-to-device copy light.
#
# The training order comes from `EpochSampler`, a permutation that depends only on the seed
# and the epoch. Two policies run with the same seed therefore see the same images in the same
# order with the same augmentation, whoever runs them and whenever.

# %%
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TRAIN_AUGMENT = T.Compose(
    [
        T.RandomRotation(CONFIG["rotation_deg"]),
        T.RandomResizedCrop(
            CONFIG["image_size"], scale=CONFIG["crop_scale"], ratio=CONFIG["crop_ratio"]
        ),
        T.ColorJitter(brightness=CONFIG["jitter"], contrast=CONFIG["jitter"]),
    ]
)


def read_image(path: Path, size: int) -> np.ndarray:
    """Grayscale uint8 image resized directly to ``size`` x ``size``."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


class CheXpertDataset(Dataset):
    """Frontal images as uint8 (1, S, S) tensors with raw labels in {-1, 0, 1} and row index."""

    def __init__(self, df: pd.DataFrame, augment: bool) -> None:
        self.paths = [DATA_ROOT / p for p in df["path"]]
        self.labels = torch.from_numpy(label_matrix(df))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        x = torch.from_numpy(read_image(self.paths[i], CONFIG["image_size"]))[None]
        if self.augment:
            x = TRAIN_AUGMENT(x)
        return x, self.labels[i], i


class EpochSampler(Sampler[int]):
    """Permutation of ``range(n)`` determined by (seed, epoch) alone; call ``set_epoch``."""

    def __init__(self, n: int, seed: int) -> None:
        self.n, self.seed, self.epoch = n, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        return iter(np.random.default_rng([self.seed, self.epoch]).permutation(self.n).tolist())

    def __len__(self) -> int:
        return self.n


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and ``random`` in a loader worker from its torch seed; one OpenCV thread."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002 - third-party code may draw from the global RNG
    random.seed(worker_seed)
    cv2.setNumThreads(0)


def make_loader(
    df: pd.DataFrame, train: bool, seed: int, num_workers: int | None = None
) -> DataLoader:
    """Loader over ``df``: seeded order and augmentation for training, fixed order otherwise."""
    workers = CONFIG["num_workers"] if num_workers is None else num_workers
    dataset = CheXpertDataset(df, augment=train)
    return DataLoader(
        dataset,
        batch_size=RUN_LIMITS["batch_size"],
        sampler=EpochSampler(len(dataset), seed) if train else None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        # Notebook-defined classes cannot be pickled to spawned or forkserver workers. Fork is the
        # Linux default up to Python 3.13 but not from 3.14, so ask for it explicitly.
        multiprocessing_context="fork" if workers > 0 and sys.platform == "linux" else None,
    )


def to_model_input(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 (B, 1, S, S) batch to a normalised float (B, 3, S, S) tensor on ``device``."""
    x = x.to(device, non_blocking=True).float().div_(255).expand(-1, 3, -1, -1)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    if CONFIG["channels_last"]:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


# Sanity checks on a small slice of the training split.
probe = TRAIN.iloc[:64]
x0, y0, i0 = CheXpertDataset(probe, augment=True)[0]
size = CONFIG["image_size"]
assert x0.shape == (1, size, size) and x0.dtype == torch.uint8, (x0.shape, x0.dtype)
assert y0.dtype == torch.int8 and set(y0.tolist()) <= {-1, 0, 1} and i0 == 0

order = list(EpochSampler(1000, SEED))
assert order == list(EpochSampler(1000, SEED)), "same seed and epoch must give the same order"
assert sorted(order) == list(range(1000))
later = EpochSampler(1000, SEED)
later.set_epoch(1)
assert list(later) != order, "the order must change between epochs"
other_seed = next(s for s in CONFIG["seeds"] if s != SEED)
assert list(EpochSampler(1000, other_seed)) != order, "the order must change with the seed"


def first_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """First augmented batch (images, row ids) of a fresh two-worker loader over ``probe``."""
    x, _, idx = next(iter(make_loader(probe, train=True, seed=seed, num_workers=2)))
    return x, idx


xa, ia = first_batch(SEED)
xb, ib = first_batch(SEED)
assert torch.equal(ia, ib) and torch.equal(xa, xb), "same seed must give identical batches"
assert not torch.equal(first_batch(other_seed)[0], xa), "another seed must change the batch"

batch = to_model_input(xa, torch.device("cpu"))
assert batch.shape == (len(xa), 3, size, size) and batch.dtype == torch.float32
print(f"loader ok: batch {tuple(batch.shape)}, order and augmentation reproducible per seed")

# %% [markdown]
# # Part 2: Model, policies and metrics
#
# ## 2.1 Model
#
# DenseNet-121 with ImageNet weights, every layer trainable. The model is split into
# `extract_features` (the final ReLU feature map, globally average-pooled to 1024 values) and
# `classify` (optional dropout, then one linear layer with 5 outputs, or 15 for U-MultiClass).
# The split lets MC Dropout compute the backbone once and repeat only the cheap head.
#
# `cam` returns class activation maps: the head weights for the chosen output rows applied to
# the final 10 x 10 feature map (plus the bias), so the spatial mean of a map equals the logit.
# The head is initialised from the run seed alone, so it does not depend on how much random
# state earlier cells used.

# %%
DENSENET_FEATURES = 1024


class DenseNetClassifier(nn.Module):
    """ImageNet DenseNet-121 backbone, optional dropout and a linear head."""

    def __init__(
        self, n_outputs: int, dropout_p: float, seed: int, pretrained: bool = True
    ) -> None:
        super().__init__()
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = torchvision.models.densenet121(weights=weights).features
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = nn.Linear(DENSENET_FEATURES, n_outputs)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Final ReLU feature map, (B, 1024, h, w); 10 x 10 for a 320 x 320 input."""
        return F.relu(self.features(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled backbone features, (B, 1024)."""
        return self.feature_map(x).mean(dim=(2, 3))

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Logits from pooled features: dropout (if any), then the linear head."""
        return self.head(self.dropout(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.extract_features(x))

    @torch.no_grad()
    def cam(self, x: torch.Tensor, rows: list[int]) -> torch.Tensor:
        """Class activation maps for head outputs ``rows``, (B, len(rows), h, w)."""
        fmap = self.feature_map(x)
        weight, bias = self.head.weight[rows], self.head.bias[rows]
        return torch.einsum("kc,bchw->bkhw", weight, fmap) + bias.view(1, -1, 1, 1)


def count_trainable_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Sanity checks without downloading weights (random backbone, one 320 x 320 batch of two).
probe_x = torch.randn(2, 3, CONFIG["image_size"], CONFIG["image_size"])
model = DenseNetClassifier(5, dropout_p=0.3, seed=SEED, pretrained=False).eval()
with torch.no_grad():
    feats = model.extract_features(probe_x)
    logits = model(probe_x)
    assert feats.shape == (2, DENSENET_FEATURES) and logits.shape == (2, 5)
    assert torch.allclose(logits, model.classify(feats)), "forward != classify(extract_features)"
    maps = model.cam(probe_x, rows=[0, 3])
    assert maps.shape == (2, 2, 10, 10), maps.shape
    assert torch.allclose(maps.mean(dim=(2, 3)), logits[:, [0, 3]], atol=1e-4), "CAM mean != logit"
    model.train()
    assert not torch.equal(model.classify(feats), model.classify(feats)), "dropout inactive"
    model.eval()
    assert torch.equal(model.classify(feats), model.classify(feats)), "dropout active in eval"

same = DenseNetClassifier(5, dropout_p=0.0, seed=SEED, pretrained=False)
assert torch.equal(same.head.weight, model.head.weight), "head init must depend only on the seed"
other = DenseNetClassifier(5, dropout_p=0.0, seed=other_seed, pretrained=False)
assert not torch.equal(other.head.weight, model.head.weight), "head init must change with seed"

backbone_params = 6_953_856  # torchvision DenseNet-121 without its 1000-class classifier
for n_out in (5, 15):
    n = count_trainable_params(DenseNetClassifier(n_out, 0.0, SEED, pretrained=False))
    assert n == backbone_params + DENSENET_FEATURES * n_out + n_out, (n_out, n)
print(f"model ok: {count_trainable_params(model):,} trainable parameters with a 5-output head")
del model, same, other

# %% [markdown]
# ## 2.2 Policy registry
#
# Each policy is one entry: the size of the head, the training loss, how logits become a
# probability per pathology, and how the raw validation labels are resolved for model
# selection. Everything else (backbone, data, optimiser, schedule) is shared.
#
# | Policy | Uncertain label u in training | Loss | Validation labels |
# |---|---|---|---|
# | `u_zeros` | 0 | BCE | u -> 0 |
# | `u_ones` | 1 | BCE | u -> 1 |
# | `u_ignore` | left out | masked BCE | u excluded |
# | `u_multiclass` | third class | 3-way cross-entropy per pathology | u excluded |
# | `u_ones_lsr` | fresh Uniform(0.55, 0.85) draw each step | BCE with soft targets | u -> 1 |
# | `u_zeros_mc` | 0, dropout 0.3 before the head | BCE | u -> 0 |
# | `u_zeros_posw` (extra) | 0, positive-class weights | weighted BCE | u -> 0 |
#
# Losses are computed in float32 even under mixed precision. The masked loss averages over the
# entries that are kept, so a batch whose labels are all uncertain contributes 0, not NaN.
# U-MultiClass encodes negative, positive and uncertain as classes 0, 1 and 2; its score is
# softmax P(positive). `certain_labels` (u excluded) is the policy-independent view used for
# the certain-only validation AUROC that every run logs.

# %%
N_PATH = len(PATHOLOGIES)


def binary_targets(y: torch.Tensor, uncertain_as: float) -> torch.Tensor:
    """Float targets from raw labels: 1 positive, 0 negative, ``uncertain_as`` for -1."""
    return torch.where(y == -1, uncertain_as, (y == 1).float())


def lsr_targets(y: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """U-Ones + LSR targets: a fresh Uniform(lo, hi) draw for every uncertain entry."""
    lo, hi = CONFIG["lsr_range"]
    draws = torch.empty(y.shape).uniform_(lo, hi, generator=generator).to(y.device)
    return torch.where(y == -1, draws, (y == 1).float())


def multiclass_classes(y: torch.Tensor) -> torch.Tensor:
    """U-MultiClass class index per entry: negative 0, positive 1, uncertain 2."""
    return torch.where(y == -1, 2, y).long()


def bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor | None = None):
    """Mean binary cross-entropy with logits, in float32."""
    return F.binary_cross_entropy_with_logits(logits.float(), targets, pos_weight=pos_weight)


def masked_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BCE over certain entries only: summed loss / max(number kept, 1)."""
    keep = (y != -1).float()
    per_entry = F.binary_cross_entropy_with_logits(
        logits.float(), (y == 1).float(), reduction="none"
    )
    return (per_entry * keep).sum() / keep.sum().clamp(min=1)


def multiclass_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """3-way cross-entropy per pathology on (B, 15) logits, averaged over all entries."""
    return F.cross_entropy(logits.float().reshape(-1, 3), multiclass_classes(y).reshape(-1))


def multiclass_probs(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the 3 classes of each pathology, (B, 5, 3)."""
    return torch.softmax(logits.float().reshape(-1, N_PATH, 3), dim=-1)


def multiclass_renormalised(logits: torch.Tensor) -> torch.Tensor:
    """Alternative U-MultiClass score P(pos) / (P(pos) + P(neg)), (B, 5)."""
    probs = multiclass_probs(logits)
    return probs[..., 1] / (probs[..., 0] + probs[..., 1])


def resolve(uncertain_as: float):
    """Validation-label rule mapping raw -1 to ``uncertain_as`` (NaN means excluded)."""
    return lambda y: np.where(y == -1, uncertain_as, y).astype(np.float64)


certain_labels = resolve(np.nan)

# Positive-class weights for the extra experiment: n_neg / n_pos on the training split, u -> 0.
train_positive = (label_matrix(TRAIN) == 1).sum(axis=0)
POS_WEIGHT = torch.tensor((len(TRAIN) - train_positive) / train_positive, dtype=torch.float32)


@dataclass(frozen=True)
class Policy:
    """One uncertainty policy: head size, loss, score and validation-label rule."""

    name: str
    n_outputs: int
    loss: Callable[[torch.Tensor, torch.Tensor, torch.Generator], torch.Tensor]
    score: Callable[[torch.Tensor], torch.Tensor]
    val_labels: Callable[[np.ndarray], np.ndarray]
    dropout_p: float = 0.0
    mc: bool = False
    extra: bool = False

    @property
    def cam_rows(self) -> list[int]:
        """Head rows whose CAMs are shown: the positive class of each pathology."""
        return (
            [3 * j + 1 for j in range(N_PATH)] if self.n_outputs == 3 * N_PATH else [*range(N_PATH)]
        )


def sigmoid_score(logits: torch.Tensor) -> torch.Tensor:
    """Per-pathology probability from (B, 5) logits."""
    return torch.sigmoid(logits.float())


def zeros_loss(logits: torch.Tensor, y: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """U-Zeros loss (also used by the MC Dropout variant): BCE with u -> 0."""
    return bce(logits, binary_targets(y, 0.0))


POLICIES = {
    p.name: p
    for p in [
        Policy(
            name="u_zeros",
            n_outputs=N_PATH,
            loss=zeros_loss,
            score=sigmoid_score,
            val_labels=resolve(0),
        ),
        Policy(
            name="u_ones",
            n_outputs=N_PATH,
            loss=lambda z, y, g: bce(z, binary_targets(y, 1.0)),
            score=sigmoid_score,
            val_labels=resolve(1),
        ),
        Policy(
            name="u_ignore",
            n_outputs=N_PATH,
            loss=lambda z, y, g: masked_bce(z, y),
            score=sigmoid_score,
            val_labels=certain_labels,
        ),
        Policy(
            name="u_multiclass",
            n_outputs=3 * N_PATH,
            loss=lambda z, y, g: multiclass_ce(z, y),
            score=lambda z: multiclass_probs(z)[..., 1],
            val_labels=certain_labels,
        ),
        Policy(
            name="u_ones_lsr",
            n_outputs=N_PATH,
            loss=lambda z, y, g: bce(z, lsr_targets(y, g)),
            score=sigmoid_score,
            val_labels=resolve(1),
        ),
        Policy(
            name="u_zeros_mc",
            n_outputs=N_PATH,
            loss=zeros_loss,
            score=sigmoid_score,
            val_labels=resolve(0),
            dropout_p=CONFIG["mc_dropout_p"],
            mc=True,
        ),
        Policy(
            name="u_zeros_posw",
            n_outputs=N_PATH,
            loss=lambda z, y, g: bce(z, binary_targets(y, 0.0), POS_WEIGHT.to(z.device)),
            score=sigmoid_score,
            val_labels=resolve(0),
            extra=True,
        ),
    ]
}
MAIN_POLICIES = [name for name, p in POLICIES.items() if not p.extra]
assert POLICY in POLICIES, f"POLICY must be one of {list(POLICIES)}, got {POLICY!r}"

print(f"policy {POLICY!r} selected; positive-class weights {np.round(POS_WEIGHT.numpy(), 2)}")

# Hand-computed checks on a 2 x 5 example; the expected values use NumPy formulas, not torch.
y_ex = torch.tensor([[1, 0, -1, -1, 1], [0, -1, 1, 0, -1]], dtype=torch.int8)
z_ex = torch.tensor([[2.0, -1.0, 0.5, -0.5, 0.0], [1.0, 0.3, -2.0, 0.7, -1.5]])
z_np, y_np = z_ex.numpy().astype(np.float64), y_ex.numpy()


def np_bce(z: np.ndarray, t: np.ndarray, w: np.ndarray | float = 1.0) -> np.ndarray:
    """Per-entry BCE with logits, positive term weighted by ``w`` (NumPy reference)."""
    log_sig, log_one_minus = -np.logaddexp(0, -z), -np.logaddexp(0, z)
    return -(w * t * log_sig + (1 - t) * log_one_minus)


def check(name: str, expected: float, generator: torch.Generator | None = None, z=z_ex) -> None:
    """Assert that policy ``name``'s loss on the example equals ``expected``."""
    got = POLICIES[name].loss(z, y_ex, generator).item()
    assert abs(got - expected) < 1e-5, f"{name}: loss {got:.6f}, expected {expected:.6f}"


positive, negative, uncertain = y_np == 1, y_np == 0, y_np == -1
check("u_zeros", np_bce(z_np, positive).mean())
check("u_zeros_mc", np_bce(z_np, positive).mean())
check("u_ones", np_bce(z_np, ~negative).mean())
check("u_ignore", np_bce(z_np, positive)[~uncertain].mean())
pw = POS_WEIGHT.numpy().astype(np.float64)
check("u_zeros_posw", np_bce(z_np, positive, pw).mean())

lsr_draw = lsr_targets(y_ex, torch.Generator().manual_seed(0)).numpy()
lo, hi = CONFIG["lsr_range"]
assert ((lsr_draw[uncertain] >= lo) & (lsr_draw[uncertain] <= hi)).all(), "LSR target out of range"
assert (lsr_draw[~uncertain] == positive[~uncertain]).all(), "LSR changed a certain label"
check("u_ones_lsr", np_bce(z_np, lsr_draw).mean(), torch.Generator().manual_seed(0))
second_draw = lsr_targets(y_ex, torch.Generator().manual_seed(1)).numpy()
assert not np.allclose(lsr_draw[uncertain], second_draw[uncertain]), "LSR draws must vary"

expected_classes = [[1, 0, 2, 2, 1], [0, 2, 1, 0, 2]]
assert multiclass_classes(y_ex).tolist() == expected_classes, "U-MultiClass must map -1 to class 2"
z15 = torch.linspace(-1.5, 1.5, 30).reshape(2, 15)
z15_np = z15.numpy().astype(np.float64).reshape(2, N_PATH, 3)
log_softmax = z15_np - np.log(np.exp(z15_np).sum(axis=-1, keepdims=True))
classes = np.where(uncertain, 2, y_np)
expected_ce = -np.take_along_axis(log_softmax, classes[..., None], axis=-1).mean()
check("u_multiclass", expected_ce, z=z15)
probs = np.exp(log_softmax)
assert np.allclose(POLICIES["u_multiclass"].score(z15).numpy(), probs[..., 1], atol=1e-6)
renorm = probs[..., 1] / (probs[..., 0] + probs[..., 1])
assert np.allclose(multiclass_renormalised(z15).numpy(), renorm, atol=1e-6)

all_uncertain = torch.full((2, N_PATH), -1, dtype=torch.int8)
z_grad = z_ex.clone().requires_grad_(True)
loss = POLICIES["u_ignore"].loss(z_grad, all_uncertain, None)
loss.backward()
assert loss.item() == 0.0 and torch.isfinite(z_grad.grad).all(), "all-uncertain batch must give 0"

for name, policy in POLICIES.items():
    logits = (z15 if policy.n_outputs == 3 * N_PATH else z_ex).clone().requires_grad_(True)
    policy.loss(logits, y_ex, torch.Generator().manual_seed(0)).backward()
    assert torch.isfinite(logits.grad).all(), f"{name}: non-finite gradient"
    scores = policy.score(logits.detach())
    assert scores.shape == (2, N_PATH) and ((scores >= 0) & (scores <= 1)).all(), name
    assert len(policy.cam_rows) == N_PATH and max(policy.cam_rows) < policy.n_outputs, name

assert np.array_equal(POLICIES["u_zeros"].val_labels(y_np), np.where(uncertain, 0, y_np))
assert np.array_equal(POLICIES["u_ones_lsr"].val_labels(y_np), np.where(uncertain, 1, y_np))
assert np.isnan(POLICIES["u_multiclass"].val_labels(y_np)[uncertain]).all()
assert np.isfinite(POS_WEIGHT.numpy()).all() and (POS_WEIGHT > 0).all(), "bad positive weights"
assert len(MAIN_POLICIES) == 6
print(f"policies ok: {', '.join(POLICIES)}")

# %% [markdown]
# ## 2.3 Metrics
#
# All metrics are written in NumPy so the evaluation is fully visible; scikit-learn appears
# only in the checks at the end of this cell, as an independent reference.
#
# - **AUROC** (main metric) from average ranks, so tied scores count one half.
# - **AUPRC** as average precision over distinct score thresholds.
# - **F1** at the threshold that maximises Youden's J (TPR - FPR) on internal validation.
# - **ECE** over 10 equal-width bins of the positive-class probability, and the **Brier** score.
# - **Predictive entropy** (in nats) of the MC Dropout mean probability.
#
# Confidence intervals resample *patients*, not images, because images of one patient are not
# independent. One set of 1,000 resamples (fixed seed) is drawn per evaluation set and reused
# for every policy and metric, so differences between policies are paired. The paired p-value
# for "policy A differs from U-Zeros" is
# p = min(1, 2 min((1 + #{d <= 0}) / (B + 1), (1 + #{d >= 0}) / (B + 1))) over the B bootstrap
# differences d, as in the proposal.

# %%
BOOTSTRAP_SEED = CONFIG["split_seed"]


def _sorted_counts(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct thresholds (descending) with cumulative true and false positives at each."""
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[last]
    return s_sorted[last], tps, last + 1 - tps


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve via average ranks; NaN if only one class is present."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    values, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    mid_ranks = np.cumsum(counts) - (counts - 1) / 2
    rank_sum = mid_ranks[inverse][y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision over distinct thresholds; NaN if there are no positives."""
    if y.sum() == 0:
        return np.nan
    _, tps, fps = _sorted_counts(y, s)
    precision, recall = tps / (tps + fps), tps / tps[-1]
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Score threshold maximising TPR - FPR (predict positive when score >= threshold)."""
    thresholds, tps, fps = _sorted_counts(y, s)
    j = tps / max(y.sum(), 1) - fps / max(len(y) - y.sum(), 1)
    return float(thresholds[np.argmax(j)])


def f1_at(y: np.ndarray, s: np.ndarray, threshold: float) -> float:
    """F1 score when predicting positive for score >= threshold (0 if undefined)."""
    pred = s >= threshold
    tp = np.sum(pred & (y == 1))
    denominator = 2 * tp + np.sum(pred & (y == 0)) + np.sum(~pred & (y == 1))
    return float(2 * tp / denominator) if denominator else 0.0


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = CONFIG["ece_bins"]) -> float:
    """Expected calibration error over equal-width bins (lo, hi] of the probability."""
    bins = np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1], right=True)
    total = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        total += in_bin.sum() * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(total / len(y))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared difference between probability and label."""
    return float(np.mean((p - y) ** 2))


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy (nats) of a Bernoulli with probability ``p``, elementwise."""
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def per_pathology(metric: Callable, labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """``metric`` per pathology column, skipping entries whose label is NaN (excluded)."""
    results = []
    for j in range(labels.shape[1]):
        keep = ~np.isnan(labels[:, j])
        results.append(metric(labels[keep, j], scores[keep, j]))
    return np.array(results)


def bootstrap_indices(patient_ids: np.ndarray, n_resamples: int, seed: int) -> list[np.ndarray]:
    """Row indices of ``n_resamples`` patient-level bootstrap resamples (whole patients)."""
    _, group, counts = np.unique(patient_ids, return_inverse=True, return_counts=True)
    order = np.argsort(group, kind="mergesort")
    starts = np.cumsum(counts) - counts
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(counts), len(counts))
        sizes = counts[chosen]
        offsets = np.repeat(starts[chosen] - (np.cumsum(sizes) - sizes), sizes)
        resamples.append(order[offsets + np.arange(sizes.sum())])
    return resamples


def bootstrap_ci(statistic: Callable[[np.ndarray], float], resamples: list[np.ndarray], n: int):
    """Point estimate on all ``n`` rows and 95% percentile interval over the resamples."""
    values = np.array([statistic(idx) for idx in resamples])
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return statistic(np.arange(n)), float(lo), float(hi)


def paired_difference(
    stat_a: Callable[[np.ndarray], float],
    stat_b: Callable[[np.ndarray], float],
    resamples: list[np.ndarray],
    n: int,
) -> dict:
    """Difference A - B with 95% CI and two-sided paired bootstrap p-value."""
    d = np.array([stat_a(idx) - stat_b(idx) for idx in resamples])
    d = d[np.isfinite(d)]
    b = len(d)
    p = min(1.0, 2 * min((1 + np.sum(d <= 0)) / (b + 1), (1 + np.sum(d >= 0)) / (b + 1)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    full = np.arange(n)
    return {"diff": stat_a(full) - stat_b(full), "lo": float(lo), "hi": float(hi), "p": p}


# Checks: hand-worked values, then agreement with scikit-learn on random data with ties.
assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5, "a tie must count one half"
assert np.isnan(auroc(np.array([1, 1]), np.array([0.2, 0.3])))

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)

check_rng = np.random.default_rng(0)
for _ in range(20):
    y_r = check_rng.integers(0, 2, 300)
    s_r = np.round(check_rng.random(300) * 0.6 + 0.4 * y_r, 2)  # rounding creates ties
    assert abs(auroc(y_r, s_r) - roc_auc_score(y_r, s_r)) < 1e-12
    assert abs(auprc(y_r, s_r) - average_precision_score(y_r, s_r)) < 1e-12
    assert abs(brier(y_r, s_r) - brier_score_loss(y_r, s_r)) < 1e-12
    fpr, tpr, thr = roc_curve(y_r, s_r)
    assert youden_threshold(y_r, s_r) == thr[np.argmax(tpr - fpr)]
    t = youden_threshold(y_r, s_r)
    assert abs(f1_at(y_r, s_r, t) - f1_score(y_r, s_r >= t)) < 1e-12

# Perfect calibration: in every bin the positive rate equals the (constant) probability.
p_cal = np.repeat(np.arange(0.05, 1, 0.1), 20)
y_cal = np.concatenate([np.r_[np.ones(k), np.zeros(20 - k)] for k in range(1, 20, 2)])
assert ece(y_cal, p_cal) < 1e-12, "ECE must be 0 for perfectly calibrated predictions"
assert abs(ece(np.r_[np.ones(5), np.zeros(5)], np.ones(10)) - 0.5) < 1e-12
# A probability on a bin edge belongs to the lower bin: 0.1 is in (0, 0.1], not (0.1, 0.2].
edge_ece = ece(np.array([0, 0, 1, 1]), np.array([0.1, 0.1, 0.15, 0.15]))
assert abs(edge_ece - 0.475) < 1e-12, edge_ece
assert abs(binary_entropy(np.array([0.5]))[0] - np.log(2)) < 1e-9
assert binary_entropy(np.array([0.0, 1.0])).max() < 1e-9

pids = np.array(["a", "b", "b", "c", "c", "c"])
for idx in bootstrap_indices(pids, 50, BOOTSTRAP_SEED):
    picked = pd.Series(pids[idx]).value_counts()
    size = pd.Series(pids).value_counts()
    assert all(picked[p] % size[p] == 0 for p in picked.index), "a patient was split"
first = bootstrap_indices(pids, 5, BOOTSTRAP_SEED)
again = bootstrap_indices(pids, 5, BOOTSTRAP_SEED)
assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True)), "not reproducible"

y_r = check_rng.integers(0, 2, 200)
s_r = check_rng.random(200) + 0.3 * y_r
groups = np.repeat(np.arange(100), 2)
boot = bootstrap_indices(groups, 200, BOOTSTRAP_SEED)
same = paired_difference(
    lambda i: auroc(y_r[i], s_r[i]), lambda i: auroc(y_r[i], s_r[i]), boot, 200
)
assert same == {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0}, same
always_better = paired_difference(lambda i: 1.0, lambda i: 0.0, boot, 200)
assert always_better["p"] == 2 / (len(boot) + 1), "p-value must be two-sided"
point, lo, hi = bootstrap_ci(lambda i: auroc(y_r[i], s_r[i]), boot, 200)
assert lo < point < hi
print(f"metrics ok: AUROC {point:.3f} (95% CI {lo:.3f}-{hi:.3f}) on a random check set")

# %% [markdown]
# # Part 3: Training run (`MODE == "train"`)
#
# ## 3.1 Training functions
#
# One recipe for every policy (proposal Table III): AdamW (lr 1e-4, betas 0.9/0.999, weight
# decay 1e-5), batch 32, up to 10 epochs, learning rate x0.1 when the validation mean AUROC has
# not improved for one epoch, early stopping after three epochs without improvement, FP16 mixed
# precision on the GPU. The policy only supplies the head size, the loss and the validation
# labels.
#
# Early stopping and the scheduler follow the validation mean AUROC computed with the policy's
# own validation labels (the proposal's rule); that checkpoint is `best.pt`. Because those labels
# differ between policies, the epoch with the best *certain-only* AUROC is also kept
# (`best_certain.pt`) for the selection-sensitivity table T9.
#
# Monitoring never slows a step: one log line every 50 steps, gradient norm and loss scale
# every 200 steps, and a background thread that samples `nvidia-smi` every 10 seconds. A
# non-finite loss stops the run with the reason recorded. Checkpoints stay in the Kaggle
# working directory and are never pushed.

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = CONFIG["amp"] and DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = CONFIG["cudnn_benchmark"]
assert CONFIG["optimizer"] == "AdamW"


class TrainingFailure(RuntimeError):
    """Training had to stop (for example a non-finite loss); the message is the reason."""


def make_logger(path: Path) -> Callable[[str], None]:
    """Logger that prints a line and appends it to ``path``."""

    def log(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")

    return log


def build_model(policy: Policy, seed: int, pretrained: bool = True) -> DenseNetClassifier:
    """The shared DenseNet-121 with this policy's head, on ``DEVICE``."""
    model = DenseNetClassifier(policy.n_outputs, policy.dropout_p, seed, pretrained).to(DEVICE)
    if CONFIG["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    return model


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """AdamW with the frozen protocol settings."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        betas=CONFIG["betas"],
        weight_decay=CONFIG["weight_decay"],
    )


class GpuTrace:
    """Background thread sampling ``nvidia-smi`` into a CSV; does nothing without a GPU."""

    FIELDS = "utilization.gpu,memory.used,clocks.sm,power.draw"

    def __init__(self, path: Path, interval_s: float = 10.0) -> None:
        self.path, self.interval_s = path, interval_s
        self.enabled = DEVICE.type == "cuda" and shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        query = ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"]
        while True:
            out = subprocess.run(query + ["-i", "0"], capture_output=True, text=True).stdout
            with open(self.path, "a") as f:
                f.write(f"{time.time() - self._start:.0f},{out.strip()}\n")
            if self._stop.wait(self.interval_s):
                return

    def __enter__(self) -> "GpuTrace":
        if self.enabled:
            self.path.write_text(f"time_s,{self.FIELDS}\n")
            self._start = time.time()
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.enabled:
            self._stop.set()
            self._thread.join()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
    epoch: int,
    grad_log: list[dict],
    log: Callable[[str], None],
) -> dict:
    """One pass over ``loader``; returns the mean loss, seconds and images per second."""
    model.train()
    loader.sampler.set_epoch(epoch)
    n_steps, seen = len(loader), 0
    loss_sum = torch.zeros((), device=DEVICE)  # summed on the device: no sync per step
    start = time.perf_counter()
    for step, (x, y, _) in enumerate(loader, start=1):
        x, y = to_model_input(x, DEVICE), y.to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(x)
        loss = policy.loss(logits, y, generator)  # float32 inside every policy loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if step % RUN_LIMITS["grad_stats_every"] == 0:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf)  # measure only
            grad_log.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "grad_norm": norm.item(),
                    "loss_scale": scaler.get_scale(),
                }
            )
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach() * len(x)
        seen += len(x)
        if step % CONFIG["log_every"] == 0 or step == n_steps:
            mean_loss = (loss_sum / seen).item()
            if not math.isfinite(mean_loss):
                raise TrainingFailure(f"non-finite loss in epoch {epoch} by step {step}")
            elapsed = time.perf_counter() - start
            eta_min = (n_steps - step) * elapsed / step / 60
            lr = optimizer.param_groups[0]["lr"]
            log(
                f"epoch {epoch} step {step}/{n_steps} loss {mean_loss:.4f} "
                f"{seen / elapsed:.0f} img/s eta {eta_min:.1f} min lr {lr:.1e} "
                f"scale {scaler.get_scale():.0f}"
            )
    seconds = time.perf_counter() - start
    return {"train_loss": mean_loss, "train_seconds": seconds, "images_per_sec": seen / seconds}


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """Logits for every row of an unshuffled loader, in row order, as float32."""
    model.eval()
    outputs = []
    for x, _, _ in loader:
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            outputs.append(model(to_model_input(x, DEVICE)).float().cpu())
    return torch.cat(outputs).numpy()


def scores_from_logits(policy: Policy, logits: np.ndarray) -> np.ndarray:
    """Per-pathology probabilities from saved logits."""
    return policy.score(torch.from_numpy(logits)).numpy()


def validation_aurocs(policy: Policy, y_raw: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Mean validation AUROC with policy-resolved labels, and with certain labels only."""
    resolved = per_pathology(auroc, policy.val_labels(y_raw), scores)
    certain = per_pathology(auroc, certain_labels(y_raw), scores)
    return float(np.nanmean(resolved)), float(np.nanmean(certain))


def fit(
    policy: Policy,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int,
    max_epochs: int,
    run_dir: Path,
    ckpt_dir: Path,
    log: Callable[[str], None],
    on_epoch_end: Callable[[], None] | None = None,
) -> tuple[nn.Module, dict]:
    """Train one policy; return the model restored to ``best.pt`` and a run summary."""
    seed_everything(seed)
    model = build_model(policy, seed)
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=CONFIG["plateau_factor"], patience=CONFIG["plateau_patience"]
    )
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
    generator = torch.Generator().manual_seed(seed)  # LSR target draws
    train_loader = make_loader(train_df, train=True, seed=seed)
    val_loader = make_loader(val_df, train=False, seed=seed)
    y_val = label_matrix(val_df)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, grad_log = [], []
    best = {"best.pt": (-math.inf, 0), "best_certain.pt": (-math.inf, 0)}
    stale, stop_reason = 0, "max_epochs"
    for epoch in range(1, max_epochs + 1):
        stats = train_one_epoch(
            model, train_loader, policy, optimizer, scaler, generator, epoch, grad_log, log
        )
        scores = scores_from_logits(policy, predict_logits(model, val_loader))
        val_auroc, val_certain = validation_aurocs(policy, y_val, scores)
        if not (math.isfinite(val_auroc) and math.isfinite(val_certain)):
            raise TrainingFailure(f"validation AUROC is undefined in epoch {epoch}")
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auroc)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0.0
        history.append(
            {
                "epoch": epoch,
                **stats,
                "val_auroc": val_auroc,
                "val_auroc_certain": val_certain,
                "lr": lr,
                "peak_mem_gb": peak_gb,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "metrics_epoch.csv", index=False)
        pd.DataFrame(grad_log, columns=["epoch", "step", "grad_norm", "loss_scale"]).to_csv(
            run_dir / "grad_stats.csv", index=False
        )
        improved = val_auroc > best["best.pt"][0]
        if improved:
            best["best.pt"] = (val_auroc, epoch)
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        if val_certain > best["best_certain.pt"][0]:
            best["best_certain.pt"] = (val_certain, epoch)
            if improved:  # same epoch: one file, copied, so the two are identical by construction
                shutil.copyfile(ckpt_dir / "best.pt", ckpt_dir / "best_certain.pt")
            else:
                torch.save(model.state_dict(), ckpt_dir / "best_certain.pt")
        stale = 0 if improved else stale + 1
        log(
            f"epoch {epoch} done: val AUROC {val_auroc:.4f} (certain-only {val_certain:.4f}), "
            f"best epoch {best['best.pt'][1]}, {stats['images_per_sec']:.0f} img/s"
        )
        if on_epoch_end is not None:
            on_epoch_end()
        if stale >= CONFIG["early_stop_patience"]:
            stop_reason = "early_stop"
            break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=DEVICE, weights_only=True))
    summary = {
        "best_epoch": best["best.pt"][1],
        "best_certain_epoch": best["best_certain.pt"][1],
        "epochs_run": len(history),
        "stop_reason": stop_reason,
        "train_seconds": sum(h["train_seconds"] for h in history),
        "images_per_sec": float(np.mean([h["images_per_sec"] for h in history])),
        "peak_mem_gb": max(h["peak_mem_gb"] for h in history),
    }
    return model, summary


# Check: every policy's loss falls on a repeated small batch (random backbone, 64 x 64 input).
small_x = F.interpolate(batch[:8], size=64, mode="bilinear", align_corners=False)
small_y = y_ex.repeat(4, 1)
for name, policy in POLICIES.items():
    torch.manual_seed(SEED)
    net = DenseNetClassifier(policy.n_outputs, policy.dropout_p, SEED, pretrained=False)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(SEED)
    losses = []
    for _ in range(8):
        step_loss = policy.loss(net(small_x), small_y, gen)
        opt.zero_grad()
        step_loss.backward()
        opt.step()
        losses.append(step_loss.item())
    assert losses[-1] < losses[0], f"{name}: loss did not fall on a repeated batch: {losses}"
print(f"training functions ok on {DEVICE}; loss falls on a repeated batch for every policy")

# %% [markdown]
# ## 3.2 Run it
#
# Trains the selected `POLICY` at `SEED` and writes everything a run produces into
# `runs/<policy>_s<seed>/` (`runs_smoke/` for a smoke pass): the per-epoch metrics, gradient
# statistics, the GPU trace, the training log and `run_info.json`. `run_info.json` records the
# policy, seed, code version, the full `CONFIG`, the split and data fingerprints, the hardware
# and the outcome, and its `status` moves from `running` to `trained` (and to `complete` once
# the predictions are written in section 3.3) or `failed` with the reason. A run that dies
# mid-way therefore never looks finished.
#
# Checkpoints go to `checkpoints/`, which is never pushed.

# %%
RUN_NAME = f"{POLICY}_s{SEED}"
RUN_DIR = REPO_DIR / ("runs_smoke" if SMOKE else "runs") / RUN_NAME
CKPT_DIR = REPO_DIR / "checkpoints" / RUN_NAME


def environment_info() -> dict:
    """Hardware and library versions recorded with every run."""
    on_gpu = DEVICE.type == "cuda"
    return {
        "gpu_name": torch.cuda.get_device_name() if on_gpu else "cpu",
        "gpu_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }


def write_run_info(status: str, **extra) -> dict:
    """Create or update ``run_info.json`` with a new status and fields; return the record."""
    path = RUN_DIR / "run_info.json"
    if path.exists():
        info = json.loads(path.read_text())
    else:
        info = {
            "policy": POLICY,
            "seed": SEED,
            "code_version": CONFIG["code_version"],
            "smoke": SMOKE,
            "config": CONFIG,
            "run_limits": RUN_LIMITS,
            "split_sha256": SPLIT_SHA256,
            "data_fingerprint": DATA_FINGERPRINT,
            "run_by": RUN_BY,
            **environment_info(),
        }
    info.update(status=status, **extra)
    path.write_text(json.dumps(info, indent=2))
    return info


if MODE == "train":
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"{RUN_DIR} is not empty; remove it to start this run again")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    policy = POLICIES[POLICY]
    train_df, val_df = TRAIN, VAL
    if SMOKE:
        n_train = min(RUN_LIMITS["train_images"], len(TRAIN))
        n_val = min(RUN_LIMITS["val_images"], len(VAL))
        train_df = TRAIN.sample(n=n_train, random_state=CONFIG["split_seed"], ignore_index=True)
        val_df = VAL.sample(n=n_val, random_state=CONFIG["split_seed"], ignore_index=True)

    log = make_logger(RUN_DIR / "train.log")
    log(f"run {RUN_NAME} by {RUN_BY} on {DEVICE} ({'smoke' if SMOKE else 'full'} run)")
    log(f"{len(train_df):,} training and {len(val_df):,} validation images")
    log(f"split {SPLIT_SHA256[:12]}, data {DATA_FINGERPRINT[:12]}")
    write_run_info(
        "running",
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_train_images=len(train_df),
        n_val_images=len(val_df),
    )
    try:
        with GpuTrace(RUN_DIR / "gpu_trace.csv"):
            model, summary = fit(
                policy,
                train_df,
                val_df,
                SEED,
                RUN_LIMITS["max_epochs"],
                RUN_DIR,
                CKPT_DIR,
                log,
            )
    except Exception as err:
        log(f"FAILED: {err!r}")
        write_run_info("failed", failure=repr(err))
        raise
    info = write_run_info("trained", trainable_params=count_trainable_params(model), **summary)
    log(f"trained: best epoch {info['best_epoch']} of {info['epochs_run']} ({info['stop_reason']})")

    # Checks on what the run wrote.
    metrics = pd.read_csv(RUN_DIR / "metrics_epoch.csv")
    assert len(metrics) == info["epochs_run"] and not metrics.isna().any().any(), "bad epoch log"
    steps = math.ceil(len(train_df) / RUN_LIMITS["batch_size"])
    expected_rows = info["epochs_run"] * (steps // RUN_LIMITS["grad_stats_every"])
    n_grad_rows = len(pd.read_csv(RUN_DIR / "grad_stats.csv"))
    assert n_grad_rows == expected_rows, (
        f"grad_stats has {n_grad_rows} rows, expected {expected_rows}"
    )
    if info["best_epoch"] == info["best_certain_epoch"]:
        assert sha256_file(CKPT_DIR / "best.pt") == sha256_file(CKPT_DIR / "best_certain.pt")
    if DEVICE.type == "cuda":
        assert len(pd.read_csv(RUN_DIR / "gpu_trace.csv")) >= 1, "GPU trace is empty"
    print(metrics.to_string(index=False, float_format="%.4f"))

# %% [markdown]
# ## 3.3 Predictions and run artifacts
#
# After training, the model (restored to `best.pt`) predicts the internal-validation images and
# the test images, and the run folder is completed:
#
# - `val_preds.npz`, `test_preds.npz`: image path (the row id), logits and scores. **They contain
#   no CheXpert labels**: the repository is public and the dataset licence forbids sharing the
#   data, so results mode joins the labels back from the CSVs by path.
# - `val_preds_certain.npz`, `test_preds_certain.npz`: the same for `best_certain.pt`, written
#   only when that checkpoint comes from a different epoch.
# - `mc_val.npz`, `mc_test.npz` (MC Dropout policy only): the mean of 20 stochastic passes,
#   its predictive entropy and the dropout-off score. `val_preds.npz` of that run is the
#   dropout-off prediction, so both versions are available to the results tables. The backbone
#   runs once; dropout and the linear head are applied 20 times to the cached 1024-d features,
#   which is exactly 20 full passes because the backbone is deterministic at inference.
# - `cam_test.npz`, `cam_val_subset.npz`: class activation maps (10 x 10) for every test image
#   and the fixed qualitative validation images.
# - Cost measurements in `run_info.json`: inference time per image, GFLOPs per image and, for
#   the MC policy, the time of the 20 head passes per image.
#
# Finally the status becomes `complete`; a run without it is ignored by results mode.

# %%
LABEL_KEYS = {"label", "labels", "y", "y_raw", "y_true"}


def save_npz(file: Path, /, **arrays: np.ndarray) -> None:
    """Write arrays to a compressed .npz; refuses names that suggest CheXpert labels."""
    assert not LABEL_KEYS & set(arrays), f"{file.name}: labels must not be saved in run files"
    np.savez_compressed(file, **arrays)


def load_checkpoint(model: nn.Module, name: str) -> None:
    """Load ``checkpoints/<run>/<name>`` into ``model``."""
    model.load_state_dict(torch.load(CKPT_DIR / name, map_location=DEVICE, weights_only=True))


def predict_df(model: nn.Module, df: pd.DataFrame) -> np.ndarray:
    """Logits for every row of ``df``, in row order."""
    return predict_logits(model, make_loader(df, train=False, seed=SEED))


def write_predictions(
    policy: Policy, split: str, df: pd.DataFrame, logits: np.ndarray, suffix: str = ""
) -> np.ndarray:
    """Save path, logits and scores (never labels) as ``<split>_preds<suffix>.npz``."""
    scores = scores_from_logits(policy, logits)
    save_npz(
        RUN_DIR / f"{split}_preds{suffix}.npz",
        path=df["path"].to_numpy(),
        logits=logits,
        scores=scores,
    )
    return scores


@torch.no_grad()
def extract_features_df(model: DenseNetClassifier, df: pd.DataFrame) -> torch.Tensor:
    """Pooled 1024-d backbone features for every row of ``df``, (N, 1024) on ``DEVICE``."""
    model.eval()
    outputs = []
    for x, _, _ in make_loader(df, train=False, seed=SEED):
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            outputs.append(model.extract_features(to_model_input(x, DEVICE)).float())
    return torch.cat(outputs)


@torch.no_grad()
def mc_probabilities(
    model: DenseNetClassifier,
    features: torch.Tensor,
    score: Callable[[torch.Tensor], torch.Tensor],
    passes: int,
    seed: int,
) -> torch.Tensor:
    """(passes, N, 5) probabilities: dropout and head applied ``passes`` times to ``features``.

    Only the dropout layer is stochastic; the generator is seeded so the masks repeat.
    """
    model.eval()
    model.dropout.train()
    devices = [features.device.index or 0] if features.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            return torch.stack([score(model.classify(features)) for _ in range(passes)])
    finally:
        model.eval()


def mc_arrays(
    model: DenseNetClassifier, policy: Policy, df: pd.DataFrame, deterministic: np.ndarray
) -> dict[str, np.ndarray]:
    """MC Dropout outputs for ``df``: mean probability, predictive entropy, dropout-off score."""
    features = extract_features_df(model, df)
    probs = mc_probabilities(model, features, policy.score, CONFIG["mc_passes"], SEED)
    mean = probs.mean(dim=0).cpu().numpy()
    return {
        "path": df["path"].to_numpy(),
        "mean": mean,
        "entropy": binary_entropy(mean.astype(np.float64)).astype(np.float32),
        "deterministic": deterministic,
    }


def mc_equivalence_gap(model: DenseNetClassifier, policy: Policy, x: torch.Tensor) -> float:
    """Largest difference between cached-feature MC and a naive loop of full passes on ``x``."""
    passes = CONFIG["mc_passes"]
    cached = mc_probabilities(model, model.extract_features(x), policy.score, passes, SEED)
    model.eval()
    model.dropout.train()
    devices = [x.device.index or 0] if x.device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(SEED)
        naive = torch.stack([policy.score(model(x)) for _ in range(passes)])
    model.eval()
    return (cached - naive).abs().max().item()


def compute_cams(model: DenseNetClassifier, df: pd.DataFrame, rows: list[int]) -> np.ndarray:
    """Class activation maps for head ``rows``, (N, len(rows), 10, 10), in float32."""
    model.eval()
    maps = []
    for x, _, _ in make_loader(df, train=False, seed=SEED):
        maps.append(model.cam(to_model_input(x, DEVICE), rows).float().cpu())
    return torch.cat(maps).numpy()


def random_batch(batch_size: int) -> torch.Tensor:
    """A model-ready batch of random images, for timing and FLOP counting."""
    size = CONFIG["image_size"]
    raw = torch.randint(0, 256, (batch_size, 1, size, size), dtype=torch.uint8)
    return to_model_input(raw, DEVICE)


@torch.no_grad()
def time_per_image(fn: Callable[[], object], batch_size: int, warmup: int, timed: int) -> float:
    """Milliseconds per image for ``fn`` (one batch), after warm-up, with GPU synchronisation."""
    sync = torch.cuda.synchronize if DEVICE.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(timed):
        fn()
    sync()
    return (time.perf_counter() - start) / (timed * batch_size) * 1000


@torch.no_grad()
def gflops_per_image(model: nn.Module) -> float:
    """GFLOPs (multiply-adds counted as two) of one forward pass on a single image."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with FlopCounterMode(display=False) as counter:
        model(random_batch(1))
    return counter.get_total_flops() / 1e9


if MODE == "train":
    ids_val, ids_test = val_df["path"].to_numpy(), TEST["path"].to_numpy()
    model.eval()

    # Primary checkpoint (best.pt): validation and test predictions.
    val_logits, test_logits = predict_df(model, val_df), predict_df(model, TEST)
    val_scores = write_predictions(policy, "val", val_df, val_logits)
    test_scores = write_predictions(policy, "test", TEST, test_logits)
    recomputed = validation_aurocs(policy, label_matrix(val_df), val_scores)[0]
    logged = float(metrics.loc[metrics["epoch"] == info["best_epoch"], "val_auroc"].iloc[0])
    assert abs(recomputed - logged) < 1e-3, f"saved predictions give {recomputed}, log {logged}"
    log(f"predictions written; validation AUROC {recomputed:.4f} matches the epoch log")

    # Second checkpoint (best_certain.pt): its own files only when it is a different epoch.
    certain_differs = info["best_certain_epoch"] != info["best_epoch"]
    load_checkpoint(model, "best_certain.pt")
    if certain_differs:
        write_predictions(policy, "val", val_df, predict_df(model, val_df), "_certain")
        write_predictions(policy, "test", TEST, predict_df(model, TEST), "_certain")
    else:
        same = predict_df(model, TEST)
        assert np.allclose(same, test_logits, atol=1e-3), "best_certain.pt should equal best.pt"
    load_checkpoint(model, "best.pt")

    # MC Dropout: cached-feature passes must equal a naive loop of full passes.
    if policy.mc:
        probe = next(iter(make_loader(val_df.head(4), train=False, seed=SEED, num_workers=0)))
        gap = mc_equivalence_gap(model, policy, to_model_input(probe[0], DEVICE))
        assert gap < 1e-5, f"cached MC differs from {CONFIG['mc_passes']} full passes by {gap:.2e}"
        mc_val = mc_arrays(model, policy, val_df, val_scores)
        mc_test = mc_arrays(model, policy, TEST, test_scores)
        for name, arrays in [("val", mc_val), ("test", mc_test)]:
            save_npz(RUN_DIR / f"mc_{name}.npz", **arrays)
            assert ((arrays["mean"] >= 0) & (arrays["mean"] <= 1)).all()
            assert arrays["entropy"].min() >= 0 and arrays["entropy"].max() <= np.log(2) + 1e-6
        assert not np.allclose(mc_val["mean"], mc_val["deterministic"]), "dropout had no effect"
        log(f"MC Dropout outputs written (cached vs naive passes differ by {gap:.1e})")

    # Class activation maps: every test image and the fixed qualitative validation images.
    cam_val_df = qual_val.head(RUN_LIMITS["qual_images"]) if SMOKE else qual_val
    for name, frame in [("test", TEST), ("val_subset", cam_val_df)]:
        maps = compute_cams(model, frame, policy.cam_rows)
        assert maps.shape == (len(frame), N_PATH, 10, 10) and np.isfinite(maps).all()
        save_npz(RUN_DIR / f"cam_{name}.npz", path=frame["path"].to_numpy(), cam=maps)

    # Cost: inference time on a model-ready random batch, FLOPs and the MC head cost.
    batch = RUN_LIMITS["batch_size"]
    warmup, timed = (3, 20) if DEVICE.type == "cuda" else (1, 2)
    x_time = random_batch(batch)
    with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
        ms_per_image = time_per_image(lambda: model(x_time), batch, warmup, timed)
    costs = {
        "inference_ms_per_image": ms_per_image,
        "inference_batch_size": batch,
        "gflops_per_image": gflops_per_image(model),
    }
    assert costs["gflops_per_image"] > 0 and costs["inference_ms_per_image"] > 0
    if policy.mc:
        feats = model.extract_features(x_time).detach()
        costs["mc_head_ms_per_image"] = time_per_image(
            lambda: mc_probabilities(model, feats, policy.score, CONFIG["mc_passes"], SEED),
            batch,
            warmup,
            timed,
        )
    log(f"inference {ms_per_image:.2f} ms/image, {costs['gflops_per_image']:.1f} GFLOPs/image")

    # Final check of the run folder: expected files only, small enough to push, no labels.
    expected = {
        "run_info.json", "metrics_epoch.csv", "grad_stats.csv", "train.log",
        "val_preds.npz", "test_preds.npz", "cam_test.npz", "cam_val_subset.npz",
    }  # fmt: skip
    expected |= {"gpu_trace.csv"} if DEVICE.type == "cuda" else set()
    expected |= {"val_preds_certain.npz", "test_preds_certain.npz"} if certain_differs else set()
    expected |= {"mc_val.npz", "mc_test.npz"} if policy.mc else set()
    written = {f.name for f in RUN_DIR.iterdir()}
    assert written == expected, f"unexpected run files: {written ^ expected}"
    for f in RUN_DIR.glob("*.npz"):
        with np.load(f) as data:
            assert not LABEL_KEYS & set(data.files), f"{f.name} contains labels"
    size_mb = sum(f.stat().st_size for f in RUN_DIR.iterdir()) / 1e6
    assert size_mb < 25, f"run folder is {size_mb:.1f} MB; it must stay small enough to push"

    write_run_info(
        "complete",
        certain_checkpoint_differs=certain_differs,
        n_val_predictions=len(val_df),
        n_test_predictions=len(TEST),
        **costs,
    )
    log(f"run complete: {len(written)} files, {size_mb:.2f} MB in {RUN_DIR}")
