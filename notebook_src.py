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
import platform
import random
import re
import subprocess
import sys
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
        batch_size=CONFIG["batch_size"],
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
