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
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

CONFIG = {
    "code_version": "v1",
    "pathologies": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
    "image_size": 320,
    "val_fraction": 0.10,
    "split_seed": 42,
    "seeds": (42, 43, 44),
    "batch_size": 32,
    "max_epochs": 10,
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

check_rng = np.random.default_rng(0)
for frame in (TRAIN_ALL, TEST):
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
