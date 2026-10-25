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
import io
import json
import math
import platform
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
#
# `RunPusher` publishes a run folder to its own branch (`run/<policy>_s<seed>`) from a
# throw-away repository, so the push can only ever contain that folder. It is used by the
# training run (sections 3.2 and 3.4).

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


assert not any(s in scrub(f"x {GITHUB_PAT} {_AUTH_HEADER} y") for s in _SECRETS), "scrub leaks"


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


def remote_url() -> str:
    """URL of the GitHub repository; the ``REMOTE_URL`` variable overrides it (local tests)."""
    if os.environ.get("REMOTE_URL"):
        return os.environ["REMOTE_URL"]
    assert GITHUB_REPO, "Kaggle Secret GITHUB_REPO (owner/name) is not set"
    return f"https://github.com/{GITHUB_REPO}.git"


def clone_repo(ref: str, dest: Path) -> Path:
    """Shallow-clone ``GITHUB_REPO`` at tag or branch ``ref`` into ``dest`` (reused if present)."""
    if not (dest / ".git").is_dir():
        run_git("clone", "--depth", "1", "--branch", ref, remote_url(), str(dest), auth=True)
    head = run_git("rev-parse", "--short", "HEAD", cwd=dest).strip()
    print(f"repo {GITHUB_REPO} at {ref} ({head}) in {dest}")
    return dest


PUSH_SUFFIXES = {".json", ".csv", ".log", ".npz"}  # nothing else may leave the notebook
PUSH_MAX_MB = 25


class RunPusher:
    """Pushes one run folder to its own orphan branch, never touching ``main`` or any code.

    The branch is ``run/<policy>_s<seed>`` (``smoke/...`` for a smoke pass) and holds only the
    run folder, committed from a throw-away repository outside the working directory. ``claim``
    makes the first push and refuses a run branch that already exists, so two people who picked
    the same policy and seed cannot overwrite each other. Every later push is a plain
    fast-forward. Smoke branches are disposable and are simply overwritten.
    """

    def __init__(
        self, run_dir: Path, repo_dir: Path, smoke: bool, who: str, log: Callable[[str], None]
    ) -> None:
        self.run_dir, self.smoke, self.who, self.log = run_dir, smoke, who, log
        self.rel = run_dir.relative_to(repo_dir).as_posix()  # path inside the branch
        self.branch = f"{'smoke' if smoke else 'run'}/{run_dir.name}"
        self.enabled = bool(os.environ.get("REMOTE_URL") or (GITHUB_PAT and GITHUB_REPO))
        self.work: Path | None = None

    def claim(self) -> None:
        """Check the branch is free, create the throw-away repository and push the first state."""
        if not self.enabled:
            if IS_KAGGLE:
                raise RuntimeError(
                    "Kaggle Secrets GITHUB_PAT and GITHUB_REPO are not both set, so this run could "
                    "not be pushed. Attach them (Add-ons > Secrets) and run again."
                )
            self.log(
                "push skipped: no GITHUB_PAT/GITHUB_REPO; outputs stay in the working directory"
            )
            return
        url = remote_url()
        if (
            not self.smoke
            and run_git("ls-remote", "--heads", url, f"refs/heads/{self.branch}", auth=True).strip()
        ):
            raise RuntimeError(
                f"branch {self.branch} already exists on the remote: someone has already started "
                "this policy and seed. Pick another policy or seed, or ask the builder to delete "
                "the branch if that run failed (a fixed run needs a new version tag)."
            )
        self.work = Path(tempfile.mkdtemp(prefix="push_"))
        run_git("init", "-q", "-b", self.branch, str(self.work))
        for key, value in [
            ("user.name", self.who),
            ("user.email", f"{self.who}@users.noreply.github.com"),
            ("commit.gpgsign", "false"),
            # Behave the same on every machine: a global ignore file (for example `*.log`)
            # would silently drop run files, and global hooks have no business here.
            ("core.excludesFile", os.devnull),
            ("core.hooksPath", os.devnull),
        ]:
            run_git("config", key, value, cwd=self.work)
        run_git("remote", "add", "origin", url, cwd=self.work)
        self.push(f"{self.run_dir.name}: started", strict=True, force=self.smoke)
        self.log(f"pushing to branch {self.branch} of {url}")

    def _commit(self, message: str) -> None:
        """Copy the run folder into the repository after checking it is safe, and commit."""
        files = sorted(self.run_dir.iterdir())
        bad = [f.name for f in files if f.suffix not in PUSH_SUFFIXES or not f.is_file()]
        assert not bad, f"refusing to push {bad}: only {sorted(PUSH_SUFFIXES)} files may be pushed"
        size_mb = sum(f.stat().st_size for f in files) / 1e6
        assert size_mb < PUSH_MAX_MB, (
            f"run folder is {size_mb:.1f} MB, over the {PUSH_MAX_MB} MB cap"
        )
        secrets = [s.encode() for s in _SECRETS]
        leaked = [f.name for f in files if any(s in f.read_bytes() for s in secrets)]
        assert not leaked, f"refusing to push {leaked}: contains the GitHub token"
        dest = self.work / self.rel
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(self.run_dir, dest)
        run_git("add", "-A", cwd=self.work)
        if run_git("status", "--porcelain", cwd=self.work).strip():
            run_git("commit", "-q", "-m", message, cwd=self.work)

    def push(self, message: str, strict: bool = False, force: bool = False) -> bool:
        """Commit the run folder and push it; True on success.

        With ``strict=False`` a failure only logs a warning, so a network blip never kills a
        training run; with ``strict=True`` the push is tried three times and then raises.
        """
        if self.work is None:
            return False
        assert not force or self.smoke, "only smoke branches may be overwritten"
        try:
            self._commit(message)
            for attempt in range(3 if strict else 1):
                try:
                    cmd = ["push", "-q", *(["--force"] if force else [])]
                    run_git(
                        *cmd, "origin", f"HEAD:refs/heads/{self.branch}", cwd=self.work, auth=True
                    )
                    return True
                except RuntimeError:
                    if attempt == 2 or not strict:
                        raise
                    time.sleep(5 * (attempt + 1))
        except Exception as err:
            if strict:
                raise
            self.log(f"warning: push to {self.branch} failed, the run continues: {err}")
        return False

    def verify(self) -> str:
        """Check the remote branch is at our last commit and holds exactly the run folder."""
        head = run_git("rev-parse", "HEAD", cwd=self.work).strip()
        out = run_git("ls-remote", "origin", f"refs/heads/{self.branch}", cwd=self.work, auth=True)
        assert out.split()[:1] == [head], f"remote {self.branch} is not at the pushed commit"
        tree = run_git("ls-tree", "-r", "--name-only", "HEAD", cwd=self.work).split()
        expected = {f"{self.rel}/{f.name}" for f in self.run_dir.iterdir()}
        assert set(tree) == expected, (
            f"pushed files differ from the run folder: {set(tree) ^ expected}"
        )
        return head[:10]

    def close(self) -> None:
        """Delete the throw-away repository."""
        if self.work is not None:
            shutil.rmtree(self.work, ignore_errors=True)
            self.work = None


def find_data_root(inputs: Path = Path("/kaggle/input")) -> Path:
    """Locate the folder holding CheXpert-small's ``train.csv`` and ``valid.csv``.

    ``DATA_ROOT`` wins when set. On Kaggle the attached inputs are searched one directory level
    at a time, down to four levels, stopping at the first level that holds the two files, so the
    image folders (about 220,000 files) are never walked.
    """
    if os.environ.get("DATA_ROOT"):
        candidates = [Path(os.environ["DATA_ROOT"]).resolve()]
    elif IS_KAGGLE:
        level, candidates = [inputs], []
        for _ in range(5):
            candidates = [d for d in level if (d / "train.csv").is_file()]
            if candidates:
                break
            level = [child for d in level for child in sorted(d.iterdir()) if child.is_dir()]
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


SAVED = {"tables": [], "figures": []}  # names written so far, listed by the export index


def save_table(df: pd.DataFrame, name: str, float_format: str = "%.1f") -> None:
    """Write ``df`` to ``results/tables/<name>.csv`` and ``.tex`` (booktabs)."""
    if name not in SAVED["tables"]:
        SAVED["tables"].append(name)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    latex = df.to_latex(index=False, escape=True, float_format=lambda x: float_format % x)
    (TABLES_DIR / f"{name}.tex").write_text(latex)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Write ``fig`` to ``results/figures/<name>.pdf`` and a 300 dpi ``.png``."""
    if name not in SAVED["figures"]:
        SAVED["figures"].append(name)
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


def paired_summary(d: np.ndarray, point: float) -> dict:
    """Difference ``point`` with its 95% CI and two-sided paired p-value from bootstrap ``d``."""
    d = d[np.isfinite(d)]
    b = len(d)
    p = min(1.0, 2 * min((1 + np.sum(d <= 0)) / (b + 1), (1 + np.sum(d >= 0)) / (b + 1)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "p": float(p)}


def paired_difference(
    stat_a: Callable[[np.ndarray], float],
    stat_b: Callable[[np.ndarray], float],
    resamples: list[np.ndarray],
    n: int,
) -> dict:
    """Difference A - B with 95% CI and two-sided paired bootstrap p-value."""
    d = np.array([stat_a(idx) - stat_b(idx) for idx in resamples])
    full = np.arange(n)
    return paired_summary(d, stat_a(full) - stat_b(full))


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
#
# The run claims its GitHub branch before training starts (see `RunPusher` in section 0.3) and
# pushes the folder after every epoch, so a session that dies still leaves its history there.

# %%
RUN_NAME = f"{POLICY}_s{SEED}"
RUNS_SUBDIR = "runs_smoke" if SMOKE else "runs"
RUN_DIR = REPO_DIR / RUNS_SUBDIR / RUN_NAME
CKPT_DIR = REPO_DIR / "checkpoints" / RUN_NAME


IO_THREADS = 64  # threads for the cache warm-up; speed only, never affects results


def warm_file_cache(paths: list[Path], log: Callable[[str], None]) -> None:
    """Read every file once so the operating system keeps it in memory for the epochs to come.

    The first read of an image from Kaggle's dataset mount takes about 0.3 s, so four loader
    workers manage only about 12 images/s, while a cached image costs almost nothing. Waiting
    for storage uses no CPU, so many threads can overlap the waits: this pass reads in parallel
    and discards the bytes. Training and its results are unchanged; it only moves the slow
    first read out of the epoch loop.
    """

    def read(path: Path) -> int:
        with open(path, "rb") as f:
            return len(f.read())

    start, total = time.perf_counter(), 0
    with ThreadPoolExecutor(IO_THREADS) as pool:
        for done, size in enumerate(pool.map(read, paths), start=1):
            total += size
            if done % 25_000 == 0 or done == len(paths):
                speed = done / (time.perf_counter() - start)
                progress = f"{done:,}/{len(paths):,} files, {total / 1e9:.1f} GB"
                log(f"cache warm-up: {progress}, {speed:.0f} files/s")


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
    pusher = RunPusher(RUN_DIR, REPO_DIR, SMOKE, RUN_BY, log)
    pusher.claim()  # stops here, before any training, if the branch is taken
    needed = pd.concat([train_df["path"], val_df["path"], TEST["path"], qual_val["path"]])
    warm_file_cache([DATA_ROOT / path for path in needed.unique()], log)
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
                on_epoch_end=lambda: pusher.push(f"{RUN_NAME}: epoch log"),
            )
    except Exception as err:
        log(f"FAILED: {err!r}")
        write_run_info("failed", failure=repr(err))
        pusher.push(f"{RUN_NAME}: failed")
        raise
    info = write_run_info("trained", trainable_params=count_trainable_params(model), **summary)
    log(f"trained: best epoch {info['best_epoch']} of {info['epochs_run']} ({info['stop_reason']})")
    pusher.push(f"{RUN_NAME}: trained")

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
    """Write arrays to a compressed .npz; refuses labels and any array that needs pickling."""
    assert not LABEL_KEYS & set(arrays), f"{file.name}: labels must not be saved in run files"
    objects = [name for name, a in arrays.items() if a.dtype == object]
    assert not objects, f"{file.name}: {objects} would need pickle; use fixed-width strings"
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
        path=df["path"].to_numpy(dtype=str),
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
        "path": df["path"].to_numpy(dtype=str),
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
        save_npz(RUN_DIR / f"cam_{name}.npz", path=frame["path"].to_numpy(dtype=str), cam=maps)

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
        with np.load(f) as data:  # reading every array proves none needs pickle
            arrays = {name: data[name] for name in data.files}
        assert not LABEL_KEYS & set(arrays), f"{f.name} contains labels"
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

# %% [markdown]
# ## 3.4 Push to GitHub
#
# Pushes the finished run folder (status `complete`) to `run/<policy>_s<seed>`, then checks that
# the branch on GitHub ends at that commit and holds exactly the files of the run folder. This
# final push is the one that matters: it is tried three times and the notebook fails loudly if
# it does not land, so a "successful" run is never silently missing from the branch. (If it
# fails, the files are still in this notebook's output.) The branch holds the run folder only:
# no code, no checkpoints (`best.pt` stays in the notebook output) and no CheXpert labels.

# %%
if MODE == "train":
    if pusher.work is None:
        print("push skipped (no GitHub secrets); the run folder stays in the working directory")
    else:
        pusher.push(f"{RUN_NAME}: complete", strict=True)
        tip = pusher.verify()
        pushed_names = {f.name for f in RUN_DIR.iterdir()}
        assert not any(name.endswith((".pt", ".pth", ".ckpt")) for name in pushed_names)
        print(f"pushed {len(pushed_names)} files to branch {pusher.branch} (commit {tip})")
        pusher.close()


# %% [markdown]
# # Part 4: Results (`MODE == "results"`)
#
# ## 4.1 Fetch the runs, inventory and consistency check
#
# Every run trained in its own notebook and pushed its folder to `run/<policy>_s<seed>`. This
# part downloads those folders and lists every expected run in an **inventory**, with runs that
# are not there (yet) shown as `missing`. Only a run that is `complete` and passes every check
# below is used; any other run is listed with the reason. A run is comparable with the others
# when it has
#
# - the same code version and the same frozen `CONFIG`,
# - the same train/validation split and the same dataset files (their fingerprints),
# - the same test images in the same order, and
# - the policy and seed its branch name says, and all the files a run must write.
#
# `N_SEEDS` chooses how many seeds are analysed: `"1"` uses seed 42 only (the proposal's
# protocol), `"2"` and `"3"` add seeds 43 and 44, and `"auto"` takes the largest number for
# which every policy has finished every seed. An explicit number with runs missing stops here
# with the list of what is missing, so a partial seed layer can never leak into the tables. The
# extra class-weighting run (`u_zeros_posw`, seed 42) is loaded separately and is not part of
# the six-policy comparison.

# %%
BRANCH_PREFIX = "smoke" if SMOKE else "run"
FETCH_DIR = REPO_DIR / "fetched_runs"
EXTRA_POLICY = next(name for name, policy_ in POLICIES.items() if policy_.extra)
BASE_FILES = {
    "run_info.json", "metrics_epoch.csv", "grad_stats.csv", "train.log",
    "val_preds.npz", "test_preds.npz", "cam_test.npz", "cam_val_subset.npz",
}  # fmt: skip


@dataclass
class Run:
    """One run folder fetched from GitHub, with the reasons it cannot be used (if any)."""

    policy: str
    seed: int
    dir: Path | None = None
    info: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.policy}_s{self.seed}"

    @property
    def status(self) -> str:
        """``missing`` without a branch, else the status the run itself recorded."""
        return "missing" if self.dir is None else self.info.get("status", "unknown")

    @property
    def usable(self) -> bool:
        return self.status == "complete" and not self.problems


def fetch_runs(names: list[str], dest: Path) -> dict[str, Path]:
    """Download the folders of the run branches that exist; return ``{run name: folder}``."""
    remote = remote_url()
    listing = run_git("ls-remote", "--heads", remote, f"refs/heads/{BRANCH_PREFIX}/*", auth=True)
    on_remote = {line.split("refs/heads/")[1] for line in listing.splitlines()}
    branches = [f"{BRANCH_PREFIX}/{n}" for n in names if f"{BRANCH_PREFIX}/{n}" in on_remote]
    found: dict[str, Path] = {}
    if not branches:
        return found
    with tempfile.TemporaryDirectory(prefix="fetch_") as tmp:
        scratch = Path(tmp)
        run_git("init", "-q", "--bare", str(scratch))
        specs = [f"+refs/heads/{b}:refs/heads/{b}" for b in branches]
        run_git("fetch", "-q", "--depth", "1", remote, *specs, cwd=scratch, auth=True)
        for branch in branches:
            name = branch.split("/", 1)[1]
            archive = ["git", "archive", "--format=tar", branch, f"{RUNS_SUBDIR}/{name}"]
            tarball = subprocess.run(archive, cwd=scratch, capture_output=True, check=True).stdout
            shutil.rmtree(dest / RUNS_SUBDIR / name, ignore_errors=True)
            with tarfile.open(fileobj=io.BytesIO(tarball)) as tar:
                tar.extractall(dest, filter="data")
            found[name] = dest / RUNS_SUBDIR / name
    return found


def load_run(policy: str, seed: int, folder: Path | None) -> Run:
    """A ``Run`` with its ``run_info.json`` read (a missing or broken file is a problem)."""
    run = Run(policy, seed, folder)
    if folder is not None:
        try:
            run.info = json.loads((folder / "run_info.json").read_text())
        except (OSError, json.JSONDecodeError) as err:
            run.problems.append(f"run_info.json unreadable: {err}")
    return run


def expected_files(run: Run) -> set[str]:
    """Files a complete run folder holds, which depend on the policy and on its checkpoints."""
    files = set(BASE_FILES)
    if run.info.get("certain_checkpoint_differs"):
        files |= {"val_preds_certain.npz", "test_preds_certain.npz"}
    if POLICIES[run.policy].mc:
        files |= {"mc_val.npz", "mc_test.npz"}
    if run.info.get("gpu_count"):
        files.add("gpu_trace.csv")
    return files


def check_run(run: Run, test_paths: list[str]) -> list[str]:
    """Reasons ``run`` cannot be combined with the others; empty when it is fine."""
    info = run.info
    if info.get("status") != "complete":
        return []  # an unfinished run is not an error: its status column says it
    version = CONFIG["code_version"]
    checks = [
        (
            (info.get("policy"), info.get("seed")) == (run.policy, run.seed),
            "policy or seed differs from the branch name",
        ),
        (info.get("code_version") == version, f"code version is not {version!r}"),
        (info.get("config") == json.loads(json.dumps(CONFIG)), "CONFIG differs from this notebook"),
        (SMOKE or info.get("run_limits") == RUN_LIMITS, "run limits differ from this notebook"),
        (info.get("split_sha256") == SPLIT_SHA256, "different train/validation split"),
        (info.get("data_fingerprint") == DATA_FINGERPRINT, "different dataset files"),
        (info.get("smoke") == SMOKE, f"smoke flag is {info.get('smoke')}, notebook has {SMOKE}"),
    ]
    problems = [text for ok, text in checks if not ok]
    present, expected = {f.name for f in run.dir.iterdir()}, expected_files(run)
    if present != expected:
        problems.append(f"files differ from the expected set: {sorted(present ^ expected)}")
    elif list(np.load(run.dir / "test_preds.npz")["path"]) != test_paths:
        problems.append("test images (or their order) differ from this notebook's test set")
    return problems


def describe(run: Run) -> str:
    """``name: status`` followed by the run's problems, for error messages."""
    return f"{run.name}: {run.status}" + "".join(f"; {text}" for text in run.problems)


def choose_seeds(runs: dict[tuple[str, int], Run], n_seeds: str) -> list[int]:
    """Seeds to analyse for ``N_SEEDS``; raises with the list of unusable runs if impossible."""
    seeds = list(CONFIG["seeds"])
    layer_ok = {s: all(runs[(p, s)].usable for p in MAIN_POLICIES) for s in seeds}
    if n_seeds == "auto":
        chosen = []
        for seed in seeds:
            if not layer_ok[seed]:
                break
            chosen.append(seed)
        if not chosen:
            raise RuntimeError(f"no complete layer: every policy at seed {seeds[0]} is needed")
        return chosen
    wanted = seeds[: int(n_seeds)]
    unusable = [runs[(p, s)] for s in wanted for p in MAIN_POLICIES if not runs[(p, s)].usable]
    if unusable:
        lines = "\n  ".join(describe(run) for run in unusable)
        raise RuntimeError(f"N_SEEDS={n_seeds} needs runs that are not usable:\n  {lines}")
    return wanted


def inventory_table(runs: list[Run], used: set[str]) -> pd.DataFrame:
    """One row per expected run: who ran it, where, how far it got and whether it is analysed."""
    rows = []
    for run in runs:
        info = run.info
        limit = (info.get("run_limits") or {}).get("max_epochs", "?")
        hours = info["train_seconds"] / 3600 if "train_seconds" in info else None
        rows.append(
            {
                "run": run.name,
                "who": info.get("run_by", "-"),
                "status": run.status,
                "gpu": info.get("gpu_name", "-"),
                "epochs": f"{info['epochs_run']}/{limit}" if "epochs_run" in info else "-",
                "best_epoch": info.get("best_epoch", "-"),
                "train_hours": f"{hours:.2f}" if hours is not None else "-",
                "used": "yes" if run.name in used else "no",
                "problems": "; ".join(run.problems),
            }
        )
    return pd.DataFrame(rows)


def protocol_table(runs: list[Run]) -> pd.DataFrame:
    """Settings every analysed run shares (the protocol the paper states), plus the hardware."""
    cfg, first = runs[0].info["config"], runs[0].info
    size, val = cfg["image_size"], cfg["val_fraction"]
    gpus = sorted({r.info["gpu_name"] for r in runs})
    software = sorted({f"torch {r.info['torch']}, CUDA {r.info['cuda']}" for r in runs})
    rows = {
        "code version": cfg["code_version"],
        "backbone": "DenseNet-121, ImageNet weights, all layers fine-tuned",
        "input": f"{size} x {size}, grayscale copied to 3 channels, ImageNet normalisation",
        "split": f"patient level, {1 - val:.0%} / {val:.0%}, seed {cfg['split_seed']}",
        "optimiser": f"{cfg['optimizer']}, lr {cfg['lr']}, weight decay {cfg['weight_decay']}",
        "schedule": (
            f"lr x{cfg['plateau_factor']} after {cfg['plateau_patience']} epoch without gain, "
            f"early stop after {cfg['early_stop_patience']}"
        ),
        "batch size / max epochs": f"{cfg['batch_size']} / {cfg['max_epochs']}",
        "mixed precision": str(cfg["amp"]),
        "augmentation": (
            f"rotation {cfg['rotation_deg']} deg, crop scale {tuple(cfg['crop_scale'])}, "
            f"jitter {cfg['jitter']}, no flip"
        ),
        "MC Dropout": f"p = {cfg['mc_dropout_p']}, {cfg['mc_passes']} passes",
        "label smoothing range": str(tuple(cfg["lsr_range"])),
        "bootstrap": f"{cfg['bootstrap_resamples']} patient-level resamples, 95% intervals",
        "split SHA-256": first["split_sha256"][:16],
        "data fingerprint": first["data_fingerprint"][:16],
        "hardware": ", ".join(gpus),
        "software": "; ".join(software),
    }
    return pd.DataFrame({"setting": list(rows), "value": list(rows.values())})


if MODE == "results":
    test_paths = TEST["path"].tolist()
    expected_runs = [(p, s) for s in CONFIG["seeds"] for p in MAIN_POLICIES] + [(EXTRA_POLICY, 42)]
    fetched = fetch_runs([f"{p}_s{s}" for p, s in expected_runs], FETCH_DIR)
    ALL_RUNS = {(p, s): load_run(p, s, fetched.get(f"{p}_s{s}")) for p, s in expected_runs}
    for run in ALL_RUNS.values():
        if run.dir is not None and not run.problems:
            run.problems = check_run(run, test_paths)

    try:
        RESULT_SEEDS = choose_seeds(ALL_RUNS, N_SEEDS)
        failure = None
    except RuntimeError as err:
        RESULT_SEEDS, failure = [], err
    RUNS = {(p, s): ALL_RUNS[(p, s)] for s in RESULT_SEEDS for p in MAIN_POLICIES}
    EXTRA_RUN = ALL_RUNS[(EXTRA_POLICY, 42)]
    if not EXTRA_RUN.usable:
        EXTRA_RUN = None

    shown = CONFIG["seeds"] if N_SEEDS == "auto" else CONFIG["seeds"][: int(N_SEEDS)]
    shown_runs = [ALL_RUNS[(p, s)] for s in shown for p in MAIN_POLICIES]
    shown_runs.append(ALL_RUNS[(EXTRA_POLICY, 42)])
    used = {r.name for r in RUNS.values()} | ({EXTRA_RUN.name} if EXTRA_RUN else set())
    inventory = inventory_table(shown_runs, used)
    print(inventory.drop(columns="problems").to_string(index=False))
    for run in shown_runs:
        if run.problems:
            print(f"  {run.name}: {'; '.join(run.problems)}")
    save_table(inventory, "run_inventory")
    if failure is not None:
        raise failure
    if not EXTRA_RUN:
        print(f"note: the extra run {EXTRA_POLICY}_s42 is not usable; E8 will be skipped")

    gpu_names = {r.info["gpu_name"] for r in RUNS.values()}
    if len(gpu_names) > 1:
        print(f"warning: the runs used different GPUs: {sorted(gpu_names)}")
    protocol = protocol_table(list(RUNS.values()))
    save_table(protocol, "protocol")
    print(protocol.to_string(index=False))
    extra = "yes" if EXTRA_RUN else "no"
    print(f"analysing seeds {RESULT_SEEDS}: {len(RUNS)} runs, extra run {extra}")

# %% [markdown]
# ## 4.2 Predictions, labels and bootstrap values
#
# Run folders hold image paths, logits and scores but **no labels**; the labels are joined back
# from the dataset CSVs by image path (every path must be found). The test labels are the
# radiologist-consensus labels of `valid.csv`; the internal-validation labels keep the original
# -1 and are used for thresholds, checkpoint selection and the entropy analysis.
#
# Score used for each policy in the tables: the sigmoid probability, the softmax P(positive) for
# U-MultiClass, and for U-Zeros+MC the **mean of the 20 MC Dropout passes** (the dropout-off score
# is kept for the ablation in T6). F1 uses a threshold per pathology: Youden's J on the validation
# predictions with the policy's own validation labels.
#
# Every metric is computed once per run on the test set and on each of the shared patient-level
# bootstrap resamples. With several seeds the interval comes from the **seed-averaged**
# bootstrap: each replicate averages the metric over the seeds on the same resample (predictions
# are never averaged), and differences pair the same seed across policies.

# %%
POLICY_LABELS = {
    "u_zeros": "U-Zeros",
    "u_ones": "U-Ones",
    "u_ignore": "U-Ignore",
    "u_multiclass": "U-MultiClass",
    "u_ones_lsr": "U-Ones+LSR",
    "u_zeros_mc": "U-Zeros+MC",
    "u_zeros_posw": "U-Zeros+PosW",
}
# One fixed colour per policy from a palette checked for colour-blind separation of neighbouring
# colours. Some hues are light on white, so every figure also carries a legend and markers.
POLICY_COLORS = dict(
    zip(
        MAIN_POLICIES,
        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"],
        strict=True,
    )
) | {EXTRA_POLICY: "#6b6a66"}
POLICY_MARKERS = dict(zip(MAIN_POLICIES, "osD^vP", strict=True)) | {EXTRA_POLICY: "X"}
METRICS = ["AUROC", "AUPRC", "F1", "ECE", "Brier"]
LOWER_IS_BETTER = {"ECE", "Brier"}
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
    }
)


@dataclass
class Preds:
    """Scores of one run on the validation and test images (the runs store no labels)."""

    val: np.ndarray  # (n_val, 5) scores used in the tables (MC mean for the MC policy)
    test: np.ndarray
    val_det: np.ndarray  # dropout-off scores of best.pt (the same as val/test except for MC)
    test_det: np.ndarray
    val_cert: np.ndarray  # dropout-off scores of best_certain.pt
    test_cert: np.ndarray
    val_logits: np.ndarray
    test_logits: np.ndarray
    mc_val: dict | None = None
    mc_test: dict | None = None


def load_arrays(folder: Path, name: str) -> dict[str, np.ndarray]:
    """All arrays of ``folder/name`` (an .npz written without pickling)."""
    with np.load(folder / name) as data:
        return {key: data[key] for key in data.files}


def load_preds(run: Run) -> tuple[Preds, np.ndarray]:
    """Predictions of a run and its validation image paths."""
    folder, mc = run.dir, POLICIES[run.policy].mc
    val, test = load_arrays(folder, "val_preds.npz"), load_arrays(folder, "test_preds.npz")
    differs = run.info.get("certain_checkpoint_differs")
    cert_val = load_arrays(folder, "val_preds_certain.npz") if differs else val
    cert_test = load_arrays(folder, "test_preds_certain.npz") if differs else test
    mc_val = load_arrays(folder, "mc_val.npz") if mc else None
    mc_test = load_arrays(folder, "mc_test.npz") if mc else None
    if mc:
        assert np.array_equal(mc_val["deterministic"], val["scores"]), "MC run: dropout-off differs"
    preds = Preds(
        val=mc_val["mean"] if mc else val["scores"],
        test=mc_test["mean"] if mc else test["scores"],
        val_det=val["scores"],
        test_det=test["scores"],
        val_cert=cert_val["scores"],
        test_cert=cert_test["scores"],
        val_logits=val["logits"],
        test_logits=test["logits"],
        mc_val=mc_val,
        mc_test=mc_test,
    )
    return preds, val["path"]


def youden_thresholds(policy_name: str, val_scores: np.ndarray) -> np.ndarray:
    """Per-pathology F1 thresholds: Youden's J on validation with the policy's own labels."""
    labels = POLICIES[policy_name].val_labels(Y_VAL)
    thresholds = []
    for j in range(N_PATH):
        keep = ~np.isnan(labels[:, j])
        y, s = labels[keep, j], val_scores[keep, j]
        thresholds.append(youden_threshold(y, s) if 0 < y.sum() < len(y) else 0.5)
    return np.array(thresholds)


def test_metrics(scores: np.ndarray, thresholds: np.ndarray, idx=slice(None)) -> np.ndarray:
    """(5 metrics, 5 pathologies) on the test rows ``idx``; NaN where a pathology has one class."""
    out = np.full((len(METRICS), N_PATH), np.nan)
    for j in range(N_PATH):
        y, s = Y_TEST[idx, j], scores[idx, j]
        out[:, j] = [
            auroc(y, s),
            auprc(y, s),
            f1_at(y, s, thresholds[j]),
            ece(y, s),
            brier(y, s),
        ]
    return out


def variant_scores(policy_name: str, preds: Preds, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """(validation, test) scores of a run variant: main, off (dropout off), cert, renorm."""
    if variant == "main":
        return preds.val, preds.test
    if variant == "off":
        return preds.val_det, preds.test_det
    if variant == "cert":
        return preds.val_cert, preds.test_cert
    assert variant == "renorm" and policy_name == "u_multiclass", variant
    both = (preds.val_logits, preds.test_logits)
    return tuple(multiclass_renormalised(torch.from_numpy(z)).numpy() for z in both)


def boot_values(key: tuple[str, int, str]) -> tuple[np.ndarray, np.ndarray]:
    """Metrics of run variant ``(policy, seed, variant)``: point (5, 5), bootstrap (B, 5, 5)."""
    if key not in BOOT:
        policy_name, seed, variant = key
        val_scores, test_scores = variant_scores(policy_name, PREDS[(policy_name, seed)], variant)
        thresholds = youden_thresholds(policy_name, val_scores)
        point = test_metrics(test_scores, thresholds)
        values = np.stack([test_metrics(test_scores, thresholds, idx) for idx in TEST_BOOT])
        BOOT[key] = (point, values)
    return BOOT[key]


@dataclass
class Agg:
    """A policy variant across the analysed seeds: per-seed points and seed-averaged bootstrap."""

    point_seed: np.ndarray  # (n_seeds, 5 metrics, 5 pathologies)
    boot: np.ndarray  # (B, 5 metrics, 5 pathologies): mean over seeds on each resample


def agg(policy_name: str, variant: str = "main", seeds: list[int] | None = None) -> Agg:
    """Aggregate a policy variant over ``seeds`` (default: all analysed seeds)."""
    keys = [(policy_name, s, variant) for s in (seeds or RESULT_SEEDS)]
    points = np.stack([boot_values(k)[0] for k in keys])
    values = np.stack([boot_values(k)[1] for k in keys]).mean(axis=0)
    return Agg(points, values)


def pick(arr: np.ndarray, metric: str, j: int | None) -> np.ndarray:
    """Metric ``metric`` of pathology ``j`` (``None``: mean over the five pathologies)."""
    x = arr[..., METRICS.index(metric), :]
    return np.nanmean(x, axis=-1) if j is None else x[..., j]


def summary(a: Agg, metric: str, j: int | None = None) -> dict:
    """Mean over seeds, 95% bootstrap interval and SD across seeds (None for one seed)."""
    points, boot = pick(a.point_seed, metric, j), pick(a.boot, metric, j)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    sd = float(points.std(ddof=1)) if len(points) > 1 else None
    return {"point": float(points.mean()), "lo": float(lo), "hi": float(hi), "sd": sd}


def fmt(s: dict, digits: int = 3) -> str:
    """``0.812 [0.751, 0.866]`` with ``± sd`` appended when there are several seeds."""
    text = f"{s['point']:.{digits}f} [{s['lo']:.{digits}f}, {s['hi']:.{digits}f}]"
    return text + (f" ± {s['sd']:.{digits}f}" if s["sd"] is not None else "")


def paired(a: Agg, b: Agg, metric: str, j: int | None = None) -> dict:
    """A - B for ``metric``: difference, 95% CI, paired p-value and SD over seeds."""
    d_seed = pick(a.point_seed, metric, j) - pick(b.point_seed, metric, j)
    d_boot = pick(a.boot, metric, j) - pick(b.boot, metric, j)
    out = paired_summary(d_boot, d_seed.mean())
    out["sd"] = float(d_seed.std(ddof=1)) if len(d_seed) > 1 else None
    return out


def fmt_diff(d: dict, digits: int = 3) -> str:
    """``+0.012 [-0.004, 0.027]`` with ``± sd`` over seeds when there are several."""
    text = f"{d['diff']:+.{digits}f} [{d['lo']:+.{digits}f}, {d['hi']:+.{digits}f}]"
    return text + (f" ± {d['sd']:.{digits}f}" if d["sd"] is not None else "")


def fmt_p(p: float) -> str:
    """A p-value to three decimals."""
    return f"{p:.3f}"


def roc_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False and true positive rates along the ROC curve, starting at (0, 0)."""
    _, tps, fps = _sorted_counts(y, s)
    return np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]]


def policy_legend(fig: plt.Figure, policies: list[str], y: float = -0.02) -> None:
    """One shared legend below ``fig`` with each policy's colour and marker."""
    handles = [
        plt.Line2D([], [], color=POLICY_COLORS[p], marker=POLICY_MARKERS[p], label=POLICY_LABELS[p])
        for p in policies
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(policies), frameon=False,
        bbox_to_anchor=(0.5, y),
    )  # fmt: skip


if MODE == "results":
    BOOT: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    USED_RUNS = dict(RUNS) | ({(EXTRA_POLICY, 42): EXTRA_RUN} if EXTRA_RUN else {})
    PREDS, val_path_sets = {}, {}
    for key, run in USED_RUNS.items():
        PREDS[key], val_path_sets[key] = load_preds(run)
    first_paths = next(iter(val_path_sets.values()))
    assert all(np.array_equal(first_paths, v) for v in val_path_sets.values()), "val images differ"

    # Labels come from the dataset CSVs, joined by image path (every path must be found).
    missing_val = set(first_paths) - set(VAL["path"])
    assert not missing_val, f"validation paths not in the dataset: {sorted(missing_val)[:3]}"
    VAL_ROWS = VAL.set_index("path").loc[first_paths].reset_index()
    Y_VAL = label_matrix(VAL_ROWS)  # raw: 1, 0 (negative or not mentioned), -1
    Y_TEST = label_matrix(TEST).astype(np.float64)  # radiologist consensus: 0 or 1
    assert set(np.unique(Y_TEST)) <= {0.0, 1.0}
    for key, preds in PREDS.items():
        assert preds.val.shape == (len(VAL_ROWS), N_PATH) and preds.test.shape == Y_TEST.shape, key

    n_boot = CONFIG["bootstrap_resamples"]
    TEST_BOOT = bootstrap_indices(TEST["patient_id"].to_numpy(), n_boot, BOOTSTRAP_SEED)
    start = time.perf_counter()
    for key in USED_RUNS:
        boot_values((*key, "main"))
    print(
        f"{len(PREDS)} runs, {len(VAL_ROWS):,} validation and {len(TEST)} test images, "
        f"{n_boot} patient-level resamples ({time.perf_counter() - start:.0f} s)"
    )
    # Cross-check against the independently written metric code: the point estimate of the first
    # run must equal scikit-learn's AUROC.
    first_key = next(iter(USED_RUNS))
    point = BOOT[(*first_key, "main")][0]
    expected = [roc_auc_score(Y_TEST[:, j], PREDS[first_key].test[:, j]) for j in range(N_PATH)]
    assert np.allclose(point[0], expected), "AUROC differs from scikit-learn"

# %% [markdown]
# ## 4.3 E1 to E3: baseline, hard-label policies and beyond (T1, T2, T3, T6, T7)
#
# **E1 (baseline).** How well does U-Zeros do under the common protocol? Its rows in T1 and T2.
#
# **E2 (hard-label policies).** Do U-Ones, U-Ignore and U-MultiClass beat U-Zeros? T1 gives the
# mean over the five pathologies of AUROC, AUPRC, F1, ECE and Brier with 95% patient-level
# bootstrap intervals, T2 the per-pathology AUROC and AUPRC, and T3 the paired AUROC
# difference against U-Zeros with its interval and paired-bootstrap p-value. The five p-values
# in each block are **not** corrected for multiple comparisons; read them together with the
# intervals. The test set holds only about 200 studies, so intervals are wide and small
# differences are not significant. Figures: ROC overlays and a forest plot of the differences.
#
# **E3 (beyond hard labels).** Do label smoothing (U-Ones+LSR) and MC Dropout help? They are
# rows of the same tables. T6 separates the effect of *training with dropout* from *MC
# inference*; T7 compares the two U-MultiClass scores.


# %%
def show_table(title: str, table: pd.DataFrame) -> None:
    """Print ``table`` under a title."""
    print(f"\n{title}\n{table.to_string(index=False)}")


def overall_table() -> pd.DataFrame:
    """T1: the five metrics averaged over pathologies, per policy, with 95% intervals."""
    rows = []
    for name in MAIN_POLICIES:
        a = agg(name)
        rows.append({"Policy": POLICY_LABELS[name]} | {m: fmt(summary(a, m)) for m in METRICS})
    return pd.DataFrame(rows)


def by_pathology_table(metric: str) -> pd.DataFrame:
    """T2: ``metric`` per pathology and on average, per policy, with 95% intervals."""
    rows = []
    for name in MAIN_POLICIES:
        a = agg(name)
        cells = {PATHOLOGIES[j]: fmt(summary(a, metric, j)) for j in range(N_PATH)}
        rows.append({"Policy": POLICY_LABELS[name]} | cells | {"Mean": fmt(summary(a, metric))})
    return pd.DataFrame(rows)


def pathology_columns() -> list[tuple[str, int | None]]:
    """``[("Mean", None), ("Atelectasis", 0), ...]`` for tables and panels."""
    return [("Mean", None)] + [(PATHOLOGIES[j], j) for j in range(N_PATH)]


def paired_table(metric: str = "AUROC") -> pd.DataFrame:
    """T3: ``metric`` difference to U-Zeros, per policy and pathology, with interval and p."""
    base, rows = agg("u_zeros"), []
    for name in MAIN_POLICIES[1:]:
        for label, j in pathology_columns():
            d = paired(agg(name), base, metric, j)
            row = {"Policy": POLICY_LABELS[name], "Pathology": label}
            rows.append(row | {f"{metric} difference": fmt_diff(d), "p": fmt_p(d["p"])})
    return pd.DataFrame(rows)


def mc_table() -> pd.DataFrame:
    """T6: the MC Dropout policy with dropout off and with MC inference, against U-Zeros."""
    base, off, mc = agg("u_zeros"), agg("u_zeros_mc", "off"), agg("u_zeros_mc")
    rows = []
    for label, a in [
        ("U-Zeros (trained without dropout)", base),
        ("Trained with dropout, dropout off", off),
        ("Trained with dropout, MC mean (20 passes)", mc),
    ]:
        row = {"Setting": label} | {
            m: fmt(summary(a, m)) for m in ["AUROC", "AUPRC", "ECE", "Brier"]
        }
        vs_base = paired(a, base, "AUROC") if a is not base else None
        vs_off = paired(a, off, "AUROC") if a is mc else None
        row["AUROC - U-Zeros"] = fmt_diff(vs_base) if vs_base else "-"
        row["p (U-Zeros)"] = fmt_p(vs_base["p"]) if vs_base else "-"
        row["AUROC - dropout off"] = fmt_diff(vs_off) if vs_off else "-"
        row["p (dropout off)"] = fmt_p(vs_off["p"]) if vs_off else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def multiclass_table() -> pd.DataFrame:
    """T7: U-MultiClass scored by softmax P(pos) and by P(pos) / (P(pos) + P(neg))."""
    primary, renorm = agg("u_multiclass"), agg("u_multiclass", "renorm")
    rows = []
    for label, a in [
        ("P(positive)  (used in T1-T3)", primary),
        ("P(pos) / (P(pos) + P(neg))", renorm),
    ]:
        row = {"Score": label} | {m: fmt(summary(a, m)) for m in ["AUROC", "AUPRC"]}
        d = paired(a, primary, "AUROC") if a is renorm else None
        row["AUROC - P(positive)"] = fmt_diff(d) if d else "-"
        row["p"] = fmt_p(d["p"]) if d else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_roc_overlay() -> None:
    """ROC curves of the six policies for each pathology (first analysed seed, test set)."""
    seed = RESULT_SEEDS[0]
    fig, axes = plt.subplots(1, N_PATH, figsize=(PAGE_WIDTH_IN, 2.0), sharex=True, sharey=True)
    for j, ax in enumerate(axes):
        for name in MAIN_POLICIES:
            fpr, tpr = roc_points(Y_TEST[:, j], PREDS[(name, seed)].test[:, j])
            ax.plot(fpr, tpr, color=POLICY_COLORS[name], lw=0.9)
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.6, ls=":")
        ax.set_title(PATHOLOGIES[j])
        ax.set_xlabel("False positive rate")
        ax.set_aspect("equal")
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout()
    policy_legend(fig, MAIN_POLICIES, y=-0.06)
    save_figure(fig, "E2_roc_overlay")
    show_figure(fig)


def plot_forest() -> None:
    """AUROC difference to U-Zeros with 95% intervals: mean and each pathology."""
    base, others = agg("u_zeros"), MAIN_POLICIES[1:]
    fig, axes = plt.subplots(1, N_PATH + 1, figsize=(PAGE_WIDTH_IN, 2.2), sharey=True)
    for ax, (title, j) in zip(axes, pathology_columns(), strict=True):
        for k, name in enumerate(others):
            d = paired(agg(name), base, "AUROC", j)
            err = [[max(d["diff"] - d["lo"], 0)], [max(d["hi"] - d["diff"], 0)]]
            ax.errorbar(d["diff"], -k, xerr=err, fmt=POLICY_MARKERS[name], capsize=1.5, lw=0.8,
                        color=POLICY_COLORS[name])  # fmt: skip
        ax.axvline(0, color="0.5", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("AUROC difference")
    axes[0].set_yticks([-k for k in range(len(others))], [POLICY_LABELS[n] for n in others])
    fig.tight_layout()
    save_figure(fig, "E2_forest_vs_u_zeros")
    show_figure(fig)


if MODE == "results":
    T1 = overall_table()
    save_table(T1, "T1_overall")
    show_table("T1: mean over the five pathologies (95% bootstrap interval; ± SD over seeds)", T1)
    T2 = by_pathology_table("AUROC")
    save_table(T2, "T2_auroc_by_pathology")
    show_table("T2: AUROC per pathology", T2)
    T2B = by_pathology_table("AUPRC")
    save_table(T2B, "T2b_auprc_by_pathology")
    show_table("T2b: AUPRC per pathology", T2B)
    T3 = paired_table()
    save_table(T3, "T3_auroc_vs_u_zeros")
    show_table("T3: AUROC difference to U-Zeros (paired bootstrap, uncorrected p)", T3)
    plot_roc_overlay()
    plot_forest()
    T6 = mc_table()
    save_table(T6, "T6_mc_dropout_ablation")
    show_table("T6: MC Dropout, dropout off versus MC inference", T6)
    T7 = multiclass_table()
    save_table(T7, "T7_multiclass_scores")
    show_table("T7: U-MultiClass scores", T7)

# %% [markdown]
# ## 4.4 E4: generalisation across uncertain prevalence
#
# **Purpose.** Which policy gains most where uncertain labels are common? For every pathology the
# table puts the uncertain prevalence in the analysis training split next to each policy's AUROC
# difference to U-Zeros (± SD across seeds when there are several), and ranks the six policies.
# With five pathologies this is descriptive only: no correlation test is run.


# %%
def objective5_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uncertain prevalence next to each policy's AUROC gain over U-Zeros, and the rank table."""
    base, rows, ranks = agg("u_zeros"), [], []
    order = UNCERTAIN_PREVALENCE[PATHOLOGIES].sort_values(ascending=False).index
    for path in order:
        j = PATHOLOGIES.index(path)
        row = {"Pathology": path, "Uncertain (%)": f"{UNCERTAIN_PREVALENCE[path]:.1f}"}
        auroc_by_policy = {name: summary(agg(name), "AUROC", j)["point"] for name in MAIN_POLICIES}
        for name in MAIN_POLICIES[1:]:
            d = paired(agg(name), base, "AUROC", j)
            row[POLICY_LABELS[name]] = f"{d['diff']:+.3f}" + (
                f" ± {d['sd']:.3f}" if d["sd"] else ""
            )
        best = max(auroc_by_policy, key=auroc_by_policy.get)
        rows.append(row | {"Best policy": POLICY_LABELS[best]})
        ordered = sorted(MAIN_POLICIES, key=lambda n: -auroc_by_policy[n])
        ranks.append(
            {"Pathology": path, "Uncertain (%)": f"{UNCERTAIN_PREVALENCE[path]:.1f}"}
            | {POLICY_LABELS[n]: ordered.index(n) + 1 for n in MAIN_POLICIES}
        )
    return pd.DataFrame(rows), pd.DataFrame(ranks)


def plot_objective5() -> None:
    """AUROC gain over U-Zeros against the uncertain prevalence of each pathology."""
    base = agg("u_zeros")
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    x = UNCERTAIN_PREVALENCE[PATHOLOGIES].to_numpy()
    for name in MAIN_POLICIES[1:]:
        diffs = [paired(agg(name), base, "AUROC", j) for j in range(N_PATH)]
        y = np.array([d["diff"] for d in diffs])
        sd = [d["sd"] for d in diffs]
        order = np.argsort(x)
        ax.plot(x[order], y[order], color=POLICY_COLORS[name], lw=0.6, alpha=0.6)
        ax.errorbar(
            x, y, yerr=None if sd[0] is None else sd, fmt=POLICY_MARKERS[name], capsize=1.5,
            color=POLICY_COLORS[name], label=POLICY_LABELS[name], lw=0.8,
        )  # fmt: skip
    ax.axhline(0, color="0.5", lw=0.6)
    top = ax.get_ylim()[1]
    for j, path in enumerate(PATHOLOGIES):
        ax.annotate(path, (x[j], top), rotation=90, fontsize=5, va="top", ha="right", color="0.3")
    ax.set_xlabel("Uncertain labels in the training split (%)")
    ax.set_ylabel("AUROC difference to U-Zeros")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "E4_gain_vs_uncertain_prevalence")
    show_figure(fig)


if MODE == "results":
    T_OBJ5, T_OBJ5_RANK = objective5_tables()
    save_table(T_OBJ5, "T_objective5_gain_vs_prevalence")
    show_table(
        "E4: AUROC difference to U-Zeros by uncertain prevalence (rows: most to least)", T_OBJ5
    )
    save_table(T_OBJ5_RANK, "T_objective5_rank")
    show_table("E4: rank of each policy per pathology (1 = highest AUROC)", T_OBJ5_RANK)
    plot_objective5()

# %% [markdown]
# ## 4.5 E5: calibration (T4, reliability diagrams)
#
# **Purpose.** Which policy is best calibrated? T4 gives the Expected Calibration Error (10
# equal-width bins) and the Brier score, averaged over pathologies and per pathology. The
# reliability diagrams pool the five pathologies for one seed: a calibrated model lies on the
# diagonal; marker area grows with the number of predictions in the bin.


# %%
def reliability_bins(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(3, bins) mean predicted probability, observed positive rate and count per non-empty bin."""
    edges = np.linspace(0, 1, CONFIG["ece_bins"] + 1)[1:-1]
    bins = np.digitize(p, edges, right=True)
    cells = [(p[bins == b].mean(), y[bins == b].mean(), (bins == b).sum()) for b in np.unique(bins)]
    return np.array(cells).T


def plot_reliability() -> None:
    """Reliability diagram of each policy (first analysed seed, test set, pathologies pooled)."""
    seed = RESULT_SEEDS[0]
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH_IN, 4.3), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, MAIN_POLICIES, strict=True):
        y, p = Y_TEST.ravel(), PREDS[(name, seed)].test.ravel()
        mean_p, rate, count = reliability_bins(y, p)
        assert abs(np.sum(count * np.abs(rate - mean_p)) / len(y) - ece(y, p)) < 1e-9
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls=":")
        ax.plot(mean_p, rate, color=POLICY_COLORS[name], lw=0.8)
        ax.scatter(mean_p, rate, s=8 + 60 * count / count.max(), color=POLICY_COLORS[name],
                   marker=POLICY_MARKERS[name], zorder=3)  # fmt: skip
        ax.set_title(f"{POLICY_LABELS[name]} (ECE {ece(y, p):.3f})")
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("Mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed positive rate")
    fig.tight_layout()
    save_figure(fig, "E5_reliability")
    show_figure(fig)


if MODE == "results":
    T4 = T1[["Policy", "ECE", "Brier"]]
    save_table(T4, "T4_calibration")
    show_table("T4: calibration (mean over pathologies, 95% bootstrap interval)", T4)
    T4B = by_pathology_table("ECE")
    save_table(T4B, "T4b_ece_by_pathology")
    show_table("T4b: ECE per pathology", T4B)
    plot_reliability()

# %% [markdown]
# ## 4.6 E6: does MC Dropout entropy reflect the doubt in the reports?
#
# **Purpose.** Is the predictive entropy of the MC Dropout model higher on images whose training
# label was *uncertain*? The test set has no uncertain labels, so this uses the internal
# validation images (which keep the original -1): per pathology the images are grouped as
# negative (including *not mentioned*), positive and uncertain. The table gives the mean
# entropy per group, the gap *uncertain minus certain* and the AUROC of entropy for telling
# uncertain from certain images, all with 95% patient-level bootstrap intervals. The figure
# and intervals use the first analysed seed; the gap of every seed is listed separately.

# %%
ENTROPY_COLUMNS = ["negative", "positive", "uncertain", "gap", "auroc"]


def entropy_stats(entropy: np.ndarray, idx=slice(None)) -> np.ndarray:
    """(5 pathologies, 5): mean entropy of three label groups, the gap and the entropy AUROC."""
    out = np.full((N_PATH, len(ENTROPY_COLUMNS)), np.nan)
    for j in range(N_PATH):
        e, lab = entropy[idx, j], Y_VAL[idx, j]
        unc = lab == -1
        groups = [e[lab == 0], e[lab == 1], e[unc]]
        means = [g.mean() if len(g) else np.nan for g in groups]
        if unc.any() and (~unc).any():
            out[j] = [*means, means[2] - e[~unc].mean(), auroc(unc.astype(int), e)]
        else:
            out[j, :3] = means
    return out


def entropy_table(entropy: np.ndarray, resamples: list[np.ndarray]) -> pd.DataFrame:
    """E6 table with 95% bootstrap intervals for one seed's validation entropy."""
    point = entropy_stats(entropy)
    values = np.stack([entropy_stats(entropy, idx) for idx in resamples])
    lo, hi = np.nanpercentile(values, [2.5, 97.5], axis=0)
    rows = []
    for j, path in enumerate(PATHOLOGIES):
        counts = [(Y_VAL[:, j] == g).sum() for g in (0, 1, -1)]
        row = {"Pathology": path, "n (neg/pos/unc)": "/".join(str(c) for c in counts)}
        for k, column in enumerate(ENTROPY_COLUMNS):
            row[column] = f"{point[j, k]:.3f} [{lo[j, k]:.3f}, {hi[j, k]:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows).rename(
        columns={"gap": "gap (unc - certain)", "auroc": "entropy AUROC"}
    )


def plot_entropy(entropy: np.ndarray) -> None:
    """Entropy of the three label groups for each pathology."""
    colors = ["#2a78d6", "#eb6834", "#6250d6"]
    fig, axes = plt.subplots(1, N_PATH, figsize=(PAGE_WIDTH_IN, 2.0), sharey=True)
    for j, ax in enumerate(axes):
        data = [entropy[Y_VAL[:, j] == g, j] for g in (0, 1, -1)]
        box = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                         medianprops={"color": "black", "lw": 0.8})  # fmt: skip
        for patch, color in zip(box["boxes"], colors, strict=True):
            patch.set(facecolor=color, alpha=0.7, lw=0.5)
        ax.set_xticks([1, 2, 3], ["neg", "pos", "unc"])
        ax.set_title(PATHOLOGIES[j])
    axes[0].set_ylabel("Predictive entropy (nats)")
    fig.tight_layout()
    save_figure(fig, "E6_entropy_by_label_group")
    show_figure(fig)


if MODE == "results":
    mc_entropy = {s: PREDS[("u_zeros_mc", s)].mc_val["entropy"] for s in RESULT_SEEDS}
    val_boot = bootstrap_indices(VAL_ROWS["patient_id"].to_numpy(), n_boot, BOOTSTRAP_SEED)
    T_ENTROPY = entropy_table(mc_entropy[RESULT_SEEDS[0]], val_boot)
    save_table(T_ENTROPY, "T_entropy_by_label_group")
    show_table(f"E6: MC Dropout entropy on validation images (seed {RESULT_SEEDS[0]})", T_ENTROPY)
    gaps = {f"gap seed {s}": entropy_stats(e)[:, 3] for s, e in mc_entropy.items()}
    T_ENTROPY_SEEDS = pd.DataFrame(
        {"Pathology": PATHOLOGIES} | {k: v.round(3) for k, v in gaps.items()}
    )
    save_table(T_ENTROPY_SEEDS, "T_entropy_gap_by_seed", float_format="%.3f")
    show_table("E6: entropy gap (uncertain minus certain) in every seed", T_ENTROPY_SEEDS)
    plot_entropy(mc_entropy[RESULT_SEEDS[0]])

# %% [markdown]
# ## 4.7 E7 and E8: computational cost and class weighting (T5, T10)
#
# **E7 purpose.** What does each policy cost? T5 lists trainable parameters, hardware, training
# time (total and per epoch), training throughput, inference time per image, peak GPU memory,
# GFLOPs per image and mean GPU utilisation, from the numbers each run recorded (mean ± SD over
# seeds when there are several). T5b compares the cost of MC inference by re-running only
# dropout and the linear head on cached backbone features with 20 full forward passes.
#
# **E8 purpose.** Does positive-class weighting help the baseline (the extra experiment promised
# in the proposal)? T10 compares `u_zeros_posw` with `u_zeros` at seed 42 only, with one paired
# difference, so no multiplicity question arises. It is skipped when that run is absent.


# %%
def run_costs(run: Run) -> dict:
    """Cost figures of one run from its ``run_info.json``, epoch log and GPU trace."""
    info = run.info
    epochs = pd.read_csv(run.dir / "metrics_epoch.csv")
    trace_file = run.dir / "gpu_trace.csv"
    util = np.nan
    if trace_file.is_file():
        util = pd.read_csv(trace_file, skipinitialspace=True)["utilization.gpu"].mean()
    return {
        "params_m": info["trainable_params"] / 1e6,
        "train_h": info["train_seconds"] / 3600,
        "min_per_epoch": epochs["train_seconds"].mean() / 60,
        "img_per_s": info["images_per_sec"],
        "infer_ms": info["inference_ms_per_image"],
        "peak_gb": info["peak_mem_gb"],
        "gflops": info["gflops_per_image"],
        "gpu_util": util,
        "mc_head_ms": info.get("mc_head_ms_per_image", np.nan),
    }


def mean_sd(values: list[float], digits: int = 2) -> str:
    """``mean`` or ``mean ± sd`` of ``values``; ``-`` when there is nothing to show."""
    arr = np.array(values, dtype=float)
    if np.isnan(arr).all():
        return "-"
    text = f"{np.nanmean(arr):.{digits}f}"
    return text + (f" ± {np.nanstd(arr, ddof=1):.{digits}f}" if len(arr) > 1 else "")


def cost_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """T5 (cost per policy) and T5b (cost of MC inference)."""
    rows, costs_by_policy = [], {}
    for name in MAIN_POLICIES:
        runs = [RUNS[(name, s)] for s in RESULT_SEEDS]
        costs = [run_costs(r) for r in runs]
        costs_by_policy[name] = costs
        col = lambda key, costs=costs: [c[key] for c in costs]  # noqa: E731
        rows.append(
            {
                "Policy": POLICY_LABELS[name],
                "Params (M)": mean_sd(col("params_m")),
                "GPU": ", ".join(sorted({r.info["gpu_name"] for r in runs})),
                "Train (h)": mean_sd(col("train_h")),
                "Min/epoch": mean_sd(col("min_per_epoch")),
                "Train img/s": mean_sd(col("img_per_s"), 0),
                "Infer ms/img": mean_sd(col("infer_ms")),
                "Peak mem (GB)": mean_sd(col("peak_gb")),
                "GFLOPs/img": mean_sd(col("gflops")),
                "GPU util (%)": mean_sd(col("gpu_util"), 0),
            }
        )
    mc_costs, passes = costs_by_policy["u_zeros_mc"], CONFIG["mc_passes"]
    infer = [c["infer_ms"] for c in mc_costs]
    head = [c["mc_head_ms"] for c in mc_costs]
    cached = [i + h for i, h in zip(infer, head, strict=True)]
    naive = [passes * i for i in infer]
    mc_row = {
        "Single pass (ms/img)": mean_sd(infer, 3),
        f"Dropout + head x{passes} (ms/img)": mean_sd(head, 3),
        "MC, cached backbone (ms/img)": mean_sd(cached, 3),
        f"MC, {passes} full passes (ms/img)": mean_sd(naive, 3),
        "Speed-up": mean_sd([n / c for n, c in zip(naive, cached, strict=True)], 1),
    }
    return pd.DataFrame(rows), pd.DataFrame([mc_row])


def class_weighting_table() -> pd.DataFrame:
    """T10: u_zeros_posw next to u_zeros at seed 42, with the paired AUROC difference."""
    base, weighted = agg("u_zeros", seeds=[42]), agg(EXTRA_POLICY, seeds=[42])
    d = paired(weighted, base, "AUROC")
    rows = []
    for name, a in [("u_zeros", base), (EXTRA_POLICY, weighted)]:
        row = {"Policy": POLICY_LABELS[name]} | {m: fmt(summary(a, m)) for m in METRICS}
        rows.append(row | {"AUROC - U-Zeros": "-", "p": "-"})
    rows[1]["AUROC - U-Zeros"], rows[1]["p"] = fmt_diff(d), fmt_p(d["p"])
    return pd.DataFrame(rows)


if MODE == "results":
    T5, T5B = cost_tables()
    save_table(T5, "T5_cost")
    show_table("T5: computational cost (mean ± SD over seeds)", T5)
    save_table(T5B, "T5b_mc_inference_cost")
    show_table("T5b: cost of MC Dropout inference", T5B)
    if EXTRA_RUN is not None:
        T10 = class_weighting_table()
        save_table(T10, "T10_class_weighting")
        show_table("T10: positive-class weights on the baseline (seed 42)", T10)
    else:
        print("E8 skipped: the run u_zeros_posw_s42 is not available")

# %% [markdown]
# ## 4.8 Seed variability and checkpoint selection (T8, T9)
#
# **T8** (only with several seeds): per policy the mean AUROC of every seed, its mean, SD and
# range, and whether the ranking of the six policies is the same in every seed. **T9**: the
# proposal selects the checkpoint with the best *policy-resolved* validation AUROC, which is a
# different target for each policy. T9 repeats the comparison with the checkpoint chosen by the
# policy-independent *certain-only* validation AUROC (dropout off for the MC policy) and shows
# whether any ranking changes. Headline tables keep the proposal's rule.


# %%
def seed_table() -> tuple[pd.DataFrame, bool]:
    """T8: mean test AUROC of each policy per seed with spread, and whether rankings agree."""
    per_seed = {
        name: [pick(boot_values((name, s, "main"))[0], "AUROC", None) for s in RESULT_SEEDS]
        for name in MAIN_POLICIES
    }
    rows = []
    for name, values in per_seed.items():
        row = {"Policy": POLICY_LABELS[name]}
        row |= {f"seed {s}": f"{v:.3f}" for s, v in zip(RESULT_SEEDS, values, strict=True)}
        row |= {"Mean": f"{np.mean(values):.3f}", "SD": f"{np.std(values, ddof=1):.3f}"}
        rows.append(row | {"Range": f"{min(values):.3f} to {max(values):.3f}"})
    rankings = [
        tuple(sorted(MAIN_POLICIES, key=lambda n: -per_seed[n][k]))
        for k in range(len(RESULT_SEEDS))
    ]
    return pd.DataFrame(rows), len(set(rankings)) == 1


def selection_table() -> tuple[pd.DataFrame, bool]:
    """T9: mean AUROC with the primary and the certain-only checkpoint, and the two rankings."""
    primary = {n: summary(agg(n, "off"), "AUROC")["point"] for n in MAIN_POLICIES}
    certain = {n: summary(agg(n, "cert"), "AUROC")["point"] for n in MAIN_POLICIES}
    rank = lambda scores: {  # noqa: E731
        n: i + 1 for i, n in enumerate(sorted(scores, key=lambda n: -scores[n]))
    }
    rank_primary, rank_certain = rank(primary), rank(certain)
    rows = []
    for name in MAIN_POLICIES:
        same = sum(
            RUNS[(name, s)].info["best_epoch"] == RUNS[(name, s)].info["best_certain_epoch"]
            for s in RESULT_SEEDS
        )
        rows.append(
            {
                "Policy": POLICY_LABELS[name],
                "Primary checkpoint": f"{primary[name]:.3f}",
                "Certain-only checkpoint": f"{certain[name]:.3f}",
                "Same epoch": f"{same}/{len(RESULT_SEEDS)} runs",
                "Rank (primary)": rank_primary[name],
                "Rank (certain-only)": rank_certain[name],
            }
        )
    return pd.DataFrame(rows), rank_primary == rank_certain


if MODE == "results":
    if len(RESULT_SEEDS) > 1:
        T8, SAME_RANKING = seed_table()
        save_table(T8, "T8_seed_variability")
        show_table("T8: mean test AUROC per seed", T8)
        print(f"ranking of the six policies identical in every seed: {SAME_RANKING}")
    else:
        SAME_RANKING = None
        print("T8 skipped: only one seed is analysed")
    T9, SELECTION_STABLE = selection_table()
    save_table(T9, "T9_checkpoint_selection")
    show_table("T9: checkpoint chosen by policy-resolved versus certain-only validation AUROC", T9)
    print(f"ranking unchanged by the selection rule: {SELECTION_STABLE}")

# %% [markdown]
# ## 4.9 Qualitative evaluation
#
# All figures come from the saved predictions and activation maps of the first analysed seed
# (no GPU needed); images are read from the dataset. The same cases are shown for every policy.
#
# 1. **Example gallery.** Per pathology, two random test images each of true positives, true
#    negatives, false positives and false negatives of the *baseline* (fixed seed, at the
#    baseline's validation threshold), with the probability of every policy in the caption
#    (Z U-Zeros, O U-Ones, I U-Ignore, M U-MultiClass, L U-Ones+LSR, C U-Zeros+MC).
# 2. **Activation maps.** For a true positive, a false negative and a false positive of the
#    baseline, and for one validation image whose label was *uncertain*, the class activation map
#    of each policy (head weights times the last 10 x 10 feature map, scaled to its own
#    minimum and maximum, so it shows *where* the model looks and not how strongly).
# 3. **Behaviour on uncertain-labelled images.** The predicted probability of every policy on
#    the validation entries whose original label is -1: this shows the effect of the labelling
#    choice directly.
# 4. **Confusion matrices** on the test set at each policy's validation Youden threshold.
# 5. **Failure cases.** The most confident wrong predictions of the baseline and of the best
#    policy on the test set.

# %%
POLICY_INITIALS = dict(zip(MAIN_POLICIES, "ZOIMLC", strict=True))
INITIALS_KEY = " · ".join(f"{POLICY_INITIALS[n]} {POLICY_LABELS[n]}" for n in MAIN_POLICIES)


def load_image(path: str) -> np.ndarray:
    """A dataset image as the model sees it: grayscale, resized to the input size."""
    return read_image(DATA_ROOT / path, CONFIG["image_size"])


def score_caption(scores: dict[str, float]) -> str:
    """``Z 0.71 O 0.80 I 0.65`` over two lines for the six policies' probabilities."""
    parts = [f"{POLICY_INITIALS[n]} {scores[n]:.2f}" for n in MAIN_POLICIES]
    return " ".join(parts[:3]) + "\n" + " ".join(parts[3:])


def pick_cases(j: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Two random test rows each of TP, TN, FP and FN of the baseline for pathology ``j``."""
    base = PREDS[("u_zeros", QSEED)]
    positive = Y_TEST[:, j] == 1
    predicted = base.test[:, j] >= youden_thresholds("u_zeros", base.val)[j]
    masks = {"TP": positive & predicted, "TN": ~positive & ~predicted,
             "FP": ~positive & predicted, "FN": positive & ~predicted}  # fmt: skip
    return {
        k: rng.choice(np.flatnonzero(m), size=min(2, int(m.sum())), replace=False)
        for k, m in masks.items()
    }


def plot_gallery(j: int, cases: dict[str, np.ndarray]) -> None:
    """2 x 4 grid of the selected cases of pathology ``j`` with all policies' probabilities."""
    fig, axes = plt.subplots(2, 4, figsize=(PAGE_WIDTH_IN, 3.9))
    slots = [(c, k) for c in ("TP", "TN", "FP", "FN") for k in (0, 1)]
    for ax, (category, k) in zip(
        axes.T.flat, slots, strict=True
    ):  # TP, TN in row 0 / FP, FN in row 1
        ax.axis("off")
        if k >= len(cases[category]):
            continue
        i = cases[category][k]
        ax.imshow(load_image(TEST["path"].iloc[i]), cmap="gray")
        scores = {n: PREDS[(n, QSEED)].test[i, j] for n in MAIN_POLICIES}
        ax.set_title(f"{category}\n{score_caption(scores)}", fontsize=5.5)
    fig.suptitle(f"{PATHOLOGIES[j]}: cases of the baseline.  {INITIALS_KEY}", fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5)
    save_figure(fig, f"Q1_gallery_{PATHOLOGIES[j].replace(' ', '_')}")
    show_figure(fig)


def get_cams(name: str, split: str) -> tuple[dict[str, int], np.ndarray]:
    """Row index by image path and the maps (N, 5, 10, 10) of a run's ``cam_<split>.npz``."""
    if (name, split) not in CAMS:
        data = load_arrays(RUNS[(name, QSEED)].dir, f"cam_{split}.npz")
        CAMS[(name, split)] = ({p: i for i, p in enumerate(data["path"])}, data["cam"])
    return CAMS[(name, split)]


def cam_overlay(ax: plt.Axes, image: np.ndarray, cam: np.ndarray) -> None:
    """Draw ``image`` with its 10 x 10 activation map, upsampled and scaled to [0, 1]."""
    size = image.shape[::-1]
    heat = cv2.resize(cam.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    ax.imshow(image, cmap="gray")
    ax.imshow(heat, cmap="magma", alpha=0.5)
    ax.axis("off")


def plot_cam_comparison(j: int, cases: dict[str, np.ndarray]) -> None:
    """Rows: baseline TP, FN, FP and an uncertain validation image; columns: image + 6 policies."""
    rows = [
        (c, "test", TEST["path"].iloc[cases[c][0]], f"label {int(Y_TEST[cases[c][0], j])}")
        for c in ("TP", "FN", "FP")
        if len(cases[c])
    ]
    val_index, _ = get_cams("u_zeros", "val_subset")
    uncertain = [p for p in val_index if VAL_LABELS.loc[p, PATHOLOGIES[j]] == -1]
    if uncertain:
        rows.append(("uncertain", "val_subset", uncertain[0], "label -1"))
    fig, axes = plt.subplots(
        len(rows), 7, figsize=(PAGE_WIDTH_IN, 1.05 * len(rows) + 0.5), squeeze=False
    )
    for r, (kind, split, path, note) in enumerate(rows):
        image = load_image(path)
        axes[r, 0].imshow(image, cmap="gray")
        axes[r, 0].axis("off")
        axes[r, 0].text(-0.04, 0.5, f"{kind}\n{note}", transform=axes[r, 0].transAxes,
                        rotation=90, ha="right", va="center", fontsize=5.5)  # fmt: skip
        for c, name in enumerate(MAIN_POLICIES, start=1):
            index, maps = get_cams(name, split)
            cam_overlay(axes[r, c], image, maps[index[path], j])
    axes[0, 0].set_title("image", fontsize=6)
    for c, name in enumerate(MAIN_POLICIES, start=1):
        axes[0, c].set_title(POLICY_LABELS[name], fontsize=6)
    fig.suptitle(f"{PATHOLOGIES[j]}: class activation maps", fontsize=6)
    fig.tight_layout()
    save_figure(fig, f"Q2_cam_{PATHOLOGIES[j].replace(' ', '_')}")
    show_figure(fig)


def uncertain_probabilities() -> tuple[pd.DataFrame, dict]:
    """Table and data of each policy's probability on validation entries labelled uncertain."""
    data = {
        name: [PREDS[(name, QSEED)].val[Y_VAL[:, j] == -1, j] for j in range(N_PATH)]
        for name in MAIN_POLICIES
    }
    rows = []
    for j, path in enumerate(PATHOLOGIES):
        row = {"Pathology": path, "n": len(data["u_zeros"][j])}
        row |= {
            POLICY_LABELS[n]: f"{data[n][j].mean():.3f}" if len(data[n][j]) else "-"
            for n in MAIN_POLICIES
        }
        rows.append(row)
    return pd.DataFrame(rows), data


def plot_uncertain_probabilities(data: dict) -> None:
    """Box plots of the probability on uncertain-labelled validation entries, per pathology."""
    fig, axes = plt.subplots(1, N_PATH, figsize=(PAGE_WIDTH_IN, 2.1), sharey=True)
    for j, ax in enumerate(axes):
        values = [data[n][j] for n in MAIN_POLICIES]
        if all(len(v) for v in values):
            box = ax.boxplot(values, widths=0.6, showfliers=False, patch_artist=True,
                             medianprops={"color": "black", "lw": 0.8})  # fmt: skip
            for patch, name in zip(box["boxes"], MAIN_POLICIES, strict=True):
                patch.set(facecolor=POLICY_COLORS[name], alpha=0.8, lw=0.5)
        ax.set_xticks(range(1, 7), [POLICY_INITIALS[n] for n in MAIN_POLICIES])
        ax.set_title(f"{PATHOLOGIES[j]} (n={len(values[0])})")
    axes[0].set_ylabel("Predicted probability")
    fig.tight_layout()
    policy_legend(fig, MAIN_POLICIES, y=-0.06)
    save_figure(fig, "Q3_probability_on_uncertain_labels")
    show_figure(fig)


def confusion_counts(name: str, j: int) -> np.ndarray:
    """[[TN, FP], [FN, TP]] on the test set at the policy's validation Youden threshold."""
    preds = PREDS[(name, QSEED)]
    predicted = preds.test[:, j] >= youden_thresholds(name, preds.val)[j]
    y = Y_TEST[:, j] == 1
    return np.array([[(~y & ~predicted).sum(), (~y & predicted).sum()],
                     [(y & ~predicted).sum(), (y & predicted).sum()]])  # fmt: skip


def confusion_table() -> pd.DataFrame:
    """Long table of confusion counts per policy and pathology."""
    rows = []
    for name in MAIN_POLICIES:
        for j, path in enumerate(PATHOLOGIES):
            (tn, fp), (fn, tp) = confusion_counts(name, j)
            rows.append(
                {
                    "Policy": POLICY_LABELS[name],
                    "Pathology": path,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                    "TP": tp,
                }
            )
    return pd.DataFrame(rows)


def plot_confusion() -> None:
    """Grid of 2 x 2 confusion matrices: pathologies in rows, policies in columns."""
    fig, axes = plt.subplots(N_PATH, 6, figsize=(PAGE_WIDTH_IN, 6.4))
    for j in range(N_PATH):
        for c, name in enumerate(MAIN_POLICIES):
            ax, counts = axes[j, c], confusion_counts(name, j)
            shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
            ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
            for (r, k), value in np.ndenumerate(counts):
                ax.text(k, r, str(value), ha="center", va="center", fontsize=6,
                        color="white" if shares[r, k] > 0.6 else "black")  # fmt: skip
            ax.set_xticks([0, 1], ["neg", "pos"] if j == N_PATH - 1 else ["", ""])
            ax.set_yticks([0, 1], ["neg", "pos"] if c == 0 else ["", ""])
            ax.tick_params(length=0)
            ax.spines[:].set_visible(False)
            if j == 0:
                ax.set_title(POLICY_LABELS[name], fontsize=6)
        axes[j, 0].set_ylabel(PATHOLOGIES[j].replace(" ", "\n"), fontsize=6)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    note = "Rows: radiologist label. Columns: prediction at the validation Youden threshold."
    fig.text(0.5, 0.0, note, ha="center", fontsize=6)
    save_figure(fig, "Q4_confusion_matrices")
    show_figure(fig)


def confident_errors(name: str, top: int = 5) -> pd.DataFrame:
    """The ``top`` most confident wrong test predictions of a policy over all pathologies."""
    preds = PREDS[(name, QSEED)]
    thresholds = youden_thresholds(name, preds.val)
    predicted = preds.test >= thresholds
    wrong = predicted != (Y_TEST == 1)
    confidence = np.where(wrong, np.where(predicted, preds.test, 1 - preds.test), -1.0)
    rows = []
    for rank, flat in enumerate(np.argsort(-confidence, axis=None)[:top], start=1):
        i, j = np.unravel_index(flat, confidence.shape)
        if confidence[i, j] < 0:
            break
        row = {"Policy": POLICY_LABELS[name], "Rank": rank, "Image": TEST["path"].iloc[i]}
        row |= {"Pathology": PATHOLOGIES[j], "Label": int(Y_TEST[i, j])}
        rows.append(row | {"Score": round(float(preds.test[i, j]), 3)})
    return pd.DataFrame(rows)


def plot_failures(errors: pd.DataFrame) -> None:
    """Images of the confident errors: one row per policy."""
    names = list(dict.fromkeys(errors["Policy"]))
    fig, axes = plt.subplots(
        len(names), 5, figsize=(PAGE_WIDTH_IN, 1.7 * len(names) + 0.3), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for r, policy in enumerate(names):
        for c, row in enumerate(errors[errors["Policy"] == policy].itertuples()):
            axes[r, c].imshow(load_image(row.Image), cmap="gray")
            axes[r, c].set_title(
                f"{row.Pathology}\nlabel {row.Label}, score {row.Score:.2f}", fontsize=5.5
            )
        axes[r, 0].text(-0.04, 0.5, policy, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=6)  # fmt: skip
    fig.suptitle("Most confident wrong test predictions", fontsize=6)
    fig.tight_layout()
    save_figure(fig, "Q5_failure_cases")
    show_figure(fig)


if MODE == "results":
    QSEED = RESULT_SEEDS[0]
    CAMS: dict[tuple[str, str], tuple[dict[str, int], np.ndarray]] = {}
    VAL_LABELS = pd.DataFrame(label_matrix(VAL), index=VAL["path"], columns=PATHOLOGIES)
    case_rng = np.random.default_rng(BOOTSTRAP_SEED)
    for j in range(N_PATH):
        cases = pick_cases(j, case_rng)
        plot_gallery(j, cases)
        plot_cam_comparison(j, cases)
    T_UNC, unc_data = uncertain_probabilities()
    save_table(T_UNC, "Q3_probability_on_uncertain_labels")
    show_table(
        f"Mean predicted probability on validation entries labelled uncertain (seed {QSEED})", T_UNC
    )
    plot_uncertain_probabilities(unc_data)
    T_CONF = confusion_table()
    save_table(T_CONF, "Q4_confusion_counts")
    plot_confusion()
    point = {n: summary(agg(n), "AUROC")["point"] for n in MAIN_POLICIES}
    ranked = sorted(MAIN_POLICIES, key=lambda n: -point[n])
    best_policy = ranked[0] if ranked[0] != "u_zeros" else ranked[1]
    errors = pd.concat(
        [confident_errors("u_zeros"), confident_errors(best_policy)], ignore_index=True
    )
    save_table(errors, "Q5_failure_cases")
    show_table("Most confident wrong test predictions", errors)
    plot_failures(errors)

# %% [markdown]
# ## 4.10 Training record of every run
#
# Each run trained in its own notebook, but only this notebook is submitted, so the training
# record of **all** runs is rebuilt here from the files each run pushed (`metrics_epoch.csv`,
# `train.log`, `grad_stats.csv`, `gpu_trace.csv`, `run_info.json`): the per-epoch table, the
# stopping reason, the beginning and end of the training log and the gradient and GPU summaries
# of every policy (and the extra class-weighting run), then training curves of all policies on
# shared axes. With several seeds the tables show the first seed, the curves show the mean over
# seeds with the minimum-to-maximum band, and the long table `T_training_record` lists every run.

# %%
EPOCH_COLUMNS = ["epoch", "train_loss", "val_auroc", "val_auroc_certain", "lr", "train_seconds",
                 "images_per_sec", "peak_mem_gb"]  # fmt: skip


def load_training(run: Run) -> pd.DataFrame:
    """The per-epoch log of a run with a ``run`` column."""
    frame = pd.read_csv(run.dir / "metrics_epoch.csv")
    frame.insert(0, "run", run.name)
    return frame


def print_training_record(run: Run) -> None:
    """Epoch table, stopping reason, log excerpt and gradient/GPU summaries of one run."""
    info = run.info
    who = f"run by {info['run_by']} on {info['gpu_name']}"
    print(f"\n=== {run.name} ({POLICY_LABELS[run.policy]}), {who} ===")
    print(
        f"best epoch {info['best_epoch']} of {info['epochs_run']} run, "
        f"stopped by {info['stop_reason']}; trained {info['train_seconds'] / 60:.1f} min "
        f"at {info['images_per_sec']:.0f} images/s"
    )
    print(load_training(run)[EPOCH_COLUMNS].to_string(index=False, float_format="%.4f"))
    grads = pd.read_csv(run.dir / "grad_stats.csv")
    if len(grads):
        norm = grads["grad_norm"]
        print(
            f"gradient norm: median {norm.median():.2f}, max {norm.max():.2f}; "
            f"loss scale: min {grads['loss_scale'].min():.0f}"
        )
    trace_file = run.dir / "gpu_trace.csv"
    if trace_file.is_file():
        trace = pd.read_csv(trace_file, skipinitialspace=True)
        print(
            f"GPU: utilisation {trace['utilization.gpu'].mean():.0f}% on average, "
            f"memory up to {trace['memory.used'].max() / 1024:.1f} GiB"
        )
    lines = (run.dir / "train.log").read_text().splitlines()
    print("log:", *lines[:2], "  ...", *lines[-2:], sep="\n  ")


def seed_band(
    frames: list[pd.DataFrame], column: str
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Epochs with the mean, minimum and maximum of ``column`` over the runs that reached them."""
    epochs = sorted(set().union(*[set(f["epoch"]) for f in frames]))
    values = [
        [f.loc[f["epoch"] == e, column].iloc[0] for f in frames if e in set(f["epoch"])]
        for e in epochs
    ]
    return epochs, [np.mean(v) for v in values], [min(v) for v in values], [max(v) for v in values]


def plot_training_curves() -> None:
    """Validation AUROC of all policies on shared axes, and each policy's training loss."""
    logs = {n: [load_training(RUNS[(n, s)]) for s in RESULT_SEEDS] for n in MAIN_POLICIES}
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH_IN, 2.4), sharex=True)
    for ax, column, title in [
        (axes[0], "val_auroc", "Validation mean AUROC (policy-resolved labels)"),
        (axes[1], "val_auroc_certain", "Validation mean AUROC (certain labels only)"),
    ]:
        for name in MAIN_POLICIES:
            epochs, mean, low, high = seed_band(logs[name], column)
            ax.plot(epochs, mean, color=POLICY_COLORS[name], marker=POLICY_MARKERS[name], lw=0.9)
            if len(RESULT_SEEDS) > 1:
                ax.fill_between(epochs, low, high, color=POLICY_COLORS[name], alpha=0.15, lw=0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("AUROC")
    fig.tight_layout()
    policy_legend(fig, MAIN_POLICIES, y=-0.07)
    save_figure(fig, "T_training_curves_val_auroc")
    show_figure(fig)

    fig, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH_IN, 3.6), sharex=True)
    for ax, name in zip(axes.flat, MAIN_POLICIES, strict=True):
        epochs, mean, low, high = seed_band(logs[name], "train_loss")
        ax.plot(epochs, mean, color=POLICY_COLORS[name], marker=POLICY_MARKERS[name], lw=0.9)
        if len(RESULT_SEEDS) > 1:
            ax.fill_between(epochs, low, high, color=POLICY_COLORS[name], alpha=0.15, lw=0)
        ax.set_title(POLICY_LABELS[name])
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Training loss (each policy's own loss)")
    fig.tight_layout()
    save_figure(fig, "T_training_curves_loss")
    show_figure(fig)


if MODE == "results":
    shown_runs = [RUNS[(n, RESULT_SEEDS[0])] for n in MAIN_POLICIES] + (
        [EXTRA_RUN] if EXTRA_RUN else []
    )
    for run in shown_runs:
        print_training_record(run)
    record = pd.concat([load_training(r) for r in USED_RUNS.values()], ignore_index=True)
    save_table(record, "T_training_record", float_format="%.4f")
    print(f"\nT_training_record: {len(record)} epoch rows from {len(USED_RUNS)} runs")
    plot_training_curves()

# %% [markdown]
# ## 4.11 Export index
#
# Every table was written as CSV and LaTeX (`booktabs`) and every figure as vector PDF and a
# 300 dpi PNG at the moment it was made, into `results/tables/` and `results/figures/`. This
# cell checks that every file exists and lists what was produced.


# %%
def export_index() -> pd.DataFrame:
    """One row per exported table and figure; fails if any file is missing."""
    rows = []
    for kind, names, directory, suffixes in [
        ("table", SAVED["tables"], TABLES_DIR, (".csv", ".tex")),
        ("figure", SAVED["figures"], FIGURES_DIR, (".pdf", ".png")),
    ]:
        for name in names:
            files = [directory / f"{name}{suffix}" for suffix in suffixes]
            missing = [f.name for f in files if not f.is_file()]
            assert not missing, f"export incomplete: {missing}"
            kb = sum(f.stat().st_size for f in files) / 1024
            rows.append(
                {"kind": kind, "name": name, "files": ", ".join(suffixes), "kB": f"{kb:.0f}"}
            )
    return pd.DataFrame(rows)


if MODE == "results":
    EXPORT = export_index()
    print(EXPORT.to_string(index=False))
    n_tables, n_figures = len(SAVED["tables"]), len(SAVED["figures"])
    print(f"\n{n_tables} tables and {n_figures} figures written to {REPO_DIR / 'results'}")

# %% [markdown]
# ## 4.12 Findings
#
# Sentences with the numbers generated from the results above, so the paper can quote them
# without retyping. They report what the data show, including intervals that contain zero; the
# test set has only a few hundred images, so most differences between policies are small
# compared with their uncertainty. P-values are paired-bootstrap p-values, **not** corrected for
# the five comparisons per row.


# %%
def excludes_zero(d: dict) -> bool:
    """True if the 95% interval of a paired difference does not contain zero."""
    return d["lo"] > 0 or d["hi"] < 0


def side(d: dict) -> str:
    """``excludes`` or ``includes`` zero for the interval of a paired difference."""
    return "excludes" if excludes_zero(d) else "includes"


def build_findings() -> list[str]:
    """Plain-language findings with the numbers behind them."""
    label, base = POLICY_LABELS, agg("u_zeros")
    out = [
        f"Protocol: {len(RESULT_SEEDS)} seed(s) {RESULT_SEEDS}; test set of {len(TEST)} images "
        f"from {TEST['patient_id'].nunique()} patients; {n_boot} patient-level bootstrap resamples."
    ]
    point = {n: summary(agg(n), "AUROC")["point"] for n in MAIN_POLICIES}
    best = max(point, key=point.get)
    out.append(
        f"Highest mean AUROC: {label[best]} {fmt(summary(agg(best), 'AUROC'))}; "
        f"the U-Zeros baseline has {fmt(summary(base, 'AUROC'))}."
    )
    diffs = {n: paired(agg(n), base, "AUROC") for n in MAIN_POLICIES[1:]}
    for name, d in diffs.items():
        verdict = "below" if d["p"] < 0.05 else "not below"
        out.append(
            f"{label[name]} versus U-Zeros, mean AUROC {fmt_diff(d)}: the interval {side(d)} zero "
            f"(p = {fmt_p(d['p'])}, {verdict} 0.05, uncorrected)."
        )
    big = max(diffs, key=lambda n: diffs[n]["diff"])
    small = min(diffs, key=lambda n: diffs[n]["diff"])
    out.append(
        f"Largest mean AUROC change over U-Zeros: {label[big]} ({diffs[big]['diff']:+.3f}); "
        f"smallest: {label[small]} ({diffs[small]['diff']:+.3f})."
    )
    for j, path in enumerate(PATHOLOGIES):
        by_policy = {n: summary(agg(n), "AUROC", j)["point"] for n in MAIN_POLICIES}
        top = max(by_policy, key=by_policy.get)
        head = f"{path} ({UNCERTAIN_PREVALENCE[path]:.1f}% uncertain): "
        if top == "u_zeros":
            out.append(f"{head}U-Zeros has the highest AUROC ({by_policy[top]:.3f}).")
            continue
        d = paired(agg(top), base, "AUROC", j)
        out.append(
            f"{head}highest AUROC {label[top]} ({by_policy[top]:.3f}), {fmt_diff(d)} versus "
            f"U-Zeros; the interval {side(d)} zero (p = {fmt_p(d['p'])})."
        )
    for metric in ("ECE", "Brier"):
        values = {n: summary(agg(n), metric)["point"] for n in MAIN_POLICIES}
        low = min(values, key=values.get)
        out.append(
            f"Best calibrated by {metric}: {label[low]} ({values[low]:.3f}); "
            f"U-Zeros {values['u_zeros']:.3f}."
        )
    mc = paired(agg("u_zeros_mc"), agg("u_zeros_mc", "off"), "AUROC")
    out.append(
        f"MC inference versus dropout off, mean AUROC {fmt_diff(mc)} (p = {fmt_p(mc['p'])})."
    )
    gaps = entropy_stats(mc_entropy[RESULT_SEEDS[0]])[:, 3]
    out.append(
        f"MC entropy gap (uncertain minus certain, seed {RESULT_SEEDS[0]}) is positive for "
        f"{int((gaps > 0).sum())} of {N_PATH} pathologies "
        f"(range {np.nanmin(gaps):+.3f} to {np.nanmax(gaps):+.3f})."
    )
    if EXTRA_RUN is not None:
        d = paired(agg(EXTRA_POLICY, seeds=[42]), agg("u_zeros", seeds=[42]), "AUROC")
        out.append(
            f"Positive-class weights on the baseline (seed 42): mean AUROC {fmt_diff(d)} "
            f"(p = {fmt_p(d['p'])})."
        )
    if SAME_RANKING is not None:
        same = "the same" if SAME_RANKING else "not the same"
        out.append(f"The ranking of the six policies is {same} in every seed.")
    changes = "does not change" if SELECTION_STABLE else "changes"
    out.append(f"Choosing the checkpoint by certain-only validation AUROC {changes} the ranking.")
    return out


if MODE == "results":
    FINDINGS = build_findings()
    for line in FINDINGS:
        print(f"- {line}")
    (REPO_DIR / "results" / "findings.txt").write_text("\n".join(FINDINGS) + "\n")
