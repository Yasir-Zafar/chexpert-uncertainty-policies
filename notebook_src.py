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
import platform
import random
import subprocess
from pathlib import Path

import cv2
import matplotlib
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
