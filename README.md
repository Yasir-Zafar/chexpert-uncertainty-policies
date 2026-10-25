# Uncertainty label resolution for multi-label chest X-ray classification

A controlled comparison of six ways to turn CheXpert's *uncertain* label (-1) into a training
target. One backbone (ImageNet DenseNet-121, all layers fine-tuned), one patient-level split, one
training recipe and one metric set; only the uncertainty policy changes between runs.

| Policy (`POLICY` value) | What happens to the uncertain label |
|---|---|
| `u_zeros` (baseline) | becomes 0 |
| `u_ones` | becomes 1 |
| `u_ignore` | left out of the loss |
| `u_multiclass` | kept as a third class |
| `u_ones_lsr` | becomes a soft target drawn from [0.55, 0.85] |
| `u_zeros_mc` | becomes 0; uncertainty estimated with MC Dropout (p = 0.3, 20 passes) |
| `u_zeros_posw` | extra experiment: baseline with positive-class weights (not one of the six) |

The deliverable is **one notebook**, `chexpert_uncertainty.ipynb`, that goes from data
preparation to results. It runs in two modes:

- `train`: trains one policy on a Kaggle GPU and pushes the outputs to its own GitHub branch.
- `results`: fetches every run from GitHub and builds all tables and figures on a CPU session.
  The executed `results` notebook is what gets submitted.

The design and the build plan are in [docs/agent-plan.md](docs/agent-plan.md); the approved
proposal is in [docs/](docs/).

## Repository layout

```
chexpert_uncertainty.ipynb   generated from notebook_src.py; this is what you import into Kaggle
notebook_src.py              the notebook's source (jupytext "# %%" cells); edit this one
splits/  qual/               patient split and qualitative image lists (created once from the real
                             CSVs and committed before the version is tagged; not added yet)
tools/                       fake_chexpert.py (fake data), make_nb.py, pushtest.py, status.py
Makefile  pyproject.toml     dev tasks and lint settings
flake.nix  .envrc            the dev environment (Nix + direnv)
docs/                        the plan and the proposal
```

Run outputs are not on `main`. Each run lives on its own branch, `run/<policy>_s<seed>`, which
holds only that run's folder `runs/<policy>_s<seed>/` and no code.

## Running one policy on Kaggle

Everyone runs the **same tagged version** of the notebook, on the same dataset and the same GPU
type. The builder announces these before any full run:

| Setting | Value |
|---|---|
| Notebook version (git tag) | to be announced by the builder |
| Kaggle dataset | to be announced by the builder |
| Accelerator | to be announced by the builder (one GPU only; not 2 x T4) |
| Seeds | 42 only, unless the builder announces extra seed layers (43, then 44) |

Who runs what (swap freely, the runs are independent):

| Person | Policies |
|---|---|
| Builder | `u_zeros`, `u_zeros_mc`, and the extra `u_zeros_posw` |
| Teammate 1 | `u_ones`, `u_ignore` |
| Teammate 2 | `u_multiclass`, `u_ones_lsr` |

### One-time setup in your Kaggle account

1. Verify your phone number (needed for GPU and internet) and note your weekly GPU quota.
2. Create a GitHub **fine-grained token** limited to this repository with *Contents: read and
   write*, expiring after your last run.
3. In a Kaggle notebook, open *Add-ons > Secrets* and add three secrets:
   `GITHUB_PAT` (the token), `GITHUB_REPO` (`owner/name` of this repository) and `GITHUB_USER`
   (your GitHub username).

The token lives only in Secrets. Never paste it into a cell, and keep your notebook **private**.

### Each run

1. Download `chexpert_uncertainty.ipynb` from the tagged version and import it into Kaggle.
2. In the settings panel: set the accelerator announced above, turn **Internet on**, attach the
   dataset, and attach the three secrets.
3. In the first code cell (*0.1 Run settings*) change the default of `POLICY` to your policy, for
   example `POLICY = os.environ.get("POLICY", "u_ones")`. Do not edit anything else.
4. **Save Version, then Save and Run All (Commit)**, and close the tab; it runs in the
   background.
5. When it finishes, check on GitHub that the branch `run/<policy>_s42` exists, that
   `runs/<policy>_s42/run_info.json` says `"status": "complete"`, and that no `.pt` file is
   anywhere in the branch. Send the builder one line: date, policy, GPU, wall-clock time and the
   best epoch.

A full run is about 10 epochs (up to a few hours on one GPU). The epoch log is pushed to your
branch after every epoch, so a session that dies still leaves its history behind.

### Rules

1. Smoke pass first, by the builder only: with `SMOKE=1` a run trains on a small subset for two
   epochs and pushes to `smoke/<policy>_s<seed>` (smoke branches are overwritten freely). The
   builder checks the files, that no token appears in any log and that throughput is acceptable.
   Teammates start full runs only after the builder announces it.
2. Start full runs only after the builder's baseline `u_zeros` has finished two epochs and looks
   sane (validation AUROC rising, no NaN).
3. Push only to your own `run/...` branches, never to `main`. A full run refuses to start if its
   branch already exists, which means someone has already started that policy and seed; pick
   another or ask the builder.
4. If a run fails or looks wrong, do not edit the notebook and rerun. Tell the builder: a fix
   needs a new version tag and a rerun of every affected run.

### Watching progress

```sh
python tools/status.py        # one row per run branch: who, epoch, validation AUROC, img/s, status
```

It needs only git, reads the pushed epoch logs, and ignores smoke branches.

## Results and submission

`MODE=results` runs on a Kaggle **CPU** session (no GPU quota) with the dataset attached. It
fetches all runs from their branches, checks they used the same split, data, code version and
settings, and builds the tables, figures and qualitative analysis. Because only the executed
results notebook is submitted, it also shows the training record of every run (per-epoch tables,
log excerpts, curves) rather than leaving it on GitHub. Tables go to `results/tables/` (CSV and
LaTeX) and figures to `results/figures/` (PDF and PNG).

## Local development

The environment is a Nix flake loaded by direnv (`direnv allow`, or `nix develop`): Python 3.13
to match Kaggle, CPU PyTorch, ruff, pre-commit and make. The real dataset is licensed and never
leaves Kaggle; locally everything runs against a small fake dataset.

```sh
make hooks      # once: install the pre-commit hooks (ruff, token and kaggle.json blocker, notebook check)
make check      # lint, hook test, fake data, data-prep reruns, headless run of the notebook
make smoke      # one smoke training per policy on the fake data (about 8 minutes on a CPU)
make pushtest   # the GitHub push and tools/status.py against a local bare repository
make nb         # regenerate chexpert_uncertainty.ipynb after editing notebook_src.py
```

Edit `notebook_src.py`, never the `.ipynb`: the pre-commit hook rejects a commit whose notebook
is out of date, so run `make nb` and commit both files together.

## Data and licence

CheXpert may not be redistributed. The repository contains no images and no labels: the split and
qualitative lists hold patient ids and image paths only, and pushed run files hold image paths,
scores and logits but no labels, which results mode joins back from the dataset's CSVs. Model
checkpoints (`*.pt`) stay in the Kaggle notebook output and are never pushed.
