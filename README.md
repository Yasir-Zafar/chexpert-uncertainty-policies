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
splits/  qual/               patient split and qualitative image lists, created once from the real
                             CSVs; Kaggle runs only read them
tools/                       fake_chexpert.py (fake data), make_nb.py, pushtest.py, status.py,
                             resulttest.py, run_until.py (dev checks)
Makefile  pyproject.toml     dev tasks and lint settings
flake.nix  .envrc            the dev environment (Nix + direnv)
docs/                        the plan and the proposal
```

Run outputs are not on `main`. Each run lives on its own branch, `run/<policy>_s<seed>`, which
holds only that run's folder `runs/<policy>_s<seed>/` and no code.

## Running one policy on Kaggle

Everyone runs the **same tagged version** of the notebook, on the same dataset and the same GPU
type. These are the settings:

| Setting | Value |
|---|---|
| Notebook version (git tag) | `v1`, announced by the builder once the checks on Kaggle are done |
| Kaggle dataset | `ashery/chexpert` (the CheXpert-v1.0-small mirror); attach this one only |
| Accelerator | GPU T4 x2 (the notebook uses one of the two GPUs; about 120 images/s) |
| Environment | keep the default; each run records its Python, torch and CUDA versions |
| Seeds | 42 only, unless the builder announces extra seed layers (43, then 44) |
| Time per run | about 4.5 hours for 10 epochs, plus about 5 minutes of cache warm-up at the start |

Who runs what (swap freely, the runs are independent):

| Person | Policies |
|---|---|
| Builder | `u_zeros`, `u_zeros_mc`, and the extra `u_zeros_posw` |
| Teammate 1 | `u_ones`, `u_ignore` |
| Teammate 2 | `u_multiclass`, `u_ones_lsr` |

Two runs per person use about 10 hours of the 30-hour weekly GPU quota, and every extra seed
layer costs the same again.

### One-time setup in your Kaggle account

1. Verify your phone number (needed for GPU and internet) and note your weekly GPU quota.
2. Create a GitHub **fine-grained token**: Repository access *Only select repositories*, this
   repository only; Repository permissions *Contents: Read and write* and nothing else; expiry
   after your last run. With read-only access a run stops at its first push with
   `403 ... Permission ... denied`.
3. In a Kaggle notebook, open *Add-ons > Secrets* and add three secrets, each attached to the
   notebook: `GITHUB_PAT` (the token), `GITHUB_REPO` (`owner/name` of this repository) and
   `GITHUB_USER` (your GitHub username).

The token lives only in Secrets. Never paste it into a cell, and keep your notebook **private**.
Do not link the notebook to GitHub: that would commit notebook versions into the repository.

### Each run

1. Download `chexpert_uncertainty.ipynb` from the tagged version and import it into Kaggle
   (File > Import notebook).
2. In the settings panel: accelerator *GPU T4 x2*, **Internet on**, *Add Input* the dataset above,
   and attach the three secrets.
3. In the first code cell (*0.1 Run settings*) change the default of `POLICY` to your policy, for
   example `POLICY = os.environ.get("POLICY", "u_ones")`. Do not edit anything else.
4. **Save Version, then Save and Run All (Commit)**, and close the tab; it runs in the
   background.
5. When it finishes, check on GitHub that the branch `run/<policy>_s42` exists, that
   `runs/<policy>_s42/run_info.json` says `"status": "complete"`, and that no `.pt` file is
   anywhere in the branch. Send the builder one line: date, policy, GPU, wall-clock time and the
   best epoch.

What the log shows, in order: the versions of Python and the libraries, the repository version
that was cloned, the dataset folder, the row counts (`train.csv: 223,414 rows`), the training and
validation image counts, `cache warm-up` lines (every image is read once so the epochs run from
memory instead of waiting for storage), then about one `epoch N step ... img/s` line per 50 steps
and one `epoch N done` line per epoch. From the second epoch on, `img/s` should be near 120; tell
the builder if it stays below about 100. The epoch log is pushed to your branch after every epoch,
so a session that dies still leaves its history behind.

### Rules

1. Smoke pass first, by the builder only: in cell 0.1 set the `SMOKE` default to `"1"` (and
   `REPO_REF` to `"main"` while there is no tag). A smoke run trains on 2,000 images for two
   epochs and pushes to `smoke/<policy>_s<seed>` (smoke branches are overwritten freely). Its log
   says `(smoke run)` and `2,000 training and 500 validation images`; stop it at once if it says
   `(full run)`. The builder checks the files, that no token appears in any log and that
   throughput is acceptable. Teammates start full runs only after the builder announces it.
2. Start full runs only after the builder's baseline `u_zeros` has finished two epochs and looks
   sane (validation AUROC rising, no NaN).
3. Push only to your own `run/...` branches, never to `main`. A full run refuses to start if its
   branch already exists, which means someone has already started that policy and seed; pick
   another or ask the builder.
4. If a run fails or looks wrong, do not edit the notebook and rerun. Tell the builder: a fix
   needs a new version tag and a rerun of every affected run.

### If it stops

| Message | Cause and fix |
|---|---|
| `Kaggle Secret GITHUB_REPO ... is not set`, or `GITHUB_PAT and GITHUB_REPO are not both set` | the secrets are missing, or not attached to this notebook |
| `git push failed ... 403 ... Permission ... denied` | the token cannot write: set *Contents* to *Read and write* for this repository |
| `Remote branch v1 not found` | the tag does not exist yet (the builder announces it) |
| `splits/split_seed42.csv is missing` | the split files are not in the version being cloned |
| `expected one dataset folder with train.csv and valid.csv` | no dataset attached, or two |
| `train.csv: N rows, official mirror has 223414` | the wrong dataset mirror |
| `... is not empty; remove it to start this run again` | an earlier attempt in this session left files: run `!rm -rf /kaggle/working/repo/runs /kaggle/working/repo/runs_smoke /kaggle/working/repo/checkpoints` and run again |
| `branch run/<policy>_s42 already exists` | someone started that policy and seed; tell the builder, do not delete it yourself |

### Watching progress

```sh
python tools/status.py        # one row per run branch: who, epoch, validation AUROC, img/s, status
```

It needs only git, reads the pushed epoch logs, and ignores smoke branches.

## When all runs are done

Run branches are data, not code: **never merge them**, and ignore GitHub's "Compare & pull
request" banner on them. Each holds one run's folder and nothing else, and `main` does not change
when a run finishes. The results notebook reads the branches directly by name.

1. Check that every expected run is `complete`: `python tools/status.py`, or the run inventory
   that results mode prints first.
2. Import the notebook once more and, in cell 0.1, change only `MODE` to `"results"` (keep
   `SMOKE` at `"0"`: results mode reads the `run/...` branches, not the `smoke/...` ones).
3. Settings: a **CPU** session (no GPU quota), Internet on, the same dataset attached (the image
   galleries need it) and the three secrets attached.
4. Save Version, then Save and Run All (Commit).

Results mode lists every expected run in an inventory, stops with a list if one is missing or
does not match the others (code version, settings, split, data, test images), and otherwise
builds the tables, figures, qualitative analysis and findings. Because only the executed results
notebook is submitted, it also shows the training record of every run (per-epoch tables, log
excerpts, curves) rather than leaving it on GitHub.

What to submit and keep:
- **Submit** the executed results notebook (the code with all outputs), downloaded from the saved
  version.
- **For the paper**, take the tables (CSV and LaTeX) and figures (PDF and PNG) from the
  notebook's output folder, `results/tables/` and `results/figures/`.
- **Keep** the run branches and the `v1` tag as the record of the runs and the code. The
  `results/` folder is ignored by git; add a final copy with `git add -f results` through a pull
  request only if you want it in the repository.

## Local development

The environment is a Nix flake loaded by direnv (`direnv allow`, or `nix develop`): Python 3.13
to match Kaggle, CPU PyTorch, ruff, pre-commit and make. The real dataset is licensed and never
leaves Kaggle; locally everything runs against a small fake dataset.

```sh
make hooks      # once: install the pre-commit hooks (ruff, token and kaggle.json blocker, notebook check)
make check      # lint, hook test, fake data, data-prep reruns, headless run of the notebook
make smoke      # one smoke training per policy on the fake data (about 8 minutes on a CPU)
make pushtest   # the GitHub push and tools/status.py against a local bare repository
make fakeruns   # once, about 45 minutes: 21 fake runs on a local bare repo for the next target
make resulttest # results mode on those runs: seed choice, tampering, missing runs, exports
make nb         # regenerate chexpert_uncertainty.ipynb after editing notebook_src.py
```

Edit `notebook_src.py`, never the `.ipynb`: the pre-commit hook rejects a commit whose notebook
is out of date, so run `make nb` and commit both files together.

## Data and licence

CheXpert may not be redistributed. The repository contains no images and no labels: the split and
qualitative lists hold patient ids and image paths only, and pushed run files hold image paths,
scores and logits but no labels, which results mode joins back from the dataset's CSVs. Model
checkpoints (`*.pt`) stay in the Kaggle notebook output and are never pushed.
