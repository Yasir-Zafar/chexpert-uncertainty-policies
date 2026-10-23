# Plan v2: one graded notebook, six runs, one results pass

Supersedes `agent_plan.md` and `tickets.md`. Scope is code only; the paper is handled separately.
The proposal's figures and locked settings are taken as correct.

---

## Project context (read this first)

**What this is.** A group research project in computer vision: identify a problem, design a deep-learning solution, evaluate it, and write an IEEE-format paper. Title: *Evaluating Uncertainty Label Resolution Methods for Multi-Label Chest X-Ray Classification*. Team of three. This plan covers only the code deliverable: one Jupyter notebook with all code from data preparation to results. The paper is written separately.

**The problem.** CheXpert labels each chest X-ray with 14 observations using a rule-based labeler that reads the radiology report. Each observation is positive (1), negative (0), uncertain (-1) or not mentioned (blank). Uncertain labels come from hedging language in the report, and they are a large share for some findings: among the five pathologies studied, from 3.6% (Cardiomegaly) to 15.1% (Atelectasis). There is no standard way to turn -1 into a training target. Irvin et al. found that no single heuristic wins everywhere (U-Ones beat U-Zeros on Atelectasis, U-MultiClass was best on Cardiomegaly), and comparisons across papers are confounded by different architectures, splits, pretraining and noise handling.

**The study (the "solution").** A controlled comparison. One backbone (ImageNet-pretrained DenseNet-121, all layers fine-tuned), one dataset (CheXpert-small, frontal AP and PA images only), five pathologies (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion), one patient-level split, one training and evaluation protocol. The only thing that changes between runs is how -1 is handled during training. "Not mentioned" is treated as 0 in every policy.

| Run | Policy | What happens to the uncertain label u | Loss and head |
|---|---|---|---|
| R1 | U-Zeros (baseline) | u becomes 0 | BCE, 5 sigmoid outputs |
| R2 | U-Ones | u becomes 1 | BCE, 5 sigmoid outputs |
| R3 | U-Ignore | u is left out of the loss | masked BCE, 5 sigmoid outputs |
| R4 | U-MultiClass | u is kept as a third class | 3-class cross-entropy per pathology, 15 logits |
| R5 | U-Ones + LSR | u becomes a soft target drawn from a fixed range, [0.55, 0.85] | BCE with soft targets |
| R6 | U-Zeros + MC Dropout | u becomes 0; uncertainty is estimated at inference (dropout p = 0.3, 20 passes) | BCE, dropout before the head |

U-SelfTrained is excluded from the main comparison (it needs a second training stage, so it would add another difference between methods). Table III of the proposal also promises one extra experiment: positive-class weights on the baseline.

**Objectives and where this plan covers them**

| Proposal objective | Covered by |
|---|---|
| 1. Review and categorize existing approaches | Already done in the paper's literature review; no code |
| 2. Implement a common evaluation protocol (fixed split, backbone, metric set) | Notebook parts 1 to 3, frozen settings in cell 2 |
| 3. Compare U-Ignore, U-Zeros, U-Ones, U-MultiClass | Experiment E2 |
| 4. Evaluate label smoothing and Bayesian uncertainty quantification in the same protocol | Experiment E3 (plus E5 calibration, E6 entropy) |
| 5. Which strategy generalizes best across pathologies with different uncertain prevalence | Experiment E4 (descriptive: five pathologies, no formal correlation) |

**Data roles (easy to mix up)**
- **Training data:** the official `train.csv`, frontal images only, split by patient 90% / 10% with seed 42 into *train* and *internal validation*. The split is created once and saved to a file.
- **Internal validation:** keeps the original -1 labels. Used for early stopping, learning-rate scheduling, threshold selection, model selection and the MC Dropout entropy analysis.
- **Test data:** the official 200-study validation set (`valid.csv`), labelled by consensus of three board-certified radiologists, with no uncertain class. Never used for tuning. The official 500-study test set is hidden, so it cannot be used.
- **Metrics:** AUROC (main), AUPRC, F1 at the Youden threshold chosen on internal validation, ECE and Brier score; predictive entropy for the MC variant. 95% bootstrap confidence intervals (1,000 resamples) and paired bootstrap tests against U-Zeros.

**Team and logistics.** Three people. One person builds the notebook; all three run models (two each), plus the extra class-weighting run. Training happens on free Kaggle GPU notebooks. Each run's outputs are pushed to its own GitHub branch, and a final CPU-only results run combines them into the tables and figures.

**Glossary.** *u*: the uncertain label (-1). *Policy*: a rule for turning u into a training target. *Run*: one policy trained once at one seed. *Mode*: `train` (one run on GPU) or `results` (combine all runs on CPU). *Seed layer*: all six policies at one seed.

---

## 0. What changed and why

The assignment's final submission is **one notebook containing all the code from data preparation to results**, plus the paper. Implementation and code quality alone is 15 of the points, and qualitative evaluation is worth as much as the results tables (10 each). So:

- No Python package, CI, hash machinery or 46-step build. One master notebook, built cell by cell.
- The six policies are six entries in one registry cell. Teammates change one line to run theirs.
- Added what the first plan lacked: a qualitative section (examples, class activation maps, confusion matrices, failure cases) and a computational-cost section (parameters, training time, inference time, GPU, memory).
- Experiments are organised as named experiments E1 to E7, each with a one-line purpose, because the guidance asks for that.
- Seeds are an option. The default stays one seed (42) per policy, as in the proposal. If time allows, results mode can use up to three seeds (42, 43, 44) and report mean and standard deviation. That is 18 runs in total, 12 of them optional.

---

## 1. Rubric to code

| Rubric item (points) | What earns it |
|---|---|
| Implementation and code quality (15) | One notebook that runs top to bottom in both modes from a clean session; markdown header per section; small functions with docstrings; one frozen CONFIG cell; inline sanity asserts; library versions printed; no dead code |
| Experimental execution and completeness (10) | All six policies trained under one frozen protocol; a consistency cell proves identical split, data, code version and settings across runs; per-epoch logs for every run |
| Results and quantitative evaluation, tables (10) | Tables T1 to T7 with 95% CIs, paired differences against U-Zeros, objective-5 table, calibration, cost table; all exported as LaTeX and CSV |
| Qualitative evaluation (10) | Example gallery, activation-map comparison across the six policies, behaviour on uncertain-labelled images, confusion matrices, failure cases |
| Analysis and interpretation (5) | Plots chosen to answer "why" (uncertain-case behaviour, entropy), plus an auto-generated findings cell with the numbers the paper quotes |

Discussion, limitations, conclusion and LaTeX quality are your paper. The notebook's job is to hand you the numbers and figures.

---

## 2. Design

**One master notebook, `chexpert_uncertainty.ipynb`, two modes.**

| Mode | Where | What it does |
|---|---|---|
| `train` | Kaggle GPU, once per policy | data prep, train one policy, save predictions and artifacts, push them to GitHub |
| `results` | Kaggle CPU (no GPU quota) | data prep and stats, fetch the six runs from GitHub, all tables, figures, qualitative analysis, export |

The file submitted is the executed `results` notebook: all code visible, outputs for E1 to E7 and the qualitative section. The six executed `train` notebooks stay on GitHub as the training logs.

**Authoring.** Write `notebook_src.py` with `# %%` cell markers (jupytext) and generate the `.ipynb` from it. Plain text is easy to edit cell by cell, run as a script and lint. The generated `.ipynb` is what gets uploaded to Kaggle. The `.py` is the source of truth.

**Local development without the licensed data.** `tools/fake_chexpert.py` builds a tiny CheXpert-shaped dataset (frontal and lateral rows, AP and PA, -1/0/1/blank labels, random grayscale JPEGs). Environment variables `MODE`, `POLICY`, `SEED`, `SMOKE`, `DATA_ROOT` let the whole notebook run on CPU against it. The tool is not part of the submission, and the notebook contains no fake-data code.

**GitHub logging.**
- Each run pushes `runs/<policy>_s<seed>/` to branch `run/<policy>_s<seed>` from inside the notebook, using a fine-grained token from Kaggle Secrets. Progress (the epoch log) is pushed after every epoch, so a dead session still leaves history.
- Kaggle's built-in "Link to GitHub" (File menu) can also be switched on to keep a copy of each notebook version. It saves the notebook, not output files, and users report it won't recommit under an unchanged file name, so treat it as a convenience and rely on the push cell.
- Results mode fetches the six branches and reads `runs/` from them.
- Checkpoints (`best.pt`, about 28 MB) are never pushed. They stay in the Kaggle notebook output.

**Repo layout**

```
uncert-cxr/
  chexpert_uncertainty.ipynb
  notebook_src.py
  README.md
  Makefile  pyproject.toml (ruff config)  .pre-commit-config.yaml
  docs/                         (this plan, the proposal PDF)
  splits/split_seed42.csv
  qual/qual_ids.csv
  tools/fake_chexpert.py  tools/status.py
  runs/<policy>_s<seed>/        (arrives from the run branches)
  results/tables/  results/figures/
```

---

## 3. Settings

**Cell 1 is the only cell anyone edits for a run.**

```python
MODE = os.environ.get("MODE", "train")
POLICY = os.environ.get("POLICY", "u_zeros")
SEED = int(os.environ.get("SEED", 42))
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_SEEDS = os.environ.get("N_SEEDS", "1")
```

`SEED` is read in train mode and must be 42, 43 or 44. `N_SEEDS` is read in results mode: `"1"` (default) uses seed 42 only, `"2"` uses 42 and 43, `"3"` uses all three, and `"auto"` uses the largest count for which every policy has every one of those seeds complete. An explicit count with runs missing stops with a list of what is missing.

**Cell 2 is the frozen protocol.** It never changes per run, and results mode asserts it is identical across the six runs.

```python
CONFIG = {
    "code_version": "v1",
    "data_root": os.environ.get("DATA_ROOT", "/kaggle/input/chexpert"),
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
}
```

Not in CONFIG, by design: head, loss, dropout and MC settings live in the policy registry (cell 10).

**Defaults where the proposal is silent** (printed in the notebook's protocol cell so the paper can state them):

| Topic | Default |
|---|---|
| Validation labels per policy | U-Ignore and U-MultiClass drop uncertain entries from validation AUROC. U-Zeros, U-Ones, LSR (maps u to 1) and MC (maps u to 0) use the hard mapping. A policy-independent **certain-only** validation AUROC is logged for every policy every epoch |
| Masked loss (U-Ignore) | sum of per-entry loss over unmasked entries divided by max(unmasked count, 1), per batch; all-uncertain batch gives 0, not NaN |
| LSR sampling | a fresh Uniform(0.55, 0.85) draw for every uncertain entry at every iteration, from a seeded generator |
| U-MultiClass score | classes neg=0, pos=1, unc=2, logits reshaped (B, 5, 3). Primary AUROC score is softmax P(positive); P(pos)/(P(pos)+P(neg)) is reported separately |
| Entropy analysis | the test set has no uncertain labels, so predictive entropy is compared on the internal validation split (which keeps -1): certain-negative, certain-positive, uncertain |
| MC variant | also evaluated with dropout off, to separate the effect of training with dropout from the effect of MC inference |
| Bootstrap | patient-level resampling (image rows grouped by patient), one set of resample indices reused for every policy and metric so comparisons are paired |
| Paired p-value | p = min(1, 2 * min((1 + #{d <= 0}) / (B + 1), (1 + #{d >= 0}) / (B + 1))) with d the paired difference. Holm correction is the default for headline claims: within each row of T3 (the mean and each pathology), the five comparisons against U-Zeros form one family. Raw and Holm-adjusted p-values are both shown, next to the mean difference and its CI |
| ECE | 10 equal-width bins on the positive-class probability |
| F1 threshold | Youden's J chosen on the internal validation predictions, per pathology, with the validation labels above |
| Resize | direct to 320 x 320, no letterboxing |
| GPU | one GPU only (two T4s would split the batch and change BatchNorm statistics); one accelerator type for all six runs |
| Seeds | One seed (42) per policy is the primary protocol. Extra seeds 43 and 44 change only head initialisation, sample order, augmentation, dropout masks and LSR draws; the data split always stays at `split_seed` 42. Two policies run with the same seed see the identical sample order. Predictions are never averaged across seeds (no ensembling); metrics are averaged instead |
| Seed statistics | With several seeds, tables report mean and SD across seeds, and the CI is a seed-averaged patient-level bootstrap (section 5). Paired differences pair the same seed across policies |
| Selection check | The proposal's rule (best policy-resolved validation AUROC) stays the primary checkpoint for every headline table. Because that target differs per policy, a second checkpoint chosen by the policy-independent certain-only AUROC is also evaluated and reported in T9 |
| Class weighting | None in the six policy runs (Table III). The extra experiment uses BCE `pos_weight` = n_neg / n_pos per pathology, computed on the training split with uncertain labels resolved to 0, on U-Zeros only |
| Determinism | Seeds are fixed everywhere, but deterministic algorithms are not forced: they can slow some operations, and FP16 training is not bitwise reproducible anyway. Reproducibility here means same code version, same split and same seed, not identical bits |
| Speed settings | `channels_last` and `cudnn_benchmark` are decided once, by an A/B benchmark on the chosen GPU at GATE-2, then frozen in CONFIG for every run |

---

## 4. Notebook cell map

Each cell group has a markdown header explaining it. Inline `assert`s replace a separate test suite.

**Part 0: setup**
- **C0** Title, research questions, how to run (modes, which cell to edit).
- **C1** Run settings (above, including `N_SEEDS`). **C2** CONFIG (above). Imports, seed helper, version print.
- **C3** Environment: Kaggle detection, secrets, git helpers (`clone_repo`, `push_run`, `fetch_runs`) with the token passed per command and scrubbed from logs, dataset root discovery.

**Part 1: data preparation**
- **C4** Read both CSVs, strip the path prefix, filter to Frontal then AP/PA (print the row count after each), parse patient and study ids, blanks to 0, keep -1. Mirror check: `train.csv` 223,414 rows and `valid.csv` 234 rows before filtering.
- **C5** Dataset statistics: positive / negative / uncertain / not-mentioned per pathology for full train, analysis train, internal validation and test; images and patients per split; bar chart of uncertain prevalence. Writes `results/tables/dataset_stats.csv` and `.tex`.
- **C6** Patient-level 90/10 split, seed 42. If `splits/split_seed42.csv` exists in the repo it is loaded; otherwise (builder only, once) it is created and committed. Asserts: no patient in both parts, validation fraction within 0.5 percentage points of 10%, SHA-256 printed.
- **C7** Fixed qualitative id lists (`qual/qual_ids.csv`), created once with a fixed seed **before any training**: every test image, plus a 600-image internal-validation subset stratified so each pathology has at least 50 uncertain-labelled images.
- **C8** Dataset class (uint8 1 x 320 x 320, raw labels in {0, 1, -1}), CPU transforms (rotation, random resized crop, brightness and contrast jitter, no flip), GPU-side normalisation (3 channels, ImageNet mean and std), seeded sampler (seed + epoch, so every policy sees the same order), worker seeding.

**Part 2: model, policies, metrics**
- **C9** DenseNet-121 (ImageNet weights, all layers trainable). `extract_features` (global-average-pooled 1024-d), `classify` (optional Dropout then `Linear(1024, 5)` or `Linear(1024, 15)`), `cam` (class activation maps: head weights times the final 10 x 10 ReLU feature map, no gradients; for U-MultiClass use the positive-class row of each pathology), `count_trainable_params`.
- **C10** **Policy registry**, the "classifier cells". Six entries (plus a seventh, `u_zeros_posw`, flagged `extra` and excluded from the six-policy comparison), each with `head`, `loss(logits, y_raw, generator)`, `score(logits)`, `val_labels(y_raw)`, `dropout_p`, `mc`. Inline asserts: hand-computed losses on a 2 x 5 example for each policy, all-uncertain batch under U-Ignore is 0, LSR targets inside [0.55, 0.85], U-MultiClass encodes {0, 1, -1} as {0, 1, 2}. `u_zeros_posw` is `u_zeros` with BCE `pos_weight` = n_neg / n_pos per pathology, computed on the training split with uncertain labels resolved to 0 (assert: weights are finite and positive).
- **C11** Metrics (pure NumPy): AUROC, AUPRC, Youden threshold, F1, ECE, Brier, binary entropy, mutual information, patient-level bootstrap with shared indices, paired difference with CI and p-value, and the Holm adjustment (asserted against a hand-worked example). Asserts against scikit-learn on random data and ECE = 0 on perfectly calibrated synthetic data.

**Part 3: training run (`MODE == "train"`)**
- **C12** `train_one_epoch` (AMP, GradScaler, policy loss in float32, NaN/Inf guard that fails the run with the reason recorded), `evaluate` (validation mean AUROC with policy-resolved labels, plus certain-only AUROC), early stopping, plateau scheduler on validation mean AUROC, best-checkpoint restore. Early stopping and the scheduler follow the policy-resolved AUROC (the proposal's rule); the epoch with the best certain-only AUROC is also kept as a second checkpoint (`best_certain.pt`, the same file when the two epochs coincide). Records seconds per epoch, images per second and peak GPU memory. Monitoring is throttled so it never costs step time (section 11): one log line every `log_every` steps, gradient norm and loss scale to `grad_stats.csv` every `grad_stats_every` steps, and a background thread sampling `nvidia-smi` into `gpu_trace.csv`.
- **C13** Run it. `SMOKE` uses 2,000 training and 500 validation images for 1 to 2 epochs.
- **C14** Inference and artifacts: validation and test predictions for the primary checkpoint, and again for the second (certain-only) checkpoint when it differs (inference only, no extra training); MC variant caches backbone features once and applies dropout plus the head 20 times (mathematically the same as 20 full passes, assert within 1e-5 on a small batch); CAMs for the `qual_ids` images (and every test image); inference timing (batch 32, warm-up, `cuda.synchronize`); GFLOPs per image from PyTorch's built-in FLOP counter; writes the run folder.
- **C15** Push to `run/<policy>_s<seed>`.

**Part 4: results (`MODE == "results"`)**
- **C16** Fetch the runs for the seeds selected by `N_SEEDS` (six policies per seed); consistency check (same CONFIG, `code_version`, split hash, data fingerprint, test row order; seed is the only thing that may differ between runs of the same policy; same GPU name is a warning; every run `complete`). A seed counts only if all six policies have it complete; incomplete layers are listed and ignored. The extra run `u_zeros_posw`, if present, is loaded separately and held to the same CONFIG and `code_version`. Print the protocol table.
- **C17** E1 to E3 quantitative tables and plots (below).
- **C18** E4 objective-5 table and plot.
- **C19** E5 calibration. **C20** E6 uncertainty analysis. **C21** E7 computational cost and the E8 class-weighting comparison (skipped with a message if the extra run is absent).
- **C22** Qualitative section (below).
- **C23** Training curves (with several seeds: mean line with a min-max band).
- **C24** Export every table (CSV and LaTeX `booktabs`) and every figure (vector PDF plus 300 dpi PNG, single-column 3.5 in or double-column 7.16 in) to `results/`, and print an index of what was written.
- **C25** Findings cell: best policy overall and per pathology, largest and smallest gains over U-Zeros, whether each CI excludes zero and whether the Holm-adjusted p-value is below 0.05, written out as sentences with numbers. With several seeds it also states whether the policy ranking is the same in every seed.

---

## 5. Experiments (each begins with a one-line purpose in the notebook)

| ID | Name | Question | Output |
|---|---|---|---|
| E1 | Baseline | How well does the U-Zeros baseline do under the common protocol? | T1 row, T2 row |
| E2 | Hard-label policies | Do U-Ones, U-Ignore and U-MultiClass beat U-Zeros? | T1 (mean AUROC, AUPRC, F1, ECE, Brier with CIs), T2 per pathology, T3 paired differences with CI and p, ROC overlays, difference forest plot |
| E3 | Beyond hard labels | Do label smoothing (U-Ones + LSR) and Bayesian uncertainty (U-Zeros + MC Dropout) help in the same protocol? | same tables, plus the MC dropout-off ablation (T6) and U-MultiClass score variants (T7) |
| E4 | Generalisation across uncertain prevalence | Which policy gains most where uncertainty is common? | objective-5 table: per pathology, uncertain prevalence in the analysis split next to each policy's AUROC difference versus U-Zeros, ranking per pathology, scatter figure. Descriptive only, no correlation test |
| E5 | Calibration | Which policy is best calibrated? | ECE and Brier table (T4), reliability diagrams |
| E6 | Uncertainty signal | Is MC predictive entropy higher on uncertain-labelled studies? | entropy by group per pathology with CI and Mann-Whitney U, AUROC of entropy for uncertain versus certain, distribution figure |
| E7 | Computational cost | What does each policy cost? | T5: trainable parameters, GPU, training time (total and per epoch), images per second, inference ms per image, peak GPU memory, GFLOPs per image, mean GPU utilisation, MC inference cost (head-only cached versus naive 20 passes) |
| E8 | Class weighting (the extra experiment in Table III) | Does positive-class weighting help the baseline? | T10: `u_zeros_posw` next to `u_zeros` (all metrics with CIs, one paired difference with CI and p-value; a single comparison needs no multiplicity correction). Seed 42 only, reported separately from the six-policy comparison |

An optional literature reference row (published values from the cited paper, typed in from the paper itself, labelled not directly comparable) can sit under T1.

**With more than one seed (`N_SEEDS` of 2 or 3).** The single-seed path is unchanged and stays the default. When extra seeds exist, results mode adds:
- **Tables:** mean and SD across seeds for every metric (SD shown only when there are at least two seeds). The CI becomes a seed-averaged bootstrap: each replicate resamples test patients once, computes the metric for every seed on that same resample, and averages over seeds. Predictions are never averaged across seeds, so this is not ensembling.
- **Paired differences against U-Zeros:** pair the same seed across policies (policy at seed s versus U-Zeros at seed s), average the difference over seeds within each replicate, and report the CI, the p-value and the SD of the per-seed differences.
- **E4:** error bars (SD across seeds) on the objective-5 plot. **E7:** mean and SD of the cost columns.
- **T8, seed variability:** per policy, mean, SD and range of mean AUROC, and whether the ranking of the six policies is identical in every seed.
- **Seed 42 only:** the qualitative section and the example figures. E6 reports its entropy gaps per seed alongside the seed-42 figure.

**T9, selection sensitivity:** mean AUROC per policy under the primary checkpoint and under the certain-only checkpoint, and whether the ranking of the six policies changes. Headline tables stay on the proposal's selection rule.

**Qualitative section (C22), all from saved predictions and CAMs, no GPU:**
1. **Example gallery:** per pathology, 2 true positives, 2 true negatives, 2 false positives, 2 false negatives (chosen from the baseline), each image with the six policies' probabilities in the caption.
2. **Activation-map comparison:** the same image across six columns, one row per policy, for success and failure cases and for uncertain-labelled validation images.
3. **Behaviour on uncertain-labelled images:** distribution of predicted probability under each policy for entries labelled -1. This is the figure that shows the effect of the labelling change directly.
4. **Confusion matrices** at each policy's validation Youden threshold, per pathology.
5. **Failure cases:** the most confident wrong predictions on the test set for the baseline and the best policy.

---

## 6. What each run saves (`runs/<policy>_s<seed>/`, all small)

`run_info.json` (policy, seed, `code_version`, CONFIG, split SHA-256, data fingerprint = SHA-256 over `train.csv` and `valid.csv`, who ran it, GPU name and count, torch and CUDA versions, status, best epoch, epochs run, stop reason, trainable parameters, training seconds, images per second, inference ms per image, peak GPU memory), `metrics_epoch.csv`, `grad_stats.csv`, `gpu_trace.csv`, `train.log`, `val_preds.npz`, `test_preds.npz` (row ids, patient ids, raw labels, scores, logits), `val_preds_certain.npz` and `test_preds_certain.npz` (same, from the certain-only checkpoint), `cam_test.npz`, `cam_val_subset.npz`, and for the MC run `mc_val.npz` and `mc_test.npz` (sample mean, entropy, mutual information, deterministic score). Test labels are read only after the best checkpoint is chosen, and only to predict.

---

## 7. Running it (three people, two runs each)

| Person | Runs |
|---|---|
| Builder | `u_zeros` (baseline), `u_zeros_mc` |
| Teammate 1 | `u_ones`, `u_ignore` |
| Teammate 2 | `u_multiclass`, `u_ones_lsr` |

Swap freely; the runs are independent. The extra experiment promised in Table III, `u_zeros_posw` (seed 42), is a seventh run: the builder takes it by default, and it can go to anyone with spare quota. It is not part of the six-policy comparison, the seed layers or the six-per-seed checks.

**Teammate steps:** download the `.ipynb` from the repo, import it into Kaggle, set accelerator and internet on, attach the dataset, attach the secrets `GITHUB_PAT`, `GITHUB_REPO`, `GITHUB_USER`, keep the notebook private, edit `POLICY` in cell 1, **Save Version, Save and Run All (Commit)**. Close the tab; it runs in the background.

**Rules**
1. Everyone runs the same tagged notebook version (`v1`), the same dataset mirror and the same accelerator.
2. Each teammate runs a `SMOKE = True` pass first and checks: files on their branch, no token in any log, `.pt` not in git, row counts as expected.
3. Teammates start full runs only after the builder's baseline has finished two epochs and looks sane (validation mean AUROC rising, no NaN, throughput acceptable). The epoch log appears on `run/u_zeros_s42` while it trains.
4. Teammates push only to their own `run/...` branches, never to `main`.
5. If a run fails or looks wrong, do not edit and rerun. Tell the builder; a fix means a new version tag and a rerun of every affected run.

**Compute.** Quota is per account (read it off your Kaggle account page; it is not published and can change), so each person needs only the GPU time for two runs. The first smoke run prints training images per second. Rule of thumb: at about 150 images per second an epoch over roughly 170,000 images takes about 19 minutes and a full 10-epoch run about 3 hours, well inside a 12-hour session. Compare the P100 and one T4 once and pick one for all six runs; the T4 has tensor cores and the P100 does not, so FP16 may favour the T4. If throughput falls well short, first try `num_workers` 3 to 6 and channels-last; GPU-side augmentation is the last resort and needs the team's sign-off because it changes the augmentation implementation.

**Extra seeds (optional).** Seed layers go in order: seed 43 for all six policies, then seed 44. A layer is usable only when all six policies have finished it, so finish a layer across the team before starting the next; results mode lists and ignores incomplete layers. Set `SEED` in cell 1 and use the same tagged notebook version. Each run lands on its own branch `run/<policy>_s<seed>`, so an extra run can never overwrite seed 42. Finished runs are stored there (predictions, logs, activation maps, cost numbers) and are never rerun when a layer is added: going from two seeds to three costs only the six seed-44 runs, and going from one to three costs twelve. Only results mode is rerun, which is a CPU job. The one exception is a protocol change: a new `code_version` makes older runs ineligible, and results mode will say so. Quota rough estimate (unmeasured, replace with GATE-2 numbers): at 1 to 3.5 hours per run, one seed layer costs each account about 2 to 7 GPU hours, so three seeds is about 6 to 21 hours per account. A layer can also wait for the next weekly quota reset.

Results mode runs on a Kaggle CPU session with the dataset attached (needed for the image galleries), so it costs no GPU quota.

---

## 8. Build order: six chunks, one at a time

Each chunk is a separate prompt to the coding agent, finished and tested on the fake dataset before the next starts.

| Chunk | Cells | Done when |
|---|---|---|
| 1 | C0 to C7 plus `tools/fake_chexpert.py`, the `Makefile`, the ruff config and the pre-commit hook | `make check` passes (ruff clean, fake data builds, cells run); on fake data: counts printed after each filter; stats table written; split file created, reloaded and byte-identical; no patient overlap; the pre-commit hook rejects a file containing a fake `github_pat_` string |
| 2 | C8 to C11 | All inline asserts pass (policy losses, shapes, `forward == classify(extract_features)`, metrics versus scikit-learn, identical sample order across two seeded runs) |
| 3 | C12 to C14 | `SMOKE=1` for all six policies and `u_zeros_posw` on CPU writes every artifact; MC cached-feature result matches a naive 20-pass loop; loss falls on a repeated batch for each policy; `SEED=43` changes the sample order while two policies with the same seed see the identical order; both checkpoint paths write their files, identical when the best epochs coincide; grad stats written; the GPU-trace thread starts and stops cleanly (skipped on CPU); the FLOP counter returns a positive number |
| 4 | C15, notebook generation, the README and `tools/status.py` | Push against a local bare repo: progress push then final push both appear, `.pt` excluded, token absent from output; `tools/status.py` prints one row per run branch (policy, who, epoch, validation AUROC, images per second, status) from the pushed logs |
| 5 | C16 to C21, C23, C25 | Results mode on fake runs with `N_SEEDS` of 1, 2 and 3 produces every table without error; `auto` picks the right count and an explicit count with runs missing stops with the list; a policy compared with itself gives difference 0 and p = 1; one AUROC checked by hand; the `N_SEEDS=1` tables equal the seed-42 slice of the `N_SEEDS=3` tables |
| 6 | C22, C24 | All qualitative figures and exports render from the fake runs; index of outputs printed |

Chunks 5 and 6 can be built while the real runs are training.

---

## 9. Risks that affect development

| Risk | Handling |
|---|---|
| Kaggle mirror differs from the official layout | C4 row-count mirror check; the dataset slug and path prefix are fixed once and shared |
| Loader limits throughput on 4 vCPUs | Throughput printed at the first smoke run; workers, channels-last, then sign-off for GPU-side augmentation |
| Runs from different code or data versions get combined | `code_version`, split hash, data fingerprint and CONFIG checked in C16 |
| Token leakage | Fine-grained single-repo token, per-command auth header, log scrubbing, private notebooks, token only in Secrets |
| Secrets stripped when pushing a notebook with the Kaggle command line | Use the browser Save and Run All; test the command-line route on a tiny notebook before relying on it |
| FP16 overflow or NaN | Loss in float32, GradScaler, NaN guard that fails fast with the reason |
| Test set is about 200 studies | Patient-level bootstrap; wide CIs are expected and reported as they are |
| Session dies mid-run | Epoch log pushed every epoch; rerun from scratch |
| Extra seed layer only half finished | A seed counts only if all six policies finished it; results mode lists and ignores incomplete layers, so a partial layer never contaminates the tables |

---

## 10. Cards

Size: S up to 1 hour, M 1 to 3 hours, L 3 to 6 hours of hands-on time (excluding GPU wait). A card is ready when everything under "Depends on" is done.

| ID | Owner | Size | Depends on | Done when |
|---|---|---|---|---|
| SET-1 | Builder | S | none | Repo created, both teammates have write access, `owner/name` shared |
| SET-2 | Builder | M | none | One dataset mirror chosen and confirmed acceptable under the CheXpert agreement; it attaches in a Kaggle notebook; `train.csv` has 223,414 rows and `valid.csv` 234; slug and path prefix recorded and shared |
| SET-3-B | Builder | S | SET-1 | Kaggle phone-verified, quota noted, fine-grained token (single repo, Contents read and write), Secrets `GITHUB_PAT`, `GITHUB_REPO`, `GITHUB_USER` created |
| SET-3-T1 | Teammate 1 | S | SET-1 | Same as SET-3-B in their accounts; token expiry set after their last run |
| SET-3-T2 | Teammate 2 | S | SET-1 | Same as SET-3-B in their accounts |
| BLD-1 | Builder | L | SET-2 | Chunk 1 done, including the `Makefile` and pre-commit hook |
| BLD-2 | Builder | L | BLD-1 | Chunk 2 done |
| BLD-3 | Builder | L | BLD-2 | Chunk 3 done |
| BLD-4 | Builder | M | BLD-3, SET-1 | Chunk 4 done; `.ipynb` generated and committed; README has the teammate steps from section 7; `tools/status.py` works |
| GATE-1 | Builder | M | BLD-4, SET-3-B | Real-data smoke on Kaggle for `u_zeros_mc` and `u_multiclass`: files on their branches, no token in logs, `.pt` not in git, expected row counts, throughput printed, data fingerprint recorded; monitoring overhead check: step time over 200 steps with logging, grad stats and GPU trace on versus off differs by under 1%, otherwise lower the frequencies |
| GATE-2 | Builder | S | GATE-1 | P100 versus T4 compared, one accelerator chosen; 200-step A/B of `channels_last` and `cudnn_benchmark` on it, fastest pair frozen in CONFIG; version tag `v1` created, teammates told the tag and the accelerator |
| RUN-R1 | Builder | L | GATE-2 | `u_zeros` complete on `run/u_zeros_s42`; after epoch 2 post "go" to the team; sanity checks (below) pass |
| RUN-R6 | Builder | L | RUN-R1 | `u_zeros_mc` complete, including `mc_val.npz` and `mc_test.npz` |
| RUN-R7 | Builder (or anyone with spare quota) | L | RUN-R1 | Extra experiment: `u_zeros_posw` seed 42 complete on `run/u_zeros_posw_s42`; sanity checks pass |
| RUN-R2 | Teammate 1 | L | GATE-2, SET-3-T1, RUN-R1 (epoch 2 ok) | Smoke pass first, then full `u_ones` run; sanity checks pass |
| RUN-R3 | Teammate 1 | L | RUN-R2 | Full `u_ignore` run; sanity checks pass |
| RUN-R4 | Teammate 2 | L | GATE-2, SET-3-T2, RUN-R1 (epoch 2 ok) | Smoke pass first, then full `u_multiclass` run; sanity checks pass |
| RUN-R5 | Teammate 2 | L | RUN-R4 | Full `u_ones_lsr` run; sanity checks pass |
| BLD-5 | Builder | L | BLD-3 | Chunk 5 done |
| BLD-6 | Builder | L | BLD-5 | Chunk 6 done |
| RES-1 | Builder | M | BLD-6, RUN-R1, RUN-R2, RUN-R3, RUN-R4, RUN-R5, RUN-R6, RUN-R7 | Results mode run on a Kaggle CPU session with consistency checks passing; `results/` committed; executed notebook downloaded |
| REV-1 | Teammate 1 | S | RES-1 | Two policies' mean and per-pathology AUROC and AUPRC recomputed independently from `test_preds.npz` and matched to the tables, or discrepancies reported |
| REV-2 | Teammate 2 | S | RES-1 | Notebook run top to bottom from a clean session in results mode; unclear cells and missing explanations reported |
| SUB-1 | Builder | S | REV-1, REV-2 | Final executed notebook and exported tables and figures handed to the paper (the OPT-RES versions if extra seeds were run); README states the seed count; no secrets in git history |

**Sanity checks for every full run:** `run_info.json` has status `complete`, the right `code_version` and the team's split hash and data fingerprint; no NaN in `metrics_epoch.csv`; validation mean AUROC rises over the first epochs and is not below about 0.75; test row count equals the C4 count; one line sent to the builder with date, policy, GPU, wall-clock and best epoch.

### Optional: extra seeds (default stays at one)

Only if time and quota allow. Layers are all-or-nothing and go in order (section 7). Same sanity checks as above; the manifest seed must match the branch name.

| ID | Owner | Size | Depends on | Done when |
|---|---|---|---|---|
| OPT-GO | Builder | S | RUN-R1 | Measured hours per run, remaining time and remaining quota per account reviewed with the team; decision recorded in the README: 1, 2 or 3 seeds |
| OPT-S43-B | Builder | L | OPT-GO, RUN-R1, RUN-R6 | Seed 43 complete for `u_zeros` and `u_zeros_mc` |
| OPT-S43-T1 | Teammate 1 | L | OPT-GO, RUN-R2, RUN-R3 | Seed 43 complete for `u_ones` and `u_ignore` |
| OPT-S43-T2 | Teammate 2 | L | OPT-GO, RUN-R4, RUN-R5 | Seed 43 complete for `u_multiclass` and `u_ones_lsr` |
| OPT-S44-B | Builder | L | OPT-S43-B, OPT-S43-T1, OPT-S43-T2 | Seed 44 complete for `u_zeros` and `u_zeros_mc` |
| OPT-S44-T1 | Teammate 1 | L | OPT-S43-B, OPT-S43-T1, OPT-S43-T2 | Seed 44 complete for `u_ones` and `u_ignore` |
| OPT-S44-T2 | Teammate 2 | L | OPT-S43-B, OPT-S43-T1, OPT-S43-T2 | Seed 44 complete for `u_multiclass` and `u_ones_lsr` |
| OPT-RES | Builder | M | RES-1, OPT-S43-B, OPT-S43-T1, OPT-S43-T2 (plus OPT-S44-B, OPT-S44-T1, OPT-S44-T2 for three seeds) | Results mode rerun with `N_SEEDS` set to 2, 3 or `auto`; consistency check passes; the seed-42 slice of every table equals RES-1's tables; T8 and the error bars are present; executed notebook and `results/` replace the single-seed versions |


---

## 11. Tooling

**Rule: nothing here may add measurable time to an epoch.** Every item runs outside the training loop, runs only every N steps, runs in a background thread, or runs only in smoke and benchmark passes. GATE-1 checks this: step time with all monitoring on versus off must differ by under 1%.

| Group | Tool | What it gives you | Cost to epoch time |
|---|---|---|---|
| Dev | `ruff` (lint and format) via `make lint` | Clean, consistent code for the 15-point quality item | none, never runs in training |
| Dev | `Makefile`: `make fake` (build the fake dataset), `make smoke` (all seven policies on CPU), `make nb` (jupytext conversion), `make check` (lint, smoke, results mode on the fake runs, headless `nbconvert --execute` of the generated notebook) | One command proves the notebook runs top to bottom | none |
| Dev | `pre-commit` hook: ruff plus a pattern check that blocks `github_pat_`, `ghp_` strings and `kaggle.json` | Keeps tokens out of git on three laptops | none |
| Monitoring | One throttled log line every `log_every` steps: loss, images per second, ETA, learning rate, loss scale. No progress bar (bars flood Kaggle commit logs) | Readable logs, early sign of a slow run | negligible, one print per 50 steps |
| Monitoring | Gradient norm and AMP loss scale every `grad_stats_every` steps to `grad_stats.csv` | Explains NaNs, overflow and unstable runs | negligible, once per 200 steps |
| Monitoring | Background thread running `nvidia-smi` every 10 s into `gpu_trace.csv` (utilisation, memory, clock, power) | Shows whether epochs are GPU-bound or loader-bound; feeds the cost table | none, separate thread |
| Monitoring | `tools/status.py`: reads the pushed progress from the run branches and prints one table of all runs | The team sees who is at which epoch; the go signal for teammates | none, runs on your laptop |
| Speed | `channels_last` memory format with AMP; `cudnn.benchmark` on or off; both decided by a 200-step A/B at GATE-2 | Faster convolutions on tensor-core GPUs (the T4 has them) | reduces or neutral |
| Speed | Loader settings: `persistent_workers`, `prefetch_factor=4`, `pin_memory`, `non_blocking` copies, GPU-side normalisation, `cv2.setNumThreads(0)` | Keeps the GPU fed | reduces |
| Reporting | `torch.utils.flop_counter.FlopCounterMode`, once per run | GFLOPs per image for the cost table, no new dependency | none, one forward pass |
| Diagnosis | `torch.profiler` over about 20 steps, only in a benchmark pass at GATE-1, never in production runs | Locates the bottleneck if throughput misses the gate | none in production |

**Left out on purpose**
- **Weights & Biases, TensorBoard, MLflow:** each adds an account, keys and notebook clutter. The per-epoch CSV pushed to GitHub plus `tools/status.py` already give live curves and a system of record.
- **PyTorch Lightning, `torchmetrics`:** they hide the training loop and metrics that the rubric grades as your own implementation.
- **`torch.compile`:** compile time and extra risk for an uncertain gain.
- **Forced deterministic algorithms:** can slow operations; seeds are still fixed (section 3, Determinism).
- **DALI (GPU JPEG decoding):** extra complexity; the throughput gate decides whether loader speed is a problem at all.
