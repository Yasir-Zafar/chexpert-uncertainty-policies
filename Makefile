# Dev tasks. Run inside the nix dev shell (direnv loads it automatically).
PY := python

.PHONY: lint fmt

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .

.PHONY: hooks hooktest

hooks:
	pre-commit install

# The secret hook must reject a fake token; a passing run here is a failure.
hooktest:
	@tmp=$$(mktemp -d); \
	printf 'token = "github_pat_%s"\n' "$$(printf 'A%.0s' $$(seq 40))" > $$tmp/leak.txt; \
	if pre-commit run no-github-tokens --files $$tmp/leak.txt >/dev/null 2>&1; then \
		echo "hooktest FAILED: fake token was not blocked"; rm -rf $$tmp; exit 1; \
	else echo "hooktest ok: fake token blocked"; rm -rf $$tmp; fi

FAKE_ROOT := fake_data/chexpert

.PHONY: fake

fake: $(FAKE_ROOT)/train.csv

$(FAKE_ROOT)/train.csv: tools/fake_chexpert.py
	rm -rf $(FAKE_ROOT)
	$(PY) tools/fake_chexpert.py --out $(FAKE_ROOT)

# Notebook against the fake data. REPO_DIR keeps fake splits and results out of the real repo
# folders, so a fake split can never be committed as splits/split_seed42.csv.
FAKE_ENV := DATA_ROOT=$(FAKE_ROOT) REPO_DIR=fake_data/repo MPLBACKEND=Agg

.PHONY: prep

prep: fake
	$(FAKE_ENV) $(PY) tools/run_until.py "Part 3"

.PHONY: check

# Fresh fake repo, two runs of the data-preparation cells: the second must reload the split and qualitative files
# unchanged (byte-identical), proving they are created once and only read afterwards.
check: lint hooktest fake
	rm -rf fake_data/repo
	$(FAKE_ENV) $(PY) tools/run_until.py "Part 3" > /dev/null
	sha256sum fake_data/repo/splits/split_seed42.csv fake_data/repo/qual/qual_ids.csv > fake_data/created.sha
	$(FAKE_ENV) $(PY) tools/run_until.py "Part 3" > fake_data/rerun.log
	! grep -E "^(created|warning)" fake_data/rerun.log
	sha256sum --check --quiet fake_data/created.sha
	$(MAKE) nbtest
	@echo "check ok"

POLICY_NAMES := u_zeros u_ones u_ignore u_multiclass u_ones_lsr u_zeros_mc u_zeros_posw

# A smoke training on the fake data: tiny subset and batch keep CPU memory and time small.
SMOKE_ENV := $(FAKE_ENV) SMOKE=1 SMOKE_TRAIN=64 SMOKE_VAL=48 SMOKE_QUAL=16 SMOKE_BATCH=8 MODE=train

.PHONY: smoke

# One smoke run per policy on the fake data, each in its own process. The tiny subset and batch
# keep CPU memory and time small; a real smoke pass on Kaggle uses the notebook's defaults.
smoke: fake
	rm -rf fake_data/repo/runs_smoke fake_data/repo/checkpoints
	for p in $(POLICY_NAMES); do \
		echo "smoke $$p"; \
		$(SMOKE_ENV) POLICY=$$p \
			$(PY) notebook_src.py > fake_data/smoke_$$p.log 2>&1 \
			|| { tail -25 fake_data/smoke_$$p.log; exit 1; }; \
	done
	@ls fake_data/repo/runs_smoke

.PHONY: pushtest

# The GitHub push (C15) against a local bare repository: one smoke run end to end, then the
# branch rules. Takes about as long as one `make smoke` policy.
pushtest: fake
	$(PY) tools/pushtest.py

NB := chexpert_uncertainty.ipynb

.PHONY: nb nbcheck nbtest

# The notebook teammates import into Kaggle is generated from notebook_src.py and committed.
nb:
	$(PY) tools/make_nb.py

nbcheck:
	$(PY) tools/make_nb.py --check

# The committed notebook is current and runs top to bottom headlessly in a Jupyter kernel: a smoke
# training (the results part is covered by `make resulttest`). Its own repo dir keeps its run
# folder apart from `make smoke`.
nbtest: nbcheck fake
	rm -rf fake_data/nbrun fake_data/nbrepo
	$(SMOKE_ENV) REPO_DIR=fake_data/nbrepo POLICY=u_zeros_mc $(PY) -m nbconvert --to notebook --execute $(NB) \
		--ExecutePreprocessor.timeout=-1 --output-dir fake_data/nbrun --output executed.ipynb

RESULTS_REMOTE := fake_data/results_remote.git
FAKE_SEEDS := 42 43 44

.PHONY: fakeruns

# Inputs for results mode: 7 policies x 3 seeds as smoke trainings, pushed to a local bare
# repository standing in for GitHub (read them back with SMOKE=1 MODE=results). The notebook is
# snapshotted first so all runs use the same code even if it is edited meanwhile. Slow: about 25
# minutes on a CPU.
fakeruns: fake
	rm -rf $(RESULTS_REMOTE) fake_data/runsrepo
	git init -q --bare $(RESULTS_REMOTE)
	cat notebook_src.py > fake_data/nb_snapshot.py
	for s in $(FAKE_SEEDS); do for p in $(POLICY_NAMES); do \
		echo "fake run $$p seed $$s"; \
		$(SMOKE_ENV) REPO_DIR=fake_data/runsrepo REMOTE_URL=$(CURDIR)/$(RESULTS_REMOTE) \
			POLICY=$$p SEED=$$s $(PY) fake_data/nb_snapshot.py \
			> fake_data/fakerun_$${p}_$$s.log 2>&1 \
			|| { tail -25 fake_data/fakerun_$${p}_$$s.log; exit 1; }; \
	done; done
	git -C $(RESULTS_REMOTE) for-each-ref --format='%(refname:short)' refs/heads | wc -l

.PHONY: resulttest

# Results mode, cells 4.1 onwards, against the fake runs (needs `make fakeruns` first).
resulttest: fake
	$(PY) tools/resulttest.py
