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
	$(FAKE_ENV) MODE=results $(PY) notebook_src.py

.PHONY: check

# Fresh fake repo, two notebook runs: the second must reload the split and qualitative files
# unchanged (byte-identical), proving they are created once and only read afterwards.
check: lint hooktest fake
	rm -rf fake_data/repo
	$(FAKE_ENV) MODE=results $(PY) notebook_src.py > /dev/null
	sha256sum fake_data/repo/splits/split_seed42.csv fake_data/repo/qual/qual_ids.csv > fake_data/created.sha
	$(FAKE_ENV) MODE=results $(PY) notebook_src.py > fake_data/rerun.log
	! grep -E "^(created|warning)" fake_data/rerun.log
	sha256sum --check --quiet fake_data/created.sha
	$(MAKE) nbtest
	@echo "check ok"

POLICY_NAMES := u_zeros u_ones u_ignore u_multiclass u_ones_lsr u_zeros_mc u_zeros_posw

.PHONY: smoke

# One smoke run per policy on the fake data, each in its own process. The tiny subset and batch
# keep CPU memory and time small; a real smoke pass on Kaggle uses the notebook's defaults.
smoke: fake
	rm -rf fake_data/repo/runs_smoke fake_data/repo/checkpoints
	for p in $(POLICY_NAMES); do \
		echo "smoke $$p"; \
		$(FAKE_ENV) SMOKE=1 SMOKE_TRAIN=64 SMOKE_VAL=48 SMOKE_QUAL=16 SMOKE_BATCH=8 MODE=train POLICY=$$p \
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

# The committed notebook is current and runs top to bottom headlessly (results mode, fake data).
nbtest: nbcheck fake
	rm -rf fake_data/nbrun
	$(FAKE_ENV) MODE=results $(PY) -m nbconvert --to notebook --execute $(NB) \
		--ExecutePreprocessor.timeout=-1 --output-dir fake_data/nbrun --output executed.ipynb
