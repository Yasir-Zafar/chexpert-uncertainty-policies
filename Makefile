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
