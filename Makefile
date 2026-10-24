# Dev tasks. Run inside the nix dev shell (direnv loads it automatically).
PY := python

.PHONY: lint fmt

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .
