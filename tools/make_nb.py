"""Generate ``chexpert_uncertainty.ipynb`` from ``notebook_src.py`` (the source of truth).

The notebook is what teammates import into Kaggle, so it must be a pure function of the source:
the same bytes every time, no outputs, no random cell ids (nbformat 4.4 has none) and a fixed
kernelspec. ``--check`` exits 1 if the committed notebook is out of date.

Usage: python tools/make_nb.py [--check]
"""

import sys
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "notebook_src.py"
TARGET = ROOT / "chexpert_uncertainty.ipynb"
METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def build() -> str:
    """The notebook JSON generated from the source."""
    nb = jupytext.read(SOURCE, fmt="py:percent")
    nb.metadata = METADATA
    nb.nbformat_minor = 4  # no cell ids, so the file does not change between runs
    for cell in nb.cells:
        cell.pop("id", None)
    nbformat.validate(nb)
    assert not any(c.get("outputs") for c in nb.cells), "the generated notebook has outputs"
    return nbformat.writes(nb) + "\n"


if __name__ == "__main__":
    text = build()
    if sys.argv[1:] == ["--check"]:
        if not TARGET.is_file() or TARGET.read_text() != text:
            sys.exit(f"{TARGET.name} is out of date: run `make nb` and commit it")
        print(f"{TARGET.name} is up to date")
    else:
        TARGET.write_text(text)
        n_cells = text.count('"cell_type"')
        print(f"wrote {TARGET.name} ({n_cells} cells)")
