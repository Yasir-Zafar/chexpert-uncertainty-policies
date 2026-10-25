"""Run ``notebook_src.py`` up to, but not including, the part whose heading contains a marker.

Dev tool: the data-preparation cells (parts 0 to 2) behave the same in every mode, so checks
that only need them can run them without training anything or needing finished runs.

Usage: python tools/run_until.py "Part 3"
"""

import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "notebook_src.py"


def main() -> None:
    """Execute the notebook source before the cell holding the heading ``# # <marker>``."""
    source = SOURCE.read_text()
    heading = source.index(f"# # {sys.argv[1]}")
    cut = source.rindex("# %%", 0, heading)  # the markdown cell that holds the heading
    exec(compile(source[:cut], str(SOURCE), "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
