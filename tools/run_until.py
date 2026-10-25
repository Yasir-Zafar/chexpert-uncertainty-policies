"""Run ``notebook_src.py`` up to, but not including, the part whose heading contains a marker.

Dev tool: the data-preparation cells (parts 0 to 2) behave the same in every mode, so checks
that only need them can run them without training anything or needing finished runs.

Usage: python tools/run_until.py "Part 3"   (or "1.5" to stop before the image-loading cells)
"""

import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "notebook_src.py"


def main() -> None:
    """Execute the notebook source before the cell holding the heading ``<marker>``.

    The marker names a part (``"Part 3"``) or a numbered section (``"1.5"``).
    """
    source = SOURCE.read_text()
    marker = sys.argv[1]
    heading = min(
        i for i in (source.find(f"# # {marker}"), source.find(f"# ## {marker}")) if i >= 0
    )
    cut = source.rindex("# %%", 0, heading)  # the markdown cell that holds the heading
    exec(compile(source[:cut], str(SOURCE), "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
