"""Print one table with a row per run branch pushed to the repository (policy, who, progress).

The notebook pushes ``run_info.json`` and ``metrics_epoch.csv`` to the branch
``run/<policy>_s<seed>`` after every epoch, so this shows who is at which epoch while the runs
train. Smoke branches (``smoke/...``) are ignored. Needs git and the standard library only.

Usage: python tools/status.py [--remote URL_OR_PATH]   (default: ``origin`` of this clone)
"""

import argparse
import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ["run", "who", "epoch", "val_auroc", "img/s", "status"]


def git(*args: str, cwd: Path = ROOT, check: bool = True) -> str:
    """Run git and return stdout; with ``check`` a failure exits with git's message."""
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and done.returncode != 0:
        sys.exit(f"git {args[0]} failed:\n{done.stderr.strip()}")
    return done.stdout if done.returncode == 0 else ""


def fetch_run_branches(remote: str, scratch: Path) -> list[str]:
    """Fetch the tip of every ``run/*`` branch into a bare repository in ``scratch``."""
    git("init", "-q", "--bare", str(scratch))
    git("fetch", "-q", "--depth", "1", remote, "+refs/heads/run/*:refs/heads/run/*", cwd=scratch)
    return sorted(
        git("for-each-ref", "--format=%(refname:short)", "refs/heads/run", cwd=scratch).split()
    )


def run_row(scratch: Path, branch: str) -> list[str]:
    """Table row for ``run/<policy>_s<seed>``: who, latest epoch, its AUROC and speed, status."""
    name = branch.removeprefix("run/")

    def read(file: str) -> str:
        return git("show", f"{branch}:runs/{name}/{file}", cwd=scratch, check=False)

    try:
        info = json.loads(read("run_info.json") or "{}")
    except json.JSONDecodeError:
        info = {}
    epochs = list(csv.DictReader(io.StringIO(read("metrics_epoch.csv"))))
    last = epochs[-1] if epochs else None
    limits = info.get("run_limits") or {}
    total = limits.get("max_epochs") or (info.get("config") or {}).get("max_epochs") or "?"
    return [
        name,
        str(info.get("run_by", "?")),
        f"{last['epoch'] if last else 0}/{total}",
        f"{float(last['val_auroc']):.4f}" if last else "-",
        f"{float(last['images_per_sec']):.0f}" if last else "-",
        str(info.get("status", "unknown")),
    ]


def render(rows: list[list[str]]) -> str:
    """Left-aligned text table, header first."""
    table = [HEADER, *rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(HEADER))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in table
    ]
    return "\n".join(line.rstrip() for line in lines)


def main() -> None:
    """Fetch the run branches and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--remote", help="repository URL or path (default: origin of this clone)")
    remote = parser.parse_args().remote or git("remote", "get-url", "origin").strip()
    if Path(remote).exists():  # git runs in a scratch directory, so a local path must be absolute
        remote = str(Path(remote).resolve())
    with tempfile.TemporaryDirectory(prefix="status_") as tmp:
        branches = fetch_run_branches(remote, Path(tmp))
        if not branches:
            sys.exit(f"no run/* branches on {remote} yet")
        print(render([run_row(Path(tmp), branch) for branch in branches]))


if __name__ == "__main__":
    main()
