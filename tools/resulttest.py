"""Dev check for results mode, cell 4.1 (fetch, inventory, consistency); not part of the submission.

Needs the fake runs: run ``make fakeruns`` once (about half an hour on a CPU) to put 7 policies x
3 seeds on a local bare repository standing in for GitHub; ``make resulttest`` then runs this.

Part 1 loads the notebook in results mode against those runs and checks the seed choice for every
``N_SEEDS``, the inventory and the protocol table. Part 2 tampers with copies of a run (settings,
split, data, files, test order) and checks that each change is reported. Part 3 deletes a run
branch from a copy of the remote and checks that an explicit ``N_SEEDS`` stops with the list of
what is missing while ``auto`` falls back to the complete layers.

Usage: python tools/resulttest.py
"""

import contextlib
import dataclasses
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REMOTE = ROOT / "fake_data" / "results_remote.git"
ENV = {
    "DATA_ROOT": str(ROOT / "fake_data" / "chexpert"),
    "REPO_DIR": str(ROOT / "fake_data" / "resrepo"),
    "MPLBACKEND": "Agg",
    "SMOKE": "1",
    "MODE": "results",
    "N_SEEDS": "auto",
    "REMOTE_URL": str(REMOTE),
}


def check(condition: bool, message: str) -> None:
    """Fail the whole test with ``message``; print it when it passes."""
    if not condition:
        sys.exit(f"resulttest FAILED: {message}")
    print(f"  ok  {message}")


def load_notebook() -> dict:
    """Run the notebook source in results mode and return its namespace."""
    os.environ.update(ENV)
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(ROOT / "notebook_src.py"), run_name="notebook")


def part1_selection(ns: dict) -> None:
    """Seed choice, inventory and protocol on the complete set of fake runs."""
    print("part 1: the complete set")
    check(ns["RESULT_SEEDS"] == [42, 43, 44], "auto picks all three complete seed layers")
    check(len(ns["RUNS"]) == 18 and ns["EXTRA_RUN"] is not None, "18 runs plus the extra run")
    choose, runs = ns["choose_seeds"], ns["ALL_RUNS"]
    check([choose(runs, n) for n in "123"] == [[42], [42, 43], [42, 43, 44]], "N_SEEDS 1, 2, 3")
    inventory = ns["inventory"]
    check(len(inventory) == 19 and set(inventory["status"]) == {"complete"}, "inventory: 19 runs")
    check((inventory["used"] == "yes").all(), "all of them are used with auto")
    check(inventory["run"].is_unique and not inventory["problems"].any(), "no problems found")
    tables = Path(ENV["REPO_DIR"]) / "results" / "tables"
    check((tables / "run_inventory.csv").is_file(), "inventory table written")
    protocol = dict(zip(ns["protocol"]["setting"], ns["protocol"]["value"], strict=True))
    check("DenseNet-121" in protocol["backbone"] and protocol["hardware"] == "cpu", "protocol")

    layer_44_broken = dataclasses.replace(
        runs[("u_ones", 44)], info={**runs[("u_ones", 44)].info, "status": "running"}
    )
    partial = {**runs, ("u_ones", 44): layer_44_broken}
    check(choose(partial, "auto") == [42, 43], "auto drops a layer with an unfinished run")
    try:
        choose(partial, "3")
    except RuntimeError as err:
        check("u_ones_s44: running" in str(err), "an explicit count stops and names the run")
    else:
        check(False, "an explicit count must stop when a run is unfinished")


def tampered(ns: dict, edit) -> list[str]:
    """Problems ``check_run`` reports for a copy of u_zeros_mc seed 43 after ``edit(folder)``."""
    original = ns["ALL_RUNS"][("u_zeros_mc", 43)]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "run"
        shutil.copytree(original.dir, folder)
        edit(folder)
        copy = dataclasses.replace(
            original, dir=folder, info=json.loads((folder / "run_info.json").read_text())
        )
        return ns["check_run"](copy, ns["TEST"]["path"].tolist())


def edit_info(**changes):
    """An edit that sets fields of ``run_info.json``."""

    def edit(folder: Path) -> None:
        path = folder / "run_info.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))

    return edit


def part2_tampering(ns: dict) -> None:
    """Every kind of inconsistency is reported, and an untouched copy is clean."""
    print("part 2: tampering with a copy of one run")
    check(tampered(ns, lambda folder: None) == [], "an untouched copy has no problems")
    config = dict(ns["ALL_RUNS"][("u_zeros_mc", 43)].info["config"], lr=0.5)
    cases = [
        (edit_info(config=config), "CONFIG differs"),
        (edit_info(code_version="v0"), "code version"),
        (edit_info(split_sha256="0" * 64), "split"),
        (edit_info(data_fingerprint="0" * 64), "dataset files"),
        (edit_info(smoke=False), "smoke flag"),
        (edit_info(policy="u_ones"), "policy or seed"),
        (edit_info(seed=42), "policy or seed"),
        (lambda folder: (folder / "extra.csv").write_text("x"), "files differ"),
        (lambda folder: (folder / "mc_val.npz").unlink(), "files differ"),
    ]
    for edit, expected in cases:
        problems = tampered(ns, edit)
        check(any(expected in text for text in problems), f"reported: {expected}")

    def reverse_test_order(folder: Path) -> None:
        with np.load(folder / "test_preds.npz") as data:
            arrays = {name: data[name][::-1] for name in data.files}
        np.savez_compressed(folder / "test_preds.npz", **arrays)

    check(any("test images" in t for t in tampered(ns, reverse_test_order)), "reported: test order")


def part3_missing_run() -> None:
    """Delete one run branch from a copy of the remote and run the notebook as a script."""
    print("part 3: a missing run")
    with tempfile.TemporaryDirectory() as tmp:
        remote = Path(tmp) / "remote.git"
        shutil.copytree(REMOTE, remote)
        subprocess.run(
            ["git", "-C", str(remote), "branch", "-D", "smoke/u_ones_s44"],
            check=True,
            capture_output=True,
        )

        def run(n_seeds: str) -> subprocess.CompletedProcess:
            env = {**os.environ, **ENV, "REMOTE_URL": str(remote), "N_SEEDS": n_seeds}
            return subprocess.run(
                [sys.executable, str(ROOT / "notebook_src.py")],
                env=env,
                capture_output=True,
                text=True,
            )

        strict = run("3")
        check(strict.returncode != 0 and "u_ones_s44: missing" in strict.stderr, "N_SEEDS=3 stops")
        check("missing" in strict.stdout, "the inventory is printed before it stops")
        auto = run("auto")
        check(
            auto.returncode == 0 and "analysing seeds [42, 43]" in auto.stdout, "auto uses 42, 43"
        )


if __name__ == "__main__":
    os.chdir(ROOT)
    heads = subprocess.run(
        ["git", "-C", str(REMOTE), "for-each-ref", "--format=%(refname)", "refs/heads"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len(heads) < 21:
        sys.exit(f"run `make fakeruns` first: the fake remote holds {len(heads)} of 21 runs")
    namespace = load_notebook()
    part1_selection(namespace)
    part2_tampering(namespace)
    part3_missing_run()
    print("resulttest ok")
