"""Dev check for the GitHub push (cell C15); not part of the submission.

Everything runs against a local bare repository standing in for GitHub, with a fake token set.

Part 1 runs a smoke training of one policy end to end and inspects the branch it pushed: the
progress pushes and the final push are all there, only the run folder is in the tree, no
checkpoint, no token anywhere in the history.

Part 2 loads the notebook's helpers (results mode, nothing is trained) and checks the rules that
need a real run branch: a second claim of the same policy and seed is refused, a checkpoint or
the token in the run folder is never pushed, a failed progress push does not stop a run, a
failed final push does, smoke branches are overwritten and ``main`` is never touched.

Usage: python tools/pushtest.py [rules]   (``make pushtest`` runs it in the dev shell)
"""

import contextlib
import io
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAKE_ROOT = ROOT / "fake_data" / "chexpert"
FAKE_REPO = ROOT / "fake_data" / "repo"
REMOTE = ROOT / "fake_data" / "remote.git"
MOVED = ROOT / "fake_data" / "remote_moved.git"
TOKEN = "fake-pat-for-pushtest-0123456789abcdef"

ENV = {
    "DATA_ROOT": str(FAKE_ROOT),
    "REPO_DIR": str(FAKE_REPO),
    "MPLBACKEND": "Agg",
    "REMOTE_URL": str(REMOTE),
    "GITHUB_PAT": TOKEN,
    "GITHUB_REPO": "owner/fake",
    "GITHUB_USER": "tester",
}


def git(*args: str, cwd: Path = REMOTE) -> str:
    """Run git in the bare remote (or ``cwd``) and return stdout."""
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout


def check(condition: bool, message: str) -> None:
    """Fail the whole test with ``message``; print it when it passes."""
    if not condition:
        sys.exit(f"pushtest FAILED: {message}")
    print(f"  ok  {message}")


def raises(exc: type[Exception], fn, text: str = "") -> bool:
    """True if ``fn()`` raises ``exc`` and the message contains ``text``."""
    try:
        fn()
    except exc as err:
        return text in str(err)
    return False


def fresh_remote() -> None:
    """Replace the bare repository by an empty one."""
    shutil.rmtree(REMOTE, ignore_errors=True)
    subprocess.run(["git", "init", "-q", "--bare", str(REMOTE)], check=True)


def tree_files(rev: str) -> list[str]:
    """File paths in the remote tree at ``rev``."""
    return git("ls-tree", "-r", "--name-only", rev).split()


def commits(branch: str) -> list[str]:
    """Commits on ``branch``, oldest first."""
    return git("rev-list", "--reverse", branch).split()


def part1_end_to_end() -> None:
    """Smoke run with the push switched on, then inspect the pushed branch."""
    print("part 1: smoke run pushed to a local bare repository")
    fresh_remote()
    for stale in ("runs_smoke", "checkpoints"):
        shutil.rmtree(FAKE_REPO / stale, ignore_errors=True)
    env = {
        **os.environ,
        **ENV,
        "MODE": "train",
        "POLICY": "u_zeros",
        "SMOKE": "1",
        "SMOKE_TRAIN": "64",
        "SMOKE_VAL": "48",
        "SMOKE_QUAL": "16",
        "SMOKE_BATCH": "8",
    }
    done = subprocess.run(
        [sys.executable, "notebook_src.py"], cwd=ROOT, env=env, capture_output=True, text=True
    )
    if done.returncode != 0:
        print(done.stdout[-3000:], done.stderr[-3000:])
        sys.exit("pushtest FAILED: the smoke run failed")
    check(TOKEN not in done.stdout + done.stderr, "token absent from the notebook's output")

    branch, folder = "smoke/u_zeros_s42", "runs_smoke/u_zeros_s42"
    heads = git("for-each-ref", "--format=%(refname:short)", "refs/heads").split()
    check(heads == [branch], f"the only branch is {branch}")
    history = commits(branch)
    # started, epoch 1, epoch 2, trained, complete
    check(len(history) == 5, f"progress pushes and the final push all landed ({len(history)})")
    check(
        '"status": "running"' in git("show", f"{history[0]}:{folder}/run_info.json"),
        "the first push records the run as running",
    )
    epoch1 = git("show", f"{history[1]}:{folder}/metrics_epoch.csv").strip().splitlines()
    check(len(epoch1) == 2, "the first progress push holds the epoch-1 row")
    check('"status": "complete"' in git("show", f"{branch}:{folder}/run_info.json"), "complete")

    local = {f.name for f in (FAKE_REPO / folder).iterdir()}
    check(set(tree_files(branch)) == {f"{folder}/{n}" for n in local}, "tree = run folder only")
    check(not any(Path(f).suffix in {".pt", ".pth"} for f in tree_files(branch)), ".pt excluded")
    for commit in history:
        hits = subprocess.run(
            ["git", "grep", "-a", "-F", "-l", TOKEN, commit], cwd=REMOTE, capture_output=True
        )
        check(hits.returncode == 1, f"token absent from commit {commit[:8]}")


def part2_rules() -> None:
    """Load the notebook helpers and check the branch rules."""
    print("part 2: branch rules")
    fresh_remote()
    os.environ.update(ENV, MODE="results")  # the process ends after the test: no need to restore
    with contextlib.redirect_stdout(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "notebook_src.py"), run_name="notebook")
    time.sleep = lambda seconds: None  # the push retries must not slow the test

    pusher_class, scrub, header = ns["RunPusher"], ns["scrub"], ns["_AUTH_HEADER"]
    leaked = scrub(f"a {TOKEN} b {header} c")
    check(TOKEN not in leaked and header not in leaked, "scrub removes the token and the header")

    work = Path(tempfile.mkdtemp(prefix="pushtest_"))
    run = work / "runs" / "demo_s42"
    run.mkdir(parents=True)
    (run / "run_info.json").write_text('{"status": "running"}')
    logs: list[str] = []

    def make(name: str, smoke: bool = False):
        return pusher_class(run, work, smoke=smoke, who=name, log=logs.append)

    alice = make("alice")
    alice.claim()
    check(alice.branch == "run/demo_s42", "a real run goes to run/<policy>_s<seed>")
    (run / "metrics_epoch.csv").write_text("epoch,val_auroc\n1,0.7\n")
    check(alice.push("epoch 1"), "a progress push succeeds")
    branch = "run/demo_s42"
    check(len(commits(branch)) == 2, "each push is one more commit on the same branch")
    before = commits(branch)
    tip = before[-1]

    check(
        raises(RuntimeError, make("bob").claim, "already exists"),
        "a second claim of the same policy and seed is refused",
    )
    check(commits(branch)[-1] == tip, "the refused claim left the branch untouched")
    check(raises(AssertionError, lambda: alice.push("x", force=True)), "no force on a run branch")

    (run / "best.pt").write_bytes(b"weights")
    check(raises(AssertionError, lambda: alice.push("x", strict=True), ".pt"), "checkpoint refused")
    check(alice.push("x") is False, "a refused progress push only warns")
    (run / "best.pt").unlink()
    (run / "train.log").write_text(f"oops {TOKEN}\n")
    check(raises(AssertionError, lambda: alice.push("x", strict=True), "token"), "token refused")
    (run / "train.log").write_text("fine\n")

    REMOTE.rename(MOVED)  # the remote disappears: a network outage
    check(alice.push("during the outage") is False, "a failed progress push does not stop the run")
    check(
        raises(RuntimeError, lambda: alice.push("final", strict=True)), "a failed final push does"
    )
    MOVED.rename(REMOTE)
    check(alice.push("after the outage", strict=True), "pushing works again after the outage")
    check(commits(branch)[: len(before)] == before, "earlier commits are kept (fast-forward only)")
    check(alice.verify() == commits(branch)[-1][:10], "verify finds the remote at our commit")
    check(set(tree_files(branch)) == {f"runs/demo_s42/{f.name}" for f in run.iterdir()}, "tree")
    alice.close()

    one, two = make("carol", smoke=True), make("dave", smoke=True)
    one.claim()
    old = commits("smoke/demo_s42")[-1]
    two.claim()  # same smoke branch: overwritten, not refused
    new = commits("smoke/demo_s42")
    check(len(new) == 1 and new[-1] != old, "a smoke branch is overwritten by the next smoke run")
    heads = git("for-each-ref", "--format=%(refname:short)", "refs/heads").split()
    check("main" not in heads and sorted(heads) == ["run/demo_s42", "smoke/demo_s42"], "no main")

    pusher_class.claim.__globals__["GITHUB_PAT"] = None
    os.environ.pop("REMOTE_URL")
    off = make("erin")
    off.claim()
    check(off.work is None and not off.push("x"), "without secrets (off Kaggle) pushing is skipped")
    check(any("push skipped" in line for line in logs), "and the log says so")
    shutil.rmtree(work)


if __name__ == "__main__":
    os.chdir(ROOT)
    only_rules = sys.argv[1:] == ["rules"]  # `rules` skips the slow smoke run (part 1)
    if not only_rules:
        part1_end_to_end()
    part2_rules()
    print("pushtest ok")
