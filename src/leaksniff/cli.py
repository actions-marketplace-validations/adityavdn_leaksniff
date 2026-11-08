"""Command line: `leaksniff path/ [more paths]`."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .core import LEVELS, Finding, check_path

SKIP_DIRS = {".git", ".venv", "venv", "env", "site-packages", "node_modules", ".ipynb_checkpoints",
             "__pycache__", ".tox", "build", "dist"}
COLORS = {"error": "\033[31m", "warning": "\033[33m", "info": "\033[36m"}


def iter_files(paths: list[str]):
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            yield p
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(files):
                if name.endswith((".py", ".ipynb")):
                    yield Path(root) / name


def render(f: Finding, color: bool) -> str:
    tag = f"{f.code} {f.level}"
    if color:
        tag = f"{COLORS[f.level]}{tag}\033[0m"
    return f"{f.where()}: {tag}: {f.message}\n    fix: {f.hint}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="leaksniff", description="Find data leakage and evaluation "
                                 "mistakes in ML scripts and notebooks.")
    ap.add_argument("paths", nargs="*", default=["."], help="files or folders (.py, .ipynb)")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--min-level", choices=list(LEVELS), default="warning",
                    help="lowest level to show (default: warning)")
    ap.add_argument("--fail-on", choices=list(LEVELS), default="warning",
                    help="lowest level that makes the exit code 1 (default: warning)")
    ap.add_argument("--ignore", default="", help="comma-separated codes to skip, e.g. LS004,LS009")
    ap.add_argument("--version", action="version", version=f"leaksniff {__version__}")
    args = ap.parse_args(argv)

    ignore = {c.strip().upper() for c in args.ignore.split(",") if c.strip()}
    shown, failing, files = [], False, 0
    for path in iter_files(args.paths):
        files += 1
        for f in check_path(path):
            if f.code in ignore:
                continue
            if LEVELS[f.level] >= LEVELS[args.fail_on]:
                failing = True
            if LEVELS[f.level] >= LEVELS[args.min_level]:
                shown.append(f)

    if args.format == "json":
        print(json.dumps([f.to_dict() for f in shown]))
    else:
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        for f in shown:
            print(render(f, color))
        bad = len({f.path for f in shown})
        print(f"\n{len(shown)} problem(s) in {bad} of {files} file(s)." if shown
              else f"No leaks found in {files} file(s).")
    return 1 if failing else 0
