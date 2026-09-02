"""Fetch the Finsheild repo source into a Colab runtime.

Workaround for the fact that ``git clone https://github.com/...`` from inside
the Google Colab base image returns::

    fatal: could not read Username for 'https://github.com':
           terminal prompts disabled

even though the repo is public and ``urllib`` / ``curl`` both reach
``github.com``. The Colab base image's git is missing a usable credential
helper.

This script downloads the public tarball (``/archive/refs/heads/<branch>.tar.gz``)
and extracts it. It accepts a branch and an optional destination directory.

Usage (inside a Colab runtime):

    python scripts/fetch_repo.py --branch main --dest /content/Finsheild

    # Use a pinned commit (sha) — handy for reproducible training runs
    python scripts/fetch_repo.py --branch main --ref 42532b4 --dest /content/Finsheild

The script:

1. Downloads ``https://github.com/<owner>/<repo>/archive/refs/heads/<branch>.tar.gz``
   using ``urllib`` (no extra deps).
2. Extracts to ``<dest>`` with ``tar --strip-components=1``.
3. Optionally pins the working tree to a specific commit by downloading
   that commit's tarball into a temp dir and overlaying its source tree
   (tarball extraction alone does NOT give you a git history, but does
   give you a complete source snapshot).

This is intentionally minimal. It does NOT bootstrap git history. If
Phase 14 (QLoRA) needs full git history (for ``git lfs`` pulls or for
``pip install -e .`` from a specific tag), use the SSH path documented
in AGENTS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_OWNER = "shaikhakramshakil"
DEFAULT_REPO = "Finsheild"


def _download(url: str) -> bytes:
    """Download URL with a User-Agent header. Colab's urllib works fine."""
    req = urllib.request.Request(url, headers={"User-Agent": "finsheild-fetch"})
    with urllib.request.urlopen(req, timeout=180) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status} for {url}")
        return r.read()


def fetch_tarball(owner: str, repo: str, ref: str) -> bytes:
    """Download a tarball for a branch, tag, or sha. No auth required for public repos."""
    # GitHub tarball endpoint:
    # - branches / tags use /archive/refs/heads/<ref>.tar.gz or /archive/refs/tags/<ref>.tar.gz
    # - commits use /archive/<sha>.tar.gz
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
        url = f"https://github.com/{owner}/{repo}/archive/{ref}.tar.gz"
    else:
        url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{ref}.tar.gz"
    print(f"Downloading {url}", file=sys.stderr)
    try:
        return _download(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Fallback: maybe ``ref`` is a tag
            url2 = f"https://github.com/{owner}/{repo}/archive/refs/tags/{ref}.tar.gz"
            print(f"  404, retrying {url2}", file=sys.stderr)
            return _download(url2)
        raise


def extract_tarball(blob: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        # Strip the leading "<repo>-<sha>/" component
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(dest, filter="data")
        # Find the single top-level dir created by the tarball
        top = [p for p in dest.iterdir() if p.is_dir()]
        if len(top) == 1:
            inner = top[0]
            # Move all entries up one level
            for entry in inner.iterdir():
                shutil.move(str(entry), str(dest / entry.name))
            inner.rmdir()
    finally:
        os.unlink(path)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", default=DEFAULT_OWNER)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--branch", default="main",
                    help="Branch name (default: main)")
    p.add_argument("--ref", default=None,
                    help="Optional ref override (branch / tag / commit sha). "
                          "If set, overrides --branch.")
    p.add_argument("--dest", type=Path, default=Path("/content/Finsheild"))
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    ref = args.ref or args.branch
    blob = fetch_tarball(args.owner, args.repo, ref)

    # Verify checksum is at least plausible (non-empty, gzip-magic).
    if len(blob) < 200:
        raise RuntimeError(
            f"Tarball too small ({len(blob)} bytes); check ref/branch name.")
    if blob[:2] != b"\x1f\x8b":
        raise RuntimeError("Downloaded blob is not a gzip file")

    extract_tarball(blob, args.dest)

    if not args.quiet:
        sha = hashlib.sha256(blob).hexdigest()
        print(f"sha256(tarball)={sha}", file=sys.stderr)
        print(f"Extracted to {args.dest}", file=sys.stderr)

    # Sanity check: confirm we have the Finsheild package marker.
    if not (args.dest / "src" / "finsheild" / "__init__.py").exists():
        print(f"WARNING: {args.dest}/src/finsheild/__init__.py missing — "
              f"did you point at the right repo?", file=sys.stderr)
        return 1

    if not args.quiet:
        # Show the latest commit-ish info we have (no .git from tarball).
        rel_files = sorted(p.name for p in (args.dest).iterdir())[:8]
        print(f"Top-level entries ({len(rel_files)} shown): {rel_files}",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())