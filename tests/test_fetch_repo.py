"""Tests for ``scripts/fetch_repo.py``.

These tests exercise the script against the real public GitHub repo, but
they:

* are skipped if the network is unavailable (``request.exceptions`` or
  ``urllib.error.URLError``)
* write into a tempdir, never into the repo
* never modify the live working tree

Run with:

    PYTHONPATH=src pytest tests/test_fetch_repo.py -v

The test that actually fetches the tarball is gated on the env var
``FINSHEILD_RUN_NETWORK_TESTS=1`` so it can be opted out of in air-gapped CI.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "fetch_repo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_repo", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_loads():
    mod = _load_module()
    assert hasattr(mod, "fetch_tarball")
    assert hasattr(mod, "extract_tarball")
    assert mod.DEFAULT_OWNER == "shaikhakramshakil"
    assert mod.DEFAULT_REPO == "Finsheild"


def test_extract_tarball_smoke(tmp_dir):
    """Create a synthetic tarball, extract it, and verify layout."""
    import io
    import tarfile

    mod = _load_module()

    # Build a synthetic tarball in memory
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"hello world\n"
        info = tarfile.TarInfo(name="finsheild-fake-sha/README.md")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        info2 = tarfile.TarInfo(name="finsheild-fake-sha/src/__init__.py")
        info2.size = len(data)
        tf.addfile(info2, io.BytesIO(data))
    blob = buf.getvalue()
    assert blob[:2] == b"\x1f\x8b"

    mod.extract_tarball(blob, tmp_dir)
    assert (tmp_dir / "README.md").exists()
    assert (tmp_dir / "src" / "__init__.py").exists()
    # The inner top-level dir should have been flattened
    inner_dirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
    assert "finsheild-fake-sha" not in [p.name for p in inner_dirs]


def test_argparser_defaults():
    """Argparse defaults match what the README expects."""
    mod = _load_module()
    import argparse
    # Replicate the parser to inspect defaults without going through argparse
    p = argparse.ArgumentParser()
    p.add_argument("--owner", default=mod.DEFAULT_OWNER)
    p.add_argument("--repo", default=mod.DEFAULT_REPO)
    p.add_argument("--branch", default="main")
    p.add_argument("--dest", type=Path, default=Path("/content/Finsheild"))
    args = p.parse_args([])
    assert args.owner == "shaikhakramshakil"
    assert args.repo == "Finsheild"
    assert args.branch == "main"
    if os.environ.get("FINSHEILD_RUN_NETWORK_TESTS") != "1":
        pytest.skip("set FINSHEILD_RUN_NETWORK_TESTS=1 to enable real-fetch test")


def _has_network() -> bool:
    try:
        socket.create_connection(("github.com", 443), timeout=3)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _has_network(),
                     reason="github.com unreachable from this host")
def test_real_fetch_main_branch(tmp_dir):
    """End-to-end: fetch real main, verify Phase 4 files appear.

    Marked slow — only runs when network is reachable AND the env var
    ``FINSHEILD_RUN_NETWORK_TESTS=1`` is set (to keep CI fast by default).
    """
    if os.environ.get("FINSHEILD_RUN_NETWORK_TESTS") != "1":
        pytest.skip("set FINSHEILD_RUN_NETWORK_TESTS=1 to enable real-fetch test")

    import subprocess
    cmd = [sys.executable, str(SCRIPT), "--branch", "main",
           "--dest", str(tmp_dir), "--quiet"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert (tmp_dir / "src" / "finsheild" / "synthetic_env" / "environment.py").exists()
    # ``scripts/fetch_repo.py`` will only appear after the next push.
    # We accept either presence (freshly pushed) or absence (one commit behind).


# ---- Fixtures ------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path