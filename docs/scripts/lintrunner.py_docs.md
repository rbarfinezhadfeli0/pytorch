# Documentation: `scripts/lintrunner.py`

## File Metadata

- **Path**: `scripts/lintrunner.py`
- **Size**: 5,851 bytes (5.71 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
"""
Wrapper script to run the isolated hook version of lintrunner.

This allows developers to easily run lintrunner (including with -a for auto-fixes)
using the same isolated environment that the pre-push hook uses, without having
to manually activate/deactivate virtual environments.

Usage:
    python scripts/lintrunner.py          # Check mode (same as git push)
    python scripts/lintrunner.py -a       # Auto-fix mode
    python scripts/lintrunner.py --help   # Show lintrunner help

This module also provides shared functionality for lintrunner hash management.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Find repository root using git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        sys.exit("❌ Not in a git repository")


def compute_file_hash(path: Path) -> str:
    """Returns SHA256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_stored_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text().strip()
    except Exception:
        return None


# Venv location - change this if the path changes
HOOK_VENV_PATH = ".git/hooks/linter/.venv"


def get_hook_venv_path() -> Path:
    """Get the path to the hook virtual environment."""
    repo_root = find_repo_root()
    return repo_root / HOOK_VENV_PATH


def find_hook_venv() -> Path:
    """Locate the isolated hook virtual environment."""
    venv_dir = get_hook_venv_path()

    if not venv_dir.exists():
        sys.exit(
            f"❌ Hook virtual environment not found at {venv_dir}\n"
            "   Please set this up by running: python scripts/setup_hooks.py"
        )

    return venv_dir


def check_lintrunner_installed(venv_dir: Path) -> None:
    """Check if lintrunner is installed in the given venv, exit if not."""
    result = subprocess.run(
        [
            "uv",
            "pip",
            "show",
            "--python",
            str(venv_dir / "bin" / "python"),
            "lintrunner",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        sys.exit(
            "❌ lintrunner is required but was not found in the hook environment. "
            "Please run `python scripts/setup_hooks.py` to reinstall."
        )
    print("✅ lintrunner is already installed")


def run_lintrunner(venv_dir: Path, args: list[str]) -> int:
    """Run lintrunner command in the specified venv and return exit code."""
    # Run lintrunner directly from the venv's bin directory with environment setup
    lintrunner_exe = venv_dir / "bin" / "lintrunner"
    cmd = [str(lintrunner_exe)] + args
    env = os.environ.copy()

    # PATH: Ensures lintrunner can find other tools in the venv (like python, pip, etc.)
    env["PATH"] = str(venv_dir / "bin") + os.pathsep + env.get("PATH", "")
    # VIRTUAL_ENV: Tells tools like pip_init.py that we're in a venv (prevents --user flag issues)
    env["VIRTUAL_ENV"] = str(venv_dir)

    # Note: Progress tends to be slightly garbled due to terminal control sequences,
    # but functionality and final results will be correct
    return subprocess.call(cmd, env=env)


def initialize_lintrunner_if_needed(venv_dir: Path) -> None:
    """Check if lintrunner needs initialization and run init if needed."""
    repo_root = find_repo_root()
    lintrunner_toml_path = repo_root / ".lintrunner.toml"
    initialized_hash_path = venv_dir / ".lintrunner_plugins_hash"

    if not lintrunner_toml_path.exists():
        print("⚠️ No .lintrunner.toml found. Skipping init.")
        return

    current_hash = compute_file_hash(lintrunner_toml_path)
    stored_hash = read_stored_hash(initialized_hash_path)

    if current_hash != stored_hash:
        print("🔁 Running `lintrunner init` …", file=sys.stderr)
        result = run_lintrunner(venv_dir, ["init"])
        if result != 0:
            sys.exit(f"❌ lintrunner init failed")
        initialized_hash_path.write_text(current_hash)
    else:
        print("✅ Lintrunner plugins already initialized and up to date.")


def main() -> None:
    """Run lintrunner in the isolated hook environment."""
    venv_dir = find_hook_venv()
    python_exe = venv_dir / "bin" / "python"

    if not python_exe.exists():
        sys.exit(f"❌ Python executable not found at {python_exe}")

    try:
        print(f"🐍 Virtual env being used: {venv_dir}", file=sys.stderr)

        # 1. Ensure lintrunner binary is available in the venv
        check_lintrunner_installed(venv_dir)

        # 2. Check for plugin updates and re-init if needed
        initialize_lintrunner_if_needed(venv_dir)

        # 3. Run lintrunner with any passed arguments and propagate its exit code
        args = sys.argv[1:]
        result = run_lintrunner(venv_dir, args)

        # If lintrunner failed and we're not already in auto-fix mode, suggest the wrapper
        if result != 0 and "-a" not in args:
            print(
                "\n💡 To auto-fix these issues, run: python scripts/lintrunner.py -a",
                file=sys.stderr,
            )

        sys.exit(result)

    except KeyboardInterrupt:
        print("\n  Lintrunner interrupted by user (KeyboardInterrupt)", file=sys.stderr)
        sys.exit(1)  # Tell git push to fail


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Wrapper script to run the isolated hook version of lintrunner.This allows developers to easily run lintrunner (including with -a for auto-fixes)using the same isolated environment that the pre-push hook uses, without havingto manually activate/deactivate virtual environments.Usage:    python scripts/lintrunner.py          # Check mode (same as git push)    python scripts/lintrunner.py -a       # Auto-fix mode    python scripts/lintrunner.py --help   # Show lintrunner helpThis module also provides shared functionality for lintrunner hash management.

This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `find_repo_root`, `compute_file_hash`, `read_stored_hash`, `get_hook_venv_path`, `find_hook_venv`, `check_lintrunner_installed`, `run_lintrunner`, `initialize_lintrunner_if_needed`, `main`

**Key imports**: annotations, hashlib, os, shlex, shutil, subprocess, sys, Path


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `hashlib`
- `os`
- `shlex`
- `shutil`
- `subprocess`
- `sys`
- `pathlib`: Path


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`scripts`):

- [`lint_xrefs.sh_docs.md`](./lint_xrefs.sh_docs.md)
- [`build_host_protoc.sh_docs.md`](./build_host_protoc.sh_docs.md)
- [`setup_hooks.py_docs.md`](./setup_hooks.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`lint_urls.sh_docs.md`](./lint_urls.sh_docs.md)
- [`install_triton_wheel.sh_docs.md`](./install_triton_wheel.sh_docs.md)


## Cross-References

- **File Documentation**: `lintrunner.py_docs.md`
- **Keyword Index**: `lintrunner.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
