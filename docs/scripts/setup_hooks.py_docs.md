# Documentation: `scripts/setup_hooks.py`

## File Metadata

- **Path**: `scripts/setup_hooks.py`
- **Size**: 4,502 bytes (4.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This file handles **configuration or setup**.

## Original Source

```python
#!/usr/bin/env python3
"""
Bootstrap Git pre‑push hook with isolated virtual environment.

✓ Requires uv to be installed (fails if not available)
✓ Creates isolated venv in .git/hooks/linter/.venv/ for hook dependencies
✓ Installs lintrunner only in the isolated environment
✓ Creates direct git hook that bypasses pre-commit

Run this from the repo root (inside or outside any project venv):

    python scripts/setup_hooks.py

IMPORTANT: The generated git hook references scripts/lintrunner.py. If users checkout
branches that don't have this file, git push will fail with "No such file or directory".
Users would need to either:
1. Re-run the old setup_hooks.py from that branch, or
2. Manually delete .git/hooks/pre-push to disable hooks temporarily, or
3. Switch back to a branch with the new scripts/lintrunner.py
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path


# Add scripts directory to Python path so we can import lintrunner module
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Import shared functions from lintrunner module
from lintrunner import find_repo_root, get_hook_venv_path


# Restore sys.path to avoid affecting other imports
sys.path.pop(0)


# ───────────────────────────────────────────
# Helper utilities
# ───────────────────────────────────────────
def run(cmd: list[str], cwd: Path = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ensure_uv() -> None:
    if which("uv"):
        return

    sys.exit(
        "\n❌  uv is required but was not found on your PATH.\n"
        "    Please install uv first using the instructions at:\n"
        "    https://docs.astral.sh/uv/getting-started/installation/\n"
        "    Then rerun  python scripts/setup_hooks.py\n"
    )


if sys.platform.startswith("win"):
    print(
        "\n⚠️  Lintrunner is not supported on Windows, so there are no pre-push hooks to add. Exiting setup.\n"
    )
    sys.exit(0)

# ───────────────────────────────────────────
# 1. Setup isolated hook environment
# ───────────────────────────────────────────

ensure_uv()

# Find repo root and setup hook directory
repo_root = find_repo_root()
venv_dir = get_hook_venv_path()
hooks_dir = venv_dir.parent.parent  # Go from .git/hooks/linter/.venv to .git/hooks


print(f"Setting up isolated hook environment in {venv_dir}")

# Create isolated virtual environment for hooks
if venv_dir.exists():
    print("Removing existing hook venv...")
    shutil.rmtree(venv_dir)

run(["uv", "venv", str(venv_dir), "--python", "3.10"])

# Install lintrunner in the isolated environment
print("Installing lintrunner in isolated environment...")
run(
    ["uv", "pip", "install", "--python", str(venv_dir / "bin" / "python"), "lintrunner"]
)

# ───────────────────────────────────────────
# 2. Create direct git pre-push hook
# ───────────────────────────────────────────

pre_push_hook = hooks_dir / "pre-push"
python_exe = venv_dir / "bin" / "python"
lintrunner_script_path_quoted = shlex.quote(
    str(repo_root / "scripts" / "lintrunner.py")
)

hook_script = f"""#!/bin/bash
set -e

# Check if lintrunner script exists (user might be on older commit)
if [ ! -f {lintrunner_script_path_quoted} ]; then
    echo "⚠️  {lintrunner_script_path_quoted} not found - skipping linting (likely on an older commit)"
    exit 0
fi

# Run lintrunner wrapper using the isolated venv's Python
{shlex.quote(str(python_exe))} {lintrunner_script_path_quoted}
"""

print(f"Creating git pre-push hook at {pre_push_hook}")
pre_push_hook.write_text(hook_script)
pre_push_hook.chmod(0o755)  # Make executable

print(
    "\n✅  Isolated hook environment created and pre‑push hook is active.\n"
    "   Lintrunner will now run automatically on every `git push`.\n"
    f"   Hook dependencies are isolated in {venv_dir}\n"
)

```



## High-Level Overview

"""Bootstrap Git pre‑push hook with isolated virtual environment.✓ Requires uv to be installed (fails if not available)✓ Creates isolated venv in .git/hooks/linter/.venv/ for hook dependencies✓ Installs lintrunner only in the isolated environment✓ Creates direct git hook that bypasses pre-commitRun this from the repo root (inside or outside any project venv):    python scripts/setup_hooks.pyIMPORTANT: The generated git hook references scripts/lintrunner.py. If users checkoutbranches that don't have this file, git push will fail with "No such file or directory".Users would need to either:1. Re-run the old setup_hooks.py from that branch, or2. Manually delete .git/hooks/pre-push to disable hooks temporarily, or3. Switch back to a branch with the new scripts/lintrunner.py

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `run`, `which`, `ensure_uv`

**Key imports**: annotations, shlex, shutil, subprocess, sys, Path, lintrunner module, find_repo_root, get_hook_venv_path


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `shlex`
- `shutil`
- `subprocess`
- `sys`
- `pathlib`: Path
- `lintrunner module`
- `lintrunner`: find_repo_root, get_hook_venv_path


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
- [`README.md_docs.md`](./README.md_docs.md)
- [`lint_urls.sh_docs.md`](./lint_urls.sh_docs.md)
- [`lintrunner.py_docs.md`](./lintrunner.py_docs.md)
- [`install_triton_wheel.sh_docs.md`](./install_triton_wheel.sh_docs.md)


## Cross-References

- **File Documentation**: `setup_hooks.py_docs.md`
- **Keyword Index**: `setup_hooks.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
