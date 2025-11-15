# Documentation: `.ci/lumen_cli/cli/lib/common/utils.py`

## File Metadata

- **Path**: `.ci/lumen_cli/cli/lib/common/utils.py`
- **Size**: 3,644 bytes (3.56 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
General Utility helpers for CLI tasks.
"""

import logging
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def run_command(
    cmd: str,
    use_shell: bool = False,
    log_cmd: bool = True,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    check: bool = True,
) -> int:
    """Run a command with optional shell execution."""
    if use_shell:
        args = cmd
        log_prefix = "[shell]"
        executable = "/bin/bash"
    else:
        args = shlex.split(cmd)
        log_prefix = "[cmd]"
        executable = None

    if log_cmd:
        display_cmd = cmd if use_shell else " ".join(args)
        logger.info("%s %s", log_prefix, display_cmd)

    run_env = {**os.environ, **(env or {})}

    proc = subprocess.run(
        args,
        shell=use_shell,
        executable=executable,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=cwd,
        env=run_env,
        check=False,
    )

    if check and proc.returncode != 0:
        logger.error(
            "%s Command failed (exit %s): %s", log_prefix, proc.returncode, cmd
        )
        raise subprocess.CalledProcessError(
            proc.returncode, args if not use_shell else cmd
        )

    return proc.returncode


def str2bool(value: Optional[str]) -> bool:
    """Convert environment variables to boolean values."""
    if not value:
        return False
    if not isinstance(value, str):
        raise ValueError(
            f"Expected a string value for boolean conversion, got {type(value)}"
        )
    value = value.strip().lower()

    true_value_set = {"1", "true", "t", "yes", "y", "on", "enable", "enabled", "found"}
    false_value_set = {"0", "false", "f", "no", "n", "off", "disable"}

    if value in true_value_set:
        return True
    if value in false_value_set:
        return False
    raise ValueError(f"Invalid string value for boolean conversion: {value}")


@contextmanager
def temp_environ(updates: dict[str, str]):
    """
    Temporarily set environment variables and restore them after the block.
    Args:
        updates: Dict of environment variables to set.
    """
    missing = object()
    old: dict[str, str | object] = {k: os.environ.get(k, missing) for k in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for k, v in old.items():
            if v is missing:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v  # type: ignore[arg-type]


@contextmanager
def working_directory(path: str):
    """
    Temporarily change the working directory inside a context.
    """
    if not path:
        # No-op context
        yield
        return
    prev_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)


def get_wheels(
    output_dir: Path,
    max_depth: Optional[int] = None,
) -> list[str]:
    """Return a list of wheels found in the given output directory."""
    root = Path(output_dir)
    if not root.exists():
        return []
    items = []
    for dirpath, _, filenames in os.walk(root):
        depth = Path(dirpath).relative_to(root).parts
        if max_depth is not None and len(depth) > max_depth:
            continue
        for fname in sorted(filenames):
            if fname.endswith(".whl"):
                pkg = fname.split("-")[0]
                relpath = str((Path(dirpath) / fname).relative_to(root))
                items.append({"pkg": pkg, "relpath": relpath})
    return items

```



## High-Level Overview

"""General Utility helpers for CLI tasks.

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `run_command`, `str2bool`, `temp_environ`, `working_directory`, `get_wheels`

**Key imports**: logging, os, shlex, subprocess, sys, contextmanager, Path, Optional


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/lumen_cli/cli/lib/common`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `os`
- `shlex`
- `subprocess`
- `sys`
- `contextlib`: contextmanager
- `pathlib`: Path
- `typing`: Optional


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

Files in the same folder (`.ci/lumen_cli/cli/lib/common`):

- [`gh_summary.py_docs.md`](./gh_summary.py_docs.md)
- [`docker_helper.py_docs.md`](./docker_helper.py_docs.md)
- [`git_helper.py_docs.md`](./git_helper.py_docs.md)
- [`envs_helper.py_docs.md`](./envs_helper.py_docs.md)
- [`pip_helper.py_docs.md`](./pip_helper.py_docs.md)
- [`cli_helper.py_docs.md`](./cli_helper.py_docs.md)
- [`logger.py_docs.md`](./logger.py_docs.md)
- [`path_helper.py_docs.md`](./path_helper.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
