# Documentation: `tools/linter/adapters/pip_init.py`

## File Metadata

- **Path**: `tools/linter/adapters/pip_init.py`
- **Size**: 3,022 bytes (2.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
"""
Initializer script that installs stuff to pip.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time


def run_command(
    args: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(args, env=env, text=True, encoding="utf-8", check=True)
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="pip initializer")
    parser.add_argument(
        "packages",
        nargs="+",
        help="pip packages to install",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "--dry-run", help="do not install anything, just print what would be done."
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET if args.verbose else logging.DEBUG,
        stream=sys.stderr,
    )

    env: dict[str, str] = {
        **os.environ,
        "UV_PYTHON": sys.executable,
        "UV_PYTHON_DOWNLOADS": "never",
        "FORCE_COLOR": "1",
        "CLICOLOR_FORCE": "1",
    }
    uv_index = env.get("UV_INDEX", env.get("PIP_EXTRA_INDEX_URL"))
    if uv_index:
        env["UV_INDEX"] = uv_index

    # If we are in a global install, use `--user` to install so that you do not
    # need root access in order to initialize linters.
    #
    # However, `pip install --user` interacts poorly with virtualenvs (see:
    # https://bit.ly/3vD4kvl) and conda (see: https://bit.ly/3KG7ZfU). So in
    # these cases perform a regular installation.
    in_conda = env.get("CONDA_PREFIX") is not None
    in_virtualenv = env.get("VIRTUAL_ENV") is not None
    need_user_flag = not in_conda and not in_virtualenv

    uv: str | None = shutil.which("uv")
    is_uv_managed_python = "uv/python" in sys.base_prefix.replace("\\", "/")
    if uv and (is_uv_managed_python or not need_user_flag):
        pip_args = [uv, "pip", "install"]
    elif sys.executable:
        pip_args = [sys.executable, "-mpip", "install"]
    else:
        pip_args = ["pip3", "install"]

    if need_user_flag:
        pip_args.append("--user")

    pip_args.extend(args.packages)

    for package in args.packages:
        package_name, _, version = package.partition("=")
        if version == "":
            raise RuntimeError(
                f"Package {package_name} did not have a version specified. "
                "Please specify a version to produce a consistent linting experience."
            )

    dry_run = args.dry_run == "1"
    if dry_run:
        print(f"Would have run: {pip_args}")
        sys.exit(0)

    run_command(pip_args, env=env)


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Initializer script that installs stuff to pip.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `run_command`, `main`

**Key imports**: annotations, argparse, logging, os, shutil, subprocess, sys, time


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `logging`
- `os`
- `shutil`
- `subprocess`
- `sys`
- `time`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`tools/linter/adapters`):

- [`grep_linter.py_docs.md`](./grep_linter.py_docs.md)
- [`import_linter.py_docs.md`](./import_linter.py_docs.md)
- [`gha_linter.py_docs.md`](./gha_linter.py_docs.md)
- [`actionlint_linter.py_docs.md`](./actionlint_linter.py_docs.md)
- [`pyfmt_linter.py_docs.md`](./pyfmt_linter.py_docs.md)
- [`mypy_linter.py_docs.md`](./mypy_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `pip_init.py_docs.md`
- **Keyword Index**: `pip_init.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
