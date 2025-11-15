# Documentation: build_with_debinfo.py

## File Metadata
- **Path**: `tools/build_with_debinfo.py`
- **Size**: 4134 bytes
- **Lines**: 125
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
#!/usr/bin/env python3
# Tool quickly rebuild one or two files with debug info
# Mimics following behavior:
# - touch file
# - ninja -j1 -v -n torch_python | sed -e 's/-O[23]/-g/g' -e 's#\[[0-9]\+\/[0-9]\+\] \+##' |sh
# - Copy libs from build/lib to torch/lib folder

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


PYTORCH_ROOTDIR = Path(__file__).resolve().parent.parent
TORCH_DIR = PYTORCH_ROOTDIR / "torch"
TORCH_LIB_DIR = TORCH_DIR / "lib"
BUILD_DIR = PYTORCH_ROOTDIR / "build"
BUILD_LIB_DIR = BUILD_DIR / "lib"


def check_output(args: list[str], cwd: str | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd).decode("utf-8")


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Incremental build PyTorch with debinfo")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("files", nargs="*")
    return parser.parse_args()


def get_lib_extension() -> str:
    if sys.platform == "linux":
        return "so"
    if sys.platform == "darwin":
        return "dylib"
    raise RuntimeError(f"Unsupported platform {sys.platform}")


def create_symlinks() -> None:
    """Creates symlinks from build/lib to torch/lib"""
    if not TORCH_LIB_DIR.exists():
        raise RuntimeError(f"Can't create symlinks as {TORCH_LIB_DIR} does not exist")
    if not BUILD_LIB_DIR.exists():
        raise RuntimeError(f"Can't create symlinks as {BUILD_LIB_DIR} does not exist")
    for torch_lib in TORCH_LIB_DIR.glob(f"*.{get_lib_extension()}"):
        if torch_lib.is_symlink():
            continue
        build_lib = BUILD_LIB_DIR / torch_lib.name
        if not build_lib.exists():
            raise RuntimeError(f"Can't find {build_lib} corresponding to {torch_lib}")
        torch_lib.unlink()
        torch_lib.symlink_to(build_lib)


def has_build_ninja() -> bool:
    return (BUILD_DIR / "build.ninja").exists()


def is_devel_setup() -> bool:
    output = check_output([sys.executable, "-c", "import torch;print(torch.__file__)"])
    return output.strip() == str(TORCH_DIR / "__init__.py")


def create_build_plan() -> list[tuple[str, str]]:
    output = check_output(
        ["ninja", "-j1", "-v", "-n", "torch_python"], cwd=str(BUILD_DIR)
    )
    rc = []
    for line in output.split("\n"):
        if not line.startswith("["):
            continue
        line = line.split("]", 1)[1].strip()
        if line.startswith(": &&") and line.endswith("&& :"):
            line = line[4:-4]
        line = line.replace("-O2", "-g").replace("-O3", "-g")
        # Build Metal shaders with debug information
        if "xcrun metal " in line and "-frecord-sources" not in line:
            line += " -frecord-sources -gline-tables-only"
        try:
            name = line.split("-o ", 1)[1].split(" ")[0]
            rc.append((name, line))
        except IndexError:
            print(f"Skipping {line} as it does not specify output file")
    return rc


def main() -> None:
    if sys.platform == "win32":
        print("Not supported on Windows yet")
        sys.exit(-95)
    if not is_devel_setup():
        print(
            "Not a devel setup of PyTorch, "
            "please run `python -m pip install --no-build-isolation -v -e .` first"
        )
        sys.exit(-1)
    if not has_build_ninja():
        print("Only ninja build system is supported at the moment")
        sys.exit(-1)
    args = parse_args()
    for file in args.files:
        if file is None:
            continue
        Path(file).touch()
    build_plan = create_build_plan()
    if len(build_plan) == 0:
        return print("Nothing to do")
    if len(build_plan) > 100:
        print("More than 100 items needs to be rebuild, run `ninja torch_python` first")
        sys.exit(-1)
    for idx, (name, cmd) in enumerate(build_plan):
        print(f"[{idx + 1} / {len(build_plan)}] Building {name}")
        if args.verbose:
            print(cmd)
        subprocess.check_call(["sh", "-c", cmd], cwd=BUILD_DIR)
    create_symlinks()


if __name__ == "__main__":
    main()

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Functions
This file defines 8 function(s): check_output, parse_args, get_lib_extension, create_symlinks, has_build_ninja, is_devel_setup, create_build_plan, main


## Key Components

The file contains 428 words across 125 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4134 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
