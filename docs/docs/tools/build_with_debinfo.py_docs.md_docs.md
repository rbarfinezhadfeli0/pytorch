# Documentation: `docs/tools/build_with_debinfo.py_docs.md`

## File Metadata

- **Path**: `docs/tools/build_with_debinfo.py_docs.md`
- **Size**: 7,132 bytes (6.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/build_with_debinfo.py`

## File Metadata

- **Path**: `tools/build_with_debinfo.py`
- **Size**: 4,134 bytes (4.04 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
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

"""Creates symlinks from build/lib to torch/lib"""    if not TORCH_LIB_DIR.exists():        raise RuntimeError(f"Can't create symlinks as {TORCH_LIB_DIR} does not exist")    if not BUILD_LIB_DIR.exists():        raise RuntimeError(f"Can't create symlinks as {BUILD_LIB_DIR} does not exist")    for torch_lib in TORCH_LIB_DIR.glob(f"*.{get_lib_extension()}"):

This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `check_output`, `parse_args`, `get_lib_extension`, `create_symlinks`, `has_build_ninja`, `is_devel_setup`, `create_build_plan`, `main`

**Key imports**: annotations, subprocess, sys, Path, Any, ArgumentParser, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `subprocess`
- `sys`
- `pathlib`: Path
- `typing`: Any
- `argparse`: ArgumentParser
- `torch`


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

Files in the same folder (`tools`):

- [`BUCK.bzl_docs.md`](./BUCK.bzl_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`render_junit.py_docs.md`](./render_junit.py_docs.md)
- [`extract_scripts.py_docs.md`](./extract_scripts.py_docs.md)
- [`nvcc_fix_deps.py_docs.md`](./nvcc_fix_deps.py_docs.md)
- [`update_masked_docs.py_docs.md`](./update_masked_docs.py_docs.md)
- [`optional_submodules.py_docs.md`](./optional_submodules.py_docs.md)
- [`gen_vulkan_spv.py_docs.md`](./gen_vulkan_spv.py_docs.md)
- [`generated_dirs.txt_docs.md`](./generated_dirs.txt_docs.md)
- [`build_libtorch.py_docs.md`](./build_libtorch.py_docs.md)


## Cross-References

- **File Documentation**: `build_with_debinfo.py_docs.md`
- **Keyword Index**: `build_with_debinfo.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/tools`):

- [`git_add_generated_dirs.sh_docs.md_docs.md`](./git_add_generated_dirs.sh_docs.md_docs.md)
- [`update_masked_docs.py_docs.md_docs.md`](./update_masked_docs.py_docs.md_docs.md)
- [`bazel.bzl_docs.md_docs.md`](./bazel.bzl_docs.md_docs.md)
- [`nightly_hotpatch.py_docs.md_docs.md`](./nightly_hotpatch.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`extract_scripts.py_docs.md_docs.md`](./extract_scripts.py_docs.md_docs.md)
- [`bazel.bzl_kw.md_docs.md`](./bazel.bzl_kw.md_docs.md)
- [`build_with_debinfo.py_kw.md_docs.md`](./build_with_debinfo.py_kw.md_docs.md)
- [`gen_flatbuffers.sh_kw.md_docs.md`](./gen_flatbuffers.sh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `build_with_debinfo.py_docs.md_docs.md`
- **Keyword Index**: `build_with_debinfo.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
