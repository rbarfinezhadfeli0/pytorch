# Documentation: `docs/tools/build_pytorch_libs.py_docs.md`

## File Metadata

- **Path**: `docs/tools/build_pytorch_libs.py_docs.md`
- **Size**: 6,677 bytes (6.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/build_pytorch_libs.py`

## File Metadata

- **Path**: `tools/build_pytorch_libs.py`
- **Size**: 3,931 bytes (3.84 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import os
import platform
import subprocess

from .optional_submodules import checkout_nccl
from .setup_helpers.cmake import CMake, USE_NINJA
from .setup_helpers.env import (
    check_env_flag,
    check_negative_env_flag,
    IS_64BIT,
    IS_WINDOWS,
)


def _get_vc_env(vc_arch: str) -> dict[str, str]:
    try:
        from setuptools import distutils  # type: ignore[import,attr-defined]

        return distutils._msvccompiler._get_vc_env(vc_arch)  # type: ignore[no-any-return]
    except AttributeError:
        from setuptools._distutils import (
            _msvccompiler,  # type: ignore[import,attr-defined]
        )

        return _msvccompiler._get_vc_env(vc_arch)  # type: ignore[no-any-return,attr-defined]


def _overlay_windows_vcvars(env: dict[str, str]) -> dict[str, str]:
    vc_arch = "x64" if IS_64BIT else "x86"

    if platform.machine() == "ARM64":
        vc_arch = "x64_arm64"

        # First Win11 Windows on Arm build version that supports x64 emulation
        # is 10.0.22000.
        win11_1st_version = (10, 0, 22000)
        current_win_version = tuple(
            int(version_part) for version_part in platform.version().split(".")
        )
        if current_win_version < win11_1st_version:
            vc_arch = "x86_arm64"
            print(
                "Warning: 32-bit toolchain will be used, but 64-bit linker "
                "is recommended to avoid out-of-memory linker error!"
            )
            print(
                "Warning: Please consider upgrading to Win11, where x64 "
                "emulation is enabled!"
            )

    vc_env = _get_vc_env(vc_arch)
    # Keys in `_get_vc_env` are always lowercase.
    # We turn them into uppercase before overlaying vcvars
    # because OS environ keys are always uppercase on Windows.
    # https://stackoverflow.com/a/7797329
    vc_env = {k.upper(): v for k, v in vc_env.items()}
    for k, v in env.items():
        uk = k.upper()
        if uk not in vc_env:
            vc_env[uk] = v
    return vc_env


def _create_build_env() -> dict[str, str]:
    # XXX - our cmake file sometimes looks at the system environment
    # and not cmake flags!
    # you should NEVER add something to this list. It is bad practice to
    # have cmake read the environment
    my_env = os.environ.copy()
    if IS_WINDOWS and USE_NINJA:
        # When using Ninja under Windows, the gcc toolchain will be chosen as
        # default. But it should be set to MSVC as the user's first choice.
        my_env = _overlay_windows_vcvars(my_env)
        my_env.setdefault("CC", "cl")
        my_env.setdefault("CXX", "cl")
    return my_env


def build_pytorch(
    version: str | None,
    cmake_python_library: str | None,
    build_python: bool,
    rerun_cmake: bool,
    cmake_only: bool,
    cmake: CMake,
) -> None:
    my_env = _create_build_env()
    if (
        not check_negative_env_flag("USE_DISTRIBUTED")
        and not check_negative_env_flag("USE_CUDA")
        and not check_negative_env_flag("USE_NCCL")
        and not check_env_flag("USE_SYSTEM_NCCL")
    ):
        checkout_nccl()
    build_test = not check_negative_env_flag("BUILD_TEST")
    cmake.generate(
        version, cmake_python_library, build_python, build_test, my_env, rerun_cmake
    )
    if cmake_only:
        return
    build_custom_step = os.getenv("BUILD_CUSTOM_STEP")
    if build_custom_step:
        try:
            output = subprocess.check_output(
                build_custom_step,
                shell=True,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print("Command output:")
            print(output)
        except subprocess.CalledProcessError as e:
            print("Command failed with return code:", e.returncode)
            print("Output (stdout and stderr):")
            print(e.output)
            raise
    cmake.build(my_env)

```



## High-Level Overview


This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_get_vc_env`, `_overlay_windows_vcvars`, `_create_build_env`, `build_pytorch`

**Key imports**: annotations, os, platform, subprocess, checkout_nccl, CMake, USE_NINJA, distutils  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `os`
- `platform`
- `subprocess`
- `.optional_submodules`: checkout_nccl
- `.setup_helpers.cmake`: CMake, USE_NINJA
- `setuptools`: distutils  


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `build_pytorch_libs.py_docs.md`
- **Keyword Index**: `build_pytorch_libs.py_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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
- [`build_with_debinfo.py_docs.md_docs.md`](./build_with_debinfo.py_docs.md_docs.md)
- [`extract_scripts.py_docs.md_docs.md`](./extract_scripts.py_docs.md_docs.md)
- [`bazel.bzl_kw.md_docs.md`](./bazel.bzl_kw.md_docs.md)
- [`build_with_debinfo.py_kw.md_docs.md`](./build_with_debinfo.py_kw.md_docs.md)
- [`gen_flatbuffers.sh_kw.md_docs.md`](./gen_flatbuffers.sh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `build_pytorch_libs.py_docs.md_docs.md`
- **Keyword Index**: `build_pytorch_libs.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
