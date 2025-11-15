# Documentation: `docs/tools/setup_helpers/env.py_docs.md`

## File Metadata

- **Path**: `docs/tools/setup_helpers/env.py_docs.md`
- **Size**: 6,456 bytes (6.30 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/setup_helpers/env.py`

## File Metadata

- **Path**: `tools/setup_helpers/env.py`
- **Size**: 3,504 bytes (3.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import os
import platform
import struct
from itertools import chain
from typing import cast, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


CMAKE_MINIMUM_VERSION_STRING = "3.27"

IS_WINDOWS = platform.system() == "Windows"
IS_DARWIN = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

IS_64BIT = struct.calcsize("P") == 8

BUILD_DIR = "build"


def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def check_negative_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["OFF", "0", "NO", "FALSE", "N"]


def gather_paths(env_vars: Iterable[str]) -> list[str]:
    return list(chain(*(os.getenv(v, "").split(os.pathsep) for v in env_vars)))


def lib_paths_from_base(base_path: str) -> list[str]:
    return [os.path.join(base_path, s) for s in ["lib/x64", "lib", "lib64"]]


# We promised that CXXFLAGS should also be affected by CFLAGS
if "CFLAGS" in os.environ and "CXXFLAGS" not in os.environ:
    os.environ["CXXFLAGS"] = os.environ["CFLAGS"]


class BuildType:
    """Checks build type. The build type will be given in :attr:`cmake_build_type_env`. If :attr:`cmake_build_type_env`
    is ``None``, then the build type will be inferred from ``CMakeCache.txt``. If ``CMakeCache.txt`` does not exist,
    os.environ['CMAKE_BUILD_TYPE'] will be used.

    Args:
      cmake_build_type_env (str): The value of os.environ['CMAKE_BUILD_TYPE']. If None, the actual build type will be
        inferred.

    """

    def __init__(self, cmake_build_type_env: str | None = None) -> None:
        if cmake_build_type_env is not None:
            self.build_type_string = cmake_build_type_env
            return

        cmake_cache_txt = os.path.join(BUILD_DIR, "CMakeCache.txt")
        if os.path.isfile(cmake_cache_txt):
            # Found CMakeCache.txt. Use the build type specified in it.
            from .cmake_utils import get_cmake_cache_variables_from_file

            with open(cmake_cache_txt) as f:
                cmake_cache_vars = get_cmake_cache_variables_from_file(f)
            # Normally it is anti-pattern to determine build type from CMAKE_BUILD_TYPE because it is not used for
            # multi-configuration build tools, such as Visual Studio and XCode. But since we always communicate with
            # CMake using CMAKE_BUILD_TYPE from our Python scripts, this is OK here.
            self.build_type_string = cast(str, cmake_cache_vars["CMAKE_BUILD_TYPE"])
        else:
            self.build_type_string = os.environ.get("CMAKE_BUILD_TYPE", "Release")

    def is_debug(self) -> bool:
        "Checks Debug build."
        return self.build_type_string == "Debug"

    def is_rel_with_deb_info(self) -> bool:
        "Checks RelWithDebInfo build."
        return self.build_type_string == "RelWithDebInfo"

    def is_release(self) -> bool:
        "Checks Release build."
        return self.build_type_string == "Release"


# hotpatch environment variable 'CMAKE_BUILD_TYPE'. 'CMAKE_BUILD_TYPE' always prevails over DEBUG or REL_WITH_DEB_INFO.
if "CMAKE_BUILD_TYPE" not in os.environ:
    if check_env_flag("DEBUG"):
        os.environ["CMAKE_BUILD_TYPE"] = "Debug"
    elif check_env_flag("REL_WITH_DEB_INFO"):
        os.environ["CMAKE_BUILD_TYPE"] = "RelWithDebInfo"
    else:
        os.environ["CMAKE_BUILD_TYPE"] = "Release"

build_type = BuildType()

```



## High-Level Overview

"""Checks build type. The build type will be given in :attr:`cmake_build_type_env`. If :attr:`cmake_build_type_env`    is ``None``, then the build type will be inferred from ``CMakeCache.txt``. If ``CMakeCache.txt`` does not exist,    os.environ['CMAKE_BUILD_TYPE'] will be used.

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BuildType`

**Functions defined**: `check_env_flag`, `check_negative_env_flag`, `gather_paths`, `lib_paths_from_base`, `__init__`, `is_debug`, `is_rel_with_deb_info`, `is_release`

**Key imports**: annotations, os, platform, struct, chain, cast, TYPE_CHECKING, Iterable, get_cmake_cache_variables_from_file


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/setup_helpers`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `os`
- `platform`
- `struct`
- `itertools`: chain
- `typing`: cast, TYPE_CHECKING
- `collections.abc`: Iterable
- `.cmake_utils`: get_cmake_cache_variables_from_file


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/setup_helpers`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`generate_code.py_docs.md`](./generate_code.py_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`gen_version_header.py_docs.md`](./gen_version_header.py_docs.md)
- [`gen_unboxing.py_docs.md`](./gen_unboxing.py_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`cmake_utils.py_docs.md`](./cmake_utils.py_docs.md)
- [`cmake.py_docs.md`](./cmake.py_docs.md)
- [`gen.py_docs.md`](./gen.py_docs.md)


## Cross-References

- **File Documentation**: `env.py_docs.md`
- **Keyword Index**: `env.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/setup_helpers`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/setup_helpers`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/setup_helpers`):

- [`BUILD.bazel_docs.md_docs.md`](./BUILD.bazel_docs.md_docs.md)
- [`gen_version_header.py_kw.md_docs.md`](./gen_version_header.py_kw.md_docs.md)
- [`cmake_utils.py_docs.md_docs.md`](./cmake_utils.py_docs.md_docs.md)
- [`gen.py_kw.md_docs.md`](./gen.py_kw.md_docs.md)
- [`generate_code.py_kw.md_docs.md`](./generate_code.py_kw.md_docs.md)
- [`build.bzl_kw.md_docs.md`](./build.bzl_kw.md_docs.md)
- [`cmake_utils.py_kw.md_docs.md`](./cmake_utils.py_kw.md_docs.md)
- [`gen_unboxing.py_kw.md_docs.md`](./gen_unboxing.py_kw.md_docs.md)
- [`env.py_kw.md_docs.md`](./env.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `env.py_docs.md_docs.md`
- **Keyword Index**: `env.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
