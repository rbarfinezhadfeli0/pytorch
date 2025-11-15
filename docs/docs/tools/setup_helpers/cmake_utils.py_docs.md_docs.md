# Documentation: `docs/tools/setup_helpers/cmake_utils.py_docs.md`

## File Metadata

- **Path**: `docs/tools/setup_helpers/cmake_utils.py_docs.md`
- **Size**: 5,503 bytes (5.37 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/setup_helpers/cmake_utils.py`

## File Metadata

- **Path**: `tools/setup_helpers/cmake_utils.py`
- **Size**: 2,867 bytes (2.80 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
"""
This is refactored from cmake.py to avoid circular imports issue with env.py,
which calls get_cmake_cache_variables_from_file
"""

from __future__ import annotations

import re
from typing import IO, Optional, Union


CMakeValue = Optional[Union[bool, str]]


def convert_cmake_value_to_python_value(
    cmake_value: str, cmake_type: str
) -> CMakeValue:
    r"""Convert a CMake value in a string form to a Python value.

    Args:
      cmake_value (string): The CMake value in a string form (e.g., "ON", "OFF", "1").
      cmake_type (string): The CMake type of :attr:`cmake_value`.

    Returns:
      A Python value corresponding to :attr:`cmake_value` with type :attr:`cmake_type`.
    """

    cmake_type = cmake_type.upper()
    up_val = cmake_value.upper()
    if cmake_type == "BOOL":
        # https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html#genex:BOOL
        return not (
            up_val in ("FALSE", "OFF", "N", "NO", "0", "", "NOTFOUND")
            or up_val.endswith("-NOTFOUND")
        )
    elif cmake_type == "FILEPATH":
        if up_val.endswith("-NOTFOUND"):
            return None
        else:
            return cmake_value
    else:  # Directly return the cmake_value.
        return cmake_value


def get_cmake_cache_variables_from_file(
    cmake_cache_file: IO[str],
) -> dict[str, CMakeValue]:
    r"""Gets values in CMakeCache.txt into a dictionary.

    Args:
      cmake_cache_file: A CMakeCache.txt file object.
    Returns:
      dict: A ``dict`` containing the value of cached CMake variables.
    """

    results = {}
    for i, line in enumerate(cmake_cache_file, 1):
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            # Blank or comment line, skip
            continue

        # Almost any character can be part of variable name and value. As a practical matter, we assume the type must be
        # valid if it were a C variable name. It should match the following kinds of strings:
        #
        #   USE_CUDA:BOOL=ON
        #   "USE_CUDA":BOOL=ON
        #   USE_CUDA=ON
        #   USE_CUDA:=ON
        #   Intel(R) MKL-DNN_SOURCE_DIR:STATIC=/path/to/pytorch/third_party/ideep/mkl-dnn
        #   "OpenMP_COMPILE_RESULT_CXX_openmp:experimental":INTERNAL=FALSE
        matched = re.match(
            r'("?)(.+?)\1(?::\s*([a-zA-Z_-][a-zA-Z0-9_-]*)?)?\s*=\s*(.*)', line
        )
        if matched is None:  # Illegal line
            raise ValueError(f"Unexpected line {i} in {repr(cmake_cache_file)}: {line}")
        _, variable, type_, value = matched.groups()
        if type_ is None:
            type_ = ""
        if type_.upper() in ("INTERNAL", "STATIC"):
            # CMake internal variable, do not touch
            continue
        results[variable] = convert_cmake_value_to_python_value(value, type_)

    return results

```



## High-Level Overview

"""This is refactored from cmake.py to avoid circular imports issue with env.py,which calls get_cmake_cache_variables_from_file

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `convert_cmake_value_to_python_value`, `get_cmake_cache_variables_from_file`

**Key imports**: annotations, re, IO, Optional, Union


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/setup_helpers`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `re`
- `typing`: IO, Optional, Union


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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
- [`env.py_docs.md`](./env.py_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`cmake.py_docs.md`](./cmake.py_docs.md)
- [`gen.py_docs.md`](./gen.py_docs.md)


## Cross-References

- **File Documentation**: `cmake_utils.py_docs.md`
- **Keyword Index**: `cmake_utils.py_kw.md`
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

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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
- [`gen.py_kw.md_docs.md`](./gen.py_kw.md_docs.md)
- [`generate_code.py_kw.md_docs.md`](./generate_code.py_kw.md_docs.md)
- [`build.bzl_kw.md_docs.md`](./build.bzl_kw.md_docs.md)
- [`cmake_utils.py_kw.md_docs.md`](./cmake_utils.py_kw.md_docs.md)
- [`gen_unboxing.py_kw.md_docs.md`](./gen_unboxing.py_kw.md_docs.md)
- [`env.py_docs.md_docs.md`](./env.py_docs.md_docs.md)
- [`env.py_kw.md_docs.md`](./env.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cmake_utils.py_docs.md_docs.md`
- **Keyword Index**: `cmake_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
