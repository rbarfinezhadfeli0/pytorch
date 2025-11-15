# Documentation: `tools/setup_helpers/gen_version_header.py`

## File Metadata

- **Path**: `tools/setup_helpers/gen_version_header.py`
- **Size**: 2,667 bytes (2.60 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
# Ideally, there would be a way in Bazel to parse version.txt
# and use the version numbers from there as substitutions for
# an expand_template action. Since there isn't, this silly script exists.

from __future__ import annotations

import argparse
import os
from typing import cast


Version = tuple[int, int, int]


def parse_version(version: str) -> Version:
    """
    Parses a version string into (major, minor, patch) version numbers.

    Args:
      version: Full version number string, possibly including revision / commit hash.

    Returns:
      An int 3-tuple of (major, minor, patch) version numbers.
    """
    # Extract version number part (i.e. toss any revision / hash parts).
    version_number_str = version
    for i in range(len(version)):
        c = version[i]
        if not (c.isdigit() or c == "."):
            version_number_str = version[:i]
            break

    return cast(Version, tuple([int(n) for n in version_number_str.split(".")]))


def apply_replacements(replacements: dict[str, str], text: str) -> str:
    """
    Applies the given replacements within the text.

    Args:
      replacements (dict): Mapping of str -> str replacements.
      text (str): Text in which to make replacements.

    Returns:
      Text with replacements applied, if any.
    """
    for before, after in replacements.items():
        text = text.replace(before, after)
    return text


def main(args: argparse.Namespace) -> None:
    with open(args.version_path) as f:
        version = f.read().strip()
    (major, minor, patch) = parse_version(version)

    replacements = {
        "@TORCH_VERSION_MAJOR@": str(major),
        "@TORCH_VERSION_MINOR@": str(minor),
        "@TORCH_VERSION_PATCH@": str(patch),
    }

    # Create the output dir if it doesn't exist.
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.template_path) as input:
        with open(args.output_path, "w") as output:
            for line in input:
                output.write(apply_replacements(replacements, line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate version.h from version.h.in template",
    )
    parser.add_argument(
        "--template-path",
        required=True,
        help="Path to the template (i.e. version.h.in)",
    )
    parser.add_argument(
        "--version-path",
        required=True,
        help="Path to the file specifying the version",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for expanded template (i.e. version.h)",
    )
    args = parser.parse_args()
    main(args)

```



## High-Level Overview

"""    Parses a version string into (major, minor, patch) version numbers.    Args:      version: Full version number string, possibly including revision / commit hash.    Returns:      An int 3-tuple of (major, minor, patch) version numbers.

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `parse_version`, `apply_replacements`, `main`

**Key imports**: annotations, argparse, os, cast


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/setup_helpers`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `os`
- `typing`: cast


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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
- [`gen_unboxing.py_docs.md`](./gen_unboxing.py_docs.md)
- [`env.py_docs.md`](./env.py_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`cmake_utils.py_docs.md`](./cmake_utils.py_docs.md)
- [`cmake.py_docs.md`](./cmake.py_docs.md)
- [`gen.py_docs.md`](./gen.py_docs.md)


## Cross-References

- **File Documentation**: `gen_version_header.py_docs.md`
- **Keyword Index**: `gen_version_header.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
