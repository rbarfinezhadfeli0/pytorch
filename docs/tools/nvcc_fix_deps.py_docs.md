# Documentation: `tools/nvcc_fix_deps.py`

## File Metadata

- **Path**: `tools/nvcc_fix_deps.py`
- **Size**: 3,414 bytes (3.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
"""Tool to fix the nvcc's dependency file output

Usage: python nvcc_fix_deps.py nvcc [nvcc args]...

This wraps nvcc to ensure that the dependency file created by nvcc with the
-MD flag always uses absolute paths. nvcc sometimes outputs relative paths,
which ninja interprets as an unresolved dependency, so it triggers a rebuild
of that file every time.

The easiest way to use this is to define:

CMAKE_CUDA_COMPILER_LAUNCHER="python;tools/nvcc_fix_deps.py;ccache"

"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TextIO


def resolve_include(path: Path, include_dirs: list[Path]) -> Path:
    for include_path in include_dirs:
        abs_path = include_path / path
        if abs_path.exists():
            return abs_path

    paths = "\n    ".join(str(d / path) for d in include_dirs)
    raise RuntimeError(
        f"""
ERROR: Failed to resolve dependency:
    {path}
Tried the following paths, but none existed:
    {paths}
"""
    )


def repair_depfile(depfile: TextIO, include_dirs: list[Path]) -> None:
    changes_made = False
    out = ""
    for line in depfile:
        if ":" in line:
            colon_pos = line.rfind(":")
            out += line[: colon_pos + 1]
            line = line[colon_pos + 1 :]

        line = line.strip()

        if line.endswith("\\"):
            end = " \\"
            line = line[:-1].strip()
        else:
            end = ""

        path = Path(line)
        if not path.is_absolute():
            changes_made = True
            path = resolve_include(path, include_dirs)
        out += f"    {path}{end}\n"

    # If any paths were changed, rewrite the entire file
    if changes_made:
        depfile.seek(0)
        depfile.write(out)
        depfile.truncate()


PRE_INCLUDE_ARGS = ["-include", "--pre-include"]
POST_INCLUDE_ARGS = ["-I", "--include-path", "-isystem", "--system-include"]


def extract_include_arg(include_dirs: list[Path], i: int, args: list[str]) -> None:
    def extract_one(name: str, i: int, args: list[str]) -> str | None:
        arg = args[i]
        if arg == name:
            return args[i + 1]
        if arg.startswith(name):
            arg = arg[len(name) :]
            return arg[1:] if arg[0] == "=" else arg
        return None

    for name in PRE_INCLUDE_ARGS:
        path = extract_one(name, i, args)
        if path is not None:
            include_dirs.insert(0, Path(path).resolve())
            return

    for name in POST_INCLUDE_ARGS:
        path = extract_one(name, i, args)
        if path is not None:
            include_dirs.append(Path(path).resolve())
            return


if __name__ == "__main__":
    ret = subprocess.run(
        sys.argv[1:], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr
    )

    depfile_path = None
    include_dirs = []

    # Parse only the nvcc arguments we care about
    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "-MF":
            depfile_path = Path(args[i + 1])
        elif arg == "-c":
            # Include the base path of the cuda file
            include_dirs.append(Path(args[i + 1]).resolve().parent)
        else:
            extract_include_arg(include_dirs, i, args)

    if depfile_path is not None and depfile_path.exists():
        with depfile_path.open("r+") as f:
            repair_depfile(f, include_dirs)

    sys.exit(ret.returncode)

```



## High-Level Overview

"""Tool to fix the nvcc's dependency file outputUsage: python nvcc_fix_deps.py nvcc [nvcc args]...This wraps nvcc to ensure that the dependency file created by nvcc with the-MD flag always uses absolute paths. nvcc sometimes outputs relative paths,which ninja interprets as an unresolved dependency, so it triggers a rebuildof that file every time.The easiest way to use this is to define:CMAKE_CUDA_COMPILER_LAUNCHER="python;tools/nvcc_fix_deps.py;ccache"

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `resolve_include`, `repair_depfile`, `extract_include_arg`, `extract_one`

**Key imports**: annotations, subprocess, sys, Path, TextIO


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
- `typing`: TextIO


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
- [`update_masked_docs.py_docs.md`](./update_masked_docs.py_docs.md)
- [`optional_submodules.py_docs.md`](./optional_submodules.py_docs.md)
- [`gen_vulkan_spv.py_docs.md`](./gen_vulkan_spv.py_docs.md)
- [`generated_dirs.txt_docs.md`](./generated_dirs.txt_docs.md)
- [`build_libtorch.py_docs.md`](./build_libtorch.py_docs.md)


## Cross-References

- **File Documentation**: `nvcc_fix_deps.py_docs.md`
- **Keyword Index**: `nvcc_fix_deps.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
