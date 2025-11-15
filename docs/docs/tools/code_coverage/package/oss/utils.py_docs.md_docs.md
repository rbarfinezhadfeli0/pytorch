# Documentation: `docs/tools/code_coverage/package/oss/utils.py_docs.md`

## File Metadata

- **Path**: `docs/tools/code_coverage/package/oss/utils.py_docs.md`
- **Size**: 5,594 bytes (5.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/code_coverage/package/oss/utils.py`

## File Metadata

- **Path**: `tools/code_coverage/package/oss/utils.py`
- **Size**: 3,182 bytes (3.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import os
import subprocess

from ..util.setting import CompilerType, TestType, TOOLS_FOLDER
from ..util.utils import print_error, remove_file


def get_oss_binary_folder(test_type: TestType) -> str:
    assert test_type in {TestType.CPP, TestType.PY}
    # TODO: change the way we get binary file -- binary may not in build/bin ?
    return os.path.join(
        get_pytorch_folder(), "build/bin" if test_type == TestType.CPP else "test"
    )


def get_oss_shared_library() -> list[str]:
    lib_dir = os.path.join(get_pytorch_folder(), "build", "lib")
    return [
        os.path.join(lib_dir, lib)
        for lib in os.listdir(lib_dir)
        if lib.endswith(".dylib")
    ]


def get_oss_binary_file(test_name: str, test_type: TestType) -> str:
    assert test_type in {TestType.CPP, TestType.PY}
    binary_folder = get_oss_binary_folder(test_type)
    binary_file = os.path.join(binary_folder, test_name)
    if test_type == TestType.PY:
        # add python to the command so we can directly run the script by using binary_file variable
        binary_file = "python " + binary_file
    return binary_file


def get_llvm_tool_path() -> str:
    return os.environ.get(
        "LLVM_TOOL_PATH", "/usr/local/opt/llvm/bin"
    )  # set default as llvm path in dev server, on mac the default may be /usr/local/opt/llvm/bin


def get_pytorch_folder() -> str:
    # TOOLS_FOLDER in oss: pytorch/tools/code_coverage
    return os.path.abspath(
        os.environ.get("PYTORCH_FOLDER", os.path.dirname(os.path.dirname(TOOLS_FOLDER)))
    )


def detect_compiler_type() -> CompilerType | None:
    # check if user specifies the compiler type
    user_specify = os.environ.get("CXX", None)
    if user_specify:
        if user_specify in ["clang", "clang++"]:
            return CompilerType.CLANG
        elif user_specify in ["gcc", "g++"]:
            return CompilerType.GCC

        raise RuntimeError(f"User specified compiler is not valid {user_specify}")

    # auto detect
    auto_detect_result = subprocess.check_output(
        ["cc", "-v"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    if "clang" in auto_detect_result:
        return CompilerType.CLANG
    elif "gcc" in auto_detect_result:
        return CompilerType.GCC
    raise RuntimeError(f"Auto detected compiler is not valid {auto_detect_result}")


def clean_up_gcda() -> None:
    gcda_files = get_gcda_files()
    for item in gcda_files:
        remove_file(item)


def get_gcda_files() -> list[str]:
    folder_has_gcda = os.path.join(get_pytorch_folder(), "build")
    if os.path.isdir(folder_has_gcda):
        # TODO use glob
        # output = glob.glob(f"{folder_has_gcda}/**/*.gcda")
        output = subprocess.check_output(["find", folder_has_gcda, "-iname", "*.gcda"])
        return output.decode("utf-8").split("\n")
    else:
        return []


def run_oss_python_test(binary_file: str) -> None:
    # python test script
    try:
        subprocess.check_call(
            binary_file, shell=True, cwd=get_oss_binary_folder(TestType.PY)
        )
    except subprocess.CalledProcessError:
        print_error(f"Binary failed to run: {binary_file}")

```



## High-Level Overview


This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_oss_binary_folder`, `get_oss_shared_library`, `get_oss_binary_file`, `get_llvm_tool_path`, `get_pytorch_folder`, `detect_compiler_type`, `clean_up_gcda`, `get_gcda_files`, `run_oss_python_test`

**Key imports**: annotations, os, subprocess, CompilerType, TestType, TOOLS_FOLDER, print_error, remove_file


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/code_coverage/package/oss`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `os`
- `subprocess`
- `..util.setting`: CompilerType, TestType, TOOLS_FOLDER
- `..util.utils`: print_error, remove_file


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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

Files in the same folder (`tools/code_coverage/package/oss`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`run.py_docs.md`](./run.py_docs.md)
- [`init.py_docs.md`](./init.py_docs.md)
- [`cov_json.py_docs.md`](./cov_json.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/code_coverage/package/oss`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/code_coverage/package/oss`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/tools/code_coverage/package/oss`):

- [`init.py_docs.md_docs.md`](./init.py_docs.md_docs.md)
- [`run.py_docs.md_docs.md`](./run.py_docs.md_docs.md)
- [`run.py_kw.md_docs.md`](./run.py_kw.md_docs.md)
- [`cov_json.py_kw.md_docs.md`](./cov_json.py_kw.md_docs.md)
- [`cov_json.py_docs.md_docs.md`](./cov_json.py_docs.md_docs.md)
- [`init.py_kw.md_docs.md`](./init.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`utils.py_kw.md_docs.md`](./utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md_docs.md`
- **Keyword Index**: `utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
