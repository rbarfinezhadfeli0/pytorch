# Documentation: `docs/tools/code_coverage/package/oss/init.py_docs.md`

## File Metadata

- **Path**: `docs/tools/code_coverage/package/oss/init.py_docs.md`
- **Size**: 7,503 bytes (7.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/code_coverage/package/oss/init.py`

## File Metadata

- **Path**: `tools/code_coverage/package/oss/init.py`
- **Size**: 5,141 bytes (5.02 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import argparse
import os
from typing import cast

from ..util.setting import (
    CompilerType,
    JSON_FOLDER_BASE_DIR,
    LOG_DIR,
    Option,
    Test,
    TestList,
    TestType,
)
from ..util.utils import (
    clean_up,
    create_folder,
    print_log,
    raise_no_test_found_exception,
    remove_file,
    remove_folder,
)
from ..util.utils_init import add_arguments_utils, create_folders, get_options
from .utils import (
    clean_up_gcda,
    detect_compiler_type,
    get_llvm_tool_path,
    get_oss_binary_folder,
    get_pytorch_folder,
)


BLOCKED_PYTHON_TESTS = {
    "run_test.py",
    "test_dataloader.py",
    "test_multiprocessing.py",
    "test_multiprocessing_spawn.py",
    "test_utils.py",
}


def initialization() -> tuple[Option, TestList, list[str]]:
    # create folder if not exists
    create_folders()
    # add arguments
    parser = argparse.ArgumentParser()
    parser = add_arguments_utils(parser)
    parser = add_arguments_oss(parser)
    # parse arguments
    (options, args_interested_folder, args_run_only, arg_clean) = parse_arguments(
        parser
    )
    # clean up
    if arg_clean:
        clean_up_gcda()
        clean_up()
    # get test lists
    test_list = get_test_list(args_run_only)
    # get interested folder -- final report will only over these folders
    interested_folders = empty_list_if_none(args_interested_folder)
    # print initialization information
    print_init_info()
    # remove last time's log
    remove_file(os.path.join(LOG_DIR, "log.txt"))
    return (options, test_list, interested_folders)


def add_arguments_oss(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--run-only",
        help="only run certain test(s), for example: atest test_nn.py.",
        nargs="*",
        default=None,
    )

    return parser


def parse_arguments(
    parser: argparse.ArgumentParser,
) -> tuple[Option, list[str] | None, list[str] | None, bool | None]:
    # parse args
    args = parser.parse_args()
    # get option
    options = get_options(args)
    return (options, args.interest_only, args.run_only, args.clean)


def get_test_list_by_type(run_only: list[str] | None, test_type: TestType) -> TestList:
    test_list: TestList = []
    binary_folder = get_oss_binary_folder(test_type)
    g = os.walk(binary_folder)
    for _, _, file_list in g:
        for file_name in file_list:
            if run_only is not None and file_name not in run_only:
                continue
            # target pattern in oss is used in printing report -- which tests we have run
            test: Test = Test(
                name=file_name,
                target_pattern=file_name,
                test_set="",
                test_type=test_type,
            )
            test_list.append(test)
    return test_list


def get_test_list(run_only: list[str] | None) -> TestList:
    test_list: TestList = []
    # add c++ test list
    test_list.extend(get_test_list_by_type(run_only, TestType.CPP))
    # add python test list
    py_run_only = get_python_run_only(run_only)
    test_list.extend(get_test_list_by_type(py_run_only, TestType.PY))

    # not find any test to run
    if not test_list:
        raise_no_test_found_exception(
            get_oss_binary_folder(TestType.CPP), get_oss_binary_folder(TestType.PY)
        )
    return test_list


def empty_list_if_none(arg_interested_folder: list[str] | None) -> list[str]:
    if arg_interested_folder is None:
        return []
    # if this argument is specified, just return itself
    return arg_interested_folder


def gcc_export_init() -> None:
    remove_folder(JSON_FOLDER_BASE_DIR)
    create_folder(JSON_FOLDER_BASE_DIR)


def get_python_run_only(args_run_only: list[str] | None) -> list[str]:
    # if user specifies run-only option
    if args_run_only:
        return args_run_only

    # if not specified, use default setting, different for gcc and clang
    if detect_compiler_type() == CompilerType.GCC:
        return ["run_test.py"]
    else:
        # for clang, some tests will result in too large intermediate files that can't be merged by llvm, we need to skip them
        run_only: list[str] = []
        binary_folder = get_oss_binary_folder(TestType.PY)
        g = os.walk(binary_folder)
        for _, _, file_list in g:
            for file_name in file_list:
                if file_name in BLOCKED_PYTHON_TESTS or not file_name.endswith(".py"):
                    continue
                run_only.append(file_name)
            # only run tests in the first-level folder in test/
            break
        return run_only


def print_init_info() -> None:
    print_log("pytorch folder: ", get_pytorch_folder())
    print_log("cpp test binaries folder: ", get_oss_binary_folder(TestType.CPP))
    print_log("python test scripts folder: ", get_oss_binary_folder(TestType.PY))
    print_log("compiler type: ", cast(CompilerType, detect_compiler_type()).value)
    print_log(
        "llvm tool folder (only for clang, if you are using gcov please ignore it): ",
        get_llvm_tool_path(),
    )

```



## High-Level Overview


This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `initialization`, `add_arguments_oss`, `parse_arguments`, `get_test_list_by_type`, `get_test_list`, `empty_list_if_none`, `gcc_export_init`, `get_python_run_only`, `print_init_info`

**Key imports**: annotations, argparse, os, cast, add_arguments_utils, create_folders, get_options


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/code_coverage/package/oss`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `os`
- `typing`: cast
- `..util.utils_init`: add_arguments_utils, create_folders, get_options


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`tools/code_coverage/package/oss`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`run.py_docs.md`](./run.py_docs.md)
- [`cov_json.py_docs.md`](./cov_json.py_docs.md)


## Cross-References

- **File Documentation**: `init.py_docs.md`
- **Keyword Index**: `init.py_kw.md`
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

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/tools/code_coverage/package/oss`):

- [`run.py_docs.md_docs.md`](./run.py_docs.md_docs.md)
- [`run.py_kw.md_docs.md`](./run.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`cov_json.py_kw.md_docs.md`](./cov_json.py_kw.md_docs.md)
- [`cov_json.py_docs.md_docs.md`](./cov_json.py_docs.md_docs.md)
- [`init.py_kw.md_docs.md`](./init.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`utils.py_kw.md_docs.md`](./utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `init.py_docs.md_docs.md`
- **Keyword Index**: `init.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
