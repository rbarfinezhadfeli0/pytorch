# Documentation: `docs/tools/test/test_cmake.py_docs.md`

## File Metadata

- **Path**: `docs/tools/test/test_cmake.py_docs.md`
- **Size**: 8,312 bytes (8.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/test/test_cmake.py`

## File Metadata

- **Path**: `tools/test/test_cmake.py`
- **Size**: 3,879 bytes (3.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import contextlib
import os
import typing
import unittest
import unittest.mock

import tools.setup_helpers.cmake
import tools.setup_helpers.env  # noqa: F401 unused but resolves circular import


if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


T = typing.TypeVar("T")


class TestCMake(unittest.TestCase):
    @unittest.mock.patch("multiprocessing.cpu_count")
    def test_build_jobs(self, mock_cpu_count: unittest.mock.MagicMock) -> None:
        """Tests that the number of build jobs comes out correctly."""
        mock_cpu_count.return_value = 13
        cases = [
            # MAX_JOBS, USE_NINJA, IS_WINDOWS,         want
            (("8", True, False), ["-j", "8"]),  # noqa: E201,E241
            ((None, True, False), None),  # noqa: E201,E241
            (("7", False, False), ["-j", "7"]),  # noqa: E201,E241
            ((None, False, False), ["-j", "13"]),  # noqa: E201,E241
            (("6", True, True), ["-j", "6"]),  # noqa: E201,E241
            ((None, True, True), None),  # noqa: E201,E241
            (("11", False, True), ["-j", "11"]),  # noqa: E201,E241
            ((None, False, True), ["-j", "13"]),  # noqa: E201,E241
        ]
        for (max_jobs, use_ninja, is_windows), want in cases:
            with self.subTest(
                MAX_JOBS=max_jobs, USE_NINJA=use_ninja, IS_WINDOWS=is_windows
            ):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(env_var("MAX_JOBS", max_jobs))
                    stack.enter_context(
                        unittest.mock.patch.object(
                            tools.setup_helpers.cmake, "USE_NINJA", use_ninja
                        )
                    )
                    stack.enter_context(
                        unittest.mock.patch.object(
                            tools.setup_helpers.cmake, "IS_WINDOWS", is_windows
                        )
                    )

                    cmake = tools.setup_helpers.cmake.CMake()

                    with unittest.mock.patch.object(cmake, "run") as cmake_run:
                        cmake.build({})

                    cmake_run.assert_called_once()
                    (call,) = cmake_run.mock_calls
                    build_args, _ = call.args

                if want is None:
                    self.assertNotIn("-j", build_args)
                else:
                    self.assert_contains_sequence(build_args, want)

    @staticmethod
    def assert_contains_sequence(
        sequence: Sequence[T], subsequence: Sequence[T]
    ) -> None:
        """Raises an assertion if the subsequence is not contained in the sequence."""
        if len(subsequence) == 0:
            return  # all sequences contain the empty subsequence

        # Iterate over all windows of len(subsequence). Stop if the
        # window matches.
        for i in range(len(sequence) - len(subsequence) + 1):
            candidate = sequence[i : i + len(subsequence)]
            assert len(candidate) == len(subsequence)  # sanity check
            if candidate == subsequence:
                return  # found it
        raise AssertionError(f"{subsequence} not found in {sequence}")


@contextlib.contextmanager
def env_var(key: str, value: str | None) -> Iterator[None]:
    """Sets/clears an environment variable within a Python context."""
    # Get the previous value and then override it.
    previous_value = os.environ.get(key)
    set_env_var(key, value)
    try:
        yield
    finally:
        # Restore to previous value.
        set_env_var(key, previous_value)


def set_env_var(key: str, value: str | None) -> None:
    """Sets/clears an environment variable."""
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()

```



## High-Level Overview

"""Tests that the number of build jobs comes out correctly."""        mock_cpu_count.return_value = 13        cases = [            # MAX_JOBS, USE_NINJA, IS_WINDOWS,         want            (("8", True, False), ["-j", "8"]),  # noqa: E201,E241            ((None, True, False), None),  # noqa: E201,E241            (("7", False, False), ["-j", "7"]),  # noqa: E201,E241            ((None, False, False), ["-j", "13"]),  # noqa: E201,E241            (("6", True, True), ["-j", "6"]),  # noqa: E201,E241            ((None, True, True), None),  # noqa: E201,E241            (("11", False, True), ["-j", "11"]),  # noqa: E201,E241            ((None, False, True), ["-j", "13"]),  # noqa: E201,E241        ]        for (max_jobs, use_ninja, is_windows), want in cases:            with self.subTest(                MAX_JOBS=max_jobs, USE_NINJA=use_ninja, IS_WINDOWS=is_windows            ):                with contextlib.ExitStack() as stack:                    stack.enter_context(env_var("MAX_JOBS", max_jobs))                    stack.enter_context(                        unittest.mock.patch.object(                            tools.setup_helpers.cmake, "USE_NINJA", use_ninja                        )                    )                    stack.enter_context(                        unittest.mock.patch.object(                            tools.setup_helpers.cmake, "IS_WINDOWS", is_windows                        )

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCMake`

**Functions defined**: `test_build_jobs`, `assert_contains_sequence`, `env_var`, `set_env_var`

**Key imports**: annotations, contextlib, os, typing, unittest, unittest.mock, tools.setup_helpers.cmake, tools.setup_helpers.env  , if typing.TYPE_CHECKING, Iterator, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `contextlib`
- `os`
- `typing`
- `unittest`
- `unittest.mock`
- `tools.setup_helpers.cmake`
- `tools.setup_helpers.env  `
- `if typing.TYPE_CHECKING`
- `collections.abc`: Iterator, Sequence


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

This is a test file. Run it with:

```bash
python tools/test/test_cmake.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/test`):

- [`test_upload_stats_lib.py_docs.md`](./test_upload_stats_lib.py_docs.md)
- [`test_codegen.py_docs.md`](./test_codegen.py_docs.md)
- [`linter_test_case.py_docs.md`](./linter_test_case.py_docs.md)
- [`test_upload_gate.py_docs.md`](./test_upload_gate.py_docs.md)
- [`test_gen_backend_stubs.py_docs.md`](./test_gen_backend_stubs.py_docs.md)
- [`test_gb_registry_linter.py_docs.md`](./test_gb_registry_linter.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_set_linter.py_docs.md`](./test_set_linter.py_docs.md)
- [`gen_oplist_test.py_docs.md`](./gen_oplist_test.py_docs.md)
- [`test_upload_test_stats.py_docs.md`](./test_upload_test_stats.py_docs.md)


## Cross-References

- **File Documentation**: `test_cmake.py_docs.md`
- **Keyword Index**: `test_cmake.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/tools/test/test_cmake.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/test`):

- [`test_gen_backend_stubs.py_kw.md_docs.md`](./test_gen_backend_stubs.py_kw.md_docs.md)
- [`test_upload_stats_lib.py_kw.md_docs.md`](./test_upload_stats_lib.py_kw.md_docs.md)
- [`test_cmake.py_kw.md_docs.md`](./test_cmake.py_kw.md_docs.md)
- [`test_upload_test_stats.py_docs.md_docs.md`](./test_upload_test_stats.py_docs.md_docs.md)
- [`test_codegen_model.py_docs.md_docs.md`](./test_codegen_model.py_docs.md_docs.md)
- [`test_codegen.py_docs.md_docs.md`](./test_codegen.py_docs.md_docs.md)
- [`test_vulkan_codegen.py_kw.md_docs.md`](./test_vulkan_codegen.py_kw.md_docs.md)
- [`test_set_linter.py_docs.md_docs.md`](./test_set_linter.py_docs.md_docs.md)
- [`test_gb_registry_linter.py_kw.md_docs.md`](./test_gb_registry_linter.py_kw.md_docs.md)
- [`test_upload_test_stats.py_kw.md_docs.md`](./test_upload_test_stats.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_cmake.py_docs.md_docs.md`
- **Keyword Index**: `test_cmake.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
