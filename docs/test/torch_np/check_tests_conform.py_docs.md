# Documentation: `test/torch_np/check_tests_conform.py`

## File Metadata

- **Path**: `test/torch_np/check_tests_conform.py`
- **Size**: 2,310 bytes (2.26 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import sys
import textwrap
from pathlib import Path


def check(path):
    """Check a test file for common issues with pytest->pytorch conversion."""
    print(path.name)
    print("=" * len(path.name), "\n")

    src = path.read_text().split("\n")
    for num, line in enumerate(src):
        if is_comment(line):
            continue

        # module level test functions
        if line.startswith("def test"):
            report_violation(line, num, header="Module-level test function")

        # test classes must inherit from TestCase
        if line.startswith("class Test") and "TestCase" not in line:
            report_violation(
                line, num, header="Test class does not inherit from TestCase"
            )

        # last vestiges of pytest-specific stuff
        if "pytest.mark" in line:
            report_violation(line, num, header="pytest.mark.something")

        for part in ["pytest.xfail", "pytest.skip", "pytest.param"]:
            if part in line:
                report_violation(line, num, header=f"stray {part}")

        if textwrap.dedent(line).startswith("@parametrize"):
            # backtrack to check
            nn = num
            for nn in range(num, -1, -1):
                ln = src[nn]
                if "class Test" in ln:
                    # hack: large indent => likely an inner class
                    if len(ln) - len(ln.lstrip()) < 8:
                        break
            else:
                report_violation(line, num, "off-class parametrize")
            if not src[nn - 1].startswith("@instantiate_parametrized_tests"):
                report_violation(
                    line, num, f"missing instantiation of parametrized tests in {ln}?"
                )


def is_comment(line):
    return textwrap.dedent(line).startswith("#")


def report_violation(line, lineno, header):
    print(f">>>> line {lineno} : {header}\n {line}\n")


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 2:
        raise ValueError("Usage : python check_tests_conform path/to/file/or/dir")

    path = Path(argv[1])

    if path.is_dir():
        # run for all files in the directory (no subdirs)
        for this_path in path.glob("test*.py"):
            #   breakpoint()
            check(this_path)
    else:
        check(path)

```



## High-Level Overview

"""Check a test file for common issues with pytest->pytorch conversion."""    print(path.name)    print("=" * len(path.name), "\n")    src = path.read_text().split("\n")    for num, line in enumerate(src):        if is_comment(line):            continue        # module level test functions        if line.startswith("def test"):            report_violation(line, num, header="Module-level test function")        # test classes must inherit from TestCase        if line.startswith("class Test") and "TestCase" not in line:            report_violation(                line, num, header="Test class does not inherit from TestCase"            )        # last vestiges of pytest-specific stuff        if "pytest.mark" in line:            report_violation(line, num, header="pytest.mark.something")        for part in ["pytest.xfail", "pytest.skip", "pytest.param"]:            if part in line:                report_violation(line, num, header=f"stray {part}")        if textwrap.dedent(line).startswith("@parametrize"):            # backtrack to check            nn = num            for nn in range(num, -1, -1):                ln = src[nn]                if "class Test" in ln:                    # hack: large indent => likely an inner class                    if len(ln) - len(ln.lstrip()) < 8:                        break            else:                report_violation(line, num, "off-class parametrize")            if not src[nn - 1].startswith("@instantiate_parametrized_tests"):                report_violation(                    line, num, f"missing instantiation of parametrized tests in {ln}?"                )

This Python file contains 5 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `check`, `is_comment`, `report_violation`

**Key imports**: sys, textwrap, Path


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `textwrap`
- `pathlib`: Path


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
python test/torch_np/check_tests_conform.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/torch_np`):

- [`test_random.py_docs.md`](./test_random.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_ufuncs_basic.py_docs.md`](./test_ufuncs_basic.py_docs.md)
- [`test_function_base.py_docs.md`](./test_function_base.py_docs.md)
- [`test_basic.py_docs.md`](./test_basic.py_docs.md)
- [`test_binary_ufuncs.py_docs.md`](./test_binary_ufuncs.py_docs.md)
- [`test_indexing.py_docs.md`](./test_indexing.py_docs.md)
- [`test_ndarray_methods.py_docs.md`](./test_ndarray_methods.py_docs.md)
- [`conftest.py_docs.md`](./conftest.py_docs.md)
- [`test_reductions.py_docs.md`](./test_reductions.py_docs.md)


## Cross-References

- **File Documentation**: `check_tests_conform.py_docs.md`
- **Keyword Index**: `check_tests_conform.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
