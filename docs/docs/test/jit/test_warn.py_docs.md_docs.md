# Documentation: `docs/test/jit/test_warn.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_warn.py_docs.md`
- **Size**: 6,825 bytes (6.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_warn.py`

## File Metadata

- **Path**: `test/jit/test_warn.py`
- **Size**: 3,701 bytes (3.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import io
import os
import sys
import warnings
from contextlib import redirect_stderr

import torch
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestWarn(JitTestCase):
    def test_warn(self):
        @torch.jit.script
        def fn():
            warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_only_once(self):
        @torch.jit.script
        def fn():
            for _ in range(10):
                warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_only_once_in_loop_func(self):
        def w():
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            for _ in range(10):
                w()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_once_per_func(self):
        def w1():
            warnings.warn("I am warning you")

        def w2():
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            w1()
            w2()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_once_per_func_in_loop(self):
        def w1():
            warnings.warn("I am warning you")

        def w2():
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            for _ in range(10):
                w1()
                w2()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_multiple_calls_multiple_warnings(self):
        @torch.jit.script
        def fn():
            warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_multiple_calls_same_func_diff_stack(self):
        def warn(caller: str):
            warnings.warn("I am warning you from " + caller)

        @torch.jit.script
        def foo():
            warn("foo")

        @torch.jit.script
        def bar():
            warn("bar")

        f = io.StringIO()
        with redirect_stderr(f):
            foo()
            bar()

        FileCheck().check_count(
            str="UserWarning: I am warning you from foo",
            count=1,
            exactly=True,
        ).check_count(
            str="UserWarning: I am warning you from bar",
            count=1,
            exactly=True,
        ).run(f.getvalue())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 1 class(es) and 21 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestWarn`

**Functions defined**: `test_warn`, `fn`, `test_warn_only_once`, `fn`, `test_warn_only_once_in_loop_func`, `w`, `fn`, `test_warn_once_per_func`, `w1`, `w2`, `fn`, `test_warn_once_per_func_in_loop`, `w1`, `w2`, `fn`, `test_warn_multiple_calls_multiple_warnings`, `fn`, `test_warn_multiple_calls_same_func_diff_stack`, `warn`, `foo`

**Key imports**: io, os, sys, warnings, redirect_stderr, torch, FileCheck, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `os`
- `sys`
- `warnings`
- `contextlib`: redirect_stderr
- `torch`
- `torch.testing`: FileCheck
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


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

This is a test file. Run it with:

```bash
python test/jit/test_warn.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/jit`):

- [`test_dataclasses.py_docs.md`](./test_dataclasses.py_docs.md)
- [`test_recursive_script.py_docs.md`](./test_recursive_script.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_python_builtins.py_docs.md`](./test_python_builtins.py_docs.md)
- [`test_functional_blocks.py_docs.md`](./test_functional_blocks.py_docs.md)
- [`test_hooks_modules.py_docs.md`](./test_hooks_modules.py_docs.md)
- [`mydecorator.py_docs.md`](./mydecorator.py_docs.md)
- [`test_union.py_docs.md`](./test_union.py_docs.md)
- [`test_python_bindings.py_docs.md`](./test_python_bindings.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)


## Cross-References

- **File Documentation**: `test_warn.py_docs.md`
- **Keyword Index**: `test_warn.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/jit`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python docs/test/jit/test_warn.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/jit`):

- [`test_attr.py_kw.md_docs.md`](./test_attr.py_kw.md_docs.md)
- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_dataclasses.py_docs.md_docs.md`](./test_dataclasses.py_docs.md_docs.md)
- [`test_aten_pow.py_kw.md_docs.md`](./test_aten_pow.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_graph_rewrite_passes.py_kw.md_docs.md`](./test_graph_rewrite_passes.py_kw.md_docs.md)
- [`test_module_containers.py_kw.md_docs.md`](./test_module_containers.py_kw.md_docs.md)
- [`test_complex.py_kw.md_docs.md`](./test_complex.py_kw.md_docs.md)
- [`test_types.py_kw.md_docs.md`](./test_types.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_warn.py_docs.md_docs.md`
- **Keyword Index**: `test_warn.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
