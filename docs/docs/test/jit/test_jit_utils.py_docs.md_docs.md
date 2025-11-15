# Documentation: `docs/test/jit/test_jit_utils.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_jit_utils.py_docs.md`
- **Size**: 7,132 bytes (6.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_jit_utils.py`

## File Metadata

- **Path**: `test/jit/test_jit_utils.py`
- **Size**: 3,711 bytes (3.62 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import sys
from textwrap import dedent

import torch
from torch.testing._internal import jit_utils


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Tests various JIT-related utility functions.
class TestJitUtils(JitTestCase):
    # Tests that POSITIONAL_OR_KEYWORD arguments are captured.
    def test_get_callable_argument_names_positional_or_keyword(self):
        def fn_positional_or_keyword_args_only(x, y):
            return x + y

        self.assertEqual(
            ["x", "y"],
            torch._jit_internal.get_callable_argument_names(
                fn_positional_or_keyword_args_only
            ),
        )

    # Tests that POSITIONAL_ONLY arguments are ignored.
    def test_get_callable_argument_names_positional_only(self):
        code = dedent(
            """
            def fn_positional_only_arg(x, /, y):
                return x + y
        """
        )

        fn_positional_only_arg = jit_utils._get_py3_code(code, "fn_positional_only_arg")
        self.assertEqual(
            ["y"],
            torch._jit_internal.get_callable_argument_names(fn_positional_only_arg),
        )

    # Tests that VAR_POSITIONAL arguments are ignored.
    def test_get_callable_argument_names_var_positional(self):
        # Tests that VAR_POSITIONAL arguments are ignored.
        def fn_var_positional_arg(x, *arg):
            return x + arg[0]

        self.assertEqual(
            ["x"],
            torch._jit_internal.get_callable_argument_names(fn_var_positional_arg),
        )

    # Tests that KEYWORD_ONLY arguments are ignored.
    def test_get_callable_argument_names_keyword_only(self):
        def fn_keyword_only_arg(x, *, y):
            return x + y

        self.assertEqual(
            ["x"], torch._jit_internal.get_callable_argument_names(fn_keyword_only_arg)
        )

    # Tests that VAR_KEYWORD arguments are ignored.
    def test_get_callable_argument_names_var_keyword(self):
        def fn_var_keyword_arg(**args):
            return args["x"] + args["y"]

        self.assertEqual(
            [], torch._jit_internal.get_callable_argument_names(fn_var_keyword_arg)
        )

    # Tests that a function signature containing various different types of
    # arguments are ignored.
    def test_get_callable_argument_names_hybrid(self):
        code = dedent(
            """
            def fn_hybrid_args(x, /, y, *args, **kwargs):
                return x + y + args[0] + kwargs['z']
        """
        )
        fn_hybrid_args = jit_utils._get_py3_code(code, "fn_hybrid_args")
        self.assertEqual(
            ["y"], torch._jit_internal.get_callable_argument_names(fn_hybrid_args)
        )

    def test_checkscriptassertraisesregex(self):
        def fn():
            tup = (1, 2)
            return tup[2]

        self.checkScriptRaisesRegex(fn, (), Exception, "range", name="fn")

        s = dedent(
            """
        def fn():
            tup = (1, 2)
            return tup[2]
        """
        )

        self.checkScriptRaisesRegex(s, (), Exception, "range", name="fn")

    def test_no_tracer_warn_context_manager(self):
        torch._C._jit_set_tracer_state_warn(True)
        with jit_utils.NoTracerWarnContextManager():
            self.assertEqual(False, torch._C._jit_get_tracer_state_warn())
        self.assertEqual(True, torch._C._jit_get_tracer_state_warn())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview

"""            def fn_positional_only_arg(x, /, y):                return x + y

This Python file contains 1 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestJitUtils`

**Functions defined**: `test_get_callable_argument_names_positional_or_keyword`, `fn_positional_or_keyword_args_only`, `test_get_callable_argument_names_positional_only`, `fn_positional_only_arg`, `test_get_callable_argument_names_var_positional`, `fn_var_positional_arg`, `test_get_callable_argument_names_keyword_only`, `fn_keyword_only_arg`, `test_get_callable_argument_names_var_keyword`, `fn_var_keyword_arg`, `test_get_callable_argument_names_hybrid`, `fn_hybrid_args`, `test_checkscriptassertraisesregex`, `fn`, `fn`, `test_no_tracer_warn_context_manager`

**Key imports**: os, sys, dedent, torch, jit_utils, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `textwrap`: dedent
- `torch`
- `torch.testing._internal`: jit_utils
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
python test/jit/test_jit_utils.py
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

- **File Documentation**: `test_jit_utils.py_docs.md`
- **Keyword Index**: `test_jit_utils.py_kw.md`
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
python docs/test/jit/test_jit_utils.py_docs.md
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

- **File Documentation**: `test_jit_utils.py_docs.md_docs.md`
- **Keyword Index**: `test_jit_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
