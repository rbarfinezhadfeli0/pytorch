# Documentation: `docs/test/jit/test_custom_operators.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_custom_operators.py_docs.md`
- **Size**: 8,293 bytes (8.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_custom_operators.py`

## File Metadata

- **Path**: `test/jit/test_custom_operators.py`
- **Size**: 4,865 bytes (4.75 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import sys
import unittest

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


def canonical(graph):
    return torch._C._jit_pass_canonicalize(graph).str(False)


class TestCustomOperators(JitTestCase):
    def test_dynamic_op_registry(self):
        from torch._ops import _OpNamespace

        self.assertTrue(hasattr(torch, "ops"))

        if "_test" in torch.ops.__dict__:
            torch.ops.__dict__.pop("_test")

        # Don't use `hasattr()` because it will call `__getattr__`.
        self.assertNotIn("_test", torch.ops.__dict__)
        torch.ops._test
        self.assertIn("_test", torch.ops.__dict__)
        self.assertEqual(type(torch.ops._test), _OpNamespace)

        self.assertNotIn("leaky_relu", torch.ops._test.__dict__)
        op = torch.ops._test.leaky_relu
        self.assertTrue(callable(op))
        self.assertIn("leaky_relu", torch.ops._test.__dict__)
        op2 = torch.ops._test.leaky_relu
        self.assertEqual(op, op2)

    def test_getting_invalid_attr(self):
        for attr in ["__origin__", "__self__"]:
            with self.assertRaisesRegexWithHighlight(
                AttributeError,
                f"Invalid attribute '{attr}' for '_OpNamespace' '_test'",
                "",
            ):
                getattr(torch.ops._test, attr)

    def test_simply_calling_an_operator(self):
        input = torch.randn(100)
        output = torch.ops.aten.relu(input)
        self.assertEqual(output, input.relu())

    def test_default_arguments_are_used(self):
        output = torch.ops._test.leaky_relu(torch.tensor([-1.0, 1.0]))
        self.assertEqual(output, torch.tensor([-0.01, 1]))

    def test_passing_too_many_args(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"aten::relu\(\) expected at most 1 argument\(s\) but received 2 argument\(s\)",
            "",
        ):
            torch.ops.aten.relu(1, 2)

    def test_passing_too_few_args(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"aten::relu\(\) is missing value for argument 'self'.", ""
        ):
            torch.ops.aten.relu()

    def test_passing_one_positional_but_not_the_second(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"aten::type_as\(\) is missing value for argument 'other'.",
            "",
        ):
            torch.ops.aten.type_as(torch.ones(5, 5))

    def test_passing_unknown_kwargs(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Unknown keyword argument 'foo' for operator '_test::leaky_relu'",
            "",
        ):
            torch.ops._test.leaky_relu(torch.ones(5), foo=torch.ones(5))

    def test_passing_and_returning_lists(self):
        # Replace with actual test once we support lists.
        a, b = torch.rand(5), torch.rand(5)
        output = torch.ops._test.cat([a, b])
        output_ref = torch.cat([a, b])
        self.assertEqual(output, output_ref)

    def test_calling_scripted_custom_op(self):
        @torch.jit.script
        def func(x):
            return torch.ops.aten.relu(x)

        input = torch.ones(5, 5)
        self.assertEqual(func(input), input.relu())

    def test_calling_traced_custom_op(self):
        input = torch.ones(5, 5)
        func = torch.jit.trace(torch.ops.aten.relu, [input])
        self.assertEqual(func(input), input.relu())

    @unittest.skip(
        "Need to figure out default dtype differences between fbcode and oss"
    )
    def test_script_graph_for_custom_ops_matches_traced_graph(self):
        input = torch.ones(5, 5)
        trace = torch.jit.trace(torch.ops.aten.relu, [input])
        self.assertExpectedInline(
            canonical(trace.graph),
            """\
graph(%0 : Float(5, 5)):
  %1 : Float(5, 5) = aten::relu(%0)
  return (%1)
""",
        )

    def test_script_graph_contains_custom_op(self):
        @torch.jit.script
        def func(x):
            return torch.ops.aten.relu(x)

        self.assertExpectedInline(
            canonical(func.graph),
            """\
graph(%x.1 : Tensor):
  %1 : Tensor = aten::relu(%x.1)
  return (%1)
""",
        )

    def test_generic_list(self):
        self.assertEqual(torch.ops._test.get_first([["hello"]]), "hello")

    # https://github.com/pytorch/pytorch/issues/80508
    def test_where_no_scalar(self):
        x = torch.rand(1, 3, 224, 224)
        torch.ops.aten.where(x > 0.5, -1.5, 1.5)  # does not raise


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 1 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCustomOperators`

**Functions defined**: `canonical`, `test_dynamic_op_registry`, `test_getting_invalid_attr`, `test_simply_calling_an_operator`, `test_default_arguments_are_used`, `test_passing_too_many_args`, `test_passing_too_few_args`, `test_passing_one_positional_but_not_the_second`, `test_passing_unknown_kwargs`, `test_passing_and_returning_lists`, `test_calling_scripted_custom_op`, `func`, `test_calling_traced_custom_op`, `test_script_graph_for_custom_ops_matches_traced_graph`, `test_script_graph_contains_custom_op`, `func`, `test_generic_list`, `test_where_no_scalar`

**Key imports**: os, sys, unittest, torch, raise_on_run_directly, JitTestCase, _OpNamespace


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `unittest`
- `torch`
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase
- `torch._ops`: _OpNamespace


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
python test/jit/test_custom_operators.py
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

- **File Documentation**: `test_custom_operators.py_docs.md`
- **Keyword Index**: `test_custom_operators.py_kw.md`
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
python docs/test/jit/test_custom_operators.py_docs.md
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

- **File Documentation**: `test_custom_operators.py_docs.md_docs.md`
- **Keyword Index**: `test_custom_operators.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
