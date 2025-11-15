# Documentation: `test/jit/test_parametrization.py`

## File Metadata

- **Path**: `test/jit/test_parametrization.py`
- **Size**: 2,429 bytes (2.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]


import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestParametrization(JitTestCase):
    # Define some parametrization
    class Symmetric(nn.Module):
        def forward(self, X):
            return X.triu() + X.triu(1).mT

    def test_traceable(self):
        r"""Test the jit scripting and tracing of a parametrized model."""
        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", self.Symmetric())

        x = torch.randn(3, 5)
        y = model(x)

        # Check the tracing works. Because traced functions cannot be called
        # directly, we run the comparison on the activations.
        traced_model = torch.jit.trace_module(model, {"forward": x})
        y_hat = traced_model(x)
        self.assertEqual(y, y_hat)

        # Check traced model works with caching
        with parametrize.cached():
            y_hat = traced_model(x)
            self.assertEqual(y, y_hat)

        # Check the tracing throws an error when caching
        with self.assertRaisesRegex(RuntimeError, "Cannot trace a model while caching"):
            with parametrize.cached():
                traced_model = torch.jit.trace_module(model, {"forward": x})

    def test_scriptable(self):
        # TODO: Need to fix the scripting in parametrizations
        #       Currently, all the tests below will throw torch.jit.Error
        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", self.Symmetric())

        x = torch.randn(3, 5)
        y = model(x)

        with self.assertRaises(torch.jit.Error):
            # Check scripting works
            scripted_model = torch.jit.script(model)
            y_hat = scripted_model(x)
            self.assertEqual(y, y_hat)

            with parametrize.cached():
                # Check scripted model works when caching
                y_hat = scripted_model(x)
                self.assertEqual(y, y_hat)

                # Check the scripting process throws an error when caching
                with self.assertRaisesRegex(RuntimeError, "Caching is not implemented"):
                    scripted_model = torch.jit.trace_module(model)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview

r"""Test the jit scripting and tracing of a parametrized model."""        model = nn.Linear(5, 5)        parametrize.register_parametrization(model, "weight", self.Symmetric())        x = torch.randn(3, 5)        y = model(x)        # Check the tracing works. Because traced functions cannot be called        # directly, we run the comparison on the activations.        traced_model = torch.jit.trace_module(model, {"forward": x})        y_hat = traced_model(x)        self.assertEqual(y, y_hat)        # Check traced model works with caching        with parametrize.cached():            y_hat = traced_model(x)            self.assertEqual(y, y_hat)        # Check the tracing throws an error when caching        with self.assertRaisesRegex(RuntimeError, "Cannot trace a model while caching"):            with parametrize.cached():                traced_model = torch.jit.trace_module(model, {"forward": x})    def test_scriptable(self):        # TODO: Need to fix the scripting in parametrizations        #       Currently, all the tests below will throw torch.jit.Error        model = nn.Linear(5, 5)        parametrize.register_parametrization(model, "weight", self.Symmetric())        x = torch.randn(3, 5)        y = model(x)        with self.assertRaises(torch.jit.Error):

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestParametrization`, `Symmetric`

**Functions defined**: `forward`, `test_traceable`, `test_scriptable`

**Key imports**: torch, torch.nn.utils.parametrize as parametrize, nn, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn.utils.parametrize as parametrize`
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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
python test/jit/test_parametrization.py
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


## Cross-References

- **File Documentation**: `test_parametrization.py_docs.md`
- **Keyword Index**: `test_parametrization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
