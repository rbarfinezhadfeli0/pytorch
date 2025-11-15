# Documentation: `test/dynamo/test_export_mutations.py`

## File Metadata

- **Path**: `test/dynamo/test_export_mutations.py`
- **Size**: 3,325 bytes (3.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.common_utils import IS_FBCODE


class MutationExportTests(torch._dynamo.test_case.TestCase):
    def check_failure_on_export(self, mod, *args):
        with self.assertRaises(AssertionError):
            torch._dynamo.export(mod)(*args)

    def check_same_with_export(self, mod, arg):
        real_result = mod(arg)
        graph, _ = torch._dynamo.export(mod)(arg)
        result = graph(arg)
        self.assertEqual(result, real_result)

    def test_module_attribute_mutation_violation_positive_1(self):
        # Mutating attribute with a Tensor type
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.randn(3, 2)

            def forward(self, x):
                self.a = self.a.to(torch.float64)
                return x.sum() + self.a.sum()

        self.check_same_with_export(Foo(), torch.randn(3, 2))

    def test_module_attribute_mutation_violation_negative_1(self):
        # Mutating attribute with a Tensor type inside __init__ but
        # not in forward()
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.randn(3, 2)

            def forward(self, x):
                return x.sum() + self.a.to(torch.float64).sum()

        self.check_same_with_export(Foo(), torch.randn(3, 2))

    def test_module_attribute_mutation_violation_negative_2(self):
        # Mutating attribute with a Tensor type inside __init__ twice
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.randn(3, 2)
                self.a = self.a.to(torch.float64)

            def forward(self, x):
                return x.sum() + self.a.sum()

        self.check_same_with_export(Foo(), torch.randn(3, 2))

    def test_module_attribute_mutation_violation_negative_3(self):
        # Mutating local variable inside forward()
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.randn(3, 2)

            def forward(self, x):
                b = 1
                b = b * 5
                return x.sum() + self.a.sum() + b

        self.check_same_with_export(Foo(), torch.randn(3, 2))

    @unittest.skipIf(IS_FBCODE, "Broken in fbcode")
    def test_module_attribute_mutation_violation_negative_4(self):
        # Mutating attribute with a Tensor type
        # But not exporting but using eager mode as well as dynamo optimize mode
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.randn(3, 2)

            def forward(self, x):
                self.a = self.a.to(torch.float64)
                return x.sum() + self.a.sum()

        mod = Foo()
        arg = torch.randn(3, 2)
        real_result = mod(arg)
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(opt_mod(arg), real_result)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 6 class(es) and 17 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MutationExportTests`, `Foo`, `Foo`, `Foo`, `Foo`, `Foo`

**Functions defined**: `check_failure_on_export`, `check_same_with_export`, `test_module_attribute_mutation_violation_positive_1`, `__init__`, `forward`, `test_module_attribute_mutation_violation_negative_1`, `__init__`, `forward`, `test_module_attribute_mutation_violation_negative_2`, `__init__`, `forward`, `test_module_attribute_mutation_violation_negative_3`, `__init__`, `forward`, `test_module_attribute_mutation_violation_negative_4`, `__init__`, `forward`

**Key imports**: unittest, torch, torch._dynamo.test_case, torch._dynamo.testing, IS_FBCODE, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`
- `torch.testing._internal.common_utils`: IS_FBCODE


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/dynamo/test_export_mutations.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_export_mutations.py_docs.md`
- **Keyword Index**: `test_export_mutations.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
