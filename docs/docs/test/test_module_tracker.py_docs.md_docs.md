# Documentation: `docs/test/test_module_tracker.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_module_tracker.py_docs.md`
- **Size**: 6,336 bytes (6.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_module_tracker.py`

## File Metadata

- **Path**: `test/test_module_tracker.py`
- **Size**: 3,550 bytes (3.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: unknown"]

from copy import copy

import torch
from torch import nn
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TestCase,
    xfailIfTorchDynamo,
)
from torch.utils.checkpoint import checkpoint
from torch.utils.module_tracker import ModuleTracker


class TestModuleTracker(TestCase):
    # "https://github.com/pytorch/pytorch/issues/127112
    @xfailIfTorchDynamo
    def test_module_hierarchy(self):
        seen_fw = []
        seen_bw = []

        class Foo(nn.Module):
            def forward(self, x):
                x = x["a"].relu_()
                seen_fw.append((copy(tracker.parents), tracker.is_bw))
                x.register_hook(
                    lambda grad: seen_bw.append((copy(tracker.parents), tracker.is_bw))
                )
                return {"a": torch.mm(x, x)}

        class Mod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = Foo()
                self.b = nn.ModuleDict({"nest": Foo()})
                self.c = nn.ModuleList([Foo()])

            def forward(self, x):
                x = self.c[0](x)
                return self.b["nest"](self.a(x))

        mod = Mod()

        with ModuleTracker() as tracker:
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()

        self.assertEqual(
            seen_fw,
            [
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
            ],
        )

        self.assertEqual(
            seen_bw,
            [
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
            ],
        )

    @skipIfTorchDynamo("unexplained 3.13+ recursion error")
    def test_confused_hierarchy(self):
        class MyMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(2, 2)
                self.ran = False

            def forward(self, inp):
                if not self.ran:
                    self.ran = True
                    return self(inp)
                else:
                    self.ran = False
                    return self.inner(inp)

        mod = MyMod()
        inp = torch.rand(1, 2, requires_grad=True)

        # Should not fail
        with ModuleTracker():
            res = mod(inp)
            res.sum().backward()

        # Should not fail
        with ModuleTracker():
            res = checkpoint(lambda inp: mod(inp), inp)
            res.sum().backward()

    def test_bw_detection(self):
        mod = nn.Linear(2, 2)

        with ModuleTracker() as tracker:
            mod(torch.rand(2, requires_grad=True)).sum().backward()
            self.assertFalse(tracker.is_bw)
            self.assertEqual(tracker.parents, {"Global"})


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 4 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestModuleTracker`, `Foo`, `Mod`, `MyMod`

**Functions defined**: `test_module_hierarchy`, `forward`, `__init__`, `forward`, `test_confused_hierarchy`, `__init__`, `forward`, `test_bw_detection`

**Key imports**: copy, torch, nn, checkpoint, ModuleTracker


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`: copy
- `torch`
- `torch.utils.checkpoint`: checkpoint
- `torch.utils.module_tracker`: ModuleTracker


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
python test/test_module_tracker.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_module_tracker.py_docs.md`
- **Keyword Index**: `test_module_tracker.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_module_tracker.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_module_tracker.py_docs.md_docs.md`
- **Keyword Index**: `test_module_tracker.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
