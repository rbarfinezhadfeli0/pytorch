# Documentation: `docs/test/jit/test_graph_rewrite_passes.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_graph_rewrite_passes.py_docs.md`
- **Size**: 5,378 bytes (5.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_graph_rewrite_passes.py`

## File Metadata

- **Path**: `test/jit/test_graph_rewrite_passes.py`
- **Size**: 2,373 bytes (2.32 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import torch
import torch._C
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestGraphRewritePasses(JitTestCase):
    def test_fuse_linear(self):
        class FunctionalLinear(torch.nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                res = torch.matmul(x, self.weight.t())
                if self.bias is not None:
                    res.add_(self.bias)
                return res

        x1 = torch.rand(3)
        w1 = torch.rand(5, 3)
        b1 = torch.rand(5)
        for has_bias in [True, False]:
            bias = b1 if has_bias else None
            model = torch.jit.trace(FunctionalLinear(w1, bias), [x1])
            for node in model.graph.nodes():
                if node.kind() == "aten::matmul":
                    source_range_1 = node.sourceRange()
            torch._C._jit_pass_fuse_linear(model.graph)
            for node in model.graph.nodes():
                if node.kind() == "aten::linear":
                    source_range_2 = node.sourceRange()
            FileCheck().check("aten::linear").run(model.graph)
            check_not = ["aten::matmul", "aten::addmm", "aten::add_", "aten::t("]
            for cn in check_not:
                FileCheck().check_not(cn).run(model.graph)
            self.assertTrue(source_range_1 == source_range_2)
            # make sure it runs
            model(x1)

        # check matmuls are not fused
        class Matmul(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return torch.matmul(x, self.weight)

        x = torch.rand(5, 6, 5)
        w = torch.rand(5, 5, 100)
        model = torch.jit.trace(Matmul(w), [x])
        torch._C._jit_pass_fuse_linear(model.graph)
        # check 3d matmul is not fused
        FileCheck().check("aten::matmul").run(model.graph)
        FileCheck().check_not("aten::linear").run(model.graph)
        # make sure it runs
        model(x)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 3 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestGraphRewritePasses`, `FunctionalLinear`, `Matmul`

**Functions defined**: `test_fuse_linear`, `__init__`, `forward`, `__init__`, `forward`

**Key imports**: torch, torch._C, FileCheck, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._C`
- `torch.testing`: FileCheck
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


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
python test/jit/test_graph_rewrite_passes.py
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

- **File Documentation**: `test_graph_rewrite_passes.py_docs.md`
- **Keyword Index**: `test_graph_rewrite_passes.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python docs/test/jit/test_graph_rewrite_passes.py_docs.md
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

- **File Documentation**: `test_graph_rewrite_passes.py_docs.md_docs.md`
- **Keyword Index**: `test_graph_rewrite_passes.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
