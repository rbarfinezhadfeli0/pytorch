# Documentation: `docs/torch/testing/_internal/common_dist_composable.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/common_dist_composable.py_docs.md`
- **Size**: 6,381 bytes (6.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/common_dist_composable.py`

## File Metadata

- **Path**: `torch/testing/_internal/common_dist_composable.py`
- **Size**: 3,577 bytes (3.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# mypy: ignore-errors

# Owner(s): ["oncall: distributed"]


import torch
import torch.nn as nn


class UnitModule(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.l1 = nn.Linear(100, 100, device=device)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100, device=device),
            nn.ReLU(),
        )
        self.l2 = nn.Linear(100, 100, device=device)

    def forward(self, x):
        return self.l2(self.seq(self.l1(x)))


class CompositeModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.l1 = nn.Linear(100, 100, device=device)
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        self.l2 = nn.Linear(100, 100, device=device)

    def forward(self, x):
        return self.l2(self.u2(self.u1(self.l1(x))))


class UnitParamModule(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.l = nn.Linear(100, 100, device=device)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100, device=device),
            nn.ReLU(),
        )
        self.p = nn.Parameter(torch.randn((100, 100), device=device))

    def forward(self, x):
        return torch.mm(self.seq(self.l(x)), self.p)


class CompositeParamModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.l = nn.Linear(100, 100, device=device)
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        self.p = nn.Parameter(torch.randn((100, 100), device=device))
        self.register_buffer(
            "buffer", torch.randn((100, 100), device=device), persistent=True
        )

    def forward(self, x):
        a = self.u2(self.u1(self.l(x)))
        b = self.p
        return torch.mm(a, b)


class FakeSequential(nn.Module):
    # Define this class to achieve a desired nested wrapping using the module
    # wrap policy with `nn.Sequential`
    def __init__(self, *modules: tuple[nn.Module, ...]) -> None:
        super().__init__()
        self._module_sequence = list(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self._module_sequence:
            x = module(x)
        return x


class NestedSequentialModel(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        # This nested structure exercises traversal order to catch differences
        # between valid traversals (e.g. BFS and DFS variations).
        self.seq1 = nn.Sequential(
            nn.Linear(1, 1, device=device),
            FakeSequential(
                nn.Linear(1, 1, device=device),
                nn.ReLU(),
                FakeSequential(
                    nn.Linear(1, 1, device=device),
                ),
                nn.ReLU(),
            ),
            nn.Linear(1, 2, device=device),
        )
        self.lin = nn.Linear(2, 2, device=device)
        self.seq2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2, 3, device=device),
            FakeSequential(
                nn.Linear(3, 2, bias=False, device=device),
                nn.Linear(2, 4, bias=False, device=device),
            ),
        )

        # FIXME(rec): forward() is not a method, it's a local function inside __init__
        # that is never used. It should probabkly be outdented by four spaces, or removed.
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.seq2(self.lin(self.seq1(x)))

```



## High-Level Overview


This Python file contains 7 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `UnitModule`, `CompositeModel`, `UnitParamModule`, `CompositeParamModel`, `FakeSequential`, `NestedSequentialModel`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`

**Key imports**: torch, torch.nn as nn


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn as nn`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python torch/testing/_internal/common_dist_composable.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal`):

- [`common_jit.py_docs.md`](./common_jit.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_function_db.py_docs.md`](./autograd_function_db.py_docs.md)
- [`custom_op_db.py_docs.md`](./custom_op_db.py_docs.md)
- [`subclasses.py_docs.md`](./subclasses.py_docs.md)
- [`logging_utils.py_docs.md`](./logging_utils.py_docs.md)
- [`two_tensor.py_docs.md`](./two_tensor.py_docs.md)
- [`autocast_test_lists.py_docs.md`](./autocast_test_lists.py_docs.md)
- [`hypothesis_utils.py_docs.md`](./hypothesis_utils.py_docs.md)
- [`common_mkldnn.py_docs.md`](./common_mkldnn.py_docs.md)


## Cross-References

- **File Documentation**: `common_dist_composable.py_docs.md`
- **Keyword Index**: `common_dist_composable.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal`, which is part of the **core PyTorch library**.



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
python docs/torch/testing/_internal/common_dist_composable.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal`):

- [`hypothesis_utils.py_kw.md_docs.md`](./hypothesis_utils.py_kw.md_docs.md)
- [`fake_config_module2.py_kw.md_docs.md`](./fake_config_module2.py_kw.md_docs.md)
- [`static_module.py_kw.md_docs.md`](./static_module.py_kw.md_docs.md)
- [`common_pruning.py_kw.md_docs.md`](./common_pruning.py_kw.md_docs.md)
- [`composite_compliance.py_kw.md_docs.md`](./composite_compliance.py_kw.md_docs.md)
- [`common_mkldnn.py_docs.md_docs.md`](./common_mkldnn.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`common_dtype.py_docs.md_docs.md`](./common_dtype.py_docs.md_docs.md)
- [`common_methods_invocations.py_docs.md_docs.md`](./common_methods_invocations.py_docs.md_docs.md)
- [`hypothesis_utils.py_docs.md_docs.md`](./hypothesis_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `common_dist_composable.py_docs.md_docs.md`
- **Keyword Index**: `common_dist_composable.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
