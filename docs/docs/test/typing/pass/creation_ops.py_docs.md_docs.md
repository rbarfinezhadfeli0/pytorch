# Documentation: `docs/test/typing/pass/creation_ops.py_docs.md`

## File Metadata

- **Path**: `docs/test/typing/pass/creation_ops.py_docs.md`
- **Size**: 5,489 bytes (5.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/typing/pass/creation_ops.py`

## File Metadata

- **Path**: `test/typing/pass/creation_ops.py`
- **Size**: 3,209 bytes (3.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# mypy: disable-error-code="possibly-undefined"
# flake8: noqa
from typing_extensions import assert_type

import torch
from torch.testing._internal.common_utils import TEST_NUMPY


if TEST_NUMPY:
    import numpy as np

# From the docs, there are quite a few ways to create a tensor:
# https://pytorch.org/docs/stable/tensors.html

# torch.tensor()
torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
torch.tensor([0, 1])
torch.tensor(
    [[0.11111, 0.222222, 0.3333333]], dtype=torch.float64, device=torch.device("cuda:0")
)
torch.tensor(3.14159)

# torch.sparse_coo_tensor
i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
torch.sparse_coo_tensor(i, v, [2, 4])
torch.sparse_coo_tensor(i, v)
torch.sparse_coo_tensor(
    i, v, [2, 4], dtype=torch.float64, device=torch.device("cuda:0")
)
torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1])
torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, 2]), [1, 2])

# torch.as_tensor
a = [1, 2, 3]
torch.as_tensor(a)
torch.as_tensor(a, device=torch.device("cuda"))

# torch.as_strided
x = torch.randn(3, 3)
torch.as_strided(x, (2, 2), (1, 2))
torch.as_strided(x, (2, 2), (1, 2), 1)

# torch.from_numpy
if TEST_NUMPY:
    torch.from_numpy(np.array([1, 2, 3]))

# torch.zeros/zeros_like
torch.zeros(2, 3)
torch.zeros((2, 3))
torch.zeros([2, 3])
torch.zeros(5)
torch.zeros_like(torch.empty(2, 3))

# torch.ones/ones_like
torch.ones(2, 3)
torch.ones((2, 3))
torch.ones([2, 3])
torch.ones(5)
torch.ones_like(torch.empty(2, 3))

# torch.arange
torch.arange(5)
torch.arange(1, 4)
torch.arange(1, 2.5, 0.5)

# torch.range
torch.range(1, 4)
torch.range(1, 4, 0.5)

# torch.linspace
torch.linspace(3, 10, steps=5)
torch.linspace(-10, 10, steps=5)
torch.linspace(start=-10, end=10, steps=5)
torch.linspace(start=-10, end=10, steps=1)

# torch.logspace
torch.logspace(start=-10, end=10, steps=5)
torch.logspace(start=0.1, end=1.0, steps=5)
torch.logspace(start=0.1, end=1.0, steps=1)
torch.logspace(start=2, end=2, steps=1, base=2)

# torch.eye
torch.eye(3)

# torch.empty/empty_like/empty_strided
torch.empty(2, 3)
torch.empty((2, 3))
torch.empty([2, 3])
torch.empty_like(torch.empty(2, 3), dtype=torch.int64)
torch.empty_strided((2, 3), (1, 2))

# torch.full/full_like
torch.full((2, 3), 3.141592)
torch.full_like(torch.full((2, 3), 3.141592), 2.71828)

# torch.quantize_per_tensor
torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8)

# torch.quantize_per_channel
x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
quant = torch.quantize_per_channel(
    x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8
)

# torch.dequantize
torch.dequantize(x)

# torch.complex
real = torch.tensor([1, 2], dtype=torch.float32)
imag = torch.tensor([3, 4], dtype=torch.float32)
torch.complex(real, imag)

# torch.polar
abs = torch.tensor([1, 2], dtype=torch.float64)
pi = torch.acos(torch.zeros(1)).item() * 2
angle = torch.tensor([pi / 2, 5 * pi / 4], dtype=torch.float64)
torch.polar(abs, angle)

# torch.heaviside
inp = torch.tensor([-1.5, 0, 2.0])
values = torch.tensor([0.5])
torch.heaviside(inp, values)

# Parameter
p = torch.nn.Parameter(torch.empty(1))
assert_type(p, torch.nn.Parameter)

```



## High-Level Overview


This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: assert_type, torch, TEST_NUMPY, numpy as np


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/typing/pass`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `typing_extensions`: assert_type
- `torch`
- `torch.testing._internal.common_utils`: TEST_NUMPY
- `numpy as np`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/typing/pass/creation_ops.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/typing/pass`):

- [`cuda_steam.py_docs.md`](./cuda_steam.py_docs.md)
- [`torch_size.py_docs.md`](./torch_size.py_docs.md)
- [`arithmetic_ops.py_docs.md`](./arithmetic_ops.py_docs.md)
- [`disabled_jit.py_docs.md`](./disabled_jit.py_docs.md)
- [`math_ops.py_docs.md`](./math_ops.py_docs.md)
- [`distributions.py_docs.md`](./distributions.py_docs.md)


## Cross-References

- **File Documentation**: `creation_ops.py_docs.md`
- **Keyword Index**: `creation_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/typing/pass`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/typing/pass`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/typing/pass/creation_ops.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/typing/pass`):

- [`torch_size.py_kw.md_docs.md`](./torch_size.py_kw.md_docs.md)
- [`distributions.py_docs.md_docs.md`](./distributions.py_docs.md_docs.md)
- [`disabled_jit.py_docs.md_docs.md`](./disabled_jit.py_docs.md_docs.md)
- [`arithmetic_ops.py_kw.md_docs.md`](./arithmetic_ops.py_kw.md_docs.md)
- [`arithmetic_ops.py_docs.md_docs.md`](./arithmetic_ops.py_docs.md_docs.md)
- [`disabled_jit.py_kw.md_docs.md`](./disabled_jit.py_kw.md_docs.md)
- [`distributions.py_kw.md_docs.md`](./distributions.py_kw.md_docs.md)
- [`math_ops.py_kw.md_docs.md`](./math_ops.py_kw.md_docs.md)
- [`creation_ops.py_kw.md_docs.md`](./creation_ops.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `creation_ops.py_docs.md_docs.md`
- **Keyword Index**: `creation_ops.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
