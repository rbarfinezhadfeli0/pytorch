# Documentation: `torch/utils/benchmark/op_fuzzers/binary.py`

## File Metadata

- **Path**: `torch/utils/benchmark/op_fuzzers/binary.py`
- **Size**: 4,144 bytes (4.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
# mypy: allow-untyped-defs
import numpy as np
import torch

from torch.utils.benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedTensor


_MIN_DIM_SIZE = 16
_MAX_DIM_SIZE = 16 * 1024 ** 2
_POW_TWO_SIZES = tuple(2 ** i for i in range(
    int(np.log2(_MIN_DIM_SIZE)),
    int(np.log2(_MAX_DIM_SIZE)) + 1,
))


class BinaryOpFuzzer(Fuzzer):
    def __init__(self, seed, dtype=torch.float32, cuda=False) -> None:
        super().__init__(
            parameters=[
                # Dimensionality of x and y. (e.g. 1D, 2D, or 3D.)
                FuzzedParameter("dim", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),

                # Shapes for `x` and `y`.
                #       It is important to test all shapes, however
                #   powers of two are especially important and therefore
                #   warrant special attention. This is done by generating
                #   both a value drawn from all integers between the min and
                #   max allowed values, and another from only the powers of two
                #   (both distributions are loguniform) and then randomly
                #   selecting between the two.
                #       Moreover, `y` will occasionally have singleton
                #   dimensions in order to test broadcasting.
                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=_MIN_DIM_SIZE,
                        maxval=_MAX_DIM_SIZE,
                        distribution="loguniform",
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_pow2_{i}",
                        distribution={size: 1. / len(_POW_TWO_SIZES) for size in _POW_TWO_SIZES}
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_any_{i}"): 0.8,
                            ParameterAlias(f"k_pow2_{i}"): 0.2,
                        },
                        strict=True,
                    ) for i in range(3)
                ],

                [
                    FuzzedParameter(
                        name=f"y_k{i}",
                        distribution={
                            ParameterAlias(f"k{i}"): 0.8,
                            1: 0.2,
                        },
                        strict=True,
                    ) for i in range(3)
                ],

                # Steps for `x` and `y`. (Benchmarks strided memory access.)
                [
                    FuzzedParameter(
                        name=f"{name}_step_{i}",
                        distribution={1: 0.8, 2: 0.06, 4: 0.06, 8: 0.04, 16: 0.04},
                    )
                    for i in range(3)
                    for name in ("x", "y")
                ],

                # Repeatable entropy for downstream applications.
                FuzzedParameter(name="random_value", minval=0, maxval=2 ** 32 - 1, distribution="uniform"),
            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    steps=("x_step_0", "x_step_1", "x_step_2"),
                    probability_contiguous=0.75,
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    max_allocation_bytes=2 * 1024**3,  # 2 GB
                    dim_parameter="dim",
                    dtype=dtype,
                    cuda=cuda,
                ),
                FuzzedTensor(
                    name="y",
                    size=("y_k0", "y_k1", "y_k2"),
                    steps=("x_step_0", "x_step_1", "x_step_2"),
                    probability_contiguous=0.75,
                    max_allocation_bytes=2 * 1024**3,  # 2 GB
                    dim_parameter="dim",
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )

```



## High-Level Overview


This Python file contains 1 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BinaryOpFuzzer`

**Functions defined**: `__init__`

**Key imports**: numpy as np, torch, Fuzzer, FuzzedParameter, ParameterAlias, FuzzedTensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/benchmark/op_fuzzers`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `numpy as np`
- `torch`
- `torch.utils.benchmark`: Fuzzer, FuzzedParameter, ParameterAlias, FuzzedTensor


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/utils/benchmark/op_fuzzers`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`sparse_unary.py_docs.md`](./sparse_unary.py_docs.md)
- [`spectral.py_docs.md`](./spectral.py_docs.md)
- [`sparse_binary.py_docs.md`](./sparse_binary.py_docs.md)
- [`unary.py_docs.md`](./unary.py_docs.md)


## Cross-References

- **File Documentation**: `binary.py_docs.md`
- **Keyword Index**: `binary.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
