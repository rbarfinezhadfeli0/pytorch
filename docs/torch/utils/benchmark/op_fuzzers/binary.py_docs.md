# Documentation: binary.py

## File Metadata
- **Path**: `torch/utils/benchmark/op_fuzzers/binary.py`
- **Size**: 4144 bytes
- **Lines**: 107
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): BinaryOpFuzzer

### Functions
This file defines 1 function(s): __init__


## Key Components

The file contains 314 words across 107 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4144 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
