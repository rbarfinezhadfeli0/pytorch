# Documentation: `docs/torchgen/_autoheuristic/pad_mm/gen_data_pad_mm.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/_autoheuristic/pad_mm/gen_data_pad_mm.py_docs.md`
- **Size**: 7,790 bytes (7.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/_autoheuristic/pad_mm/gen_data_pad_mm.py`

## File Metadata

- **Path**: `torchgen/_autoheuristic/pad_mm/gen_data_pad_mm.py`
- **Size**: 4,717 bytes (4.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Can be **executed as a standalone script**.

## Original Source

```python
import random
import sys
from pathlib import Path
from typing import Any


sys.path.append(str(Path(__file__).absolute().parents[1]))

from benchmark_runner import BenchmarkRunner  # type: ignore[import-not-found]
from benchmark_utils import (  # type: ignore[import-not-found]
    fits_in_memory,
    get_mm_tensors,
    set_precision,
    transpose_tensors,
)

import torch
from torch._inductor.fx_passes.pad_mm import (  # type: ignore[import-not-found]
    get_alignment_size_dtype,
)
from torch._inductor.utils import fresh_cache


class BenchmarkRunnerPadMM(BenchmarkRunner):  # type: ignore[misc, no-any-unimported]
    """
    BenchmarkRunner for pad_mm. Used to generate collect training data with AutoHeuristic to learn a heuristic.
    """

    def __init__(self) -> None:
        super().__init__("pad_mm")

    def create_input(self) -> tuple[Any, ...]:
        dtype = self.get_dtype()
        set_precision(dtype)
        m, k, n = self.get_m_k_n(dtype)

        (transpose_left, transpose_right) = transpose_tensors()
        prepadded_left = self.prepadded()
        prepadded_right = self.prepadded()
        return (
            m,
            k,
            n,
            transpose_left,
            transpose_right,
            dtype,
            prepadded_left,
            prepadded_right,
        )

    def run_benchmark(
        self,
        m: int,
        k: int,
        n: int,
        transpose_left: bool,
        transpose_right: bool,
        dtype: Any,
        prepadded_left: bool,
        prepadded_right: bool,
    ) -> None:
        a, b = get_mm_tensors(
            m,
            k,
            n,
            transpose_left,
            transpose_right,
            dtype_left=dtype,
            dtype_right=dtype,
        )

        print("Benchmarking the following input:")
        print(f"m={m} k={k} n={n} dtype={dtype}")
        print(f"transpose_left={transpose_left} transpose_right={transpose_right}")
        print(f"prepadded_left={prepadded_left} prepadded_right={prepadded_right}")

        with fresh_cache():

            def mm(a: Any, b: Any) -> Any:
                return torch.mm(a, b)

            def mm_mat1_prepadded(a: Any, b: Any) -> Any:
                return torch.mm(a + 1, b)

            def mm_mat2_prepadded(a: Any, b: Any) -> Any:
                return torch.mm(a, b + 1)

            def mm_mat1_mat2_prepadded(a: Any, b: Any) -> Any:
                return torch.mm(a + 1, b + 1)

            if prepadded_left and prepadded_right:
                cf = torch.compile(mm_mat1_mat2_prepadded)
            elif prepadded_left:
                cf = torch.compile(mm_mat1_prepadded)
            elif prepadded_right:
                cf = torch.compile(mm_mat2_prepadded)
            else:
                cf = torch.compile(mm)
            cf(a, b)
            torch.compiler.reset()

    def get_random_dim(
        self, min_power2: int = 1, max_power2: int = 16, p_unaligned: float = 0.25
    ) -> int:
        aligned = random.choices([True, False], [1 - p_unaligned, p_unaligned])[0]
        if aligned:
            return 2 ** random.randint(min_power2, max_power2)  # type: ignore[no-any-return]
        else:
            # choose a random number between 2^i and 2^(i+1)
            return self.get_random_between_pow2(min_power2, max_power2)  # type: ignore[no-any-return]

    def is_aligned(self, dim: int, align_size: int) -> bool:
        return dim % align_size == 0

    def get_m_k_n(self, dtype: Any) -> tuple[int, int, int]:
        uniform = random.choices([True, False])[0]
        align_size = get_alignment_size_dtype(dtype)

        # repeat until tensors fit in memory
        while True:
            if uniform:
                m = random.randint(1, 65536)
                k = random.randint(1, 65536)
                n = random.randint(1, 65536)
            else:
                m = self.get_random_dim()
                k = self.get_random_dim()
                n = self.get_random_dim()

            if all(self.is_aligned(dim, align_size) for dim in [m, k, n]):
                # skip if already aligned
                continue

            if fits_in_memory(dtype, m, k, n):
                return (m, k, n)

    def prepadded(self, p_prepadded: float = 0.2) -> bool:
        # p_prepadded: probability that a tensor is "prepadded", i.e. pad_mm excludes time it takes to pad from benchmarking
        return random.choices([True, False], [p_prepadded, 1 - p_prepadded])[0]

    def get_dtype(self) -> Any:
        dtype_choices = [torch.float16, torch.bfloat16, torch.float32]
        return random.choices(dtype_choices)[0]


if __name__ == "__main__":
    runner = BenchmarkRunnerPadMM()
    runner.run()

```



## High-Level Overview

"""    BenchmarkRunner for pad_mm. Used to generate collect training data with AutoHeuristic to learn a heuristic.

This Python file contains 1 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BenchmarkRunnerPadMM`

**Functions defined**: `__init__`, `create_input`, `run_benchmark`, `mm`, `mm_mat1_prepadded`, `mm_mat2_prepadded`, `mm_mat1_mat2_prepadded`, `get_random_dim`, `is_aligned`, `get_m_k_n`, `prepadded`, `get_dtype`

**Key imports**: random, sys, Path, Any, BenchmarkRunner  , torch, fresh_cache


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen/_autoheuristic/pad_mm`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `random`
- `sys`
- `pathlib`: Path
- `typing`: Any
- `benchmark_runner`: BenchmarkRunner  
- `torch`
- `torch._inductor.utils`: fresh_cache


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`torchgen/_autoheuristic/pad_mm`):

- [`train_pad_mm.py_docs.md`](./train_pad_mm.py_docs.md)
- [`test_pad_mm.py_docs.md`](./test_pad_mm.py_docs.md)
- [`train_regression_pad_mm.py_docs.md`](./train_regression_pad_mm.py_docs.md)
- [`train_decision_pad_mm.py_docs.md`](./train_decision_pad_mm.py_docs.md)
- [`gen_pad_mm_a100.sh_docs.md`](./gen_pad_mm_a100.sh_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`get_padmm_dataset.sh_docs.md`](./get_padmm_dataset.sh_docs.md)
- [`generate_heuristic_pad_mm.sh_docs.md`](./generate_heuristic_pad_mm.sh_docs.md)
- [`gen_pad_mm_h100.sh_docs.md`](./gen_pad_mm_h100.sh_docs.md)


## Cross-References

- **File Documentation**: `gen_data_pad_mm.py_docs.md`
- **Keyword Index**: `gen_data_pad_mm.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torchgen/_autoheuristic/pad_mm`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torchgen/_autoheuristic/pad_mm`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torchgen/_autoheuristic/pad_mm`):

- [`gen_pad_mm_h100.sh_kw.md_docs.md`](./gen_pad_mm_h100.sh_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`test_pad_mm.py_kw.md_docs.md`](./test_pad_mm.py_kw.md_docs.md)
- [`train_regression_pad_mm.py_kw.md_docs.md`](./train_regression_pad_mm.py_kw.md_docs.md)
- [`get_padmm_dataset.sh_docs.md_docs.md`](./get_padmm_dataset.sh_docs.md_docs.md)
- [`gen_pad_mm_a100.sh_docs.md_docs.md`](./gen_pad_mm_a100.sh_docs.md_docs.md)
- [`train_pad_mm.py_kw.md_docs.md`](./train_pad_mm.py_kw.md_docs.md)
- [`get_padmm_dataset.sh_kw.md_docs.md`](./get_padmm_dataset.sh_kw.md_docs.md)
- [`generate_heuristic_pad_mm.sh_docs.md_docs.md`](./generate_heuristic_pad_mm.sh_docs.md_docs.md)


## Cross-References

- **File Documentation**: `gen_data_pad_mm.py_docs.md_docs.md`
- **Keyword Index**: `gen_data_pad_mm.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
