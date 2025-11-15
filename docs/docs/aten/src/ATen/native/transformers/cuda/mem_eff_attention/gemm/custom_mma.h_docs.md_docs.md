# Documentation: `docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h_docs.md`
- **Size**: 5,216 bytes (5.09 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h`
- **Size**: 2,491 bytes (2.43 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_multistage.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_pipelined.h>

#include <cutlass/gemm/threadblock/mma_multistage.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
template <typename Mma, int kMaxK>
struct MakeCustomMma;

template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    typename IteratorB,
    typename SmemIteratorB,
    cutlass::arch::CacheOperation::Kind CacheOpB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int Stages,
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
    int kMaxK>
struct MakeCustomMma<
    cutlass::gemm::threadblock::MmaMultistage<
        Shape,
        IteratorA,
        SmemIteratorA,
        CacheOpA,
        IteratorB,
        SmemIteratorB,
        CacheOpB,
        ElementC,
        LayoutC,
        Policy,
        Stages,
        SharedMemoryClear>,
    kMaxK> {
  // Reduce the number of stages if we don't need that many
  static int constexpr kStages =
      kMaxK == cutlass::platform::numeric_limits<int>::max()
      ? Stages
      : cutlass::const_min(
            Stages,
            (kMaxK + int(Shape::kK) - 1) / int(Shape::kK));
  using Mma = cutlass::gemm::threadblock::CustomMmaMultistage<
      Shape,
      IteratorA,
      SmemIteratorA,
      CacheOpA,
      IteratorB,
      SmemIteratorB,
      CacheOpB,
      ElementC,
      LayoutC,
      Policy,
      kStages,
      SharedMemoryClear,
      kMaxK>;
};

template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    typename IteratorB,
    typename SmemIteratorB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int kMaxK>
struct MakeCustomMma<
    cutlass::gemm::threadblock::MmaPipelined<
        Shape,
        IteratorA,
        SmemIteratorA,
        IteratorB,
        SmemIteratorB,
        ElementC,
        LayoutC,
        Policy>,
    kMaxK> {
  using Mma = cutlass::gemm::threadblock::CustomMmaPipelined<
      Shape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      Policy>;
};

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `MakeCustomMma`, `MakeCustomMma`, `MakeCustomMma`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_multistage.h`
- `ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_pipelined.h`
- `cutlass/gemm/threadblock/mma_multistage.h`
- `cutlass/gemm/threadblock/mma_pipelined.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

Files in the same folder (`aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm`):

- [`custom_mma_base.h_docs.md`](./custom_mma_base.h_docs.md)
- [`custom_mma_multistage.h_docs.md`](./custom_mma_multistage.h_docs.md)
- [`mma_from_smem.h_docs.md`](./mma_from_smem.h_docs.md)
- [`custom_mma_pipelined.h_docs.md`](./custom_mma_pipelined.h_docs.md)
- [`mma_accum_lambda_iterator.h_docs.md`](./mma_accum_lambda_iterator.h_docs.md)
- [`find_default_mma.h_docs.md`](./find_default_mma.h_docs.md)


## Cross-References

- **File Documentation**: `custom_mma.h_docs.md`
- **Keyword Index**: `custom_mma.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm`):

- [`mma_from_smem.h_docs.md_docs.md`](./mma_from_smem.h_docs.md_docs.md)
- [`custom_mma_multistage.h_docs.md_docs.md`](./custom_mma_multistage.h_docs.md_docs.md)
- [`mma_accum_lambda_iterator.h_kw.md_docs.md`](./mma_accum_lambda_iterator.h_kw.md_docs.md)
- [`custom_mma_pipelined.h_docs.md_docs.md`](./custom_mma_pipelined.h_docs.md_docs.md)
- [`mma_from_smem.h_kw.md_docs.md`](./mma_from_smem.h_kw.md_docs.md)
- [`custom_mma.h_kw.md_docs.md`](./custom_mma.h_kw.md_docs.md)
- [`custom_mma_multistage.h_kw.md_docs.md`](./custom_mma_multistage.h_kw.md_docs.md)
- [`custom_mma_base.h_docs.md_docs.md`](./custom_mma_base.h_docs.md_docs.md)
- [`custom_mma_pipelined.h_kw.md_docs.md`](./custom_mma_pipelined.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `custom_mma.h_docs.md_docs.md`
- **Keyword Index**: `custom_mma.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
