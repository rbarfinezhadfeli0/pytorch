# Documentation: `docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h_docs.md`
- **Size**: 7,972 bytes (7.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h`
- **Size**: 5,176 bytes (5.05 KB)
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
/*! \file
    \brief Cutlass provides helper template functions to figure out the right
   data structures to instantiate to run a GEMM with various parameters (see
   `cutlass/gemm/threadblock/default_mma.h`). However, due to template
   instantiation priority rules, it will only create an MmaMultiStage with
   kStages=3 (otherwise creates an MmePipelined - which is not compatible with
   FastF32). kStages=3 uses too much shared memory and we want to use kStages=2,
   so we just copy-pasted some code from `default_mma.h` and
   `default_mma_core.h` files and wrapped this template to allow our use case.

    This is really only for the FastF32 case - aka using TensorCores with fp32.
*/

#pragma once

#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/default_mma_core_simt.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm70.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    typename Enable_ = void>
struct FindDefaultMma {
  static constexpr bool AccumulatorsInRowMajor = false;
  static constexpr SharedMemoryClearOption SharedMemoryClear =
      SharedMemoryClearOption::kNone;
  using DefaultMma = cutlass::gemm::threadblock::DefaultMma<
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      LayoutC,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      Stages,
      Operator,
      AccumulatorsInRowMajor,
      SharedMemoryClear>;
};

/// Specialization for sm80 / FastF32 / multistage with kStages=2
template <
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    int kStages,
    typename Operator>
struct FindDefaultMma<
    ElementA_,
    LayoutA_,
    kAlignmentA,
    ElementB_,
    LayoutB_,
    kAlignmentB,
    ElementAccumulator,
    layout::RowMajor,
    arch::OpClassTensorOp,
    arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    kStages,
    Operator,
    typename cutlass::platform::enable_if<(kAlignmentA > 1)>::type> {
  using LayoutC = layout::RowMajor;
  using OperatorClass = arch::OpClassTensorOp;
  using ArchTag = arch::Sm80;

  using DefaultMma_ = cutlass::gemm::threadblock::DefaultMma<
      ElementA_,
      LayoutA_,
      kAlignmentA,
      ElementB_,
      LayoutB_,
      kAlignmentB,
      ElementAccumulator,
      LayoutC,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      3,
      Operator>;
  struct DefaultMma : DefaultMma_ {
    using MmaCore_ = typename DefaultMma_::MmaCore;
    // Define the threadblock-scoped multistage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
        typename MmaCore_::Shape,
        typename DefaultMma_::IteratorA,
        typename MmaCore_::SmemIteratorA,
        MmaCore_::kCacheOpA,
        typename DefaultMma_::IteratorB,
        typename MmaCore_::SmemIteratorB,
        MmaCore_::kCacheOpB,
        ElementAccumulator,
        LayoutC,
        typename MmaCore_::MmaPolicy,
        kStages>;
  };
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `cutlass`, `gemm`, `threadblock`

**Classes/Structs**: `tag`, `FindDefaultMma`, `FindDefaultMma`, `DefaultMma`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cutlass/gemm/threadblock/default_mma.h`
- `cutlass/gemm/threadblock/default_mma_core_simt.h`
- `cutlass/gemm/threadblock/default_mma_core_sm70.h`
- `cutlass/gemm/threadblock/default_mma_core_sm75.h`
- `cutlass/gemm/threadblock/default_mma_core_sm80.h`


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
- [`custom_mma.h_docs.md`](./custom_mma.h_docs.md)
- [`mma_from_smem.h_docs.md`](./mma_from_smem.h_docs.md)
- [`custom_mma_pipelined.h_docs.md`](./custom_mma_pipelined.h_docs.md)
- [`mma_accum_lambda_iterator.h_docs.md`](./mma_accum_lambda_iterator.h_docs.md)


## Cross-References

- **File Documentation**: `find_default_mma.h_docs.md`
- **Keyword Index**: `find_default_mma.h_kw.md`
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
- [`custom_mma.h_docs.md_docs.md`](./custom_mma.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `find_default_mma.h_docs.md_docs.md`
- **Keyword Index**: `find_default_mma.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
