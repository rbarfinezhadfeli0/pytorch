# Documentation: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_base.h`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_base.h`
- **Size**: 6,241 bytes (6.09 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include <cutlass/aligned_buffer.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/mma_base.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class CustomMmaBase {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  ///< Policy describing tuning details
  using Policy = Policy_;

  //
  // Dependent types
  //

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Policy::Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<
      Shape::kM / WarpGemm::kM,
      Shape::kN / WarpGemm::kN,
      Shape::kK / WarpGemm::kK>;

  /// Number of warp-level GEMM operations
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::MmaShape::kK);

  /// Number of stages
  static int const kStages = Stages;

  //
  // Nested structs
  //

  /// Shared storage object needed by threadblock-scoped GEMM
  template <typename Element, typename OperandShape, typename OperandLayout>
  struct OperandSharedStorage {
    AlignedBuffer<Element, OperandShape::kCount> buffer;
    using TensorRef = TensorRef<Element, OperandLayout>;

    CUTLASS_DEVICE
    static OperandLayout Layout() {
      return OperandLayout::packed({OperandShape::kRow, OperandShape::kColumn});
    }

    /// Returns a TensorRef to the operand
    CUTLASS_HOST_DEVICE
    TensorRef ref() {
      return TensorRef{buffer.data(), Layout()};
    }
  };

  /// Shape of the A matrix operand in shared memory
  using ShapeA = MatrixShape<
      Shape::kM + Policy::SmemPaddingA::kRow,
      Shape::kK * kStages + Policy::SmemPaddingA::kColumn>;

  /// Shape of the B matrix operand in shared memory
  using ShapeB = MatrixShape<
      Shape::kK * kStages + Policy::SmemPaddingB::kRow,
      Shape::kN + Policy::SmemPaddingB::kColumn>;

  using SharedStorageA = OperandSharedStorage<
      typename Operator::ElementA,
      ShapeA,
      typename Operator::LayoutA>;
  using SharedStorageB = OperandSharedStorage<
      typename Operator::ElementB,
      ShapeB,
      typename Operator::LayoutB>;
  using TensorRefA = typename SharedStorageA::TensorRef;
  using TensorRefB = typename SharedStorageB::TensorRef;

  struct SharedStorage {
    /// Buffer for A operand
    SharedStorageA operand_A;

    /// Buffer for B operand
    SharedStorageB operand_B;
  };

 protected:
  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A operand from shared memory
  typename Operator::IteratorA warp_tile_iterator_A_;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  typename Operator::IteratorB warp_tile_iterator_B_;

 public:
  /// Construct from tensor references
  CUTLASS_DEVICE
  CustomMmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorageA& shared_storageA,
      SharedStorageB& shared_storageB,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx)
      : warp_tile_iterator_A_(shared_storageA.ref(), lane_idx),
        warp_tile_iterator_B_(shared_storageB.ref(), lane_idx) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `cutlass`, `gemm`, `threadblock`

**Classes/Structs**: `CustomMmaBase`, `OperandSharedStorage`, `SharedStorage`, `from`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cutlass/aligned_buffer.h`
- `cutlass/arch/memory.h`
- `cutlass/array.h`
- `cutlass/cutlass.h`
- `cutlass/gemm/gemm.h`
- `cutlass/gemm/threadblock/mma_base.h`
- `cutlass/matrix_shape.h`
- `cutlass/numeric_types.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm`):

- [`custom_mma_multistage.h_docs.md`](./custom_mma_multistage.h_docs.md)
- [`custom_mma.h_docs.md`](./custom_mma.h_docs.md)
- [`mma_from_smem.h_docs.md`](./mma_from_smem.h_docs.md)
- [`custom_mma_pipelined.h_docs.md`](./custom_mma_pipelined.h_docs.md)
- [`mma_accum_lambda_iterator.h_docs.md`](./mma_accum_lambda_iterator.h_docs.md)
- [`find_default_mma.h_docs.md`](./find_default_mma.h_docs.md)


## Cross-References

- **File Documentation**: `custom_mma_base.h_docs.md`
- **Keyword Index**: `custom_mma_base.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
