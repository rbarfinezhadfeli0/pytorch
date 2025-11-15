# Documentation: `aten/src/ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_mma.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_mma.h`
- **Size**: 15,602 bytes (15.24 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma_multistage.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma_pipelined.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_mma_bf16.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp), fp16 activation & int8 weight
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultMma<cutlass::half_t,
                  LayoutA,
                  kAlignmentA,
                  uint8_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator> {

private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;

    using Mma = DqMma<half_t,
                      LayoutA,
                      kAlignmentA,
                      uint8_t,
                      LayoutB,
                      kAlignmentB,
                      half_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      2,
                      Operator>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output (OperatorClass TensorOp), fp16 activation & int4 weight
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultMma<cutlass::half_t,
                  LayoutA,
                  kAlignmentA,
                  uint4b_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator> {

private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;

    using Mma = DqMma<half_t,
                      LayoutA,
                      kAlignmentA,
                      uint4b_t,
                      LayoutB,
                      kAlignmentB,
                      half_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      2,
                      Operator>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    ///
    int kStages,
    /// Shared memory clear option
    SharedMemoryClearOption SharedMemoryClear>
struct DefaultMma<cutlass::half_t,
                  LayoutA,
                  kAlignmentA,
                  uint8_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  kStages,
                  Operator,
                  false,
                  SharedMemoryClear> {

private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;

    using Mma = DqMma<half_t,
                      LayoutA,
                      kAlignmentA,
                      uint8_t,
                      LayoutB,
                      kAlignmentB,
                      half_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      kStages,
                      Operator,
                      SharedMemoryClear>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output (OperatorClass TensorOp), fp16 activation & int4 weight
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    ///
    int kStages,
    /// Shared memory clear option
    SharedMemoryClearOption SharedMemoryClear>
struct DefaultMma<cutlass::half_t,
                  LayoutA,
                  kAlignmentA,
                  uint4b_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  kStages,
                  Operator,
                  false,
                  SharedMemoryClear> {

private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;

    using Mma = DqMma<half_t,
                      LayoutA,
                      kAlignmentA,
                      uint4b_t,
                      LayoutB,
                      kAlignmentB,
                      half_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      kStages,
                      Operator,
                      SharedMemoryClear>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

// fp16 x fp16 specialization on Ampere to use mma multistage for 2 stage. Helps avoid reg spills on
// large tile when not enough shared mem is present to do 3+ stage
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB>
struct DefaultMma<half_t,
                  LayoutA,
                  kAlignmentA,
                  half_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  arch::Sm80,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator,
                  false,
                  SharedMemoryClear,
                  GatherA,
                  GatherB> {

    // Define the MmaCore components
    // 3 is used on purpose here to trigger components for mma multistage
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        half_t,
                                                                        LayoutA,
                                                                        half_t,
                                                                        LayoutB,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        arch::OpClassTensorOp,
                                                                        3,
                                                                        Operator>;

    // Define iterators over tiles from the A operand
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<half_t, kAlignmentA>;
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        half_t,
        LayoutA,
        1,
        ThreadMapA,
        AccessTypeA,
        GatherA>;

    // Define iterators over tiles from the B operand
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<half_t, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        half_t,
        LayoutB,
        0,
        ThreadMapB,
        AccessTypeB,
        GatherB>;

    // Define the threadblock-scoped multistage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<typename MmaCore::Shape,
                                                                     IteratorA,
                                                                     typename MmaCore::SmemIteratorA,
                                                                     MmaCore::kCacheOpA,
                                                                     IteratorB,
                                                                     typename MmaCore::SmemIteratorB,
                                                                     MmaCore::kCacheOpB,
                                                                     ElementAccumulator,
                                                                     layout::RowMajor,
                                                                     typename MmaCore::MmaPolicy,
                                                                     2>;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `cutlass`, `gemm`, `threadblock`

**Classes/Structs**: `DefaultMma`, `DefaultMma`, `DefaultMma`, `DefaultMma`, `DefaultMma`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda/cutlass_extensions/gemm/threadblock`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma_multistage.h`
- `ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma_pipelined.h`
- `ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_mma_bf16.h`


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

Files in the same folder (`aten/src/ATen/native/cuda/cutlass_extensions/gemm/threadblock`):

- [`default_mma_bf16.h_docs.md`](./default_mma_bf16.h_docs.md)
- [`dq_mma_multistage.h_docs.md`](./dq_mma_multistage.h_docs.md)
- [`default_dq_mma_multistage.h_docs.md`](./default_dq_mma_multistage.h_docs.md)
- [`default_dq_mma_pipelined.h_docs.md`](./default_dq_mma_pipelined.h_docs.md)
- [`dq_mma_base.h_docs.md`](./dq_mma_base.h_docs.md)
- [`default_dq_mma.h_docs.md`](./default_dq_mma.h_docs.md)
- [`dq_mma_pipelined.h_docs.md`](./dq_mma_pipelined.h_docs.md)


## Cross-References

- **File Documentation**: `default_mma.h_docs.md`
- **Keyword Index**: `default_mma.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
