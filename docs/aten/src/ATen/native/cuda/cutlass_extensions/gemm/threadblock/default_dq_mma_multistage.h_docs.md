# Documentation: `aten/src/ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma_multistage.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma_multistage.h`
- **Size**: 17,661 bytes (17.25 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <cutlass/gemm/threadblock/default_mma.h>
#include <ATen/native/cuda/cutlass_extensions/arch/mma.h>

#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/dq_mma_multistage.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/warp/default_mma_tensor_op.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h>
#include <ATen/native/cuda/cutlass_extensions/tile_interleaved_layout.h>

#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template<
    /// Type for elementA
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator,
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
    /// Stages in GEMM
    int kStages,
    ///
    typename Operator,
    ///
    SharedMemoryClearOption SharedMemoryClear>
struct DqMma<ElementA,
             LayoutA,
             kAlignmentA,
             ElementB,
             LayoutB,
             kAlignmentB,
             ElementScale,
             LayoutScale,
             kAlignmentScale,
             ElementAccumulator,
             layout::RowMajor,
             OperatorClass,
             ArchTag,
             ThreadblockShape,
             WarpShape,
             InstructionShape,
             kStages,
             Operator,
             SharedMemoryClear,
             typename platform::enable_if<(ArchTag::kMinComputeCapability >= 80)>::type> {

    static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                  "Element A must be fp16 or bf16");

    static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
                  "Mma multistage must dequantize after ldsm");

    static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                  "Element B must be uint8 or uint4");

    static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    // Mma core does not depend on stages, so pass in at least 3 here to mma multistage pieces are created
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        ElementA,
                                                                        LayoutA,
                                                                        ElementB,
                                                                        LayoutB,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        OperatorClass,
                                                                        std::max(kStages, 3),
                                                                        Operator,
                                                                        false,
                                                                        CacheOpA,
                                                                        CacheOpB>;

    // Define iterators over tiles from the A operand
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        1,
        ThreadMapA,
        AccessTypeA>;

    // Define iterators over tiles from the B operand
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        0,
        ThreadMapB,
        AccessTypeB>;

    // ThreadMap for scale iterator
    static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
    using IteratorScaleThreadMap =
        transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                                                  MmaCore::Shape::kN / kAlignmentScale,
                                                  kAlignmentScale>;

    // Define iterators over tiles from the scale operand
    using IteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                ElementScale,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;

    using SmemIteratorScale = IteratorScale;

    using Converter = FastInterleavedAndBiasedNumericArrayConverter<ElementA,
                                                                    ElementB,
                                                                    MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape,
                                                                       IteratorA,
                                                                       typename MmaCore::SmemIteratorA,
                                                                       MmaCore::kCacheOpA,
                                                                       IteratorB,
                                                                       typename MmaCore::SmemIteratorB,
                                                                       MmaCore::kCacheOpB,
                                                                       IteratorScale,
                                                                       SmemIteratorScale,
                                                                       ElementAccumulator,
                                                                       layout::RowMajor,
                                                                       typename MmaCore::MmaPolicy,
                                                                       kStages,
                                                                       Converter,
                                                                       SharedMemoryClear>;
};

template<
    /// Type for element A
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator,
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
    /// Stages in GEMM
    int kStages,
    ///
    typename Operator,
    ///
    SharedMemoryClearOption SharedMemoryClear,
    ///
    int RowsPerTile,
    ///
    int ColumnsInterleaved>
struct DqMma<ElementA,
             LayoutA,
             kAlignmentA,
             ElementB,
             layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>,
             kAlignmentB,
             ElementScale,
             LayoutScale,
             kAlignmentScale,
             ElementAccumulator,
             layout::RowMajor,
             OperatorClass,
             ArchTag,
             ThreadblockShape,
             WarpShape,
             InstructionShape,
             kStages,
             Operator,
             SharedMemoryClear,
             typename platform::enable_if<(ArchTag::kMinComputeCapability >= 80)>::type> {

    static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                  "Element A must be fp16 or bf16");

    static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
                  "Mma multistage must dequantize after ldsm");

    static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                  "Element B must be uint8 or uint4");

    static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    // Mma core does not depend on stages, so pass in at least 3 here to mma multistage pieces are created
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        ElementA,
                                                                        LayoutA,
                                                                        ElementB,
                                                                        layout::ColumnMajor,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        OperatorClass,
                                                                        std::max(kStages, 3),
                                                                        Operator,
                                                                        false,
                                                                        CacheOpA,
                                                                        CacheOpB>;

    // Define iterators over tiles from the A operand
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        1,
        ThreadMapA,
        AccessTypeA>;

private:
    static_assert(!(MmaCore::Shape::kN % ColumnsInterleaved), "");
    static_assert(RowsPerTile == MmaCore::Shape::kK, "");

    using OriginalThreadMap       = typename MmaCore::IteratorThreadMapB;
    using OriginalWarpArrangement = typename OriginalThreadMap::Detail::WarpThreadArrangement;
    static_assert(!(OriginalWarpArrangement::kStrided % ColumnsInterleaved), "");

    using GmemIteratorShape =
        MatrixShape<MmaCore::Shape::kK * ColumnsInterleaved, MmaCore::Shape::kN / ColumnsInterleaved>;
    using GmemThreadMapB = transform::PitchLinearWarpRakedThreadMap<
        layout::PitchLinearShape<GmemIteratorShape::kRow, GmemIteratorShape::kColumn>,
        OriginalThreadMap::kThreads,
        layout::PitchLinearShape<OriginalWarpArrangement::kContiguous * ColumnsInterleaved,
                                 OriginalWarpArrangement::kStrided / ColumnsInterleaved>,
        MmaCore::kAccessSizeInBits / sizeof_bits<ElementB>::value>;

public:
    // Define iterators over tiles from the B operand
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::
        PredicatedTileAccessIterator<GmemIteratorShape, ElementB, layout::ColumnMajor, 0, GmemThreadMapB, AccessTypeB>;

    // ThreadMap for scale iterator
    static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
    using IteratorScaleThreadMap =
        transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                                                  MmaCore::Shape::kN / kAlignmentScale,
                                                  kAlignmentScale>;

    // Define iterators over tiles from the scale operand
    using IteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                ElementScale,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;

    using SmemIteratorScale = IteratorScale;

    using Converter = FastInterleavedAndBiasedNumericArrayConverter<ElementA,
                                                                    ElementB,
                                                                    MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape,
                                                                       IteratorA,
                                                                       typename MmaCore::SmemIteratorA,
                                                                       MmaCore::kCacheOpA,
                                                                       IteratorB,
                                                                       typename MmaCore::SmemIteratorB,
                                                                       MmaCore::kCacheOpB,
                                                                       IteratorScale,
                                                                       SmemIteratorScale,
                                                                       ElementAccumulator,
                                                                       layout::RowMajor,
                                                                       typename MmaCore::MmaPolicy,
                                                                       kStages,
                                                                       Converter,
                                                                       SharedMemoryClear>;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `cutlass`, `gemm`, `threadblock`

**Classes/Structs**: `tag`, `DqMma`, `tag`, `DqMma`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda/cutlass_extensions/gemm/threadblock`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cutlass/gemm/threadblock/default_mma.h`
- `ATen/native/cuda/cutlass_extensions/arch/mma.h`
- `ATen/native/cuda/cutlass_extensions/gemm/threadblock/dq_mma_multistage.h`
- `ATen/native/cuda/cutlass_extensions/gemm/warp/default_mma_tensor_op.h`
- `ATen/native/cuda/cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h`
- `ATen/native/cuda/cutlass_extensions/tile_interleaved_layout.h`
- `ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma.h`


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

- [`default_mma.h_docs.md`](./default_mma.h_docs.md)
- [`default_mma_bf16.h_docs.md`](./default_mma_bf16.h_docs.md)
- [`dq_mma_multistage.h_docs.md`](./dq_mma_multistage.h_docs.md)
- [`default_dq_mma_pipelined.h_docs.md`](./default_dq_mma_pipelined.h_docs.md)
- [`dq_mma_base.h_docs.md`](./dq_mma_base.h_docs.md)
- [`default_dq_mma.h_docs.md`](./default_dq_mma.h_docs.md)
- [`dq_mma_pipelined.h_docs.md`](./dq_mma_pipelined.h_docs.md)


## Cross-References

- **File Documentation**: `default_dq_mma_multistage.h_docs.md`
- **Keyword Index**: `default_dq_mma_multistage.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
