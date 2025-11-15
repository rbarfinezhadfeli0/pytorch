# Documentation: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_rescale_output.h`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_rescale_output.h`
- **Size**: 7,782 bytes (7.60 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory
  to match canonical tensor layouts in global memory. Epilogues support
  conversion and reduction operations.

  This is a copy of cutlass/epilogue/threadblock/epilogue.h that can
  handle "row_id" as a first argument, as uses it to get the corresponding
  `m_prime` / `s_prime` to rescale the output.
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <cassert>
#endif

#include <cutlass/aligned_buffer.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/layout/vector.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_coord.h>

#include <cutlass/gemm/gemm.h>

#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/transform/threadblock/regular_tile_iterator.h>

#include <cutlass/epilogue/threadblock/epilogue_base.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator.h>
#include <cutlass/numeric_types.h>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_pipelined.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
// output <- alpha * accumulator + beta * source
//   with:
//     alpha = 1 / s_prime (to normalize when isLast=True, 1 otherwise)
//     beta = alpha / m_prime (renormalize the output when the max changes)
//     source is the current output
template <
    typename ElementOutput_, ///< Data type used to store tensors
    typename ElementSource_, //< Data type for source (usually matches
                             //`ElementOutput`)
    int Count, ///< Number of elements computed per operation.
               ///< Usually it is 128/sizeof_bits<ElementOutput_>,
               ///< but we use 64 or 32 sometimes when there are not enough data
               ///< to store
    typename ElementAccumulator_, ///< Accumulator data type
    typename ElementCompute_, ///< Data type used to compute linear combination
    bool isFirst,
    bool isLast,
    typename FragmentAlphaBeta_,
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class MemoryEfficientAttentionNormalize {
 public:
  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;
  using FragmentAlphaBeta = FragmentAlphaBeta_;

  static FloatRoundStyle const kRound = Round;

 private:
  //
  // Data members
  //

  FragmentAlphaBeta const& s_prime_;
  FragmentAlphaBeta const& m_prime_;

 public:
  /// Constructs the function object, possibly loading from pointers in host
  /// memory
  CUTLASS_HOST_DEVICE
  MemoryEfficientAttentionNormalize(
      FragmentAlphaBeta const& s_prime,
      FragmentAlphaBeta const& m_prime)
      : s_prime_(s_prime), m_prime_(m_prime) {}

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return !isFirst;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      int row,
      FragmentAccumulator const& accumulator,
      FragmentSource const& source) const {
    assert(!isFirst);

    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round>
        source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;

    // Row sums for full masked out rows are 0, we set them to 1
    // In order to avoid NaNs in the output and instead sem them to 0.
    ElementCompute denom = s_prime_[row] == 0 ? 1 : s_prime_[row];
    ElementCompute alpha = isLast ? (1 / denom) : 1;
    ElementCompute beta = alpha * m_prime_[row];

    intermediate = mul_add_source(beta, converted_source); // X =  beta * C

    intermediate = mul_add_accumulator(
        alpha, converted_accumulator, intermediate); // D = alpha * Accum + X

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(int row, FragmentAccumulator const& accumulator)
      const {
    assert(isFirst);

    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    ComputeFragment intermediate;
    multiplies<ComputeFragment> mul_accumulator;

    // Row sums for full masked out rows are 0, we set them to 1
    // In order to avoid NaNs in the output and instead sem them to 0.
    ElementCompute denom = s_prime_[row] == 0 ? 1 : s_prime_[row];
    ElementCompute alpha = isLast ? (1 / denom) : 1;

    intermediate = mul_accumulator(
        alpha, converted_accumulator); // X =  alpha * C + uniform

    return destination_converter(intermediate);
  }
};

} // namespace thread

namespace threadblock {
template <
    typename EO,
    typename ES,
    int Count,
    typename EA,
    typename EC,
    bool F,
    bool L,
    typename FAB,
    FloatRoundStyle R>
struct ApplyEpilogueOp<thread::MemoryEfficientAttentionNormalize<
    EO,
    ES,
    Count,
    EA,
    EC,
    F,
    L,
    FAB,
    R>> {
  using Op = thread::
      MemoryEfficientAttentionNormalize<EO, ES, Count, EA, EC, F, L, FAB, R>;
  static CUTLASS_DEVICE typename Op::FragmentOutput apply(
      Op const& output_op,
      int row_id,
      typename Op::FragmentAccumulator const& accum,
      typename Op::FragmentSource const& source) {
    return output_op(row_id, accum, source);
  }
  static CUTLASS_DEVICE typename Op::FragmentOutput apply(
      Op const& output_op,
      int row_id,
      typename Op::FragmentAccumulator const& accum) {
    return output_op(row_id, accum);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `threadblock`, `cutlass`, `epilogue`, `thread`

**Classes/Structs**: `MemoryEfficientAttentionNormalize`, `ApplyEpilogueOp`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cuda/std/cassert`
- `cassert`
- `cutlass/aligned_buffer.h`
- `cutlass/array.h`
- `cutlass/cutlass.h`
- `cutlass/functional.h`
- `cutlass/layout/tensor.h`
- `cutlass/layout/vector.h`
- `cutlass/numeric_types.h`
- `cutlass/tensor_coord.h`
- `cutlass/gemm/gemm.h`
- `cutlass/transform/pitch_linear_thread_map.h`
- `cutlass/transform/threadblock/regular_tile_iterator.h`
- `cutlass/epilogue/threadblock/epilogue_base.h`
- `cutlass/epilogue/threadblock/predicated_tile_iterator.h`
- `cutlass/numeric_types.h`
- `cutlass/array.h`
- `cutlass/cutlass.h`
- `cutlass/epilogue/thread/scale_type.h`
- `cutlass/functional.h`
- `cutlass/numeric_conversion.h`
- `cutlass/numeric_types.h`
- `ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_pipelined.h`


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

Files in the same folder (`aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue`):

- [`epilogue_pipelined.h_docs.md`](./epilogue_pipelined.h_docs.md)
- [`epilogue_thread_apply_logsumexp.h_docs.md`](./epilogue_thread_apply_logsumexp.h_docs.md)


## Cross-References

- **File Documentation**: `epilogue_rescale_output.h_docs.md`
- **Keyword Index**: `epilogue_rescale_output.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
