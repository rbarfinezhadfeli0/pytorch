# Documentation: `docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_thread_apply_logsumexp.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_thread_apply_logsumexp.h_docs.md`
- **Size**: 8,640 bytes (8.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_thread_apply_logsumexp.h`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_thread_apply_logsumexp.h`
- **Size**: 6,123 bytes (5.98 KB)
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
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include <cuda_fp16.h>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, int ElementsPerAccess>
struct ArrayExponential {
  CUTLASS_HOST_DEVICE
  Array<Element, ElementsPerAccess> operator()(
      Array<Element, ElementsPerAccess> const& input) const {
    Array<Element, ElementsPerAccess> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = expf(input[i]);
    }

    return result;
  }
};

template <int ElementsPerAccess>
struct ArrayExponential<half_t, ElementsPerAccess> {
  CUTLASS_DEVICE
  Array<half_t, ElementsPerAccess> operator()(
      Array<half_t, ElementsPerAccess> const& input) const {
    Array<half_t, ElementsPerAccess> result;

    int const kVectorCount = ElementsPerAccess / 2;

    __half2 const* input_ptr =
        reinterpret_cast<__half2 const*>(input.raw_data());
    __half2* res_ptr = reinterpret_cast<__half2*>(result.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = h2exp(input_ptr[i]);
    }

    return result;
  }
};
} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies:
/// output <- (input - lse).exp()
template <
    typename ElementOutput_, // output
    typename ElementLSE_, // accumulator from LSE
    typename ElementAccumulator_, // accumulator from matmul
    typename ElementCompute_, // intermediate compute (and exp calculation)
    int ElementsPerAccess>
class ApplyLogSumExp {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementLSE = ElementLSE_;

  static int constexpr kElementsPerAccess = ElementsPerAccess;
  static int constexpr kCount = kElementsPerAccess;
  static constexpr ScaleType::Kind kScale =
      cutlass::epilogue::thread::ScaleType::NoBetaScaling;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentLSE = Array<ElementLSE, kElementsPerAccess>;
  using FragmentScaleBias = FragmentLSE; // Used by epilogue_smem_accumulator.h

 public:
  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  ApplyLogSumExp() {}

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return true;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& AB,
      FragmentLSE const& scale_unused,
      // bias used as LSE
      FragmentLSE const& bias) const {
    FragmentCompute frag_AB = NumericArrayConverter<
        ElementCompute,
        ElementAccumulator,
        kElementsPerAccess>()(AB);
    FragmentCompute frag_lse_compute =
        NumericArrayConverter<ElementCompute, ElementLSE, kElementsPerAccess>()(
            bias);
    FragmentCompute frag_compute;

    minus<FragmentCompute> minus_lse;
    detail::ArrayExponential<ElementCompute, kElementsPerAccess> apply_exp;
    frag_compute = minus_lse(frag_AB, frag_lse_compute);
    frag_compute = apply_exp(frag_compute);

    return NumericArrayConverter<
        ElementOutput,
        ElementCompute,
        kElementsPerAccess>()(frag_compute);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `cutlass`, `epilogue`, `detail`, `thread`

**Classes/Structs**: `ArrayExponential`, `ArrayExponential`, `ApplyLogSumExp`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cuda_fp16.h`
- `cutlass/array.h`
- `cutlass/cutlass.h`
- `cutlass/epilogue/thread/activation.h`
- `cutlass/functional.h`
- `cutlass/numeric_conversion.h`
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

Files in the same folder (`aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue`):

- [`epilogue_pipelined.h_docs.md`](./epilogue_pipelined.h_docs.md)
- [`epilogue_rescale_output.h_docs.md`](./epilogue_rescale_output.h_docs.md)


## Cross-References

- **File Documentation**: `epilogue_thread_apply_logsumexp.h_docs.md`
- **Keyword Index**: `epilogue_thread_apply_logsumexp.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue`):

- [`epilogue_pipelined.h_kw.md_docs.md`](./epilogue_pipelined.h_kw.md_docs.md)
- [`epilogue_thread_apply_logsumexp.h_kw.md_docs.md`](./epilogue_thread_apply_logsumexp.h_kw.md_docs.md)
- [`epilogue_rescale_output.h_docs.md_docs.md`](./epilogue_rescale_output.h_docs.md_docs.md)
- [`epilogue_pipelined.h_docs.md_docs.md`](./epilogue_pipelined.h_docs.md_docs.md)
- [`epilogue_rescale_output.h_kw.md_docs.md`](./epilogue_rescale_output.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `epilogue_thread_apply_logsumexp.h_docs.md_docs.md`
- **Keyword Index**: `epilogue_thread_apply_logsumexp.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
