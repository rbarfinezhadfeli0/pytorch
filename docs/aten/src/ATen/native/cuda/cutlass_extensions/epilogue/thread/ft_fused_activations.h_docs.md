# Documentation: `aten/src/ATen/native/cuda/cutlass_extensions/epilogue/thread/ft_fused_activations.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/cutlass_extensions/epilogue/thread/ft_fused_activations.h`
- **Size**: 3,215 bytes (3.14 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination with a maximum operation used by epilogues.
*/

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/functional.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__forceinline__ __device__ float tanh_opt(float x)
{
#if (__CUDACC_VER_MAJOR__ < 11) || (__CUDA_ARCH__ < 750)
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#else
    return fast_tanh(x);
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `cutlass`, `epilogue`, `thread`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda/cutlass_extensions/epilogue/thread`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cutlass/array.h`
- `cutlass/cutlass.h`
- `cutlass/epilogue/thread/activation.h`
- `cutlass/epilogue/thread/scale_type.h`
- `cutlass/functional.h`
- `cutlass/half.h`
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

Files in the same folder (`aten/src/ATen/native/cuda/cutlass_extensions/epilogue/thread`):



## Cross-References

- **File Documentation**: `ft_fused_activations.h_docs.md`
- **Keyword Index**: `ft_fused_activations.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
