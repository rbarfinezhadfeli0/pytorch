# Documentation: `c10/metal/expm1f.h`

## File Metadata

- **Path**: `c10/metal/expm1f.h`
- **Size**: 3,596 bytes (3.51 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Copy-and-pasted from:
// https://github.com/ml-explore/mlx/blob/99c33d011d63174f50cea37c3eede002958be6d3/mlx/backend/metal/kernels/expm1f.h

#pragma once

#include <metal_math>

// Original license copied below:
//  Copyright (c) 2015-2023 Norbert Juffa
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

namespace c10 {
namespace metal {

/* Compute exponential base e minus 1. Maximum ulp error = 0.997458

   i = rint(a/log(2)), f = a-i*log(2). Then expm1(a) = 2**i * (expm1(f)+1) - 1.
   Compute r = expm1(f). Then expm1(a)= 2 * (0.5 * 2**i * r + 0.5 * 2**i - 0.5).
   With t = 0.5*2**i, expm1(a) = 2*(r * t + t-0.5). However, for best accuracy,
   when i == 1, expm1(a)= 2*(r + 0.5), and when i == 0, expm1(a) = r.

   NOTE: Scale factor b is only applied if i < 0 or i > 1 (should be power of 2)
*/
inline float expm1f_scaled_unchecked(float a, float b) {
  float f, j, r, s, t, u, v, x, y;
  int i;

  // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
  j = ::metal::fma(1.442695f, a, 12582912.f); // 0x1.715476p0, 0x1.8p23
  j = j - 12582912.0f; // 0x1.8p23
  i = (int)j;
  f = ::metal::fma(j, -6.93145752e-1f, a);

  // approximate r = exp(f)-1 on interval [-log(2)/2, +log(2)/2]
  s = f * f;
  if (a == 0.0f)
    s = a; // ensure -0 is passed through
  // err = 0.997458  ulp1 = 11081805
  r = 1.97350979e-4f; // 0x1.9de000p-13
  r = ::metal::fma(r, f, 1.39309070e-3f); // 0x1.6d30bcp-10
  r = ::metal::fma(r, f, 8.33343994e-3f); // 0x1.1111f6p-7
  r = ::metal::fma(r, f, 4.16668020e-2f); // 0x1.55559ep-5
  r = ::metal::fma(r, f, 1.66666716e-1f); // 0x1.55555cp-3
  r = ::metal::fma(r, f, 4.99999970e-1f); // 0x1.fffffep-2
  u = (j == 1) ? (f + 0.5f) : f;
  v = ::metal::fma(r, s, u);
  s = 0.5f * b;
  t = ::metal::ldexp(s, i);
  y = t - s;
  x = (t - y) - s; // double-float canonicalization of difference
  r = ::metal::fma(v, t, x) + y;
  r = r + r;
  if (j == 0)
    r = v;
  if (j == 1)
    r = v + v;
  return r;
}

/* Compute exponential base e minus 1. max ulp err = 0.99746 */
inline float expm1f(float a) {
  float r;

  r = expm1f_scaled_unchecked(a, 1.0f);
  /* handle severe overflow and underflow */
  if (::metal::abs(a - 1.0f) > 88.0f) {
    r = ::metal::pow(2, a);
    r = ::metal::fma(r, r, -1.0f);
  }
  return r;
}

} // namespace metal
} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `metal`, `c10`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/metal`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `metal_math`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`c10/metal`):

- [`common.h_docs.md`](./common.h_docs.md)
- [`igamma.h_docs.md`](./igamma.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`error.h_docs.md`](./error.h_docs.md)
- [`reduction_utils.h_docs.md`](./reduction_utils.h_docs.md)
- [`special_math.h_docs.md`](./special_math.h_docs.md)
- [`indexing.h_docs.md`](./indexing.h_docs.md)
- [`atomic.h_docs.md`](./atomic.h_docs.md)
- [`random.h_docs.md`](./random.h_docs.md)


## Cross-References

- **File Documentation**: `expm1f.h_docs.md`
- **Keyword Index**: `expm1f.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
