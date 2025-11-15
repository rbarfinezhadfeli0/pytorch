# Documentation: `docs/aten/src/ATen/native/SpectralOpsUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/SpectralOpsUtils.h_docs.md`
- **Size**: 5,859 bytes (5.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/SpectralOpsUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/SpectralOpsUtils.h`
- **Size**: 3,282 bytes (3.21 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <string>
#include <stdexcept>
#include <sstream>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/core/TensorBase.h>

namespace at::native {

// Normalization types used in _fft_with_size
enum class fft_norm_mode {
  none,       // No normalization
  by_root_n,  // Divide by sqrt(signal_size)
  by_n,       // Divide by signal_size
};

// NOTE [ Fourier Transform Conjugate Symmetry ]
//
// Real-to-complex Fourier transform satisfies the conjugate symmetry. That is,
// assuming X is the transformed K-dimensional signal, we have
//
//     X[i_1, ..., i_K] = X[j_i, ..., j_K]*,
//
//       where j_k  = (N_k - i_k)  mod N_k, N_k being the signal size at dim k,
//             * is the conjugate operator.
//
// Therefore, in such cases, FFT libraries return only roughly half of the
// values to avoid redundancy:
//
//     X[:, :, ..., :floor(N / 2) + 1]
//
// This is also the assumption in cuFFT and MKL. In ATen SpectralOps, such
// halved signal will also be returned by default (flag onesided=True).
// The following infer_ft_real_to_complex_onesided_size function calculates the
// onesided size from the twosided size.
//
// Note that this loses some information about the size of signal at last
// dimension. E.g., both 11 and 10 maps to 6. Hence, the following
// infer_ft_complex_to_real_onesided_size function takes in optional parameter
// to infer the twosided size from given onesided size.
//
// cuFFT doc: http://docs.nvidia.com/cuda/cufft/index.html#multi-dimensional
// MKL doc: https://software.intel.com/en-us/mkl-developer-reference-c-dfti-complex-storage-dfti-real-storage-dfti-conjugate-even-storage#CONJUGATE_EVEN_STORAGE

inline int64_t infer_ft_real_to_complex_onesided_size(int64_t real_size) {
  return (real_size / 2) + 1;
}

inline int64_t infer_ft_complex_to_real_onesided_size(int64_t complex_size,
                                                      int64_t expected_size=-1) {
  int64_t base = (complex_size - 1) * 2;
  if (expected_size < 0) {
    return base + 1;
  } else if (base == expected_size) {
    return base;
  } else if (base + 1 == expected_size) {
    return base + 1;
  } else {
    std::ostringstream ss;
    ss << "expected real signal size " << expected_size << " is incompatible "
       << "with onesided complex frequency size " << complex_size;
    TORCH_CHECK(false, ss.str());
  }
}

using fft_fill_with_conjugate_symmetry_fn =
    void (*)(ScalarType dtype, IntArrayRef mirror_dims, IntArrayRef half_sizes,
             IntArrayRef in_strides, const void* in_data,
             IntArrayRef out_strides, void* out_data);
DECLARE_DISPATCH(fft_fill_with_conjugate_symmetry_fn, fft_fill_with_conjugate_symmetry_stub)

// In real-to-complex transform, cuFFT and MKL only fill half of the values
// due to conjugate symmetry. This function fills in the other half of the full
// fft by using the Hermitian symmetry in the signal.
// self should be the shape of the full signal and dims.back() should be the
// one-sided dimension.
// See NOTE [ Fourier Transform Conjugate Symmetry ]
TORCH_API void _fft_fill_with_conjugate_symmetry_(const Tensor& self, IntArrayRef dims);

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `fft_norm_mode`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `string`
- `stdexcept`
- `sstream`
- `c10/core/ScalarType.h`
- `c10/util/ArrayRef.h`
- `c10/util/Exception.h`
- `ATen/native/DispatchStub.h`
- `ATen/core/TensorBase.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `SpectralOpsUtils.h_docs.md`
- **Keyword Index**: `SpectralOpsUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `SpectralOpsUtils.h_docs.md_docs.md`
- **Keyword Index**: `SpectralOpsUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
