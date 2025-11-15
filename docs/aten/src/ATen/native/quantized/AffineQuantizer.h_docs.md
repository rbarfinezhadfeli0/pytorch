# Documentation: `aten/src/ATen/native/quantized/AffineQuantizer.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/AffineQuantizer.h`
- **Size**: 3,715 bytes (3.63 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>

namespace at::native {

TORCH_API Tensor& quantize_tensor_per_tensor_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point);
TORCH_API Tensor& quantize_tensor_per_channel_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    Tensor zero_points,
    int64_t axis);

TORCH_API Tensor& quantize_tensor_per_channel_float_qparams(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

TORCH_API Tensor& dequantize_tensor_per_tensor_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point);
TORCH_API Tensor& dequantize_tensor_per_channel_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    Tensor zero_points,
    int64_t axis);
TORCH_API Tensor& dequantize_tensor_per_channel_float_qparams(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using quantize_tensor_per_tensor_affine_fn =
    void (*)(const Tensor& rtensor, Tensor& qtensor, double scale, int64_t zero_point);

using quantize_tensor_per_channel_affine_fn = void (*)(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using quantize_tensor_per_channel_float_qparams_fn = void (*)(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using dequantize_tensor_per_tensor_affine_fn =
    void (*)(const Tensor& qtensor, Tensor& rtensor, double scale, int64_t zero_point);

using dequantize_tensor_per_channel_affine_fn = void (*)(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using dequantize_tensor_per_channel_float_qparams_fn = void (*)(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using quantize_tensor_per_tensor_affine_sub_byte_fn =
    void (*)(const Tensor& rtensor, Tensor& qtensor, float scale, float zero_point);

using dequantize_tensor_per_tensor_affine_sub_byte_fn =
    void (*)(const Tensor& qtensor, Tensor& rtensor, float scale, float zero_point);

DECLARE_DISPATCH(
    quantize_tensor_per_tensor_affine_fn,
    quantize_tensor_per_tensor_affine_stub)
DECLARE_DISPATCH(
    quantize_tensor_per_channel_affine_fn,
    quantize_tensor_per_channel_affine_stub)
DECLARE_DISPATCH(
    quantize_tensor_per_channel_float_qparams_fn,
    quantize_tensor_per_channel_float_qparams_stub)

DECLARE_DISPATCH(
    dequantize_tensor_per_tensor_affine_fn,
    dequantize_tensor_per_tensor_affine_stub)
DECLARE_DISPATCH(
    dequantize_tensor_per_channel_affine_fn,
    dequantize_tensor_per_channel_affine_stub)
DECLARE_DISPATCH(
    dequantize_tensor_per_channel_float_qparams_fn,
    dequantize_tensor_per_channel_float_qparams_stub)

DECLARE_DISPATCH(
    quantize_tensor_per_tensor_affine_sub_byte_fn,
    quantize_tensor_per_tensor_affine_sub_byte_stub)

DECLARE_DISPATCH(
    dequantize_tensor_per_tensor_affine_sub_byte_fn,
    dequantize_tensor_per_tensor_affine_sub_byte_stub)

template <typename T>
TORCH_API Tensor quantize_tensor(
    Tensor rtensor,
    Tensor qtensor,
    double scale,
    int64_t zero_point);
template <typename T>
TORCH_API Tensor dequantize_tensor(
    Tensor qtensor,
    Tensor rtensor,
    double scale,
    int64_t zero_point);

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/quantized/AffineQuantizerBase.h`


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

Files in the same folder (`aten/src/ATen/native/quantized`):

- [`QTensor.cpp_docs.md`](./QTensor.cpp_docs.md)
- [`FakeQuantAffine.h_docs.md`](./FakeQuantAffine.h_docs.md)
- [`qconv_unpack.cpp_docs.md`](./qconv_unpack.cpp_docs.md)
- [`AffineQuantizerBase.h_docs.md`](./AffineQuantizerBase.h_docs.md)
- [`AffineQuantizerBase.cpp_docs.md`](./AffineQuantizerBase.cpp_docs.md)
- [`PackedParams.h_docs.md`](./PackedParams.h_docs.md)
- [`Copy.cpp_docs.md`](./Copy.cpp_docs.md)
- [`qlinear_unpack.cpp_docs.md`](./qlinear_unpack.cpp_docs.md)
- [`TensorFactories.cpp_docs.md`](./TensorFactories.cpp_docs.md)


## Cross-References

- **File Documentation**: `AffineQuantizer.h_docs.md`
- **Keyword Index**: `AffineQuantizer.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
