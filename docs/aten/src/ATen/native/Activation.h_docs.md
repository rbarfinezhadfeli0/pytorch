# Documentation: `aten/src/ATen/native/Activation.h`

## File Metadata

- **Path**: `aten/src/ATen/native/Activation.h`
- **Size**: 3,515 bytes (3.43 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/native/DispatchStub.h>
#include <ATen/native/Gelu.h>
#include <c10/util/Exception.h>

namespace c10 {
class Scalar;
}

namespace at {
struct TensorIterator;
struct TensorIteratorBase;
class TensorBase;
}

namespace at::native {

using structured_activation_fn = void (*)(TensorIteratorBase&);
using structured_activation_backward_fn = void (*)(TensorIteratorBase&);

using activation_fn = void (*)(TensorIterator&);
using activation_backward_fn = void (*)(TensorIterator&);
using softplus_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
using softplus_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
using threshold_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
using hardtanh_backward_fn = void (*)(TensorIterator&, const c10::Scalar&, const c10::Scalar&);
using hardsigmoid_fn = void(*)(TensorIteratorBase&);
using hardsigmoid_backward_fn = void(*)(TensorIteratorBase&);
using hardswish_fn = void(*)(TensorIterator&);
using hardswish_backward_fn = void(*)(TensorIterator&);
using shrink_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using softshrink_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using shrink_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using elu_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&, const c10::Scalar&);
using elu_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&, const c10::Scalar&, bool);
using leaky_relu_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using leaky_relu_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using log_sigmoid_cpu_fn = void (*)(TensorBase&, TensorBase&, const TensorBase&);
using gelu_fn = void (*)(TensorIteratorBase&, GeluType);
using gelu_backward_fn = void (*)(TensorIteratorBase&, GeluType);
using glu_jvp_fn = void (*)(TensorIteratorBase&);

DECLARE_DISPATCH(elu_fn, elu_stub)
DECLARE_DISPATCH(elu_backward_fn, elu_backward_stub)
DECLARE_DISPATCH(softplus_fn, softplus_stub)
DECLARE_DISPATCH(softplus_backward_fn, softplus_backward_stub)
DECLARE_DISPATCH(log_sigmoid_cpu_fn, log_sigmoid_cpu_stub)
DECLARE_DISPATCH(activation_backward_fn, log_sigmoid_backward_stub)
DECLARE_DISPATCH(threshold_fn, threshold_stub)
DECLARE_DISPATCH(gelu_fn, GeluKernel)
DECLARE_DISPATCH(gelu_backward_fn, GeluBackwardKernel)
DECLARE_DISPATCH(hardtanh_backward_fn, hardtanh_backward_stub)
DECLARE_DISPATCH(hardsigmoid_fn, hardsigmoid_stub)
DECLARE_DISPATCH(hardsigmoid_backward_fn, hardsigmoid_backward_stub)
DECLARE_DISPATCH(hardswish_fn, hardswish_stub)
DECLARE_DISPATCH(hardswish_backward_fn, hardswish_backward_stub)
DECLARE_DISPATCH(shrink_fn, hardshrink_stub)
DECLARE_DISPATCH(softshrink_fn, softshrink_stub)
DECLARE_DISPATCH(shrink_backward_fn, shrink_backward_stub)
DECLARE_DISPATCH(leaky_relu_fn, leaky_relu_stub)
DECLARE_DISPATCH(leaky_relu_backward_fn, leaky_relu_backward_stub)
DECLARE_DISPATCH(structured_activation_fn, glu_stub)
DECLARE_DISPATCH(activation_backward_fn, glu_backward_stub)
DECLARE_DISPATCH(glu_jvp_fn, glu_jvp_stub)
DECLARE_DISPATCH(structured_activation_fn, silu_stub)
DECLARE_DISPATCH(structured_activation_backward_fn, silu_backward_stub)
DECLARE_DISPATCH(structured_activation_fn, mish_stub)
DECLARE_DISPATCH(activation_backward_fn, mish_backward_stub)
DECLARE_DISPATCH(activation_fn, prelu_stub)
DECLARE_DISPATCH(activation_backward_fn, prelu_backward_stub)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`, `c10`

**Classes/Structs**: `Scalar`, `TensorIterator`, `TensorIteratorBase`, `TensorBase`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/DispatchStub.h`
- `ATen/native/Gelu.h`
- `c10/util/Exception.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `Activation.h_docs.md`
- **Keyword Index**: `Activation.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
