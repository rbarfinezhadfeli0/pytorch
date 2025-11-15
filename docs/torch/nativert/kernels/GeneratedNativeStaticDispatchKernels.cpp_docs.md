# Documentation: `torch/nativert/kernels/GeneratedNativeStaticDispatchKernels.cpp`

## File Metadata

- **Path**: `torch/nativert/kernels/GeneratedNativeStaticDispatchKernels.cpp`
- **Size**: 10,826 bytes (10.57 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// @generated
// @lint-ignore-every CLANGTIDY HOWTOEVEN
#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>

#include <torch/nativert/kernels/KernelRegistry.h>

#include <iterator>

namespace torch::nativert {

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.view_as_real.default",
    aten_view_as_real_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::view_as_real(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.view_as_complex.default",
    aten_view_as_complex_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::view_as_complex(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.real.default", aten_real_default, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::real(self);
  return;
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.imag.default", aten_imag_default, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::imag(self);
  return;
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten._conj.default", aten__conj_default, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::_conj(self);
  return;
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.conj.default", aten_conj_default, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::conj(self);
  return;
})

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.resolve_conj.default",
    aten_resolve_conj_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::resolve_conj(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.resolve_neg.default",
    aten_resolve_neg_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::resolve_neg(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten._neg_view.default",
    aten__neg_view_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::_neg_view(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.diagonal.default",
    aten_diagonal_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto offset = KernelInput(1).toInt();
      const auto dim1 = KernelInput(2).toInt();
      const auto dim2 = KernelInput(3).toInt();
      KernelOutput(0) = at::native::diagonal(self, offset, dim1, dim2);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.linalg_diagonal.default",
    aten_linalg_diagonal_default,
    {
      const auto& A = KernelInput(0).toTensor();
      const auto offset = KernelInput(1).toInt();
      const auto dim1 = KernelInput(2).toInt();
      const auto dim2 = KernelInput(3).toInt();
      KernelOutput(0) = at::native::linalg_diagonal(A, offset, dim1, dim2);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.expand_as.default",
    aten_expand_as_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      KernelOutput(0) = at::native::expand_as(self, other);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.flatten.using_ints",
    aten_flatten_using_ints,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto start_dim = KernelInput(1).toInt();
      const auto end_dim = KernelInput(2).toInt();
      KernelOutput(0) = at::native::flatten(self, start_dim, end_dim);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.movedim.int", aten_movedim_int, {
  const auto& self = KernelInput(0).toTensor();
  const auto source = KernelInput(1).toInt();
  const auto destination = KernelInput(2).toInt();
  KernelOutput(0) = at::native::movedim(self, source, destination);
  return;
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.moveaxis.int", aten_moveaxis_int, {
  const auto& self = KernelInput(0).toTensor();
  const auto source = KernelInput(1).toInt();
  const auto destination = KernelInput(2).toInt();
  KernelOutput(0) = at::native::moveaxis(self, source, destination);
  return;
})

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.numpy_T.default",
    aten_numpy_T_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::numpy_T(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.matrix_H.default",
    aten_matrix_H_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::matrix_H(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.mT.default", aten_mT_default, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::mT(self);
  return;
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.mH.default", aten_mH_default, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::mH(self);
  return;
})

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.adjoint.default",
    aten_adjoint_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::adjoint(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.ravel.default", aten_ravel_default, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::ravel(self);
  return;
})

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.reshape_as.default",
    aten_reshape_as_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      KernelOutput(0) = at::native::reshape_as(self, other);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.detach.default",
    aten_detach_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::detach(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.squeeze.default",
    aten_squeeze_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::squeeze(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.squeeze.dim", aten_squeeze_dim, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  KernelOutput(0) = at::native::squeeze(self, dim);
  return;
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.t.default", aten_t_default, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::t(self);
  return;
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.transpose.int", aten_transpose_int, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim0 = KernelInput(1).toInt();
  const auto dim1 = KernelInput(2).toInt();
  KernelOutput(0) = at::native::transpose(self, dim0, dim1);
  return;
})

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.unsqueeze.default",
    aten_unsqueeze_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim = KernelInput(1).toInt();
      KernelOutput(0) = at::native::unsqueeze(self, dim);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.view_as.default",
    aten_view_as_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      KernelOutput(0) = at::native::view_as(self, other);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.positive.default",
    aten_positive_default,
    {
      const auto& self = KernelInput(0).toTensor();
      KernelOutput(0) = at::native::positive(self);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten._autocast_to_reduced_precision.default",
    aten__autocast_to_reduced_precision_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto cuda_enabled = KernelInput(1).toBool();
      const auto cpu_enabled = KernelInput(2).toBool();
      const auto cuda_dtype = KernelInput(3).toScalarType();
      const auto cpu_dtype = KernelInput(4).toScalarType();
      KernelOutput(0) = at::native::_autocast_to_reduced_precision(
          self, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten._autocast_to_full_precision.default",
    aten__autocast_to_full_precision_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto cuda_enabled = KernelInput(1).toBool();
      const auto cpu_enabled = KernelInput(2).toBool();
      KernelOutput(0) = at::native::_autocast_to_full_precision(
          self, cuda_enabled, cpu_enabled);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.swapaxes.default",
    aten_swapaxes_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto axis0 = KernelInput(1).toInt();
      const auto axis1 = KernelInput(2).toInt();
      KernelOutput(0) = at::native::swapaxes(self, axis0, axis1);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.swapdims.default",
    aten_swapdims_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim0 = KernelInput(1).toInt();
      const auto dim1 = KernelInput(2).toInt();
      KernelOutput(0) = at::native::swapdims(self, dim0, dim1);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.unfold.default",
    aten_unfold_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dimension = KernelInput(1).toInt();
      const auto size = KernelInput(2).toInt();
      const auto step = KernelInput(3).toInt();
      KernelOutput(0) = at::native::unfold(self, dimension, size, step);
      return;
    })

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.alias.default", aten_alias_default, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::alias(self);
  return;
})

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/kernels`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/CPUFunctions.h`
- `ATen/InferSize.h`
- `ATen/NativeFunctions.h`
- `ATen/Parallel.h`
- `ATen/ScalarOps.h`
- `ATen/TensorUtils.h`
- `ATen/cpu/vec/functional.h`
- `ATen/cpu/vec/vec.h`
- `ATen/native/EmbeddingBag.h`
- `ATen/native/Fill.h`
- `ATen/native/IndexingUtils.h`
- `ATen/native/NonSymbolicBC.h`
- `ATen/native/Resize.h`
- `ATen/native/SharedReduceOps.h`
- `ATen/native/TensorAdvancedIndexing.h`
- `ATen/native/cpu/SerialStackImpl.h`
- `ATen/native/layer_norm.h`
- `ATen/native/quantized/cpu/fbgemm_utils.h`
- `ATen/native/quantized/cpu/qembeddingbag.h`
- `ATen/native/quantized/cpu/qembeddingbag_prepack.h`
- `ATen/quantized/QTensorImpl.h`
- `ATen/quantized/Quantizer.h`
- `c10/core/ScalarType.h`
- `c10/core/WrapDimMinimal.h`
- `c10/util/irange.h`
- `torch/nativert/kernels/KernelRegistry.h`
- `iterator`


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

Files in the same folder (`torch/nativert/kernels`):

- [`PrimKernelRegistry.cpp_docs.md`](./PrimKernelRegistry.cpp_docs.md)
- [`KernelRegistry.h_docs.md`](./KernelRegistry.h_docs.md)
- [`AutoFunctionalizeKernel.cpp_docs.md`](./AutoFunctionalizeKernel.cpp_docs.md)
- [`ETCallDelegateKernel.cpp_docs.md`](./ETCallDelegateKernel.cpp_docs.md)
- [`HigherOrderKernel.cpp_docs.md`](./HigherOrderKernel.cpp_docs.md)
- [`KernelHandlerRegistry.cpp_docs.md`](./KernelHandlerRegistry.cpp_docs.md)
- [`NativeKernels.cpp_docs.md`](./NativeKernels.cpp_docs.md)
- [`KernelFactory.cpp_docs.md`](./KernelFactory.cpp_docs.md)
- [`AutoFunctionalizeKernel.h_docs.md`](./AutoFunctionalizeKernel.h_docs.md)
- [`HigherOrderKernel.h_docs.md`](./HigherOrderKernel.h_docs.md)


## Cross-References

- **File Documentation**: `GeneratedNativeStaticDispatchKernels.cpp_docs.md`
- **Keyword Index**: `GeneratedNativeStaticDispatchKernels.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
