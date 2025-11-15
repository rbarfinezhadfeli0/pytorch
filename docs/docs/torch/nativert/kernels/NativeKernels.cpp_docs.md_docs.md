# Documentation: `docs/torch/nativert/kernels/NativeKernels.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/kernels/NativeKernels.cpp_docs.md`
- **Size**: 6,463 bytes (6.31 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/kernels/NativeKernels.cpp`

## File Metadata

- **Path**: `torch/nativert/kernels/NativeKernels.cpp`
- **Size**: 3,955 bytes (3.86 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nativert/kernels/KernelRegistry.h>

#include <ATen/NativeFunctions.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>

namespace torch::nativert {

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.slice.Tensor", aten_slice_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& dim = KernelInput(1).toInt();
  const auto& start = KernelInput(2).toOptional<int64_t>();
  const auto& end = KernelInput(3).toOptional<int64_t>();
  const auto& step = KernelInput(4).toInt();
  KernelOutput(0) = at::native::slice(self, dim, start, end, step);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.sym_size.int", aten_sym_size_int, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  auto& out = KernelOutput(0);
  TORCH_CHECK(dim >= 0 && dim < self.dim(), "Invalid dimension");
  out = self.sym_size(dim);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.reshape.default", aten_reshape, {
  const auto& self = KernelInput(0).toTensor();
  const auto& shape = KernelInput(1).toIntVector();
  KernelOutput(0) = at::native::reshape(self, shape);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.view.default", aten_view, {
  const auto& self = KernelInput(0).toTensor();
  const auto& size = KernelInput(1).toIntVector();
  KernelOutput(0) = at::native::view(self, size);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.permute.default", aten_permute, {
  const auto& self = KernelInput(0).toTensor();
  const auto& dims = KernelInput(1).toDimVector();
  KernelOutput(0) = at::native::permute(self, dims);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.select.int", aten_select, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto index = KernelInput(2).toInt();
  KernelOutput(0) = at::native::select(self, dim, index);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.split.Tensor", aten_split_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto split_size = KernelInput(1).toInt();
  const auto dim = KernelInput(2).toInt();
  KernelOutput(0) = at::native::split(self, split_size, dim);
})

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.split_with_sizes.default",
    aten_split_with_sizes,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& split_sizes = KernelInput(1).toIntList();
      const auto dim = KernelInput(2).toInt();
      KernelOutput(0) =
          at::native::split_with_sizes(self, split_sizes.vec(), dim);
    })

REGISTER_NATIVE_CPU_KERNEL(
    "torch.ops.aten.tensor_split.sections",
    aten_tensor_split_sections,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto sections = KernelInput(1).toInt();
      const auto dim = KernelInput(2).toInt();
      KernelOutput(0) =
          at::native::tensor_split_sections_symint(self, sections, dim);
    })

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.item.default", aten_item, {
  const auto& self = KernelInput(0).toTensor();
  KernelOutput(0) = at::native::item(self);
})

REGISTER_NATIVE_CPU_KERNEL("torch.ops.aten.narrow.default", aten_narrow, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  int64_t start = 0;
  if (KernelInput(2).isScalar()) {
    start = KernelInput(2).toInt();
  } else {
    auto& t = KernelInput(2).toTensor();
    start = t.item<int64_t>();
  }
  const auto length = KernelInput(3).toInt();
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.sizes()[dim];
  if (start != cur_size && start < 0) {
    start = at::maybe_wrap_dim(start, cur_size);
  }
  TORCH_CHECK(
      length >= 0 && start <= cur_size - length,
      "start (",
      start,
      ") + length (",
      length,
      ") exceeds dimension size (",
      cur_size,
      ").");
  KernelOutput(0) = at::native::slice(self, dim, start, start + length, 1);
})

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

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

- `torch/nativert/kernels/KernelRegistry.h`
- `ATen/NativeFunctions.h`
- `ATen/native/IndexingUtils.h`
- `ATen/native/NonSymbolicBC.h`


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

Files in the same folder (`torch/nativert/kernels`):

- [`PrimKernelRegistry.cpp_docs.md`](./PrimKernelRegistry.cpp_docs.md)
- [`KernelRegistry.h_docs.md`](./KernelRegistry.h_docs.md)
- [`AutoFunctionalizeKernel.cpp_docs.md`](./AutoFunctionalizeKernel.cpp_docs.md)
- [`ETCallDelegateKernel.cpp_docs.md`](./ETCallDelegateKernel.cpp_docs.md)
- [`HigherOrderKernel.cpp_docs.md`](./HigherOrderKernel.cpp_docs.md)
- [`KernelHandlerRegistry.cpp_docs.md`](./KernelHandlerRegistry.cpp_docs.md)
- [`KernelFactory.cpp_docs.md`](./KernelFactory.cpp_docs.md)
- [`AutoFunctionalizeKernel.h_docs.md`](./AutoFunctionalizeKernel.h_docs.md)
- [`HigherOrderKernel.h_docs.md`](./HigherOrderKernel.h_docs.md)


## Cross-References

- **File Documentation**: `NativeKernels.cpp_docs.md`
- **Keyword Index**: `NativeKernels.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/kernels`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/kernels`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/nativert/kernels`):

- [`ETCallDelegateKernel.cpp_docs.md_docs.md`](./ETCallDelegateKernel.cpp_docs.md_docs.md)
- [`TritonKernel.h_kw.md_docs.md`](./TritonKernel.h_kw.md_docs.md)
- [`PrimKernelRegistry.cpp_kw.md_docs.md`](./PrimKernelRegistry.cpp_kw.md_docs.md)
- [`C10Kernel.h_kw.md_docs.md`](./C10Kernel.h_kw.md_docs.md)
- [`CallTorchBindKernel.cpp_docs.md_docs.md`](./CallTorchBindKernel.cpp_docs.md_docs.md)
- [`ETCallDelegateKernel.h_kw.md_docs.md`](./ETCallDelegateKernel.h_kw.md_docs.md)
- [`AutoFunctionalizeKernel.h_kw.md_docs.md`](./AutoFunctionalizeKernel.h_kw.md_docs.md)
- [`HigherOrderKernel.h_docs.md_docs.md`](./HigherOrderKernel.h_docs.md_docs.md)
- [`C10Kernel.h_docs.md_docs.md`](./C10Kernel.h_docs.md_docs.md)
- [`CallTorchBindKernel.h_kw.md_docs.md`](./CallTorchBindKernel.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `NativeKernels.cpp_docs.md_docs.md`
- **Keyword Index**: `NativeKernels.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
