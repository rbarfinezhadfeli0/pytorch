# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Common.h`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Common.h`
- **Size**: 2,865 bytes (2.80 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```c
#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/blob.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/view_native.h>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function_hook.h>

#include <c10/core/Allocator.h>

#include <include/openreg.h>

namespace at::native::openreg {

class MemoryGuard {
 public:
  template <typename... Args>
  explicit MemoryGuard(const Args&... args) {
    (find_and_unprotect_tensors(args), ...);
  }

  ~MemoryGuard() noexcept {
    for (void* ptr : unprotected_pointers_) {
      orMemoryProtect(ptr);
    }
  }

  MemoryGuard(const MemoryGuard&) = delete;
  MemoryGuard& operator=(const MemoryGuard&) = delete;
  MemoryGuard(MemoryGuard&&) = delete;
  MemoryGuard& operator=(MemoryGuard&&) = delete;

 private:
  template <typename T>
  void find_and_unprotect_tensors(const T& item) {
    if constexpr (std::is_base_of_v<at::TensorBase, T>) {
      unprotect_if_needed(item);
    } else if constexpr (std::is_same_v<T, c10::IValue>) {
      if (item.isTensor()) {
        unprotect_if_needed(item.toTensor());
      } else if (item.isTensorList()) {
        for (const at::Tensor& tensor : item.toTensorListRef()) {
          unprotect_if_needed(tensor);
        }
      } else if (item.isList()) {
        for (const c10::IValue& element : item.toListRef()) {
          find_and_unprotect_tensors(element);
        }
      } else if (item.isGenericDict()) {
        for (const auto& [key, value] : item.toGenericDict()) {
          find_and_unprotect_tensors(key);
          find_and_unprotect_tensors(value);
        }
      }
    }
  }

  void unprotect_if_needed(const at::TensorBase& tensor) {
    if (!tensor.defined() || !tensor.has_storage()) {
      return;
    }

    void* ptr = tensor.data_ptr();
    orPointerAttributes attr;

    if (orPointerGetAttributes(&attr, ptr) != orSuccess ||
        attr.type != orMemoryTypeDevice) {
      return;
    }

    auto [it, inserted] = unprotected_pointers_.insert(attr.pointer);
    if (inserted) {
      orMemoryUnprotect(attr.pointer);
    }
  }

  std::unordered_set<void*> unprotected_pointers_;
};

} // namespace at::native::openreg

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `MemoryGuard`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/EmptyTensor.h`
- `ATen/TensorIterator.h`
- `ATen/TensorOperators.h`
- `ATen/core/blob.h`
- `ATen/native/CPUFallback.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/UnaryOps.h`
- `ATen/native/quantized/AffineQuantizer.h`
- `ATen/native/transformers/attention.h`
- `ATen/native/transformers/sdp_utils_cpp.h`
- `ATen/ops/_local_scalar_dense_native.h`
- `ATen/ops/_reshape_alias_native.h`
- `ATen/ops/abs_native.h`
- `ATen/ops/as_strided_cpu_dispatch.h`
- `ATen/ops/copy_native.h`
- `ATen/ops/quantize_per_tensor_native.h`
- `ATen/ops/resize_as_native.h`
- `ATen/ops/resize_native.h`
- `ATen/ops/set_cpu_dispatch.h`
- `ATen/ops/set_native.h`
- `ATen/ops/view_native.h`
- `torch/csrc/autograd/custom_function.h`
- `torch/csrc/autograd/function_hook.h`
- `c10/core/Allocator.h`
- `include/openreg.h`


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

This is a test file. Run it with:

```bash
python test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Common.h
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native`):

- [`Minimal.cpp_docs.md`](./Minimal.cpp_docs.md)
- [`Minimal.h_docs.md`](./Minimal.h_docs.md)
- [`Extra.h_docs.md`](./Extra.h_docs.md)
- [`Extra.cpp_docs.md`](./Extra.cpp_docs.md)


## Cross-References

- **File Documentation**: `Common.h_docs.md`
- **Keyword Index**: `Common.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
