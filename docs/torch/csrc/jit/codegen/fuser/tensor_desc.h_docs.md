# Documentation: `torch/csrc/jit/codegen/fuser/tensor_desc.h`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/fuser/tensor_desc.h`
- **Size**: 2,701 bytes (2.64 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <torch/csrc/Export.h>

#include <algorithm>
#include <ostream>
#include <vector>

namespace torch::jit::fuser {

// type information needed by the compiler for input/outputs
// contiguity[i] is true if the dim i is contiguous with dim i + 1.
// contiguity.back() == true means strides.back() == 1.
struct TORCH_API TensorDesc {
  at::ScalarType scalar_type;
  std::vector<bool> contiguity;

  TensorDesc(const at::ScalarType& type, const std::vector<bool>& contiguity)
      : scalar_type{type}, contiguity{contiguity} {
    if (contiguity.empty()) {
      nDim_ = 0;
    } else {
      nDim_ = std::count(contiguity.begin(), contiguity.end(), false) +
          (lastIsContiguous() ? 1 : 0);
    }
  }

  // Delegating constructors
  TensorDesc(
      const at::ScalarType& type,
      const at::IntArrayRef& sizes,
      const at::IntArrayRef& strides)
      : TensorDesc(type, TensorDesc::findContiguous(sizes, strides)) {}

  TensorDesc(const at::Tensor& t)
      : TensorDesc(t.scalar_type(), t.sizes(), t.strides()) {}

  TensorDesc(const c10::TensorTypePtr& type)
      : TensorDesc(
            type->scalarType().value(),
            type->sizes().concrete_sizes().value(),
            type->strides().concrete_sizes().value()) {}

  // number of dimensions after contiguity compression
  size_t nDim() const {
    return nDim_;
  }

  // True iff innermost stride is 1
  bool lastIsContiguous() const {
    return (contiguity.empty() || contiguity.back());
  }

  static std::vector<bool> findContiguous(
      const at::IntArrayRef& sizes,
      const at::IntArrayRef& strides) {
    AT_ASSERT(sizes.size() == strides.size());
    std::vector<bool> cont(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      const auto expected_stride =
          (i + 1 < sizes.size()) ? sizes[i + 1] * strides[i + 1] : 1;
      cont[i] = (strides[i] == expected_stride);
    }
    return cont;
  }

  bool operator==(const TensorDesc& desc) const {
    return scalar_type == desc.scalar_type && contiguity == desc.contiguity;
  }

  bool operator!=(const TensorDesc& desc) const {
    return !(*this == desc);
  }

  static size_t hash(const TensorDesc& spec) {
    return c10::get_hash(
        spec.scalar_type,
        spec.nDim_,
        std::hash<std::vector<bool>>{}(spec.contiguity));
  }

 private:
  size_t nDim_;
};

inline std::ostream& operator<<(std::ostream& out, const TensorDesc& d) {
  out << d.scalar_type << "[";
  for (const auto b : d.contiguity)
    out << b << ";";
  out << "]";
  return out;
}

} // namespace torch::jit::fuser

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/fuser`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/core/jit_type.h`
- `c10/util/Exception.h`
- `c10/util/hash.h`
- `torch/csrc/Export.h`
- `algorithm`
- `ostream`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/csrc/jit/codegen/fuser`):

- [`compiler.h_docs.md`](./compiler.h_docs.md)
- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`kernel_spec.h_docs.md`](./kernel_spec.h_docs.md)
- [`executor.h_docs.md`](./executor.h_docs.md)
- [`fallback.h_docs.md`](./fallback.h_docs.md)
- [`arg_spec.h_docs.md`](./arg_spec.h_docs.md)
- [`fused_kernel.h_docs.md`](./fused_kernel.h_docs.md)
- [`tensor_info.h_docs.md`](./tensor_info.h_docs.md)
- [`executor.cpp_docs.md`](./executor.cpp_docs.md)


## Cross-References

- **File Documentation**: `tensor_desc.h_docs.md`
- **Keyword Index**: `tensor_desc.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
