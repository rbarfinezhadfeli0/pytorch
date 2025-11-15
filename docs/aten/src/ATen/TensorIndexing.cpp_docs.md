# Documentation: `aten/src/ATen/TensorIndexing.cpp`

## File Metadata

- **Path**: `aten/src/ATen/TensorIndexing.cpp`
- **Size**: 3,488 bytes (3.41 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/TensorIndexing.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

namespace at {
namespace indexing {

const EllipsisIndexType Ellipsis = EllipsisIndexType();

std::ostream& operator<<(std::ostream& stream, const Slice& slice) {
  stream << slice.start() << ":" << slice.stop() << ":" << slice.step();
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index) {
  if (tensor_index.is_none()) {
    stream << "None";
  } else if (tensor_index.is_ellipsis()) {
    stream << "...";
  } else if (tensor_index.is_integer()) {
    stream << tensor_index.integer();
  } else if (tensor_index.is_boolean()) {
    stream << std::boolalpha << tensor_index.boolean();
  } else if (tensor_index.is_slice()) {
    stream << tensor_index.slice();
  } else if (tensor_index.is_tensor()) {
    stream << tensor_index.tensor();
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices) {
  stream << "(";
  for (const auto i : c10::irange(tensor_indices.size())) {
    stream << tensor_indices[i];
    if (i < tensor_indices.size() - 1) stream << ", ";
  }
  stream << ")";
  return stream;
}

// This mirrors `THPVariable_setitem` in torch/csrc/autograd/python_variable_indexing.cpp
// for "the assigned value is a Scalar" case
static inline void set_item(const Tensor& self, ArrayRef<TensorIndex> indices, const Scalar& v) {
  Tensor value;

  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::Device self_device = self.device();

    // TODO: This qint special case looks very suspicious...
    if (isQIntType(self.scalar_type())) {
      value = at::indexing::scalarToTensor(v, device(kCPU).dtype(kFloat), at::Device(kCPU));
    } else if (self_device.is_cuda()) {
      value = at::indexing::scalarToTensor(v, self.options(), at::Device(kCPU));
    } else {
      value = at::indexing::scalarToTensor(v, self.options(), self_device);
    }
  }

  set_item(self, indices, value);
}

} // namespace indexing

Tensor Tensor::index(ArrayRef<at::indexing::TensorIndex> indices) const {
  TORCH_CHECK(!indices.empty(), "Passing an empty index list to Tensor::index() is not valid syntax");
  OptionalDeviceGuard device_guard(device_of(*this));
  return at::indexing::get_item(*this, indices);
}
Tensor Tensor::index(std::initializer_list<at::indexing::TensorIndex> indices) const {
  return index(ArrayRef<at::indexing::TensorIndex>(indices));
}

Tensor & Tensor::index_put_(ArrayRef<at::indexing::TensorIndex> indices, Tensor const & rhs) {
  TORCH_CHECK(!indices.empty(), "Passing an empty index list to Tensor::index_put_() is not valid syntax");
  OptionalDeviceGuard device_guard(device_of(*this));
  at::indexing::set_item(*this, indices, rhs);
  return *this;
}
Tensor & Tensor::index_put_(ArrayRef<at::indexing::TensorIndex> indices, const Scalar& v) {
  TORCH_CHECK(!indices.empty(), "Passing an empty index list to Tensor::index_put_() is not valid syntax");
  OptionalDeviceGuard device_guard(device_of(*this));
  at::indexing::set_item(*this, indices, v);
  return *this;
}
Tensor & Tensor::index_put_(std::initializer_list<at::indexing::TensorIndex> indices, Tensor const & rhs) {
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), rhs);
}
Tensor & Tensor::index_put_(std::initializer_list<at::indexing::TensorIndex> indices, const Scalar& v) {
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), v);
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `indexing`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/TensorIndexing.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`


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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `TensorIndexing.cpp_docs.md`
- **Keyword Index**: `TensorIndexing.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
