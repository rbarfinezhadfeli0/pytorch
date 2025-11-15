# Documentation: `docs/torch/csrc/utils/tensor_apply.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/tensor_apply.cpp_docs.md`
- **Size**: 5,943 bytes (5.80 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/tensor_apply.cpp`

## File Metadata

- **Path**: `torch/csrc/utils/tensor_apply.cpp`
- **Size**: 3,494 bytes (3.41 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/utils/tensor_apply.h>

#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_scalars.h>

using namespace at;

namespace torch::utils {

struct StridedData {
  StridedData(const Tensor& tensor)
      : data(tensor.data_ptr()),
        strides(tensor.strides()),
        elementSize(tensor.element_size()) {}

  void* data;
  IntArrayRef strides;
  int64_t elementSize;

  void step(int dim) {
    data = (char*)data + (strides[dim] * elementSize);
  }
};

template <size_t N>
static void recursive_apply(
    IntArrayRef sizes,
    ScalarType scalarType,
    int64_t dim,
    PyObject* fn,
    std::array<StridedData, N> strided_data) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  if (dim == ndim) {
    auto args = THPObjectPtr(PyTuple_New(N));
    if (!args)
      throw python_error();
    for (const auto i : c10::irange(N)) {
      PyObject* arg = load_scalar(strided_data[i].data, scalarType);
      if (!arg)
        throw python_error();
      PyTuple_SET_ITEM(args.get(), i, arg);
    }
    auto ret = THPObjectPtr(PyObject_CallObject(fn, args.get()));
    if (!ret)
      throw python_error();
    store_scalar(strided_data[0].data, scalarType, ret.get());
    return;
  }

  auto n = sizes[dim];
  for ([[maybe_unused]] const auto i : c10::irange(n)) {
    recursive_apply(sizes, scalarType, dim + 1, fn, strided_data);
    for (auto& td : strided_data) {
      td.step(dim);
    }
  }
}

const Tensor& apply_(const Tensor& self, PyObject* fn) {
  if (self.is_meta()) {
    return self; // Just skip
  }
  TORCH_CHECK_TYPE(
      self.device().is_cpu(), "apply_ is only implemented on CPU tensors");
  auto scalarType = self.scalar_type();
  recursive_apply<1>(self.sizes(), scalarType, 0, fn, {{self}});
  return self;
}

const Tensor& map_(const Tensor& self, const Tensor& other_, PyObject* fn) {
  TORCH_CHECK_TYPE(
      other_.options().type_equal(self.options()),
      "map_: expected ",
      self.toString(),
      " for 'other' (got ",
      other_.toString(),
      ")");
  if (self.is_meta()) {
    return self; // Just skip
  }
  TORCH_CHECK_TYPE(
      self.device().is_cpu(), "map_ is only implemented on CPU tensors");
  c10::MaybeOwned<Tensor> other = expand_inplace(self, other_, "map_");
  auto scalarType = self.scalar_type();
  recursive_apply<2>(self.sizes(), scalarType, 0, fn, {{self, *other}});
  return self;
}

const Tensor& map2_(
    const Tensor& self,
    const Tensor& x_,
    const Tensor& y_,
    PyObject* fn) {
  TORCH_CHECK_TYPE(
      x_.options().type_equal(self.options()),
      "map2_: expected ",
      self.toString(),
      " for argument 'x' (got ",
      x_.toString(),
      ")");
  TORCH_CHECK_TYPE(
      y_.options().type_equal(self.options()),
      "map2_: expected ",
      self.toString(),
      " for argument 'y' (got ",
      y_.toString(),
      ")");
  if (self.is_meta()) {
    return self; // Just skip
  }
  TORCH_CHECK_TYPE(
      (self.device().is_cpu() && x_.device().is_cpu() && y_.device().is_cpu()),
      "map2_ is only implemented on CPU tensors");
  auto others = expand_inplace(self, x_, y_, "map2_");
  auto scalarType = self.scalar_type();
  recursive_apply<3>(
      self.sizes(),
      scalarType,
      0,
      fn,
      {{self, *std::get<0>(others), *std::get<1>(others)}});
  return self;
}

} // namespace torch::utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `at`

**Classes/Structs**: `StridedData`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/utils/tensor_apply.h`
- `ATen/ExpandUtils.h`
- `ATen/TensorUtils.h`
- `c10/util/irange.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/utils/python_numbers.h`
- `torch/csrc/utils/python_scalars.h`


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

Files in the same folder (`torch/csrc/utils`):

- [`tensor_list.h_docs.md`](./tensor_list.h_docs.md)
- [`disable_torch_function.cpp_docs.md`](./disable_torch_function.cpp_docs.md)
- [`tensor_new.cpp_docs.md`](./tensor_new.cpp_docs.md)
- [`cpp_stacktraces.cpp_docs.md`](./cpp_stacktraces.cpp_docs.md)
- [`numpy_stub.h_docs.md`](./numpy_stub.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`six.h_docs.md`](./six.h_docs.md)
- [`python_scalars.h_docs.md`](./python_scalars.h_docs.md)


## Cross-References

- **File Documentation**: `tensor_apply.cpp_docs.md`
- **Keyword Index**: `tensor_apply.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/utils`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/utils`):

- [`python_tuples.h_kw.md_docs.md`](./python_tuples.h_kw.md_docs.md)
- [`six.h_kw.md_docs.md`](./six.h_kw.md_docs.md)
- [`tensor_types.cpp_docs.md_docs.md`](./tensor_types.cpp_docs.md_docs.md)
- [`tensor_list.h_kw.md_docs.md`](./tensor_list.h_kw.md_docs.md)
- [`verbose.h_kw.md_docs.md`](./verbose.h_kw.md_docs.md)
- [`invalid_arguments.cpp_kw.md_docs.md`](./invalid_arguments.cpp_kw.md_docs.md)
- [`tensor_apply.h_kw.md_docs.md`](./tensor_apply.h_kw.md_docs.md)
- [`cuda_enabled.h_docs.md_docs.md`](./cuda_enabled.h_docs.md_docs.md)
- [`tensor_layouts.h_docs.md_docs.md`](./tensor_layouts.h_docs.md_docs.md)
- [`variadic.h_kw.md_docs.md`](./variadic.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `tensor_apply.cpp_docs.md_docs.md`
- **Keyword Index**: `tensor_apply.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
