# Documentation: `docs/torch/csrc/utils/nested.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/nested.cpp_docs.md`
- **Size**: 5,423 bytes (5.30 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/nested.cpp`

## File Metadata

- **Path**: `torch/csrc/utils/nested.cpp`
- **Size**: 2,997 bytes (2.93 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/nested.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/torch.h>
#include <stdexcept>
#include <vector>

namespace torch::utils {

// NB: device_idx here is NOT a DeviceIndex, but index into PythonArgs
static c10::TensorOptions typeIdWithDefault(
    PythonArgs& r,
    int device_idx,
    c10::DispatchKey dispatch_key) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
  if (!r.isNone(device_idx)) {
    options = options.device(r.device(device_idx));
  }
  return options;
}

at::Tensor nested_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    torch::PythonArgs& r) {
  TORCH_CHECK(r.idx == 0, "nested_tensor(): invalid arguments");

  PyObject* data = r.pyobject(0);
  // Check if data is a list: Only List[Tensor] and List[List...[Scalar]] are
  // accepted for now
  TORCH_CHECK_TYPE(
      PyList_Check(data),
      "Only lists (List[Tensor] and List[List...[Scalar]]) are accepted in nested_tensor");

  auto dtype_val = r.scalartypeWithDefault(1, scalar_type);
  auto tensor_options = typeIdWithDefault(r, 2, dispatch_key);
  bool pin_memory = r.toBool(3);
  bool args_requires_grad = r.toBool(4);

  TORCH_CHECK(
      PyList_Size(data) >= 0,
      "Something went really wrong and your list has negative size");

  // Check whether we are dealing with lists of tensors or not
  std::vector<at::Tensor> new_list(PyList_Size(data));
  for (const auto i : c10::irange(PyList_Size(data))) {
    THPObjectPtr elem = THPObjectPtr(PyList_GetItemRef(data, i));
    if (THPVariable_Check(elem.get())) {
      new_list[i] = THPVariable_Unpack(elem.get()).detach();
      TORCH_CHECK(
          !new_list[i].is_nested(),
          "We do not accept nested tensors as input to nested tensors");
      TORCH_CHECK(
          new_list[i].layout() == kStrided,
          "We do not accept non-strided layouts as input to nested tensors");
    } else {
      PythonArgs elem_r(r);
      std::array<PyObject*, 6> elem_args = {
          elem.get(), // data
          r.args[1], // dtpye
          nullptr, // device (cpu)
          nullptr, // no pinned memory
          r.args[4], // requires grad
          nullptr // names
      };
      elem_r.args = elem_args.data();
      new_list[i] = tensor_ctor(dispatch_key, scalar_type, elem_r);
    }
  }

  at::ScalarType final_dtype = dtype_val;
  if (r.isNone(1) && !new_list.empty()) {
    final_dtype = c10::typeMetaToScalarType(new_list[0].dtype());
  }
  at::Device final_device = tensor_options.device();
  if (r.isNone(2) && !new_list.empty()) {
    final_device = new_list[0].device();
  }
  auto out = at::_nested_tensor_from_tensor_list(
      new_list, final_dtype, std::nullopt, final_device, pin_memory);
  out.requires_grad_(args_requires_grad);
  return out;
}

} // namespace torch::utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/NestedTensorImpl.h`
- `c10/core/ScalarType.h`
- `torch/csrc/python_headers.h`
- `torch/csrc/utils/nested.h`
- `torch/csrc/utils/pybind.h`
- `torch/csrc/utils/tensor_new.h`
- `torch/torch.h`
- `stdexcept`
- `vector`


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
- [`tensor_apply.cpp_docs.md`](./tensor_apply.cpp_docs.md)
- [`cpp_stacktraces.cpp_docs.md`](./cpp_stacktraces.cpp_docs.md)
- [`numpy_stub.h_docs.md`](./numpy_stub.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`six.h_docs.md`](./six.h_docs.md)
- [`python_scalars.h_docs.md`](./python_scalars.h_docs.md)


## Cross-References

- **File Documentation**: `nested.cpp_docs.md`
- **Keyword Index**: `nested.cpp_kw.md`
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

- **File Documentation**: `nested.cpp_docs.md_docs.md`
- **Keyword Index**: `nested.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
