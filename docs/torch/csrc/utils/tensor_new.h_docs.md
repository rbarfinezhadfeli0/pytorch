# Documentation: `torch/csrc/utils/tensor_new.h`

## File Metadata

- **Path**: `torch/csrc/utils/tensor_new.h`
- **Size**: 4,031 bytes (3.94 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <ATen/core/Tensor.h>

namespace torch::utils {

// NOTE: [torch.tensor, lift_fresh, and device movement]
//
// The `only_lift_cpu_tensors` flag controls what happens on torch.tensor([1, 2,
// 3], device="cuda") (or any non-CPU devices).
//
// If false (default):
// - the data gets moved into a CPU Tensor
// - then, it gets moved to cuda (via .to)
// - finally, we call lift_fresh() on it.
// Steps 1 and 2 happen with all modes disabled.
//
// If true:
// - the data gets moved into a CPU Tensor (with correct dtype)
// - we call lift_fresh() on it
// - finally, we move it to cuda (via .to)
// Step 1 happens with all modes disabled.
//
// `only_lift_cpu_tensors=true` is useful to prevent CUDA initialization under
// FakeTensorMode because it avoids moving concrete data to CUDA.
TORCH_API bool only_lift_cpu_tensors();
TORCH_API void set_only_lift_cpu_tensors(bool value);

at::Tensor base_tensor_ctor(PyObject* args, PyObject* kwargs);
TORCH_PYTHON_API at::Tensor legacy_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
at::Tensor legacy_tensor_new(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
at::Tensor indexing_tensor_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    std::optional<at::Device> device,
    PyObject* data);
at::Tensor sparse_coo_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
void _validate_sparse_coo_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);

at::Tensor sparse_compressed_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor sparse_csr_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor sparse_csc_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor sparse_bsr_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor sparse_bsc_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);

void _validate_sparse_compressed_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
void _validate_sparse_csr_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
void _validate_sparse_csc_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
void _validate_sparse_bsr_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
void _validate_sparse_bsc_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);

at::Tensor tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor as_tensor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);
at::Tensor new_tensor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
at::Tensor new_ones(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
at::Tensor tensor_frombuffer(
    PyObject* buffer,
    at::ScalarType dtype,
    int64_t count,
    int64_t offset,
    bool requires_grad);
at::Tensor tensor_fromDLPack(PyObject* data);
at::Tensor asarray(
    PyObject* obj,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Device> device,
    std::optional<bool> copy,
    bool requires_grad);
} // namespace torch::utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 31 function(s).

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

- `torch/csrc/python_headers.h`
- `torch/csrc/utils/python_arg_parser.h`
- `ATen/core/Tensor.h`


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

Files in the same folder (`torch/csrc/utils`):

- [`tensor_list.h_docs.md`](./tensor_list.h_docs.md)
- [`disable_torch_function.cpp_docs.md`](./disable_torch_function.cpp_docs.md)
- [`tensor_new.cpp_docs.md`](./tensor_new.cpp_docs.md)
- [`tensor_apply.cpp_docs.md`](./tensor_apply.cpp_docs.md)
- [`cpp_stacktraces.cpp_docs.md`](./cpp_stacktraces.cpp_docs.md)
- [`numpy_stub.h_docs.md`](./numpy_stub.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`six.h_docs.md`](./six.h_docs.md)
- [`python_scalars.h_docs.md`](./python_scalars.h_docs.md)


## Cross-References

- **File Documentation**: `tensor_new.h_docs.md`
- **Keyword Index**: `tensor_new.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
