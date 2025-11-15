# Documentation: `torch/csrc/dynamo/guards.h`

## File Metadata

- **Path**: `torch/csrc/dynamo/guards.h`
- **Size**: 3,890 bytes (3.80 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <c10/core/GradMode.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::dynamo {

PyObject* torch_c_dynamo_guards_init();

// interfaces for extra_state and eval_frame.c because RootGuardManager class is
// not visible there.
void* convert_to_root_guard_manager(py::object root);
bool run_root_guard_manager(void* root, FrameLocalsMapping* f_locals);

extern thread_local bool tls_is_in_mode_without_ignore_compile_internals;

void set_is_in_mode_without_ignore_compile_internals(bool value);

// If we're in a mode with ignore_compile_internals=False, we WON'T mask
// Python keys from guard checking (they should be visible, so eager fallback is
// possible). Otherwise (invisible mode or no mode), we WILL mask Python keys to
// avoid guard failures on the dispatch keyset at runtime.
bool get_is_in_mode_without_ignore_compile_internals();

struct LocalState {
  // TLS state that changes operators
  c10::impl::LocalDispatchKeySet dispatch_modifier;
  c10::DispatchKeySet override_dispatch_key_set;
  bool grad_mode_enabled;
  bool should_mask_python_keys;

  at::DispatchKeySet apply(at::DispatchKeySet ks) const {
    if (override_dispatch_key_set.empty()) {
      auto result =
          (ks | dispatch_modifier.included_) - dispatch_modifier.excluded_;

      if (should_mask_python_keys) {
        result = result -
            c10::DispatchKeySet(
                     {c10::DispatchKey::Python,
                      c10::DispatchKey::PythonTLSSnapshot});
      }

      return result;
    } else {
      return override_dispatch_key_set;
    }
  }

  LocalState()
      : dispatch_modifier(c10::impl::tls_local_dispatch_key_set()),
        override_dispatch_key_set(c10::BackendComponent::InvalidBit),
        grad_mode_enabled(at::GradMode::is_enabled()),
        should_mask_python_keys(
            !get_is_in_mode_without_ignore_compile_internals()) {}

  void overrideDispatchKeySet(c10::DispatchKeySet ks) {
    override_dispatch_key_set = ks;
  }
};

class TensorCheck {
 public:
  TensorCheck(
      const LocalState& state,
      PyTypeObject* pt,
      const at::Tensor& v,
      c10::DispatchKeySet dispatch_key_set,
      std::vector<std::optional<c10::SymInt>> dynamic_dims_sizes,
      std::vector<std::optional<c10::SymInt>> dynamic_dims_strides);

  TensorCheck(
      const LocalState& state,
      PyTypeObject* pt,
      c10::DispatchKeySet dispatch_key_set,
      at::ScalarType dtype,
      at::DeviceIndex device_index,
      bool requires_grad,
      std::vector<std::optional<c10::SymInt>> dynamic_dims_sizes,
      std::vector<std::optional<c10::SymInt>> dynamic_dims_strides);

  bool check(const LocalState& state, const at::Tensor& v);
  bool check(
      const LocalState& state,
      const c10::DispatchKeySet& dispatch_key_set,
      const at::ScalarType& dtype,
      const c10::Device& device,
      const c10::SymIntArrayRef& dynamic_dims_sizes,
      const c10::SymIntArrayRef& dynamic_dims_strides,
      const bool& requires_grad);
  std::string check_verbose(
      const LocalState& state,
      const at::Tensor& v,
      const std::string& tensor_name);

  PyTypeObject* pytype;

 private:
  uint64_t dispatch_key_; // DispatchKeySet includes device/layout
  at::ScalarType dtype_;
  // Note(voz): While dispatch_key_ is sufficiently representative of a device
  // In that keys are more granular AND device specific - they do not
  // necessarily capture device indices correctly.
  at::DeviceIndex device_index_;
  bool requires_grad_;
  // NB: These are unset if dynamic shapes is enabled.
  std::vector<std::optional<c10::SymInt>> sizes_;
  std::vector<std::optional<c10::SymInt>> strides_;
  // Not strictly required for dense tensors, but nested tensors need it.
  int64_t dim_;
};

} // namespace torch::dynamo

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `is`, `LocalState`, `TensorCheck`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/GradMode.h`
- `torch/csrc/dynamo/framelocals_mapping.h`
- `torch/csrc/python_headers.h`
- `torch/csrc/utils/pybind.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/csrc/dynamo`):

- [`cpython_defs.c_docs.md`](./cpython_defs.c_docs.md)
- [`eval_frame_cpp.h_docs.md`](./eval_frame_cpp.h_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`python_compiled_autograd.cpp_docs.md`](./python_compiled_autograd.cpp_docs.md)
- [`cpython_defs.h_docs.md`](./cpython_defs.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`framelocals_mapping.h_docs.md`](./framelocals_mapping.h_docs.md)
- [`compiled_autograd.cpp_docs.md`](./compiled_autograd.cpp_docs.md)
- [`extra_state.h_docs.md`](./extra_state.h_docs.md)
- [`eval_frame.c_docs.md`](./eval_frame.c_docs.md)


## Cross-References

- **File Documentation**: `guards.h_docs.md`
- **Keyword Index**: `guards.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
