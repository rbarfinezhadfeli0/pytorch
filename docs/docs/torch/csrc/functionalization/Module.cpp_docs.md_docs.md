# Documentation: `docs/torch/csrc/functionalization/Module.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/functionalization/Module.cpp_docs.md`
- **Size**: 4,741 bytes (4.63 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/functionalization/Module.cpp`

## File Metadata

- **Path**: `torch/csrc/functionalization/Module.cpp`
- **Size**: 2,770 bytes (2.71 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/functionalization/Module.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/FunctionalStorageImpl.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/FunctionalizeFallbackKernel.h>
#include <memory>

namespace torch::functionalization {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Create a `torch._C._functionalization` Python module.
  auto functionalization = m.def_submodule(
      "_functionalization", "functionalization related pybind.");

  // Retrieve the ViewMeta sequence of a given functional tensor.
  functionalization.def("get_view_meta_sequence", [](const at::Tensor& tensor) {
    TORCH_INTERNAL_ASSERT(
        at::functionalization::impl::isFunctionalTensor(tensor));
    auto impl = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
    return impl->view_metas();
  });

  // Applies the given ViewMeta sequence to the given base.
  functionalization.def(
      "apply_view_meta_sequence",
      [](const at::Tensor& base,
         const std::vector<std::shared_ptr<at::functionalization::ViewMeta>>&
             sequence) {
        return at::functionalization::impl::apply_view_meta_sequence(
            base, sequence);
      });

  // Binding for InverseReturnMode.
  py::enum_<at::functionalization::InverseReturnMode>(
      functionalization, "InverseReturnMode")
      .value("AlwaysView", at::functionalization::InverseReturnMode::AlwaysView)
      .value("NeverView", at::functionalization::InverseReturnMode::NeverView)
      .value(
          "ViewOrScatterInverse",
          at::functionalization::InverseReturnMode::ViewOrScatterInverse);

  // Create bindings for the ViewMeta base class.
  //
  // Needed so that we can take a list of ViewMeta objects as parameter.
  // Specifically, in the Python-side, we will have a list of derived ViewMeta
  // classes. We need to tell pybind11 that all of those are, in fact, instances
  // of different ViewMeta sub-types.
  py::class_<
      at::functionalization::ViewMeta,
      std::shared_ptr<at::functionalization::ViewMeta>>(
      functionalization, "ViewMeta")
      .def_property_readonly(
          "has_symbolic_inputs",
          [](const std::shared_ptr<at::functionalization::ViewMeta>& meta) {
            return meta->has_symbolic_inputs;
          });

  // Bindings for `ViewMeta` specializations manually implemented.
  create_binding_with_pickle<at::functionalization::resize__ViewMeta>(
      functionalization);
  create_binding_with_pickle<at::functionalization::_unsafe_view_ViewMeta>(
      functionalization);

  // Bindings for `ViewMeta` specializations automatically generated.
  initGenerated(functionalization.ptr());
}

} // namespace torch::functionalization

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/functionalization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/functionalization/Module.h`
- `torch/csrc/utils/pybind.h`
- `ATen/FunctionalStorageImpl.h`
- `ATen/FunctionalTensorWrapper.h`
- `ATen/FunctionalizeFallbackKernel.h`
- `memory`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/functionalization`):

- [`Module.h_docs.md`](./Module.h_docs.md)


## Cross-References

- **File Documentation**: `Module.cpp_docs.md`
- **Keyword Index**: `Module.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/functionalization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/functionalization`, which is part of the **core PyTorch library**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/functionalization`):

- [`Module.h_docs.md_docs.md`](./Module.h_docs.md_docs.md)
- [`Module.h_kw.md_docs.md`](./Module.h_kw.md_docs.md)
- [`Module.cpp_kw.md_docs.md`](./Module.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Module.cpp_docs.md_docs.md`
- **Keyword Index**: `Module.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
