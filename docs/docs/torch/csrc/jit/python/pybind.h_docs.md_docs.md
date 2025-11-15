# Documentation: `docs/torch/csrc/jit/python/pybind.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/python/pybind.h_docs.md`
- **Size**: 10,864 bytes (10.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/python/pybind.h`

## File Metadata

- **Path**: `torch/csrc/jit/python/pybind.h`
- **Size**: 7,957 bytes (7.77 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/core/ivalue.h>
#include <ATen/core/symbol.h>
#include <c10/util/irange.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace torch::jit {

// This is a variant of shared_ptr that "sees through" a wrapper.
// We use it to convert Value, Node, Block and node to "wrapped" Python
// values. When we destruct the C++ object, the wrapper's pointer will
// be set to 0 and any future dereferencing will throw. We need this
// because the Python objects may hang around after the C++ object
// has already been destroyed.
// This also needs the magic type_caster below, which is from the
// workaround offered in https://github.com/pybind/pybind11/issues/2751
template <typename T>
class unwrapping_shared_ptr {
  static_assert(
      std::is_same_v<T, torch::jit::Value> ||
          std::is_same_v<T, torch::jit::Node> ||
          std::is_same_v<T, torch::jit::Block>,
      "unwrapping type only defined for Graph object types");

 private:
  std::shared_ptr<torch::jit::Wrap<T>> impl;

 public:
  unwrapping_shared_ptr() : impl({}) {}
  explicit unwrapping_shared_ptr(T* p) : impl(p->wrap()) {
    impl->clear_cb = &clear_registered_instances;
  }
  T* get() const {
    if (!impl->elem) {
      throw std::logic_error("has been invalidated");
    }
    return impl->elem;
  }
  // we need to disable the overloaded & for PyBind11 < 2.3 due.
  // see https://github.com/pybind/pybind11/pull/1435
#if (PYBIND11_VERSION_MAJOR > 2) || \
    ((PYBIND11_VERSION_MAJOR == 2) && (PYBIND11_VERSION_MINOR >= 3))
  T** operator&() {
    if (!impl->elem) {
      throw std::logic_error("has been invalidated");
    }
    return &(impl->elem);
  }
#endif
};

} // namespace torch::jit

PYBIND11_DECLARE_HOLDER_TYPE(T, torch::jit::unwrapping_shared_ptr<T>, true)

namespace pybind11::detail {

#define CREATE_UNWRAPPING_CASTER(Class)                                                   \
  template <>                                                                             \
  struct type_caster<Class> : public type_caster_base<Class> {                            \
   public:                                                                                \
    using type = Class;                                                                   \
    using holder_type = torch::jit::unwrapping_shared_ptr<Class>;                         \
                                                                                          \
    bool load(handle src, bool convert) {                                                 \
      return load_impl<type_caster<Class>>(src, convert);                                 \
    }                                                                                     \
                                                                                          \
    explicit operator type*() {                                                           \
      return static_cast<type*>(value);                                                   \
    }                                                                                     \
    explicit operator type&() {                                                           \
      return *static_cast<type*>(value);                                                  \
    }                                                                                     \
                                                                                          \
   protected:                                                                             \
    friend class type_caster_generic;                                                     \
                                                                                          \
    bool load_value(const value_and_holder& v_h) {                                        \
      if (v_h.holder_constructed()) {                                                     \
        value = v_h.template holder<holder_type>().get();                                 \
        return true;                                                                      \
      } else {                                                                            \
        throw cast_error(                                                                 \
            "Unable to cast from non-held to held instance (#Class& to Holder<#Class>)"); \
      }                                                                                   \
    }                                                                                     \
  }

CREATE_UNWRAPPING_CASTER(torch::jit::Node);
CREATE_UNWRAPPING_CASTER(torch::jit::Value);
CREATE_UNWRAPPING_CASTER(torch::jit::Block);

#undef CREATE_UNWRAPPING_CASTER

template <>
struct type_caster<torch::jit::IValue> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::IValue, _("IValue"));

  bool load(handle src, bool /*unused*/) {
    try {
      value = torch::jit::toTypeInferredIValue(src);
      return true;
    } catch (std::exception& e) {
      return false;
    }
  }

  static handle cast(
      torch::jit::IValue src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return torch::jit::toPyObject(std::move(src)).release();
  }
};

template <>
struct type_caster<torch::jit::Symbol> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::Symbol, _("Symbol"));

  bool load(handle src, bool /*unused*/) {
    // TODO: Is there a way to py::cast that doesn't raise an exception on
    // failure?  Can we catch pybind11::cast_error here instead?
    std::string src_str;
    try {
      src_str = py::cast<std::string>(src);
    } catch (std::exception& e) {
      return false;
    }
    value = torch::jit::Symbol::fromQualString(src_str);
    return true;
  }

  static handle cast(
      torch::jit::Symbol src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(std::string(src.toQualString()), return_value_policy::copy)
        .release();
  }
};

template <>
struct type_caster<torch::jit::AttributeKind> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::AttributeKind, _("AttributeKind"));

  bool load(handle src, bool /*unused*/) {
    return false;
  }

  static handle cast(
      torch::jit::AttributeKind src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(
               std::string(torch::jit::toString(src)),
               return_value_policy::copy)
        .release();
  }
};

// See https://github.com/pybind/pybind11/issues/637
using ListCasterBase = pybind11::detail::
    list_caster<std::vector<torch::jit::Node*>, torch::jit::Node*>;
template <>
struct type_caster<std::vector<torch::jit::Node*>> : ListCasterBase {
  static handle cast(
      const std::vector<torch::jit::Node*>& src,
      return_value_policy /*unused*/,
      handle parent) {
    return ListCasterBase::cast(src, return_value_policy::reference, parent);
  }
  static handle cast(
      const std::vector<torch::jit::Node*>* src,
      return_value_policy pol,
      handle parent) {
    return cast(*src, pol, parent);
  }
};

} // namespace pybind11::detail

namespace torch::jit {

static inline py::tuple tuple_tail(const py::tuple& tup) {
  py::tuple r(tup.size() - 1);
  for (const auto i : c10::irange(1, tup.size())) {
    r[i - 1] = tup[i];
  }
  return r;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `pybind11`, `torch`, `py`

**Classes/Structs**: `the`, `unwrapping_shared_ptr`, `type_caster`, `type_caster_generic`, `type_caster`, `type_caster`, `type_caster`, `type_caster`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/python`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/python_headers.h`
- `ATen/core/ivalue.h`
- `ATen/core/symbol.h`
- `c10/util/irange.h`
- `torch/csrc/DynamicTypes.h`
- `torch/csrc/THP.h`
- `torch/csrc/autograd/variable.h`
- `torch/csrc/jit/frontend/tracer.h`
- `torch/csrc/jit/python/pybind_utils.h`
- `torch/csrc/utils/pybind.h`
- `pybind11/functional.h`
- `pybind11/pybind11.h`
- `pybind11/stl.h`


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

Files in the same folder (`torch/csrc/jit/python`):

- [`python_ir.cpp_docs.md`](./python_ir.cpp_docs.md)
- [`python_custom_class.cpp_docs.md`](./python_custom_class.cpp_docs.md)
- [`python_ivalue.h_docs.md`](./python_ivalue.h_docs.md)
- [`pybind_utils.cpp_docs.md`](./pybind_utils.cpp_docs.md)
- [`opaque_obj.h_docs.md`](./opaque_obj.h_docs.md)
- [`update_graph_executor_opt.h_docs.md`](./update_graph_executor_opt.h_docs.md)
- [`python_tree_views.h_docs.md`](./python_tree_views.h_docs.md)
- [`python_ir.h_docs.md`](./python_ir.h_docs.md)
- [`python_tracer.h_docs.md`](./python_tracer.h_docs.md)
- [`python_arg_flatten.h_docs.md`](./python_arg_flatten.h_docs.md)


## Cross-References

- **File Documentation**: `pybind.h_docs.md`
- **Keyword Index**: `pybind.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/python`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/python`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/csrc/jit/python`):

- [`opaque_obj.h_kw.md_docs.md`](./opaque_obj.h_kw.md_docs.md)
- [`script_init.h_docs.md_docs.md`](./script_init.h_docs.md_docs.md)
- [`python_tree_views.cpp_docs.md_docs.md`](./python_tree_views.cpp_docs.md_docs.md)
- [`python_dict.cpp_docs.md_docs.md`](./python_dict.cpp_docs.md_docs.md)
- [`python_tree_views.h_docs.md_docs.md`](./python_tree_views.h_docs.md_docs.md)
- [`opaque_obj.h_docs.md_docs.md`](./opaque_obj.h_docs.md_docs.md)
- [`python_custom_class.cpp_docs.md_docs.md`](./python_custom_class.cpp_docs.md_docs.md)
- [`python_tracer.cpp_kw.md_docs.md`](./python_tracer.cpp_kw.md_docs.md)
- [`python_interpreter.cpp_kw.md_docs.md`](./python_interpreter.cpp_kw.md_docs.md)
- [`python_tracer.cpp_docs.md_docs.md`](./python_tracer.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `pybind.h_docs.md_docs.md`
- **Keyword Index**: `pybind.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
