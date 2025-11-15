# Documentation: `docs/torch/csrc/jit/python/python_list.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/python/python_list.cpp_docs.md`
- **Size**: 12,766 bytes (12.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/python/python_list.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/python/python_list.cpp`
- **Size**: 10,128 bytes (9.89 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/core/ivalue.h>
#include <c10/util/irange.h>
#include <pybind11/detail/common.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_list.h>
#include <torch/csrc/utils/pybind.h>
#include <stdexcept>

namespace torch::jit {

IValue ScriptListIterator::next() {
  if (iter_ == end_) {
    throw py::stop_iteration();
  }

  IValue result = *iter_;

  // Advance the iterator for next time.
  iter_++;

  return result;
}

bool ScriptListIterator::done() const {
  return iter_ == end_;
}

namespace {
py::list scriptListToPyList(const ScriptList& src) {
  py::list out(src.len());
  auto iter = src.iter();

  size_t i = 0;
  while (!iter.done()) {
    auto val = iter.next();
    // TODO: Handle nested dictionaries.
    if (val.isList()) {
      out[i] = scriptListToPyList(val);
    } else {
      out[i] = toPyObject(val);
    }
    ++i;
  }

  return out;
}
} // namespace

void initScriptListBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<ScriptListIterator>(m, "ScriptListIterator")
      .def(
          "__next__",
          [](ScriptListIterator& iter) {
            auto result = iter.next();
            return toPyObject(result);
          })
      .def("__iter__", [](ScriptListIterator& iter) { return iter; });

  py::class_<ScriptList, std::shared_ptr<ScriptList>>(m, "ScriptList")
      .def(py::init([](py::list list) {
        TypePtr type = nullptr;

        if (!list.empty()) {
          // If the source list is nonempty, try to infer its type.
          auto inferred_type = tryToInferType(list);

          if (!inferred_type.success()) {
            std::stringstream ss;
            ss << "Unable to infer type of list: " << inferred_type.reason();
            throw JITException(ss.str());
          }

          type = inferred_type.type();
        } else {
          // If is empty, assume the type is List[Tensor] as is done in
          // TorchScript code.
          type = ListType::create(TensorType::getInferred());
        }

        auto data = toIValue(std::move(list), type);
        return std::make_shared<ScriptList>(data);
      }))
      .def(
          "__repr__",
          [](const std::shared_ptr<ScriptList>& self) {
            return toPyObject(self->repr());
          })
      .def(
          "__bool__",
          [](const std::shared_ptr<ScriptList>& self) {
            return toPyObject(self->toBool());
          })
      .def(
          "__len__",
          [](const std::shared_ptr<ScriptList>& self) {
            return toPyObject(static_cast<int64_t>(self->len()));
          })
      .def(
          "__contains__",
          [](const std::shared_ptr<ScriptList>& self, py::object elem) {
            try {
              return toPyObject(self->contains(
                  toIValue(std::move(elem), self->type()->getElementType())));
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "__getitem__",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::diff_type idx) {
            try {
              auto value = self->getItem(idx);
              return toPyObject(value);
            } catch (const std::out_of_range& e) {
              throw py::index_error();
            }
          },
          py::return_value_policy::
              reference_internal) // Return value is a reference to an object
                                  // that resides in the ScriptList
      .def(
          "__getitem__",
          [](const std::shared_ptr<ScriptList>& self, const py::slice& slice) {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;

            if (!slice.compute(
                    self->len(), &start, &stop, &step, &slicelength)) {
              throw py::error_already_set();
            }

            auto seq = std::make_shared<ScriptList>(self->type());

            for ([[maybe_unused]] const auto i [[maybe_unused]] :
                 c10::irange(slicelength)) {
              seq->append(self->getItem(static_cast<ptrdiff_t>(start)));
              start += step;
            }

            return seq;
          })
      .def(
          "__setitem__",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::diff_type idx,
             py::object value) {
            try {
              self->setItem(
                  idx,
                  toIValue(std::move(value), self->type()->getElementType()));
            } catch (const std::out_of_range& e) {
              throw py::index_error();
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "__setitem__",
          [](const std::shared_ptr<ScriptList>& self,
             const py::slice& slice,
             const py::list& value) {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;

            if (!slice.compute(
                    self->len(), &start, &stop, &step, &slicelength)) {
              throw py::error_already_set();
            }

            if (slicelength != value.size()) {
              throw std::runtime_error(
                  "Left and right hand size of slice assignment have different sizes");
            }

            for (const auto i : c10::irange(slicelength)) {
              try {
                self->setItem(
                    static_cast<ptrdiff_t>(start),
                    toIValue(value[i], self->type()->getElementType()));
              } catch (const py::cast_error& e) {
                throw py::type_error();
              }
              start += step;
            }
          })
      .def(
          "__delitem__",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::diff_type idx) {
            try {
              self->delItem(idx);
            } catch (const std::out_of_range& e) {
              throw py::index_error();
            }
          })
      .def(
          "__iter__",
          [](const std::shared_ptr<ScriptList>& self) { return self->iter(); },
          py::keep_alive<0, 1>()) // ScriptList needs to be alive at least as
                                  // long as the iterator
      .def(
          "count",
          [](const std::shared_ptr<ScriptList>& self, py::object value) {
            try {
              return self->count(
                  toIValue(std::move(value), self->type()->getElementType()));

            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "remove",
          [](const std::shared_ptr<ScriptList>& self, py::object value) {
            try {
              return self->remove(
                  toIValue(std::move(value), self->type()->getElementType()));
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "append",
          [](const std::shared_ptr<ScriptList>& self, py::object value) {
            try {
              return self->append(
                  toIValue(std::move(value), self->type()->getElementType()));
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "clear",
          [](const std::shared_ptr<ScriptList>& self) { self->clear(); })
      .def(
          "extend",
          [](const std::shared_ptr<ScriptList>& self, py::list list) {
            try {
              self->extend(toIValue(std::move(list), self->type()));
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "extend",
          [](const std::shared_ptr<ScriptList>& self,
             const py::iterable& iter) {
            ScriptList iter_list(self->type());

            try {
              for (py::handle obj : iter) {
                iter_list.append(toIValue(
                    py::reinterpret_borrow<py::object>(obj),
                    self->type()->getElementType()));
              }
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }

            self->extend(toIValue(py::cast(iter_list), self->type()));
          })
      .def(
          "pop",
          [](const std::shared_ptr<ScriptList>& self) {
            return toPyObject(self->pop());
          })
      .def(
          "pop",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::diff_type idx) { return toPyObject(self->pop(idx)); })
      .def(
          "insert",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::diff_type idx,
             py::object obj) {
            try {
              self->insert(
                  toIValue(std::move(obj), self->type()->getElementType()),
                  idx);
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(py::pickle(
          [](const ScriptList& data) { // __getstate__
            return scriptListToPyList(data);
          },
          [](py::list list) { // __setstate__
            TypePtr type = nullptr;

            if (!list.empty()) {
              // If the source list is nonempty, try to infer its type.
              auto inferred_type = tryToInferType(list);

              if (!inferred_type.success()) {
                std::stringstream ss;
                ss << "Unable to infer type of list: "
                   << inferred_type.reason();
                throw JITException(ss.str());
              }

              type = inferred_type.type();
            } else {
              // If is empty, assume the type is List[Tensor] as is done in
              // TorchScript code.
              type = ListType::create(TensorType::getInferred());
            }

            auto data = toIValue(std::move(list), type);
            return std::make_shared<ScriptList>(data);
          }));
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `void`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/python`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue.h`
- `c10/util/irange.h`
- `pybind11/detail/common.h`
- `pybind11/pytypes.h`
- `torch/csrc/jit/python/pybind_utils.h`
- `torch/csrc/jit/python/python_list.h`
- `torch/csrc/utils/pybind.h`
- `stdexcept`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `python_list.cpp_docs.md`
- **Keyword Index**: `python_list.cpp_kw.md`
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

- **Serialization**: Uses pickle - be cautious with untrusted data

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

- **File Documentation**: `python_list.cpp_docs.md_docs.md`
- **Keyword Index**: `python_list.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
