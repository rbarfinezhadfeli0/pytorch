# Documentation: `c10/core/SafePyObject.h`

## File Metadata

- **Path**: `c10/core/SafePyObject.h`
- **Size**: 3,773 bytes (3.68 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <c10/macros/Export.h>
#include <c10/util/python_stub.h>
#include <utility>

namespace c10 {

// This is an safe owning holder for a PyObject, akin to pybind11's
// py::object, with two major differences:
//
//  - It is in c10/core; i.e., you can use this type in contexts where
//    you do not have a libpython dependency
//
//  - It is multi-interpreter safe (ala torchdeploy); when you fetch
//    the underlying PyObject* you are required to specify what the current
//    interpreter context is and we will check that you match it.
//
// It is INVALID to store a reference to a Tensor object in this way;
// you should just use TensorImpl directly in that case!
struct C10_API SafePyObject {
  // Steals a reference to data
  SafePyObject(PyObject* data, c10::impl::PyInterpreter* pyinterpreter)
      : data_(data), pyinterpreter_(pyinterpreter) {}
  SafePyObject(SafePyObject&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        pyinterpreter_(other.pyinterpreter_) {}
  // For now it's not used, so we just disallow it.
  SafePyObject& operator=(SafePyObject&&) = delete;

  SafePyObject(SafePyObject const& other)
      : data_(other.data_), pyinterpreter_(other.pyinterpreter_) {
    if (data_ != nullptr) {
      (*pyinterpreter_)->incref(data_);
    }
  }

  SafePyObject& operator=(SafePyObject const& other) {
    if (this == &other) {
      return *this; // Handle self-assignment
    }
    if (other.data_ != nullptr) {
      (*other.pyinterpreter_)->incref(other.data_);
    }
    if (data_ != nullptr) {
      (*pyinterpreter_)->decref(data_, /*has_pyobj_slot*/ false);
    }
    data_ = other.data_;
    pyinterpreter_ = other.pyinterpreter_;
    return *this;
  }

  ~SafePyObject() {
    if (data_ != nullptr) {
      (*pyinterpreter_)->decref(data_, /*has_pyobj_slot*/ false);
    }
  }

  c10::impl::PyInterpreter& pyinterpreter() const {
    return *pyinterpreter_;
  }
  PyObject* ptr(const c10::impl::PyInterpreter* /*interpreter*/) const;

  // stop tracking the current object, and return it
  PyObject* release() {
    auto rv = data_;
    data_ = nullptr;
    return rv;
  }

 private:
  PyObject* data_;
  c10::impl::PyInterpreter* pyinterpreter_;
};

// A newtype wrapper around SafePyObject for type safety when a python object
// represents a specific type. Note that `T` is only used as a tag and isn't
// actually used for any true purpose.
template <typename T>
struct SafePyObjectT : private SafePyObject {
  SafePyObjectT(PyObject* data, c10::impl::PyInterpreter* pyinterpreter)
      : SafePyObject(data, pyinterpreter) {}
  ~SafePyObjectT() = default;
  SafePyObjectT(SafePyObjectT&& other) noexcept : SafePyObject(other) {}
  SafePyObjectT(SafePyObjectT const&) = delete;
  SafePyObjectT& operator=(SafePyObjectT const&) = delete;
  SafePyObjectT& operator=(SafePyObjectT&&) = delete;

  using SafePyObject::ptr;
  using SafePyObject::pyinterpreter;
  using SafePyObject::release;
};

// Like SafePyObject, but non-owning.  Good for references to global PyObjects
// that will be leaked on interpreter exit.  You get a copy constructor/assign
// this way.
struct C10_API SafePyHandle {
  SafePyHandle() : data_(nullptr), pyinterpreter_(nullptr) {}
  SafePyHandle(PyObject* data, c10::impl::PyInterpreter* pyinterpreter)
      : data_(data), pyinterpreter_(pyinterpreter) {}

  c10::impl::PyInterpreter& pyinterpreter() const {
    return *pyinterpreter_;
  }
  PyObject* ptr(const c10::impl::PyInterpreter* /*interpreter*/) const;
  void reset() {
    data_ = nullptr;
    pyinterpreter_ = nullptr;
  }
  operator bool() {
    return data_;
  }

 private:
  PyObject* data_;
  c10::impl::PyInterpreter* pyinterpreter_;
};

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `C10_API`, `SafePyObjectT`, `C10_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/impl/PyInterpreter.h`
- `c10/macros/Export.h`
- `c10/util/python_stub.h`
- `utility`


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

Files in the same folder (`c10/core`):

- [`DispatchKey.cpp_docs.md`](./DispatchKey.cpp_docs.md)
- [`CopyBytes.h_docs.md`](./CopyBytes.h_docs.md)
- [`OptionalRef.h_docs.md`](./OptionalRef.h_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`MemoryFormat.h_docs.md`](./MemoryFormat.h_docs.md)
- [`SafePyObject.cpp_docs.md`](./SafePyObject.cpp_docs.md)
- [`DeviceType.cpp_docs.md`](./DeviceType.cpp_docs.md)
- [`SymBool.cpp_docs.md`](./SymBool.cpp_docs.md)
- [`Allocator.cpp_docs.md`](./Allocator.cpp_docs.md)
- [`SymbolicShapeMeta.cpp_docs.md`](./SymbolicShapeMeta.cpp_docs.md)


## Cross-References

- **File Documentation**: `SafePyObject.h_docs.md`
- **Keyword Index**: `SafePyObject.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
