# Documentation: `torch/csrc/python_dimname.cpp`

## File Metadata

- **Path**: `torch/csrc/python_dimname.cpp`
- **Size**: 3,605 bytes (3.52 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_dimname.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch {

struct InternedStringsTable {
  InternedStringsTable() = default;
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~InternedStringsTable();
  InternedStringsTable(const InternedStringsTable&) = delete;
  InternedStringsTable& operator=(InternedStringsTable const&) = delete;
  InternedStringsTable(InternedStringsTable&&) = delete;
  InternedStringsTable& operator=(InternedStringsTable&&) = delete;

  std::optional<at::Dimname> lookup(PyObject* obj);
  // Precondition: obj is an interned python string.
  void addMapping(PyObject* obj, at::Dimname dimname);

 private:
  ska::flat_hash_map<PyObject*, at::Dimname> py_interned_string_to_dimname_;
};

static InternedStringsTable kPyInternedStringToDimname;

// NOLINTNEXTLINE(bugprone-exception-escape)
InternedStringsTable::~InternedStringsTable() {
  // If python is already dead, leak the wrapped python objects
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    for (auto it = py_interned_string_to_dimname_.begin();
         it != py_interned_string_to_dimname_.end();
         ++it) {
      // See Note [References to python interned strings]
      Py_DECREF(it->first);
    }
  }
}

std::optional<at::Dimname> InternedStringsTable::lookup(PyObject* obj) {
  auto it = py_interned_string_to_dimname_.find(obj);
  if (it == py_interned_string_to_dimname_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void InternedStringsTable::addMapping(PyObject* obj, at::Dimname dimname) {
  // Note [References to python interned strings]
  // If a Python interned string has no references to it, then it gets
  // deallocated, invalidating this mapping. Let's immortalize the string by
  // holding a refcount to it and releasing it in the destructor
  Py_INCREF(obj);
  py_interned_string_to_dimname_.emplace(obj, dimname);
}

} // namespace torch

bool THPUtils_checkDimname(PyObject* obj) {
  return obj == Py_None || THPUtils_checkString(obj);
}

// To avoid ambiguity with IntArrayRef, we parse obj as a DimnameList if
// it is a list or tuple and its first elt is a Dimname
bool THPUtils_checkDimnameList(PyObject* obj) {
  auto tuple = PyTuple_Check(obj);
  if (!tuple && !PyList_Check(obj)) {
    return false;
  }
  // NOLINTNEXTLINE(bugprone-branch-clone)
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  if (size == 0) {
    return true;
  }
  PyObject* first_elt =
      tuple ? PyTuple_GET_ITEM(obj, 0) : PyList_GET_ITEM(obj, 0);
  return THPUtils_checkDimname(first_elt);
}

at::Dimname THPDimname_parse(PyObject* obj) {
  if (obj == Py_None) {
    return at::Dimname::wildcard();
  }

  TORCH_CHECK_TYPE(
      THPUtils_checkString(obj),
      "expected None or string for Dimname but got ",
      Py_TYPE(obj)->tp_name);

  if (!THPUtils_isInterned(obj)) {
    // internStringInPlace decrefs obj and increfs the result. Because we're
    // not actually returning the result to the user, we need to undo these.
    // See
    // https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_InternInPlace
    Py_INCREF(obj);
    THPUtils_internStringInPlace(&obj);
    Py_DECREF(obj);
  }

  auto maybeDimname = torch::kPyInternedStringToDimname.lookup(obj);
  if (maybeDimname) {
    return *maybeDimname;
  }

  const auto name = THPUtils_unpackString(obj);
  auto dimname = at::Dimname::fromSymbol(at::Symbol::dimname(name));
  torch::kPyInternedStringToDimname.addMapping(obj, dimname);
  return dimname;
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `InternedStringsTable`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/flat_hash_map.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/python_dimname.h`
- `torch/csrc/utils/python_strings.h`


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

Files in the same folder (`torch/csrc`):

- [`itt_wrapper.cpp_docs.md`](./itt_wrapper.cpp_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`Export.h_docs.md`](./Export.h_docs.md)
- [`MemoryFormat.h_docs.md`](./MemoryFormat.h_docs.md)
- [`Size.h_docs.md`](./Size.h_docs.md)
- [`stub.c_docs.md`](./stub.c_docs.md)
- [`Device.h_docs.md`](./Device.h_docs.md)
- [`Layout.h_docs.md`](./Layout.h_docs.md)
- [`Exceptions.h_docs.md`](./Exceptions.h_docs.md)
- [`PyInterpreter.h_docs.md`](./PyInterpreter.h_docs.md)


## Cross-References

- **File Documentation**: `python_dimname.cpp_docs.md`
- **Keyword Index**: `python_dimname.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
