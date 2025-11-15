# Documentation: `docs/torch/csrc/dynamo/framelocals_mapping.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/dynamo/framelocals_mapping.cpp_docs.md`
- **Size**: 9,213 bytes (9.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/dynamo/framelocals_mapping.cpp`

## File Metadata

- **Path**: `torch/csrc/dynamo/framelocals_mapping.cpp`
- **Size**: 6,729 bytes (6.57 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/dynamo/framelocals_mapping.h>

#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/debug_macros.h>

#define Py_BUILD_CORE
#include <internal/pycore_code.h>
#undef Py_BUILD_CORE

#if IS_PYTHON_3_11_PLUS

// Our own version of PyFrame_GetLocals.
// Also combines functionality from frame_init_get_vars and frame_get_var.
// PyFrame_GetLocals:
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1213
// frame_init_get_vars:
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1136
// frame_get_var:
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1162
// PyFrame_GetLocals returns the frame locals dict.
// frame_init_get_vars initializes free variables from the closure.
// frame_get_var fetches the variable value from the frame given the index
// NOTE: hidden variables are not included.
// Returns a new reference.
FrameLocalsMapping::FrameLocalsMapping(FrameLocalsFrameType* frame)
    : _code_obj(py::cast<py::object>((PyObject*)F_CODE(frame))) {
  PyCodeObject* co = F_CODE(frame);
  _framelocals.resize(co->co_nlocalsplus, nullptr);

#if IS_PYTHON_3_15_PLUS
  TORCH_CHECK(false, "Python 3.15+");
#elif IS_PYTHON_3_14_PLUS
  if (!frame->stackpointer) {
    return;
  }
#else
  if (!frame->stacktop) {
    return;
  }
#endif

  auto update_framelocals = [&](int i, PyObject* value) {
    _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);

    if (kind & CO_FAST_FREE && !(co->co_flags & CO_OPTIMIZED)) {
      return;
    }

#if IS_PYTHON_3_12_PLUS
    if (kind & CO_FAST_HIDDEN) {
      return;
    }
#endif

    if (kind & CO_FAST_FREE) {
      CHECK(value != nullptr && PyCell_Check(value));
      value = PyCell_GET(value);
    }

    DEBUG_CHECK(0 <= i && i < _framelocals.size());
    _framelocals[i] = value;
  };

  auto offset = co->co_nlocalsplus - co->co_nfreevars;
#if IS_PYTHON_3_15_PLUS
  TORCH_CHECK(false, "Python 3.15+");
#elif IS_PYTHON_3_14_PLUS
  for (int i = 0; i < offset; i++) {
    update_framelocals(
        i, THP_PyStackRef_AsPyObjectBorrow(&frame->localsplus[i]));
  }
#else
  for (int i = 0; i < offset; i++) {
    update_framelocals(i, frame->localsplus[i]);
  }
#endif

  // Get references to closure variables
#if IS_PYTHON_3_15_PLUS
  PyObject* closure;
  TORCH_CHECK(false, "Python 3.15+");
#else
  PyObject* closure = FUNC(frame)->func_closure;
#endif
  for (int i = 0; i < co->co_nfreevars; i++) {
    update_framelocals(offset + i, PyTuple_GET_ITEM(closure, i));
  }

  // NOTE no need to move the instruction pointer to after COPY_FREE_VARS
  // since we don't actually copy free vars from the closure to the frame
  // localsplus.
}

void FrameLocalsMapping::_realize_dict() {
  _dict = py::dict();
  py::tuple framelocals_names = code_framelocals_names(_code_obj);

  auto nlocalsplus = ((PyCodeObject*)_code_obj.ptr())->co_nlocalsplus;
  DEBUG_CHECK(nlocalsplus == _framelocals.size());
  for (int i = 0; i < nlocalsplus; i++) {
    if (_framelocals[i]) {
      _dict[framelocals_names[i]] = _framelocals[i];
    }
  }
}

py::tuple code_framelocals_names(py::handle code) {
  CHECK(PyCode_Check(code.ptr()));
  return py::cast<py::tuple>(((PyCodeObject*)code.ptr())->co_localsplusnames);
}

#else

// Based on
// https://github.com/python/cpython/blob/5f24da9d75bb0150781b17ee4706e93e6bb364ea/Objects/frameobject.c#L1016
FrameLocalsMapping::FrameLocalsMapping(FrameLocalsFrameType* frame)
    : _code_obj(py::cast<py::object>((PyObject*)F_CODE(frame))) {
  PyCodeObject* co = (PyCodeObject*)_code_obj.ptr();
  auto nlocals =
      std::min<int>(co->co_nlocals, (int)PyTuple_GET_SIZE(co->co_varnames));
  auto ncells = PyCode_GetNCellvars(co);
  auto nfree = PyCode_GetNFreevars(co);

  _framelocals.resize(co->co_nlocals + ncells + nfree, nullptr);

  auto update_framelocals = [&](int i, bool deref) {
    DEBUG_CHECK(0 <= i && i < _framelocals.size());
    PyObject* value = frame->f_localsplus[i];
    if (deref) {
      CHECK(value != nullptr && PyCell_Check(value));
      value = PyCell_GET(value);
    }
    _framelocals[i] = value;
  };

  // locals
  for (int i = 0; i < nlocals; i++) {
    update_framelocals(i, false);
  }

  // cellvars
  for (int i = 0; i < ncells; i++) {
    update_framelocals(co->co_nlocals + i, true);
  }

  // freevars
  if (co->co_flags & CO_OPTIMIZED) {
    for (int i = 0; i < nfree; i++) {
      update_framelocals(co->co_nlocals + ncells + i, true);
    }
  }
}

void FrameLocalsMapping::_realize_dict() {
  _dict = py::dict();
  py::tuple framelocals_names = code_framelocals_names(_code_obj);
  PyCodeObject* co = (PyCodeObject*)_code_obj.ptr();

  auto update_mapping = [&](int i) {
    DEBUG_CHECK(0 <= i && i < _framelocals.size());
    PyObject* value = _framelocals[i].ptr();
    // NOTE: CPython's PyFrame_FastToLocalsWithError/map_to_dict
    // removes the local name from the locals dict if the value is NULL.
    // This is likely so that if a local variable is deleted in the fastlocals,
    // PyFrame_FastToLocalsWithError will also remove it from frame->f_locals.
    // Since we create the locals dict from scratch every time (and only
    // before a frame is run), we probably don't need to account for this
    // codepath, saving us from unnecessarily calling _dict.pop().
    // It is unexpected that multiple fastlocal values corresponding to
    // the same variable name have both a null and non-null value.
    if (value != nullptr) {
      _dict[framelocals_names[i]] = value;
    }
  };

  // locals
  py::tuple varnames = _code_obj.attr("co_varnames");
  auto nlocals = std::min(co->co_nlocals, (int)varnames.size());
  for (int i = 0; i < nlocals; i++) {
    update_mapping(i);
  }

  // cellvars
  auto ncells = PyCode_GetNCellvars(co);
  for (int i = 0; i < ncells; i++) {
    update_mapping(co->co_nlocals + i);
  }

  // freevars
  if (co->co_flags & CO_OPTIMIZED) {
    auto nfree = PyCode_GetNFreevars(co);
    for (int i = 0; i < nfree; i++) {
      update_mapping(co->co_nlocals + ncells + i);
    }
  }
}

py::tuple code_framelocals_names(py::handle code) {
  CHECK(PyCode_Check(code.ptr()));
  py::tuple names = code.attr("co_varnames") + code.attr("co_cellvars");
  if (((PyCodeObject*)code.ptr())->co_flags & CO_OPTIMIZED) {
    names += code.attr("co_freevars");
  }
  return names;
}

#endif

PyObject* FrameLocalsMapping::get(int idx) {
  DEBUG_CHECK(0 <= idx && idx < _framelocals.size());
  return _framelocals[idx].ptr();
}

PyDictObject* framelocals_mapping_to_dict(FrameLocalsMapping* map) {
  return map->to_dict();
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/dynamo/framelocals_mapping.h`
- `torch/csrc/dynamo/cpython_defs.h`
- `torch/csrc/dynamo/cpython_includes.h`
- `torch/csrc/dynamo/debug_macros.h`
- `internal/pycore_code.h`


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

- **File Documentation**: `framelocals_mapping.cpp_docs.md`
- **Keyword Index**: `framelocals_mapping.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/dynamo`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/dynamo`):

- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`cpython_includes.h_kw.md_docs.md`](./cpython_includes.h_kw.md_docs.md)
- [`stackref_bridge.c_docs.md_docs.md`](./stackref_bridge.c_docs.md_docs.md)
- [`eval_frame.c_docs.md_docs.md`](./eval_frame.c_docs.md_docs.md)
- [`extra_state.h_docs.md_docs.md`](./extra_state.h_docs.md_docs.md)
- [`cache_entry.h_kw.md_docs.md`](./cache_entry.h_kw.md_docs.md)
- [`compiled_autograd.h_docs.md_docs.md`](./compiled_autograd.h_docs.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`extra_state.h_kw.md_docs.md`](./extra_state.h_kw.md_docs.md)
- [`extra_state.cpp_kw.md_docs.md`](./extra_state.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `framelocals_mapping.cpp_docs.md_docs.md`
- **Keyword Index**: `framelocals_mapping.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
