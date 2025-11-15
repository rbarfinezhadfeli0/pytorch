# Documentation: `docs/torch/csrc/dynamo/framelocals_mapping.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/dynamo/framelocals_mapping.h_docs.md`
- **Size**: 5,191 bytes (5.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/dynamo/framelocals_mapping.h`

## File Metadata

- **Path**: `torch/csrc/dynamo/framelocals_mapping.h`
- **Size**: 2,781 bytes (2.72 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/utils/python_compat.h>

#ifdef __cplusplus

#include <string>
#include <unordered_map>

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>

extern "C" {

#if IS_PYTHON_3_11_PLUS
using FrameLocalsFrameType = _PyInterpreterFrame;
#else
using FrameLocalsFrameType = PyFrameObject;
#endif // IS_PYTHON_3_11_PLUS

/**
 * Utility to view a frame's localsplus (locals + cells + freevars)
 * in C/C++ and Python, without changing the state of the frame.
 *
 * Notes on usage:
 *  - C/C++ can directly read the frame's localsplus using an index.
 *  - Cell/free variables are unboxed.
 *  - Can be converted into a dict for use in Python.
 *    The dict is constructed once per FrameLocalsMapping, lazily.
 *  - Lifetime should not exceed the lifetime of the frame
 *
 * How do guards use FrameLocalsMapping?
 * - When a guard accesses a frame's localsplus, we find the index of the
 *   variable name in the frame's code object and create a
 *   FrameLocalsGuardAccessor.
 * - We create a FrameLocalsMapping for the frame that we pass on to guard eval.
 * - LeafGuards/GuardManagers/GuardAccessors now need to define how they
 *   handle FrameLocalsMapping. By default, the FrameLocalsMapping is converted
 *   to a Python dict and the guard check is performed on the resulting dict.
 * - Some guard checks don't actually depend on the input arguments, e.g. they
 *   only check global state. In this case, no dict conversion of
 *   FrameLocalsMapping is done.
 * - FrameLocalsGuardAccessor is like DictGetItemGuardAccessor, except it knows
 *   how to handle FrameLocalsMapping - by using the framelocals variable name
 *   index that it was given when it was built.
 */
typedef struct VISIBILITY_HIDDEN FrameLocalsMapping {
 private:
  py::object _code_obj;
  // can't use localsplus directly due to closure variables:
  // - in 3.11+, the closure vars in the frame's closure object and
  //   the corresponding localsplus entry is nullptr
  // - regardless of Python version, we need to unbox the cell variable
  std::vector<py::handle> _framelocals;

  py::object _dict{py::none()};

  void _realize_dict();

 public:
  explicit FrameLocalsMapping(FrameLocalsFrameType* frame);

  PyObject* get(int idx);

  bool dict_realized() const {
    return _dict.is_none();
  }

  // Borrowed reference
  PyDictObject* to_dict() {
    if (this->dict_realized()) {
      _realize_dict();
    }
    return (PyDictObject*)_dict.ptr();
  }
} FrameLocalsMapping;

#else

// opaque type for C
typedef struct FrameLocalsMapping FrameLocalsMapping;

#endif

// Borrowed reference
PyDictObject* framelocals_mapping_to_dict(FrameLocalsMapping* map);

#ifdef __cplusplus
} // extern "C"

py::tuple code_framelocals_names(py::handle code);
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `VISIBILITY_HIDDEN`, `FrameLocalsMapping`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/utils/python_compat.h`
- `string`
- `unordered_map`
- `torch/csrc/dynamo/utils.h`
- `torch/csrc/utils/pybind.h`


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
- [`compiled_autograd.cpp_docs.md`](./compiled_autograd.cpp_docs.md)
- [`extra_state.h_docs.md`](./extra_state.h_docs.md)
- [`eval_frame.c_docs.md`](./eval_frame.c_docs.md)


## Cross-References

- **File Documentation**: `framelocals_mapping.h_docs.md`
- **Keyword Index**: `framelocals_mapping.h_kw.md`
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

- **File Documentation**: `framelocals_mapping.h_docs.md_docs.md`
- **Keyword Index**: `framelocals_mapping.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
