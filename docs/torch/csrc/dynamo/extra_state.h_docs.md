# Documentation: `torch/csrc/dynamo/extra_state.h`

## File Metadata

- **Path**: `torch/csrc/dynamo/extra_state.h`
- **Size**: 6,428 bytes (6.28 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <Python.h>

#include <torch/csrc/dynamo/framelocals_mapping.h>

#ifdef __cplusplus

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>
#include <list>

namespace py = pybind11;

extern "C" {

#else

#include <stdbool.h>

#endif

enum FrameAction {
  DEFAULT, // look through the cache, compile if not found
  SKIP, // eager
  RUN_ONLY, // look through the cache, run eager if not found
};

typedef struct FrameExecStrategy {
  enum FrameAction cur_action; // action to take for current frame
  enum FrameAction recursive_action; // action to take for recursive frames
} FrameExecStrategy;

// Points to the extra scratch space on the code object
extern Py_ssize_t extra_index;

// function to call when cache lookup errors
extern PyObject* guard_error_hook;

typedef PyObject FrameState;
typedef struct CacheEntry CacheEntry;

// ExtraState encasulates CacheEntry and FrameState. ExtraState is the highest
// level of abstraction of what is stored on the extra code object. Previously,
// we saved different parts on different extra indexes.  We prefer this way
// because of cleaner abstraction and faster SetExtra access.

#ifdef __cplusplus

typedef struct VISIBILITY_HIDDEN PrecompileEntry {
  py::object guard_manager;
  py::object code;
  void* root_mgr;

  PrecompileEntry(py::object gm, py::object c);
} PrecompileEntry;

typedef struct VISIBILITY_HIDDEN ExtraState {
  // A pointer to the orig_code object to prevent race conditions in invalidate
  // function.
  PyCodeObject* orig_code;
  std::list<PrecompileEntry> precompile_entries;
  // List of cache entries for compiled code objects
  std::list<CacheEntry> cache_entry_list;
  // Frame state to detect dynamic shape dims
  py::dict frame_state;
  // Actions to apply to all frames with this code object
  FrameExecStrategy strategy{DEFAULT, DEFAULT};

  ExtraState(PyCodeObject* orig_code_arg);
  CacheEntry* get_first_entry();
  void move_to_front(CacheEntry* cache_entry);
  void move_to_back(CacheEntry* cache_entry);
  void invalidate(CacheEntry* cache_entry, py::object deleted_guard_manager);
} ExtraState;

#else

typedef struct ExtraState ExtraState;
typedef struct PrecompileEntry PrecompileEntry;

#endif

// Helper to extra the cache_entry from the extra state.
// Ownership contract
// args
//  - extra_state: Borrowed
// return
//  - CacheEntry: Borrowed.
CacheEntry* extract_cache_entry(ExtraState* extra_state);

// Returns either the previously stored frame state or an empty dict.
// Ownership contract
// args
//  - extra_state: Borrowed
// return
//  - extra_state->frame_state: Borrowed.
FrameState* extract_frame_state(ExtraState* extra_state);

// Returns the FrameExecStrategy stored in extra_state.
// Ownership contract
// args
//  - extra_state: Borrowed
FrameExecStrategy extra_state_get_exec_strategy(ExtraState* extra_state);

// Set the FrameExecStrategy to be done to all frames with code object
// corresponding to this extra_state. Ownership contract
// - extra_state: Borrowed
void extra_state_set_exec_strategy(
    ExtraState* extra_state,
    FrameExecStrategy strategy);

// Ownership contract
// args
//  - code: Borrowed
// return
//  - extra_state: Borrowed.
ExtraState* get_extra_state(PyCodeObject* code);

// This is passed as freefunc to _PyEval_RequestCodeExtraIndex. This acts as a
// deleter for the object on extra scratch space. This function is called
// internally in _PyCode_SetExtra and also during the code deallocation.

// Destroys the extra state by deleting cache_entry, frame state and finally
// freeing the constructed extra state.

// Developer note - You should not call this function directly. This is called
// directly inside set_extra_state. If you are in a situation trying to call
// this function, consider if set_extra_state should be called.
void destroy_extra_state(void* obj);

// Clears the existing object sitting on the extra scratch spance and sets it
// up with the new state. Note that _PyCode_SetExtra calls the
// destroy_extra_state deleter internally, and therefore we don't call it
// explicitly here.

// Ownership contract
// args
//  - extra_state: Stolen
// return
//  - there is no return, but the extra_state is stolen, so it becomes
//  set_extra_state responsibility to clean it up. It will be deleted during
//  the reset_code, when the set_extra_state is called with NULL.

// Invariant - Dont set the extra state for the extra state that is already on
// the code object. Otherwise, we will first free up the old extra state
// (which is also the new extra state) and write something invalid on the
// scratch space.
void set_extra_state(PyCodeObject* code, ExtraState* extra_state);

// Creates a new extra state and put it on the extra scratch space of the code
// object.

// Ownership contract
// args
//  - code: Borrowed
// return:
//   - extra_state: New reference.
// These references are then further passed to set_extra_state which becomes
// the final owner of these references.
ExtraState* init_and_set_extra_state(PyCodeObject* code);

// Lookup the cache held by extra_state.
// Ownership contract
// args
//  - extra_state: Borrowed
// return:
//   - Py_None or PyCodeObject: Borrowed reference.
//   - Py_None or PyObject: Trace id of the compiled code.
void lookup(
    ExtraState* extra_state,
    FrameLocalsMapping* f_locals,
    PyObject* backend,
    PyObject** maybe_cached_code,
    const char** trace_annotation,
    bool is_skip_guard_eval_unsafe);

// Create a new cache entry at extra_state holding on to guarded_code.
// Ownership contract
// args
//  - extra_state: Borrowed
//  - guarded_code: Borrowed
// return:
//  - cache_entry: Borrowed reference
CacheEntry* create_cache_entry(
    ExtraState* extra_state,
    PyObject* guraded_code,
    PyObject* callback);

// Extracts the backend fn from the callback.
PyObject* get_backend(PyObject* callback);

#ifdef __cplusplus

} // extern "C"

// Returns the list of CacheEntry corresponding to code_obj.
// Warning: returns references whose lifetimes are controlled by C++
py::list _debug_get_cache_entry_list(const py::handle& code_obj);
void _reset_precompile_entries(const py::handle& code_obj);
void _load_precompile_entry(
    const py::handle& code_obj,
    py::object guard_manager,
    py::object dynamo_code);
py::list _debug_get_precompile_entries(const py::handle& code_obj);
void _set_lru_cache(py::object boolean);

#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `py`

**Classes/Structs**: `FrameExecStrategy`, `CacheEntry`, `VISIBILITY_HIDDEN`, `VISIBILITY_HIDDEN`, `ExtraState`, `PrecompileEntry`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `Python.h`
- `torch/csrc/dynamo/framelocals_mapping.h`
- `torch/csrc/dynamo/utils.h`
- `torch/csrc/utils/pybind.h`
- `list`
- `stdbool.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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
- [`eval_frame.c_docs.md`](./eval_frame.c_docs.md)


## Cross-References

- **File Documentation**: `extra_state.h_docs.md`
- **Keyword Index**: `extra_state.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
