# Documentation: `docs/torch/csrc/stable/c/shim.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/stable/c/shim.h_docs.md`
- **Size**: 5,179 bytes (5.06 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/stable/c/shim.h`

## File Metadata

- **Path**: `torch/csrc/stable/c/shim.h`
- **Size**: 3,332 bytes (3.25 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#ifndef STABLE_TORCH_SHIM
#define STABLE_TORCH_SHIM

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <torch/csrc/stable/version.h>

// This header defines stable C API extensions for backward/forward
// compatibility when calling ATen operations through the dispatcher.
//
// This is separate from the main AOTI shim to provide versioning capabilities
// for schema changes in native ATen functions.

#ifdef __cplusplus
extern "C" {
#endif

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
using StableIValue = uint64_t;

// Has the same semantic as aoti_torch_call_dispatcher, but takes an
// additional argument for the extension build version. This is
// needed for backward compatibility when calling native functions via
// the dispatcher. The caller should pass in the libtorch version the
// extension is building with (NOT target version).
AOTI_TORCH_EXPORT AOTITorchError torch_call_dispatcher(
    const char* opName,
    const char* overloadName,
    StableIValue* stack,
    uint64_t extension_build_version);

// Version-aware variant of aoti_torch_library_impl that takes an
// extension_build_version parameter for backward compatibility
AOTI_TORCH_EXPORT AOTITorchError torch_library_impl(
    TorchLibraryHandle self,
    const char* name,
    void (*fn)(StableIValue*, uint64_t, uint64_t),
    uint64_t extension_build_version);

struct StableListOpaque;
using StableListHandle = StableListOpaque*;

// returns an owning reference of a StableList. callee is responsible for
// freeing memory.
AOTI_TORCH_EXPORT AOTITorchError
torch_new_list_reserve_size(size_t size, StableListHandle* ret);

AOTI_TORCH_EXPORT AOTITorchError
torch_list_size(StableListHandle list_handle, size_t* size);

AOTI_TORCH_EXPORT AOTITorchError torch_list_get_item(
    StableListHandle list_handle,
    size_t index,
    StableIValue* element);

AOTI_TORCH_EXPORT AOTITorchError torch_list_set_item(
    StableListHandle list_handle,
    size_t index,
    StableIValue element);

AOTI_TORCH_EXPORT AOTITorchError
torch_list_push_back(StableListHandle list_handle, StableIValue element);

// deletes the underlying list referenced by list_handle
AOTI_TORCH_EXPORT AOTITorchError
torch_delete_list(StableListHandle list_handle);

// Helper function to parse device string using c10::Device
// Returns device type and index via output parameters
AOTI_TORCH_EXPORT AOTITorchError torch_parse_device_string(
    const char* device_string,
    uint32_t* out_device_type,
    int32_t* out_device_index);

// Parallel utility APIs for stable ABI
// Function pointer type for parallel_for callback
// The callback receives begin and end indices for a range to process
typedef void (*ParallelFunc)(int64_t begin, int64_t end, void* ctx);

AOTI_TORCH_EXPORT AOTITorchError torch_parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    ParallelFunc func,
    void* ctx);

// Get the current thread index in a parallel region
// Returns 0 if not in a parallel region
AOTI_TORCH_EXPORT AOTITorchError torch_get_thread_idx(uint32_t* out_thread_idx);

// Get the number of threads for the parallel backend
AOTI_TORCH_EXPORT AOTITorchError
torch_get_num_threads(uint32_t* out_num_threads);

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STABLE_TORCH_SHIM

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `StableListOpaque`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/stable/c`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/inductor/aoti_torch/c/shim.h`
- `torch/csrc/stable/version.h`


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

Files in the same folder (`torch/csrc/stable/c`):



## Cross-References

- **File Documentation**: `shim.h_docs.md`
- **Keyword Index**: `shim.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/stable/c`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/stable/c`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/stable/c`):

- [`shim.h_kw.md_docs.md`](./shim.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `shim.h_docs.md_docs.md`
- **Keyword Index**: `shim.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
