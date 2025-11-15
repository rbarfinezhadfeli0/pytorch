# Documentation: `docs/torch/csrc/inductor/aoti_torch/c/shim_mps.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/inductor/aoti_torch/c/shim_mps.h_docs.md`
- **Size**: 5,282 bytes (5.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/inductor/aoti_torch/c/shim_mps.h`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_torch/c/shim_mps.h`
- **Size**: 3,189 bytes (3.11 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#ifndef AOTI_TORCH_SHIM_MPS
#define AOTI_TORCH_SHIM_MPS

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

struct AOTIMetalKernelFunctionOpaque;
using AOTIMetalKernelFunctionHandle = AOTIMetalKernelFunctionOpaque*;

struct AOTIMetalShaderLibraryOpaque;
using AOTIMetalShaderLibraryHandle = AOTIMetalShaderLibraryOpaque*;

#ifdef __cplusplus
extern "C" {
#endif

// MetalShaderLibrary functions
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_create_shader_library(
    const char* metal_shader_source,
    AOTIMetalShaderLibraryHandle* library_handle);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle library_handle);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle library_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* function_handle);

// MetalKernelFunction functions
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_mps_start_encoding(AOTIMetalKernelFunctionHandle func);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AtenTensorHandle tensor);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    int64_t val);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length,
    uint64_t group_size);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_dispatch_array(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size,
    const uint64_t* group_size,
    size_t group_size_size);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_mps_malloc(void** buffer, size_t num_bytes);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_free(void* ptr);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_memcpy(
    void* buffer,
    size_t constant_offset,
    size_t bytes_read,
    size_t data_size,
    uint8_t* constants_start);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_copy_buffer(
    void* src_buffer,
    void* dst_buffer,
    size_t data_size,
    size_t src_offset,
    size_t dst_offset);

// C callback function type for command block execution
typedef void (*aoti_torch_mps_command_block_callback_t)(
    AOTIMetalKernelFunctionHandle func,
    void* user_data);

// Shared callback function for std::function trampoline
AOTI_TORCH_EXPORT void aoti_torch_mps_shared_callback(
    AOTIMetalKernelFunctionHandle func,
    void* user_data);

// Pure C version using function pointer and user data for trampoline pattern
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_run_command_block(
    AOTIMetalKernelFunctionHandle func,
    aoti_torch_mps_command_block_callback_t callback,
    void* user_data);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOTI_TORCH_SHIM_MPS

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `AOTIMetalKernelFunctionOpaque`, `AOTIMetalShaderLibraryOpaque`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_torch/c`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/inductor/aoti_torch/c/shim.h`


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

Files in the same folder (`torch/csrc/inductor/aoti_torch/c`):

- [`macros.h_docs.md`](./macros.h_docs.md)
- [`shim_cpu.h_docs.md`](./shim_cpu.h_docs.md)
- [`shim.h_docs.md`](./shim.h_docs.md)
- [`shim_xpu.h_docs.md`](./shim_xpu.h_docs.md)
- [`shim_deprecated.h_docs.md`](./shim_deprecated.h_docs.md)


## Cross-References

- **File Documentation**: `shim_mps.h_docs.md`
- **Keyword Index**: `shim_mps.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/inductor/aoti_torch/c`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/inductor/aoti_torch/c`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/inductor/aoti_torch/c`):

- [`shim_deprecated.h_kw.md_docs.md`](./shim_deprecated.h_kw.md_docs.md)
- [`shim_cpu.h_docs.md_docs.md`](./shim_cpu.h_docs.md_docs.md)
- [`shim_xpu.h_docs.md_docs.md`](./shim_xpu.h_docs.md_docs.md)
- [`macros.h_kw.md_docs.md`](./macros.h_kw.md_docs.md)
- [`shim.h_kw.md_docs.md`](./shim.h_kw.md_docs.md)
- [`shim_deprecated.h_docs.md_docs.md`](./shim_deprecated.h_docs.md_docs.md)
- [`shim_mps.h_kw.md_docs.md`](./shim_mps.h_kw.md_docs.md)
- [`macros.h_docs.md_docs.md`](./macros.h_docs.md_docs.md)
- [`shim_cpu.h_kw.md_docs.md`](./shim_cpu.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `shim_mps.h_docs.md_docs.md`
- **Keyword Index**: `shim_mps.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
