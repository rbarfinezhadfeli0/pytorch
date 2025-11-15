# Documentation: `docs/torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cuh_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cuh_docs.md`
- **Size**: 4,863 bytes (4.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cuh`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cuh`
- **Size**: 2,385 bytes (2.33 KB)
- **Type**: CUDA Header File
- **Extension**: `.cuh`

## File Purpose

This is a cuda header file that is part of the PyTorch project.

## Original Source

```
#pragma once

#include <c10/macros/Macros.h>
#include <ATen/ATen.h>

#define NVSHMEM_CHECK(stmt, msg)                                             \
  do {                                                                       \
    int result = (stmt);                                                     \
    TORCH_CHECK(                                                             \
        result == 0,                                                         \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + msg + \
            ". Error code: " + std::to_string(result));                      \
  } while (0)

namespace c10d::nvshmem_extension {

// Check if NVSHMEM is available
TORCH_API bool is_nvshmem_available();

// Initializes the device state in CUmodule so that it’s able to perform NVSHMEM
// operations.
TORCH_API void nvshmemx_cumodule_init(uintptr_t module);

TORCH_API void nvshmem_put(at::Tensor& tensor, const int64_t peer);

TORCH_API void nvshmem_get(at::Tensor& tensor, const int64_t peer);

at::Tensor nvshmem_broadcast(at::Tensor& input, const int64_t root, const std::string& group_name);

TORCH_API void nvshmem_wait_for_signal(at::Tensor& sigpad, int64_t signal, int64_t peer);

TORCH_API void nvshmem_put_with_signal(at::Tensor& tensor, at::Tensor& sigpad, int64_t signal, int64_t peer);

at::Tensor nvshmem_all_to_all(
    at::Tensor& input,
    at::Tensor& out,
    std::string group_name);

void all_to_all_vdev(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    std::string group_name);

void all_to_all_vdev_2d(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    std::string group_name,
    std::optional<int64_t> major_align = std::nullopt);

void all_to_all_vdev_2d_offset(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits_offsets,
    at::Tensor& out_splits_offsets,
    std::string group_name);

void tile_reduce(
    at::Tensor& in_tile,
    at::Tensor& out_tile,
    int64_t root,
    std::string group_name,
    std::string reduce_op = "sum");

void multi_root_tile_reduce(
    at::ArrayRef<at::Tensor> in_tiles,
    at::Tensor& out_tile,
    at::ArrayRef<int64_t> roots,
    std::string group_name,
    std::string reduce_op = "sum");

} // namespace c10d::nvshmem_extension

```



## High-Level Overview

This file is part of the PyTorch framework located at `torch/csrc/distributed/c10d/symm_mem`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d/symm_mem`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/macros/Macros.h`
- `ATen/ATen.h`


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

Files in the same folder (`torch/csrc/distributed/c10d/symm_mem`):

- [`SymmetricMemory.cpp_docs.md`](./SymmetricMemory.cpp_docs.md)
- [`CUDASymmetricMemoryOps.cu_docs.md`](./CUDASymmetricMemoryOps.cu_docs.md)
- [`NVSHMEMSymmetricMemory.cu_docs.md`](./NVSHMEMSymmetricMemory.cu_docs.md)
- [`SymmetricMemory.hpp_docs.md`](./SymmetricMemory.hpp_docs.md)
- [`DMAConnectivity.hpp_docs.md`](./DMAConnectivity.hpp_docs.md)
- [`DMAConnectivity.cpp_docs.md`](./DMAConnectivity.cpp_docs.md)
- [`nvshmem_team_manager.hpp_docs.md`](./nvshmem_team_manager.hpp_docs.md)
- [`nvshmem_extension.cu_docs.md`](./nvshmem_extension.cu_docs.md)
- [`CUDASymmetricMemoryUtils.cpp_docs.md`](./CUDASymmetricMemoryUtils.cpp_docs.md)


## Cross-References

- **File Documentation**: `nvshmem_extension.cuh_docs.md`
- **Keyword Index**: `nvshmem_extension.cuh_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/c10d/symm_mem`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/c10d/symm_mem`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/distributed/c10d/symm_mem`):

- [`SymmetricMemory.hpp_docs.md_docs.md`](./SymmetricMemory.hpp_docs.md_docs.md)
- [`CUDASymmetricMemory.hpp_docs.md_docs.md`](./CUDASymmetricMemory.hpp_docs.md_docs.md)
- [`DMAConnectivity.cpp_docs.md_docs.md`](./DMAConnectivity.cpp_docs.md_docs.md)
- [`CudaDMAConnectivity.cpp_docs.md_docs.md`](./CudaDMAConnectivity.cpp_docs.md_docs.md)
- [`NCCLSymmetricMemory.cu_kw.md_docs.md`](./NCCLSymmetricMemory.cu_kw.md_docs.md)
- [`CUDASymmetricMemory.cu_kw.md_docs.md`](./CUDASymmetricMemory.cu_kw.md_docs.md)
- [`nvshmem_extension.cu_docs.md_docs.md`](./nvshmem_extension.cu_docs.md_docs.md)
- [`DMAConnectivity.hpp_docs.md_docs.md`](./DMAConnectivity.hpp_docs.md_docs.md)
- [`CUDASymmetricMemory-inl.h_kw.md_docs.md`](./CUDASymmetricMemory-inl.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `nvshmem_extension.cuh_docs.md_docs.md`
- **Keyword Index**: `nvshmem_extension.cuh_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
