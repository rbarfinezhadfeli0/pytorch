# Documentation: `docs/torch/csrc/cuda/nccl.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/cuda/nccl.h_docs.md`
- **Size**: 8,287 bytes (8.09 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/cuda/nccl.h`

## File Metadata

- **Path**: `torch/csrc/cuda/nccl.h`
- **Size**: 5,871 bytes (5.73 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstddef>
#include <optional>
#include <vector>

// NCCL BFloat16 is enabled only for CUDA 11+ and NCCL versions 2.10+, or for
// HIP 3.1+
#if defined(__CUDA_BF16_TYPES_EXIST__)
#define HAS_NCCL_BF16_DATATYPE \
  ((NCCL_MAJOR > 2) || (NCCL_MAJOR == 2) && (NCCL_MINOR >= 10))
#elif defined(USE_ROCM) && (TORCH_HIP_VERSION >= 301)
#define HAS_NCCL_BF16_DATATYPE 1
#else
#define HAS_NCCL_BF16_DATATYPE 0
#endif

namespace torch::cuda::nccl {

/* The following are copied from <nccl.h> and redefined in torch::cuda::nccl
 * namespace */
/* pytorch should only use the following definition within pytorch scope */

/* Opaque handle to communicator to ncclComm*, this will reinterpret as ncclComm
 * in nccl.cpp */
typedef void* ncclComm_t;

/** redefine nccl unique ID in torch scope. this should be identical to native
 * nccl impp. */
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct {
  // NOLINTNEXTLINE(*array*)
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

/* Error type */
enum class ncclResult {
  Success = 0,
  UnhandledCudaError = 1,
  SystemError = 2,
  InternalError = 3,
  InvalidArgument = 4,
  InvalidUsage = 5,
  RemoteError = 6,
  InProgress = 7,
  NumResults = 8
};

/* Reduction operation selector */
enum class ncclRedOp { Sum = 0, Prod = 1, Max = 2, Min = 3, NumOps = 4 };

/* Data types */
enum class ncclDataType {
  Int8 = 0,
  Char = 0,
  Uint8 = 1,
  Int32 = 2,
  Int = 2,
  Uint32 = 3,
  Int64 = 4,
  Uint64 = 5,
  Float16 = 6,
  Half = 6,
  Float32 = 7,
  Float = 7,
  Float64 = 8,
  Double = 8,
  Bfloat16 = 9,
  NumTypes = 10
};

// RAII helper class to manage NCCL group API and CUDA free mutex.
// The destructor is allowed to throw since this helper class only
// manages group and lock lifetimes.
struct TORCH_CUDA_CPP_API AutoNcclGroup {
  AutoNcclGroup();
  AutoNcclGroup(ncclComm_t comm, bool comm_nonblocking);
  ~AutoNcclGroup() noexcept(false);
  ncclComm_t comm_;
  bool comm_nonblocking_;
};

// NOTE: this is exposed only so that python_nccl.cpp can some of these helpers.
// Don't use them outside of these files.
namespace detail {

TORCH_CUDA_CPP_API void throw_nccl_error(ncclResult status);

inline void NCCL_CHECK(ncclResult status) {
  if (status != ncclResult::Success) {
    throw_nccl_error(status);
  }
}

TORCH_CUDA_CPP_API at::ArrayRef<ncclComm_t> get_communicators(
    at::TensorList inputs);
TORCH_CUDA_CPP_API void check_inputs(
    at::TensorList inputs,
    at::TensorList outputs,
    size_t input_multiplier,
    size_t output_multiplier);
TORCH_CUDA_CPP_API void check_inputs(
    at::TensorList inputs,
    const at::Tensor& output,
    int root,
    size_t input_multiplier,
    size_t output_multiplier);

} // namespace detail

using comm_list = std::vector<ncclComm_t>;
using stream_list = std::vector<std::optional<at::cuda::CUDAStream>>;

TORCH_CUDA_CPP_API std::uint64_t version();
TORCH_CUDA_CPP_API const char* version_suffix();

bool is_available(at::TensorList tensors);

TORCH_CUDA_CPP_API void get_unique_id(ncclUniqueId& id);
TORCH_CUDA_CPP_API ncclComm_t
comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank);
TORCH_CUDA_CPP_API void comm_destroy(ncclComm_t comm);

TORCH_CUDA_CPP_API void broadcast(
    at::TensorList tensors,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

size_t get_max_count();

TORCH_CUDA_CPP_API void reduce(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& output,
    int32_t root = 0,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_CPP_API void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root = 0,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_CPP_API void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_CPP_API void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_CPP_API void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream,
    int32_t root = 0);

TORCH_CUDA_CPP_API void all_gather(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_CPP_API void gather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream,
    int32_t root = 0);

TORCH_CUDA_CPP_API void all2all_single_equal_split(
    at::Tensor& input,
    at::Tensor& output,
    int size,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream);

TORCH_CUDA_CPP_API void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType type,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream);

TORCH_CUDA_CPP_API void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream);

TORCH_CUDA_CPP_API void send(
    const at::Tensor& input,
    ncclComm_t comm,
    at::cuda::CUDAStream stream,
    int dst);

TORCH_CUDA_CPP_API void recv(
    at::Tensor& output,
    ncclComm_t comm,
    at::cuda::CUDAStream stream,
    int src);
} // namespace torch::cuda::nccl

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 25 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`

**Classes/Structs**: `ncclResult`, `ncclRedOp`, `ncclDataType`, `to`, `only`, `TORCH_CUDA_CPP_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/cuda/CUDAContext.h`
- `cstddef`
- `optional`
- `vector`


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

Files in the same folder (`torch/csrc/cuda`):

- [`python_comm.cpp_docs.md`](./python_comm.cpp_docs.md)
- [`python_comm.h_docs.md`](./python_comm.h_docs.md)
- [`memory_snapshot.cpp_docs.md`](./memory_snapshot.cpp_docs.md)
- [`GdsFile.h_docs.md`](./GdsFile.h_docs.md)
- [`GreenContext.cpp_docs.md`](./GreenContext.cpp_docs.md)
- [`CUDAPluggableAllocator.cpp_docs.md`](./CUDAPluggableAllocator.cpp_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`Module.h_docs.md`](./Module.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)


## Cross-References

- **File Documentation**: `nccl.h_docs.md`
- **Keyword Index**: `nccl.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/cuda`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/cuda`):

- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`python_nccl.h_docs.md_docs.md`](./python_nccl.h_docs.md_docs.md)
- [`THCP.h_docs.md_docs.md`](./THCP.h_docs.md_docs.md)
- [`GreenContext.cpp_kw.md_docs.md`](./GreenContext.cpp_kw.md_docs.md)
- [`CUDAPluggableAllocator.cpp_docs.md_docs.md`](./CUDAPluggableAllocator.cpp_docs.md_docs.md)
- [`GdsFile.cpp_kw.md_docs.md`](./GdsFile.cpp_kw.md_docs.md)
- [`python_comm.cpp_kw.md_docs.md`](./python_comm.cpp_kw.md_docs.md)
- [`GdsFile.cpp_docs.md_docs.md`](./GdsFile.cpp_docs.md_docs.md)
- [`Module.cpp_docs.md_docs.md`](./Module.cpp_docs.md_docs.md)
- [`CUDAPluggableAllocator.h_docs.md_docs.md`](./CUDAPluggableAllocator.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `nccl.h_docs.md_docs.md`
- **Keyword Index**: `nccl.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
