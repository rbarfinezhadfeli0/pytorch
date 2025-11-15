# Documentation: `docs/torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp_docs.md`
- **Size**: 5,184 bytes (5.06 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp`
- **Size**: 2,325 bytes (2.27 KB)
- **Type**: C++ Header File
- **Extension**: `.hpp`

## File Purpose

This is a c++ header file that is part of the PyTorch project.

## Original Source

```cpp
#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

namespace c10d::intra_node_comm {

using namespace c10d::symmetric_memory;

constexpr size_t kMaxDevices = 8;
constexpr size_t kDefaultBufferSize = 10ull * 1024 * 1024;

using NvlMesh = std::array<std::array<size_t, kMaxDevices>, kMaxDevices>;

enum class Topology : uint8_t {
  UNKNOWN = 0,
  FULLY_CONNECTED = 1,
};

enum class AllReduceAlgo : uint8_t {
  NONE = 0,
  ONE_SHOT = 1,
  TWO_SHOT = 2,
};

// NOTE: this class will be be removed soon in favor of SymmetricMemory
class TORCH_API IntraNodeComm : public c10::intrusive_ptr_target {
 public:
  IntraNodeComm(
      c10::intrusive_ptr<c10d::Store> store,
      size_t rank,
      size_t worldSize,
      std::optional<size_t> bufferSize = std::nullopt);

  ~IntraNodeComm() override;

  static bool isEnabled();

  /**
   * Performs rendezvous.
   * If rendezvous fails, the IntraNodeComm object will be in an invalid
   * state and it is the caller's responsibility to dispose it.
   */
  bool rendezvous();

  /**
   * Selects a AllReduceAlgo that we think will outperform nccl.
   * Returns AllReduceAlgo::NONE if we don't think we can outperform nccl.
   */
  AllReduceAlgo selectAllReduceAlgo(const at::Tensor& input);

  at::Tensor allReduce(const at::Tensor& input, AllReduceAlgo algo);

 private:
  at::Tensor oneShotAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  at::Tensor twoShotAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  c10::intrusive_ptr<Store> store_;
  size_t rank_;
  size_t worldSize_;
  size_t bufferSize_;

  /**
   * Members initialized after rendezvous
   */
  bool isInitialized_ = false;
  int deviceIdx_{0};
  Topology topology_ = Topology::UNKNOWN;
  void* symmetricMemoryPtr_ = nullptr;
  c10::intrusive_ptr<SymmetricMemory> symmetricMemory_ = nullptr;
};

class IntraNodeCommWork : public c10d::Work {
 public:
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    return true;
  }
};

TORCH_API int64_t getIntraNodeCommUsageCounter();

bool isIntraNodeCommSupported();
} // namespace c10d::intra_node_comm

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `Topology`, `AllReduceAlgo`, `will`, `TORCH_API`, `IntraNodeCommWork`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d/symm_mem`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `c10/cuda/CUDAStream.h`
- `torch/csrc/distributed/c10d/Store.hpp`
- `torch/csrc/distributed/c10d/Work.hpp`
- `torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp`


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

Files in the same folder (`torch/csrc/distributed/c10d/symm_mem`):

- [`SymmetricMemory.cpp_docs.md`](./SymmetricMemory.cpp_docs.md)
- [`CUDASymmetricMemoryOps.cu_docs.md`](./CUDASymmetricMemoryOps.cu_docs.md)
- [`NVSHMEMSymmetricMemory.cu_docs.md`](./NVSHMEMSymmetricMemory.cu_docs.md)
- [`SymmetricMemory.hpp_docs.md`](./SymmetricMemory.hpp_docs.md)
- [`DMAConnectivity.hpp_docs.md`](./DMAConnectivity.hpp_docs.md)
- [`DMAConnectivity.cpp_docs.md`](./DMAConnectivity.cpp_docs.md)
- [`nvshmem_team_manager.hpp_docs.md`](./nvshmem_team_manager.hpp_docs.md)
- [`nvshmem_extension.cu_docs.md`](./nvshmem_extension.cu_docs.md)
- [`nvshmem_extension.cuh_docs.md`](./nvshmem_extension.cuh_docs.md)
- [`CUDASymmetricMemoryUtils.cpp_docs.md`](./CUDASymmetricMemoryUtils.cpp_docs.md)


## Cross-References

- **File Documentation**: `intra_node_comm.hpp_docs.md`
- **Keyword Index**: `intra_node_comm.hpp_kw.md`
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
- [`nvshmem_extension.cuh_docs.md_docs.md`](./nvshmem_extension.cuh_docs.md_docs.md)
- [`DMAConnectivity.cpp_docs.md_docs.md`](./DMAConnectivity.cpp_docs.md_docs.md)
- [`CudaDMAConnectivity.cpp_docs.md_docs.md`](./CudaDMAConnectivity.cpp_docs.md_docs.md)
- [`NCCLSymmetricMemory.cu_kw.md_docs.md`](./NCCLSymmetricMemory.cu_kw.md_docs.md)
- [`CUDASymmetricMemory.cu_kw.md_docs.md`](./CUDASymmetricMemory.cu_kw.md_docs.md)
- [`nvshmem_extension.cu_docs.md_docs.md`](./nvshmem_extension.cu_docs.md_docs.md)
- [`DMAConnectivity.hpp_docs.md_docs.md`](./DMAConnectivity.hpp_docs.md_docs.md)
- [`CUDASymmetricMemory-inl.h_kw.md_docs.md`](./CUDASymmetricMemory-inl.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `intra_node_comm.hpp_docs.md_docs.md`
- **Keyword Index**: `intra_node_comm.hpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
