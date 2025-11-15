# Documentation: `docs/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp_docs.md`
- **Size**: 5,892 bytes (5.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp`
- **Size**: 2,966 bytes (2.90 KB)
- **Type**: C++ Header File
- **Extension**: `.hpp`

## File Purpose

This is a c++ header file that is part of the PyTorch project.

## Original Source

```cpp
#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

namespace c10d {
namespace symmetric_memory {

bool device_has_multicast_support(int device_idx);

bool allow_overlapping_devices();

// Query environment variable to get the backend used for CUDA Symmetric Memory.
std::string getSymmMemBackendCUDA();

class IpcChannel {
 public:
  IpcChannel();
  ~IpcChannel();

  void send_fd(int dst_pid, int fd);
  int recv_fd();

  std::vector<int> all_gather_fds(
      int rank,
      const std::vector<int>& pids,
      int fd);

  int broadcast_fds(
      int rank,
      int src_rank,
      const std::vector<int>& pids,
      int fd);

 private:
  static std::string get_socket_name(int pid);

  std::string socket_name_;
  int socket_;
};

// A set of store-based exchange methods with a preset prefix typically type of
// the SymmetricMemory.  Most used as static instances at respective
// SymmetricMemory implementation files.
class StoreExchange {
 public:
  StoreExchange(const std::string& store_prefix)
      : store_prefix_(store_prefix) {}

  // Put template function in header file so that compiler can easily access it.
  template <typename T>
  std::vector<T> all_gather(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size,
      T val) {
    static_assert(std::is_trivially_copyable_v<T>);

    std::vector<std::string> peer_keys;
    peer_keys.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      std::ostringstream oss;
      oss << store_prefix_ << "/" << seq_id_ << "/" << r;
      peer_keys.push_back(oss.str());
    }
    ++seq_id_;

    {
      std::vector<uint8_t> payload(
          reinterpret_cast<uint8_t*>(&val),
          reinterpret_cast<uint8_t*>(&val) + sizeof(T));
      store->set(peer_keys[rank], payload);
    }

    std::vector<T> peer_vals;
    peer_vals.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      if (r == rank) {
        peer_vals.push_back(val);
        continue;
      }
      store->wait({peer_keys[r]});
      auto payload = store->get(peer_keys[r]);
      TORCH_CHECK(payload.size() == sizeof(T));
      T peer_val{};
      std::memcpy(&peer_val, payload.data(), sizeof(T));
      peer_vals.push_back(peer_val);
    }
    return peer_vals;
  }

  void barrier(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size) {
    // TODO: implement an efficient one?
    all_gather(store, rank, world_size, 0);
  }

 private:
  const std::string store_prefix_;
  size_t seq_id_ = 0;
};

// Returns a pointer of virtual address that is mapped to the physical memory
// held by the handle.
void map_block(
    void** ptr,
    c10d::symmetric_memory::HandleType handle,
    size_t size,
    int device_idx);

} // namespace symmetric_memory
} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `symmetric_memory`, `c10d`

**Classes/Structs**: `IpcChannel`, `StoreExchange`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d/symm_mem`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/Store.hpp`
- `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp`
- `torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `CUDASymmetricMemoryUtils.hpp_docs.md`
- **Keyword Index**: `CUDASymmetricMemoryUtils.hpp_kw.md`
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

- **File Documentation**: `CUDASymmetricMemoryUtils.hpp_docs.md_docs.md`
- **Keyword Index**: `CUDASymmetricMemoryUtils.hpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
