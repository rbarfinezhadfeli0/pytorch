# Documentation: `docs/torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cpp_docs.md`
- **Size**: 9,434 bytes (9.21 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cpp`
- **Size**: 6,632 bytes (6.48 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp>

#if defined(USE_ROCM)
#include <rocm_smi/rocm_smi.h>
#endif

namespace c10d::intra_node_comm {

static std::vector<std::string> ENABLE_INTRA_NODE_COMM = {
    "ENABLE_INTRA_NODE_COMM"};
// Forces detectedTopology() to return Topology::FULLY_CONNECTED, so
// IntraNodeComm can be used even without NVLink connection. This is only used
// for testing purposes.
static std::vector<std::string> TEST_INTRA_NODE_COMM = {"TEST_INTRA_NODE_COMM"};
static int intraNodeCommIdx = 0;

/**
 * Query the nvlink connection among devices.
 */
static NvlMesh getNvlMesh(const std::vector<int>& rankToDeviceIdx) {
#if !defined(USE_RCOM)
  auto connectivity = detect_dma_connectivity(c10::DeviceType::CUDA, "nvlink");
  NvlMesh nvlMesh = {};
  for (size_t srcRank = 0; srcRank < kMaxDevices; ++srcRank) {
    for (size_t dstRank = 0; dstRank < kMaxDevices; ++dstRank) {
      if (srcRank < rankToDeviceIdx.size() &&
          dstRank < rankToDeviceIdx.size()) {
        nvlMesh[srcRank][dstRank] =
            connectivity
                ->matrix[rankToDeviceIdx[srcRank]][rankToDeviceIdx[dstRank]];
      }
    }
  }
  return nvlMesh;
#else
  NvlMesh nvlMesh = {};
  const auto worldSize = rankToDeviceIdx.size();
  // For each device, loop over devices connected to it
  for (size_t idx = 0; idx < worldSize; ++idx) {
    for (size_t link = 0; link < kMaxDevices; ++link) {
      if (idx == link)
        continue;

      bool conn = false;
      auto ret = rsmi_is_P2P_accessible(idx, link, &conn);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(ERROR)
            << "IntraNodeComm: getNvlMesh: rsmi_is_P2P_accessible returned error ret="
            << ret;
        return {};
      }

      if (conn) {
        nvlMesh[idx][link] += 1;
      }
    }
  }
  return nvlMesh;
#endif
}

/**
 * Detect topology given a NvlMesh.
 */
static Topology detectTopology(const NvlMesh nvlMesh, size_t worldSize) {
  if (getCvarBool(TEST_INTRA_NODE_COMM, false)) {
    return Topology::FULLY_CONNECTED;
  }
  bool fullyConnected = true;
  for (size_t i = 0; i < worldSize - 1; ++i) {
    for (size_t j = i + 1; j < worldSize; ++j) {
      if (nvlMesh[i][j] == 0 || nvlMesh[j][i] == 0) {
        fullyConnected = false;
      }
    }
  }
  if (fullyConnected) {
    LOG(INFO) << "IntraNodeComm: Topology::FULLY_CONNECTED";
    return Topology::FULLY_CONNECTED;
  }
  LOG(INFO) << "IntraNodeComm: Topology::UNKNOWN";
  return Topology::UNKNOWN;
}

IntraNodeComm::IntraNodeComm(
    c10::intrusive_ptr<c10d::Store> store,
    size_t rank,
    size_t worldSize,
    std::optional<size_t> bufferSize)
    : store_(std::move(store)),
      rank_(rank),
      worldSize_(worldSize),
      bufferSize_(bufferSize.has_value() ? *bufferSize : kDefaultBufferSize) {}

IntraNodeComm::~IntraNodeComm() {
  if (!isInitialized_) {
    return;
  }
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  allocator->free(symmetricMemoryPtr_);
}

bool IntraNodeComm::isEnabled() {
  return getCvarBool(ENABLE_INTRA_NODE_COMM, false);
}

/**
 * Use c10d::Store to perform allgather on a trivially copyable type.
 */
template <typename T>
static std::vector<T> storeAllGather(
    const c10::intrusive_ptr<c10d::Store>& store,
    const std::string& prefix,
    size_t rank,
    size_t worldSize,
    T val) {
  static_assert(std::is_trivially_copyable_v<T>);

  std::vector<std::string> peerKeys;
  for (size_t r = 0; r < worldSize; ++r) {
    std::ostringstream oss;
    oss << prefix << "-" << r;
    peerKeys.push_back(oss.str());
  }

  {
    std::vector<uint8_t> payload(
        reinterpret_cast<uint8_t*>(&val),
        reinterpret_cast<uint8_t*>(&val) + sizeof(T));
    store->set(peerKeys[rank], payload);
  }

  std::vector<T> peerVals;
  for (size_t r = 0; r < worldSize; ++r) {
    if (r == rank) {
      peerVals.push_back(val);
      continue;
    }
    store->wait({peerKeys[r]});
    auto payload = store->get(peerKeys[r]);
    TORCH_CHECK(payload.size() == sizeof(T));
    T peerVal{};
    std::memcpy(&peerVal, payload.data(), sizeof(T));
    peerVals.push_back(peerVal);
  }
  return peerVals;
}

bool IntraNodeComm::rendezvous() {
  if (isInitialized_) {
    return true;
  }
  if (!isIntraNodeCommSupported() || worldSize_ < 2 ||
      worldSize_ > kMaxDevices) {
    return false;
  }

  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  deviceIdx_ = at::cuda::current_device();

  // Exchange hostname and device bus ID
  struct DevInfo {
    // NOLINTNEXTLINE
    char hostname[HOST_NAME_MAX + 1];
    int deviceIdx;
  };

  DevInfo devInfo{};
  gethostname(devInfo.hostname, sizeof(devInfo.hostname));
  devInfo.deviceIdx = deviceIdx_;

#if defined(USE_ROCM)
  auto ret = rsmi_init(0);
  if (ret != RSMI_STATUS_SUCCESS) {
    LOG(ERROR) << "IntraNodeComm:: rendezvous failed in rsmi_init, ret=" << ret;
    return false;
  }
#endif

  auto peerDevInfos =
      storeAllGather(store_, "handshake-0", rank_, worldSize_, devInfo);

  std::vector<int> rankToDeviceIdx;
  for (const auto& info : peerDevInfos) {
    if (strcmp(info.hostname, peerDevInfos.front().hostname) != 0) {
      LOG(WARNING) << "Aborting IntraNodeComm::rendezvous because some "
                      "participants are not on the same host ("
                   << info.hostname << ", " << devInfo.hostname << ")";
      return false;
    }
    rankToDeviceIdx.emplace_back(info.deviceIdx);
  }

  {
    std::unordered_set uniqueDeviceIdxs(
        rankToDeviceIdx.begin(), rankToDeviceIdx.end());
    if (uniqueDeviceIdxs.size() != worldSize_) {
      LOG(WARNING)
          << "Skipping IntraNodeComm::rendezvous() because participants have "
             "overlapping devices. To resolve this, call torch.cuda.set_device() "
             "before init_process_group().";
      return false;
    }
  }

  // Query nvlink connection
  auto nvlMesh = getNvlMesh(rankToDeviceIdx);

  // Detect topology
  topology_ = detectTopology(nvlMesh, worldSize_);
  if (topology_ != Topology::FULLY_CONNECTED) {
    return false;
  }

  auto groupName = "IntraNodeComm" + std::to_string(intraNodeCommIdx++);
  set_group_info(
      groupName, static_cast<int>(rank_), static_cast<int>(worldSize_), store_);
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  symmetricMemoryPtr_ = allocator->alloc(bufferSize_, deviceIdx_, groupName);
  symmetricMemory_ = allocator->rendezvous(symmetricMemoryPtr_, std::nullopt);
  isInitialized_ = true;
  return true;
}

} // namespace c10d::intra_node_comm

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `DevInfo`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d/symm_mem`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/Utils.hpp`
- `torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp`
- `torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp`
- `rocm_smi/rocm_smi.h`


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

- **File Documentation**: `intra_node_comm.cpp_docs.md`
- **Keyword Index**: `intra_node_comm.cpp_kw.md`
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

- **File Documentation**: `intra_node_comm.cpp_docs.md_docs.md`
- **Keyword Index**: `intra_node_comm.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
