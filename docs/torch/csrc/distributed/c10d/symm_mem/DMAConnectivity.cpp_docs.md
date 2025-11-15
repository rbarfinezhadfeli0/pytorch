# Documentation: `torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.cpp`
- **Size**: 2,654 bytes (2.59 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>
#include <utility>

namespace {

std::string get_detector_key(
    c10::DeviceType device_type,
    const std::string& connection_type) {
  std::ostringstream oss;
  oss << device_type << "/" << connection_type;
  return oss.str();
}

class DetectorMap {
 public:
  DetectorMap(const DetectorMap&) = delete;
  DetectorMap& operator=(const DetectorMap&) = delete;
  DetectorMap(DetectorMap&&) = delete;
  DetectorMap& operator=(DetectorMap&&) = delete;
  ~DetectorMap() = default;
  static DetectorMap& get() {
    static DetectorMap instance;
    return instance;
  }

  void register_detector(
      c10::DeviceType device_type,
      const std::string& connection_type,
      c10::intrusive_ptr<c10d::DMAConnectivityDetector> detector) {
    auto key = get_detector_key(device_type, connection_type);
    detector_map_[key] = std::move(detector);
  }

  c10::intrusive_ptr<c10d::DMAConnectivity> detect(
      c10::DeviceType device_type,
      const std::string& connection_type) {
    auto key = get_detector_key(device_type, connection_type);
    {
      auto it = cached_.find(key);
      if (it != cached_.end()) {
        return it->second;
      }
    }

    auto it = detector_map_.find(key);
    TORCH_CHECK(
        it != detector_map_.end(),
        "DMA connectivity detector for ",
        device_type,
        " over ",
        connection_type,
        " is not available");
    auto detector = it->second;
    auto connectivity = detector->detect();
    cached_[key] = connectivity;
    return connectivity;
  }

 private:
  DetectorMap() = default;

  std::unordered_map<
      std::string,
      c10::intrusive_ptr<c10d::DMAConnectivityDetector>>
      detector_map_;

  std::unordered_map<std::string, c10::intrusive_ptr<c10d::DMAConnectivity>>
      cached_;
};

} // namespace

namespace c10d {

DMAConnectivity::DMAConnectivity(
    c10::DeviceType device_type,
    std::string connection_type,
    std::vector<std::vector<int>> matrix)
    : device_type(device_type),
      connection_type(std::move(connection_type)),
      matrix(std::move(matrix)) {}

void register_dma_connectivity_detector(
    c10::DeviceType device_type,
    const std::string& connection_type,
    c10::intrusive_ptr<DMAConnectivityDetector> detector) {
  return DetectorMap::get().register_detector(
      device_type, connection_type, std::move(detector));
}

c10::intrusive_ptr<DMAConnectivity> detect_dma_connectivity(
    c10::DeviceType device_type,
    const std::string& connection_type) {
  return DetectorMap::get().detect(device_type, connection_type);
}

} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `c10d`

**Classes/Structs**: `DetectorMap`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d/symm_mem`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp`
- `utility`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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
- [`nvshmem_team_manager.hpp_docs.md`](./nvshmem_team_manager.hpp_docs.md)
- [`nvshmem_extension.cu_docs.md`](./nvshmem_extension.cu_docs.md)
- [`nvshmem_extension.cuh_docs.md`](./nvshmem_extension.cuh_docs.md)
- [`CUDASymmetricMemoryUtils.cpp_docs.md`](./CUDASymmetricMemoryUtils.cpp_docs.md)


## Cross-References

- **File Documentation**: `DMAConnectivity.cpp_docs.md`
- **Keyword Index**: `DMAConnectivity.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
