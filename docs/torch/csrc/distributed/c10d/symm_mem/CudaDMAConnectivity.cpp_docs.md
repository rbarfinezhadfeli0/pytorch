# Documentation: `torch/csrc/distributed/c10d/symm_mem/CudaDMAConnectivity.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/symm_mem/CudaDMAConnectivity.cpp`
- **Size**: 4,866 bytes (4.75 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <fmt/printf.h>

#include <cuda_runtime.h>
#include <nvml.h>

namespace {

constexpr int max_nvlinks = 64;

std::string get_bus_id(int device_idx) {
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_idx));
  return fmt::sprintf(
      NVML_DEVICE_PCI_BUS_ID_FMT,
      prop.pciDomainID,
      prop.pciBusID,
      prop.pciDeviceID);
}

struct C10_EXPORT NVLinkDetector : public c10d::DMAConnectivityDetector {
  c10::intrusive_ptr<c10d::DMAConnectivity> detect() override {
    int num_devices = 0;
    C10_CUDA_CHECK(cudaGetDeviceCount(&num_devices));

    std::vector<std::vector<int>> matrix;
    matrix.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      matrix.emplace_back(num_devices, 0);
    }

    // Obtain the bus_id for all visible devices
    std::unordered_map<std::string, int> bus_id_to_device_idx;
    bus_id_to_device_idx.reserve(num_devices);
    std::vector<std::string> bus_ids;
    bus_ids.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      auto bus_id = get_bus_id(i);
      bus_id_to_device_idx.emplace(bus_id, i);
      bus_ids.push_back(std::move(bus_id));
    }

    static constexpr const char* warning_msg =
        "PyTorch features that use NVLinkDetector may assume no NVLink presence.";

    auto driver_api = c10::cuda::DriverAPI::get();
    if (driver_api->nvmlInit_v2_() != NVML_SUCCESS) {
      LOG(WARNING)
          << "NVLinkDetector: Failed to initialize NVML via nvmlInit_v2. "
          << warning_msg;
      return c10::make_intrusive<c10d::DMAConnectivity>(
          c10::DeviceType::CUDA, "nvlink", std::move(matrix));
    }

    // Obtain the nvml device for all bus_ids
    std::vector<nvmlDevice_t> nvml_devices(num_devices, nullptr);
    for (int i = 0; i < num_devices; ++i) {
      auto res = driver_api->nvmlDeviceGetHandleByPciBusId_v2_(
          bus_ids[i].c_str(), &nvml_devices[i]);
      if (res != NVML_SUCCESS) {
        LOG(WARNING) << "NVLinkDetector: Failed to obtain NVML device via "
                     << "nvmlDeviceGetHandleByPciBusId_v2. " << warning_msg;
        return c10::make_intrusive<c10d::DMAConnectivity>(
            c10::DeviceType::CUDA, "nvlink", std::move(matrix));
      }
    }

    std::vector<int> switch_link_count(num_devices, 0);
    for (int i = 0; i < num_devices; ++i) {
      for (int link = 0; link < max_nvlinks; ++link) {
        nvmlIntNvLinkDeviceType_t deviceType{};
        auto ret = driver_api->nvmlDeviceGetNvLinkRemoteDeviceType_(
            nvml_devices[i], link, &deviceType);
        if (ret != NVML_SUCCESS) {
          // We've exhausted the NVLinks connected to this device. This error
          // is benign. There doesn't seem to be a reliable way to obtain the
          // maximum link value that can be passed to the API. Therefore, we
          // simply increment the link value until the API fails or we reach a
          // predefined maximum value.
          break;
        }
        // Remote device is GPU
        if (deviceType == NVML_NVLINK_DEVICE_TYPE_GPU) {
          nvmlPciInfo_t pciInfo;
          auto res = driver_api->nvmlDeviceGetNvLinkRemotePciInfo_v2_(
              nvml_devices[i], link, &pciInfo);
          if (res != NVML_SUCCESS) {
            LOG(WARNING) << "NVLinkDetector: Failed to obtain NVML device via "
                         << "nvmlDeviceGetHandleByPciBusId_v2. " << warning_msg;
            return c10::make_intrusive<c10d::DMAConnectivity>(
                c10::DeviceType::CUDA, "nvlink", std::move(matrix));
          }
          auto it = bus_id_to_device_idx.find(pciInfo.busId);
          if (it != bus_id_to_device_idx.end()) {
            if (i != it->second) {
              matrix[i][it->second] += 1;
            }
          }
          // Remote device is NVSwitch
        } else if (deviceType == NVML_NVLINK_DEVICE_TYPE_SWITCH) {
          switch_link_count[i] += 1;
        }
      }
    }

    // Process NVSwitch connections.
    // For simplicity, we assume that all NVSwitches are interconnected.
    for (int i = 0; i < num_devices; ++i) {
      for (int j = 0; j < num_devices; ++j) {
        if (i == j) {
          continue;
        }
        matrix[i][j] += std::min(switch_link_count[i], switch_link_count[j]);
      }
    }

    return c10::make_intrusive<c10d::DMAConnectivity>(
        c10::DeviceType::CUDA, "nvlink", std::move(matrix));
  }
};

struct RegisterDetector {
  RegisterDetector() {
    register_dma_connectivity_detector(
        c10::DeviceType::CUDA, "nvlink", c10::make_intrusive<NVLinkDetector>());
  }
};

static RegisterDetector register_detector_;

} // namespace
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `C10_EXPORT`, `RegisterDetector`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d/symm_mem`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp`
- `c10/cuda/CUDAException.h`
- `c10/cuda/driver_api.h`
- `fmt/printf.h`
- `cuda_runtime.h`
- `nvml.h`


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

- **File Documentation**: `CudaDMAConnectivity.cpp_docs.md`
- **Keyword Index**: `CudaDMAConnectivity.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
