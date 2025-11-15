# Documentation: `docs/aten/src/ATen/cuda/PeerToPeerAccess.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cuda/PeerToPeerAccess.cpp_docs.md`
- **Size**: 8,414 bytes (8.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cuda/PeerToPeerAccess.cpp`

## File Metadata

- **Path**: `aten/src/ATen/cuda/PeerToPeerAccess.cpp`
- **Size**: 5,729 bytes (5.59 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/cuda/PeerToPeerAccess.h>

#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <vector>

namespace at::cuda {

static std::vector<int8_t> p2pAccessEnabled_;
static std::vector<int8_t> fabricAccessEnabled_;
static int64_t num_devices_ = -1;

namespace detail {

void init_p2p_access_cache(int64_t num_devices) {
  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  // Currently the max number of gpus in P2P group is 8, so if there are more
  // we enable P2P in groups of 8
  p2pAccessEnabled_.clear();
  p2pAccessEnabled_.resize(num_devices * num_devices, -1);
  num_devices_ = num_devices;

  for (const auto i : c10::irange(num_devices)) {
    p2pAccessEnabled_[i * num_devices + i] = 1;
  }
  fabricAccessEnabled_.clear();
  fabricAccessEnabled_.resize(num_devices, -1);
}

} // namespace detail

bool get_p2p_access(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);

  TORCH_CHECK(dev >= 0 || dev < num_devices_, dev, " is not a device");
  TORCH_CHECK(
      dev_to_access >= 0 || dev_to_access < num_devices_,
      dev_to_access,
      " is not a device");
  TORCH_INTERNAL_ASSERT(num_devices_ >= 0, "p2p access cache not initialized");

  auto& cache = p2pAccessEnabled_[dev * num_devices_ + dev_to_access];

  if (cache != -1) {
    return cache;
  }

  int result = 0;
  C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&result, dev, dev_to_access));
  cache = result ? 1 : 0;
  if (cache) {
    CUDACachingAllocator::enablePeerAccess(dev, dev_to_access);
  }

  return cache;
}

namespace {
#if !defined USE_ROCM && defined CUDA_VERSION && CUDA_VERSION >= 12040 && defined PYTORCH_C10_DRIVER_API_SUPPORTED

nvmlDevice_t get_nvml_device(c10::DeviceIndex dev) {
  static bool nvml_init [[maybe_unused]] = []() {
    TORCH_INTERNAL_ASSERT(NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_());
    return true;
  }();

  auto prop = at::cuda::getDeviceProperties(dev);
  char pci_id // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      [NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  snprintf(
      pci_id,
      sizeof(pci_id),
      NVML_DEVICE_PCI_BUS_ID_FMT,
      prop->pciDomainID,
      prop->pciBusID,
      prop->pciDeviceID);

  nvmlDevice_t nvml_device = nullptr;
  TORCH_INTERNAL_ASSERT(
      NVML_SUCCESS ==
      DriverAPI::get()->nvmlDeviceGetHandleByPciBusId_v2_(
          pci_id, &nvml_device));
  return nvml_device;
}

bool isFabricSupported() {
  // 1. try allocating memory
  CUmemGenericAllocationHandle handle = 0;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

  size_t granularity{};
  const auto driver_api = c10::cuda::DriverAPI::get();
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemGetAllocationGranularity_(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

  auto status = driver_api->cuMemCreate_(&handle, granularity, &prop, 0);
  if (status != CUDA_SUCCESS) {
    LOG(INFO)
        << "status " << status
        << " Could not allocate memory with FABRIC handle, falling back to fd handle exchange\n";
    return false;
  }
  // 2. check export
  CUmemFabricHandle sharedHandle;
  status = driver_api->cuMemExportToShareableHandle_(
      &sharedHandle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
  if (status != CUDA_SUCCESS) {
    LOG(INFO)
        << "status " << status
        << " Could not export FABRIC handle, falling back to fd handle exchange\n";
    driver_api->cuMemRelease_(handle);
    return false;
  }
  // 3. check import
  CUmemGenericAllocationHandle import_handle = 0;
  status = driver_api->cuMemImportFromShareableHandle_(
      &import_handle, &sharedHandle, CU_MEM_HANDLE_TYPE_FABRIC);
  if (status != CUDA_SUCCESS) {
    LOG(INFO)
        << "status " << status
        << " Could not import FABRIC handle, falling back to fd handle exchange\n";
    driver_api->cuMemRelease_(handle);
    return false;
  }
  driver_api->cuMemRelease_(import_handle);
  driver_api->cuMemRelease_(handle);
  LOG(INFO) << "using fabric to exchange memory handles\n";
  return true;
}
#endif
} // namespace

bool get_fabric_access(c10::DeviceIndex dev) {
#if !defined USE_ROCM && defined CUDA_VERSION && CUDA_VERSION >= 12040 && defined PYTORCH_C10_DRIVER_API_SUPPORTED
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);

  TORCH_CHECK(dev >= 0 || dev < num_devices_, dev, " is not a device");
  auto& cache = fabricAccessEnabled_[dev];
  if (cache != -1) {
    return cache;
  }
  auto nvml_device = get_nvml_device(dev);
  if (nvml_device != nullptr) {
    nvmlGpuFabricInfoV_t fabricInfo;
    fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    fabricInfo.version = nvmlGpuFabricInfo_v2;
    if (DriverAPI::get()->nvmlDeviceGetGpuFabricInfoV_ == nullptr) {
      return false;
    }
    TORCH_CHECK(
        NVML_SUCCESS ==
        DriverAPI::get()->nvmlDeviceGetGpuFabricInfoV_(
            nvml_device, &fabricInfo));
    auto state = fabricInfo.state != NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    if (state) {
      // now perform the full cycle of allocating - exporting - importing memory
      state = isFabricSupported();
    }
    cache = state ? 1 : 0;
    return cache;
  } else {
    return false;
  }
#else
  return false;
#endif
}

} // namespace at::cuda

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `bool`, `detail`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cuda/PeerToPeerAccess.h`
- `ATen/cuda/CUDAContext.h`
- `c10/cuda/CUDACachingAllocator.h`
- `c10/cuda/CUDAGuard.h`
- `c10/cuda/driver_api.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen/cuda`):

- [`CublasHandlePool.cpp_docs.md`](./CublasHandlePool.cpp_docs.md)
- [`llvm_basic.cpp_docs.md`](./llvm_basic.cpp_docs.md)
- [`CUDABlas.h_docs.md`](./CUDABlas.h_docs.md)
- [`jiterator.cu_docs.md`](./jiterator.cu_docs.md)
- [`CUDAGraph.h_docs.md`](./CUDAGraph.h_docs.md)
- [`llvm_jit_strings.h_docs.md`](./llvm_jit_strings.h_docs.md)
- [`llvm_complex.cpp_docs.md`](./llvm_complex.cpp_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md`](./CUDAGeneratorImpl.cpp_docs.md)
- [`cub_definitions.cuh_docs.md`](./cub_definitions.cuh_docs.md)
- [`jiterator_impl.h_docs.md`](./jiterator_impl.h_docs.md)


## Cross-References

- **File Documentation**: `PeerToPeerAccess.cpp_docs.md`
- **Keyword Index**: `PeerToPeerAccess.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/aten/src/ATen/cuda`):

- [`PhiloxCudaState.h_docs.md_docs.md`](./PhiloxCudaState.h_docs.md_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md_docs.md`](./CUDAGeneratorImpl.cpp_docs.md_docs.md)
- [`Exceptions.cpp_docs.md_docs.md`](./Exceptions.cpp_docs.md_docs.md)
- [`CUDAGeneratorImpl.cpp_kw.md_docs.md`](./CUDAGeneratorImpl.cpp_kw.md_docs.md)
- [`Sleep.h_docs.md_docs.md`](./Sleep.h_docs.md_docs.md)
- [`cub-RadixSortPairs-int64-2.cu_kw.md_docs.md`](./cub-RadixSortPairs-int64-2.cu_kw.md_docs.md)
- [`CUDASparseDescriptors.h_kw.md_docs.md`](./CUDASparseDescriptors.h_kw.md_docs.md)
- [`jiterator_impl.h_docs.md_docs.md`](./jiterator_impl.h_docs.md_docs.md)
- [`CUDAContext.h_docs.md_docs.md`](./CUDAContext.h_docs.md_docs.md)
- [`cub-RadixSortPairs-int64-4.cu_docs.md_docs.md`](./cub-RadixSortPairs-int64-4.cu_docs.md_docs.md)


## Cross-References

- **File Documentation**: `PeerToPeerAccess.cpp_docs.md_docs.md`
- **Keyword Index**: `PeerToPeerAccess.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
