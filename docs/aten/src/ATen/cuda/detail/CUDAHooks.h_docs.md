# Documentation: `aten/src/ATen/cuda/detail/CUDAHooks.h`

## File Metadata

- **Path**: `aten/src/ATen/cuda/detail/CUDAHooks.h`
- **Size**: 2,889 bytes (2.82 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/detail/CUDAHooksInterface.h>

#include <ATen/Generator.h>

// TODO: No need to have this whole header, we can just put it all in
// the cpp file

namespace at::cuda::detail {

// Set the callback to initialize Magma, which is set by
// torch_cuda_cu. This indirection is required so magma_init is called
// in the same library where Magma will be used.
TORCH_CUDA_CPP_API void set_magma_init_fn(void (*magma_init_fn)());


// The real implementation of CUDAHooksInterface
struct CUDAHooks : public at::CUDAHooksInterface {
  CUDAHooks(at::CUDAHooksArgs /*unused*/) {}
  void init() const override;
  Device getDeviceFromPtr(void* data) const override;
  bool isPinnedPtr(const void* data) const override;
  const Generator& getDefaultGenerator(
      DeviceIndex device_index = -1) const override;
  Generator getNewGenerator(
      DeviceIndex device_index = -1) const override;
  bool hasCUDA() const override;
  bool hasMAGMA() const override;
  bool hasCuDNN() const override;
  bool hasCuSOLVER() const override;
  bool hasCuBLASLt() const override;
  bool hasROCM() const override;
  bool hasCKSDPA() const override;
  bool hasCKGEMM() const override;
  const at::cuda::NVRTC& nvrtc() const override;
  DeviceIndex current_device() const override;
  bool isBuilt() const override {return true;}
  bool isAvailable() const override {return hasCUDA();}
  bool hasPrimaryContext(DeviceIndex device_index) const override;
  Allocator* getCUDADeviceAllocator() const override;
  Allocator* getPinnedMemoryAllocator() const override;
  bool compiledWithCuDNN() const override;
  bool compiledWithMIOpen() const override;
  bool supportsDilatedConvolutionWithCuDNN() const override;
  bool supportsDepthwiseConvolutionWithCuDNN() const override;
  bool supportsBFloat16ConvolutionWithCuDNNv8() const override;
  bool supportsBFloat16RNNWithCuDNN() const override;
  bool hasCUDART() const override;
  long versionCUDART() const override;
  long versionCuDNN() const override;
  long versionRuntimeCuDNN() const override;
  long versionCuDNNFrontend() const override;
  long versionMIOpen() const override;
  std::string showConfig() const override;
  double batchnormMinEpsilonCuDNN() const override;
  int64_t cuFFTGetPlanCacheMaxSize(DeviceIndex device_index) const override;
  void cuFFTSetPlanCacheMaxSize(DeviceIndex device_index, int64_t max_size) const override;
  int64_t cuFFTGetPlanCacheSize(DeviceIndex device_index) const override;
  void cuFFTClearPlanCache(DeviceIndex device_index) const override;
  int getNumGPUs() const override;
  DeviceIndex deviceCount() const override;
  DeviceIndex getCurrentDevice() const override;

#ifdef USE_ROCM
  bool isGPUArch(const std::vector<std::string>& archs, DeviceIndex device_index = -1) const override;
#endif
  void deviceSynchronize(DeviceIndex device_index) const override;
};

} // at::cuda::detail

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 41 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `CUDAHooks`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda/detail`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/detail/CUDAHooksInterface.h`
- `ATen/Generator.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`aten/src/ATen/cuda/detail`):

- [`BLASConstants.cu_docs.md`](./BLASConstants.cu_docs.md)
- [`BLASConstants.h_docs.md`](./BLASConstants.h_docs.md)
- [`KernelUtils.h_docs.md`](./KernelUtils.h_docs.md)
- [`OffsetCalculator.cuh_docs.md`](./OffsetCalculator.cuh_docs.md)
- [`IndexUtils.cu_docs.md`](./IndexUtils.cu_docs.md)
- [`LazyNVRTC.cpp_docs.md`](./LazyNVRTC.cpp_docs.md)
- [`LazyNVRTC.h_docs.md`](./LazyNVRTC.h_docs.md)
- [`UnpackRaw.cuh_docs.md`](./UnpackRaw.cuh_docs.md)
- [`DeviceThreadHandles.h_docs.md`](./DeviceThreadHandles.h_docs.md)


## Cross-References

- **File Documentation**: `CUDAHooks.h_docs.md`
- **Keyword Index**: `CUDAHooks.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
