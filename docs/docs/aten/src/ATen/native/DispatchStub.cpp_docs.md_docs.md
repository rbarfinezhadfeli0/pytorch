# Documentation: `docs/aten/src/ATen/native/DispatchStub.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/DispatchStub.cpp_docs.md`
- **Size**: 14,573 bytes (14.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/DispatchStub.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/DispatchStub.cpp`
- **Size**: 11,893 bytes (11.61 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/DispatchStub.h>

#include <c10/core/DeviceType.h>
#include <c10/util/Array.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>

#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif
#include <algorithm>
#include <cstdlib>
#include <cstring>

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
#include <sys/auxv.h>
#endif

namespace at::native {

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
static inline bool cpu_has_vxe()
{
  return (getauxval(AT_HWCAP) & HWCAP_S390_VXE);
}
#endif

static CPUCapability compute_cpu_capability() {
  const auto envar = c10::utils::get_env("ATEN_CPU_CAPABILITY");
  if (envar.has_value()) {
#if defined(HAVE_VSX_CPU_DEFINITION)
    if (envar == "vsx") {
      return CPUCapability::VSX;
    }
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
    if (envar == "zvector") {
      return CPUCapability::ZVECTOR;
    }
#elif defined(HAVE_SVE_CPU_DEFINITION)
    int sve_vl = cpuinfo_get_max_arm_sve_length(); //Returns maximum SVE VL supported by your HW.
#ifdef HAVE_SVE256_CPU_DEFINITION
    if (envar == "sve256") {
      if (sve_vl == 256) {
#ifdef HAVE_ARM_BF16_CPU_DEFINITION
        if (cpuinfo_has_arm_bf16()) {
          return CPUCapability::SVE256;
        }
#endif
      }
      TORCH_WARN("SVE256 capability not available on hardware. Falling back to DEFAULT");
      return CPUCapability::DEFAULT;
    }
#endif
#else
#ifdef HAVE_AVX512_CPU_DEFINITION
    if (envar == "avx512") {
      return CPUCapability::AVX512;
    }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    if (envar == "avx2") {
      return CPUCapability::AVX2;
    }
#endif
#endif
    if (envar == "default") {
      return CPUCapability::DEFAULT;
    }
    TORCH_WARN("ignoring invalid value for ATEN_CPU_CAPABILITY: ", envar.value());
  }

#if !defined(__powerpc__) && !defined(__s390x__) && !defined(HAVE_SVE_CPU_DEFINITION)
  if (cpuinfo_initialize()) {
#if defined(HAVE_AVX512_CPU_DEFINITION)
    // GCC supports some AVX512 intrinsics such as _mm512_set_epi16 only in
    // versions 9 & beyond. So, we want to ensure that only releases built with
    // supported compilers on supported hardware return CPU Capability AVX512,
    // if it's supported on the hardware PyTorch is running on.
    if (cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512bw() &&  \
        cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_fma3()) {
      return CPUCapability::AVX512;
    }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    if (cpuinfo_has_x86_avx2() && cpuinfo_has_x86_fma3()) {
      return CPUCapability::AVX2;
    }
#endif
  }
#endif

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  // vxe is needed for fp32 vector instructions
  if (cpu_has_vxe()) {
    return CPUCapability::ZVECTOR;
  }
#endif

#if defined(__linux__) && defined(HAVE_SVE_CPU_DEFINITION)
  if (cpuinfo_initialize() && cpuinfo_has_arm_sve()) {
    int sve_vl = cpuinfo_get_max_arm_sve_length(); //Returns maximum SVE VL supported by your HW.
    if (sve_vl <= 0) {
      // SVE is not supported on this system.
      // Return the default CPU capability.
      return CPUCapability::DEFAULT;
    }
    #ifdef HAVE_SVE256_CPU_DEFINITION
        if (sve_vl == 256) { // Check for SVE256
        #ifdef HAVE_ARM_BF16_CPU_DEFINITION
          if (cpuinfo_has_arm_bf16())
            return CPUCapability::SVE256;
        #endif
        }
    #endif
    // Return the default CPU capability.
    return CPUCapability::DEFAULT;
  }
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  return CPUCapability::VSX;
#else
  return CPUCapability::DEFAULT;
#endif
}

CPUCapability get_cpu_capability() {
  static CPUCapability capability = compute_cpu_capability();
  return capability;
}

DispatchResult DispatchStubImpl::try_get_call_ptr(
  const DeviceType device_type
  , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  , void *ZVECTOR
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  , void *SVE256
#endif
) {
  constexpr auto supported_devices = c10::array_of<c10::DeviceType>(
        c10::DeviceType::CPU,
        c10::DeviceType::CUDA,
        c10::DeviceType::HIP,
        c10::DeviceType::MPS,
        c10::DeviceType::MTIA,
        c10::DeviceType::XPU,
        c10::DeviceType::HPU,
        c10::DeviceType::PrivateUse1
    );
    // Check if the device type is supported.
    if (std::find(supported_devices.begin(), supported_devices.end(), device_type) == supported_devices.end()) {
        return ErrorType::DeviceNotSupported;
    }
  switch (device_type) {
    case DeviceType::CPU: {
      // Use memory_order_relaxed here since even if two threads race,
      // they will still compute the same value for cpu_dispatch_ptr.
      auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
      if (!fptr) {
        auto result = try_choose_cpu_impl(
          DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
          , AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
          , AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
          , VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
          , ZVECTOR
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
          , SVE256
#endif
        );
        if (!std::holds_alternative<ErrorType>(result)) {
          cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
        }
      return result;
      }
      return DispatchResult(fptr);
    }

    case DeviceType::CUDA:
      return cuda_dispatch_ptr != nullptr ? DispatchResult(cuda_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    case DeviceType::HIP:
      return hip_dispatch_ptr != nullptr ? DispatchResult(hip_dispatch_ptr) : ErrorType::MissingDeviceKernel;

#if defined(USE_MPS)
    case DeviceType::MPS:
      return mps_dispatch_ptr != nullptr ? DispatchResult(mps_dispatch_ptr) : ErrorType::MissingDeviceKernel;
#endif
    case DeviceType::MTIA:
      return mtia_dispatch_ptr != nullptr ? DispatchResult(mtia_dispatch_ptr) : ErrorType::MissingDeviceKernel;

#if defined(USE_XPU)
    case DeviceType::XPU:
      return xpu_dispatch_ptr != nullptr ? DispatchResult(xpu_dispatch_ptr) : ErrorType::MissingDeviceKernel;
#endif

    case DeviceType::HPU:
      return hpu_dispatch_ptr != nullptr ? DispatchResult(hpu_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    case DeviceType::PrivateUse1:
      return privateuse1_dispatch_ptr != nullptr ? DispatchResult(privateuse1_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    default:
      TORCH_INTERNAL_ASSERT(false, "An unexpected device type was provided ", device_type);
      return ErrorType::DeviceNotSupported;
  }
}

void* DispatchStubImpl::get_call_ptr(
  const DeviceType device_type
  , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  , void *ZVECTOR
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  , void *SVE256
#endif
) {

  auto result = try_get_call_ptr(
      device_type,
      DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      ,
      AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      ,
      AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      ,
      VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      ,
      ZVECTOR
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
      ,
      SVE256
#endif
  );
  if (std::holds_alternative<ErrorType>(result)) {
    auto error = std::get<ErrorType>(result);
    switch (error) {
      case ErrorType::MissingDeviceKernel:
        TORCH_INTERNAL_ASSERT(
            false, "DispatchStub: missing kernel for ", device_type);
        return nullptr;
      case ErrorType::DeviceNotSupported:
        TORCH_CHECK(false, "DispatchStub: unsupported device type", device_type);
    }
  }

  void* fptr = std::get<void*>(result);
  return fptr;
}

DispatchResult DispatchStubImpl::try_choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
    , void *ZVECTOR
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
    , void *SVE256
#endif
  ){

  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (C10_UNLIKELY(!AVX512)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      return AVX2 != nullptr ? DispatchResult(AVX2) : ErrorType::MissingDeviceKernel;
    } else {
      return DispatchResult(AVX512);
    }
  }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX2)) {
    return AVX2 != nullptr ? DispatchResult(AVX2) : ErrorType::MissingDeviceKernel;
  }
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::VSX)) {
    return VSX != nullptr ? DispatchResult(VSX) : ErrorType::MissingDeviceKernel;
  }
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::ZVECTOR)) {
    return ZVECTOR != nullptr ? DispatchResult(ZVECTOR) : ErrorType::MissingDeviceKernel;
  }
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::SVE256)) {
    if (C10_UNLIKELY(!SVE256)) {
      // dispatch to DEFAULT, since the SVE kernel is missing
      return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
    } else {
      return DispatchResult(SVE256);
    }
  }
#endif
  return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
}

void* DispatchStubImpl::choose_cpu_impl(
  void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  , void *ZVECTOR
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  , void *SVE256
#endif
) {
  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (C10_UNLIKELY(!AVX512)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
      return AVX2;
    } else {
      return AVX512;
    }
  }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX2)) {
    TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
    return AVX2;
  }
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::VSX)) {
    TORCH_INTERNAL_ASSERT(VSX, "DispatchStub: missing VSX kernel");
    return VSX;
  }
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::ZVECTOR)) {
    TORCH_INTERNAL_ASSERT(ZVECTOR, "DispatchStub: missing ZVECTOR kernel");
    return ZVECTOR;
  }
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::SVE256)) {
    if (C10_UNLIKELY(!SVE256)) {
      // dispatch to DEFAULT, since the SVE kernel is missing
      TORCH_INTERNAL_ASSERT(DEFAULT, "DispatchStub: missing default kernel");
      return DEFAULT;
    } else {
      return SVE256;
    }
  }
#endif
  TORCH_INTERNAL_ASSERT(DEFAULT, "DispatchStub: missing default kernel");
  return DEFAULT;
}

}  // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 35 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/DispatchStub.h`
- `c10/core/DeviceType.h`
- `c10/util/Array.h`
- `c10/util/Exception.h`
- `c10/util/env.h`
- `cpuinfo.h`
- `algorithm`
- `cstdlib`
- `cstring`
- `sys/auxv.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `DispatchStub.cpp_docs.md`
- **Keyword Index**: `DispatchStub.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `DispatchStub.cpp_docs.md_docs.md`
- **Keyword Index**: `DispatchStub.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
