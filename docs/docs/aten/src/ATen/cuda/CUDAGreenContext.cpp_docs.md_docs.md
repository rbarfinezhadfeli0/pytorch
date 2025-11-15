# Documentation: `docs/aten/src/ATen/cuda/CUDAGreenContext.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cuda/CUDAGreenContext.cpp_docs.md`
- **Size**: 8,988 bytes (8.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cuda/CUDAGreenContext.cpp`

## File Metadata

- **Path**: `aten/src/ATen/cuda/CUDAGreenContext.cpp`
- **Size**: 6,460 bytes (6.31 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/cuda/CUDAGreenContext.h>

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12030) && !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <stdexcept>
#include <vector>
#define HAS_CUDA_GREEN_CONTEXT() 1
#else
#define HAS_CUDA_GREEN_CONTEXT() 0
// Suppress unsued private field warnings as this class is not supposed to be called
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-private-field")
#endif

namespace at::cuda {

GreenContext::GreenContext(uint32_t device_id, uint32_t num_sms) {
#if HAS_CUDA_GREEN_CONTEXT()
  int driver_version;
  C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  TORCH_CHECK(
      driver_version >= 12080, "cuda driver too old to use green context!");
  CUcontext pctx = nullptr;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&pctx));
  if (C10_UNLIKELY(!pctx)) {
    TORCH_WARN(
        "Attempted to create a green context but"
        " there was no primary context! Creating a primary context...");

    cudaFree(0);
  }

   CUdevice device;
  device_id_ = device_id;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDeviceGet_(&device, device_id));

  // Get device resources
  CUdevResource device_resource;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(
      device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));

  // Split resources
  std::vector<CUdevResource> result(1);
  auto result_data = result.data();
  unsigned int nb_groups = 1;
  CUdevResource remaining;

  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDevSmResourceSplitByCount_(
          result_data,
          &nb_groups,
          &device_resource,
          &remaining,
          0, // default flags
          num_sms));

  TORCH_CHECK(nb_groups == 1, "Failed to create single resource group");

  // Generate resource descriptor
  CUdevResourceDesc desc;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDevResourceGenerateDesc_(
          &desc, result_data, 1));

  // Create green context
  // CU_GREEN_CTX_DEFAULT_STREAM is required per docs:
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGreenCtxCreate_(
      &green_ctx_, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));

  // Convert to regular context
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuCtxFromGreenCtx_(&context_, green_ctx_));
  TORCH_CHECK(context_, "Green ctx conversion to regular ctx failed!");
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }

  std::unique_ptr<GreenContext> GreenContext::create(
      uint32_t num_sms,
      std::optional<uint32_t> device_id) {
#if HAS_CUDA_GREEN_CONTEXT()
    if (!device_id.has_value()) {
      device_id = at::cuda::current_device();
    }
    return std::unique_ptr<GreenContext>(new GreenContext(device_id.value(), num_sms));
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }

  // Implement move operations
  GreenContext::GreenContext(GreenContext&& other) noexcept{
#if HAS_CUDA_GREEN_CONTEXT()
    device_id_ = std::exchange(other.device_id_, -1);
    green_ctx_ = std::exchange(other.green_ctx_, nullptr);
    context_ = std::exchange(other.context_, nullptr);
    parent_stream_ = std::exchange(other.parent_stream_, nullptr);
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }

  GreenContext& GreenContext::operator=(GreenContext&& other) noexcept{
#if HAS_CUDA_GREEN_CONTEXT()
    if (this != &other) {
      // Clean up current resources
      if (green_ctx_) {
        CUcontext current = nullptr;
        C10_CUDA_DRIVER_CHECK(
            c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&current));
        if (current == context_) {
          TORCH_CHECK(
              false,
              "attempting to overwrite current green ctx "
              "when it is active!");
        }
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGreenCtxDestroy_(green_ctx_));
      }

      // Take ownership of other's resources
      device_id_ = std::exchange(other.device_id_, -1);
      green_ctx_ = std::exchange(other.green_ctx_, nullptr);
      context_ = std::exchange(other.context_, nullptr);
      parent_stream_ = std::exchange(other.parent_stream_, nullptr);
    }
    return *this;
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }

  GreenContext::~GreenContext() noexcept{
#if HAS_CUDA_GREEN_CONTEXT()
    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuGreenCtxDestroy_(green_ctx_));
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }

  // Make this context current
  void GreenContext::setContext() {
#if HAS_CUDA_GREEN_CONTEXT()
    auto current_stream = c10::cuda::getCurrentCUDAStream();
    parent_stream_ = current_stream.stream();

    at::cuda::CUDAEvent ev;
    ev.record(current_stream);

    CUcontext current = nullptr;
    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&current));
    if (!current) {
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuCtxSetCurrent_(context_));
    } else {
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuCtxPushCurrent_(context_));
    }
    // currently hardcodes the new green context to use the default stream
    // TODO(eqy): consider creating a new stream if e.g., it allows interop
    // with CUDA Graph captures etc.
    auto default_stream = c10::cuda::getDefaultCUDAStream();
    ev.block(default_stream);
    c10::cuda::setCurrentCUDAStream(default_stream);
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }

  void GreenContext::popContext() {
#if HAS_CUDA_GREEN_CONTEXT()
    // see above note about stream being hardcoded to the default stream
    at::cuda::CUDAEvent ev;
    ev.record(c10::cuda::getCurrentCUDAStream());
    CUcontext popped;
    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuCtxPopCurrent_(&popped));
    TORCH_INTERNAL_ASSERT(
        popped == context_, "expected popped context to be the current ctx");
    ev.block(c10::cuda::getStreamFromExternal(parent_stream_, device_id_));
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }
} // namespace at::cuda

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 22 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `is`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cuda/CUDAGreenContext.h`
- `c10/cuda/driver_api.h`
- `stdexcept`
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

- **File Documentation**: `CUDAGreenContext.cpp_docs.md`
- **Keyword Index**: `CUDAGreenContext.cpp_kw.md`
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

- **File Documentation**: `CUDAGreenContext.cpp_docs.md_docs.md`
- **Keyword Index**: `CUDAGreenContext.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
