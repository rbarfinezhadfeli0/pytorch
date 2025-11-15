# Documentation: `docs/aten/src/ATen/native/mkldnn/xpu/detail/oneDNNContext.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/mkldnn/xpu/detail/oneDNNContext.h_docs.md`
- **Size**: 5,354 bytes (5.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/mkldnn/xpu/detail/oneDNNContext.h`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/xpu/detail/oneDNNContext.h`
- **Size**: 2,712 bytes (2.65 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Config.h>

#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <vector>

namespace at::native::onednn {

TORCH_XPU_API dnnl::memory make_onednn_memory(
    dnnl::memory::desc md,
    dnnl::engine& engine,
    void* ptr);

// Keep non-static and non-inline
bool set_onednn_verbose(int level);

// GpuEngineManager singleton
struct TORCH_XPU_API GpuEngineManager {
  static GpuEngineManager& Instance(); // Singleton

  dnnl::engine& get_engine(
      DeviceIndex device_index = c10::xpu::current_device()) {
    c10::xpu::check_device_index(device_index);
    return *engine_pool[device_index];
  }

  dnnl::engine& get_engine(const Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == kXPU);
    return get_engine(device.index());
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;
  GpuEngineManager(GpuEngineManager&&) = default;
  GpuEngineManager& operator=(GpuEngineManager&&) = default;

 protected:
  GpuEngineManager();
  ~GpuEngineManager() = default;

 private:
  std::vector<std::shared_ptr<dnnl::engine>> engine_pool;
};

// GpuStreamManager singleton
struct TORCH_XPU_API GpuStreamManager {
  static GpuStreamManager& Instance(); // Singleton

  dnnl::stream& get_stream(
      DeviceIndex device_index = c10::xpu::current_device()) {
    auto stream = c10::xpu::getCurrentXPUStream(device_index);
    auto priority = stream.priority();
    if (stream_pool[device_index][priority].find(stream) ==
        stream_pool[device_index][priority].end()) {
      stream_pool[device_index][priority][stream] =
          std::make_shared<dnnl::stream>(dnnl::sycl_interop::make_stream(
              GpuEngineManager::Instance().get_engine(device_index),
              stream.queue()));
    }
    return *stream_pool[device_index][priority][stream];
  }

  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;
  GpuStreamManager(GpuStreamManager&&) = default;
  GpuStreamManager& operator=(GpuStreamManager&&) = default;

 protected:
  GpuStreamManager() {
    c10::DeviceIndex device_count = c10::xpu::device_count_ensure_non_zero();
    stream_pool.resize(device_count);
  }
  ~GpuStreamManager() = default;

 private:
  using stream_hash_map =
      ska::flat_hash_map<c10::xpu::XPUStream, std::shared_ptr<dnnl::stream>>;
  std::vector<
      std::array<stream_hash_map, c10::xpu::max_compile_time_stream_priorities>>
      stream_pool;
};

} // namespace at::native::onednn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TORCH_XPU_API`, `TORCH_XPU_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn/xpu/detail`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `c10/core/Device.h`
- `c10/util/flat_hash_map.h`
- `c10/xpu/XPUFunctions.h`
- `c10/xpu/XPUStream.h`
- `oneapi/dnnl/dnnl.hpp`
- `oneapi/dnnl/dnnl_sycl.hpp`
- `vector`


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

Files in the same folder (`aten/src/ATen/native/mkldnn/xpu/detail`):

- [`Attention.cpp_docs.md`](./Attention.cpp_docs.md)
- [`LRUCache.h_docs.md`](./LRUCache.h_docs.md)
- [`QConv.cpp_docs.md`](./QConv.cpp_docs.md)
- [`WoQMatmul.cpp_docs.md`](./WoQMatmul.cpp_docs.md)
- [`Matmul.cpp_docs.md`](./Matmul.cpp_docs.md)
- [`Deconv.cpp_docs.md`](./Deconv.cpp_docs.md)
- [`oneDNNContext.cpp_docs.md`](./oneDNNContext.cpp_docs.md)
- [`QMatmul.cpp_docs.md`](./QMatmul.cpp_docs.md)
- [`oneDNN.h_docs.md`](./oneDNN.h_docs.md)


## Cross-References

- **File Documentation**: `oneDNNContext.h_docs.md`
- **Keyword Index**: `oneDNNContext.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/mkldnn/xpu/detail`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/mkldnn/xpu/detail`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/mkldnn/xpu/detail`):

- [`Attention.cpp_kw.md_docs.md`](./Attention.cpp_kw.md_docs.md)
- [`QConv.cpp_docs.md_docs.md`](./QConv.cpp_docs.md_docs.md)
- [`Utils.cpp_docs.md_docs.md`](./Utils.cpp_docs.md_docs.md)
- [`QMatmul.cpp_docs.md_docs.md`](./QMatmul.cpp_docs.md_docs.md)
- [`oneDNN.h_kw.md_docs.md`](./oneDNN.h_kw.md_docs.md)
- [`DnnlExt.h_kw.md_docs.md`](./DnnlExt.h_kw.md_docs.md)
- [`Matmul.cpp_docs.md_docs.md`](./Matmul.cpp_docs.md_docs.md)
- [`Conv.cpp_docs.md_docs.md`](./Conv.cpp_docs.md_docs.md)
- [`LRUCache.h_kw.md_docs.md`](./LRUCache.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `oneDNNContext.h_docs.md_docs.md`
- **Keyword Index**: `oneDNNContext.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
