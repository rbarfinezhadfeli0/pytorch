# Documentation: `docs/torch/csrc/jit/codegen/onednn/LlgaTensorImpl.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/codegen/onednn/LlgaTensorImpl.cpp_docs.md`
- **Size**: 7,900 bytes (7.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/codegen/onednn/LlgaTensorImpl.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/onednn/LlgaTensorImpl.cpp`
- **Size**: 5,391 bytes (5.26 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <c10/core/CPUAllocator.h>
#include <torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h>

namespace torch::jit::fuser::onednn {

// Non-default dnnl::graph::allocator needs an allocator.
// We would let it use c10::GetCPUAllocator's allocator,
// which uses posix_memalign with 64 byte alignment-size.
static void* pytorch_default_allocator(size_t size, size_t alignment) {
  static c10::Allocator* c10_allocator = c10::GetCPUAllocator();
  return c10_allocator->raw_allocate(size);
}

// Non-default dnnl::graph::allocator needs a deallocator.
// We would let it use c10::GetCPUAllocator's deallocator.
static void pytorch_default_deallocator(void* buf) {
  static c10::Allocator* c10_allocator = c10::GetCPUAllocator();
  c10_allocator->raw_deallocate(buf);
}

dnnl::engine& Engine::getEngine() {
  // Even if the default PyTorch CPU allocator would change, we'd still use the
  // stale value. In practice, we don't expect users to change the CPU allocator
  // dynamically anyway, as users preload jemalloc/tcmalloc at runtime, if they
  // would like to. But this behavior might need to be changed, as some models
  // work better with tcmalloc, while others work better with jemalloc, so
  // switching the CPU allocator at runtime can be useful.
  static dnnl::graph::allocator alloc{
      pytorch_default_allocator, pytorch_default_deallocator};
  static dnnl::engine cpu_engine = dnnl::graph::make_engine_with_allocator(
      dnnl::engine::kind::cpu, /* index = */ 0, alloc);
  return cpu_engine;
}

dnnl::stream& Stream::getStream() {
  static dnnl::stream cpu_stream{Engine::getEngine()};
  return cpu_stream;
}

LlgaTensorImpl::LlgaTensorImpl(
    at::Storage&& storage,
    const caffe2::TypeMeta& data_type,
    const LlgaTensorDesc& desc)
    : at::TensorImpl(
          std::move(storage),
          c10::DispatchKeySet(c10::DispatchKey::MkldnnCPU),
          data_type),
      desc_(desc) {
  set_sizes_and_strides(desc.sizes(), desc.strides());
  refresh_numel();
}

at::Tensor LlgaTensorImpl::llga_to_aten_tensor(LlgaTensorImpl* llgaImpl) {
  auto aten_tensor = at::detail::make_tensor<TensorImpl>(
      std::move(llgaImpl->storage_),
      c10::DispatchKeySet(c10::DispatchKey::CPU),
      llgaImpl->data_type_);
  auto impl = aten_tensor.unsafeGetTensorImpl();
  impl->set_storage_offset(llgaImpl->storage_offset_);
  impl->set_sizes_and_strides(llgaImpl->sizes(), llgaImpl->strides());
  return aten_tensor;
}

at::Tensor empty_llga(
    const LlgaTensorDesc& desc,
    const c10::TensorOptions& options) {
  auto nbytes = desc.storage_size();

  auto allocator = at::GetCPUAllocator();
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
      allocator->allocate(nbytes),
      allocator,
      /*resizable=*/false);

  return at::detail::make_tensor<LlgaTensorImpl>(
      std::move(storage_impl), options.dtype(), desc);
}

static const LlgaTensorDesc& get_llga_desc(const at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(
      tensor.is_mkldnn(), "get_llga_desc expects Mkldnn tensor input");
  return static_cast<LlgaTensorImpl*>(tensor.unsafeGetTensorImpl())->desc();
}

dnnl::graph::tensor llga_from_aten_tensor(const at::Tensor& tensor) {
  return {
      get_llga_desc(tensor).logical_tensor(),
      torch::jit::fuser::onednn::Engine::getEngine(),
      tensor.data_ptr()};
}

using data_type = dnnl::graph::logical_tensor::data_type;

data_type LlgaTensorDesc::getLlgaDataType(at::ScalarType dt) const {
  switch (dt) {
    case at::ScalarType::Float:
      return data_type::f32;
    case at::ScalarType::BFloat16:
      return data_type::bf16;
    case at::kInt:
      return data_type::s32;
    case at::ScalarType::QInt8:
      return data_type::s8;
    case at::ScalarType::QUInt8:
      return data_type::u8;
    default:
      // If a dtype is unsupported, oneDNN Graph will make that op a wildcard in
      // the graph construction stage. Then when we would execute oneDNN Graph
      // kernels pertaining to oneDNN Graph partitions, such an op would not be
      // inside a oneDNN Graph partition, so we would not encounter inputs with
      // unsupported dtypes at the time of executing compiled partitions.
      return data_type::undef;
  }
}

LlgaTensorDesc LlgaTensorDesc::supplementTensorInfo(const at::Tensor& t) const {
  if (t.is_mkldnn()) {
    // if input tensor is of mkldnn, it's originated from an upstream
    // LLGA partition which carries opaque layout info
    return get_llga_desc(t).tid(tid_);
  } else {
    // if input tensor is not an mkldnn tensor, use default layout
    auto sizes = t.sizes().vec();
    auto strides = t.strides().vec();
    auto dtype = getLlgaDataType(t.scalar_type());
    return {tid_, sizes, strides, dtype, property_type_};
  }
}

at::ScalarType LlgaTensorDesc::aten_scalar_type() const {
  switch (dtype_) {
    case data_type::f32:
      return at::ScalarType::Float;
    case data_type::bf16:
      return at::ScalarType::BFloat16;
    case data_type::s32:
      return at::kInt;
    case data_type::s8:
      return at::ScalarType::QInt8;
    case data_type::u8:
      return at::ScalarType::QUInt8;
    default:
      TORCH_CHECK(false, "Invalid data type ", static_cast<size_t>(dtype_));
  }
}

} // namespace torch::jit::fuser::onednn

#endif // AT_MKLDNN_ENABLED()

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/onednn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `c10/core/CPUAllocator.h`
- `torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/jit/codegen/onednn`):

- [`graph_rewriter.cpp_docs.md`](./graph_rewriter.cpp_docs.md)
- [`guard_shape.cpp_docs.md`](./guard_shape.cpp_docs.md)
- [`prepare_binary.h_docs.md`](./prepare_binary.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`graph_fuser.h_docs.md`](./graph_fuser.h_docs.md)
- [`kernel.h_docs.md`](./kernel.h_docs.md)
- [`decompose_silu.cpp_docs.md`](./decompose_silu.cpp_docs.md)
- [`prepare_binary.cpp_docs.md`](./prepare_binary.cpp_docs.md)
- [`graph_helper.cpp_docs.md`](./graph_helper.cpp_docs.md)
- [`register_interface.cpp_docs.md`](./register_interface.cpp_docs.md)


## Cross-References

- **File Documentation**: `LlgaTensorImpl.cpp_docs.md`
- **Keyword Index**: `LlgaTensorImpl.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/codegen/onednn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/codegen/onednn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/jit/codegen/onednn`):

- [`graph_rewriter.cpp_docs.md_docs.md`](./graph_rewriter.cpp_docs.md_docs.md)
- [`kernel.cpp_docs.md_docs.md`](./kernel.cpp_docs.md_docs.md)
- [`decompose_silu.cpp_kw.md_docs.md`](./decompose_silu.cpp_kw.md_docs.md)
- [`defer_size_check.h_kw.md_docs.md`](./defer_size_check.h_kw.md_docs.md)
- [`graph_fuser.h_kw.md_docs.md`](./graph_fuser.h_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`graph_fuser.h_docs.md_docs.md`](./graph_fuser.h_docs.md_docs.md)
- [`interface.h_docs.md_docs.md`](./interface.h_docs.md_docs.md)
- [`layout_propagation.h_kw.md_docs.md`](./layout_propagation.h_kw.md_docs.md)
- [`graph_helper.cpp_kw.md_docs.md`](./graph_helper.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `LlgaTensorImpl.cpp_docs.md_docs.md`
- **Keyword Index**: `LlgaTensorImpl.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
