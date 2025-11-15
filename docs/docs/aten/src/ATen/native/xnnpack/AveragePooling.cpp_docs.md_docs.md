# Documentation: `docs/aten/src/ATen/native/xnnpack/AveragePooling.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/xnnpack/AveragePooling.cpp_docs.md`
- **Size**: 6,895 bytes (6.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/xnnpack/AveragePooling.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/xnnpack/AveragePooling.cpp`
- **Size**: 4,360 bytes (4.26 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_XNNPACK

#include <ATen/native/utils/Factory.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/xnnpack/Pooling.h>

namespace at::native::xnnpack {

static inline std::vector<size_t> get_mem_format_aware_shape(const at::Tensor& in) {
  const auto mem_format = in.suggest_memory_format();
  const auto& sizes = in.sizes();
  std::vector<size_t> ret(sizes.begin(), sizes.end());
  if (mem_format == c10::MemoryFormat::ChannelsLast) {
    // NCHW -> NHWC
    // 0123 -> 0231
    ret[1] = sizes[2]; /* H */
    ret[2] = sizes[3]; /* W */
    ret[3] = sizes[1]; /* C */
  } else if (mem_format == c10::MemoryFormat::ChannelsLast3d) {
    // NCDHW -> NDHWC
    // 01234 -> 02341
    ret[1] = sizes[2]; /* D */
    ret[2] = sizes[3]; /* H */
    ret[3] = sizes[4]; /* W */
    ret[4] = sizes[1]; /* C */
  }
  return ret;
}

bool use_global_average_pool(const Tensor& input) {
  return xnnpack::available() && (1 <= input.ndimension()) &&
      (input.device().is_cpu()) && (kFloat == input.scalar_type()) &&
      !input.requires_grad() && true;
}

Tensor global_average_pool(const Tensor& input) {
  using namespace internal;

  const Tensor input_padded_contig_nhwc =
      mobile::allocate_padded_contiguous_if_needed(
          input, MemoryFormat::ChannelsLast);

  Tensor output = mobile::empty_with_tail_padding(
      {
          input_padded_contig_nhwc.size(Layout::Activation4D::batch),
          input_padded_contig_nhwc.size(Layout::Activation4D::channels),
          1,
          1,
      },
      input_padded_contig_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      input_padded_contig_nhwc.opt_names());

  // Create XNNPACK Subgraph
  xnn_subgraph_t subgraph_ptr = nullptr;
  xnn_status status = xnn_create_subgraph(
    /*external_value_ids=*/2,
    /*flags=*/0,
    &subgraph_ptr);
  TORCH_CHECK(
      status == xnn_status_success,
      "xnn create subgraph failed(", status,")!");
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
      subgraph_ptr, &xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_VALUE_ID, output_id = XNN_INVALID_VALUE_ID;


  const auto& input_shape = get_mem_format_aware_shape(input_padded_contig_nhwc);
  status = xnn_define_tensor_value(
    subgraph_ptr,
    xnn_datatype_fp32,
    input_shape.size(),
    input_shape.data(),
    nullptr,
    0,
    XNN_VALUE_FLAG_EXTERNAL_INPUT,
    &input_id
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "defining xnn input failed(", status,")!");

  const auto& output_shape = get_mem_format_aware_shape(output);
  status = xnn_define_tensor_value(
    subgraph_ptr,
    xnn_datatype_fp32,
    output_shape.size(),
    output_shape.data(),
    nullptr,
    1,
    XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
    &output_id
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "defining xnn output failed(", status,")!");

  std::vector<size_t> reduce_dims{1, 2};
  status = xnn_define_static_reduce(
    subgraph_ptr,
    xnn_reduce_mean,
    reduce_dims.size(),
    reduce_dims.data(),
    input_id,
    output_id,
    0
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "defining xnn static reduce failed(", status,")!");

  // create runtime
  xnn_runtime_t runtime_ptr = nullptr;
  status = xnn_create_runtime_v2(subgraph_ptr, caffe2::pthreadpool_(), 0, &runtime_ptr);
  TORCH_CHECK(
      status == xnn_status_success,
      "xnn create runtime failed(", status,")!");
  TORCH_CHECK(
      runtime_ptr != nullptr,
      "xnn create runtime failed because runtime_ptr is null");
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime_ptr, &xnn_delete_runtime);

  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input_padded_contig_nhwc.data_ptr<float>()},
    xnn_external_value{output_id, output.data_ptr<float>()}};

  status = xnn_setup_runtime(
    runtime_ptr,
    external.size(),
    external.data());
  TORCH_CHECK(
      status == xnn_status_success,
      "xnn setup runtime failed(", status,")!");
  status = xnn_invoke_runtime(runtime_ptr);
  TORCH_CHECK(
      status == xnn_status_success,
      "xnn invoke runtime failed(", status,")!");

  return output.to(input.suggest_memory_format());
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `internal`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/utils/Factory.h`
- `ATen/native/xnnpack/Common.h`
- `ATen/native/xnnpack/Engine.h`
- `ATen/native/xnnpack/Pooling.h`


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

Files in the same folder (`aten/src/ATen/native/xnnpack`):

- [`Engine.h_docs.md`](./Engine.h_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`ChannelShuffle.cpp_docs.md`](./ChannelShuffle.cpp_docs.md)
- [`Convolution.h_docs.md`](./Convolution.h_docs.md)
- [`RegisterOpContextClass.cpp_docs.md`](./RegisterOpContextClass.cpp_docs.md)
- [`Common.h_docs.md`](./Common.h_docs.md)
- [`Convolution.cpp_docs.md`](./Convolution.cpp_docs.md)
- [`Activation.cpp_docs.md`](./Activation.cpp_docs.md)
- [`Linear.h_docs.md`](./Linear.h_docs.md)
- [`Shim.cpp_docs.md`](./Shim.cpp_docs.md)


## Cross-References

- **File Documentation**: `AveragePooling.cpp_docs.md`
- **Keyword Index**: `AveragePooling.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/xnnpack`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/xnnpack`):

- [`MaxPooling.cpp_docs.md_docs.md`](./MaxPooling.cpp_docs.md_docs.md)
- [`Convolution.cpp_docs.md_docs.md`](./Convolution.cpp_docs.md_docs.md)
- [`Common.h_kw.md_docs.md`](./Common.h_kw.md_docs.md)
- [`Pooling.h_docs.md_docs.md`](./Pooling.h_docs.md_docs.md)
- [`RegisterOpContextClass.cpp_kw.md_docs.md`](./RegisterOpContextClass.cpp_kw.md_docs.md)
- [`AveragePooling.cpp_kw.md_docs.md`](./AveragePooling.cpp_kw.md_docs.md)
- [`OpContext.cpp_kw.md_docs.md`](./OpContext.cpp_kw.md_docs.md)
- [`ChannelShuffle.cpp_docs.md_docs.md`](./ChannelShuffle.cpp_docs.md_docs.md)
- [`MaxPooling.cpp_kw.md_docs.md`](./MaxPooling.cpp_kw.md_docs.md)
- [`Common.h_docs.md_docs.md`](./Common.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `AveragePooling.cpp_docs.md_docs.md`
- **Keyword Index**: `AveragePooling.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
