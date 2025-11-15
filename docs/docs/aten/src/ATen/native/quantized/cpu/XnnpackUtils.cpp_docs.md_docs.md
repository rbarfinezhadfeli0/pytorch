# Documentation: `docs/aten/src/ATen/native/quantized/cpu/XnnpackUtils.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/XnnpackUtils.cpp_docs.md`
- **Size**: 5,343 bytes (5.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/XnnpackUtils.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/XnnpackUtils.cpp`
- **Size**: 2,819 bytes (2.75 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_XNNPACK

#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
#include <c10/util/irange.h>

namespace at::native::xnnp_utils {

std::vector<size_t> get_mem_format_aware_shape(const at::Tensor& in) {
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

template <typename PT>
void q8_copy_int8_weight_and_add_offset(const at::Tensor& in, at::Tensor& out) {
  using T = typename PT::underlying;
  static constexpr auto offset = std::is_same_v<T, uint8_t> ? 128 : 0;
  TORCH_CHECK(
      in.scalar_type() == c10::kQInt8,
      "q8_copy_int8_weight_and_add_offset: Expected input weight data type ",
      toString(c10::kQInt8),
      " but got ",
      toString(in.scalar_type()))
  const int8_t* in_ptr =
      reinterpret_cast<const int8_t*>(in.data_ptr<c10::qint8>());
  T* out_ptr = reinterpret_cast<T*>(out.data_ptr<PT>());

  for (const auto i : c10::irange(in.numel())) {
    out_ptr[i] = static_cast<T>(static_cast<int32_t>(in_ptr[i]) + offset);
  }
}

template void q8_copy_int8_weight_and_add_offset<c10::quint8>(
    const at::Tensor& in,
    at::Tensor& out);
template void q8_copy_int8_weight_and_add_offset<c10::qint8>(
    const at::Tensor& in,
    at::Tensor& out);

/*
 * Stolen from fbgemm_utils::ConvertConvWeightsToChannelLastTensor to avoid
 * dependence on USE_FBGEMM. Reorder weights to the format xnnpack expects.
 * TODO: add a 3d variant.
 */
template <>
Tensor convert_conv_weights_to_channel_last_tensor<2>(
    const at::Tensor& src,
    int groups,
    bool transpose) {
  return transpose ?
                   // 2D conv transpose weight transform
                   // IC OC/G KH KW -> G OC/G KH KW IC/G
      [&]() {
        auto ic_g_oc_g_hw_tensors = src.chunk(groups);
        for (auto& tensor : ic_g_oc_g_hw_tensors) {
          tensor = tensor.unsqueeze(0);
        }
        auto fused_tensor = at::cat(ic_g_oc_g_hw_tensors);
        set_quantizer_(fused_tensor, src.quantizer());
        return fused_tensor.permute({0, 2, 3, 4, 1})
            .contiguous(c10::MemoryFormat::Contiguous);
      }()
                   // 2d conv weight transform
                   : src.contiguous(c10::MemoryFormat::ChannelsLast);
}
} // namespace at::native::xnnp_utils

#endif // USE_XNNPACK

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/quantized/Quantizer.h`
- `ATen/native/quantized/cpu/XnnpackUtils.h`
- `c10/util/irange.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu`):

- [`ACLUtils.cpp_docs.md`](./ACLUtils.cpp_docs.md)
- [`LinearUnpackImpl.cpp_docs.md`](./LinearUnpackImpl.cpp_docs.md)
- [`UpSampleNearest3d.cpp_docs.md`](./UpSampleNearest3d.cpp_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`QnnpackUtils.h_docs.md`](./QnnpackUtils.h_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md`](./qembeddingbag_unpack.cpp_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`TensorOperators.cpp_docs.md`](./TensorOperators.cpp_docs.md)
- [`XnnpackUtils.h_docs.md`](./XnnpackUtils.h_docs.md)
- [`qconv_dynamic.cpp_docs.md`](./qconv_dynamic.cpp_docs.md)


## Cross-References

- **File Documentation**: `XnnpackUtils.cpp_docs.md`
- **Keyword Index**: `XnnpackUtils.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu`):

- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`init_qnnpack.cpp_docs.md_docs.md`](./init_qnnpack.cpp_docs.md_docs.md)
- [`qelu.cpp_kw.md_docs.md`](./qelu.cpp_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)
- [`qclamp.cpp_docs.md_docs.md`](./qclamp.cpp_docs.md_docs.md)
- [`qembeddingbag_prepack.h_docs.md_docs.md`](./qembeddingbag_prepack.h_docs.md_docs.md)
- [`qdropout.cpp_docs.md_docs.md`](./qdropout.cpp_docs.md_docs.md)
- [`qelu.cpp_docs.md_docs.md`](./qelu.cpp_docs.md_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md_docs.md`](./qembeddingbag_unpack.cpp_docs.md_docs.md)
- [`LinearUnpackImpl.cpp_kw.md_docs.md`](./LinearUnpackImpl.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `XnnpackUtils.cpp_docs.md_docs.md`
- **Keyword Index**: `XnnpackUtils.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
