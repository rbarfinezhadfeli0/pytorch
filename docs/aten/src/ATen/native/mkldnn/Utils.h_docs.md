# Documentation: `aten/src/ATen/native/mkldnn/Utils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/Utils.h`
- **Size**: 4,521 bytes (4.42 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Config.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/strides.h>
#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif
#include <vector>

#if AT_MKLDNN_ENABLED()
#include <ideep/tensor.hpp>
#endif // AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,
    IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps, bool inplace = false);

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef dilation,
    bool ceil_mode);

void check_mkldnn_binary_fusion_inputs(
    const Tensor& input,
    const Tensor& other,
    const Tensor& weight,
    const Tensor& bias);

inline std::vector<int64_t> padding_r(
    IntArrayRef padding, IntArrayRef output_padding)
{
  // ConvTranspose padding adjustment
  //
  // PyTorch uses padding/output_padding:
  //   osize = (isize - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
  //
  // MKLDNN uses padding_l/padding_r:
  //   osize = (isize - 1) * stride - padding_l - padding_r + dilation * (kernel_size - 1) + 1
  //
  // So: padding_l = padding, padding_r = padding - output_padding
  //
  auto dim = padding.size();
  std::vector<int64_t> pad_r(dim);
  for (const auto d : c10::irange(dim)) {
    pad_r[d] = padding[d] - output_padding[d];
  }
  return pad_r;
}

// Make sure input has default contiguous strides if it's contiguous tensors for better performance.
// For example, for tensor of size = [1, 1280], stride = [0, 1], we'll convert it to size = [1, 1280], stride = [1280, 1]
// before calling oneDNN for better performance.
inline Tensor may_convert_to_default_contiguous_strides(const Tensor& input) {
  auto input_size = input.sizes().vec();
  auto input_stride = input.strides().vec();
  auto input_default_contiguous_strides = c10::contiguous_strides(input_size);
  if (input.is_contiguous() && input_stride != c10::IntArrayRef(input_default_contiguous_strides)) {
     return input.as_strided(input_size, input_default_contiguous_strides);
  }
  return input;
}

#if AT_MKLDNN_ENABLED()

using AttrFunction = std::function<ideep::attr_t(
    torch::List<std::optional<at::Scalar>>,
    std::optional<std::string_view>)>;

const std::map<std::string_view, AttrFunction>& fusion_unary_attr_map();

const std::map<std::string_view, ideep::algorithm>& fusion_unary_alg_map();

const std::map<std::string_view, ideep::algorithm>& fusion_binary_alg_map();

#endif // AT_MKLDNN_ENABLED()
}

#if defined(__aarch64__)
inline bool mkldnn_bf16_device_check_arm() {
  return cpuinfo_initialize() && cpuinfo_has_arm_bf16();
}

inline bool is_arm_neoverse() {
  return (cpuinfo_initialize() && cpuinfo_get_uarchs_count() == 1 &&
          (cpuinfo_get_uarch(0)->uarch == cpuinfo_uarch_neoverse_v1 ||
           cpuinfo_get_uarch(0)->uarch == cpuinfo_uarch_neoverse_v2 ||
           cpuinfo_get_uarch(0)->uarch == cpuinfo_uarch_neoverse_n1 ||
           cpuinfo_get_uarch(0)->uarch == cpuinfo_uarch_neoverse_n2));
}
#else
constexpr bool mkldnn_bf16_device_check_arm() {
  return false;
}

constexpr bool is_arm_neoverse() {
  return false;
}
#endif

#if AT_MKLDNN_ENABLED()
inline bool mkldnn_bf16_device_check() {
#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
  // Use ideep to check bf16 on X64 as cpuinfo has no avx_ne_convert check.
  return ideep::has_bf16_type_support();
#else
  return mkldnn_bf16_device_check_arm();
#endif
}

inline bool mkldnn_fp16_device_check() {
#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
  return ideep::has_fp16_type_support();
#else
  return false;
#endif
}

#else
inline bool mkldnn_bf16_device_check() {
  return false;
}
inline bool mkldnn_fp16_device_check() {
  return false;
}
#endif

inline void mkldnn_check_low_precision(ScalarType input_t, std::string name) {
  if (input_t == ScalarType::BFloat16) {
    TORCH_CHECK(
        mkldnn_bf16_device_check(),
        name,
        ": bf16 path needs the cpu support avx_ne_convert or avx512bw, avx512vl and avx512dq");
  } else if (input_t == ScalarType::Half) {
    TORCH_CHECK(
        mkldnn_fp16_device_check(),
        name,
        ": fp16 path needs the cpu support avx_ne_convert or avx512_fp16");
  }
}

}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `native`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/core/List.h`
- `ATen/core/Tensor.h`
- `c10/util/ArrayRef.h`
- `c10/util/strides.h`
- `cpuinfo.h`
- `vector`
- `ideep/tensor.hpp`


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

Files in the same folder (`aten/src/ATen/native/mkldnn`):

- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`Gelu.cpp_docs.md`](./Gelu.cpp_docs.md)
- [`Conv.h_docs.md`](./Conv.h_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`Matmul.cpp_docs.md`](./Matmul.cpp_docs.md)
- [`TensorShape.cpp_docs.md`](./TensorShape.cpp_docs.md)
- [`RNN.cpp_docs.md`](./RNN.cpp_docs.md)
- [`RegisterMkldnnOpContextClass.cpp_docs.md`](./RegisterMkldnnOpContextClass.cpp_docs.md)
- [`Copy.cpp_docs.md`](./Copy.cpp_docs.md)


## Cross-References

- **File Documentation**: `Utils.h_docs.md`
- **Keyword Index**: `Utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
