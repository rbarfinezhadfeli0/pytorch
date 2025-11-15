# Documentation: `docs/aten/src/ATen/native/quantized/cpu/QuantUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/QuantUtils.h_docs.md`
- **Size**: 10,978 bytes (10.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/QuantUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/QuantUtils.h`
- **Size**: 8,247 bytes (8.05 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/TensorOperators.h>
#include <c10/util/irange.h>
#include <algorithm>
#include <cmath>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/quantize_per_channel_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace quant_utils {
namespace {
  float RawUint16ToFp16(unsigned short value) {
    // Convert raw 16 bits half precision floating point number
    // to single precision floating point number.
    const unsigned short sign_bits = value >> 15;
    const unsigned short exponent_bits = value >> 10 & 0x1f;
    const unsigned short significand_bits = value & 0x3ff;

    const float sign = sign_bits ? -1 : 1;
    const float significand =
        1 + significand_bits * 0.0009765625f; // 0.0009765625f = 0x1p-10 = 2^-10;
    const float exponent = exponent_bits - 0xf;

    return sign * std::ldexp(significand, exponent);
}

template <typename T>
bool CheckAndSaturate(T max_val, T* element) {
  if (*element > max_val) {
    *element = max_val;
    return true;
  }
  if (*element < -max_val) {
    *element = -max_val;
    return true;
  }
  return false;
}
}
using namespace std;
// A structure to hold quantization parameters 'scale' and 'zero_point'.
// The meaning of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
struct TensorQuantizationParams {
  double scale;
  std::int32_t zero_point;
  int precision;
};

// Use fp16_min as the small scale cutoff because we don't want to use scales in
// fp16 subnormal range. This is to be consistent with Glow and FakeLowP
// implementation for NNPI.
constexpr float SMALL_SCALE_THRESHOLD = 6.1e-5f;

// Following implementation should be identical to fbgemm::ChooseQuantizationParams
inline TensorQuantizationParams ChooseQuantizationParams(
    float min,
    float max,
    int32_t qmin,
    int32_t qmax,
    bool preserve_sparsity = false,
    bool force_scale_power_of_two = false,
    bool reduce_range = false) {
  TORCH_CHECK(
      min <= max,
      "In ChooseQuantizationParams, min should be less than or equal to max");

  if (reduce_range) {
    qmin = qmin/2;
    qmax = qmax/2;
  }
  if (min < 0 && max > 0 && preserve_sparsity) {
    int symmetric_qmin = -((qmax - qmin) / 2 + 1);
    int symmetric_qmax = (qmax - qmin) / 2;
    double max_scale =
        std::max(fabs(min / symmetric_qmin), fabs(max / symmetric_qmax));
    min = max_scale * symmetric_qmin;
    max = max_scale * symmetric_qmax;
  }

  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  TORCH_CHECK(
      qmin < qmax,
      "In ChooseQuantizationParams, qmin should be less than qmax");

  // Use double precision for intermediate computation but use single precision
  // in final number to reflect the actual number used during quantization.
  double scale = (static_cast<double>(max) - min) / (qmax - qmin);
  // If scale is 0 or too small so its reciprocal is infinity, we arbitrary
  // adjust the scale to 0.1 . We want to avoid scale's reciprocal being
  // infinity because some of fbgemm code pre-computes scale's reciprocal to do
  // multiplication instead of division in the time critical part of code.
  if (float(scale) == 0.0f || std::isinf(1.0f / float(scale))) {
    scale = 0.1;
  }
  TORCH_CHECK(scale > 0, "quantization scale should be > 0");

  if (force_scale_power_of_two) {
    if (scale < 1) {
      scale = 1.0 / (1 << static_cast<int>(floor(log(1.0 / scale) / log(2))));
    } else {
      scale = 1 << static_cast<int>(ceil(log(scale) / log(2)));
    }
  }

  // Cut off small scale
  if (scale < SMALL_SCALE_THRESHOLD) {
    float org_scale = scale;
    scale = SMALL_SCALE_THRESHOLD;
    // Adjust the min and max based on the new scale
    if (min == 0.0f) {
      max = SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else if (max == 0.0f) {
      min = -SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else {
      float amplifier = SMALL_SCALE_THRESHOLD / org_scale;
      min *= amplifier;
      max *= amplifier;
    }
  }

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  double zero_point_from_min = qmin - min / scale;
  double zero_point_from_max = qmax - max / scale;
  double zero_point_from_min_error =
      std::abs(qmin) - std::abs(min / scale);
  double zero_point_from_max_error =
      std::abs(qmax) - std::abs(max / scale);
  double initial_zero_point =
      zero_point_from_min_error < zero_point_from_max_error
      ? zero_point_from_min
      : zero_point_from_max;

  // for symmetric quantization (preserve_sparsity == true), we force zero_point
  // to be a middle value between qmin and qmax.
  // If either min or max is 0, then we just use 0 as zero_point.
  if (min < 0 && max > 0 && preserve_sparsity) {
    initial_zero_point = static_cast<double>(qmin + qmax) / 2;
  }

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with zero
  // padding).
  int32_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = nearbyint(initial_zero_point);
  }

  TensorQuantizationParams result;
  result.scale = scale;
  result.zero_point = nudged_zero_point;
  return result;
}

// This function helps to convert the Conv1D dimensions usable by the Conv2d op.
constexpr int64_t kConv1dSqueezeDim = 0;
[[maybe_unused]] static torch::List<int64_t> MakeArgForConv1d(
    const torch::List<int64_t>& arg,
    int64_t base_value) {
  TORCH_CHECK(!arg.empty(), "Argument must have elements.");
  torch::List<int64_t> result({arg.get(0), base_value});
  if (arg.size() == 1) {
    result[1] = arg.get(0);
  } else {
    result[1] = arg.get(1);
  }
  result[kConv1dSqueezeDim] = base_value;
  return result;
}

// The range for using FP16 quantization of weights requires that the elements
// should be in the range of [5.96e-8, 65504]. If it is out of range, then the
// number will be saturated to max or min representable values by FP16.
inline void HandleWeightsSaturation(int64_t N, float* weight) {
  const float kFp16Max = RawUint16ToFp16(0x7BFF);
  bool found_out_of_range = false;
  for (const auto i : c10::irange(N)) {
    bool saturate = CheckAndSaturate<float>(kFp16Max, weight + i);
    if (saturate) {
      found_out_of_range = true;
    }
  }
  if (found_out_of_range) {
    TORCH_WARN("FOUND weight out of range ");
  }
}

// Util function for quantizing bias.
inline at::Tensor QuantizeBias(
    bool is_per_channel,
    const at::Tensor& bias,
    const at::Tensor& weight_contig,
    double input_scale) {
  at::Tensor qbias;
  if (is_per_channel) {
    auto bias_quant_scales =
        weight_contig.q_per_channel_scales() * input_scale;
    auto bias_zp = at::zeros(bias_quant_scales.sizes(), c10::kInt);
    qbias = at::native::quantize_per_channel(
        bias, bias_quant_scales, bias_zp, 0, c10::kQInt32);
  } else {
    qbias = at::native::quantize_per_tensor(
        bias, weight_contig.q_scale() * input_scale, 0, c10::kQInt32);
  }
  return qbias;
}

} // namespace quant_utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `std`, `quant_utils`

**Classes/Structs**: `TensorQuantizationParams`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/core/List.h`
- `ATen/TensorOperators.h`
- `c10/util/irange.h`
- `algorithm`
- `cmath`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/quantize_per_tensor_native.h`
- `ATen/ops/quantize_per_channel_native.h`
- `ATen/ops/zeros.h`


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

- **File Documentation**: `QuantUtils.h_docs.md`
- **Keyword Index**: `QuantUtils.h_kw.md`
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

- **File Documentation**: `QuantUtils.h_docs.md_docs.md`
- **Keyword Index**: `QuantUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
