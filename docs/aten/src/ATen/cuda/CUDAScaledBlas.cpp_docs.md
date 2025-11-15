# Documentation: `aten/src/ATen/cuda/CUDAScaledBlas.cpp`

## File Metadata

- **Path**: `aten/src/ATen/cuda/CUDAScaledBlas.cpp`
- **Size**: 9,696 bytes (9.47 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <cstdint>
#include <c10/util/typeid.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/cuda/tunable/TunableGemm.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/cuda/RowwiseScaledMM.h>
#include <ATen/native/cuda/ScaledGroupMM.h>
#include <ATen/native/cuda/GroupMM.h>
#include <ATen/ceil_div.h>

#ifdef USE_FBGEMM_GENAI
#include <fbgemm_gpu/torch_ops.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/_scaled_mm_native.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/vdot_native.h>
#endif

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace at::cuda::scaled {

/**
 * Both inputs must be fp8,
 * Each needs a single scale, {Tensorwise (float)}
 */
bool check_tensorwise_recipe(c10::ScalarType type_a,
                             std::vector<ScalingType>& recipe_a,
                             ArrayRef<Tensor>& scales_a,
                             c10::ScalarType type_b,
                             std::vector<ScalingType>& recipe_b,
                             ArrayRef<Tensor>& scales_b) {
  // both types must be fp8
  if (!isFloat8Type(type_a) || !isFloat8Type(type_b)) {
    return false;
  }

  // 1 scale each, {Tensorwise, float}
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }
  // Need {Blockwise_1x32, e8m0} for A & B
  if (recipe_a[0] != ScalingType::TensorWise) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float) return false;
  if (recipe_b[0] != ScalingType::TensorWise) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float) return false;

  return true;
}

/**
 * Both inputs must be fp8,
 * Each needs scales, {Rowwise (float)}
 */
bool check_rowwise_recipe(c10::ScalarType type_a,
                             std::vector<ScalingType>& recipe_a,
                             ArrayRef<Tensor>& scales_a,
                             c10::ScalarType type_b,
                             std::vector<ScalingType>& recipe_b,
                             ArrayRef<Tensor>& scales_b) {
  // both types must be fp8
  if (!isFloat8Type(type_a) || !isFloat8Type(type_b)) {
    return false;
  }

  // 1 scale each, {Tensorwise, float}
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // Need {RowWise, dp32} for A & B
  if (recipe_a[0] != ScalingType::RowWise) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float) return false;
  if (recipe_b[0] != ScalingType::RowWise) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float) return false;

  return true;
}


/**
 * Two-level scaling, canonical NVFP4
 * Both inputs must be fp4
 * A, B need 2 scales, {Blockwise_1x16 (e4m3), Tensorwise (fp32)}
 */
bool check_nvfp4_recipe(c10::ScalarType type_a,
                        std::vector<ScalingType>& recipe_a,
                        ArrayRef<Tensor>& scales_a,
                        c10::ScalarType type_b,
                        std::vector<ScalingType>& recipe_b,
                        ArrayRef<Tensor>& scales_b) {
  // both types must be fp4
  if (type_a != ScalarType::Float4_e2m1fn_x2 || type_b != ScalarType::Float4_e2m1fn_x2) {
    return false;
  }

  // 2 scales, 2 recipes for each input
  if (scales_a.size() != 2 || recipe_a.size() != 2 || scales_b.size() != 2 || recipe_b.size() != 2) {
    return false;
  }

  // Need {Blockwise_1x16, e4m3 for scale[0], Tensorwise, fp32 for scale[1]}
  if (recipe_a[0] != ScalingType::BlockWise1x16 || recipe_a[1] != ScalingType::TensorWise) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float8_e4m3fn || scales_a[1].scalar_type() != ScalarType::Float) return false;
  if (recipe_b[0] != ScalingType::BlockWise1x16 || recipe_b[1] != ScalingType::TensorWise) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float8_e4m3fn || scales_b[1].scalar_type() != ScalarType::Float) return false;

  return true;
}

/**
 * Single-level scaling, what PyT currently understands
 * Both inputs must be fp4
 * A, B need 1 scale, {Blockwise_1x16 (e4m3)}
 */
bool check_nvfp4_recipe_single_scale
                       (c10::ScalarType type_a,
                        std::vector<ScalingType>& recipe_a,
                        ArrayRef<Tensor>& scales_a,
                        c10::ScalarType type_b,
                        std::vector<ScalingType>& recipe_b,
                        ArrayRef<Tensor>& scales_b) {
  // both types must be fp4
  if (type_a != ScalarType::Float4_e2m1fn_x2 || type_b != ScalarType::Float4_e2m1fn_x2) {
    return false;
  }

  // 2 scales, 2 recipes for each input
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // Need {Blockwise_1x16, e4m3 for scale[0], Tensorwise, fp32 for scale[1]}
  if (recipe_a[0] != ScalingType::BlockWise1x16) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float8_e4m3fn) return false;
  if (recipe_b[0] != ScalingType::BlockWise1x16) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float8_e4m3fn) return false;

  return true;
}

/**
 * Both inputs must be fp8
 * A, B must only have 1 scale each, A: {Blockwise_1x128 (float), B: {Blockwise_128x128 (float)
 */
bool check_deepseek_recipe(ScalingType expected_recipe_a,
                           ScalingType expected_recipe_b,
                           c10::ScalarType type_a,
                           std::vector<ScalingType>& recipe_a,
                           ArrayRef<Tensor>& scales_a,
                           c10::ScalarType type_b,
                           std::vector<ScalingType>& recipe_b,
                           ArrayRef<Tensor>& scales_b) {
  // both types must be fp8
  if (type_a != ScalarType::Float8_e4m3fn || type_b != ScalarType::Float8_e4m3fn) {
    return false;
  }

  // 1 scales, 1 recipes for each input
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // Need {Blockwise_1x128, float} for A, {Blockwise_128x128, float} for B
  if (recipe_a[0] != expected_recipe_a) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float) return false;
  if (recipe_b[0] != expected_recipe_b) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float) return false;

  return true;
}

/**
 * Both inputs must be fp8
 * A, B must have 1 scale each, {Blockwise_1x32, e8m0}
 */
bool check_mxfp8_recipe(c10::ScalarType type_a,
                        std::vector<ScalingType>& recipe_a,
                        ArrayRef<Tensor>& scales_a,
                        c10::ScalarType type_b,
                        std::vector<ScalingType>& recipe_b,
                        ArrayRef<Tensor>& scales_b) {
  // both types must be fp8
  if (type_a != ScalarType::Float8_e4m3fn || type_b != ScalarType::Float8_e4m3fn) {
    return false;
  }

  // 1 scales, 1 recipes for each input
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // Need {Blockwise_1x32, e8m0} for A & B
  if (recipe_a[0] != ScalingType::BlockWise1x32) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float8_e8m0fnu) return false;
  if (recipe_b[0] != ScalingType::BlockWise1x32) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float8_e8m0fnu) return false;

  return true;
}

/**
 * Both inputs must be fp4
 * A, B must have 1 scale each, {Blockwise_1x32, e8m0}
 */
bool check_mxfp4_recipe(c10::ScalarType type_a,
                        std::vector<ScalingType>& recipe_a,
                        ArrayRef<Tensor>& scales_a,
                        c10::ScalarType type_b,
                        std::vector<ScalingType>& recipe_b,
                        ArrayRef<Tensor>& scales_b) {
  // both types must be fp4
  if (type_a != ScalarType::Float4_e2m1fn_x2 || type_b != ScalarType::Float4_e2m1fn_x2) {
    return false;
  }

  // 1 scales, 1 recipes for each input
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // Need {Blockwise_1x32, e8m0} for A & B
  if (recipe_a[0] != ScalingType::BlockWise1x32) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float8_e8m0fnu) return false;
  if (recipe_b[0] != ScalingType::BlockWise1x32) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float8_e8m0fnu) return false;

  return true;
}

} // namespace at::native::cuda::blas::scaled

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 24 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cstdint`
- `c10/util/typeid.h`
- `c10/util/Exception.h`
- `c10/util/SmallVector.h`
- `c10/core/Scalar.h`
- `c10/core/ScalarType.h`
- `c10/util/Exception.h`
- `ATen/core/Tensor.h`
- `ATen/core/NamedTensor.h`
- `ATen/Dispatch.h`
- `ATen/ExpandUtils.h`
- `ATen/OpMathType.h`
- `ATen/TensorUtils.h`
- `ATen/cuda/CUDABlas.h`
- `ATen/cuda/tunable/Tunable.h`
- `ATen/cuda/tunable/TunableGemm.h`
- `ATen/native/Resize.h`
- `c10/util/MaybeOwned.h`
- `ATen/native/GroupedMMUtils.h`
- `ATen/native/cuda/RowwiseScaledMM.h`
- `ATen/native/cuda/ScaledGroupMM.h`
- `ATen/native/cuda/GroupMM.h`
- `ATen/ceil_div.h`
- `fbgemm_gpu/torch_ops.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_addmm_activation_native.h`
- `ATen/ops/_efficientzerotensor.h`
- `ATen/ops/_scaled_mm_native.h`
- `ATen/ops/_unsafe_view_native.h`


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

- **File Documentation**: `CUDAScaledBlas.cpp_docs.md`
- **Keyword Index**: `CUDAScaledBlas.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
