# Documentation: `docs/aten/src/ATen/cuda/CUDAScaledBlas.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cuda/CUDAScaledBlas.h_docs.md`
- **Size**: 8,662 bytes (8.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cuda/CUDAScaledBlas.h`

## File Metadata

- **Path**: `aten/src/ATen/cuda/CUDAScaledBlas.h`
- **Size**: 5,366 bytes (5.24 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
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

static bool _scaled_mm_allowed_device(bool sm90_only=false, bool sm100_only=false) {
#ifdef USE_ROCM
    static const std::vector<std::string> archs = {
        "gfx942",
#if ROCM_VERSION >= 60300
        "gfx1200", "gfx1201",
#endif
#if ROCM_VERSION >= 60500
        "gfx950"
#endif
    };
    return at::detail::getCUDAHooks().isGPUArch(archs);
#else
    auto dprops = at::cuda::getCurrentDeviceProperties();

    if (sm90_only || sm100_only) {
      return (sm90_only && dprops->major == 9) || (sm100_only && dprops->major == 10);
    } else {
      return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
    }
#endif
}

#ifdef USE_ROCM
static bool _scaled_mm_is_fnuz() {
    return at::detail::getCUDAHooks().isGPUArch({"gfx942"});
}
#endif
/**
 * Track concrete implementations available
 */
enum class ScaledGemmImplementation {
  NONE = 0,
  TENSORWISE_TENSORWISE = 1,
  ROWWISE_ROWWISE = 2,
  BLOCK_128x128_1x128 = 3,
  BLOCK_1x128_128x128 = 4,
  BLOCK_1x128_1x128 = 5,
  MXFP8_MXFP8 = 6,
  NVFP4_NVFP4 = 7,
  NVFP4_NVFP4_SINGLE_SCALE = 8,
  MXFP4_MXFP4 = 9,
};

/**
 * Convert passed int (enum) from python back into a
 * strictly-typed enum
 */
template <class EnumType, class ArrayType>
std::vector<EnumType> convert_int_to_enum(ArrayType& v) {
  std::vector<EnumType> converted;
  converted.reserve(v.size());

  for (auto vi : v) {
    converted.push_back(static_cast<EnumType>(vi));
  }
  return converted;
}

bool check_tensorwise_recipe(c10::ScalarType,
                             std::vector<ScalingType>&,
                             ArrayRef<Tensor>&,
                             c10::ScalarType,
                             std::vector<ScalingType>&,
                             ArrayRef<Tensor>&);


bool check_rowwise_recipe(c10::ScalarType,
                             std::vector<ScalingType>&,
                             ArrayRef<Tensor>&,
                             c10::ScalarType,
                             std::vector<ScalingType>&,
                             ArrayRef<Tensor>&);

bool check_nvfp4_recipe(c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&,
                        c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&);

bool check_nvfp4_recipe_single_scale
                       (c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&,
                        c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&);

bool check_deepseek_recipe(ScalingType,
                           ScalingType,
                           c10::ScalarType,
                           std::vector<ScalingType>&,
                           ArrayRef<Tensor>&,
                           c10::ScalarType,
                           std::vector<ScalingType>&,
                           ArrayRef<Tensor>&);

bool check_mxfp8_recipe(c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&,
                        c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&);

bool check_mxfp4_recipe(c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&,
                        c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&);

} // namespace at::native::cuda::blas::scaled

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `ScaledGemmImplementation`, `EnumType`, `ArrayType`


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

- **File Documentation**: `CUDAScaledBlas.h_docs.md`
- **Keyword Index**: `CUDAScaledBlas.h_kw.md`
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

- **File Documentation**: `CUDAScaledBlas.h_docs.md_docs.md`
- **Keyword Index**: `CUDAScaledBlas.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
