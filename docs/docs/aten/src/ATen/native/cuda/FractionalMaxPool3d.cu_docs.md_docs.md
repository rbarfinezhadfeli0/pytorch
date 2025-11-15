# Documentation: `docs/aten/src/ATen/native/cuda/FractionalMaxPool3d.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/FractionalMaxPool3d.cu_docs.md`
- **Size**: 14,604 bytes (14.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/FractionalMaxPool3d.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/FractionalMaxPool3d.cu`
- **Size**: 11,415 bytes (11.15 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/native/FractionalMaxPooling.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/fractional_max_pool3d_backward_native.h>
#include <ATen/ops/fractional_max_pool3d_native.h>
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace at::native {

using namespace at::cuda::detail;

namespace {

template <typename scalar_t, typename accscalar_t>
__device__ inline int64_t get_intervals(
  accscalar_t sample,
  int64_t index,
  int64_t inputSize,
  int64_t outputSize,
  int64_t poolSize) {
    accscalar_t alpha = static_cast<accscalar_t>(inputSize - poolSize) /
      static_cast<accscalar_t>(outputSize - 1);
    if (index == outputSize - 1) {
      return inputSize - poolSize;
    } else {
      return static_cast<int64_t>((index + sample) * alpha) - \
        static_cast<int64_t>(sample * alpha);
    }
  }

template <typename scalar_t>
__global__ void fractional_max_pool3d_out_frame(
  PackedTensorAccessor64<const scalar_t, 5> input,
  PackedTensorAccessor64<scalar_t, 5> output,
  PackedTensorAccessor64<int64_t, 5> indices,
  PackedTensorAccessor64<const scalar_t, 3> samples,
  int64_t poolSizeT, int64_t poolSizeH, int64_t poolSizeW) {
    using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
    // Output (t, h, w) point that this thread is responsible for
    int64_t ourOutputPoint = threadIdx.x + ((int64_t) blockIdx.x) * blockDim.x;
    int64_t plane = blockIdx.y;
    int64_t batch = blockIdx.z;
    // Each thread generates a specific output point
    if (ourOutputPoint < output.size(2) * output.size(3) *
      output.size(4)){
      int64_t outputT = ourOutputPoint / (output.size(3) *
                    output.size(4));
      int64_t outputH = (ourOutputPoint / output.size(4)) %
                    output.size(3);
      int64_t outputW = ourOutputPoint % output.size(4);

      int64_t poolT = get_intervals<scalar_t,accscalar_t>(
        static_cast<accscalar_t>(samples[batch][plane][0]),
        outputT, input.size(2), output.size(2), poolSizeT);
      int64_t poolH = get_intervals<scalar_t, accscalar_t>(
        static_cast<accscalar_t>(samples[batch][plane][1]),
        outputH, input.size(3), output.size(3), poolSizeH);
      int64_t poolW = get_intervals<scalar_t, accscalar_t>(
        static_cast<accscalar_t>(samples[batch][plane][2]),
        outputW, input.size(4), output.size(4), poolSizeW);

      scalar_t maxVal = at::numeric_limits<scalar_t>::lower_bound();
      int64_t maxIndex = poolT * input.size(3) * input.size(4) + poolH * input.size(4) + poolW;

      for(int64_t t = poolT; t < poolT + poolSizeT; ++ t) {
        for (int64_t h = poolH; h < poolH + poolSizeH; ++h) {
          if(poolSizeW < 2 || poolSizeW > 7) {
            for (int64_t w = poolW; w < poolW + poolSizeW; ++w) {
              scalar_t val = input[batch][plane][t][h][w];
              // for consistency with THNN, favor the first max
              if (val > maxVal || at::_isnan(val)) {
                maxIndex = t * input.size(3) *
                  input.size(4) + h * input.size(4) + w;
                maxVal = val;
              }
            }
          } else {
            for (int64_t i = 0; i < poolSizeW; ++i) {
              int64_t w = i + poolW;
              scalar_t val = input[batch][plane][t][h][w];
              // for consistency with THNN, favor the first max
              if (val > maxVal || at::_isnan(val)) {
                maxIndex = t * input.size(3) * input.size(4) +
                  h * input.size(4) + w;
                maxVal = val;
              }
            }
          }
        }
      }

      indices[batch][plane][outputT][outputH][outputW] = maxIndex;
      output[batch][plane][outputT][outputH][outputW] = maxVal;
    }
  }

template <typename scalar_t>
__global__ void fractional_max_pool3d_backward_out_frame(
  PackedTensorAccessor64<scalar_t, 5> gradInput,
  PackedTensorAccessor64<const scalar_t, 5> gradOutput,
  PackedTensorAccessor64<const int64_t, 5> indices) {
  // Output (h, w) point that this thread is responsible for
  int64_t ourOutputPoint = threadIdx.x + ((int64_t) blockIdx.x) * blockDim.x;
  int64_t plane = blockIdx.y;
  int64_t batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < gradOutput.size(2) *
    gradOutput.size(3) * gradOutput.size(4)) {
    int64_t outputW = ourOutputPoint % gradOutput.size(4);
    int64_t outputH = (ourOutputPoint / gradOutput.size(4)) %
                      gradOutput.size(3);
    int64_t outputT = ourOutputPoint / (gradOutput.size(3) *
                      gradOutput.size(4));

    int64_t index = indices[batch][plane][outputT][outputH][outputW];
    CUDA_KERNEL_ASSERT(index >= 0);
    int64_t inputW = index % gradInput.size(4);
    int64_t inputH = (index / gradInput.size(4)) %
      gradInput.size(3);
    int64_t inputT = index / (gradInput.size(3) *
      gradInput.size(4));
    CUDA_KERNEL_ASSERT(inputT < gradInput.size(2));

    gpuAtomicAddNoReturn(
      &gradInput[batch][plane][inputT][inputH][inputW],
      gradOutput[batch][plane][outputT][outputH][outputW]
      );
    }
  }

void fractional_max_pool3d_backward_out_cuda_template(
  Tensor& gradInput,
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef output_size,
  const Tensor& indices) {
    int64_t dimt = 1;
    int64_t dimh = 2;
    int64_t dimw = 3;

    int64_t outputT = output_size[0];
    int64_t outputH = output_size[1];
    int64_t outputW = output_size[2];

    int64_t ndims = input.ndimension();
    if (ndims == 5) {
      dimt++;
      dimh++;
      dimw++;
    }

    /* sizes */
    int64_t inputT = input.size(dimt);
    int64_t inputH = input.size(dimh);
    int64_t inputW = input.size(dimw);

    TORCH_CHECK(
      outputT == gradOutput.size(dimt),
      "fractional_max_pool3d_backward_out_cuda_template(): ",
      "gradOutput time unexpected"
    );
    TORCH_CHECK(
      outputH == gradOutput.size(dimh),
      "fractional_max_pool3d_backward_out_cuda_template(): ",
      "gradOutput height unexpected"
    );
    TORCH_CHECK(
      outputW == gradOutput.size(dimw),
      "fractional_max_pool3d_backward_out_cuda_template(): ",
      "gradOutput width unexpected"
    );

    /* resize */
    gradInput.resize_as_(input);
    gradInput.zero_();

    auto gradInput_ = gradInput;
    auto gradOutput_ = gradOutput;
    auto indices_ = indices;

    if(ndims == 4) {
      gradInput_ = gradInput_.reshape({1, gradInput.size(0), inputT,
                                       inputH, inputW});
      gradOutput_ = gradOutput_.reshape({1, gradOutput.size(0), outputT,
                                         outputH, outputW});
      indices_ = indices_.reshape({1, indices.size(0), outputT, outputH,
                                   outputW});
    }

    if (gradInput.numel() == 0) {
      return;
    }

    /* backprop */
    // block is limited to 4 warps
    // grid handles overflow per each plane
    int64_t outputPlaneSize = gradOutput_.size(2) *
      gradOutput_.size(3) * gradOutput_.size(4);
    dim3 grid(
      (outputPlaneSize + 127) / 128, // ceil(outputPlaneSize / 128)
      gradInput_.size(1),
      gradInput_.size(0));
    dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gradOutput.scalar_type(),
      "fractional_max_pool3d_backward_out_frame",
      [&] {
        fractional_max_pool3d_backward_out_frame<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          gradInput_.packed_accessor64<scalar_t, 5>(),
          gradOutput_.packed_accessor64<const scalar_t, 5>(),
          indices_.packed_accessor64<const int64_t, 5>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    );
  }

}// namespace

TORCH_IMPL_FUNC(fractional_max_pool3d_out_cuda) (
  const Tensor& input,
  int64_t poolSizeT,
  int64_t poolSizeH,
  int64_t poolSizeW,
  int64_t outputT,
  int64_t outputH,
  int64_t outputW,
  const Tensor& randomSamples,
  int64_t numBatch,
  int64_t numPlanes,
  int64_t inputT,
  int64_t inputH,
  int64_t inputW,
  const Tensor& output,
  const Tensor& indices) {
  fractional_max_pool_check_shape</*ndim*/ 3>(input, randomSamples);

  auto output_ = output;
  auto indices_ = indices;
  auto input_ = input;

  int64_t ndims = input_.ndimension();
  if(ndims == 4) {
    output_ = output_.reshape({1, numPlanes, outputT, outputH, outputW});
    indices_ = indices_.reshape({1, numPlanes, outputT, outputH, outputW});
    input_ = input_.reshape({1, numPlanes, inputT, inputH, inputW});
  }
  if (output_.numel() == 0) {
    return;
  }

  // block is limited to 4 warps
  // grid handles overflow per each plane
  int64_t outputPlaneSize = output_.size(2) *
    output_.size(3) * output_.size(4);
  dim3 grid(
    (outputPlaneSize + 127) / 128, // ceil(outputPlaneSize / 128)
    input_.size(1),
    input_.size(0));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    input.scalar_type(),
    "fractional_max_pool3d_out_frame",
    [&]{
      fractional_max_pool3d_out_frame<scalar_t>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_.packed_accessor64<const scalar_t, 5>(),
        output_.packed_accessor64<scalar_t, 5>(),
        indices_.packed_accessor64<int64_t, 5>(),
        randomSamples.packed_accessor64<const scalar_t, 3>(),
        poolSizeT, poolSizeH, poolSizeW
      );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  );
}

Tensor& fractional_max_pool3d_backward_out_cuda(const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef /*pool_size*/,
  IntArrayRef output_size,
  const at::Tensor& indices,
  at::Tensor& gradInput) {
    // See Note [Writing Nondeterministic Operations]
    // Nondeterministic because of atomicAdd usage
    globalContext().alertNotDeterministic("fractional_max_pool3d_backward_out_cuda");
    fractional_max_pool3d_backward_out_cuda_template(
      gradInput,
      gradOutput_,
      input,
      output_size,
      indices
    );
    return gradInput;
  }

Tensor fractional_max_pool3d_backward_cuda(
  const at::Tensor& gradOutput,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices) {
    // See Note [Writing Nondeterministic Operations]
    // Nondeterministic because of atomicAdd usage
    globalContext().alertNotDeterministic("fractional_max_pool3d_backward_cuda");
    Tensor gradInput = at::empty({0}, input.options());
    fractional_max_pool3d_backward_out_cuda_template(
      gradInput,
      gradOutput,
      input,
      output_size,
      indices
    );
    return gradInput;
 }

}// namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `TORCH_IMPL_FUNC`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/cuda/Atomic.cuh`
- `ATen/cuda/CUDAContext.h`
- `ATen/cuda/NumericLimits.cuh`
- `ATen/cuda/detail/IndexUtils.cuh`
- `ATen/cuda/detail/TensorInfo.cuh`
- `ATen/cuda/detail/KernelUtils.h`
- `ATen/NumericUtils.h`
- `ATen/TensorUtils.h`
- `ATen/Utils.h`
- `ATen/native/FractionalMaxPooling.h`
- `c10/macros/Macros.h`
- `c10/util/Exception.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty.h`
- `ATen/ops/fractional_max_pool3d_backward_native.h`
- `ATen/ops/fractional_max_pool3d_native.h`
- `algorithm`
- `cfloat`
- `cmath`


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

Files in the same folder (`aten/src/ATen/native/cuda`):

- [`LogcumsumexpKernel.cu_docs.md`](./LogcumsumexpKernel.cu_docs.md)
- [`WeightNorm.cu_docs.md`](./WeightNorm.cu_docs.md)
- [`SparseBinaryOpIntersectionKernel.cu_docs.md`](./SparseBinaryOpIntersectionKernel.cu_docs.md)
- [`jit_utils.cpp_docs.md`](./jit_utils.cpp_docs.md)
- [`ReduceNormKernel.cu_docs.md`](./ReduceNormKernel.cu_docs.md)
- [`BinaryMiscOpsKernels.cu_docs.md`](./BinaryMiscOpsKernels.cu_docs.md)
- [`RowwiseScaledMM.h_docs.md`](./RowwiseScaledMM.h_docs.md)
- [`fused_adamw_amsgrad_impl.cuh_docs.md`](./fused_adamw_amsgrad_impl.cuh_docs.md)
- [`Col2Im.cu_docs.md`](./Col2Im.cu_docs.md)
- [`DistributionRandomKernel.cu_docs.md`](./DistributionRandomKernel.cu_docs.md)


## Cross-References

- **File Documentation**: `FractionalMaxPool3d.cu_docs.md`
- **Keyword Index**: `FractionalMaxPool3d.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/cuda`):

- [`DeviceSqrt.cuh_kw.md_docs.md`](./DeviceSqrt.cuh_kw.md_docs.md)
- [`UnaryGeometricAsinKernel.cu_kw.md_docs.md`](./UnaryGeometricAsinKernel.cu_kw.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`fused_adamw_impl.cu_docs.md_docs.md`](./fused_adamw_impl.cu_docs.md_docs.md)
- [`TensorTopK.h_kw.md_docs.md`](./TensorTopK.h_kw.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`FusedSgdKernel.cu_docs.md_docs.md`](./FusedSgdKernel.cu_docs.md_docs.md)
- [`Distributions.cu_kw.md_docs.md`](./Distributions.cu_kw.md_docs.md)
- [`block_reduce.cuh_docs.md_docs.md`](./block_reduce.cuh_docs.md_docs.md)
- [`fused_adagrad_impl.cuh_kw.md_docs.md`](./fused_adagrad_impl.cuh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `FractionalMaxPool3d.cu_docs.md_docs.md`
- **Keyword Index**: `FractionalMaxPool3d.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
