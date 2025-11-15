# Documentation: `docs/aten/src/ATen/native/cuda/RreluWithNoise.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/RreluWithNoise.cu_docs.md`
- **Size**: 8,529 bytes (8.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/RreluWithNoise.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/RreluWithNoise.cu`
- **Size**: 5,729 bytes (5.59 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/cuda/DistributionTemplates.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/rrelu_with_noise_native.h>
#endif


namespace at::native {

template <typename scalar_t, int unroll_factor, typename F>
#if __CUDA_ARCH__ >= 350 || defined USE_ROCM
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void rrelu_with_noise_cuda_kernel(
    int numel,
    PhiloxCudaState philox_args,
    scalar_t* output,
    const scalar_t* input,
    scalar_t* noise,
    double lower,
    double upper,
    const F& random_func) {
  auto seeds = at::cuda::philox::unpack(philox_args);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              idx,
              std::get<1>(seeds),
              &state);

  int grid_stride = blockDim.x * gridDim.x * unroll_factor;
  int rounded_size = ((numel - 1) / grid_stride + 1) * grid_stride;
  double range = upper - lower;

  for (int linear_index = idx; linear_index < rounded_size; linear_index += grid_stride) {
    auto rand = random_func(&state);

    // ensure that (&rand.x)[ii] is safe
    static_assert(sizeof(rand)/sizeof(rand.x) == unroll_factor, "");

    #pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li >= numel) {
        continue;
      }
      scalar_t r = static_cast<scalar_t>((&rand.x)[ii]);
      r = r * range + lower;
      if (input[li] <= 0) {
        output[li] = input[li] * r;
        noise[li] = r;
      } else {
        output[li] = input[li];
        noise[li] = static_cast<scalar_t>(1);
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t>
inline void _rrelu_with_noise_cuda_train(
    Tensor& output,
    const Tensor& input_,
    Tensor& noise_,
    const Scalar& lower_,
    const Scalar& upper_,
    std::optional<Generator> generator) {
  auto input = input_.contiguous();
  auto noise = noise_.contiguous();
  Tensor tmp_output = output.contiguous();

  int64_t numel = input.numel();
  const int unroll_factor = std::is_same_v<scalar_t, double> ? 2 : 4;
  auto [counter_offset, grid, block] = calc_execution_policy(numel, unroll_factor);

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(
      generator, cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }

  const scalar_t* input_data = input.const_data_ptr<scalar_t>();
  scalar_t* noise_data = noise.mutable_data_ptr<scalar_t>();
  scalar_t* output_data = tmp_output.mutable_data_ptr<scalar_t>();

  double lower = lower_.to<double>();
  double upper = upper_.to<double>();

  auto stream = at::cuda::getCurrentCUDAStream();

  if (std::is_same_v<scalar_t, double>) {
    rrelu_with_noise_cuda_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
        numel,
        rng_engine_inputs,
        output_data,
        input_data,
        noise_data,
        lower,
        upper,
        [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand_uniform2_double(state);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // half and float
    rrelu_with_noise_cuda_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
        numel,
        rng_engine_inputs,
        output_data,
        input_data,
        noise_data,
        lower, upper,
        [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand_uniform4(state);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (!output.is_contiguous()) {
    output.copy_(tmp_output);
  }
}

Tensor& rrelu_with_noise_out_cuda(const Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator,
    Tensor& output) {
  at::native::resize_output(output, self.sizes());

  if (self.numel() == 0) {
    return output;
  }

  TensorArg self_arg{self, "self", 1}, noise_arg{noise, "noise", 2},
      output_arg{output, "output", 3};
  checkAllSameGPU("rrelu_with_noise_out_cuda", {self_arg, noise_arg, output_arg});

  if (training) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        self.scalar_type(), "rrelu_with_noise_out_cuda", [&] {
          _rrelu_with_noise_cuda_train<scalar_t>(
              output, self, noise, lower, upper, generator);
        });
  }
  else {
    auto lower_tensor = lower.to<double>();
    auto upper_tensor = upper.to<double>();
    Scalar negative_slope = (lower_tensor + upper_tensor) / 2;
    at::leaky_relu_out(output, self, negative_slope);
  }
  return output;
}

Tensor rrelu_with_noise_cuda(
    const Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  Tensor output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::native::rrelu_with_noise_out_cuda(self, noise, lower, upper, training, generator, output);
}

Tensor& rrelu_with_noise_cuda_(
    Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  return at::native::rrelu_with_noise_out_cuda(
      self, noise, lower, upper, training, generator, self);
}

}  // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/cuda/CUDAGeneratorImpl.h`
- `ATen/native/cuda/DistributionTemplates.h`
- `ATen/native/Resize.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/leaky_relu.h`
- `ATen/ops/rrelu_with_noise_native.h`


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

- **File Documentation**: `RreluWithNoise.cu_docs.md`
- **Keyword Index**: `RreluWithNoise.cu_kw.md`
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

- **File Documentation**: `RreluWithNoise.cu_docs.md_docs.md`
- **Keyword Index**: `RreluWithNoise.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
