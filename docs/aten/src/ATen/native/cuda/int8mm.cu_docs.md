# Documentation: `aten/src/ATen/native/cuda/int8mm.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/int8mm.cu`
- **Size**: 2,488 bytes (2.43 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace at::native {

__global__ void weight_int8pack_mm_kernel(
    const float* x,
    const int8_t* w,
    const float* scale,
    float* out,
    int B,
    int K,
    int N) {
  // one thread per output element: [B, N]
  int b = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= B || n >= N)
    return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += x[b * K + k] * static_cast<float>(w[n * K + k]);
  }

  out[b * N + n] = acc * scale[n];
}

void launch_weight_int8pack_mm_cuda_kernel(
    const Tensor& x,
    const Tensor& w_int8,
    const Tensor& scale,
    Tensor& out) {
  const int B = x.size(0);
  const int K = x.size(1);
  const int N = w_int8.size(0);

  const dim3 block(16, 16);
  const dim3 grid((N + block.x - 1) / block.x, (B + block.y - 1) / block.y);

  auto stream = at::cuda::getCurrentCUDAStream();

  weight_int8pack_mm_kernel<<<grid, block, 0, stream>>>(
      x.data_ptr<float>(),
      w_int8.data_ptr<int8_t>(),
      scale.data_ptr<float>(),
      out.data_ptr<float>(),
      B,
      K,
      N);
}

// Main GPU entry point
at::Tensor _weight_int8pack_mm_cuda(
    const at::Tensor& x,
    const at::Tensor& w_int8,
    const at::Tensor& scale) {
  // --- Check inputs ---
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w_int8.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");

  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(w_int8.dim() == 2, "w must be 2D");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D");

  TORCH_CHECK(
      x.size(1) == w_int8.size(1),
      "K dimension mismatch: x.size(1) != w.size(1)");
  TORCH_CHECK(
      w_int8.size(0) == scale.size(0),
      "Output dim mismatch: w.size(0) != scale.size(0)");

  // --- Determine shapes ---
  auto B = x.size(0); // batch size
  auto N = w_int8.size(0); // output dim

  // Ensure inputs are in the correct types for the kernel
  auto x_f32 = x.to(at::kFloat);
  auto w_int8_contiguous = w_int8.contiguous();
  auto scale_f32 = scale.to(at::kFloat);

  // --- Allocate output ---
  auto out = at::empty({B, N}, x_f32.options());

  // --- Launch kernel ---
  launch_weight_int8pack_mm_cuda_kernel(
      x_f32, w_int8_contiguous, scale_f32, out);

  return out.to(x.dtype());
}

} // namespace at::native

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

- `ATen/ATen.h`
- `ATen/core/Tensor.h`
- `ATen/cuda/CUDAContext.h`
- `c10/cuda/CUDAGuard.h`


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

- **File Documentation**: `int8mm.cu_docs.md`
- **Keyword Index**: `int8mm.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
