# Documentation: `aten/src/ATen/native/sparse/cuda/SparseSemiStructuredApply.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/sparse/cuda/SparseSemiStructuredApply.cu`
- **Size**: 3,756 bytes (3.67 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/accumulate.h>

#if defined(USE_ROCM) || defined(_MSC_VER)
#else
#include <ATen/native/sparse/cuda/SparseSemiStructuredPack.h>
#endif

namespace at::native {

#if defined(USE_ROCM) || defined(_MSC_VER)
#else
template <typename KT>
__global__ void __launch_bounds__(32 /* num_threads */)
  sparse_semi_structured_apply_kernel(typename KT::Params p)
{
  KT::sparse_semi_structured_apply_kernel(p);
}

// Apply a 2:4 sparsify pattern computed with
// `_sparse_semi_structured_tile` to another Tensor
template <bool kIsMeta, typename Element>
std::tuple<Tensor, Tensor> _sparse_semi_structured_apply_typed(Tensor input, Tensor threads_masks)
{
  using KT = KernelTypes<Element>;
  // TODO: Technically we should be able to deal with that
  // by running on the transpose of `input` and swapping
  // `packed` & `packed_t`.
  // This would require to adapt the `threads_masks` a bit tho.
  if (input.stride(1) != 1) {
    input = input.contiguous();
  }
  std::optional<at::cuda::CUDAGuard> device_guard;
  if (!kIsMeta) {
    device_guard.emplace(input.device());
  }

  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input.stride(1) == 1);
  TORCH_CHECK(input.stride(0) % 8 == 0);
  TORCH_CHECK(input.size(1) % 32 == 0, "Wrong alignment shape[1]");

  auto roundedx = cutlass::round_up(input.size(0), kWarpX);
  auto roundedy = cutlass::round_up(input.size(1), kWarpY);
  at::Tensor packed =
      at::empty({roundedx, cutlass::ceil_div(roundedy, 2)}, input.options());
  at::Tensor packed_trans =
      at::empty({roundedy, cutlass::ceil_div(roundedx, 2)}, input.options());

  typename KT::Params p;
  p.input = (Element const*)input.data_ptr();
  p.input_s0 = input.stride(0);
  p.input_dim0 = input.size(0);
  p.input_dim1 = input.size(1);

  p.packed = (Element*)packed.data_ptr();
  p.packed_stride = packed.stride(0);
  p.packed_trans = (Element*)packed_trans.data_ptr();
  p.packed_trans_stride = packed_trans.stride(0);

  p.threads_masks = (uint64_t*)threads_masks.data_ptr();

  TORCH_CHECK(threads_masks.dim() == 3);
  TORCH_CHECK(
      threads_masks.size(0) == p.getBlocksGrid().x * p.getThreadsGrid().x);
  TORCH_CHECK(
      threads_masks.size(1) == p.getBlocksGrid().y * p.getThreadsGrid().y);
  TORCH_CHECK(threads_masks.stride(1) == sizeof(p.threads_masks[0]));
  TORCH_CHECK(threads_masks.size(2) == sizeof(p.threads_masks[0]));
  TORCH_CHECK(threads_masks.stride(2) == 1);
  TORCH_CHECK(threads_masks.scalar_type() == at::ScalarType::Byte);

  if (!kIsMeta) {
    size_t smem_bytes = 0;
    sparse_semi_structured_apply_kernel<KT>
        <<<p.getBlocksGrid(),
           p.getThreadsGrid(),
           smem_bytes,
           at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return std::make_tuple(packed, packed_trans);
}
#endif

std::tuple<Tensor, Tensor> _sparse_semi_structured_apply(const Tensor& input, const Tensor& threads_masks) // Returned by `_sparse_semi_structured_tile`
{
#if defined(USE_ROCM) || defined(_MSC_VER)
  TORCH_CHECK(false, "_sparse_semi_structured_apply: not supported");
  return std::make_tuple(Tensor{}, Tensor{});
#else
  TORCH_CHECK(
    input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16,
    "Unsupported dtype - only `float16` and `bfloat16` are supported currently"
  );
  auto result = (input.scalar_type() == at::ScalarType::Half)
            ? _sparse_semi_structured_apply_typed<false, cutlass::half_t>(input, threads_masks)
            : _sparse_semi_structured_apply_typed<false, cutlass::bfloat16_t>(input, threads_masks);
  return result;
#endif
}

} // namespace

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/sparse/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/sparse/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ScalarOps.h`
- `ATen/Tensor.h`
- `ATen/Functions.h`
- `ATen/Utils.h`
- `c10/cuda/CUDAGuard.h`
- `c10/util/accumulate.h`
- `ATen/native/sparse/cuda/SparseSemiStructuredPack.h`


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

Files in the same folder (`aten/src/ATen/native/sparse/cuda`):

- [`cuSPARSELtOps.cpp_docs.md`](./cuSPARSELtOps.cpp_docs.md)
- [`SparseCsrTensorMath.cu_docs.md`](./SparseCsrTensorMath.cu_docs.md)
- [`SparseSemiStructuredOps.cu_docs.md`](./SparseSemiStructuredOps.cu_docs.md)
- [`SparseCUDABlas.h_docs.md`](./SparseCUDABlas.h_docs.md)
- [`SparseMatMul.cu_docs.md`](./SparseMatMul.cu_docs.md)
- [`SparseCUDATensorMath.cuh_docs.md`](./SparseCUDATensorMath.cuh_docs.md)
- [`cuSPARSELtOps.h_docs.md`](./cuSPARSELtOps.h_docs.md)
- [`SparseBlas.cpp_docs.md`](./SparseBlas.cpp_docs.md)
- [`StaticSort.h_docs.md`](./StaticSort.h_docs.md)
- [`SparseSemiStructuredApplyDense.cu_docs.md`](./SparseSemiStructuredApplyDense.cu_docs.md)


## Cross-References

- **File Documentation**: `SparseSemiStructuredApply.cu_docs.md`
- **Keyword Index**: `SparseSemiStructuredApply.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
