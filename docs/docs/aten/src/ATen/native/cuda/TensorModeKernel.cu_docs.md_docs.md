# Documentation: `docs/aten/src/ATen/native/cuda/TensorModeKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/TensorModeKernel.cu_docs.md`
- **Size**: 12,921 bytes (12.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/TensorModeKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/TensorModeKernel.cu`
- **Size**: 9,845 bytes (9.61 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/TensorModeKernel.cuh>
#include <ATen/native/cuda/TensorModeKernel.h>
#include <ATen/Dispatch.h>
#include <ATen/native/NonEmptyUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/core/DeviceArray.h>

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace at::native {

template <typename scalar_t>
struct ModeImpl {
  std::tuple<scalar_t, int64_t> operator()(
      scalar_t *iter_begin,
      scalar_t *iter_end) {
    at::cuda::ThrustAllocator thrust_allocator;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto policy = thrust::cuda::par(thrust_allocator).on(stream);

    const auto n_element = iter_end - iter_begin;
    auto cuda_allocator = at::cuda::getCUDADeviceAllocator();
    auto sort_buffer = c10::DeviceArray<int64_t>(*cuda_allocator, n_element);
    auto sort_buffer_ptr = thrust::device_pointer_cast(sort_buffer.get());
    auto count_from_zero_iter = thrust::make_counting_iterator(int64_t{0});
    thrust::copy_n(policy, count_from_zero_iter, n_element, sort_buffer_ptr);


    // Sort the input data. The original indices of the data are stored in
    // sort_buffer_ptr
    thrust::sort_by_key(policy, iter_begin, iter_end, sort_buffer_ptr);

    // Count # of unique elements via an inner product between adjacent elements.
    // Add 1 if two neighboring element are not equal.
    int unique = 1 +
        thrust::inner_product(
                    policy,
                    iter_begin,
                    iter_end - 1,
                    iter_begin + 1,
                    0,
                    thrust::plus<int>(),
                    thrust::not_equal_to<scalar_t>());

    // Count frequency of each element
    auto keys = c10::DeviceArray<scalar_t>(*cuda_allocator, unique);
    auto counts = c10::DeviceArray<int64_t>(*cuda_allocator, unique);

    auto keys_ptr = thrust::device_pointer_cast(keys.get());
    auto counts_ptr = thrust::device_pointer_cast(counts.get());

    thrust::reduce_by_key(
        policy,
        iter_begin,
        iter_end,
        thrust::constant_iterator<int>(1),
        keys_ptr,
        counts_ptr);

    // Find index of maximum count
    auto it = thrust::max_element(policy, counts_ptr, counts_ptr + unique);
    scalar_t mode = keys_ptr[it - counts_ptr];

    // Find first index within which it occurs
    auto position_iter = thrust::find(policy, iter_begin, iter_end, mode);

    // Translate to original non-sorted index
    TORCH_INTERNAL_ASSERT(position_iter != iter_end);
    int64_t index = sort_buffer_ptr[position_iter - iter_begin];
    return {mode, index};
  }
};

struct EqualsMode {
  bool mode;

  C10_DEVICE bool operator()(const uint8_t x) {
    return static_cast<bool>(x) == mode;
  }
};

template <>
struct ModeImpl<bool> {
  std::tuple<bool, int64_t> operator()(
      const bool *first,
      const bool *last) {
    at::cuda::ThrustAllocator thrust_allocator;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto policy = thrust::cuda::par(thrust_allocator).on(stream);

    // For bool, we can skip finding the unique elements since there
    // are only two possible values.

    // See NOTE [Loading boolean values]
    auto first_bytes = reinterpret_cast<const uint8_t*>(first);
    auto last_bytes = reinterpret_cast<const uint8_t*>(last);

    const auto numel = last - first;
    const auto num_true = thrust::count_if(
        policy,
        first_bytes,
        last_bytes,
        [] GPU_LAMBDA (uint8_t x) {
          return static_cast<bool>(x);
        }
      );
    const auto num_false = (numel - num_true);
    const auto mode = num_true > num_false;

    // Find first index within which it occurs
    const auto position_iter = thrust::find_if(
        policy, first_bytes, last_bytes, EqualsMode{mode});
    const int64_t index = position_iter - first_bytes;
    return {mode, index};
  }
};

template <typename scalar_t>
void calculate_mode(
    const TensorBase& values,
    const TensorBase& indices,
    const TensorBase& self,
    std::vector<int64_t>& position,
    int dim) {

  TORCH_INTERNAL_ASSERT(self.is_contiguous());

  // Because the input is contiguous, we want to get a reference to the
  // location of the buffer at the innermost dimension that we are going
  // to calculate the mode for --> we do this by manually doing the stride
  // calculations to get an offset
  //
  // Yes, mutating self is a code smell, but we clone self before
  // entering the bowels of this implementation.
  //
  // See [Note: CUDA torch.mode clones self]
  scalar_t* data = self.mutable_data_ptr<scalar_t>();
  for (int64_t i = 0; i < static_cast<int64_t>(position.size()); i++) {
    data += position[i] * ensure_nonempty_stride(self, i);
  }

  int64_t ndim = ensure_nonempty_dim(self.dim());
  int64_t n_element = ensure_nonempty_size(self, ndim - 1);

  scalar_t* iter_begin = data;
  scalar_t* iter_end = data + n_element;

  auto [mode, index] = ModeImpl<scalar_t>{}(iter_begin, iter_end);

  // Place mode, index in output
  scalar_t* values_data = values.mutable_data_ptr<scalar_t>();
  int64_t* indices_data = indices.mutable_data_ptr<int64_t>();

  for (int64_t i = 0; i < static_cast<int64_t>(position.size()); i++) {
    int64_t pos = position[i];
    values_data += ensure_nonempty_stride(values, i) * pos;
    indices_data += ensure_nonempty_stride(indices, i) * pos;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaMemcpyAsync(
      values_data, &mode, sizeof(scalar_t), cudaMemcpyHostToDevice, stream));
  //memcpy_and_sync will synchronize results
  at::cuda::memcpy_and_sync(indices_data, &index, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
}

template <typename scalar_t>
void apply_mode(
    const TensorBase& values,
    const TensorBase& indices,
    const TensorBase& self,
    std::vector<int64_t>& position,
    int dim,
    int curDim) {
  // Because we have transposed the Tensor, the data for the dimension we are
  // mode'ing along is always in the innermost dimension
  int64_t ndim = ensure_nonempty_dim(self.dim());
  if (curDim == ndim - 1) {
    calculate_mode<scalar_t>(values, indices, self, position, dim);
  } else {
    for (int i = 0; i < ensure_nonempty_size(self, curDim); ++i) {
      position[curDim] = i;
      apply_mode<scalar_t>(values, indices, self, position, dim, curDim + 1);
    }
  }
}

template <int64_t size, typename scalar_t>
void handle_fused_mode(
    dim3 grid,
    const TensorBase& self,
    cuda::detail::TensorInfo<scalar_t, unsigned int>& ti_values,
    cuda::detail::TensorInfo<int64_t, unsigned int>& ti_indices,
    int64_t slice_size,
    int64_t slices) {
  constexpr int num_threads = size / 2;
  int warp_size = at::cuda::warp_size();
  TORCH_INTERNAL_ASSERT(num_threads % warp_size == 0 &&
                num_threads <= cuda_utils::kCUDABlockReduceMaxThreads(), "");
  const auto memsize =
      (sizeof(scalar_t) * size) + (2 * size * sizeof(unsigned int));
  compute_mode<scalar_t, size>
      <<<grid, num_threads, memsize, at::cuda::getCurrentCUDAStream()>>>(
          self.const_data_ptr<scalar_t>(), ti_values, ti_indices, slice_size, slices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void fused_mode(
    const TensorBase& values,
    const TensorBase& indices,
    const TensorBase& self,
    int64_t slice_size,
    int64_t slices) {
  // Set-up TensorInfo structs for passing to kernel
  auto ti_values = cuda::detail::getTensorInfo<scalar_t, unsigned int>(values);
  auto ti_indices = cuda::detail::getTensorInfo<int64_t, unsigned int>(indices);

  // The number of blocks is the number of slices that we need to calculate
  // the mode for. Each block is responsible for computing a single mode
  dim3 grid;
  getGridFromTiles(slices, grid);

  // The blocksize is two elements per thread, rounded up to the nearest power
  // of 2
  auto ceilPowerOf2 = nextHighestPowerOf2(slice_size);

  // Tradeoff between compilation time and the number of specializations.
  // Ideally we would have one handle_fused_mode for each power of 2
  switch (ceilPowerOf2) {
    case 2048:
      handle_fused_mode<2048, scalar_t>(
          grid, self, ti_values, ti_indices, slice_size, slices);
      break;
    case 1024:
    case 512:
    case 256:
      handle_fused_mode<1024, scalar_t>(
          grid, self, ti_values, ti_indices, slice_size, slices);
      break;
    case 128:
    case 64:
    case 32:
    case 16:
    case 8:
    case 4:
    case 2:
      handle_fused_mode<128, scalar_t>(
          grid, self, ti_values, ti_indices, slice_size, slices);
      break;
    case 1:
    default:
      TORCH_INTERNAL_ASSERT(false);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

void launch_fused_mode_kernel(
    const TensorBase &values, const TensorBase &indices, const TensorBase &self,
    int64_t slice_size, int64_t slices) {
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, self.scalar_type(), "cuda_mode", [&] {
    fused_mode<scalar_t>(values, indices, self, slice_size, slices);
  });
}

void launch_apply_mode_kernel(const TensorBase &values, const TensorBase &indices,
                              const TensorBase &self, int64_t dim, int64_t ndim) {
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, self.scalar_type(), "cuda_mode", [&] {
    // Position will store the dimension values we are processing
    std::vector<int64_t> position(ndim - 1, 0);

    apply_mode<scalar_t>(values, indices, self, position, dim, 0);
  });
}

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `ModeImpl`, `EqualsMode`, `ModeImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/cuda/TensorModeKernel.cuh`
- `ATen/native/cuda/TensorModeKernel.h`
- `ATen/Dispatch.h`
- `ATen/native/NonEmptyUtils.h`
- `ATen/cuda/detail/IndexUtils.cuh`
- `ATen/cuda/ThrustAllocator.h`
- `c10/core/DeviceArray.h`
- `thrust/count.h`
- `thrust/device_ptr.h`
- `thrust/device_vector.h`
- `thrust/execution_policy.h`
- `thrust/extrema.h`
- `thrust/find.h`
- `thrust/inner_product.h`
- `thrust/iterator/constant_iterator.h`
- `thrust/sequence.h`
- `thrust/sort.h`


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

- **File Documentation**: `TensorModeKernel.cu_docs.md`
- **Keyword Index**: `TensorModeKernel.cu_kw.md`
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

- **File Documentation**: `TensorModeKernel.cu_docs.md_docs.md`
- **Keyword Index**: `TensorModeKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
