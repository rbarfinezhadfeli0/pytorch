# Documentation: `aten/src/ATen/native/cpu/CatKernel.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/CatKernel.cpp`
- **Size**: 2,373 bytes (2.32 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/CatKernel.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

struct InputMeta {
  const void* data_ptr;
  int64_t inner_size;

  InputMeta(const Tensor& t, int64_t dim, int64_t inner)
      : data_ptr(t.const_data_ptr()), inner_size(t.sizes()[dim] * inner) {}
};

template <typename scalar_t>
void cat_serial_kernel_impl(
    const Tensor& result,
    const MaterializedITensorListRef& tensors,
    int64_t dim) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dim >= 0 && dim < result.dim(),
      "dim out of range in cat_serial_kernel_impl");
  int64_t outer =
      result.numel() / (result.sizes()[dim] * result.strides()[dim]);
  scalar_t* result_data = result.data_ptr<scalar_t>();
  int64_t ninputs = static_cast<int64_t>(tensors.size());
  std::vector<InputMeta> inputs;
  inputs.reserve(ninputs);
  for (const Tensor& tensor : tensors) {
    inputs.emplace_back(tensor, dim, result.strides()[dim]);
  }

  using Vec = vec::Vectorized<scalar_t>;
  scalar_t* result_ptr = result_data;
  for (const auto i : c10::irange(outer)) {
    for (const auto j : c10::irange(ninputs)) {
      int64_t local_inner = inputs[j].inner_size;
      const scalar_t* input_ptr =
          (const scalar_t*)(inputs[j].data_ptr) + i * local_inner;
      int64_t d = 0;
      for (; d < local_inner - (local_inner % Vec::size()); d += Vec::size()) {
        Vec in_vec = Vec::loadu(input_ptr + d);
        in_vec.store(result_ptr + d);
      }
#if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
#pragma unroll
#endif
      for (; d < local_inner; d++) {
        result_ptr[d] = input_ptr[d];
      }
      result_ptr += local_inner;
    }
  }
}

void cat_serial_kernel(
    const Tensor& result,
    const MaterializedITensorListRef& tensors,
    int64_t dim) {
  AT_DISPATCH_V2(
      result.scalar_type(),
      "cat_serial_kernel",
      AT_WRAP(
          [&]() { cat_serial_kernel_impl<scalar_t>(result, tensors, dim); }),
      AT_EXPAND(AT_FLOATING_TYPES),
      kBFloat16,
      kHalf,
      AT_EXPAND(AT_FLOAT8_TYPES));
}

} // anonymous namespace

REGISTER_DISPATCH(cat_serial_stub, &cat_serial_kernel)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `REGISTER_DISPATCH`, `at`

**Classes/Structs**: `InputMeta`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/Dispatch_v2.h`
- `ATen/cpu/vec/functional.h`
- `ATen/cpu/vec/vec.h`
- `ATen/native/cpu/CatKernel.h`
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

Files in the same folder (`aten/src/ATen/native/cpu`):

- [`UpSampleKernelAVXAntialias.h_docs.md`](./UpSampleKernelAVXAntialias.h_docs.md)
- [`SparseFactories.cpp_docs.md`](./SparseFactories.cpp_docs.md)
- [`UnfoldBackwardKernel.cpp_docs.md`](./UnfoldBackwardKernel.cpp_docs.md)
- [`int8mm_kernel.cpp_docs.md`](./int8mm_kernel.cpp_docs.md)
- [`LerpKernel.cpp_docs.md`](./LerpKernel.cpp_docs.md)
- [`UpSampleKernel.cpp_docs.md`](./UpSampleKernel.cpp_docs.md)
- [`scaled_modified_bessel_k0.cpp_docs.md`](./scaled_modified_bessel_k0.cpp_docs.md)
- [`DistributionKernels.cpp_docs.md`](./DistributionKernels.cpp_docs.md)
- [`CopyKernel.cpp_docs.md`](./CopyKernel.cpp_docs.md)
- [`SampledAddmmKernel.cpp_docs.md`](./SampledAddmmKernel.cpp_docs.md)


## Cross-References

- **File Documentation**: `CatKernel.cpp_docs.md`
- **Keyword Index**: `CatKernel.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
