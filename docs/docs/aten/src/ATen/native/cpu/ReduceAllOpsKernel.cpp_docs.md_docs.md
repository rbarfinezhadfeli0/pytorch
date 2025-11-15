# Documentation: `docs/aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp_docs.md`
- **Size**: 11,291 bytes (11.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp`
- **Size**: 8,378 bytes (8.18 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOpsUtils.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/OpMathType.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at::native {
namespace {

using namespace vec;

template <typename scalar_t, typename func_t, typename vec_func_t>
inline void reduce_all_impl_vec(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op,
    vec_func_t vop) {
  using Vec = Vectorized<opmath_type<scalar_t>>;
  const int64_t input_numel = input.numel();
  auto input_data = input.const_data_ptr<scalar_t>();
  // NOTE: parallel_reduce not support bool type
  scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t /*ident*/) -> scalar_t {
      scalar_t partial_out = vec::reduce_all<scalar_t>(
        [=](Vec x, Vec y) { return vop(x, y); },
        input_data + start,
        end - start);
      return partial_out;
    }, op);
  output.fill_(result);
}

// For operation not support in avx/avx2
template <typename scalar_t, typename func_t>
inline void reduce_all_impl(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op) {
  const int64_t input_numel = input.numel();
  auto input_data = input.const_data_ptr<scalar_t>();
  scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t ident) -> scalar_t {
      scalar_t partial_out = ident;
      for (const auto i : c10::irange(start, end)) {
         partial_out = op(partial_out, input_data[i]);
      }
      return partial_out;
    }, op);
  output.fill_(result);
}

void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    TensorIterator iter = TensorIteratorConfig()
      .add_input(input)
      .build();
    bool result_data  = true;
    cpu_serial_kernel(iter, [&](const bool a) -> void {
      result_data = result_data && a;
    });
    result.fill_(result_data);
  } else if(input.scalar_type() == ScalarType::Long) {
    // for int64_t, vectorized implementation have performance issue,
    // just use scalar path
    reduce_all_impl<int64_t>(result, input, upper_bound<int64_t>(),
      [=](int64_t a, int64_t b) -> int64_t { return min_impl(a, b); });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "min_all", [&] {
      using Vec = Vectorized<opmath_type<scalar_t>>;
      reduce_all_impl_vec<scalar_t>(result, input, upper_bound<scalar_t>(),
        [=] (scalar_t a , scalar_t b) -> scalar_t { return min_impl(a, b); },
        [=](Vec a, Vec b) -> Vec { return minimum(a, b); });
    });
  }
}

void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    TensorIterator iter = TensorIteratorConfig()
      .add_input(input)
      .build();
    bool result_data  = false;
    cpu_serial_kernel(iter, [&](const bool a) -> void {
      result_data = result_data || a;
    });
    result.fill_(result_data);
  } else if (input.scalar_type() == ScalarType::Long) {
    // for int64_t, vectorized implementation have performance issue,
    // just use scalar path
    reduce_all_impl<int64_t>(result, input, lower_bound<int64_t>(),
      [=](int64_t a, int64_t b) -> int64_t { return max_impl(a, b); });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "max_all", [&] {
      using Vec = Vectorized<opmath_type<scalar_t>>;
      reduce_all_impl_vec<scalar_t>(result, input, lower_bound<scalar_t>(),
        [=] (scalar_t a , scalar_t b) -> scalar_t { return max_impl(a, b); },
        [=](Vec a, Vec b) -> Vec { return maximum(a, b); });
    });
  }
}

// For operation not support in avx/avx2
template <typename scalar_t, typename func_t1, typename func_t2>
inline void reduce_all_impl_two_outputs(
    Tensor& output1,
    Tensor& output2,
    const Tensor& input,
    const std::pair<scalar_t, scalar_t>& ident_v,
    func_t1 reduce_chunk_func,
    func_t2 reduce_acc_func) {
  using scalar_t_pair = std::pair<scalar_t, scalar_t>;
  const int64_t input_numel = input.numel();
  auto input_data = input.const_data_ptr<scalar_t>();
  scalar_t_pair result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t_pair& ident) -> scalar_t_pair {
      scalar_t_pair partial_out(ident);
      for (const auto i : c10::irange(start, end)) {
         partial_out = reduce_chunk_func(partial_out, input_data[i]);
      }
      return partial_out;
    },
    reduce_acc_func
  );
  output1.fill_(result.first);
  output2.fill_(result.second);
}

template <typename scalar_t, typename func_t, typename vec_func_t1, typename vec_func_t2>
inline void reduce_all_impl_vec_two_outputs(
    Tensor& output1,
    Tensor& output2,
    const Tensor& input,
    const std::pair<scalar_t, scalar_t>& ident_v,
    func_t reduce_acc_func,
    vec_func_t1 reduce_chunk_func1,
    vec_func_t2 reduce_chunk_func2) {
  using Vec = Vectorized<opmath_type<scalar_t>>;
  using scalar_t_pair = std::pair<scalar_t, scalar_t>;
  const int64_t input_numel = input.numel();
  auto input_data = input.const_data_ptr<scalar_t>();
  // NOTE: parallel_reduce not support bool type
  std::pair<scalar_t, scalar_t> result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t_pair& /* ident */) -> scalar_t_pair {
    scalar_t_pair partial_out = vec::reduce2_all<scalar_t>(
        [=](Vec x, Vec y) { return reduce_chunk_func1(x, y); },
        [=](Vec x, Vec y) { return reduce_chunk_func2(x, y); },
        input_data + start,
        end - start);
      return partial_out;
    },
    reduce_acc_func
  );
  output1.fill_(result.first);
  output2.fill_(result.second);
}

void aminmax_allreduce_kernel(
    const Tensor& input,
    Tensor& min_result,
    Tensor& max_result) {
  if (input.scalar_type() == ScalarType::Bool) {
    TensorIterator iter = TensorIteratorConfig()
      .add_input(input)
      .build();
    bool min_result_data = true;
    bool max_result_data = false;
    cpu_serial_kernel(iter, [&](const bool a) -> void {
      min_result_data = min_result_data && a;
      max_result_data = max_result_data || a;
    });
    min_result.fill_(min_result_data);
    max_result.fill_(max_result_data);
  } else if (input.scalar_type() == ScalarType::Long) {
    // for int64_t, vectorized implementation have performance issue,
    // just use scalar path
    using int64_t_pair = std::pair<int64_t, int64_t>;
    reduce_all_impl_two_outputs<int64_t>(min_result, max_result, input,
      int64_t_pair(upper_bound<int64_t>(), lower_bound<int64_t>()),
      // reduce over chunk
      [=](int64_t_pair a, int64_t b) -> int64_t_pair {
        return int64_t_pair(min_impl(a.first, b), max_impl(a.second, b));
      },
      // combine two inputs
      [=](int64_t_pair a, int64_t_pair b) -> int64_t_pair {
        return int64_t_pair(min_impl(a.first, b.first), max_impl(a.second, b.second));
      }
    );
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "aminmax_cpu", [&] {
      using Vec = Vectorized<opmath_type<scalar_t>>;
      using scalar_t_pair = std::pair<scalar_t, scalar_t>;
      reduce_all_impl_vec_two_outputs<scalar_t>(
        min_result,
        max_result,
        input,
        scalar_t_pair(upper_bound<scalar_t>(), lower_bound<scalar_t>()),
        [=] (scalar_t_pair a , scalar_t_pair b) -> scalar_t_pair {
          return scalar_t_pair(
            min_impl(a.first, b.first), max_impl(a.second, b.second));
        },
        [=](Vec a, Vec b) -> Vec { return minimum(a, b); },
        [=](Vec a, Vec b) -> Vec { return maximum(a, b); }
      );
    });
  }
}

} // namespace

REGISTER_DISPATCH(min_all_stub, &min_all_kernel_impl)
REGISTER_DISPATCH(max_all_stub, &max_all_kernel_impl)
REGISTER_DISPATCH(aminmax_allreduce_stub, &aminmax_allreduce_kernel)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 26 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vec`, `REGISTER_DISPATCH`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/native/ReduceOps.h`
- `ATen/native/ReduceAllOps.h`
- `ATen/native/ReduceOpsUtils.h`
- `ATen/Dispatch.h`
- `ATen/Parallel.h`
- `ATen/TensorIterator.h`
- `ATen/OpMathType.h`
- `ATen/native/cpu/Loops.h`
- `ATen/native/cpu/zmath.h`
- `ATen/cpu/vec/functional.h`
- `ATen/cpu/vec/vec.h`
- `c10/util/irange.h`


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

- **File Documentation**: `ReduceAllOpsKernel.cpp_docs.md`
- **Keyword Index**: `ReduceAllOpsKernel.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/aten/src/ATen/native/cpu`):

- [`BinaryOpsKernel.cpp_docs.md_docs.md`](./BinaryOpsKernel.cpp_docs.md_docs.md)
- [`MultinomialKernel.cpp_kw.md_docs.md`](./MultinomialKernel.cpp_kw.md_docs.md)
- [`AmpGradScalerKernels.cpp_docs.md_docs.md`](./AmpGradScalerKernels.cpp_docs.md_docs.md)
- [`FusedSGDKernel.cpp_docs.md_docs.md`](./FusedSGDKernel.cpp_docs.md_docs.md)
- [`scaled_modified_bessel_k1.cpp_docs.md_docs.md`](./scaled_modified_bessel_k1.cpp_docs.md_docs.md)
- [`int_mm_kernel.h_docs.md_docs.md`](./int_mm_kernel.h_docs.md_docs.md)
- [`IsContiguous.h_docs.md_docs.md`](./IsContiguous.h_docs.md_docs.md)
- [`MaxPooling.cpp_docs.md_docs.md`](./MaxPooling.cpp_docs.md_docs.md)
- [`WeightNormKernel.cpp_kw.md_docs.md`](./WeightNormKernel.cpp_kw.md_docs.md)
- [`FusedAdamKernel.cpp_docs.md_docs.md`](./FusedAdamKernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ReduceAllOpsKernel.cpp_docs.md_docs.md`
- **Keyword Index**: `ReduceAllOpsKernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
