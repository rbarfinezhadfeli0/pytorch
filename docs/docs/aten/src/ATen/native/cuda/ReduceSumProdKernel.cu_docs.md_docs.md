# Documentation: `docs/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu_docs.md`
- **Size**: 12,106 bytes (11.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/ReduceSumProdKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ReduceSumProdKernel.cu`
- **Size**: 9,066 bytes (8.85 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/jit_macros.h>
#include <ATen/OpMathType.h>

namespace at::native {

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct sum_functor {
  void operator()(TensorIterator& iter) {
    const auto sum_combine = [] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
      return a + b;
    };
    constexpr bool is_16_bits = sizeof(scalar_t) == 2;
    if constexpr (is_16_bits) {
      gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/8>(
        iter, func_wrapper<out_t>(sum_combine)
      );
    } else {
      gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>(sum_combine)
      );
    }
  }
};

// jiterated specialization for `complex<Half>`
constexpr char sum_name[] = "sum";
template <>
struct sum_functor<c10::complex<at::Half>> {
// jiterator reduction fails on windows
// Ref: https://github.com/pytorch/pytorch/issues/77305
#if AT_USE_JITERATOR() && !defined(_MSC_VER)
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    std::string func = jiterator_stringify(
    arg_t combine(arg_t a, arg_t b) {
      return a + b;
    }
    );
    jitted_gpu_reduce_kernel<sum_name, scalar_t, scalar_t>(
        iter, func, 0.);
  }
#else
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, func_wrapper<scalar_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a + b;
        }), acc_t{0.});
  }
#endif
};

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct nansum_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NanSumOps<acc_t, out_t>{});
  }
};

constexpr char nansum_name[] = "nansum";
template <typename scalar_t>
struct nansum_functor_complex {
#if AT_USE_JITERATOR()
  void operator()(TensorIterator& iter) {
    std::string func = jiterator_stringify(
        arg_t combine(arg_t a, arg_t b) {
          return a + (std::isnan(b) ? arg_t{0.} : b);
        }
    );
    jitted_gpu_reduce_kernel<nansum_name, scalar_t, scalar_t>(
        iter, func, 0.);
  }
#else
  void operator()(TensorIterator& iter) {
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, NanSumOps<acc_t, acc_t>{});
  }
#endif
};

constexpr char prod_name[] = "prod";
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct prod_functor {
  // jiterator reduction fails on windows
  // Ref: https://github.com/pytorch/pytorch/issues/77305
  #if AT_USE_JITERATOR() && !defined(_MSC_VER)
  void operator()(TensorIterator& iter) {
    std::string func = jiterator_stringify(
    arg_t combine(arg_t a, arg_t b) {
      return a * b;
    }
    );
    jitted_gpu_reduce_kernel<prod_name, scalar_t, out_t>(
        iter, func, 1.);
  }
  #else
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a * b;
        }), 1.);
  }
  #endif
};

// Workaround for the error: '*' in boolean context, suggest '&&' instead [-Werror=int-in-bool-context]
template <>
struct prod_functor<bool> {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<bool, bool>(
        iter, func_wrapper<bool>([] GPU_LAMBDA(bool a, bool b) -> bool {
          return a && b;
        }), 1);
  }
};

// jiterated specialization for `complex<Half>`
template <>
struct prod_functor<c10::complex<at::Half>> {
// jiterator reduction fails on windows
// Ref: https://github.com/pytorch/pytorch/issues/77305
#if AT_USE_JITERATOR() && !defined(_MSC_VER)
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    std::string func =
        jiterator_stringify(arg_t combine(arg_t a, arg_t b) { return a * b; });
    jitted_gpu_reduce_kernel<prod_name, scalar_t, scalar_t>(iter, func, 1.);
  }
#else
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter,
        func_wrapper<scalar_t>(
            [] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t { return a * b; }),
        acc_t{1.});
  }
#endif
};

template <typename scalar_t, typename enable = void>
struct xor_sum_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, uint64_t>(
        iter,
        func_wrapper<uint64_t>(
            [] GPU_LAMBDA(uint64_t a, uint64_t b) -> uint64_t {
              return a ^ b;
            }));
  }
};

template <typename scalar_t>
struct xor_sum_functor<scalar_t, std::enable_if_t<!std::is_integral_v<scalar_t>>> {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, double>(
        iter,
        // implicitly upcast scalar_t to double
        func_wrapper<double>([] GPU_LAMBDA(double a, double b) -> double {
          union {
            double d;
            uint64_t u;
          } a_converter, b_converter, result_converter;

          a_converter.d = a;
          b_converter.d = b;
          result_converter.u = a_converter.u ^ b_converter.u;
          // return a double, otherwise uint64_t will be cast to double
          // when accumulating and the result will be wrong
          return result_converter.d;
        }));
  }
};

template <typename scalar_t>
struct xor_sum_functor<scalar_t, std::enable_if_t<std::is_same_v<scalar_t, bool>>>  {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<bool, uint64_t>(
        iter, func_wrapper<uint64_t>([] GPU_LAMBDA(bool a, bool b) -> uint64_t {
          // Bitcast to uint64_t after the XOR operation (using != for booleans)
          return static_cast<uint64_t>(a != b);
        }));
  }
};

// The function `reduce_dispatch` below dispatches to the kernel based
// on the type of `iter`. It takes care of the common logic
// for handling Half-Precision floating types.
// Otherwise the functor `op` is called to dispatch to the kernel
// of relevant type.
//
// Note: Functor `op` should take care of all the types to be supported
//       except for `at::Half` and `at::BFloat16`.
template <
    template <
        typename scalar_t,
        typename acc_t = scalar_t,
        typename out_t = scalar_t>
    typename OpFunctor,
    typename GeneralDispatcher>
static void reduce_dispatch(TensorIterator& iter, GeneralDispatcher op) {
  if (iter.dtype() == kHalf) {
    return OpFunctor<at::Half, float>{}(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::Half, float, float>{}(iter);
  } else if (iter.dtype() == kBFloat16) {
    return OpFunctor<at::BFloat16, float>{}(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::BFloat16, float, float>{}(iter);
  }
  op(iter);
}

static void sum_kernel_cuda(TensorIterator& iter){
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kBool, kComplexHalf, iter.dtype(), "sum_cuda", [&]() {
          sum_functor<scalar_t>{}(iter);
        });
  };

  reduce_dispatch<sum_functor>(iter, general_dispatcher);
}

static void nansum_kernel_cuda(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    auto dtype = iter.dtype();
    if (at::isComplexType(dtype)) {
        AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "nansum_cuda", [&]() {
          nansum_functor_complex<scalar_t>{}(iter);
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nansum_cuda", [&]() {
          nansum_functor<scalar_t>{}(iter);
        });
    }
  };

  reduce_dispatch<nansum_functor>(iter, general_dispatcher);
}

static void prod_kernel_cuda(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kComplexHalf, kBool, iter.dtype(), "prod_cuda", [&]() {
      prod_functor<scalar_t>{}(iter);
    });
  };

  reduce_dispatch<prod_functor>(iter, general_dispatcher);
}

static void xor_sum_kernel_cuda(TensorIterator& iter) {
  // Use iter.dtype(1) to dispatch based on the type of the input tensor
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.dtype(1), "xor_sum_cuda", [&]() {
        xor_sum_functor<scalar_t>{}(iter);
      });
}

REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda)
REGISTER_DISPATCH(nansum_stub, &nansum_kernel_cuda)
REGISTER_DISPATCH(prod_stub, &prod_kernel_cuda)
REGISTER_DISPATCH(xor_sum_stub, &xor_sum_kernel_cuda)

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `sum_functor`, `sum_functor`, `nansum_functor`, `nansum_functor_complex`, `prod_functor`, `prod_functor`, `prod_functor`, `xor_sum_functor`, `xor_sum_functor`, `xor_sum_functor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/TensorIterator.h`
- `ATen/native/cuda/Reduce.cuh`
- `ATen/native/DispatchStub.h`
- `ATen/native/SharedReduceOps.h`
- `ATen/Dispatch.h`
- `ATen/native/ReduceOps.h`
- `ATen/jit_macros.h`
- `ATen/OpMathType.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `ReduceSumProdKernel.cu_docs.md`
- **Keyword Index**: `ReduceSumProdKernel.cu_kw.md`
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

- **File Documentation**: `ReduceSumProdKernel.cu_docs.md_docs.md`
- **Keyword Index**: `ReduceSumProdKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
