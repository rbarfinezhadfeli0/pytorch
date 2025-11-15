# Documentation: `docs/aten/src/ATen/native/cpu/DistanceOpsKernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/DistanceOpsKernel.cpp_docs.md`
- **Size**: 21,697 bytes (21.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/DistanceOpsKernel.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/DistanceOpsKernel.cpp`
- **Size**: 18,758 bytes (18.32 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Distance.h>

#include <algorithm>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/functional.h>
#include <c10/util/irange.h>

namespace at::native {
namespace {

template<typename scalar_t>
struct Dist {
  using Vec = vec::Vectorized<scalar_t>;

  // Depending on the value of the pnorm, there are specific implementations
  // that are much faster than std::pow(std::abs(a - b), p), but have the same
  // standard loop code for how to process the input vector. To reuse the main
  // outside loop while still guaranteeing that the compiler inlines every
  // different function on p, we break the inner norm logic into structs with
  // static functions that represent what's done differently, and template the
  // outer loop on those structs.
  //
  // The four functions are:
  //     map :      This tells how to modify (a - b) to form the component that
  //                gets summed.
  //     red :      This tells how to sum the result of map up. This is
  //                separate because the inf norm actually uses max instead of
  //                sum.
  //     finish :   This tells what to do with the aggregated value to compute
  //                the norm. Generally this is the result of val ^ (1 / p).
  //     backward : This is the gradient for that norm. Arguments are pretty
  //                self explanatory.
  //
  // There are a few cases where these aren't used. The 0 norm has no backward,
  // because it's always 0, so that's shortcircuited earlier. There's a special
  // implementation of the general backward pass when p is less than two, so
  // there's a struct with only a backward pass for this case.

  // TODO This is an inefficient way to compite sign, and can be much faster
  // using native SSE instructions that should be added to Vectorized.
  static inline Vec sign(Vec val) {
    return vec::minimum(vec::maximum(Vec(0), val.ceil()), Vec(1)) +
      vec::minimum(vec::maximum(Vec(-1), val.floor()), Vec(0));
  }

  static inline Vec abs(Vec val) {
    return val.abs();
  }

  static inline scalar_t abs(scalar_t val) {
    return std::abs(val);
  }

  static inline Vec ceil(Vec val) {
    return val.ceil();
  }

  static inline scalar_t ceil(scalar_t val) {
    return std::ceil(val);
  }

  static inline Vec min(Vec val, scalar_t other) {
    return vec::minimum(val, Vec(other));
  }

  static inline scalar_t min(scalar_t val, scalar_t other) {
    return std::min(val, other);
  }

  static inline Vec max(Vec val, Vec other) {
    return vec::maximum(val, other);
  }

  static inline scalar_t max(scalar_t val, scalar_t other) {
    return std::max(val, other);
  }

  static inline Vec pow(Vec val, Vec p) {
    return val.pow(p);
  }

  static inline scalar_t pow(scalar_t val, scalar_t p) {
    return std::pow(val, p);
  }

  // Zero norm
  template<typename data_t>
  struct zdist_calc {
    static inline data_t map(const data_t& diff, const data_t& p) { return min(ceil(abs(diff)), 1); }
    static inline data_t red(const data_t& agg, const data_t& up) { return agg + up; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return agg; }
  };

  // One norm
  template<typename data_t>
  struct odist_calc {
    static inline data_t map(const data_t& diff, const data_t& p) { return diff; }
    static inline data_t red(const data_t& agg, const data_t& up) { return agg + up; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return agg; }
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t /*dist*/, const Vec& /*p*/) { return Vec(grad) * sign(diff); }
  };

  // Special general pnorm derivative if p is less than two
  struct lttdist_calc {
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) {
      Vec result = (dist == 0.0) ? Vec(0) : (sign(diff) * diff.abs().pow(p - Vec(1)) * Vec(grad) / Vec(dist).pow(p - Vec(1)));
      result = Vec::blendv(result, Vec(0), (diff == Vec(0)) & (p < Vec(1)));
      return result;
    }
  };

  // Two norm
  template<typename data_t>
  struct tdist_calc {
    // TODO This can probably use fused add multiply to get better perf
    static inline data_t map(const data_t& diff, const data_t& p) { return diff * diff; }
    static inline data_t red(const data_t& agg, const data_t& up) { return agg + up; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return std::sqrt(agg); }
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return dist == 0.0 ? Vec(0) : Vec(grad) * diff / Vec(dist); }
  };

  // General p norm
  template<typename data_t>
  struct pdist_calc {
    static inline data_t map(const data_t& diff, const data_t& p) { return pow(diff, p); }
    static inline data_t red(const data_t& agg, const data_t& up) { return agg + up; }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return std::pow(agg, 1.0 / p); }
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return dist == 0.0 ? Vec(0) : diff * diff.abs().pow(p - Vec(2)) * Vec(grad) / Vec(dist).pow(p - Vec(1)); }
  };

  // Inf norm
  template<typename data_t>
  struct idist_calc {
    static inline data_t map(const data_t& diff, const data_t& p) { return diff; }
    static inline data_t red(const data_t& agg, const data_t& up) { return max(agg, up); }
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    // TODO This backward pass uses a very complex expression to compute (diff
    // == dist) that could be much faster if using SSE instructions.
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return Vec(grad) * sign(diff) * (Vec(1) - vec::minimum(Vec(1), (diff.abs() - Vec(dist)).abs().ceil())); }
  };

  template <typename F>
  static void run_parallel_pdist(Tensor& result, const Tensor& self, const scalar_t p) {
    const scalar_t * const self_start = self.const_data_ptr<scalar_t>();
    const scalar_t * const self_end = self_start + self.numel();
    int64_t n = self.size(0);
    int64_t m = self.size(1);

    scalar_t * const res_start = result.data_ptr<scalar_t>();
    int64_t combs = result.numel(); // n * (n - 1) / 2

    // We conceptually iterate over tuples of (i, j, k) where i is the first
    // vector from the input, j is the second, and k is the result index. This
    // parallelizes over the range of k and infers what i and j are from the
    // value of k.
    parallel_for(0, combs, internal::GRAIN_SIZE / (16 * m), [p, self_start, self_end, n, m, res_start](int64_t k, int64_t end) {
      const Vec pvec(p);
      double n2 = static_cast<double>(n) - .5;
      // The -1 accounts for floating point truncation issues
      int64_t i = static_cast<int64_t>((n2 - std::sqrt(n2 * n2 - 2.0 * static_cast<double>(k) - 1.0)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;

      const scalar_t * self_i = self_start + i * m;
      const scalar_t * self_j = self_start + j * m;
      scalar_t * res = res_start + k;
      const scalar_t * const res_end = res_start + end;

      while (res != res_end) {
        *res = F::finish(vec::map2_reduce_all<scalar_t>(
          [&pvec](Vec a, Vec b) { return F::map((a - b).abs(), pvec); },
          F::red, self_i, self_j, m), p);

        res += 1;
        self_j += m;
        if (self_j == self_end) {
          self_i += m;
          self_j = self_i + m;
        }
      }
    });
  }

  // Assumes self is nonempty, contiguous, and 2D
  static void apply_pdist(Tensor& result, const Tensor& self, const scalar_t p) {
    if (p == 0.0) {
      run_parallel_pdist<zdist_calc<Vec>>(result, self, p);
    } else if (p == 1.0) {
      run_parallel_pdist<odist_calc<Vec>>(result, self, p);
    } else if (p == 2.0) {
      run_parallel_pdist<tdist_calc<Vec>>(result, self, p);
    } else if (std::isinf(p)) {
      run_parallel_pdist<idist_calc<Vec>>(result, self, p);
    } else {
      run_parallel_pdist<pdist_calc<Vec>>(result, self, p);
    }
  }

  template <typename F>
  static void run_parallel_cdist(Tensor& result, const Tensor& t1, const Tensor& t2, const scalar_t p) {
    const scalar_t * const t1_start = t1.const_data_ptr<scalar_t>();
    const scalar_t * const t2_start = t2.const_data_ptr<scalar_t>();
    int64_t d = t1.size(0);
    int64_t r1 = t1.size(-2);
    int64_t r2 = t2.size(-2);
    int64_t m = t1.size(-1);

    scalar_t * const res_start = result.data_ptr<scalar_t>();
    int64_t combs = r1 * r2;
    int64_t size1 = r1 * m;
    int64_t size2 = r2 * m;

    parallel_for(0, combs * d, internal::GRAIN_SIZE / (16 * m), [=](int64_t start, int64_t end) {
      scalar_t * res = res_start + start;
      const scalar_t * const res_end = res_start + end;
      int64_t l = start / combs;
      int64_t k = start % combs;
      int64_t i = k / r2;
      int64_t j = k % r2;
      i = i * m;
      j = j * m;

      while (res != res_end) {
        const scalar_t * self_i = t1_start + size1 * l + i;
        const scalar_t * self_j = t2_start + size2 * l + j;

        scalar_t agg = 0;
        for (const auto x : c10::irange(m)) {
          scalar_t a = *(self_i + x);
          scalar_t b = *(self_j + x);
          agg = F::red(agg, F::map(std::abs(a-b), p));
        }
        *res = F::finish(agg, p);

        res += 1;
        j += m;
        if (j == size2) {
          j = 0;
          i += m;
          if (i == size1) {
            i = 0;
            l += 1;
          }
        }
      }
    });
  }

  static void apply_cdist(Tensor& result, const Tensor& x1, const Tensor& x2, const scalar_t p) {
    if (p == 0.0) {
      run_parallel_cdist<zdist_calc<scalar_t>>(result, x1, x2, p);
    } else if (p == 1.0) {
      run_parallel_cdist<odist_calc<scalar_t>>(result, x1, x2, p);
    } else if (p == 2.0) {
      run_parallel_cdist<tdist_calc<scalar_t>>(result, x1, x2, p);
    } else if (std::isinf(p)) {
      run_parallel_cdist<idist_calc<scalar_t>>(result, x1, x2, p);
    } else {
      run_parallel_cdist<pdist_calc<scalar_t>>(result, x1, x2, p);
    }
  }

  // This does a backward pass down a Vec column of the input
  template <typename F>
  static void backward_down_column_pdist(const scalar_t * self_i, scalar_t * res_i, const scalar_t * grad_k, const scalar_t * dist_k, const Vec& pvec, int64_t n, int64_t m, int64_t gs, int64_t count = Vec::size()) {
    for (const scalar_t * const self_end = self_i + m * n; self_i != self_end - m; self_i += m, res_i += m) {

      const Vec self_vec_i = Vec::loadu(self_i, count);
      Vec res_vec_i = Vec::loadu(res_i, count);

      const scalar_t * self_j = self_i + m;
      scalar_t * res_j = res_i + m;
      for (; self_j != self_end; self_j += m, res_j += m, grad_k += gs, dist_k += 1) {
        const Vec self_vec_j = Vec::loadu(self_j, count);
        Vec res_vec_j = Vec::loadu(res_j, count);

        Vec res = F::backward(self_vec_i - self_vec_j, *grad_k, *dist_k, pvec);
        res_vec_i = res_vec_i + res;
        res_vec_j = res_vec_j - res;

        res_vec_j.store(res_j, count);
      }

      res_vec_i.store(res_i, count);
    }
  }

  template <typename F>
  static void run_backward_parallel_pdist(Tensor& result, const Tensor & grad, const Tensor & self, const scalar_t p, const Tensor& dist) {
    const int64_t n = self.size(0);
    const int64_t m = self.size(1);
    const int64_t gs = grad.stride(0);

    const scalar_t * const grad_start = grad.const_data_ptr<scalar_t>();
    const scalar_t * const dist_start = dist.const_data_ptr<scalar_t>();
    const scalar_t * const self_start = self.const_data_ptr<scalar_t>();
    scalar_t * const res_start = result.data_ptr<scalar_t>();

    // The only way to parallelize and avoid locking requires parallelizing
    // over the columns of the input, i.e. we compute the gradient for the
    // first section of each vector independently of the second section, etc.
    at::parallel_for(0, m / Vec::size(), internal::GRAIN_SIZE / (8 * n * n), [p, n, m, gs, grad_start, dist_start, self_start, res_start](int64_t l, int64_t end) {
      const Vec pvec(p);

      const scalar_t * self_l = self_start + l * Vec::size();
      scalar_t * res_l = res_start + l * Vec::size();

      for (const scalar_t * const res_end = res_start + end * Vec::size(); res_l != res_end; self_l += Vec::size(), res_l += Vec::size()) {
        backward_down_column_pdist<F>(self_l, res_l, grad_start, dist_start, pvec, n, m, gs);
      }
    });
    const int64_t remainder = m % Vec::size();
    if (remainder) {
      backward_down_column_pdist<F>(self_start + (m - remainder), res_start + (m - remainder), grad_start, dist_start, Vec(p), n, m, gs, remainder);
    }
  }

  // Assumes self is nonempty, contiguous, and 2D and dist is also contiguous
  static void apply_backward_pdist(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
    result.fill_(0);
    if (p == 0.0) {
    } else if (p == 1.0) {
      run_backward_parallel_pdist<odist_calc<Vec>>(result, grad, self, p, dist);
    } else if (p < 2.0) {
      run_backward_parallel_pdist<lttdist_calc>(result, grad, self, p, dist);
    } else if (p == 2.0) {
      run_backward_parallel_pdist<tdist_calc<Vec>>(result, grad, self, p, dist);
    } else if (std::isinf(p)) {
      run_backward_parallel_pdist<idist_calc<Vec>>(result, grad, self, p, dist);
    } else {
      run_backward_parallel_pdist<pdist_calc<Vec>>(result, grad, self, p, dist);
    }
  }

  static void apply_backward_cdist(Tensor& result, const Tensor& grad, const Tensor& x1, const Tensor& x2, const double p, const Tensor& dist) {
    result.fill_(0);
    if (p == 0.0) {
    } else if (p == 1.0) {
      run_backward_parallel_cdist<odist_calc<Vec>>(result, grad, x1, x2, p, dist);
    } else if (p < 2.0) {
      run_backward_parallel_cdist<lttdist_calc>(result, grad, x1, x2, p, dist);
    } else if (p == 2.0) {
      run_backward_parallel_cdist<tdist_calc<Vec>>(result, grad, x1, x2, p, dist);
    } else if (std::isinf(p)) {
      run_backward_parallel_cdist<idist_calc<Vec>>(result, grad, x1, x2, p, dist);
    } else {
      run_backward_parallel_cdist<pdist_calc<Vec>>(result, grad, x1, x2, p, dist);
    }
  }


  template <typename F>
  static void run_backward_parallel_cdist(Tensor& result, const Tensor & grad, const Tensor & t1, const Tensor & t2, const scalar_t p, const Tensor& dist) {
    const int64_t r1 = t1.size(-2);
    const int64_t r2 = t2.size(-2);
    const int64_t m = t1.size(-1);
    const int64_t d = result.size(0);
    const int64_t l1_size = r1 * m;
    const int64_t l2_size = r2 * m;
    //current implementation supports only tensor that can be collapsed to 1D. However, to avoid checking if grad satisfies this assumption,
    //we call .contiguous() on grad before backward, thus stride is guaranteed to be 1
    //don't use grad.stride(-1), because if last dimension is 1, stride can be bogus.
    const int64_t gs = 1;

    const scalar_t * const grad_start = grad.const_data_ptr<scalar_t>();
    const scalar_t * const dist_start = dist.const_data_ptr<scalar_t>();
    const scalar_t * const t1_start = t1.const_data_ptr<scalar_t>();
    const scalar_t * const t2_start = t2.const_data_ptr<scalar_t>();
    scalar_t * const res_start = result.data_ptr<scalar_t>();

    at::parallel_for(0, m / Vec::size(), internal::GRAIN_SIZE / (16 * r1), [=](int64_t l, int64_t end) {
      const Vec pvec(p);

      const scalar_t * i = t1_start + l * Vec::size();
      const scalar_t * j = t2_start + l * Vec::size();
      scalar_t * res_l = res_start + l * Vec::size();

      for (const scalar_t * const res_end = res_start + end * Vec::size(); res_l != res_end; i += Vec::size(), j += Vec::size(), res_l += Vec::size()) {
        backward_down_column_cdist<F>(i, j, res_l, grad_start, dist_start, pvec, r1, r2, m, d, gs, l1_size, l2_size);
      }
    });
    const int64_t remainder = m % Vec::size();
    if (remainder) {
      backward_down_column_cdist<F>(t1_start + (m - remainder), t2_start + (m - remainder), res_start + (m - remainder), grad_start, dist_start, Vec(p), r1, r2, m, d, gs, l1_size, l2_size, remainder);
    }
  }

  template <typename F>
  static void backward_down_column_cdist(const scalar_t * t1, const scalar_t * t2, scalar_t * res, const scalar_t * grad_k, const scalar_t * dist_k, const Vec& pvec, int64_t r1, int64_t r2, int64_t m, int64_t d, int64_t gs, int64_t l1_size, int64_t l2_size, int64_t count = Vec::size()) {
    const scalar_t * t1_end = t1 + l1_size;
    const scalar_t * t2_end = t2 + l2_size;

    for ([[maybe_unused]] const auto l : c10::irange(d)) {
      for (; t1 != t1_end; t1 += m, res += m) {
        const Vec vec_t1 = Vec::loadu(t1, count);
        Vec res_vec = Vec::loadu(res, count);

        for (const scalar_t * t2_curr = t2; t2_curr != t2_end; t2_curr += m, grad_k += gs, dist_k += 1) {
          const Vec vec_t2 = Vec::loadu(t2_curr, count);
          Vec res = F::backward(vec_t1 - vec_t2, *grad_k, *dist_k, pvec);
          res_vec = res_vec + res;
        }

        res_vec.store(res, count);
      }
      t1_end += l1_size;
      t2_end += l2_size;
      t2 += l2_size;
    }
  }

};

void pdist_forward_kernel_impl(Tensor& result, const Tensor& self, const double p) {
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "pdist", [&] {
    Dist<scalar_t>::apply_pdist(result, self, p);
  });
}

void pdist_backward_kernel_impl(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "pdist_backward", [&] {
    Dist<scalar_t>::apply_backward_pdist(result, grad, self, p, dist);
  });
}

void cdist_kernel_impl(Tensor& result, const Tensor& x1, const Tensor& x2, const double p) {
  AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "cdist", [&] {
    Dist<scalar_t>::apply_cdist(result, x1, x2, p);
  });
}

void cdist_backward_kernel_impl(Tensor& result, const Tensor& grad, const Tensor& x1, const Tensor& x2, const double p, const Tensor& dist) {
  AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "cdist_backward", [&] {
    Dist<scalar_t>::apply_backward_cdist(result, grad, x1, x2, p, dist);
  });
}


}  // anonymous namespace

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl)
REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_impl)
REGISTER_DISPATCH(cdist_stub, &cdist_kernel_impl)
REGISTER_DISPATCH(cdist_backward_stub, &cdist_backward_kernel_impl)

}  // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 71 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `REGISTER_DISPATCH`, `at`

**Classes/Structs**: `Dist`, `with`, `zdist_calc`, `odist_calc`, `lttdist_calc`, `tdist_calc`, `pdist_calc`, `idist_calc`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/Distance.h`
- `algorithm`
- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/Parallel.h`
- `ATen/TensorIterator.h`
- `ATen/cpu/vec/functional.h`
- `c10/util/irange.h`


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

- **File Documentation**: `DistanceOpsKernel.cpp_docs.md`
- **Keyword Index**: `DistanceOpsKernel.cpp_kw.md`
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

- **File Documentation**: `DistanceOpsKernel.cpp_docs.md_docs.md`
- **Keyword Index**: `DistanceOpsKernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
