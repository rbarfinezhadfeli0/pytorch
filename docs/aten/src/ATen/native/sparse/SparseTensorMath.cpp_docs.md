# Documentation: `aten/src/ATen/native/sparse/SparseTensorMath.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/sparse/SparseTensorMath.cpp`
- **Size**: 79,752 bytes (77.88 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIndexing.h>
#include <ATen/native/sparse/SparseTensorMath.h>

#include <c10/util/irange.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Copy.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/SparseTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_addmm.h>
#include <ATen/ops/_sparse_addmm_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_mm_native.h>
#include <ATen/ops/_sparse_sum.h>
#include <ATen/ops/_sparse_sum_backward_native.h>
#include <ATen/ops/_sparse_sum_native.h>
#include <ATen/ops/_sparse_sparse_matmul.h>
#include <ATen/ops/_sparse_mm_reduce_impl.h>
#include <ATen/ops/_sparse_mm_reduce_impl_native.h>
#include <ATen/ops/add.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/any.h>
#include <ATen/ops/any_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/conj_physical_native.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/div.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/floor_divide_native.h>
#include <ATen/ops/hspmm_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/mv_native.h>
#include <ATen/ops/native_norm_native.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/smm_native.h>
#include <ATen/ops/sspaddmm.h>
#include <ATen/ops/sspaddmm_native.h>
#include <ATen/ops/sub_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros_native.h>
#include <ATen/ops/index.h>
#endif

#include <algorithm>

namespace at::native {

using namespace at::sparse;
// --------------------------------------------------------------------
// zero_(SparseTensor)
// --------------------------------------------------------------------

// hummu hummu
SparseTensor& zero_sparse_(SparseTensor& self) {
  AT_ASSERT(self.is_sparse());
  self.sparse_resize_and_clear_(self.sizes(), self.sparse_dim(), self.dense_dim());
  return self._coalesced_(true);
}

// NB: Don't need zeros, zeros_like, already implemented in TensorFactories

// --------------------------------------------------------------------
// mul(SparseTensor, Scalar)
// --------------------------------------------------------------------

SparseTensor& mul_out_sparse_zerodim(SparseTensor& r, const SparseTensor& t, const Tensor& value) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(value.dim() == 0);

  // Resolve a possibly sparse COO value to a strided tensor.
  Tensor value_;
  if (value.is_sparse()) {
    if (value._nnz() == 0) {
      r.resize_as_(t);
      return r.zero_();
    }
    value_ = value.values();
  } else {
    value_ = value;
  }
  // With broadcasting in action, value_ may be a 1-D tensor as long
  // as its shape is (1,).
  AT_ASSERT(value_.numel() == 1);

  if (is_same_tensor(r, t)) {
    r._values().mul_(value_);
  } else {
    r.resize_as_(t);
    auto indices = r._indices();
    indices.resize_as_(t._indices());
    indices.copy_(t._indices());
    Tensor r_values = r._values(); // Sigh... needed because mul_out takes Tensor&
    at::mul_out(r_values, t._values(), value_);
    get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
    r._coalesced_(t.is_coalesced());
  }
  return r;
}

SparseTensor& mul_out_sparse_scalar(SparseTensor& r, const SparseTensor& t, const Scalar& value) {
  return mul_out_sparse_zerodim(r, t, wrapped_scalar_tensor(value));
}

// --------------------------------------------------------------------
// neg(SparseTensor)
// --------------------------------------------------------------------

SparseTensor& neg_out_sparse(const SparseTensor& t, SparseTensor& r) {
  TORCH_CHECK(r.is_sparse(), "Tensor should be sparse");
  TORCH_CHECK(t.is_sparse(), "Tensor should be sparse");

  // copy_sparse_ does not perform the copy if it is the same tensor
  copy_sparse_to_sparse_(r, t);
  r._values().neg_();
  return r;
}

SparseTensor neg_sparse(const SparseTensor& t) {
  SparseTensor r = at::empty_like(t);
  neg_out_sparse(t, r);
  return r;
}

SparseTensor& neg_sparse_(SparseTensor& t) {
  return neg_out_sparse(t, t);
}

// --------------------------------------------------------------------
// pow(SparseTensor, Scalar)
// --------------------------------------------------------------------

// TODO: add in-place variant

SparseTensor& pow_out_sparse_scalar(const SparseTensor& t_, const Scalar& value, SparseTensor& r) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t_.is_sparse());
  TORCH_CHECK(value.toDouble() != 0, "pow: cannot raise to zeroth power on sparse tensor; it would make the result tensor dense");

  // This coalesce is why we can't easily provide an inplace variant
  SparseTensor t = t_.coalesce();

  r.resize_as_(t);
  auto indices = r._indices();
  indices.resize_as_(t._indices());
  indices.copy_(t._indices());
  Tensor r_values = r._values(); // Sigh... needed because pow_out takes Tensor&
  at::pow_out(r_values, t._values(), value);
  get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
  return r._coalesced_(t.is_coalesced());
}

SparseTensor pow_sparse_scalar(const SparseTensor& t, const Scalar& value) {
  SparseTensor r = at::empty({0}, t.options());
  pow_out_sparse_scalar(t, value, r);
  return r;
}

// --------------------------------------------------------------------
// coalesce(SparseTensor)
// --------------------------------------------------------------------

static SparseTensor& coalesce_(SparseTensor& tensor) {
  if (tensor.is_coalesced()) {
    return tensor;
  }

  SparseTensor coalesced = tensor.coalesce();
  tensor._values().resize_as_(coalesced._values());
  tensor._indices().resize_as_(coalesced._indices());
  tensor._values().copy_(coalesced._values());
  tensor._indices().copy_(coalesced._indices());
  tensor._coalesced_(true);
  return tensor;
}

// Note [Sparse Floor Division]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Uncoalesced sparse tensors cannot be floor divided correctly. Integer
// division is considered a special-case of floor division for purposes of
// this note.
// For example, an integer tensor with values=[3, 3] divided by 2 would produce
// values=[1, 1], which sum to 2 instead of 3 (=6/2).
// A float tensor with values=[3., 3.] floor divided by 2 would also produce
// values=[1., 1.] (after truncation), which sum to 2.f instead of 3.f.
// To perform floor division the sparse tensor must be coalesced first.
// --------------------------------------------------------------------
// div(SparseTensor, Scalar)
// --------------------------------------------------------------------

SparseTensor& div_out_sparse_zerodim(const SparseTensor& t, const Tensor& value, std::optional<std::string_view> rounding_mode, SparseTensor& r) {
  TORCH_CHECK(value.dim() == 0, "Sparse division requires a scalar or ",
    "zero-dim dense tensor divisor (got shape ", value.sizes(), " for divisor)");
  TORCH_CHECK(!value.is_sparse(), "Sparse division requires a scalar or ",
    "zero-dim dense tensor divisor (got a sparse divisor)");

  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());

  // See note "Sparse Floor Division"
  const bool should_coalesce = rounding_mode.has_value() && !t.is_coalesced();
  if (is_same_tensor(r, t)) {
    if (should_coalesce) {
      coalesce_(r);
    }
    r._values().div_(value, rounding_mode);
  } else {
    Tensor t_tmp = t;
    if (should_coalesce) {
      t_tmp = t.coalesce();
    }
    r.resize_as_(t_tmp);
    auto indices = r._indices();
    indices.resize_as_(t_tmp._indices());
    indices.copy_(t_tmp._indices());
    Tensor r_values = r._values(); // Sigh... needed because div_out takes Tensor&
    at::div_out(r_values, t_tmp._values(), value, rounding_mode);
    get_sparse_impl(r)->set_nnz_and_narrow(t_tmp._nnz());
    r._coalesced_(t_tmp.is_coalesced());
  }
  return r;
}

SparseTensor& div_out_sparse_zerodim(const SparseTensor& t, const Tensor& value, SparseTensor& r) {
  return div_out_sparse_zerodim(t, value, /*rounding_mode=*/std::nullopt, r);
}

Tensor div_sparse(const Tensor& self, const Tensor& value) {
  auto commonDtype = at::result_type(self, value);
  if (c10::isIntegralType(commonDtype, /*includeBool=*/true)) {
    commonDtype = typeMetaToScalarType(at::get_default_dtype());
  }
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return div_out_sparse_zerodim(self, value, result);
}

Tensor& div_sparse_(Tensor& self, const Tensor& value) {
  return div_out_sparse_zerodim(self, value, self);
}

Tensor div_sparse(const Tensor& self, const Tensor& value, std::optional<std::string_view> rounding_mode) {
  auto commonDtype = at::result_type(self, value);
  if (c10::isIntegralType(commonDtype, /*includeBool=*/true) && !rounding_mode.has_value()) {
    commonDtype = typeMetaToScalarType(at::get_default_dtype());
  }
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return div_out_sparse_zerodim(self, value, std::move(rounding_mode), result);
}

Tensor& div_sparse_(Tensor& self, const Tensor& value, std::optional<std::string_view> rounding_mode) {
  return div_out_sparse_zerodim(self, value, std::move(rounding_mode), self);
}

// --------------------------------------------------------------------
// floor_divide(SparseTensor, Scalar)
// --------------------------------------------------------------------

SparseTensor& floor_divide_out_sparse_zerodim(const SparseTensor& dividend,
  const Tensor& divisor,
  SparseTensor& result) {
  TORCH_CHECK(divisor.dim() == 0, "Sparse floor division requires a scalar or ",
    "zero-dim dense tensor divisor (got shape ", divisor.sizes(), " for divisor)");
  TORCH_CHECK(!divisor.is_sparse(), "Sparse floor division requires a scalar or ",
    "zero-dim dense tensor divisor (got a sparse divisor)");

  AT_ASSERT(result.is_sparse());
  AT_ASSERT(dividend.is_sparse());

  // Case 1: result and dividend are the same tensor
  // Performs floor division in-place
  if (is_same_tensor(result, dividend)) {

    // See note "Sparse Floor Division"
    if (!result.is_coalesced()) {
      coalesce_(result);
    }

    result._values().floor_divide_(divisor);
    return result;
  }

  // Case 2: result and dividend are different tensors
  Tensor dividend_tmp = dividend;

  // Ensures dividend_tmp is coalesced (see note above)
  if (!dividend.is_coalesced()) {
    dividend_tmp = dividend.coalesce();
  }

  // Resizes and indexes result like dividend_tmp
  result.resize_as_(dividend_tmp);
  result._indices().resize_as_(dividend_tmp._indices());
  result._indices().copy_(dividend_tmp._indices());

  // Computes result
  Tensor result_values = result._values();
  at::floor_divide_out(result_values, dividend_tmp._values(), divisor);
  get_sparse_impl(result)->set_nnz_and_narrow(dividend_tmp._nnz());
  result._coalesced_(dividend_tmp.is_coalesced());
  return result;
}

Tensor floor_divide_sparse(const Tensor& self, const Tensor& value) {
  auto commonDtype = at::result_type(self, value);
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return floor_divide_out_sparse_zerodim(self, value, result);
}

Tensor& floor_divide_sparse_(Tensor& self, const Tensor& value) {
  return floor_divide_out_sparse_zerodim(self, value, self);
}

// --------------------------------------------------------------------
// norm(SparseTensor, Scalar)
// --------------------------------------------------------------------

// Only supports floating point, FYI
Tensor norm_sparse(const SparseTensor& self, const Scalar& p) {
  AT_ASSERT(self.is_sparse());
  return norm_sparse(self, p, IntArrayRef{}, false, std::nullopt);
}

Tensor norm_sparse(const SparseTensor& self, const std::optional<Scalar>& p, IntArrayRef dim, bool keepdim, std::optional<ScalarType> dtype) {
  AT_ASSERT(self.is_sparse());
  if (!dim.empty()) {
    // Only full reductions are supported, so check if that is the case
    int64_t ndim = self.dim();
    bool passed_full_reduction_check = static_cast<size_t>(ndim) == dim.size();
    if (passed_full_reduction_check) {
      auto dim_ = dim.vec();
      maybe_wrap_dims(dim_, ndim);
      std::vector<bool> dims_check(ndim, false);
      // Need to check for duplicates, and fail if any are found
      for (auto dim_ind : dim_) {
        if (dims_check[dim_ind]) {
          passed_full_reduction_check = false;
          break;
        }
        dims_check[dim_ind] = true;
      }
    }
    TORCH_CHECK(passed_full_reduction_check,
      "norm_sparse currently only supports full reductions, so 'dim' must either be empty or contain all dimensions of the input");
  }
  TORCH_CHECK(keepdim == false, "norm_sparse currently does not support keepdim=True");
  TORCH_CHECK(!dtype.has_value(), "norm_sparse currently does not support 'dtype' argument");
  constexpr auto TWO = 2.0;
  auto p_ = p.value_or(TWO);
  return self.coalesce()._values().norm(p_);
}

// --------------------------------------------------------------------
// mv(SparseTensor, Tensor)
// --------------------------------------------------------------------

Tensor mv_sparse(const SparseTensor& self, const Tensor& vec)
{
  TORCH_CHECK(self.ndimension() == 2 &&
              vec.ndimension() == 1,
              "mv: two tensor dim should be 2 and 1, but got ",
              "SparseTensor Dim: ", self.ndimension(), "Tensor Dim: ", vec.ndimension());

  TORCH_CHECK(vec.size(-1) == self.size(-1),
              "mv: expected self.size(-1) == vec.size(-1)");

  auto result = self.matmul(vec.unsqueeze(-1));

  return result.squeeze(-1);
}

// --------------------------------------------------------------------
// add(SparseTensor, SparseTensor, Scalar)  [broadcasts]
// --------------------------------------------------------------------

Tensor add_sparse(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // TODO: Why?! Can't we just flip the order here...
  TORCH_CHECK(!(self.is_sparse() && !other.is_sparse()),
              "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return at::add_out(result, self, other, alpha);  // redispatch!
}

Tensor& add_sparse_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::add_out(self, self, other, alpha);  // redispatch!
}

// There's actually nothing sparse specific about these implementations

Tensor sub_sparse(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  sub_check(self, other);
  return native::add_sparse(self, other, -alpha);
}

Tensor& sub_sparse_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  sub_check(self, other);
  return native::add_sparse_(self, other, -alpha);
}

Tensor& sub_out_sparse(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& r) {
  sub_check(self, other);
  return at::add_out(r, self, other, -alpha);  // redispatch!
}


static SparseTensor& add_out_sparse_contiguous(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, const Scalar& value, ScalarType commonDtype) {
    // saving those because they can be overwritten when doing in-place operations
    int64_t t_nnz = t._nnz(), s_nnz = src._nnz(), max_nnz = t_nnz + s_nnz;
    bool coalesced = t.is_coalesced() && src.is_coalesced();
    int64_t sparse_dim = src.sparse_dim();

    Tensor r_indices = at::empty({src.sparse_dim(), max_nnz}, t._indices().options());

    Tensor t_values = t._values().to(commonDtype);
    Tensor s_values = src._values().to(commonDtype);

    Tensor r_values = new_values_with_size_of(s_values, max_nnz).zero_();

    int64_t blockSize = r_values.stride(0);
    int64_t r_i = 0, t_i = 0, s_i = 0;
    auto t_indices = t._indices();
    auto src_indices = src._indices();

    // NB: relies on nnz tests above
    auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
    auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
    auto src_indices_accessor = src_indices.accessor<int64_t, 2>();

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16,
        commonDtype, "cadd_sparse", [&] {
          scalar_t* t_values_ptr = t_values.data_ptr<scalar_t>();
          scalar_t* s_values_ptr = s_values.data_ptr<scalar_t>();
          scalar_t* r_values_ptr = r_values.data_ptr<scalar_t>();
          scalar_t cast_value = value.to<scalar_t>();
          while (t_i < t_nnz || s_i < s_nnz) {
            int64_t cmp;
            if (t_i >= t_nnz) {
              cmp = -1;
            } else if (s_i >= s_nnz) {
              cmp = 1;
            } else {
              cmp = 0;
              for (auto d: c10::irange(sparse_dim)) {
                if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
                  cmp = 1;
                  break;
                }
                if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
                  cmp = -1;
                  break;
                }
              }
            }
            if (cmp >= 0) {
              for (auto d: c10::irange(sparse_dim)) {
                r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
              }
              if (t_values.numel() > 0) {  // We add all elements from t_values to r_values only if t_values is not an empty tensor
                at::native::cpublas::axpy<scalar_t>(blockSize, 1,
                  t_values_ptr + t_i * blockSize, 1,
                  r_values_ptr + r_i * blockSize, 1);
              }
              t_i++;
            }
            if (cmp <= 0) {
              for (auto d: c10::irange(sparse_dim)) {
                r_indices_accessor[d][r_i] = src_indices_accessor[d][s_i];
              }
              if (s_values.numel() > 0) {  // We add all elements from s_values to r_values only if s_values is not an empty tensor
                at::native::cpublas::axpy<scalar_t>(blockSize, cast_value,
                  s_values_ptr + s_i * blockSize, 1,
                  r_values_ptr + r_i * blockSize, 1);
              }
              s_i++;
            }
            r_i++;
          }
        }
    );

    if (r.scalar_type() != commonDtype) {
      r_values = r_values.to(r.scalar_type());
    }
    get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);
    get_sparse_impl(r)->set_nnz_and_narrow(r_i);

    // TODO: I think it may be possible to track inside the loop and
    // detect when we are uncoalesced (e.g., by observing that an
    // index goes backwards) which may be more precise than using the
    // coalesced flag here.  But this is easy.
    return r._coalesced_(coalesced);
}

static SparseTensor& add_out_sparse_non_contiguous(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, const Scalar& value, ScalarType commonDtype) {
    Tensor t_values = t._values().to(commonDtype);
    Tensor s_values = src._values().to(commonDtype);

    // If `t` or `src` contains non-contiguous `values`, `at::native::cpublas::axpy` doesn't work
    // and we concat the indices and values tensors instead.
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
      commonDtype, "add_out_sparse_cpu", [&] {
          if (value.to<scalar_t>() != static_cast<scalar_t>(1)) {
            s_values = s_values.mul(value);
          }
        });

    Tensor r_indices = at::cat({t._indices(), src._indices()}, 1);
    Tensor r_values = at::cat({t_values, s_values}, 0).to(r.scalar_type());
    alias_into_sparse(r, r_indices, r_values);

    // Prevent unbounded growth of nnz
    // TODO: Improved heuristic on when to coalesce or remove need to coalesce
    if (r._nnz() > r.numel()) {
      auto c = r.coalesce();
      alias_into_sparse(r, c._indices(), c._values());
    }

    return r;
}

static Tensor& add_out_dense_sparse_cpu(Tensor& r, const Tensor& dense, const SparseTensor& sparse_, const Scalar& value);

SparseTensor& add_out_sparse_cpu(const SparseTensor& t, const SparseTensor& src, const Scalar& value, SparseTensor& r) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse_cpu(r, t, src, value);
  }
  // TODO: This test seems a bit goofy
  TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  AT_ASSERT(!t.is_cuda());  // the dispatch argument
  TORCH_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!src.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected sizes of 'self' and 'other' to match, but ", t.sizes(), " != ", src.sizes());

  auto commonDtype = promoteTypes(t.scalar_type(), src.scalar_type());

  TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in add operation");

  if (src._nnz() == 0) {
    return copy_sparse_to_sparse_(r, t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r, src, value);
  }

  TORCH_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");

  r.resize_as_(src);
  if (r.is_meta()) {
    return r;
  } else if (src._values().is_contiguous() && t._values().is_contiguous()) {
    return add_out_sparse_contiguous(r, t, src, value, commonDtype);
  } else {
    return add_out_sparse_non_contiguous(r, t, src, value, commonDtype);
  }
}

// --------------------------------------------------------------------
// add(Tensor, SparseTensor, Scalar)
//    formerly known as spcadd
// --------------------------------------------------------------------
template <typename scalar_t>
static void add_dense_sparse_worker_non_hybrid_cpu(Tensor& r, const Scalar& value, const SparseTensor& sparse, const Tensor& indices, const Tensor& values) {
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto values_accessor = values.accessor<scalar_t, 1>();

  scalar_t* r_ptr = r.data_ptr<scalar_t>();
  scalar_t cast_value = value.to<scalar_t>();
  const int64_t sparse_dim = sparse.sparse_dim();
  std::vector<int64_t> result_stride(sparse_dim);
  for (const auto d: c10::irange(sparse_dim)) {
    result_stride[d] = r.stride(d);
  }
  at::parallel_for(0, sparse._nnz(), 0, [&](int64_t start, int64_t end) {
    for (const auto k: c10::irange(start, end)) {
      int64_t index = r.storage_offset();
      for (auto d: c10::irange(sparse_dim)) {
        index += result_stride[d] * indices_accessor[d][k];
      }
      r_ptr[index] += cast_value * values_accessor[k];
    }
  });
}

template <typename scalar_t>
static inline void add_dense_sparse_worker_hybrid_cpu(Tensor& r, const Scalar& value, const SparseTensor& sparse, const Tensor& indices, const Tensor& values) {

  // Get the dense dimension element numbers of hybrid sparse tensor
  int64_t values_dense_size = values.stride(0);
  TORCH_CHECK(values.is_contiguous());
  scalar_t* v_ptr = values.data_ptr<scalar_t>();

  scalar_t* r_ptr = r.data_ptr<scalar_t>();
  TORCH_CHECK(r_ptr != nullptr);

  auto indices_accessor = indices.accessor<int64_t, 2>();
  scalar_t cast_value = value.to<scalar_t>();
  auto sparse_dim = sparse.sparse_dim();
  std::vector<int64_t> result_stride(sparse_dim);
  for (auto d : c10::irange(sparse_dim)) {
    result_stride[d] = r.stride(d);
  }

  at::parallel_for(0, sparse._nnz(), 0, [&](int64_t start, int64_t end) {
    for (auto k: c10::irange(start, end)) {
      auto r_index = r_ptr;
      for (auto d: c10::irange(sparse_dim)) {
        r_index += result_stride[d] * indices_accessor[d][k];
      }
      auto v_index = v_ptr + k * values_dense_size;
      at::native::cpublas::axpy<scalar_t>(values_dense_size, cast_value, v_index, 1, r_index, 1);
    }
  });
}

template <typename scalar_t>
static inline void add_dense_sparse_worker_non_coalesced_cpu(Tensor& r, const Scalar& value,
    const SparseTensor& sparse, const Tensor& indices, const Tensor& values) {

  // Get the dense dimension element numbers of hybrid sparse tensor
  auto values_dense_size = values.stride(0);
  TORCH_CHECK(values.is_contiguous());
  scalar_t* v_ptr = values.data_ptr<scalar_t>();
  TORCH_CHECK(v_ptr != nullptr);

  scalar_t* r_ptr = r.data_ptr<scalar_t>();
  TORCH_CHECK(r_ptr != nullptr);

  scalar_t cast_value = value.to<scalar_t>();
  auto sparse_dim = sparse.sparse_dim();

  auto indices_accessor = indices.accessor<int64_t, 2>();
  int64_t result_length = r.size(0);
  std::vector<int64_t> result_stride(sparse_dim);
  for (auto d : c10::irange(sparse_dim)) {
    result_stride[d] = r.stride(d);
  }

  auto sparse_nnz = sparse._nnz();
  int max_threads = at::get_num_threads();
  max_threads = (result_length < max_threads) ? result_length : max_threads;
  int64_t avg_chunk_down = result_length / max_threads;
  std::vector<int64_t> chuck_size(max_threads);
  for (const auto i : c10::irange(max_threads)) {
    chuck_size[i] = avg_chunk_down;
  }
  //make chunk balance among threads as 211
  for (auto i = 0 ; i < result_length % max_threads ; i++) {
    chuck_size[i] += 1;
  }
  std::vector<int64_t> chuck_sum_size(max_threads + 1);
  chuck_sum_size[0] = 0;
  for (const auto i : c10::irange(1, max_threads)) {
    chuck_sum_size[i] = chuck_sum_size[i - 1] + chuck_size[i - 1];
  }
  chuck_sum_size[max_threads] = result_length;
  at::parallel_for(0, max_threads, 0, [&](int64_t start, int64_t end) {
    for (auto k: c10::irange(start, end)) {
      int64_t chunk_begin = chuck_sum_size[k];
      int64_t chunk_end = chuck_sum_size[k + 1];
      for (const auto n: c10::irange(sparse_nnz)) {
        int64_t chunk_offset = indices_accessor[0][n];
        if (chunk_offset >= chunk_begin && chunk_offset < chunk_end) {
          int64_t r_offset = result_stride[0] * chunk_offset;
          for (const auto d : c10::irange(1, sparse_dim)) {
            r_offset += result_stride[d] * indices_accessor[d][n];
          }
          scalar_t* v_index = v_ptr + n * values_dense_size;
          auto r_index = r_ptr + r_offset;
          at::native::cpublas::axpy<scalar_t>(values_dense_size, cast_value, v_index, 1, r_index, 1);
        }
      }
    }
  });
}

Tensor& add_out_dense_sparse_cpu(Tensor& r, const Tensor& dense, const SparseTensor& sparse_, const Scalar& value) {
  TORCH_CHECK(!r.is_sparse());
  TORCH_CHECK(!dense.is_sparse());
  TORCH_CHECK(sparse_.is_sparse());

  TORCH_CHECK(!dense.is_cuda()); // dispatch argument
  TORCH_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!sparse_.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(dense.sizes().equals(sparse_.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
    dense.sizes(), " while other has size ", sparse_.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");

  auto commonDtype = promoteTypes(dense.scalar_type(), sparse_.scalar_type());
  TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in add operation");

  r.resize_as_(dense);

  auto sparse_nnz = sparse_._nnz();
  if (sparse_nnz == 0) {
    if (!is_same_tensor(r, dense)) r.copy_(dense);
    return r;
  }

  int64_t dense_dim = dense.dim();
  int64_t sparse_dim = sparse_.sparse_dim();
  Tensor resultBuffer = r;
  if (r.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(r, dense)) {
    resultBuffer.copy_(dense);
  }

  Tensor values = sparse_._values();
  bool sparse_is_coalesced = (sparse_.is_coalesced() || sparse_nnz == 1);
  bool result_is_contiguous = ((r.storage().data() != nullptr) && resultBuffer.is_contiguous());
  bool value_is_contiguous = values.is_contiguous();
  bool is_contiguous =  (result_is_contiguous && value_is_contiguous);

  SparseTensor sparse = sparse_;
  Tensor indices = sparse_._indices();
  Tensor valuesBuffer = values.to(commonDtype);
  if (is_contiguous && sparse_is_coalesced) {
    //TODO: we can optimize it for non-hybrid by not using buffers
    if (sparse_dim == dense_dim) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
          at::ScalarType::ComplexHalf, at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
          commonDtype, "add_dense_sparse_non_hybrid", [&] {
            add_dense_sparse_worker_non_hybrid_cpu<scalar_t>(resultBuffer, value, sparse_, indices, valuesBuffer);
          });
    } else {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
          at::ScalarType::ComplexHalf, at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
          commonDtype, "add_dense_sparse_hybrid", [&] {
            add_dense_sparse_worker_hybrid_cpu<scalar_t>(resultBuffer, value, sparse_, indices, valuesBuffer);
          });
    }
  } else if (is_contiguous && (sparse_dim > 0)) {
    // Handle sparse is not coalesced
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        at::ScalarType::ComplexHalf, at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
        commonDtype, "add_dense_sparse_worker_non_coalesced", [&] {
          add_dense_sparse_worker_non_coalesced_cpu<scalar_t>(resultBuffer, value, sparse_, indices, valuesBuffer);
        });
  } else {
    // Slow path for non-contiguous values and output
    // TODO: coalesce() performance may can be further improved
    sparse = sparse_.coalesce();
    indices = sparse._indices();
    values = sparse._values();
    valuesBuffer = values.to(commonDtype);
    auto indices_accessor = indices.accessor<int64_t, 2>();
    auto sparse_nnz = sparse._nnz();
    at::parallel_for(0, sparse_nnz, 100, [&](int64_t start, int64_t end) {
      for (auto k: c10::irange(start, end)) {
        Tensor dstBuffer = resultBuffer;
        for (auto d: c10::irange(sparse_dim)) {
          dstBuffer = dstBuffer.select(0, indices_accessor[d][k]);
        }
        Tensor srcBuffer = valuesBuffer.select(0, k);
        dstBuffer.add_(srcBuffer, value);
      }
    });
  }
  if (r.scalar_type() != commonDtype) {
    r.copy_(resultBuffer);
  }
  return r;
}

// --------------------------------------------------------------------
// mul(SparseTensor, SparseTensor)  [broadcasts]
// --------------------------------------------------------------------

Tensor mul_sparse(const Tensor& self, const Tensor& other) {
  auto commonDtype = at::result_type(self, other);
  // Arbitrary (dense, sparse) and (sparse, dense) multiplication is not
  // currently supported, but (0dim-dense, sparse) and (sparse, 0dim-dense) is.
  // Make sure we use the sparse exemplar for result.
  auto result_options = self.is_sparse() ?
    self.options().dtype(commonDtype) : other.options().dtype(commonDtype);
  Tensor result = at::empty({0}, result_options);
  return at::mul_out(result, self, other);  // redispatch!
}

Tensor& mul_sparse_(Tensor& self, const Tensor& other) {
  if (self.is_sparse()) {
    return at::mul_out(self, self, other);  // redispatch!
  }
  else {
    const auto res = at::mul(self, other);
    self.zero_();
    self.add_(res);
    return self;
  }
}

// A generic function to implement pointwise-like operations
// with index intersection between dense and sparse COO tensors.
// NOTE: op is always called as op(dense_values, sparse_values),
// so it is up to the user to supply right implementations for non-commutative
// operations.
template <typename binary_func_t>
static Tensor& intersection_binary_op_sparse_dense_out(
    const Tensor& d,
    const SparseTensor& s_,
    Tensor& res,
    const char* const op_name,
    const binary_func_t& op,
    const bool coalesce = false) {
  // compute broadcasted shape.
  const auto res_shape = infer_size(d.sizes(), s_.sizes());

  // Short-circuit if either s_ or d is empty.
  if (!s_._nnz() || !s_.numel() || !d.numel()) {
    const int64_t dense_dim = s_.dense_dim();
    const int64_t sparse_dim = static_cast<int64_t>(res_shape.size()) - dense_dim;
    const int64_t nnz = 0;
    const auto indices = at::empty({sparse_dim, nnz}, s_._indices().options());
    auto res_values_shape = s_._values().sizes().vec();
    res_values_shape[0] = nnz;
    const auto values = at::empty(res_values_shape, s_._values().options().dtype(res.scalar_type()));
    auto* res_impl = get_sparse_impl(res);
    res_impl->raw_resize_(sparse_dim, dense_dim, /*size=*/res_shape);
    res_impl->set_indices_and_values_unsafe(indices, values);
    res_impl->set_nnz_and_narrow(nnz);
    return res._coalesced_(true);
  }

  const auto d_dim = d.dim();
  const auto s_dim = s_.dim();

  // Always coalesce when sparse broadcasts over dense,
  // because new sparse dimensions are created and
  // repeated indices have to be eliminated because of that.
  const auto s = (coalesce || d_dim > s_dim) ? s_.coalesce() : s_;

  const auto sparse_dim = s.sparse_dim();
  const auto dense_dim = s.dense_dim();

  const auto s_indices = s._indices();
  const auto s_values = s._values();

  const auto apply_op = [&](const Tensor& d_filtered) -> Tensor& {
    const auto res_indices = s_indices.clone();
    // to(res.scalar_type) is only performed when both d and s are 0-dim.
    // This insures right type promotions with the following rules:
    // op(0-dim, 0-dim).dtype == <common dtype>
    // op(0-dim, ge-1-dim).dtype == <ge-1-dim>.dtype,
    // where ge-1-dim is a tensor with dim >= 1.
    // We do not cast if op is performed in-place.
    // The cast is required if s is 0-dim non-coalesced tensor and d is 0-dim.
    // This is because s.values is at least 1D, so
    // op(s.values, d).dtype == s.values.dtype, but we want
    // op(s.values, d).dtype == <common dtype>.
    const auto values = op(d_filtered, s_values);
    const auto res_values = is_same_tensor(s_, res) ? values : values.to(res.scalar_type());
    auto* res_impl = get_sparse_impl(res);
    res_impl->raw_resize_(sparse_dim, dense_dim, res_shape);
    res_impl->set_indices_and_values_unsafe(res_indices, res_values);
    res_impl->set_nnz_and_narrow(s._nnz());
    return res._coalesced_(s.is_coalesced());
  };

  // Easiest case: only dense dimensions intersect.
  // This means only value tensors interact.
  if (d_dim <= dense_dim) {
    return apply_op(d);
  }

  // Now we have intersection between sparse and dense dims.
  const auto sparse_dim_intersec = std::min(sparse_dim, d_dim - dense_dim);
  const auto d_start_dim_intersec = std::max<int64_t>(0, d_dim - s_dim);
  const auto s_start_dim_intersec = std::max<int64_t>(0, s_dim - d_dim);

  // Index d with s_indices to find values which
  // interact with s_values.
  const auto d_filtered = [&]() -> Tensor {
    using at::indexing::Slice;
    using at::indexing::Ellipsis;
    using at::indexing::TensorIndex;

    std::vector<TensorIndex> intersec_indices;
    intersec_indices.reserve(d_dim);

    if (d_start_dim_intersec) {
      intersec_indices.emplace_back(Ellipsis);
    }
    for (const auto i : c10::irange(sparse_dim_intersec)) {
      const auto s_idx = s_start_dim_intersec + i;
      intersec_indices.emplace_back(s_indices[s_idx]);
    }
    for (auto i = d_start_dim_intersec + sparse_dim_intersec; i < d_dim; ++i) {
      intersec_indices.emplace_back(Slice());
    }
    // we need to expand d in the dimensions it is being indexed into
    // to avoid out of bound indices
    const auto d_expanded_shape = std::vector<int64_t>(
        res_shape.end() - d_dim, res_shape.end());
    return d.expand(d_expanded_shape).index(intersec_indices);
  }();

  // When dims match or sparse is "larger", the result nnz is the same,
  // so only values get modified.
  if (s_dim >= d_dim) {
    return apply_op(d_filtered);
  }

  // Otherwise nnz gets larger, and both indices and values need an update.
  const auto d_batch_shape = d.sizes().slice(0, d_start_dim_intersec);
  const auto d_batch_len = static_cast<int64_t>(d_batch_shape.size());
  int64_t batch_count = 1;
  int64_t max_batch_dim = 0;
  std::tie(batch_count, max_batch_dim) = [d_batch_shape]() -> std::tuple<int64_t, int64_t> {
    int64_t batch_count = 1;
    int64_t max_batch_dim = 0;
    for (const auto& b : d_batch_shape) {
      batch_count *= b;
      max_batch_dim = std::max(b, max_batch_dim);
    }
    return std::make_tuple(batch_count, max_batch_dim);
  }();

  const auto res_sparse_dim = static_cast<int64_t>(d_batch_shape.size()) + sparse_dim;
  const auto res_dense_dim = dense_dim;
  const auto s_nnz = s._nnz();
  const auto res_nnz = batch_count * s_nnz;
  auto res_values_shape = s_values.sizes().vec();
  res_values_shape[0] = res_nnz;
  const auto res_values = op(d_filtered, s_values).reshape(res_values_shape);
  const auto res_indices = [&]() -> Tensor {
    const auto index_buffer = at::arange(max_batch_dim, s_indices.options());
    auto indices = at::empty({res_sparse_dim, res_nnz}, s_indices.options());
    // fill in indices corresponding to the "batch" dimensions of d.
    int64_t n_repeat_interleave = res_nnz;
    int64_t n_repeat = 1;
    for (const auto dim : c10::irange(d_batch_len)) {
      const auto dim_size = d_batch_shape[dim];
      n_repeat_interleave /= dim_size;
      // fill in indices corresponding to the "batch" dimension dim.
      // Equivalent to indices[dim].copy_(repeat_interleave(dim_index, n_repeat_interleave).repeat(n_repeat))
      const std::initializer_list<int64_t> dim_index_expanded_shape = {n_repeat, dim_size, n_repeat_interleave};
      const auto dim_index = index_buffer.slice(-1, 0, dim_size);
      const auto dim_index_expanded = dim_index.unsqueeze(0).unsqueeze_(-1).expand(dim_index_expanded_shape);
      // NOTE: indices is contiguous, so view is safe
      indices[dim].view(dim_index_expanded_shape).copy_(dim_index_expanded);
      n_repeat *= dim_size;
    }
    // fill in indices corresponding to s_indices.
    // Equivalent to indices_sparse.copy(s_indices.repeat({1, n_repeat})
    n_repeat = res_nnz / s_nnz;
    auto indices_sparse = indices.narrow(0, d_batch_len, res_sparse_dim - d_batch_len);
    const std::initializer_list<int64_t> s_indices_expanded_shape = {-1, n_repeat, s_nnz};
    const auto s_indices_expanded = s_indices.unsqueeze(1).expand(s_indices_expanded_shape);
    indices_sparse.view(s_indices_expanded_shape).copy_(s_indices_expanded);

    return indices;
  }();
  auto* res_impl = get_sparse_impl(res);
  res_impl->raw_resize_(res_sparse_dim, res_dense_dim, res_shape);
  res_impl->set_indices_and_values_unsafe(res_indices, res_values);
  res_impl->set_nnz_and_narrow(res_nnz);
  // By design of index expansion and that s is coalesced,
  // the result is also coalesced.
  return res._coalesced_(true);
}

Tensor& _mul_dense_sparse_out(const Tensor& d, const Tensor& s, Tensor& res) {
  return intersection_binary_op_sparse_dense_out(d, s, res, "mul", [](const Tensor& a, const Tensor& b) -> Tensor {
      return at::mul(a, b);
  });
}

Tensor& _mul_sparse_sparse_zero_dim_out(const Tensor& zero_dim, const Tensor& other, Tensor& r) {
  const auto is_wrapped_scalar = [](const Tensor& s) -> bool {
    return !s.dim() && s.is_coalesced();
  };

  const auto extract_vals_from_wrapped_scalar = [](const Tensor& s) -> Tensor {
    auto vals = s._values().squeeze(0);
    // if squeeze does not kill the dim, it means that
    // vals is empty with shape [0]. In such a case we
    // return a 0-dim empty tensor to avoid broadcasting
    // issues in intersection_binary_op_sparse_dense_out
    // when the sparse argument is actually 0-dim.
    if (vals.dim()) {
      return at::empty({}, vals.options());
    }
    return vals;
  };

  // The code dispatches to mul(dense, sparse), and the goal
  // is to delay calling into coalesce when converting one of
  // the sparse arguments to dense if possible.
  // This is possible when there is a 0-dim coalesced argument.

  // if is_wrapped_scalar(zero_dim)
  if (zero_dim.is_coalesced()) {
    const auto scalar_val = extract_vals_from_wrapped_scalar(zero_dim);
    return _mul_dense_sparse_out(scalar_val, other, r);
  }
  // Here zero_dim is not a wrapped scalar, so we test other.
  if (is_wrapped_scalar(other)) {
    const auto scalar_val = extract_vals_from_wrapped_scalar(other);
    return _mul_dense_sparse_out(scalar_val, zero_dim, r);
  }
  // Neither of inputs is a wrapped scalar, but zero_dim
  // is at least 0-dim, so we coalesce it to convert to
  // a scalar.
  const auto scalar_val = extract_vals_from_wrapped_scalar(zero_dim.coalesce());
  return _mul_dense_sparse_out(scalar_val, other, r);
}

DEFINE_DISPATCH(mul_sparse_sparse_out_stub);

Tensor& _mul_sparse_sparse_out(const Tensor& x, const Tensor& y, Tensor& res) {
  mul_sparse_sparse_out_stub(res.device().type(), res, x, y);
  return res;
}

SparseTensor& mul_out_sparse_cpu(const Tensor& t_, const Tensor& src_, Tensor& r) {
  AT_ASSERT(!t_.is_cuda()); // dispatch argument
  TORCH_CHECK(!r.is_cuda(), "mul: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!src_.is_cuda(), "mul: expected 'other' to be a CPU tensor, but got a CUDA tensor");
  // case mul(sparse, dense)
  if (!src_.is_sparse()) {
    return _mul_dense_sparse_out(src_, t_, r);
  }
  // case mul(dense, sparse)
  if (!t_.is_sparse()) {
    return _mul_dense_sparse_out(t_, src_, r);
  }

  // case mul(sparse, sparse) with a 0-dim input.
  if (!src_.dim()) {
    return _mul_sparse_sparse_zero_dim_out(src_, t_, r);
  }
  if (!t_.dim()) {
    return _mul_sparse_sparse_zero_dim_out(t_, src_, r);
  }

  const auto is_equal_size_inputs = t_.sizes().equals(src_.sizes());

  // mul(sparse, sparse) with inputs which broadcast only in dense dims
  if (!is_equal_size_inputs) {
    _mul_sparse_sparse_out(t_, src_, r);
    return r;
  }

  TORCH_CHECK(is_equal_size_inputs, "mul: expected 'self' and 'other' to have same sizes when both are sparse"
      ", but ", t_.sizes(), " != ", src_.sizes());

  // Short circuit when there is zero nnz
  // Not strictly necessary, but there are tests checking whether
  // resize in mul fails if run on tensors coming from .data/.detach.
  if (!t_._nnz() || !src_._nnz()) {
    r.resize_as_(t_);
    return r.zero_();
  }

  // _mul_sparse_sparse_out is faster for large inputs
  // and when either of the inputs is uncoalesced.
  if (!t_.is_coalesced() || !src_.is_coalesced()) {
    _mul_sparse_sparse_out(t_, src_, r);
    return r;
  }

  // Otherwise _mul_sparse_sparse_out might be slower
  // than the brute-force solution below.

  SparseTensor t = t_.coalesce();
  SparseTensor src = src_.coalesce();

  // saving those because they can be overwritten when doing in-place operations
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz();
  int64_t max_nnz = std::min(t_nnz, s_nnz);  // multiply by zero is zero, and can be dropped
  int64_t sparse_dim = src.sparse_dim();
  Tensor t_indices = t._indices();
  Tensor src_indices = src._indices();
  Tensor r_indices = at::empty({sparse_dim, max_nnz}, t_indices.options());

  int64_t r_i = 0, t_i = 0, s_i = 0;

  auto commonDtype = promoteTypes(t_.scalar_type(), src_.scalar_type());
  TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in mul operation");

  Tensor t_values = t._values().to(commonDtype);
  Tensor s_values = src._values().to(commonDtype);

  Tensor r_buffer = new_values_with_size_of(t_values, max_nnz).zero_();

  // NB: relies on nnz test above
  auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
  auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
  auto src_indices_accessor = src_indices.accessor<int64_t, 2>();

  // Check if we can find matching indices, and if so, write an
  // entry to the result indices vector.  Returns true if matching
  // indices were found.
  auto index_preamble = [&]() {
    for (auto d: c10::irange(sparse_dim)) {
      if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
        t_i++;
        return false;
      }
      if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
        s_i++;
        return false;
      }
    }
    for (auto d: c10::irange(sparse_dim)) {
      r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
    }
    return true;
  };

  if (t_values.dim() > 1) {
    while (t_i < t_nnz && s_i < s_nnz) {
      if (!index_preamble()) continue;
      r_buffer.select(0, r_i).addcmul_(t_values.select(0, t_i), s_values.select(0, s_i));
      r_i++;
      t_i++;
      s_i++;
    }
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::ComplexHalf, at::ScalarType::BFloat16, at::ScalarType::Half,
        commonDtype, "mul_out_sparse", [&] {
          auto r_accessor = r_buffer.accessor<scalar_t, 1>();
          auto t_accessor = t_values.accessor<scalar_t, 1>();
          auto s_accessor = s_values.accessor<scalar_t, 1>();

          while (t_i < t_nnz && s_i < s_nnz) {
            if (!index_preamble()) continue;
            r_accessor[r_i] = t_accessor[t_i] * s_accessor[s_i];
            r_i++;
            t_i++;
            s_i++;
          }
        }
    );
  }

  r.resize_as_(src);
  Tensor r_values = r_buffer.to(r.scalar_type());
  get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);
  get_sparse_impl(r)->set_nnz_and_narrow(r_i);
  return r._coalesced_(true);
}

// --------------------------------------------------------------------
// addmm(D1, S, D2, beta, alpha) -> D  [broadcasts]
//
// D = beta * D1 + alpha * mm(S, D2)
// --------------------------------------------------------------------

template <typename scalar_t>
static void s_addmm_out_sparse_dense_worker(int64_t nnz, int64_t dim_i, int64_t dim_j, int64_t dim_k, Tensor& r, const Scalar& beta, const Tensor& t, const Scalar& alpha, const Tensor& indices, const Tensor& values, const Tensor& dense) {

  // r_ = alpha * sparse * dense
  scalar_t cast_alpha = alpha.to<scalar_t>();
  scalar_t cast_beta = beta.to<scalar_t>();

  if (cast_beta == static_cast<scalar_t>(0)) {
    r.zero_();
  } else if (cast_beta == static_cast<scalar_t>(1)) {
    if (!is_same_tensor(r, t)) {
      r.copy_(t);
    }
  } else {
    at::mul_out(r, t, scalar_to_tensor(beta));
  }

  auto indices_accessor = indices.accessor<int64_t, 2>();

  auto values_accessor = values.accessor<scalar_t, 1>();
  scalar_t* dense_ptr = dense.data_ptr<scalar_t>();
  scalar_t* r_ptr = r.data_ptr<scalar_t>();

  int64_t dense_stride0 = dense.stride(0);
  int64_t dense_stride1 = dense.stride(1);
  int64_t r_stride0 = r.stride(0);
  int64_t r_stride1 = r.stride(1);
  for (auto i: c10::irange(nnz)) {
    scalar_t val = values_accessor[i];
    int64_t row = indices_accessor[0][i];
    int64_t col = indices_accessor[1][i];
    if (col >= 0 && col < dim_j && row >= 0 && row < dim_i) {
      // AXPY call is no-op over an empty vector
      if (dim_k == 0) {
        continue;
      }
      at::native::cpublas::axpy<scalar_t>(dim_k,
            cast_alpha * val,
            dense_ptr + col * dense_stride0, dense_stride1,
            r_ptr + row * r_stride0, r_stride1);
    } else {
      if (col < 0 || col >= dim_j) {
        TORCH_CHECK(false, "addmm: index out of column bound: ", col, " not between 1 and ", dim_j);
      } else {
        TORCH_CHECK(false, "addmm: index out of row bound: ", row, " not between 1 and ", dim_i);
      }
    }
  }
}

static Tensor& s_addmm_out_sparse_dense_cpu(
    Tensor& r,
    const Tensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha) {
  // TODO: This error message seems awfully opaque
  TORCH_CHECK(
      t.is_cpu(),
      "Expected all tensors to be on the same device. addmm expected 't' to be CPU tensor, but got tensor on ",
      t.device());
  TORCH_CHECK(
      r.is_cpu(),
      "Expected all tensors to be on the same device. addmm: expected 'out' to be CPU tensor, but got tensor on ",
      r.device());
  TORCH_CHECK(
      sparse_.is_cpu(),
      "Expected all tensors to be on the same device. addmm: expected 'mat1' to be a CPU tensor, but got tensor on ",
      sparse_.device());
  TORCH_CHECK(
      dense.is_cpu(),
      "Expected all tensors to be on the same device. addmm: expected 'mat2' to be a CPU tensor, but got tensor on ",
      dense.device());

  TORCH_CHECK(
      r.layout() == kStrided,
      "addmm_sparse_dense: expected strided result tensor, got tensor with layout ",
      r.layout());
  TORCH_CHECK(
      t.layout() == kStrided,
      "addmm_sparse_dense: expected 't' to have strided layout, got tensor with layout ",
      t.layout());
  TORCH_CHECK(
      sparse_.layout() == kSparse && dense.layout() == kStrided,
      "addmm_sparse_dense: expected either 'mat1' to have sparse layout and 'mat2' to have strided layout, got 'mat1' with layout ",
      sparse_.layout(),
      " and 'mat2' with layout ",
      dense.layout());

  TORCH_CHECK(sparse_.sparse_dim() == 2, "addmm: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse_.dense_dim() == 0, "addmm: scalar values expected, got ", sparse_.dense_dim(), "D values");
  T
```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 107 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/sparse`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/TensorIndexing.h`
- `ATen/native/sparse/SparseTensorMath.h`
- `c10/util/irange.h`
- `c10/util/MaybeOwned.h`
- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/native/sparse/SparseStubs.h`
- `ATen/Parallel.h`
- `ATen/SparseCsrTensorUtils.h`
- `ATen/SparseTensorImpl.h`
- `ATen/ExpandUtils.h`
- `ATen/ScalarOps.h`
- `ATen/InitialTensorOptions.h`
- `ATen/WrapDimUtilsMulti.h`
- `ATen/native/BinaryOps.h`
- `ATen/native/Copy.h`
- `ATen/native/CPUBlas.h`
- `ATen/native/SparseTensorUtils.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_sparse_addmm.h`
- `ATen/ops/_sparse_addmm_native.h`
- `ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h`
- `ATen/ops/_sparse_mm_native.h`
- `ATen/ops/_sparse_sum.h`
- `ATen/ops/_sparse_sum_backward_native.h`
- `ATen/ops/_sparse_sum_native.h`
- `ATen/ops/_sparse_sparse_matmul.h`
- `ATen/ops/_sparse_mm_reduce_impl.h`
- `ATen/ops/_sparse_mm_reduce_impl_native.h`


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

Files in the same folder (`aten/src/ATen/native/sparse`):

- [`SparseBinaryOpIntersectionCommon.h_docs.md`](./SparseBinaryOpIntersectionCommon.h_docs.md)
- [`SparseFactories.cpp_docs.md`](./SparseFactories.cpp_docs.md)
- [`ParamUtils.cpp_docs.md`](./ParamUtils.cpp_docs.md)
- [`SparseTensor.cpp_docs.md`](./SparseTensor.cpp_docs.md)
- [`ValidateCompressedIndicesKernel.cpp_docs.md`](./ValidateCompressedIndicesKernel.cpp_docs.md)
- [`SparseBlas.cpp_docs.md`](./SparseBlas.cpp_docs.md)
- [`SparseBlas.h_docs.md`](./SparseBlas.h_docs.md)
- [`SparseStubs.h_docs.md`](./SparseStubs.h_docs.md)
- [`SparseCsrTensorMath.h_docs.md`](./SparseCsrTensorMath.h_docs.md)
- [`SparseTensorMath.h_docs.md`](./SparseTensorMath.h_docs.md)


## Cross-References

- **File Documentation**: `SparseTensorMath.cpp_docs.md`
- **Keyword Index**: `SparseTensorMath.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
