# Documentation: `aten/src/ATen/native/LinearAlgebra.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/LinearAlgebra.cpp`
- **Size**: 139,009 bytes (135.75 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/int_mm_kernel.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkldnn/Matmul.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/cpu/Utils.h>
#include <c10/core/GradMode.h>
#include <c10/util/accumulate.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <variant>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_compute_linear_combination_native.h>
#include <ATen/ops/_convert_weight_to_int4pack_for_cpu_native.h>
#include <ATen/ops/_dyn_quant_matmul_4bit_native.h>
#include <ATen/ops/_dyn_quant_pack_4bit_weight_native.h>
#include <ATen/ops/_int_mm_native.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_det.h>
#include <ATen/ops/_linalg_det_native.h>
#include <ATen/ops/_linalg_slogdet.h>
#include <ATen/ops/_linalg_slogdet_native.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/_weight_int4pack_mm_for_cpu_native.h>
#include <ATen/ops/_weight_int8pack_mm_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/addbmm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addr.h>
#include <ATen/ops/addr_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/chain_matmul_native.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/det_native.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/diff.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/frobenius_norm_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/full.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/ger_native.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/inner_native.h>
#include <ATen/ops/is_complex_native.h>
#include <ATen/ops/is_floating_point_native.h>
#include <ATen/ops/kron_native.h>
#include <ATen/ops/linalg_cond.h>
#include <ATen/ops/linalg_cond_native.h>
#include <ATen/ops/linalg_det.h>
#include <ATen/ops/linalg_det_native.h>
#include <ATen/ops/linalg_diagonal_native.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_eigvalsh.h>
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_inv_ex.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_matmul_native.h>
#include <ATen/ops/linalg_matrix_exp.h>
#include <ATen/ops/linalg_matrix_exp_native.h>
#include <ATen/ops/linalg_matrix_norm.h>
#include <ATen/ops/linalg_matrix_norm_native.h>
#include <ATen/ops/linalg_matrix_power_native.h>
#include <ATen/ops/linalg_matrix_rank.h>
#include <ATen/ops/linalg_matrix_rank_native.h>
#include <ATen/ops/linalg_multi_dot_native.h>
#include <ATen/ops/linalg_norm.h>
#include <ATen/ops/linalg_norm_native.h>
#include <ATen/ops/linalg_pinv.h>
#include <ATen/ops/linalg_pinv_native.h>
#include <ATen/ops/linalg_slogdet.h>
#include <ATen/ops/linalg_slogdet_native.h>
#include <ATen/ops/linalg_solve.h>
#include <ATen/ops/linalg_svdvals.h>
#include <ATen/ops/linalg_tensorinv.h>
#include <ATen/ops/linalg_tensorinv_native.h>
#include <ATen/ops/linalg_tensorsolve.h>
#include <ATen/ops/linalg_tensorsolve_native.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/linalg_vector_norm_native.h>
#include <ATen/ops/log2.h>
#include <ATen/ops/logdet_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/ops/matrix_exp_backward_native.h>
#include <ATen/ops/matrix_exp_native.h>
#include <ATen/ops/matrix_power_native.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/movedim.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mv.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/norm.h>
#include <ATen/ops/nuclear_norm_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/outer.h>
#include <ATen/ops/outer_native.h>
#include <ATen/ops/pinverse_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/real.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/slogdet_native.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/tensordot.h>
#include <ATen/ops/unique_consecutive.h>
#include <ATen/ops/vdot_native.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif

namespace at {

namespace detail {
  static void check_linalg_norm_dtype(std::optional<ScalarType> opt_dtype, ScalarType self_dtype, const char* const name) {
    if (opt_dtype.has_value()) {
      auto dtype = opt_dtype.value();
      TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype), name, ": dtype should"
          " be floating point or complex, but got ", dtype);
      TORCH_CHECK(isComplexType(self_dtype) == isComplexType(dtype),
          name, ": dtype should be ", isComplexType(self_dtype) ? "complex" : "real",
          " for ", isComplexType(self_dtype) ? "complex" : "real", " inputs, but got ", dtype);
      TORCH_CHECK(promoteTypes(self_dtype, dtype) == dtype,
          name, ": the dtype of the input ", "(", self_dtype, ") should be convertible ",
          "without narrowing to the specified dtype (", dtype, ")");
    }
  }
}

namespace meta {

#define ADDMM_META() \
  TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "self and mat2 must have the same dtype, but got ", self.scalar_type(), " and ", mat2.scalar_type()); \
  TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "mat1 and mat2 must have the same dtype, but got ", mat1.scalar_type(), " and ", mat2.scalar_type()); \
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor"); \
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor"); \
  TORCH_CHECK( \
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (", \
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")"); \
 \
  auto names = at::namedinference::propagate_names_for_addmm(mat1, mat2, self); \
  set_output_raw_strided(0, {mat1.sizes()[0], mat2.sizes()[1]}, {}, mat1.options(), names);

TORCH_META_FUNC(addmm)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  ADDMM_META();
}

TORCH_META_FUNC(_addmm_activation)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, bool use_gelu) {
  ADDMM_META();
}

TORCH_META_FUNC(mm)(const Tensor & self, const Tensor & mat2) {
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      self.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0], "x", self.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  auto names = at::namedinference::compute_matmul_outnames(self, mat2);
  set_output_raw_strided(0, {self.sizes()[0], mat2.sizes()[1]}, {}, self.options(), names);
}

TORCH_META_FUNC(linalg_vector_norm)(const Tensor& self, const Scalar& scalar_ord, OptionalIntArrayRef opt_dim, bool keepdim, std::optional<ScalarType> opt_dtype) {
  at::native::checkFloatingOrComplex(self, "linalg.vector_norm");
  TORCH_CHECK(!at::isComplexType(scalar_ord.type()), "linalg.vector_norm: Expected a non-complex scalar as the order of norm.");

  auto dim = opt_dim.value_or(IntArrayRef{});
  // Casting a large integer to a double will just introduce an error for
  // values larger than 10^53 (same for negative numbers), so that's fine.
  auto ord = scalar_ord.toDouble();

  // For more context, see issue 52783
  // If the tensor is empty and norm < 0 || norm == infty
  //   - We cannot reduce the whole tensor
  //   - We cannot reduce over an empty dimension
  if (self.numel() == 0 && (ord < 0. || ord == INFINITY)) {
    // dim=None or dim=() reduces the whole tensor
    TORCH_CHECK(opt_dim.has_value() && !opt_dim->empty(),
      "linalg.vector_norm cannot compute the ", scalar_ord, " norm on an empty ",
      "tensor because the operation does not have an identity");
    for (auto dim_num : dim) {
      TORCH_CHECK(self.size(dim_num) != 0,
        "linalg.vector_norm cannot compute the ", scalar_ord, " norm on the dimension ", dim_num ,
        "because this dimension is empty and the operation does not have an identity");
    }
  }

  at::detail::check_linalg_norm_dtype(opt_dtype, self.scalar_type(), "linalg.vector_norm");

  auto mask = at::native::make_dim_mask(dim, self.dim());
  auto shape = at::native::shape_from_dim_mask(self, std::move(mask), keepdim);
  auto options = self.options()
                     .dtype(toRealValueType(opt_dtype.value_or(self.scalar_type())));

  set_output_raw_strided(0, shape, {}, options);
}

TORCH_META_FUNC(_linalg_det)(const Tensor& A) {
  at::native::squareCheckInputs(A, "linalg.det");
  at::native::checkFloatingOrComplex(A, "linalg.det");

  auto shape = A.sizes();
  auto ndim = shape.size();

  // det
  set_output_contiguous(0, shape.slice(0, ndim - 2), A.options());

  // LU
  auto LU_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_output_strided(1, shape, LU_strides, A.options());

  // pivots
  set_output_contiguous(2, shape.slice(0, ndim - 1), A.options().dtype(kInt));
}

TORCH_META_FUNC(_linalg_slogdet)(const Tensor& A) {
  at::native::squareCheckInputs(A, "linalg.slogdet");
  at::native::checkFloatingOrComplex(A, "linalg.slogdet", /*low_precision*/false);

  auto shape= A.sizes();
  auto ndim = shape.size();

  auto shape_outputs = shape.slice(0, ndim - 2);

  // sign
  set_output_contiguous(0, shape_outputs, A.options());

  // logabsdet
  set_output_contiguous(1, shape_outputs, A.options().dtype(toRealValueType(A.scalar_type())));

  // LU
  auto LU_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_output_strided(2, shape, LU_strides, A.options());

  // pivots
  set_output_contiguous(3, shape.slice(0, ndim - 1), A.options().dtype(kInt));
}

template <typename Meta>
static void common_checks_baddbmm_bmm(Meta& meta, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, bool is_bmm, const std::optional<Tensor>& self_baddbmm = std::nullopt) {
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");

  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();

  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];
  std::vector<int64_t> output_size {bs, res_rows, res_cols};

  TORCH_CHECK(batch2_sizes[0] == bs && batch2_sizes[1] == contraction_size,
              "Expected size for first two dimensions of batch2 tensor to be: [",
              bs, ", ", contraction_size, "] but got: [", batch2_sizes[0], ", ", batch2_sizes[1], "].");

  auto& result = meta.maybe_get_output(0);
  // 'set_output' does not resize for in-place calls
  meta.set_output_raw_strided(0, output_size, {}, batch2.options());
  const auto result_sizes = result.sizes();
  // Error is raised if called from in-place overload with incorrect shape
  TORCH_CHECK(result_sizes == output_size,
              "Expected an output tensor with shape [", output_size, "] but got shape ", result_sizes);

  std::vector<Dimname> outnames = {};
  if (!is_bmm) {
    if (self_baddbmm.has_value()) {
      const auto& self = self_baddbmm.value();
      if (beta.toComplexDouble() != 0.0) result.copy_(self);
      TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
      const auto self_sizes = self.sizes();
      TORCH_CHECK(self_sizes == output_size,
                  "Expected an input tensor shape with shape ", output_size, " but got shape: ", self_sizes);
      outnames = namedinference::compute_baddbmm_outnames(result, batch1, batch2, self);
    }
  } else {
    outnames = namedinference::compute_bmm_outnames(result, batch1, batch2);
  }

  namedinference::propagate_names_if_nonempty(
    result,
    outnames
  );
}

TORCH_META_FUNC(bmm)(const Tensor& self, const Tensor& mat2) {
    common_checks_baddbmm_bmm(*this, self, mat2, Scalar(0.0), Scalar(1.0), true);
}

TORCH_META_FUNC(baddbmm)(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  auto self_ = expand_size(self, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm");
  TORCH_CHECK(self.dtype() == batch1.dtype(), "Input dtypes must be the same, got: input ", self.dtype(), ", batch1: ", batch1.dtype(), ", batch2: ", batch2.dtype());
  common_checks_baddbmm_bmm(*this, batch1, batch2, beta, alpha, false, *self_);
}

} // namespace meta
namespace native {

DEFINE_DISPATCH(addr_stub);


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.det ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// As P is a permutation matrix
// det(P) = 1 if it's an even permutation and det(P) = -1 if it's an odd permutation
static Tensor lu_det_P(const Tensor& pivots) {
  return (at::arange(1, pivots.size(-1) + 1, pivots.options()) != pivots)
    .sum(-1, /*keepdim=*/false, /*dtype=*/at::kLong)
    .fmod_(2)
    // take 0 to 1 and 1 to -1
    .mul_(-2)
    .add_(1);
}

// Auxiliary function that returns the LU decomposition to use it in the backward
TORCH_IMPL_FUNC(_linalg_det_out)(const Tensor& A, const Tensor& result, const Tensor& LU, const Tensor& pivots) {
  // info is an aux tensor
  auto info = at::empty({0}, A.options().dtype(kInt));
  // Optimisation: lu_factor_ex requires the input to be F-contig, otherwise it copies
  // Use the transpose of if A is contiguous since det(A^T) = det(A)
  // We limit this to real matrices, but it could also be implemented for complex matrices
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(LU), const_cast<Tensor&>(pivots), const_cast<Tensor&>(info), A.is_contiguous() && !A.is_complex() ? A.mH() : A);

  // det = det_P * prod(diag(LU))
  at::mul_out(const_cast<Tensor&>(result), lu_det_P(pivots), at::prod(LU.diagonal(0, -2 ,-1), /*dim=*/-1));
}

Tensor linalg_det(const Tensor& A) {
  return std::get<0>(at::_linalg_det(A));
}

Tensor& linalg_det_out(const Tensor& A, Tensor& result) {
  auto LU = at::empty({0}, A.options());
  auto pivots = at::empty({0}, A.options().dtype(kInt));
  at::_linalg_det_out(result, LU, pivots, A);
  return result;
}

// torch.det, alias for torch.linalg.det
Tensor det(const Tensor& self) {
  return at::linalg_det(self);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.slogdet ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Auxiliary function that returns the LU decomposition to use it in the backward
TORCH_IMPL_FUNC(_linalg_slogdet_out)(const Tensor& A, const Tensor& sign, const Tensor& logabsdet, const Tensor& LU, const Tensor& pivots) {
  // info is an aux tensor
  auto info = at::empty({0}, A.options().dtype(kInt));
  // Optimisation: lu_factor_ex requires the input to be F-contig, otherwise it copies
  // Use the transpose of if A is contiguous since det(A^T) = det(A)
  // We limit this to real matrices, but it could also be implemented for complex matrices
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(LU), const_cast<Tensor&>(pivots), const_cast<Tensor&>(info), A.is_contiguous() && !A.is_complex() ? A.mH() : A);

  auto diag_U = LU.diagonal(0, -2, -1);
  // sign
  at::mul_out(const_cast<Tensor&>(sign), diag_U.sgn().prod(-1), lu_det_P(pivots));

  // logabsdet
  at::sum_out(const_cast<Tensor&>(logabsdet), diag_U.abs().log_(), -1);
}

std::tuple<Tensor, Tensor> linalg_slogdet(const Tensor& A) {
  auto out = at::_linalg_slogdet(A);
  return std::make_tuple(std::move(std::get<0>(out)), std::move(std::get<1>(out)));
}

std::tuple<Tensor&, Tensor&> linalg_slogdet_out(const Tensor& A, Tensor& sign, Tensor& logabsdet) {
  auto LU = at::empty({0}, A.options());
  auto pivots = at::empty({0}, A.options().dtype(kInt));
  at::_linalg_slogdet_out(sign, logabsdet, LU, pivots, A);
  return std::tie(sign, logabsdet);
}

// Alias
std::tuple<Tensor, Tensor> slogdet(const Tensor& A) {
  return at::linalg_slogdet(A);
}

std::tuple<Tensor&, Tensor&> slogdet_out(const Tensor& A, Tensor& sign, Tensor& logabsdet) {
  return at::linalg_slogdet_out(sign, logabsdet, A);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ logdet ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor logdet(const Tensor& A) {
  squareCheckInputs(A, "logdet");
  checkFloatingOrComplex(A, "logdet", /*low_precision*/false);
  auto [sign, logabsdet] = at::linalg_slogdet(A);

  if (A.is_complex()) {
    return sign.log() + logabsdet;
  } else {
    return at::where(sign == -1., NAN, logabsdet);
  }
}

namespace {

// This function extracts the optional Tensors for atol and rtol
// Default value for atol is zero
// Default value for rtol is eps*max(rows, cols)
// If atol is specified and rtol is not specified then default value for rtol is zero
// It is used for matrix_rank and pinv
std::tuple<Tensor, Tensor> get_atol_rtol(
    const Tensor& input,
    const std::optional<Tensor>& atol_opt,
    const std::optional<Tensor>& rtol_opt,
    const std::string_view function_name) {
  auto options = input.options();
  if (input.device().type() == kMetal || input.device().type() == kMPS) {
    options = options.dtype(ScalarType::Float);
  } else {
    options = options.dtype(ScalarType::Double);
  }
  auto atol = atol_opt.has_value() ? atol_opt.value() : at::zeros({}, options);
  checkNotComplexTolerance(atol, function_name, "atol");
  Tensor rtol;
  if (rtol_opt.has_value()) {
    rtol = rtol_opt.value();
    checkNotComplexTolerance(rtol, function_name, "rtol");
  } else {
    ScalarType real_dtype = toRealValueType(input.scalar_type());
    auto default_rtol = at::full({}, _get_epsilon(real_dtype) * std::max(input.sym_size(-1), input.sym_size(-2)), options);
    rtol = atol_opt.has_value()
           ? at::where(atol_opt.value() > 0, at::zeros({}, options), default_rtol)
           : std::move(default_rtol);
  }
  return std::make_tuple(atol, rtol);
}

std::tuple<Tensor, Tensor> get_atol_rtol(
    const Tensor& input,
    std::optional<double> atol_opt,
    std::optional<double> rtol_opt) {
  auto atol = atol_opt.has_value() ? atol_opt.value() : 0.0;
  c10::SymFloat rtol;
  if (rtol_opt.has_value()) {
    rtol = rtol_opt.value();
  } else {
    ScalarType real_dtype = toRealValueType(input.scalar_type());
    auto default_rtol = _get_epsilon(real_dtype) * std::max(input.sym_size(-1), input.sym_size(-2));
    rtol = (atol_opt.has_value() && atol_opt.value() > 0.0)
           ? 0.0
           : default_rtol;
  }
  auto options = input.options();
  if (input.device().type() == kMetal || input.device().type() == kMPS) {
    options = options.dtype(ScalarType::Float);
  } else {
    options = options.dtype(ScalarType::Double);
  }
  auto atol_tensor = at::full({}, atol, options);
  auto rtol_tensor = at::full({}, rtol, options);
  return std::make_tuple(atol_tensor, rtol_tensor);
}

} // anonymous namespace

Tensor linalg_pinv(
    const Tensor& input,
    const std::optional<Tensor>& atol_opt,
    const std::optional<Tensor>& rtol_opt,
    bool hermitian) {
  // FIXME: Whenever we have a nice lstsq, we should dispatch this function to simply be
  // `torch.lstsq(A, torch.eye(A.shape[-1]), atol=atol, rtol=rtol)`
  // with a driver that supports singular inputs
  NoTF32Guard disable_tf32;
  ScalarType t = input.scalar_type();
  TORCH_CHECK((t == ScalarType::Double || t == ScalarType::Float || t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble)
              && input.dim() >= 2,
              "linalg.pinv(", t, "{", input.sizes(), "}): expected a tensor with 2 or more dimensions "
              "of float, double, cfloat or cdouble types");

  auto [atol, rtol] = get_atol_rtol(input, atol_opt, rtol_opt, "torch.linalg.pinv");

  if (input.sym_numel() == 0) {
    // The implementation below uses operations that do not work for zero numel tensors
    // therefore we need this early return for 'input.numel() == 0' case
    // TODO: replace input.svd with linalg_svd when torch/xla can work with at::linalg_svd
    auto [U, S, V] = input.svd();
    return at::matmul(V * S.reciprocal().unsqueeze(-2), U.mH());
  }

  // If not Hermitian use singular value decomposition, else use eigenvalue decomposition
  if (!hermitian) {
    // TODO: replace input.svd with linalg_svd
    // using linalg_svd breaks pytorch/xla, see https://github.com/pytorch/xla/issues/2755
    auto [U, S, V] = input.svd();
    Tensor max_val = at::narrow(S, /*dim=*/-1, /*start=*/0, /*length=*/1);  // singular values are sorted in descending order
    Tensor tol = at::max(atol.unsqueeze(-1), rtol.unsqueeze(-1) * max_val);
    Tensor S_pseudoinv = at::where(S > tol, S.reciprocal(), at::zeros({}, S.options())).to(input.dtype());
    // computes V @ diag(S_pseudoinv) @ U.conj().T
    return at::matmul(V * S_pseudoinv.unsqueeze(-2), U.mH());
  } else {
    auto [S, U] = at::linalg_eigh(input);
    // For Hermitian matrices, singular values equal to abs(eigenvalues)
    Tensor S_abs = S.abs();
    // eigenvalues are sorted in ascending order starting with negative values, we need a maximum value of abs(eigenvalues)
    Tensor max_val = S_abs.amax(/*dim=*/-1, /*keepdim=*/true);
    Tensor tol = at::max(atol.unsqueeze(-1), rtol.unsqueeze(-1) * max_val);
    Tensor S_pseudoinv = at::where(S_abs > tol, S.reciprocal(), at::zeros({}, S.options())).to(input.dtype());
    // computes U @ diag(S_pseudoinv) @ U.conj().T
    return at::matmul(U * S_pseudoinv.unsqueeze(-2), U.mH());
  }
}

Tensor linalg_pinv(const Tensor& input, std::optional<double> atol, std::optional<double> rtol, bool hermitian) {
  auto [atol_tensor, rtol_tensor] = get_atol_rtol(input, atol, rtol);
  return at::linalg_pinv(input, atol_tensor, rtol_tensor, hermitian);
}

Tensor linalg_pinv(const Tensor& input, const Tensor& rcond, bool hermitian) {
  // For NumPy compatibility the rcond argument is used as relative tolerance
  checkNotComplexTolerance(rcond, "torch.linalg.pinv", "rcond");
  auto options = input.options();
  if (input.device().type() == kMetal || input.device().type() == kMPS) {
    options = options.dtype(ScalarType::Float);
  } else {
    options = options.dtype(ScalarType::Double);
  }
  return at::linalg_pinv(input, at::zeros({}, options), rcond, hermitian);
}

Tensor linalg_pinv(const Tensor& input, double rcond, bool hermitian) {
  // For NumPy compatibility the rcond argument is used as relative tolerance
  return at::linalg_pinv(input, 0.0, rcond, hermitian);
}

// TODO: implement _out variant avoiding copy and using already allocated storage directly
Tensor& linalg_pinv_out(
    const Tensor& input,
    const std::optional<Tensor>& atol,
    const std::optional<Tensor>& rtol,
    bool hermitian,
    Tensor& result) {
  checkSameDevice("linalg.pinv", result, input);
  checkLinalgCompatibleDtype("linalg.pinv", result, input);
  Tensor result_tmp = at::linalg_pinv(input, atol, rtol, hermitian);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor& linalg_pinv_out(
    const Tensor& input,
    std::optional<double> atol,
    std::optional<double> rtol,
    bool hermitian,
    Tensor& result) {
  checkSameDevice("linalg.pinv", result, input);
  checkLinalgCompatibleDtype("linalg.pinv", result, input);
  Tensor result_tmp = at::linalg_pinv(input, atol, rtol, hermitian);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor& linalg_pinv_out(const Tensor& input, const Tensor& rcond, bool hermitian, Tensor& result) {
  checkSameDevice("linalg.pinv", result, input);
  checkLinalgCompatibleDtype("linalg.pinv", result, input);

  Tensor result_tmp = at::linalg_pinv(input, rcond, hermitian);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor& linalg_pinv_out(const Tensor& input, double rcond, bool hermitian, Tensor& result) {
  Tensor rcond_tensor = at::full({}, rcond, input.options().dtype(ScalarType::Double));
  return at::linalg_pinv_out(result, input, rcond_tensor, hermitian);
}

Tensor pinverse(const Tensor& self, double rcond) {
  return at::linalg_pinv(self, rcond, /*hermitian=*/false);
}

// matrix_power implementation
namespace {

/**
 * @brief Raises the input matrix to the given power n
 *
 * If the exponent n is negative, the inverse of the input
 * matrix will be raised to power abs(n).
 *
 * @param self (batched) square matrix to raise to power n
 * @param n exponent to raise matrix (or matrices in batch) to
 * @param _out optional tensor to write the output to
 * @return Tensor input matrix raised to power n
 */
Tensor linalg_matrix_power_impl(
    const Tensor& self,
    int64_t n,
    std::optional<Tensor> _out) {
  NoTF32Guard disable_tf32;
  auto out = _out.value_or(Tensor());

  squareCheckInputs(self, "linalg.matrix_power");
  if (_out.has_value()) {
    checkSameDevice("matrix_power", out, self);
    checkLinalgCompatibleDtype("matrix_power", out, self);
    at::native::resize_output_symint(out, self.sym_sizes());
  }

  // For n=0 we return the identity matrix of the same shape as input.
  if (n == 0) {
    if (!_out.has_value()) {
      // Clone input to include result in the autograd graph
      out = self.clone(at::MemoryFormat::Contiguous);
    }
    return out.copy_(at::eye_symint(self.sym_size(-2), self.options()));
  }
  if (n == 1) {
    return _out.has_value() ? out.copy_(self)
                            : self.clone(at::MemoryFormat::Contiguous);
  }
  if (n == -1) {
    return _out.has_value() ? at::linalg_inv_out(out, self)
                            : at::linalg_inv(self);
  }

  // For negative n we inverte the input matrix before raising to power abs(n)
  auto a = n < 0 ? at::linalg_inv(self) : self;
  n = std::abs(n);

  // Fast paths for small powers
  if (n == 2) {
    return _out.has_value() ? at::matmul_out(out, a, a) : at::matmul(a, a);
  }
  if (n == 3) {
    return _out.has_value() ? at::matmul_out(out, at::matmul(a, a), a)
                            : at::matmul(at::matmul(a, a), a);
  }

  // This is a binary decomposition of n.
  // Moving from the least significant bit to the most significant bit
  // This is done to reduce the number of matrix multiplications
  // by raising the input matrix in powers of 2
  // The total number of matrix multiplications are
  // number of bits + number of bits that equal 1 ~ O(log n)
  // instead of O(n)
  Tensor z, result;
  while (n > 0) {
    const auto bit = n % 2;
    n = n / 2;
    z = z.defined() ? at::matmul(z, z) : a;
    if (bit == 1) {
      if (_out.has_value() && n <= 0) {
        // Last multiplication can use the out version
        return result.defined() ? at::matmul_out(out, result, z) : out.copy_(z);
      }
      result = result.defined() ? at::matmul(result, z) : z;
    }
  }

  return result;
}

} // namespace

Tensor& linalg_matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  linalg_matrix_power_impl(self, n, result);
  return result;
}

Tensor linalg_matrix_power(const Tensor& self, int64_t n) {
  return linalg_matrix_power_impl(self, n, std::nullopt);
}

Tensor& matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  return at::native::linalg_matrix_power_out(self, n, result);
}

Tensor matrix_power(const Tensor& self, int64_t n) {
  return at::native::linalg_matrix_power(self, n);
}

namespace {

// Computes the rank of 'input' and saves the result in-place in 'result'.
// 'hermitian' controls whether SVD or eigendecomposition is used for computing the singular values
// 'atol' and 'rtol' are the absolute and relative tolerances, respectively.
Tensor& matrix_rank_impl(
    const Tensor& input,
    const std::optional<Tensor>& atol_opt,
    const std::optional<Tensor>& rtol_opt,
    bool hermitian,
    Tensor& result) {
  auto [atol, rtol] = get_atol_rtol(input, atol_opt, rtol_opt, "torch.linalg.matrix_rank");

  checkSameDevice("torch.linalg.matrix_rank", result, input);
  checkSameDevice("torch.linalg.matrix_rank", atol, input, "atol");
  checkSameDevice("torch.linalg.matrix_rank", rtol, input, "rtol");
  ScalarType output_type = ScalarType::Long;
  checkLinalgCompatibleDtype("torch.linalg.matrix_rank", result.scalar_type(), output_type);

  checkNotComplexTolerance(atol, "torch.linalg.matrix_rank", "atol");
  checkNotComplexTolerance(rtol, "torch.linalg.matrix_rank", "rtol");

  // NumPy doesn't take into account possible input with no elements and it errors on max not defined for this case
  // Let's output 0 for this case, since that kind of matrices have zero number of non-zero rows, hence rank is 0.
  if (input.sym_numel() == 0) {
    result.fill_(0);
    return result;
  }

  // We compute matrix rank as the number of singular or absolute eigen values
  // that are above max(atol, rtol * max(S)) threshold
  Tensor S, max_S;
  if (!hermitian) {
    S = at::linalg_svdvals(input);
    // singular values are sorted in descending order
    max_S = at::narrow(S, /*dim=*/-1, /*start=*/0, /*length=*/1);
  } else {
    S = at::linalg_eigvalsh(input);
    S = S.abs();
    // eigenvalues are sorted in ascending order starting with negative values, we need a maximum value of abs(eigenvalues)
    max_S = S.amax(/*dim=*/-1, /*keepdim=*/true);
  }

  Tensor tol = at::max(atol.unsqueeze(-1), rtol.unsqueeze(-1) * max_S);

  if (isTensorSubclassLike(input)) {
     result = at::sum(S > tol, /*dim=*/-1);
     return result;
  }

  result = at::sum_out(result, S > tol, /*dim=*/-1);
  return result;
}

Tensor get_matrix_rank_result_tensor(const Tensor& input) {
  // Matrices or batch of matrices are allowed
  checkIsMatrix(input, "torch.linalg.matrix_rank", "input");
  // For Composite Compliance, allocate `result` of correct shape to
  // avoid resizing in `out` variant.
  // See also `NOTE [matrix rank output shape]`
  auto result_shape =
      SymIntArrayRef(input.sym_sizes().cbegin(), input.sym_sizes().cend() - 2);
  Tensor result =
      at::empty_symint(result_shape, input.options().dtype(ScalarType::Long));

  return result;
}

}  // anonymous namespace

Tensor& linalg_matrix_rank_out(
    const Tensor& input,
    const std::optional<Tensor>& atol_opt,
    const std::optional<Tensor>& rtol_opt,
    bool hermitian,
    Tensor& result) {
  // Matrices or batch of matrices are allowed
  checkIsMatrix(input, "torch.linalg.matrix_rank", "input");
  auto result_shape =
    IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2);
  at::native::resize_output(result, result_shape);
  return matrix_rank_impl(input, atol_opt, rtol_opt, hermitian, result);
}

Tensor& linalg_matrix_rank_out(const Tensor& input, std::optional<double> atol, std::optional<double> rtol, bool hermitian, Tensor& result) {
  auto [atol_tensor, rtol_tensor] = get_atol_rtol(input, atol, rtol);
  result = linalg_matrix_rank_out(input, atol_tensor, rtol_tensor, hermitian, result);
  return result;
}

Tensor linalg_matrix_rank(const Tensor& input, const std::optional<Tensor>& atol, const std::optional<Tensor>& rtol, bool hermitian) {
  auto result = get_matrix_rank_result_tensor(input);
  return matrix_rank_impl(input, atol, rtol, hermitian, result);
}

Tensor linalg_matrix_rank(const Tensor& input, std::optional<double> atol, std::optional<double> rtol, bool hermitian) {
  auto result = get_matrix_rank_result_tensor(input);

  auto [atol_tensor, rtol_tensor] = get_atol_rtol(input, atol, rtol);

  return matrix_rank_impl(input, atol_tensor, rtol_tensor, hermitian, result);
}

Tensor& linalg_matrix_rank_out(const Tensor& input, const Tensor& tol, bool hermitian, Tensor& result) {
  // For NumPy compatibility tol is not scaled with max(singular_value) if the value for tol is provided
  // It is assumed that the provided value is the absolute tolerance
  Tensor rtol = at::zeros({}, tol.options());
  result = at::linalg_matrix_rank_outf(input, tol, rtol, hermitian, result);
  return result;
}

Tensor& linalg_matrix_rank_out(const Tensor& input, double tol, bool hermitian, Tensor& result) {
  // For NumPy compatibility tol is not scaled with max(singular_value) if the value for tol is provided
  // It is assumed that the provided value is the absolute tolerance
  result = at::linalg_matrix_rank_outf(input, tol, 0.0, hermitian, result);
  return result;
}

Tensor linalg_matrix_rank(const Tensor& input, const Tensor& tol, bool hermitian) {
  auto result = get_matrix_rank_result_tensor(input);
  return matrix_rank_impl(input, tol, at::zeros({}, tol.options()), hermitian, result);
}

Tensor linalg_matrix_rank(const Tensor& input, double tol, bool hermitian) {
  auto result = get_matrix_rank_result_tensor(input);

  auto [atol_tensor, rtol_tensor] = get_atol_rtol(input, tol, 0.0);

  return matrix_rank_impl(input, atol_tensor, rtol_tensor, hermitian, result);
}

// multi_dot helper functions
namespace {

/**
 * @brief Computes the optimal matrix chain multiplication order
 *
 * Follows the dynamic programming algorithm from Cormen et al.,
 * "Introduction to Algorithms, Third Edition", Chapter 15.2,
 * p. 370-378. Note that the book uses 1-based indexing.
 *
 * The cost of multiplying two matrices with sizes p x q and q x r
 * is defined here as p * q * r. The optimal multiplication order
 * is the one that minimizes the total cost.
 *
 * @param tensors list of 2D tensors
 * @return a 2D vector s used by #matrix_chain_multiplication to construct
 *         the optimal matrix multiplication order. The optimal multiplication
 *         order for multiplying tensors i...j is to multiply tensors i...s[i, j]
 *         and tensors (s[i, j] + 1)...j first and then the result of that.
 */
std::vector<std::vector<int64_t>> matrix_chain_order(TensorList tensors) {
  const size_t n = tensors.size();

  // Tensor i has dimensions p[i] x p[i + 1]
  std::vector<int64_t> p(n + 1);
  for (const auto i : c10::irange(n)) {
    p[i] = tensors[i].size(0);
  }
  p[n] = tensors[n - 1].size(1);

  // m[i, j] = k where k is the minimum cost for multiplying tensors i...j
  std::vector<std::vector<int64_t>> m(n, std::vector<int64_t>(n, 0));

  // s[i, j] = k where k is the index at which to split the list such that
  // optimally multiplying matrices i...k and k...j first and then the resulting
  // matrices is the optimal order for multiplying matrices i...j.
  std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n));

  // Compute the optimal multiplication order
  for (const auto l : c10::irange(1, n)) {
    for (const auto i : c10::irange(n - l)) {
      const auto j = i + l;
      m[i][j] = std::numeric_limits<int64_t>::max();
      for (const auto k : c10::irange(i, j)) {
        const auto q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
        if (q < m[i][j]) {
          m[i][j] = q;
          s[i][j] = k;
        }
      }
    }
  }

  return s;
}

/**
 * @brief Recursively multiplies the tensors i...j using the given order
 *
 * @param tensors matrices to multiply together
 * @param order optimal chain multiplication order from #matrix_chain_order
 * @param i index of first tensor to be multiplied
 * @param j index of last tensor to be multiplied
 * @return Tensor result of multiplying tensors[i...j] together.
 */
Tensor matrix_chain_multiplication(
    TensorList tensors,
    const std::vector<std::vector<int64_t>>& order,
    int64_t i,
    int64_t j) {
  if (i == j) {
    return tensors[i];
  }
  return at::mm(
      matrix_chain_multiplication(tensors, order, i, order[i][j]),
      matrix_chain_multiplication(tensors, order, order[i][j] + 1, j));
}

// Implements torch.linalg.multi_dot
Tensor multi_dot_impl(TensorList _tensors, std::optional<Tensor> _out) {
  const size_t n = _tensors.size();
  TORCH_CHECK(n >= 2, "multi_dot(): expected at least 2 tensors but got ", n);

  std::vector<int64_t> out_shape;
  std::vector<Tensor> tensors(n);

  // If the first tensor is 1D of size n view it as a row vector (1, n)
  if (_tensors[0].dim() == 1) {
    tensors[0] = _tensors[0].unsqueeze(0);
  } else if (_tensors[0].dim() == 2) {
    tensors[0] = _tensors[0];
    out_shape.emplace_back(tensors[0].size(0));
  } else {
    TORCH_CHECK(
        false,
        "multi_dot(): the first tensor must be 1D or 2D but got ",
        _tensors[0].dim(),
        "D");
  }

  // If the last tensor is 1D of size n view it as a column vector (n, 1)
  if (_tensors[n - 1].dim() == 1) {
    tensors[n - 1] = _tensors[n - 1].unsqueeze(-1);
  } else if (_tensors[n - 1].dim() == 2) {
    tensors[n - 1] = _tensors[n - 1];
    out_shape.emplace_back(tensors[n - 1].size(1));
  } else {
    TORCH_CHECK(
        false,
        "multi_dot(): the last tensor must be 1D or 2D but got ",
        _tensors[n - 1].dim(),
        "D");
  }

  // Ensure middle tensors are 2D
  for (const auto i : c10::irange(1, n - 1)) {
    TORCH_CHECK(
        _tensors[i].dim() == 2,
        "multi_dot(): tensor ",
        i,
        " must be 2D but got ",
        _tensors[i].dim(),
        "D");
    tensors[i] = _tensors[i];
  }

  // Ensure all tensors have the same device and dtype and check
  // that the shapes can be multiplied
  const auto dtype = tensors[0].dtype();
  const auto device = tensors[0].device();
  for (const auto i : c10::irange(1, n)) {
    TORCH_CHECK(
        tensors[i].dtype() == dtype,
        "multi_dot(): all tensors must have be the same dtype but tensor 0 is ",
        dtype,
        " and tensor ",
        i,
        " ",
        tensors[i].dtype());
    TORCH_CHECK(
        tensors[i].device() == device,
        "multi_dot(): all tensors must be on the same device but tensor 0 is on ",
        device,
        " and tensor ",
        i,
        " on ",
        tensors[i].device());
    TORCH_CHECK(
        tensors[i - 1].size(-1) == tensors[i].size(0),
        "multi_dot(): tensors ",
        i - 1,
        " and ",
        i,
        " with shapes ",
        _tensors[i - 1].sizes(),
        " and ",
        _tensors[i].sizes(),
        " cannot be multiplied")
  }

  Tensor result;

  if (_out.has_value()) {
    auto out = *_out;
    TORCH_CHECK(
        dtype == out.dtype(),
        "multi_dot(): expected out tensor to have dtype ",
        dtype,
        " but got ",
        out.dtype());
    TORCH_CHECK(
        device == out.device(),
        "multi_dot(): expected out tensor to be on device ",
        device,
        " but got ",
        out.device());

    // If the last and last tensors have shapes (a, b) and (b, c) the
    // output has shape (a, c). If either the first or last tensor is 1D
    // a and/or c dimensions will be implicitly size 1 and will be omitted
    // from the output. e.g. for inputs (a, b) x (b) the output has shape (a,).
    at::native::resize_output(out, out_shape);

    // View output as 2D for simplicity of computation.
    result = out.view({tensors[0].size(0), tensors.back().size(-1)});
  }

  // The resize_ and view calls below are to ensure the
  // output shape respects the original dimensionality of
  // the first and last tensors which we are now viewed as 2D

  if (tensors.size() == 2) {
    return _out.has_value() ? at::mm_out(result, tensors[0], tensors[1])
                         : at::mm(tensors[0], tensors[1]).view(out_shape);
  }

  // Why the separate implementation for 3 matrices?
  // The logic for three matrices is much faster when done directly
  // Requires 1 comparison to 4 comparisons and fewer arithmetic operations
  if (tensors.size() == 3) {
    const auto a = tensors[0].size(0);
    const auto b = tensors[1].size(0);
    const auto c = tensors[2].size(0);
    const auto d = tensors[2].size(1);

    // The matrices are of size (a x b), (b x c), (c x d)
    // cost_1 is the cost of parenthesizing (a x b) and (b x c) and then
    // combining (c x d) cost_2 is the cost of parenthesizing (b x c) and (c x
    // d) and then combining (a x b)
    const auto cost_1 = (a * c) * (b + d);
    const auto cost_2 = (b * d) * (a + c);

    if (cost_1 > cost_2) {
      return _out.has_value()
          ? at::mm_out(result, tensors[0], at::mm(tensors[1], tensors[2]))
          : at::mm(tensors[0], at::mm(tensors[1], tensors[2])).view(out_shape);
    } else {
      return _out.has_value()
          ? at::mm_out(result, at::mm(tensors[0], tensors[1]), tensors[2])
          : at::mm(at::mm(tensors[0], tensors[1]), tensors[2]).view(out_shape);
    }
  }

  // Algorithm for multiplying 4 or more matrices
  const auto order = matrix_chain_order(tensors);
  const int64_t i = 0;
  const int64_t j = n - 1;

  if (_out.has_value()) {
    // We manually implement the first recursive layer here so we can use mm_out
    // for the final multiplication
    return at::mm_out(
        result,
        matrix_chain_multiplication(tensors, order, i, order[i][j]),
        matrix_chain_multiplication(tensors, order, order[i][j] + 1, j));
  }
  return matrix_chain_multiplication(tensors, order, i, j).view(out_shape);
}

} // namespace

Tensor linalg_multi_dot(TensorList tensors) {
  return multi_dot_impl(tensors, std::nullopt);
}

Tensor& linalg_multi_dot_out(TensorList tensors, Tensor& result) {
  multi_dot_impl(tensors, result);
  return result;
}

Tensor chain_matmul(TensorList matrices) {
  TORCH_WARN_ONCE(
      "torch.chain_matmul is deprecated and will be removed in a future PyTorch release. ",
      "Use torch.linalg.multi_dot instead, which accepts a list of two or more tensors rather than ",
      "multiple parameters."
  );
  checkAllSameDim(matrices, 2);

  TORCH_CHECK(
      !matrices.empty(), "chain_matmul(): Expected one or more matrices");

  if (matrices.size() == 1) {
    return matrices[0].clone();
  }

  return at::native::linalg_multi_dot(matrices);
}

Tensor& chain_matmul_out(TensorList matrices, Tensor& result) {
  TORCH_WARN_ONCE(
      "torch.chain_matmul is deprecated and will be removed in a future PyTorch release. ",
      "Use torch.linalg.multi_dot instead, which accepts a list of two or more tensors rather than ",
      "multiple parameters."
  );
  checkAllSameDim(matrices, 2);

  TORCH_CHECK(
      !matrices.empty(), "chain_matmul(): Expected one or more matrices");

  if (matrices.size() == 1) {
    at::native::resize_output(result, matrices[0].sizes());
    return result.copy_(matrices[0]);
  }

  return at::native::linalg_multi_dot_out(matrices, result);
}

static void check_1d(const Tensor& t, const char* arg, const char* fn) {
 TORCH_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D");
}

static void check_addr_scalar(const ScalarType dtype,
                              const Scalar& scalar,
                              const std::string& scalar_name) {
  TORCH_CHECK(
    !scalar.isBoolean() || dtype == ScalarType::Bool,
    "Boolean ", scalar_name, " only supported for Boolean results.");
  TORCH_CHECK(
    isFloatingType(dtype) || isComplexType(dtype) || scalar.isIntegral(true),
    "For integral input tensors, "
    "argument ", scalar_name ," must not be a floating point number.");
}

static TensorIterator build_addr_iter(Tensor& result,
                                      const Tensor& self,
                                      const Tensor& vec1,
                                      const Tensor& vec2) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");

  const auto vec1_size0 = vec1.sizes()[0];
  const auto vec2_size0 = vec2.sizes()[0];
  auto self_ = &result == &self
    ? c10::MaybeOwned<Tensor>::borrowed(self)
    : expand_size(self, {vec1_size0, vec2_size0}, "addr");
  TORCH_CHECK(
    self_->dim() == 2,
    "2D tensor expected, got ", self_->dim(), "D tensor for input"
  );
  TORCH_CHECK(
    self_->sizes()[0] == vec1_size0 && self_->sizes()[1] == vec2_size0,
    "size mismatch, input: ", self_->sizes(),
    ", v1: ", vec1.sizes(),
    ", v2: ", vec2.sizes()
  );

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(true)
    .add_output(result)
    .add_owned_const_input(*self_)
    .add_owned_const_input(vec1.reshape({vec1_size0, 1}))
    .add_const_input(vec2)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .build();
  return iter;
}

Tensor addr(const Tensor& self,
            const Tensor& vec1, const Tensor& vec2,
            const Scalar& beta, const Scalar& alpha) {
  Tensor result;
  auto iter = build_addr_iter(result, self, vec1, vec2);

  check_addr_scalar(iter.dtype(), beta, "beta");
  check_addr_scalar(iter.dtype(), alpha, "alpha");

  addr_stub(iter.device_type(), iter, beta, alpha);
  return iter.output();
}

Tensor& addr_(Tensor& self,
              const Tensor& vec1, const Tensor& vec2,
              const Scalar& beta, const Scalar& alpha) {
  return at::addr_out(self, self, vec1, vec2, beta, alpha);
}

Tensor& addr_out(const Tensor& self,
                 const Tensor& vec1, const Tensor& vec2,
                 const Scalar& beta, const Scalar& alpha, Tensor &result) {
  auto iter = build_addr_iter(result, self, vec1, vec2);

  check_addr_scalar(iter.dtype(), beta, "beta");
  check_addr_scalar(iter.dtype(), alpha, "alpha");

  addr_stub(iter.device_type(), iter, beta, alpha);
  return result;
}

// The math_addr and math_addr_out functions support backends
// other than CPU and CUDA, such as XLA.
// They are implemented using the composition of existing ops
Tensor math_addr(const Tensor& self,
                 const Tensor& vec1, const Tensor& vec2,
                 const Scalar& beta, const Scalar& alpha) {
  // when beta==0, values in self should be ignored,
  // nans and infs in self should not propagate.
  Tensor out;
  if (beta.toComplexDouble() == 0.0) {
    if (alpha.toComplexDouble() == 1.0) {
      out = at::outer(vec1, vec2);
    } else {
      out = alpha * at::outer(vec1, vec2);
    }
  } else if (beta.toComplexDouble() == 1.0) {
    if (alpha.toComplexDouble() == 1.0) {
      out = self + at::outer(vec1, vec2);
    } else {
      out = self + alpha * at::outer(vec1, vec2);
    }
  } else if (alpha.toComplexDouble() == 1.0) {
    out = beta * self + at::outer(vec1, vec2);
  } else {
    out = beta * self + alpha * at::outer(vec1, vec2);
  }
  auto result_type = c10::promoteTypes(c10::promoteTypes(self.scalar_type(), vec1.scalar_type()), vec2.scalar_type());
  return out.to(c10::TensorOptions().dtype(result_type));
}

Tensor& math_addr_out(const Tensor& self,
                      const Tensor& vec1, const Tensor& vec2,
                      const Scalar& beta, const Scalar& alpha, Tensor &result) {
  auto addr_result = at::addr(self, vec1, vec2, beta, alpha);

  // Validates safe casting
  const auto result_dtype = addr_result.scalar_type();
  TORCH_CHECK(canCast(result_dtype, result.scalar_type()),
              "result type ", result_dtype,
              " can't be cast to the desired output type ", result.scalar_type());

  at::native::resize_output(result, addr_result.sizes().vec());
  result.copy_(addr_result);
  return result;
}

// torch.ger, alias for torch.outer
Tensor& ger_out(const Tensor& self, const Tensor& vec2, Tensor &result) {
  TORCH_WARN("torch.ger is deprecated and will be removed in a future PyTorch release. "
             "Use torch.outer instead.");
  return at::outer_out(result, self, vec2);
}

Tensor ger(const Tensor& self, const Tensor& vec2) {
  return self.outer(vec2);
}

Tensor& inner_out(const Tensor& self, const Tensor& other, Tensor& out) {
  checkDeviceType("inner()", {out, self, other}, self.device().type());

  // If either self or other is a scalar just multiply them
  if (self.dim() == 0 || other.dim() == 0) {
    at::mul_out(out, self, other);
    return out;
  }

  // Last dimension should match (tensordot does not enforce this)
  TORCH_CHECK(
      self.size(-1) == other.size(-1),
      "inner() the last dimension must match on both input tensors but got shapes ",
      self.sizes(),
      " and ",
      other.sizes());

  at::tensordot_out(out, self, other, -1, -1);
  return out;
}

Tensor inner(const Tensor& self, const Tensor& other) {
  checkDeviceType("inner()", {self, other}, self.device().type());

  // If either self or other is a scalar just multiply them
  if (self.dim() == 0 || other.dim() == 0) {
    return self * other;
  }

  // Last dimension should match (tensordot does not enforce this)
  TORCH_CHECK(
      self.sym_size(-1) == other.sym_size(-1),
      "inner() the last dimension must match on both input tensors but got shapes ",
      self.sym_sizes(),
      " and ",
      other.sym_sizes());

  return at::tensordot(self, other, -1, -1);
}

Tensor& outer_out(const Tensor& self, const Tensor& vec2, Tensor &result) {
  check_1d(self, "self", "outer");
  check_1d(vec2, "vec2", "outer");

  // torch.outer is implemented as a composite op using
```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 253 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `meta`, `detail`, `native`, `at`

**Classes/Structs**: `KronImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Context.h`
- `ATen/Dispatch.h`
- `ATen/ExpandUtils.h`
- `ATen/NamedTensorUtils.h`
- `ATen/OpMathType.h`
- `ATen/Parallel.h`
- `ATen/TensorIndexing.h`
- `ATen/TensorIterator.h`
- `ATen/TensorOperators.h`
- `ATen/TensorSubclassLikeUtils.h`
- `ATen/TensorUtils.h`
- `ATen/core/Tensor.h`
- `ATen/native/CPUBlas.h`
- `ATen/native/cpu/int_mm_kernel.h`
- `ATen/native/LinearAlgebra.h`
- `ATen/native/LinearAlgebraUtils.h`
- `ATen/native/ReduceOps.h`
- `ATen/native/ReduceOpsUtils.h`
- `ATen/native/Resize.h`
- `ATen/native/mkldnn/Matmul.h`
- `ATen/native/mkldnn/Utils.h`
- `ATen/cpu/Utils.h`
- `c10/core/GradMode.h`
- `c10/util/accumulate.h`
- `c10/util/env.h`
- `c10/util/irange.h`
- `variant`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_addmm_activation_native.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `LinearAlgebra.cpp_docs.md`
- **Keyword Index**: `LinearAlgebra.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
