# Documentation: `docs/aten/src/ATen/native/sparse/SparseCsrTensor.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/sparse/SparseCsrTensor.cpp_docs.md`
- **Size**: 53,690 bytes (52.43 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/sparse/SparseCsrTensor.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/sparse/SparseCsrTensor.cpp`
- **Size**: 60,539 bytes (59.12 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// Basic functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/LinearAlgebraUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_pin_memory_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_bsr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_bsc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_compressed_tensor_with_dims_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/_validate_sparse_compressed_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_csr_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_csc_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_bsr_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_bsc_tensor_args_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/ccol_indices_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/col_indices_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/crow_indices_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/row_indices_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/select_copy.h>
#include <ATen/ops/select_copy_native.h>
#include <ATen/ops/sparse_compressed_tensor_native.h>
#include <ATen/ops/sparse_csr_tensor_native.h>
#include <ATen/ops/sparse_csc_tensor_native.h>
#include <ATen/ops/sparse_bsr_tensor_native.h>
#include <ATen/ops/sparse_bsc_tensor_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/_validate_compressed_sparse_indices.h>
#include <ATen/ops/where.h>
#endif

namespace at::native {

using namespace at::sparse_csr;

namespace {

bool solve_arange(const Tensor& input, int64_t& start, int64_t& end, int64_t& step) {
  /*
    This function solves the equation

      input == arange(start, end, step)

    for integers start, end, and step, if possible. If the solution
    exists, returns true.
  */
  int64_t n = input.numel();
  if (n == 0) {
    // a trivial solution
    start = end = 0;
    step = 1;
  } else if (n == 1) {
    // a simple solution
    start = input[0].item<int64_t>();
    end = start + 1;
    step = 1;
  } else {
    Tensor first_last = input.slice(0, 0, n, n - 1).cpu();
    int64_t start_candidate = first_last[0].item<int64_t>();
    int64_t end_candidate = first_last[1].item<int64_t>() + 1;
    if (end_candidate - start_candidate == n) {
      // a special solution
      start = start_candidate;
      end = end_candidate;
      step = 1;
    } else {
      // detect if general solution exists
      Tensor possible_steps = input.slice(0, 1).sub(input.slice(0, 0, n - 1));
      Tensor possible_step = possible_steps[0];
      if ((possible_steps.eq(possible_step)).all().item<bool>()) {
        start = start_candidate;
        end = end_candidate;
        step = possible_step.item<int64_t>();
      } else {
        // no solution
        return false;
      }
    }
  }
  return true;
}

} // end anonymous namespace

/*
  Validate the arguments to sparse compressed (CSR, CSC, BSR, and BSC)
  tensor factory functions.

  The CSR and BSR invariants for PyTorch are outlined in

    https://pearu.github.io/csr_tensor_invariants.html
    https://pearu.github.io/bsr_tensor_invariants.html

  that in what follows are generalized for all sparse compressed
  formats with support to batched and dense dimensions.
*/

static void _validate_sparse_compressed_tensor_args_worker(const Tensor& compressed_indices, const Tensor& plain_indices, const Tensor& values, const IntArrayRef size, const Layout& layout, std::optional<bool> check_pinning_) {
  // Layout must be Sparse Compressed, 2.4
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout, "validate_sparse_compressed_tensor_args", [&]{});

  const std::string layout_name = layoutToString(layout, /*upper=*/ true);
  const std::string compressed_indices_name = compressedIndicesName(layout);
  const std::string plain_indices_name = plainIndicesName(layout);
  const std::string compressed_dim_name = compressedDimName(layout);
  const std::string plain_dim_name = plainDimName(layout);
  const bool check_pinning = check_pinning_.value_or(true);

  // Layout Invariants

  // Re 3.5 and 3.6: in the case of compressed/plain indices tensors,
  // we require contiguity per-patch basis, that is, the last stride
  // of these indices must be 1. The reasoning for this is that
  // indices tensors within a patch are "atomic" in the sense that
  // sliced compressed/plain indices would not represent the indices
  // of any sparse compressed tensor as the slicing would break the
  // description of the tensor index structure.

  // 2.1
  TORCH_CHECK(plain_indices.layout() == kStrided,
              "expected ", plain_indices_name, " to be a strided tensor but got ", plain_indices.layout(), " tensor");

  // 2.2
  TORCH_CHECK(compressed_indices.layout() == kStrided,
              "expected ", compressed_indices_name, " to be a strided tensor but got ", compressed_indices.layout(), " tensor");

  const int base_ndim = 2;  // corresponds to compressed and plain indices
  const auto batch_ndim = compressed_indices.dim() - 1;
  const int block_ndim = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
                           layout, "validate_sparse_compressed_tensor_args",
                           [&] { return 0; }, [&] { return 2; });
  const auto dense_ndim = values.dim() - batch_ndim - block_ndim - 1;

  // 2.3
  TORCH_CHECK(values.layout() == kStrided,
              "expected values to be a strided tensor but got ", values.layout(), " tensor");

  // 3.7 is dropped, that is, values tensor does not need to be
  // contiguous, in general. Particular algorithms on sparse
  // compressed tensors may require contiguity though.

  // Shape and Strides invariants

  // 3.2
  TORCH_CHECK(
              batch_ndim >= 0,
              compressed_indices_name, " must have dimensionality >= 1 but got ", compressed_indices.dim());

  // 3.3
  TORCH_CHECK(
              compressed_indices.dim() == plain_indices.dim(),
              compressed_indices_name, " and ", plain_indices_name, " dimensionalities must be equal but got ",
              compressed_indices.dim(), " and ", plain_indices.dim(), ", respectively");

  // 3.4
  TORCH_CHECK(
              dense_ndim >= 0,
              "values must have dimensionality > sum of batch and block dimensionalities (=",
              batch_ndim, " + ", block_ndim, ") but got ", values.dim());

  // 3.5
  TORCH_CHECK(plain_indices.stride(-1) == 1,
              "expected ", plain_indices_name, " to be a contiguous tensor per batch");

  // 3.6
  TORCH_CHECK(compressed_indices.stride(-1) == 1,
              "expected ", compressed_indices_name, " to be a contiguous tensor per batch");

  // 3.1
  TORCH_CHECK(
              static_cast<int>(size.size()) == batch_ndim + base_ndim + dense_ndim,
              "tensor dimensionality must be sum of batch, base, and dense dimensionalities (=",
              batch_ndim, " + ", base_ndim, " + ", dense_ndim, ") but got ", size.size());

  // For CSR/CSC formats, we define blocksize=(1, 1) so that checking
  // the sparse compressed tensor invariants can be unified with the
  // BSR/BSC invariants.
  // 3.10
  DimVector blocksize{
                      (block_ndim == 2 ? std::max<int64_t>(1, values.size(batch_ndim + 1)) : 1),
                      (block_ndim == 2 ? std::max<int64_t>(1, values.size(batch_ndim + 2)) : 1),
  };
  TORCH_INTERNAL_ASSERT(blocksize.size() == 2 && blocksize[0] > 0 && blocksize[1] > 0);

  // All batch sizes must be the same and consistent with tensor batchsize, 3.1, 3.8, 3.9, 3.10
  DimVector batchsize = DimVector(size.slice(0, batch_ndim));
  DimVector compressed_indices_batchsize = DimVector(compressed_indices.sizes().slice(0, batch_ndim));
  DimVector plain_indices_batchsize = DimVector(plain_indices.sizes().slice(0, batch_ndim));
  DimVector values_batchsize = DimVector(values.sizes().slice(0, batch_ndim));
  const int64_t values_nnz = values.size(batch_ndim);
  DimVector values_blocksize = DimVector(values.sizes().slice(batch_ndim + 1, block_ndim));
  DimVector values_densesize = DimVector(values.sizes().slice(batch_ndim + 1 + block_ndim, dense_ndim));
  TORCH_CHECK(
      batchsize == compressed_indices_batchsize && batchsize == plain_indices_batchsize && batchsize == values_batchsize,
      "all batch dimensions of ", compressed_indices_name," (=", compressed_indices_batchsize, "), ", plain_indices_name," (=",
      plain_indices_batchsize, "), and values (=", values_batchsize, ") must be equal to tensor batch dimensions (=",
      batchsize, ")");

  // A tensor constitutes of full blocks, 3.1
  for (int i=0; i<block_ndim; i++) {
      TORCH_CHECK(size[batch_ndim + i] % blocksize[i] == 0,
                  "tensor shape[", batch_ndim + i, "] (=", size[batch_ndim + i],
                  ") must be divisible with blocksize[", i, "] (=", blocksize[i],
                  ") as defined by values shape");
  }
  const int64_t nrows = size[batch_ndim] / blocksize[0];
  const int64_t ncols = size[batch_ndim + 1] / blocksize[1];
  auto [compressed_dim_size, plain_dim_size] = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(layout, "validate_sparse_compressed_tensor_args",
                                                                                            [&] { return std::make_tuple(nrows, ncols); },
                                                                                            [&] { return std::make_tuple(ncols, nrows); });
  // 3.8
  TORCH_CHECK(
              compressed_indices.size(-1) == compressed_dim_size + 1,
              compressed_indices_name, ".shape[-1] must be equal to the number of ",
              compressed_dim_name, "s + 1 (=", compressed_dim_size + 1, "), but got ", compressed_indices.size(-1));
  // 3.9, 3.10
  TORCH_CHECK(
              plain_indices.size(-1) == values_nnz,
              plain_indices_name, ".shape[-1] must be equal to nnz (=", values_nnz,
              ") as defined by values.shape[", batch_ndim, "], but got ", plain_indices.size(-1));
  // Type Invariants
  auto compressed_indices_type = compressed_indices.scalar_type();
  auto plain_indices_type = plain_indices.scalar_type();
  // 1.1, 1.2, 1.3
  TORCH_CHECK(
      compressed_indices_type == plain_indices_type,
      compressed_indices_name, " and ", plain_indices_name, " must have the same dtype, bot got ",
      compressed_indices_type, " and ", plain_indices_type, ", respectively");
  TORCH_CHECK(
      compressed_indices_type == kInt || compressed_indices_type == kLong,
      compressed_indices_name, " and ", plain_indices_name, " dtype must be Int or Long, but got ",
      compressed_indices_type);

  if (compressed_indices.is_meta()) {
    TORCH_CHECK(values_nnz == 0, "expected nnz to be 0 for sparse ", layout_name, " meta tensor but got ", values_nnz);
  } else {
    // Indices invariants
    at::_validate_compressed_sparse_indices(
        /*is_crow = */layout == kSparseCsr || layout == kSparseBsr,
        compressed_indices,
        plain_indices,
        compressed_dim_size,
        plain_dim_size,
        values_nnz);
  }

  // Device Invariants
  // 4.1
  TORCH_CHECK(
      values.device().type() == kCPU || values.device().type() == kCUDA || values.device().type() == kXPU || values.device().type() == kMeta || values.device().type() == kPrivateUse1,
      "device type of values (",
      values.device().type(),
      ") must be one of CPU, CUDA, XPU, Meta or PrivateUse1")
  // 4.2, 4.3, 4.4
  TORCH_CHECK(
      compressed_indices.get_device() == values.get_device(),
      "device of ", compressed_indices_name, " (=",
      compressed_indices.device(),
      ") must match device of values (=",
      values.device(),
      ")");
  TORCH_CHECK(
      compressed_indices.get_device() == plain_indices.get_device(),
      "device of ", compressed_indices_name, " (=",
      compressed_indices.device(),
      ") must match device of ", plain_indices_name, " (=",
      plain_indices.device(),
      ")");
  if (check_pinning) {
    TORCH_CHECK(
      compressed_indices.is_pinned() == values.is_pinned(),
      "memory pinning of ", compressed_indices_name, " (=",
      compressed_indices.is_pinned(),
      ") must match memory pinning of values (=",
      values.is_pinned(),
      ")");
    TORCH_CHECK(
      compressed_indices.is_pinned() == plain_indices.is_pinned(),
      "memory pinning of ", compressed_indices_name, " (=",
      compressed_indices.is_pinned(),
      ") must match memory pinning of ", plain_indices_name, " (=",
      plain_indices.is_pinned(),
      ")");
  }

  // Autograd Invariants
  //
  // These are internal asserts because users should not be able to
  // create non-floating point dtype tensors with requires_grad flag
  // set to true.
  TORCH_INTERNAL_ASSERT(!compressed_indices.requires_grad());
  TORCH_INTERNAL_ASSERT(!plain_indices.requires_grad());
}

void _validate_sparse_compressed_tensor_args(const Tensor& compressed_indices, const Tensor& plain_indices, const Tensor& values, IntArrayRef size, Layout layout, std::optional<bool> check_pinning) {
  _validate_sparse_compressed_tensor_args_worker(compressed_indices, plain_indices, values, size, layout, check_pinning);
}

void _validate_sparse_csr_tensor_args(const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, IntArrayRef size, std::optional<bool> check_pinning) {
  _validate_sparse_compressed_tensor_args_worker(crow_indices, col_indices, values, size, kSparseCsr, check_pinning);
}

void _validate_sparse_csc_tensor_args(const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values, IntArrayRef size, std::optional<bool> check_pinning) {
  _validate_sparse_compressed_tensor_args_worker(ccol_indices, row_indices, values, size, kSparseCsc, check_pinning);
}

void _validate_sparse_bsr_tensor_args(const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, IntArrayRef size, std::optional<bool> check_pinning) {
  _validate_sparse_compressed_tensor_args_worker(crow_indices, col_indices, values, size, kSparseBsr, check_pinning);
}

void _validate_sparse_bsc_tensor_args(const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values, IntArrayRef size, std::optional<bool> check_pinning) {
  _validate_sparse_compressed_tensor_args_worker(ccol_indices, row_indices, values, size, kSparseBsc, check_pinning);
}

// Construction of CSR, CSC, BSR, and BSC tensors.

// Note: The usage of "Csr" in names like SparseCsrTensor,
// SparseCsrCPU, SparseCsrCUDA, and SparseCsrTensorImpl exists because
// of historical reasons (that ought to be removed in future) and does
// not mean that the corresponding functionality would be CSR layout
// only specific.
static SparseCsrTensor new_compressed_tensor(const TensorOptions& options) {
  // TODO: remove this comment after enabling autograd support for CSR tensor
  // constructor.
  // TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
  Layout layout = AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(options.layout(), "new_compressed_tensor", [&] { return the_layout; });
  DispatchKey dispatch_key = DispatchKey::Undefined;

  switch(options.device().type()) {
  case kCPU:
    dispatch_key = DispatchKey::SparseCsrCPU;
    break;
  case kCUDA:
    dispatch_key = DispatchKey::SparseCsrCUDA;
    break;
  case kXPU:
    dispatch_key = DispatchKey::SparseCsrXPU;
    break;
  case kMeta:
    dispatch_key = DispatchKey::SparseCsrMeta;
    break;
  case kPrivateUse1:
    dispatch_key = DispatchKey::SparseCsrPrivateUse1;
    break;
  default:
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Could not run 'new_compressed_tensor' from the '", options.device(), "' device.)");
  }

  return detail::make_tensor<SparseCsrTensorImpl>(DispatchKeySet(dispatch_key), options.device(), layout, options.dtype());
}

Tensor sparse_compressed_tensor_with_dims(
     int64_t nnz,
     int64_t dense_dim,
     c10::IntArrayRef size,
     c10::IntArrayRef blocksize,
     ScalarType index_dtype,
     std::optional<ScalarType> dtype,
     std::optional<Layout> layout,
     std::optional<Device> device,
     std::optional<bool> pin_memory) {
  // sparse_compressed_tensor_with_dims is a generalization of empty
  // that enables the specification of nnz, dense_dim, blocksize, and
  // index_dtype for sparse compressed tensors.
  //
  // sparse_compressed_tensor_with_dims indices and values tensors are
  // created as empty tensors, so the returned sparse compressed
  // tensor will not satisfy the sparse compressed tensor
  // invariants. The caller is responsible for initializing the
  // indices tensors properly.
  TORCH_CHECK(layout, "sparse_compressed_tensor_with_dims: expected sparse compressed tensor layout but got none");

  Layout layout_ = layout.value();
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout_, "sparse_compressed_tensor_with_dims", [&]{});

  constexpr int64_t sparse_dim = 2;
  int64_t batch_dim = size.size() - dense_dim - sparse_dim;
  TORCH_CHECK(batch_dim >= 0, "sparse_compressed_tensor_with_dims: dimensionality must be at least dense_dim(=",
              dense_dim, ") + sparse_dim(=", sparse_dim, "), but got ", size.size());

  TORCH_CHECK(nnz >= 0, "sparse_compressed_tensor_with_dims: nnz must be non-negative, got ", nnz);

  auto plain_indices_size = DimVector(size.slice(0, batch_dim));
  auto compressed_indices_size = DimVector(size.slice(0, batch_dim));
  auto values_size = DimVector(size.slice(0, batch_dim));

  plain_indices_size.push_back(nnz);
  values_size.push_back(nnz);

  if (layout_ == kSparseBsr || layout_ == kSparseBsc) {
    TORCH_CHECK(blocksize.size() == (size_t)sparse_dim, "sparse_compressed_tensor_with_dims: blocksize needs to be a tuple of size ",
                sparse_dim, ", but got ", blocksize.size());
    auto d0 = (layout_ == kSparseBsr ? 0 : 1);
    auto d1 = (layout_ == kSparseBsr ? 1 : 0);
    TORCH_CHECK(blocksize[0] > 0 && blocksize[1] > 0, "sparse_compressed_tensor_with_dims: blocksize needs to be positive, but got ", blocksize);
    auto compressed_size = size[compressedDimension(layout_, size, dense_dim)];
    auto plain_size = size[plainDimension(layout_, size, dense_dim)];
    TORCH_CHECK(compressed_size % blocksize[d0] == 0, "sparse_compressed_tensor_with_dims: dimension ",
                compressedDimension(layout_, size, dense_dim), " must be multiple of blocksize[", d0, "](=", blocksize[d0], ") but got ", compressed_size);
    TORCH_CHECK(plain_size % blocksize[d1] == 0, "sparse_compressed_tensor_with_dims: dimension ", plainDimension(layout_, size, dense_dim),
                " must be multiple of blocksize[", d1, "](=", blocksize[d1], ") but got ", plain_size);
    compressed_indices_size.push_back(compressed_size / blocksize[d0] + 1);
    values_size.append(DimVector(blocksize));
  } else {
    TORCH_CHECK(blocksize.empty(), "sparse_compressed_tensor_with_dims: blocksize cannot be specified for non-block layout ", layout_);
    compressed_indices_size.push_back(size[compressedDimension(layout_, size, dense_dim)] + 1);
  }

  values_size.append(DimVector(size.slice(batch_dim + sparse_dim, dense_dim)));
  TORCH_CHECK(
      index_dtype == ScalarType::Int || index_dtype == ScalarType::Long,
      "indices dtype must be Int or Long, but got ", index_dtype);

  TensorOptions options_ = TensorOptions().layout(Layout::Strided).device(device).pinned_memory(pin_memory);
  auto compressed_indices = at::empty(compressed_indices_size, options_.dtype(index_dtype));
  auto plain_indices = at::empty(plain_indices_size, options_.dtype(index_dtype));
  auto values = at::empty(values_size, options_.dtype(dtype));
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);
  SparseCsrTensor self = new_compressed_tensor(options);
  if (pin_memory.value_or(false) && !values.is_pinned()) {
    get_sparse_csr_impl(self)->set_member_tensors(compressed_indices.pin_memory(), plain_indices.pin_memory(), values.pin_memory(), size);
  } else {
    get_sparse_csr_impl(self)->set_member_tensors(compressed_indices, plain_indices, values, size);
  }
  return self;
}

Tensor _sparse_compressed_tensor_unsafe_symint(
     const Tensor& compressed_indices,
     const Tensor& plain_indices,
     const Tensor& values,
     c10::SymIntArrayRef size,
     std::optional<ScalarType> dtype,
     std::optional<Layout> layout,
     std::optional<Device> device,
     std::optional<bool> pin_memory) {
  if (!layout) {
    TORCH_CHECK(false, "sparse_compressed_tensor_unsafe expected sparse compressed tensor layout but got none");
  }
  Layout layout_ = layout.value();
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout_, "sparse_compressed_tensor_unsafe", [&]{});
  if (at::globalContext().checkSparseTensorInvariants()) {
    _validate_sparse_compressed_tensor_args_worker(compressed_indices, plain_indices, values, C10_AS_INTARRAYREF_SLOW(size), layout_, true);
  }
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);
  SparseCsrTensor self = new_compressed_tensor(options);
  if (pin_memory.value_or(false) && !values.is_pinned()) {
    get_sparse_csr_impl(self)->set_member_tensors(compressed_indices.pin_memory(), plain_indices.pin_memory(), values.pin_memory(), size);
  } else {
    get_sparse_csr_impl(self)->set_member_tensors(compressed_indices, plain_indices, values, size);
  }
  return self;
}

template <Layout required_layout>
static Tensor _sparse_compressed_tensor_unsafe_template(const Tensor& compressed_indices,
                                                 const Tensor& plain_indices,
                                                 const Tensor& values,
                                                 IntArrayRef size,
                                                 std::optional<ScalarType> dtype,
                                                 std::optional<Layout> layout,
                                                 std::optional<Device> device,
                                                 std::optional<bool> pin_memory) {
  Layout layout_ = layout.value_or(required_layout);
  TORCH_CHECK(layout_ == required_layout, "sparse compressed layout must be ",required_layout, " but got ", layout_);
  if (at::globalContext().checkSparseTensorInvariants()) {
    _validate_sparse_compressed_tensor_args_worker(compressed_indices, plain_indices, values, size, layout_, true);
  }
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);
  SparseCsrTensor self = new_compressed_tensor(options);
  if (pin_memory.value_or(false) && !values.is_pinned()) {
    get_sparse_csr_impl(self)->set_member_tensors(compressed_indices.pin_memory(), plain_indices.pin_memory(), values.pin_memory(), size);
  } else {
    get_sparse_csr_impl(self)->set_member_tensors(compressed_indices, plain_indices, values, size);
  }
  return self;
}

#define SPARSE_COMPRESSED_TENSOR_UNSAFE(KIND, REQUIRED_LAYOUT)          \
  Tensor _sparse_##KIND##_tensor_unsafe(const Tensor& compressed_indices, \
                                        const Tensor& plain_indices,    \
                                        const Tensor& values,           \
                                        IntArrayRef size,               \
                                        std::optional<ScalarType> dtype, \
                                        std::optional<Layout> layout,   \
                                        std::optional<Device> device,   \
                                        std::optional<bool> pin_memory) { \
    return _sparse_compressed_tensor_unsafe_template<REQUIRED_LAYOUT>(compressed_indices, plain_indices, values, size, dtype, layout, device, pin_memory); \
  }

SPARSE_COMPRESSED_TENSOR_UNSAFE(csr, kSparseCsr)
SPARSE_COMPRESSED_TENSOR_UNSAFE(csc, kSparseCsc)
SPARSE_COMPRESSED_TENSOR_UNSAFE(bsr, kSparseBsr)
SPARSE_COMPRESSED_TENSOR_UNSAFE(bsc, kSparseBsc)

static DimVector _estimate_sparse_compressed_tensor_size(
    const Tensor& compressed_indices,
    const Tensor& plain_indices,
    const Tensor& values,
    Layout layout) {
  const int block_ndim = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(layout, "estimate_sparse_compressed_tensor_size", [&] { return 0; }, [&] { return 2; });
  const int base_ndim = 2;  // corresponds to compressed and plain indices
  const auto batch_ndim = compressed_indices.dim() - 1;
  const std::string compressed_indices_name = compressedIndicesName(layout);
  const std::string plain_indices_name = plainIndicesName(layout);
  TORCH_CHECK(
              batch_ndim >= 0,
              compressed_indices_name, " must have dimensionality >= 1 but got ", compressed_indices.dim());
  TORCH_CHECK(
              compressed_indices.dim() == plain_indices.dim(),
              compressed_indices_name, " and ", plain_indices_name, " dimensionalities must be equal but got ",
              compressed_indices.dim(), " and ", plain_indices.dim(), ", respectively");
  const int64_t dense_ndim = values.dim() - batch_ndim - block_ndim - 1;
  TORCH_CHECK(
              dense_ndim >= 0,
              "values must have dimensionality > sum of batch and block dimensionalities (=",
              batch_ndim, " + ", block_ndim, ") but got ", values.dim());
  DimVector blocksize{
                      (block_ndim == 2 ? std::max<int64_t>(1, values.size(batch_ndim + 1)) : 1),
                      (block_ndim == 2 ? std::max<int64_t>(1, values.size(batch_ndim + 2)) : 1)
  };
  DimVector size = DimVector(compressed_indices.sizes().slice(0, batch_ndim));
  int64_t compressed_dim_size = (compressed_indices.dim() > 0 && compressed_indices.size(-1) > 0 ? compressed_indices.size(-1) - 1 : 0);
  int64_t plain_dim_size = AT_DISPATCH_INTEGRAL_TYPES(plain_indices.scalar_type(), "estimate_sparse_compressed_tensor_size",
                                                      [&]() -> int64_t {
                                                        if (plain_indices.numel() > 0) {
                                                          return plain_indices.max().item<scalar_t>() + 1;
                                                        } else {
                                                          return 0;
                                                        }
                                                      });
  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(layout, "estimate_sparse_compressed_tensor_size",
      [&]{
        size.push_back(compressed_dim_size * blocksize[0]);
        size.push_back(plain_dim_size * blocksize[1]);
      },
      [&]{
        size.push_back(plain_dim_size * blocksize[0]);
        size.push_back(compressed_dim_size * blocksize[1]);
      });
  for (int i=0; i<dense_ndim; i++) {
    int64_t j = batch_ndim + 1 + block_ndim + i;
    size.push_back((j < values.dim() ? values.size(j) : 1));
  }
  TORCH_CHECK(
              static_cast<int>(size.size()) == batch_ndim + base_ndim + dense_ndim,
              "tensor dimensionality must be sum of batch, base, and dense dimensionalities (=",
              batch_ndim, " + ", base_ndim, " + ", dense_ndim, ") but got ", size.size());
  return size;
}

// TODO: This constructor should probably use an ATen abstract method in order
// to make autograd dispatch available for the CSR constructor. See the relevant
// note in native_functions.yaml.
Tensor sparse_compressed_tensor(
    const Tensor& compressed_indices,
    const Tensor& plain_indices,
    const Tensor& values,
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {

  if (!layout) {
    TORCH_CHECK(false, "sparse_compressed_tensor expected sparse compressed tensor layout but got none");
  }
  Layout layout_ = layout.value();
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout_, "sparse_compressed_tensor", [&]{});

  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);

  return at::_sparse_compressed_tensor_unsafe(
      compressed_indices,
      plain_indices,
      values,
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

Tensor sparse_compressed_tensor(
    const Tensor& compressed_indices,
    const Tensor& plain_indices,
    const Tensor& values,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {

  if (!layout) {
    TORCH_CHECK(false, "sparse_compressed_tensor expected sparse compressed tensor layout but got none");
  }
  Layout layout_ = layout.value();
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout_, "sparse_compressed_tensor", [&]{});

  DimVector size = _estimate_sparse_compressed_tensor_size(compressed_indices, plain_indices, values, layout_);

  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);

  return at::_sparse_compressed_tensor_unsafe(
      compressed_indices,
      plain_indices,
      values,
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

#define SPARSE_COMPRESSED_TENSOR(KIND, REQUIRED_LAYOUT)                 \
  Tensor sparse_##KIND##_tensor(const Tensor& compressed_indices,       \
                                const Tensor& plain_indices,            \
                                const Tensor& values,                   \
                                std::optional<ScalarType> dtype,        \
                                std::optional<Layout> layout,           \
                                std::optional<Device> device,           \
                                std::optional<bool> pin_memory) {       \
    if (layout) {                                                       \
      TORCH_CHECK(layout.value() == REQUIRED_LAYOUT, "sparse " # KIND " layout must be ", REQUIRED_LAYOUT, " but got ", layout.value()); \
    }                                                                   \
    std::optional<Layout> layout_(REQUIRED_LAYOUT);                     \
    return at::native::sparse_compressed_tensor(compressed_indices, plain_indices, values, dtype, layout_, device, pin_memory); \
  }                                                                     \
  Tensor sparse_##KIND##_tensor(const Tensor& compressed_indices,       \
                                const Tensor& plain_indices,            \
                                const Tensor& values,                   \
                                IntArrayRef size,                       \
                                std::optional<ScalarType> dtype,        \
                                std::optional<Layout> layout,           \
                                std::optional<Device> device,           \
                                std::optional<bool> pin_memory) {       \
    if (layout) {                                                       \
      TORCH_CHECK(layout.value() == REQUIRED_LAYOUT, "sparse " # KIND " layout must be ", REQUIRED_LAYOUT, " but got ", layout.value()); \
    }                                                                   \
    std::optional<Layout> layout_(REQUIRED_LAYOUT);                     \
    return at::native::sparse_compressed_tensor(compressed_indices, plain_indices, values, size, dtype, layout_, device, pin_memory); \
  }

SPARSE_COMPRESSED_TENSOR(csr, kSparseCsr)
SPARSE_COMPRESSED_TENSOR(csc, kSparseCsc)
SPARSE_COMPRESSED_TENSOR(bsr, kSparseBsr)
SPARSE_COMPRESSED_TENSOR(bsc, kSparseBsc)

Tensor empty_sparse_compressed_symint(
    SymIntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<MemoryFormat> optional_memory_format) {
  // TODO: Don't specialize
  return empty_sparse_compressed(C10_AS_INTARRAYREF_SLOW_ALLOC(size), dtype, layout, device, pin_memory, optional_memory_format);
}

// Warning: ideally, torch.empty(..., layout=<sparse compressed
// format>) ought to be unsupported because it does not return a valid
// sparse compressed tensor without initialization of compressed
// indices. The implementation below is kept for BC.
Tensor empty_sparse_compressed(
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<MemoryFormat> optional_memory_format) {
  check_size_nonnegative(size);
  TORCH_CHECK(size.size() >= 2, "torch.empty: Only batched sparse compressed (non-block) tensors are supported, but got size ", size);

  // Strided is the default layout for torch.empty.
  Layout layout_ = layout.value_or(Layout::Strided);

  // torch.empty cannot be used to create blocked tensors because its
  // API lacks a method to specify the block size.
  AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(layout_, "empty_sparse_compressed", [&]{});

  int64_t nnz = 0;
  auto compressed_indices_size = DimVector(size.slice(0, size.size() - 2));
  auto plain_indices_and_values_size = DimVector(size.slice(0, size.size() - 2));
  compressed_indices_size.push_back(size[compressedDimension(layout_, size)] + 1);
  plain_indices_and_values_size.push_back(nnz);

  TensorOptions options = TensorOptions().dtype(ScalarType::Long).layout(Layout::Strided).device(device).pinned_memory(pin_memory);
  auto compressed_indices = at::empty(compressed_indices_size, options);
  auto plain_indices = at::empty(plain_indices_and_values_size, options);
  auto values = at::empty(plain_indices_and_values_size, options.dtype(dtype));
  // torch.empty on produces garbage so that the resulting empty
  // sparse compressed tensor may fail to satisfy the following
  // compressed sparse tensor invariants:
  //
  //   compressed_indices[..., 0] == 0
  //   compressed_indices[..., -1] == nnz.
  //   compressed_indices must be non-decreasing sequence
  //
  // Therefore, avoid using empty to create sparse compressed
  // tensors. Instead, use compressed sparse constructors directly or
  // other factory functions such as torch.zeros, etc.
  return at::_sparse_compressed_tensor_unsafe(compressed_indices,
                                              plain_indices,
                                              values,
                                              size,
                                              dtype,
                                              layout,
                                              device,
                                              pin_memory);
}

const Tensor& resize_sparse_csr_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  check_size_nonnegative(size);
  TORCH_CHECK(size.size() >= 2, "torch.resize_: Only batched sparse CSR matrices are supported, but got size ", size);
  TORCH_CHECK(
      self.size(-1) <= size[size.size() - 1],
      "torch.resize_: Resizing columns of sparse CSR tensors to a smaller value is not supported. ",
      "The original number of columns is ",
      self.size(-1),
      " while the requested new number of columns is ", size[size.size() - 1], ".");
  get_sparse_csr_impl(self)->resize_(self._nnz(), size);
  return self;
}

Tensor& copy_sparse_compressed_(Tensor& self, const Tensor& src, bool non_blocking) {
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "copy_sparse_compressed_", [&]{});
  TORCH_CHECK(
      self.layout() == src.layout(),
      "torch.copy_: copy of sparse compressed tensors having different layouts is not supported.",
      " self layout is ", self.layout(), " and src layout is ", src.layout());
  TORCH_CHECK(
      self._nnz() == src._nnz(),  // actually, values copy allows different shapes as long as operands are broadcastable
      "torch.copy_: only sparse compressed tensors with the same number of specified elements are supported.");
  auto self_compressed_dim = compressedDimension(self.layout(), self.sizes());
  auto src_compressed_dim = compressedDimension(src.layout(), src.sizes());
  auto self_compressed_dims = self.size(self_compressed_dim);
  auto src_compressed_dims = src.size(compressedDimension(src.layout(), src.sizes()));
  if (self_compressed_dim == src_compressed_dim) {
    TORCH_CHECK(self_compressed_dims == src_compressed_dims,
                "torch.copy_: expected shapes of self and src to match along dimension ",
                self_compressed_dim, " for ",
                self.layout(), " layout but the corresponding dimensions of self and src are ",
                self_compressed_dims, " and ", src_compressed_dims, ", respectively.");
  } else {
    TORCH_CHECK(self_compressed_dims == src_compressed_dims,
                "torch.copy_: expected shapes of self and src to match along dimensions ",
                self_compressed_dim, " and ", src_compressed_dim, ", respectively, for ",
                self.layout(), " layout but the corresponding dimensions of self and src are ",
                self_compressed_dims, " and ", src_compressed_dims, ", respectively.");
  }
  AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "copy_sparse_compressed_",
                                              [&]{},
                                              [&]{
                                                auto self_values = self.values();
                                                auto src_values = src.values();
                                                auto self_blocksize = DimVector(self_values.sizes().slice(self_values.dim()-2, 2));
                                                auto src_blocksize = DimVector(src_values.sizes().slice(src_values.dim()-2, 2));
                                                TORCH_CHECK(self_blocksize == src_blocksize,
                                                            "torch.copy_: copy of sparse compressed tensors having different block sizes is not supported.",
                                                            " self and src block sizes are ", self_blocksize, " and ", src_blocksize, ", respectively.");
                                              });
  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "copy_sparse_compressed_",
                                            [&]{
                                              self.crow_indices().copy_(src.crow_indices(), non_blocking);
                                              self.col_indices().copy_(src.col_indices(), non_blocking);
                                            },
                                            [&]{
                                              self.ccol_indices().copy_(src.ccol_indices(), non_blocking);
                                              self.row_indices().copy_(src.row_indices(), non_blocking);
                                            });
  self.values().copy_(src.values(), non_blocking);
  return self;
}

// Access members of CSR tensors.
int64_t _nnz_sparse_csr(const SparseCsrTensor& self) {
  return get_sparse_csr_impl(self)->nnz();
}

Tensor values_sparse_csr(const Tensor& self) {
  return get_sparse_csr_impl(self)->values().alias();
}

Tensor crow_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_ROW_COMPRESSED_LAYOUTS(self.layout(),
                                                   "crow_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->compressed_indices().alias(); });
}

Tensor col_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_ROW_COMPRESSED_LAYOUTS(self.layout(),
                                                   "col_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->plain_indices().alias(); });
}

Tensor ccol_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_COL_COMPRESSED_LAYOUTS(self.layout(),
                                                   "ccol_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->compressed_indices().alias(); });
}

Tensor row_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_COL_COMPRESSED_LAYOUTS(self.layout(),
                                                   "row_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->plain_indices().alias(); });
}

Tensor crow_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "crow_indices expected sparse row compressed tensor layout but got ", self.layout());
}

Tensor col_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "col_indices expected sparse row compressed tensor layout but got ", self.layout());
}

Tensor ccol_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "ccol_indices expected sparse column compressed tensor layout but got ", self.layout());
}

Tensor row_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "row_indices expected sparse column compressed tensor layout but got ", self.layout());
}

int64_t sparse_dim_sparse_csr(const SparseCsrTensor& self) {
  return get_sparse_csr_impl(self)->sparse_dim();
}

int64_t dense_dim_sparse_csr(const SparseCsrTensor& self) {
  return get_sparse_csr_impl(self)->dense_dim();
}

const SparseCsrTensor& resize_as_sparse_compressed_(
    const SparseCsrTensor& self,
    const SparseCsrTensor& src) {
  auto src_layout = src.layout();
  auto self_layout = self.layout();
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      src_layout, "resize_as_sparse_compressed_: src ", []() {});
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      self_layout, "resize_as_sparse_compressed_: self ", []() {});
  // Note: The impl method does all required checking to see if resize/data copy
  // on member tensors is required.
  get_sparse_csr_impl(self)->resize_as_sparse_compressed_tensor_(src);
  return self;
}

SparseCsrTensor clone_sparse_compressed(
                                        const SparseCsrTensor& self,
                                        std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  TensorOptions options = self.options();
  auto compressed_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(),
                                                                      "clone_sparse_compressed",
                                                                      [&]{ return self.crow_indices(); },
                                                                      [&]{ return self.ccol_indices(); });
  auto plain_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(),
                                                                 "clone_sparse_compressed",
                                                                 [&]{ return self.col_indices(); },
                                                                 [&]{ return self.row_indices(); });
  return at::_sparse_compressed_tensor_unsafe(
       compressed_indices.clone(),
       plain_indices.clone(),
       self.values().clone(),
       self.sizes(),
       optTypeMetaToScalarType(options.dtype_opt()),
       options.layout_opt(),
       options.device_opt(),
       options.pinned_memory_opt());
}

Tensor empty_like_sparse_csr(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  TensorOptions options =
      self.options()
          .merge_in(options_)
          .merge_memory_format(optional_memory_format);

  TORCH_CHECK(options.layout() == self.layout(),
    "empty_like with different sparse layout is not supported (self is ",
    self.layout(), " but you requested ", options.layout(), ")");
  if (options.layout() == kSparseCsr) {
    auto result = at::native::_sparse_csr_tensor_unsafe(
        self.crow_indices().to(options.device(), self.crow_indices().dtype(), false, true),
        self.col_indices().to(options.device(), self.col_indices().dtype(), false, true),
        at::empty(self.values().sizes(), options.layout(kStrided)),
        self.sizes(),
        optTypeMetaToScalarType(options.dtype()),
        self.layout(),
        options.device());
    return result;
  } else if (options.layout() == kSparseCsc) {
    auto result = at::native::_sparse_csc_tensor_unsafe(
        self.ccol_indices().to(options.device(), self.ccol_indices().dtype(), false, true),
        self.row_indices().to(options.device(), self.row_indices().dtype(), false, true),
        at::empty(self.values().sizes(), options.layout(kStrided)),
        self.sizes(),
        optTypeMetaToScalarType(options.dtype()),
        self.layout(),
        options.device());
    return result;
  } else if (options.layout() == kSparseBsr) {
    auto result = at::native::_sparse_bsr_tensor_unsafe(
        self.crow_indices().to(options.device(), self.crow_indices().dtype(), false, true),
        self.col_indices().to(options.device(), self.col_indices().dtype(), false, true),
        at::empty(self.values().sizes(), options.layout(kStrided)),
        self.sizes(),
        optTypeMetaToScalarType(options.dtype()),
        self.layout(),
        options.device());

    return result;
  } else if (options.layout() == kSparseBsc) {
    auto result = at::native::_sparse_bsc_tensor_unsafe(
        self.ccol_indices().to(options.device(), self.ccol_indices().dtype(), false, true),
        self.row_indices().to(options.device(), self.row_indices().dtype(), false, true),
        at::empty(self.values().sizes(), options.layout(kStrided)),
        self.sizes(),
        optTypeMetaToScalarType(options.dtype()),
        self.layout(),
        options.device());
    return result;
  } else if (options.layout() == kStrided) {
    return at::native::empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
  } else {
    TORCH_CHECK(false, "Layout ", options.layout(), " is not supported");
  }
}

template <bool require_view, bool require_copy>
static Tensor select_sparse_csr_worker(const Tensor& self, int64_t dim, int64_t index) {
#ifndef STRIP_ERROR_MESSAGES
  constexpr const char* select_name = (require_view ? "select()" : "select_copy()");
#endif
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(), "select", []() { return; });
  TORCH_CHECK_INDEX(
      self.dim() != 0, select_name, " cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, self.dim());
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    TORCH_CHECK_INDEX(
        false,
        select_name, ": index ",
        index,
        " out of range for tensor of size ",
        self.sizes(),
        " at dimension ",
        dim);
  }
  if (index < 0) {
    index += size;
  }

  auto select_strided = [](const Tensor& self, int64_t dim, int64_t index) {
    if (require_copy) {
      return at::select_copy(self, dim, index);
    } else {
      return self.select(dim, index);
    }
  };

  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < self.dim());

  auto new_sizes = DimVector(self.sizes());
  new_sizes.erase(new_sizes.begin() + dim);
  auto options = self.options();

  auto [compressed_indices, plain_indices] =
      AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
          self.layout(),
          "select",
          [&]() {
            return std::make_pair(self.crow_indices(), self.col_indices());
          },
          [&]() {
            return std::make_pair(self.ccol_indices(), self.row_indices());
          });
  auto n_batch = compressed_indices.dim() - 1;

  if (dim < n_batch) {
    // Selecting batch dimension
    return at::_sparse_compressed_tensor_unsafe(
        compressed_indices.select(dim, index),
        plain_indices.select(dim, index),
        select_strided(self.values(), dim, index),
        new_sizes,
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt());
  } else if (dim < n_batch + 2) {
    // Selecting sparse dimension
    TORCH_CHECK(
        n_batch == 0,
 
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/sparse`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/sparse`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/sparse`):

- [`ValidateCompressedIndicesKernel.cpp_docs.md_docs.md`](./ValidateCompressedIndicesKernel.cpp_docs.md_docs.md)
- [`SparseTensorMath.h_docs.md_docs.md`](./SparseTensorMath.h_docs.md_docs.md)
- [`SparseBlasImpl.h_kw.md_docs.md`](./SparseBlasImpl.h_kw.md_docs.md)
- [`SparseCsrTensorMath.h_docs.md_docs.md`](./SparseCsrTensorMath.h_docs.md_docs.md)
- [`SparseBlas.h_docs.md_docs.md`](./SparseBlas.h_docs.md_docs.md)
- [`FlattenIndicesKernel.cpp_kw.md_docs.md`](./FlattenIndicesKernel.cpp_kw.md_docs.md)
- [`SoftMax.cpp_docs.md_docs.md`](./SoftMax.cpp_docs.md_docs.md)
- [`SparseTensor.cpp_kw.md_docs.md`](./SparseTensor.cpp_kw.md_docs.md)
- [`SparseBinaryOpIntersectionKernel.cpp_docs.md_docs.md`](./SparseBinaryOpIntersectionKernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `SparseCsrTensor.cpp_docs.md_docs.md`
- **Keyword Index**: `SparseCsrTensor.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
