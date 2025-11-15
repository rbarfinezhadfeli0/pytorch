# Documentation: `aten/src/ATen/native/TensorShape.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/TensorShape.cpp`
- **Size**: 177,768 bytes (173.60 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/core/ATen_fwd.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/functional.h>
#include <ATen/native/Copy.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/cpu/StackKernel.h>
#include <ATen/quantized/QTensorImpl.h>
#include <c10/core/Contiguity.h>
#include <c10/core/GradMode.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <optional>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_chunk_cat_native.h>
#include <ATen/ops/_conj_copy_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_foreach_copy.h>
#include <ATen/ops/_fw_primal_copy_native.h>
#include <ATen/ops/_indices_copy_native.h>
#include <ATen/ops/_make_dual.h>
#include <ATen/ops/_make_dual_copy_native.h>
#include <ATen/ops/_mkldnn_reshape.h>
#include <ATen/ops/_mkldnn_transpose.h>
#include <ATen/ops/_neg_view_copy_native.h>
#include <ATen/ops/_reshape_alias_copy_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/_reshape_copy_native.h>
#include <ATen/ops/_reshape_from_tensor_native.h>
#include <ATen/ops/_shape_as_tensor_native.h>
#include <ATen/ops/_sparse_broadcast_to.h>
#include <ATen/ops/_sparse_broadcast_to_copy_native.h>
#include <ATen/ops/_sparse_broadcast_to_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_stack_native.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/_values_copy_native.h>
#include <ATen/ops/adjoint_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/alias_copy_native.h>
#include <ATen/ops/alias_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/as_strided_copy_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/as_strided_scatter_native.h>
#include <ATen/ops/atleast_1d.h>
#include <ATen/ops/atleast_2d.h>
#include <ATen/ops/atleast_3d.h>
#include <ATen/ops/block_diag_native.h>
#include <ATen/ops/broadcast_tensors_native.h>
#include <ATen/ops/broadcast_to_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cat_meta.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/chunk_native.h>
#include <ATen/ops/col_indices_copy_native.h>
#include <ATen/ops/column_stack_native.h>
#include <ATen/ops/concat_native.h>
#include <ATen/ops/concatenate_native.h>
#include <ATen/ops/crow_indices_copy_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/detach_copy_native.h>
#include <ATen/ops/detach_native.h>
#include <ATen/ops/diag.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/diag_embed_native.h>
#include <ATen/ops/diag_native.h>
#include <ATen/ops/diagflat_native.h>
#include <ATen/ops/diagonal.h>
#include <ATen/ops/diagonal_backward.h>
#include <ATen/ops/diagonal_backward_native.h>
#include <ATen/ops/diagonal_copy.h>
#include <ATen/ops/diagonal_copy_native.h>
#include <ATen/ops/diagonal_native.h>
#include <ATen/ops/diagonal_scatter_native.h>
#include <ATen/ops/dsplit_native.h>
#include <ATen/ops/dstack_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/expand_as_native.h>
#include <ATen/ops/expand_copy_native.h>
#include <ATen/ops/expand_native.h>
#include <ATen/ops/flatten_dense_tensors_native.h>
#include <ATen/ops/flatten_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/hsplit_native.h>
#include <ATen/ops/hstack.h>
#include <ATen/ops/hstack_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/indices_copy_native.h>
#include <ATen/ops/lift_fresh_native.h>
#include <ATen/ops/lift_native.h>
#include <ATen/ops/mH_native.h>
#include <ATen/ops/mT_native.h>
#include <ATen/ops/matrix_H_native.h>
#include <ATen/ops/meshgrid_native.h>
#include <ATen/ops/moveaxis_native.h>
#include <ATen/ops/movedim.h>
#include <ATen/ops/movedim_native.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/narrow_copy.h>
#include <ATen/ops/narrow_copy_native.h>
#include <ATen/ops/narrow_native.h>
#include <ATen/ops/new_empty_native.h>
#include <ATen/ops/new_ones_native.h>
#include <ATen/ops/numpy_T_native.h>
#include <ATen/ops/permute_copy_native.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/ravel_native.h>
#include <ATen/ops/repeat_native.h>
#include <ATen/ops/reshape_as_native.h>
#include <ATen/ops/reshape_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/row_stack_native.h>
#include <ATen/ops/select.h>
#include <ATen/ops/select_backward_native.h>
#include <ATen/ops/select_copy_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/select_scatter_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/slice_backward_native.h>
#include <ATen/ops/slice_copy_native.h>
#include <ATen/ops/slice_inverse_native.h>
#include <ATen/ops/slice_native.h>
#include <ATen/ops/slice_scatter_native.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_coo_tensor_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/split_copy_native.h>
#include <ATen/ops/split_native.h>
#include <ATen/ops/split_with_sizes.h>
#include <ATen/ops/split_with_sizes_copy_native.h>
#include <ATen/ops/split_with_sizes_native.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/squeeze_copy_native.h>
#include <ATen/ops/squeeze_native.h>
#include <ATen/ops/stack_native.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_to_size_native.h>
#include <ATen/ops/swapaxes_native.h>
#include <ATen/ops/swapdims_native.h>
#include <ATen/ops/t_copy_native.h>
#include <ATen/ops/t_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/tensor_split.h>
#include <ATen/ops/tensor_split_native.h>
#include <ATen/ops/tile_native.h>
#include <ATen/ops/transpose.h>
#include <ATen/ops/transpose_copy_native.h>
#include <ATen/ops/transpose_native.h>
#include <ATen/ops/unbind.h>
#include <ATen/ops/unbind_copy_native.h>
#include <ATen/ops/unbind_native.h>
#include <ATen/ops/unflatten_dense_tensors_native.h>
#include <ATen/ops/unflatten_native.h>
#include <ATen/ops/unfold_copy_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/unsafe_chunk_native.h>
#include <ATen/ops/unsafe_split_native.h>
#include <ATen/ops/unsafe_split_with_sizes_native.h>
#include <ATen/ops/unsqueeze_copy_native.h>
#include <ATen/ops/unsqueeze_native.h>
#include <ATen/ops/values_copy_native.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/view_as_complex_copy_native.h>
#include <ATen/ops/view_as_native.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/view_as_real_copy_native.h>
#include <ATen/ops/view_copy_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/vsplit_native.h>
#include <ATen/ops/vstack.h>
#include <ATen/ops/vstack_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros_native.h>
#endif

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace at::meta {

static inline c10::MemoryFormat cat_compute_output_memory_format(
    const MaterializedITensorListRef& inputs) {
  std::optional<c10::MemoryFormat> format = std::nullopt;
  for (const Tensor& t : inputs) {
    auto f = t.suggest_memory_format();
    if (f == c10::MemoryFormat::Contiguous) {
      return f;
    }
    if (format.has_value() && format.value() != f) {
      return c10::MemoryFormat::Contiguous;
    }
    format = f;
  }
  return format.value();
}

TORCH_PRECOMPUTE_META_FUNC(cat)(const ITensorListRef& tensors, int64_t dim) {
  // previously, size [0] tensors were the only possible empty tensors; thus, it
  // wasn't possible to cat empty tensors unless all the other tensors were
  // 1-dimensional, so we allowed these tensors to be "skipped".  We maintain
  // this behavior for backwards compatibility, but only for this specific size
  // (i.e. other empty sizes are not skipped).
  auto materialized = tensors.materialize();

  native::check_cat_no_zero_dim(materialized);
  dim = at::legacy_cat_wrap_dim(dim, materialized);

  // Checking names before the actual dimensions.
  auto maybe_outnames = namedinference::compute_cat_outnames(materialized);

  TORCH_CHECK_VALUE(
      !materialized.empty(),
      "torch.cat(): expected a non-empty list of Tensors");

  // Look for the first valid tensor.
  size_t valid = materialized.size();
  for (const auto i : c10::irange(materialized.size())) {
    if (!at::native::cat_should_skip_tensor(materialized[i].get())) {
      valid = i;
      break;
    }
  }

  bool all_contiguous = true;
  bool all_same_dtype = true;
  bool all_same_sizes_and_stride = true;
  auto memory_format = cat_compute_output_memory_format(materialized);

  // Compute what the output dtype should be:
  const auto& result = maybe_get_output();
  auto is_out_defined = result.defined();
  auto out_dtype = at::native::result_type(tensors);

  // If the output tensor is defined, we need to take it into account
  // when computing the actual output dtype and the flags.
  if (is_out_defined) {
    // Check for type promotion, if the output tensor is defined.
    TORCH_CHECK_TYPE(
        canCast(out_dtype, result.scalar_type()),
        "torch.cat(): input types can't be cast to the desired output type ",
        result.scalar_type());
    out_dtype = result.scalar_type();
    all_contiguous = result.is_contiguous(memory_format);
  }

  // Fallback 'set_output' parameters.
  // (in case we don't find a valid tensor)
  DimVector sizes{0};
  TensorOptions options =
      materialized[0].get().options().dtype(out_dtype).memory_format(
          memory_format);

  // If we found a valid tensor, check whether the input tensors
  // are compatible, i.e. we can execute `cat` on them.
  bool found_valid_tensor = valid < materialized.size();
  if (found_valid_tensor) {
    TORCH_CHECK_INDEX(
        dim <= materialized[valid].get().dim(),
        "torch.cat(): dimension ",
        dim,
        "out of range");

    // Compute the output tensor size.
    // It should have the same shape as any other valid tensor,
    // except in the dimension 'dim'.
    size_t size_at_dim = 0;
    for (const auto i : c10::irange(materialized.size())) {
      const Tensor& t = materialized[i];
      all_same_dtype = all_same_dtype && out_dtype == t.scalar_type();
      if (!at::native::cat_should_skip_tensor(t)) {
        at::native::check_cat_shape_except_dim(materialized[valid], t, dim, i);
        size_at_dim += t.size(dim);
        all_contiguous = all_contiguous && t.is_contiguous(memory_format);
        all_same_sizes_and_stride = all_same_sizes_and_stride &&
            t.sizes() == materialized[valid].get().sizes() &&
            t.strides() == materialized[valid].get().strides();
      } else {
        all_contiguous = false;
      }
    }

    // Actually set the output.
    sizes = materialized[valid].get().sizes().vec();
    sizes[dim] = size_at_dim;
    options =
        materialized[valid].get().options().dtype(out_dtype).memory_format(
            memory_format);
  }

  set_output_raw_strided(0, sizes, {}, options, maybe_outnames);
  // Checks for overlaps between the inputs and the output tensor.
  if (is_out_defined && found_valid_tensor) {
    at::assert_no_internal_overlap(result);
    for (const Tensor& t : materialized) {
      at::assert_no_overlap(result, t);
    }
  }

  return TORCH_PRECOMPUTE_STRUCT(cat)()
      .set_dim(dim)
      .set_valid(valid)
      .set_all_contiguous(all_contiguous)
      .set_all_same_dtype(all_same_dtype)
      .set_all_same_sizes_and_stride(all_same_sizes_and_stride)
      .set_memory_format(memory_format);
}
} // namespace at::meta

namespace at::native {

DEFINE_DISPATCH(cat_serial_stub);
DEFINE_DISPATCH(stack_serial_stub);

Tensor _reshape_from_tensor(const Tensor& self, const Tensor& shape_tensor) {
  TORCH_CHECK(shape_tensor.dim() == 1);
  std::vector<int64_t> shape;
  auto accessor = shape_tensor.accessor<int64_t, 1>();
  for (const auto i : c10::irange(shape_tensor.numel())) {
    shape.push_back(accessor[i]);
  }
  return self.reshape(IntArrayRef(shape));
}

Tensor _shape_as_tensor(const Tensor& self) {
  auto options = TensorOptions(at::kLong);
  return at::tensor(self.sizes(), options);
}

Tensor& set_(Tensor& result, Storage source) {
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / result.dtype().itemsize());
  return result.set_(std::move(source), 0, new_size, {});
}

// unify with cuda implementation?  This is not done to avoid a dispatch in
// resize_impl_cpu_
Tensor& set_storage_cpu_(
    Tensor& result,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  checkSetStorage(result, std::move(storage), storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt =
      stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : std::nullopt;
  // We can reuse this kernel for the meta device.
  // We just need to make sure we don't actually try to resize the (null)
  // storage.
  at::native::resize_impl_cpu_(
      result.unsafeGetTensorImpl(),
      size,
      stride_opt,
      /*resize_storage=*/!result.is_meta());
  return result;
}

Tensor& set_storage_meta__symint(
    Tensor& result,
    Storage storage,
    c10::SymInt storage_offset,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  checkSetStorage(
      result,
      storage,
      storage_offset,
      size,
      stride,
      /*check_offset_in_bounds=*/false);

  c10::SymDimVector contiguous_strides;
  if (stride.data() == nullptr) {
    // TODO: dedupe this with empty() symbolic logic
    int64_t dim = size.size();
    contiguous_strides.resize(dim);
    if (dim > 0) {
      const auto last_idx = dim - 1;
      contiguous_strides.at(last_idx) = 1;
      for (auto i = last_idx - 1; i >= 0; --i) {
        // TODO: max with 1
        contiguous_strides.at(i) =
            contiguous_strides.at(i + 1) * size.at(i + 1);
      }
    }
    stride = contiguous_strides;
  }

  // Run this before storage setting so we can access numel
  result.unsafeGetTensorImpl()->set_sizes_and_strides(
      size, stride, storage_offset);

  // Matches maybe_resize_storage_cpu no-numel behavior
  if (TORCH_GUARD_OR_TRUE(result.sym_numel().sym_ne(0))) {
    // maybe_resize_storage_cpu can handle no storage exists at all but
    // that should never be the case here
    TORCH_INTERNAL_ASSERT(storage);
    TORCH_CHECK(
        storage.resizable(), "Trying to resize storage that is not resizable");
    // All meta data pointers are the same, so we don't have to "re" allocate
    // it.  TODO: Actually this might not quite be correct if we use special
    // pointers to track whether or not fake cuda tensors are pinned or not

    // TODO: When there are unbacked SymInts, we unconditionally skip the
    // setter.  This is technically wrong, but we cannot conveniently test
    // the real condition in many cases, because a lot of people are using
    // set_ just to swizzle metadata on a tensor, they didn't actually want
    // to see if they need to resize the storage.
    //
    // The old behavior was to unconditionally set_nbytes, but I think not
    // setting it is more safe.
    if (result.sym_numel().has_hint()) {
      const auto itemsize = result.dtype().itemsize();

      c10::SymInt new_size_bytes = result.is_contiguous()
          ? at::detail::computeStorageNbytesContiguous(
                size, itemsize, std::move(storage_offset))
          : at::detail::computeStorageNbytes(
                size, stride, itemsize, std::move(storage_offset));

      if (new_size_bytes.has_hint() && storage.sym_nbytes().has_hint() &&
          (new_size_bytes > storage.sym_nbytes())) {
        storage.set_nbytes(std::move(new_size_bytes));
      }
    }
  }
  return result;
}

Tensor& set__symint(
    Tensor& result,
    const Tensor& storage,
    c10::SymInt storage_offset,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  TORCH_CHECK(
      storage.is_contiguous(),
      "passed in tensor to be used as storage must be contiguous");
  return result.set__symint(
      storage.storage(),
      storage_offset + storage.sym_storage_offset(),
      size,
      stride);
}

Tensor& set_tensor_(Tensor& result, const Tensor& source) {
  if (result.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
    return result.set__symint(
        source.storage(),
        source.sym_storage_offset(),
        source.sym_sizes(),
        source.sym_strides());
  }
  return result;
}

// this needs to be split along CPU/CUDA lines because we don't have a
// consistent way of getting the allocator to use for a device
// (c10::GetAllocator is not the same as at::cuda::getCUDADeviceAllocator().
Tensor& set_cpu_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(Storage::use_byte_size_t(), 0, c10::GetAllocator(kCPU), true);
  result.set_(std::move(storage), 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

// We can't reuse the cpu kernel here because we don't want to use the cpu
// allocator.
Tensor& set_meta_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(), 0, c10::GetAllocator(kMeta), true);
  result.set_(std::move(storage), 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

Tensor sparse_broadcast_to(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(self.is_sparse(), "input must be sparse tensor");

  const auto self_size = self.sizes();
  const int64_t new_sparse_dims = size.size() - self.dim();
  TORCH_CHECK(
      new_sparse_dims >= 0,
      "the requested broadcast shape has fewer dimensions than the input");
  const int64_t res_sparse_dim = new_sparse_dims + self.sparse_dim();

  for (int64_t i = 0; i < self.dim(); ++i) {
    TORCH_CHECK(
        self_size[i] == 1 || self_size[i] == size[i + new_sparse_dims],
        "The input's length ",
        self_size[i],
        " at dimension ",
        i,
        " does not broadcast over the requested shape of length ",
        size[i + new_sparse_dims],
        " at dimension ",
        i + new_sparse_dims);
  }

  const int64_t self_nnz = self._nnz();
  const auto self_indices = self._indices();
  int64_t nnz_expand_factor = 1;
  int64_t largest_sparse_dim_len = -1;
  int64_t min_broadcast_dim = (new_sparse_dims > 0) ? 0 : -1;
  int64_t max_unchanged_dim = -1;
  for (int64_t i = 0; i < res_sparse_dim; ++i) {
    if ((i < new_sparse_dims) || (self_size[i - new_sparse_dims] != size[i])) {
      nnz_expand_factor *= size[i];
      largest_sparse_dim_len = std::max(size[i], largest_sparse_dim_len);
      if (i >= new_sparse_dims && min_broadcast_dim == -1) {
        min_broadcast_dim = i;
      }
    } else {
      if (i >= new_sparse_dims) {
        max_unchanged_dim = i;
      }
    }
  }

  // to_broadcast conserves is_coalesced property iff only the last
  // sparse dimensions are expanded. Possible expansion of dense
  // dimensions can be discarded as it does not affect the is_coalesce
  // property.
  bool is_coalesced = !self.dim() ||
      (self.is_coalesced() &&
       (max_unchanged_dim < min_broadcast_dim || min_broadcast_dim == -1));

  // Replace non-broadcastable dims with 1 in the `size` vector {
  auto res_sparse_dim_broadcast_mask =
      at::DimVector(size.begin(), size.begin() + res_sparse_dim);
  for (int64_t i = new_sparse_dims; i < res_sparse_dim; ++i) {
    res_sparse_dim_broadcast_mask[i] =
        (size[i] == self_size[i - new_sparse_dims]) ? 1 : size[i];
  }
  // }

  // Then define for each sparse dim the number of reps for each nnz index/value
  // due to broadcasting. Repetitions do not take into account the current value
  // of nnz - this will be taken care of later {
  auto nnz_repeats = c10::DimVector(res_sparse_dim);
  nnz_repeats.back() = res_sparse_dim_broadcast_mask.back();
  for (int64_t i = res_sparse_dim - 2; i >= 0; --i) {
    nnz_repeats[i] = res_sparse_dim_broadcast_mask[i] * nnz_repeats[i + 1];
  }
  // }

  // Broadcast values. Each nnz value has to be repeated nnz_expand_factor times
  // {
  auto broadcast_values_shape = DimVector(size.size() - res_sparse_dim + 2);
  std::copy(
      size.begin() + res_sparse_dim,
      size.end(),
      broadcast_values_shape.begin() + 2);
  broadcast_values_shape[0] = self_nnz;
  broadcast_values_shape[1] = nnz_expand_factor;
  auto broadcast_values =
      self._values().unsqueeze(1).expand(broadcast_values_shape).flatten(0, 1);
  // }

  // We can return early if there are no broadcastable sparse dims
  if (largest_sparse_dim_len < 0) {
    return at::sparse_coo_tensor(
        self._indices(),
        broadcast_values,
        size,
        self.options(),
        self.is_coalesced());
  }

  auto broadcast_indices =
      self._indices().new_empty({res_sparse_dim, self_nnz * nnz_expand_factor});

  // Repeat each individual index value in dimension dim nnz_repeats[dim] /
  // size[dim] times, and then repeat the whole vector self_nnz *
  // (nnz_expand_factor / nnz_repeats[dim]) times to get the final index vector
  // - only for broadcast dims {
  const auto dim_arange =
      at::arange(largest_sparse_dim_len, self._indices().options());
  for (int64_t i = 0; i < res_sparse_dim; ++i) {
    Tensor curr_dim_idx;
    if ((i < new_sparse_dims) || (self_size[i - new_sparse_dims] != size[i])) {
      // If the dim is either a newly created sparse dim, or an already existing
      // one which is broadcastable, do the reps over an arange vector
      curr_dim_idx = dim_arange.narrow(0, 0, size[i])
                         .unsqueeze_(0)
                         .unsqueeze_(-1)
                         .expand(
                             {self_nnz * (nnz_expand_factor / nnz_repeats[i]),
                              size[i],
                              nnz_repeats[i] / size[i]});
    } else {
      // Otherwise over a slice of self._indices() of length self_nnz
      curr_dim_idx = self_indices.select(0, i - new_sparse_dims)
                         .unsqueeze_(1)
                         .expand({self_nnz, nnz_expand_factor});
    }
    broadcast_indices.select(0, i)
        .view(curr_dim_idx.sizes())
        .copy_(curr_dim_idx);
  }
  // }

  return at::sparse_coo_tensor(
      broadcast_indices, broadcast_values, size, self.options(), is_coalesced);
}

Tensor broadcast_to_symint(const Tensor& self, SymIntArrayRef size) {
  return self.expand_symint(size);
}

std::vector<Tensor> broadcast_tensors(TensorList tensors) {
  return expand_outplace(tensors);
}

static void fastCatOutDim0(
    const Tensor& out,
    const MaterializedITensorListRef& inputs) {
  auto outBytes = out.nbytes();
  char* dataPtr = reinterpret_cast<char*>(out.data_ptr());
  size_t totalBytes = 0;
  for (const Tensor& input : inputs) {
    TORCH_CHECK(outBytes >= totalBytes);
    if (input.nbytes() > 0) {
      std::memcpy(dataPtr + totalBytes, input.const_data_ptr(), input.nbytes());
    }
    totalBytes += input.nbytes();
  }
  TORCH_CHECK(outBytes == totalBytes);
}

TORCH_IMPL_FUNC(cat_out_cpu)
(const ITensorListRef& tensors,
 int64_t dim,
 int64_t valid,
 bool all_contiguous,
 bool all_same_dtype,
 bool all_same_sizes_and_stride,
 MemoryFormat memory_format,
 const Tensor& result) {
  if (result.numel() == 0) {
    return;
  }

  auto materialized = tensors.materialize();

  bool use_serial_kernel =
      result.numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1;
  ScalarType dtype = materialized[valid].get().scalar_type();
  bool serial_dtype = at::isFloatingType(dtype);
  // fast path for single thread when both inputs and result are contiguous and
  // not empty, and concat dim is 0
  if (use_serial_kernel && all_contiguous && all_same_dtype &&
      (MemoryFormat::Contiguous == memory_format)) {
    if (dim == 0) {
      fastCatOutDim0(result, materialized);
      return;
    }
    // TODO: Add fast cat for higher dimensions and support multi-threaded fast
    // cat
  }

  // fast path for single thread when both inputs and result are contiguous and
  // not empty
  if (use_serial_kernel && all_contiguous && all_same_dtype && serial_dtype) {
    cat_serial_stub(kCPU, result, materialized, dim);
    return;
  }

  int64_t offset = 0;
  if (all_same_sizes_and_stride && result.is_contiguous(memory_format) &&
      all_same_dtype) {
    const Tensor& source_slice = materialized[valid];
    auto slice_dim_size = source_slice.sizes()[dim];
    auto result_slice = result.narrow(dim, 0, slice_dim_size);
    auto result_slice_data = result_slice.data_ptr();
    auto result_stride_bytes =
        result.stride(dim) * elementSize(result.scalar_type());

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .resize_outputs(false)
                    .add_output(result_slice)
                    .add_const_input(source_slice)
                    .enforce_safe_casting_to_output(true)
                    .build();

    for (const Tensor& tensor : materialized) {
      if (cat_should_skip_tensor(tensor)) {
        continue;
      }
      auto source_data = static_cast<const char*>(tensor.const_data_ptr());
      auto result_data =
          static_cast<char*>(result_slice_data) + offset * result_stride_bytes;
      iter.unsafe_replace_operand(0, result_data);
      iter.unsafe_replace_operand(1, const_cast<char*>(source_data));
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  } else {
    for (const Tensor& tensor : materialized) {
      if (cat_should_skip_tensor(tensor)) {
        continue;
      }
      auto slice_dim_size = tensor.sizes()[dim];
      auto result_slice = result.narrow(dim, offset, slice_dim_size);

      auto iter = TensorIteratorConfig()
                      .set_check_mem_overlap(false) // Already checked above
                      .resize_outputs(false)
                      .add_output(result_slice)
                      .add_const_input(tensor)
                      .promote_inputs_to_common_dtype(true)
                      .cast_common_dtype_to_outputs(true)
                      .enforce_safe_casting_to_output(true)
                      .build();
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  }
}

Tensor& cat_out(TensorList tensors, Dimname dim, Tensor& result) {
  TORCH_CHECK_VALUE(!tensors.empty(), "expected a non-empty list of Tensors");
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

Tensor cat(TensorList tensors, Dimname dim) {
  TORCH_CHECK_VALUE(!tensors.empty(), "expected a non-empty list of Tensors");
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

// torch.concat, alias for torch.cat
Tensor& concat_out(TensorList tensors, Dimname dim, Tensor& result) {
  return cat_out(tensors, dim, result);
}

Tensor concat(TensorList tensors, Dimname dim) {
  return at::cat(tensors, dim);
}

Tensor& concat_out(TensorList tensors, int64_t dim, Tensor& result) {
  return at::cat_out(result, tensors, dim);
}

Tensor concat(TensorList tensors, int64_t dim) {
  return at::cat(tensors, dim);
}

// torch.concatenate, alias for torch.cat
Tensor& concatenate_out(TensorList tensors, Dimname dim, Tensor& result) {
  return cat_out(tensors, dim, result);
}

Tensor concatenate(TensorList tensors, Dimname dim) {
  return at::cat(tensors, dim);
}

Tensor& concatenate_out(TensorList tensors, int64_t dim, Tensor& result) {
  return at::cat_out(result, tensors, dim);
}

Tensor concatenate(TensorList tensors, int64_t dim) {
  return at::cat(tensors, dim);
}

static bool sizes_match_except(
    IntArrayRef s1,
    IntArrayRef s2,
    int64_t dim_except /* should already be wrapped */) {
  if (s1.size() != s2.size()) {
    return false;
  }
  for (const auto i : c10::irange(static_cast<int64_t>(s1.size()))) {
    if (i != dim_except && s1[i] != s2[i]) {
      return false;
    }
  }
  return true;
}

// Check to see if the shape of tensors is compatible
// for being concatenated along a given dimension.
static void check_cat_sparse_dims(
    Tensor const& t,
    int64_t pos /* used only for debug messages */,
    IntArrayRef sizes,
    int64_t wrapped,
    int64_t sparse_dim,
    int64_t dense_dim) {
  TORCH_CHECK(
      t.is_sparse(),
      "Concatenating sparse tensors, but a dense tensor was found at position ",
      pos,
      ".");
  TORCH_CHECK(
      sizes_match_except(sizes, t.sizes(), wrapped),
      "All tensors must have the same shape: ",
      sizes,
      " (except in the concatenating dimension),"
      " but found shape: ",
      t.sizes(),
      " at position ",
      pos,
      ".");
  TORCH_CHECK(
      t.sparse_dim() == sparse_dim && t.dense_dim() == dense_dim,
      "All tensors must have the same sparse_dim and dense_dim: ",
      sparse_dim,
      ", ",
      dense_dim,
      ", but tensor at position ",
      pos,
      " has ",
      t.sparse_dim(),
      ", ",
      t.dense_dim(),
      ".");
}

static Tensor cat_sparse_impl(
    const MaterializedITensorListRef& tensors,
    int64_t dim) {
  std::vector<Tensor> indices;
  std::vector<Tensor> values;
  int64_t wrapped = maybe_wrap_dim(dim, tensors[0].get().dim());
  int64_t sparse_dim = tensors[0].get().sparse_dim();
  int64_t dense_dim = tensors[0].get().dense_dim();
  IntArrayRef sizes = tensors[0].get().sizes();
  if (wrapped < sparse_dim) {
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      check_cat_sparse_dims(t, i, sizes, wrapped, sparse_dim, dense_dim);
      indices.push_back(t._indices());
      values.push_back(t._values());
    }
    Tensor idxs = at::cat(indices, 1);
    Tensor vals = at::cat(values, 0);

    // We now need to move the indices of each
    // input tensor up along `dim` by an appropriate amount.
    // E.g., if t1 has indices [[2,3,4],[5,6,7]],
    // and sizes [10, 7]
    // then torch.cat((t1,t1,t1),1) should have indices
    // [[2,3,4,2,3,4,2,3,4],[5,6,7,12,13,14,19,20,21]],
    // so we need to increase idxs[1][3:6] by 7
    // and idxs[1][6:9] by 14.
    int64_t col = 0;
    int64_t cumulative_offset = 0;
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      int64_t this_piece_size = t._nnz();
      // cumulative_offset is zero for the first piece, so
      // don't waste time doing this operation unless i > 0.
      if (i > 0) {
        idxs[wrapped].narrow(0, col, this_piece_size) += cumulative_offset;
      }
      cumulative_offset += t.size(wrapped);
      col += this_piece_size;
    }
    auto sizes_copy = sizes.vec();
    sizes_copy[wrapped] = cumulative_offset;
    return native::sparse_coo_tensor(
        idxs,
        vals,
        sizes_copy,
        optTypeMetaToScalarType(tensors[0].get().options().dtype_opt()),
        tensors[0].get().options().layout_opt(),
        tensors[0].get().options().device_opt(),
        tensors[0].get().options().pinned_memory_opt());
  } else {
    // Catting along a dense dimension requires us to create new values.
    // For illustration, consider the sparse 3d tensors t1 and t2,
    // given by t1 = [[[1,2],[3,4]], ... (zeros) ..., [[5,6],[7,8]]]
    // and t2 = [... (zeros) ..., [[9, 10], [11,12]], ... (zeros) ...],
    // Their concatenation along dimension 2 is:
    // [[[1,2,0,0],[3,4,0,0]], ... (zeros) ..., [[0,0,9,10],[0,0,11,12]], ...
    // (zeros) ..., [[5,6,0,0],[7,8,0,0]]]
    //
    // Their values tensors are, respectively,
    // [[[1,2],[3,4]],[[5,6],[7,8]]] and [[[9,10],[11,12]]].
    //
    // and so the values tensor of their concatenation along dim 2 will be:
    // [[[1,2,0,0],[3,4,0,0]],[[5,6,0,0],[7,8,0,0]],[[0,0,9,10],[0,0,11,12]]]
    //
    // which we can get by taking the values tensor of each tensor, catting it
    // with zeros of the appropriate size on the left and right, and then
    // catting all those results together.

    // The dimension in each tensor's values object that corresponds to the
    // overall dimension along which we're catting.
    int64_t values_dim = wrapped - sparse_dim + 1;
    // The final size along the catted dimension.
    const int64_t total_size = std::accumulate(
        tensors.begin(),
        tensors.end(),
        static_cast<int64_t>(0),
        [values_dim](int64_t l, const Tensor& r) {
          return l + r._values().size(values_dim);
        });
    auto zeros_sizes = tensors[0].get()._values().sizes().vec();
    int64_t cumulative_size = 0;
    std::vector<Tensor> vals_pieces;
    std::vector<Tensor> idxs_pieces;
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      check_cat_sparse_dims(t, i, sizes, wrapped, sparse_dim, dense_dim);
      // dimension 0 of values corresponds to the number of values,
      // rather than to any logical dimension of the sparse tensor.
      zeros_sizes[0] = t._values().size(0);
      zeros_sizes[values_dim] = cumulative_size;
      cumulative_size += t._values().size(values_dim);
      auto z1 = at::zeros(
          zeros_sizes,
          optTypeMetaToScalarType(t._values().options().dtype_opt()),
          t._values().options().layout_opt(),
          t._values().options().device_opt(),
          t._values().options().pinned_memory_opt());
      zeros_sizes[values_dim] = total_size - cumulative_size;
      auto z2 = at::zeros(
          zeros_sizes,
          optTypeMetaToScalarType(t._values().options().dtype_opt()),
          t._values().options().layout_opt(),
          t._values().options().device_opt(),
          t._values().options().pinned_memory_opt());
      vals_pieces.push_back(at::cat({z1, t._values(), z2}, values_dim));
      idxs_pieces.push_back(t._indices());
    }
    auto sizes_copy = sizes.vec();
    sizes_copy[wrapped] = total_size;
    // This can create an uncoalesced tensor
    return native::sparse_coo_tensor(
        at::cat(idxs_pieces, 1),
        at::cat(vals_pieces),
        sizes_copy,
        optTypeMetaToScalarType(tensors[0].get().options().dtype_opt()),
        tensors[0].get().options().layout_opt(),
        tensors[0].get().options().device_opt(),
        tensors[0].get().options().pinned_memory_opt());
  }
}

Tensor cat_sparse(const ITensorListRef& tensors, int64_t dim) {
  auto materialized = tensors.materialize();
  auto maybe_outnames = namedinference::compute_cat_outnames(materialized);
  auto result =
      cat_sparse_impl(materialized, at::legacy_cat_wrap_dim(dim, materialized));
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor block_diag(TensorList tensors) {
  Tensor result;
  if (tensors.empty()) {
    result = at::empty({1, 0});
    return result;
  }

  const Device& device = tensors[0].device();
  for (const auto tensor_idx : c10::irange(tensors.size())) {
    const Tensor& tensor = tensors[tensor_idx];

    TORCH_CHECK(
        tensor.device() == device,
        "torch.block_diag: input tensors must all be on the same device.",
        " Input 0 is on device ",
        device,
        " and input ",
        tensor_idx,
        " is on device ",
        tensor.device());
  }

  ScalarType output_scalar_type = native::result_type(tensors);
  int64_t result_dim0 = 0;
  int64_t result_dim1 = 0;
  std::vector<Tensor> tensors_2D(tensors.size());

  // Sum the dimensions of the tensors, check tensor sizes,
  // and expand all 0-D and 1-D tensors so that everything
  // is 2-D
  for (const auto tensor_idx : c10::irange(tensors.size())) {
    const Tensor& tensor = tensors[tensor_idx];
    int64_t ndims = tensor.dim();
    TORCH_CHECK(
        ndims <= 2,
        "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input ",
        tensor_idx,
        " has ",
        ndims,
        " dimensions");

    int64_t dim0 = 1;
    int64_t dim1 = 1;

    if (ndims == 2) {
      dim0 = tensor.size(0);
      dim1 = tensor.size(1);
      tensors_2D[tensor_idx] = tensor;
    } else if (ndims == 1) {
      // Switching dim 0 to dim 1 is intentional
      dim1 = tensor.size(0);
      tensors_2D[tensor_idx] = tensor.expand({dim0, dim1});
    } else {
      tensors_2D[tensor_idx] = tensor.expand({dim0, dim1});
    }
    result_dim0 += dim0;
    result_dim1 += dim1;
  }

  result = at::zeros(
      {result_dim0, result_dim1},
      tensors[0].options().dtype(output_scalar_type));

  int64_t cur_dim0 = 0;
  int64_t cur_dim1 = 0;

  // Copy each tensor into the appropriate location in the result matrix
  for (const auto& tensor : tensors_2D) {
    int64_t dim0 = tensor.size(0);
    int64_t dim1 = tensor.size(1);
    result.slice(0, cur_dim0, cur_dim0 + dim0)
        .slice(1, cur_dim1, cur_dim1 + dim1)
        .copy_(tensor);

    cur_dim0 += dim0;
    cur_dim1 += dim1;
  }

  return result;
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "chunk expects at least a 1-dimensional tensor");
  TORCH_CHECK(
      chunks > 0, "chunk expects `chunks` to be greater than 0, got: ", chunks);

  const auto dim_size = self.sym_size(dim);
  auto split_size = (dim_size + chunks - 1) / chunks;

  // We need to call split_with_sizes in the case where split_size and dimension
  // size are 0, because a call to split would discard the number of chunks
  // (because we can have an arbitrary number of 0-sized chunks adding up to 0).
  // So, call split_with_sizes with the correct number of chunks, eventually we
  // will do this for all cases.
  if (split_size == 0 && dim_size == 0) {
    std::vector<c10::SymInt> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
    return self.split_with_sizes_symint(split_sizes, dim);
  } else {
    return self.split_symint(std::move(split_size), dim);
  }
}

std::vector<Tensor> tensor_split_sections_symint(
    const Tensor& self,
    c10::SymInt sym_sections,
    int64_t dim) {
  TORCH_CHECK(
      self.dim() > 0,
      "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ",
      self.dim(),
      " dims");
  int64_t dim_ = maybe_wrap_dim(dim, self.dim());
  // NB: intentional, sections specifies number of output tensors, which
  // cannot be polymorphic
  int64_t sections = sym_sections.guard_int(__FILE__, __LINE__);
  TORCH_CHECK(
      sections > 0, "number of sections must be larger than 0, got ", sections);
  const auto dim_size = self.sym_size(dim_);
  std::vector<Tensor> splits(sections);
  auto min_split_size = dim_size / sections;
  auto num_splits_one_extra = dim_size % sections;
  c10::SymInt start_idx = 0;
  for (const auto split_idx : c10::irange(sections)) {
    auto split_size = (num_splits_one_extra > split_idx) ? (min_split_size + 1)
                                                         : min_split_size;
    splits[split_idx] =
        at::slice_symint(self, dim_, start_idx, start_idx + split_size);
    start_idx += split_size;
  }
  return splits;
}

template <typename T>
static std::vector<Tensor> _tensor_split_indices(
    const Tensor& self,
    ArrayRef<T> indices,
    int64_t dim) {
  TORCH_CHECK(
      self.dim() > 0,
      "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ",
      self.dim(),
      " dims");
  int64_t dim_ = maybe_wrap_dim(dim, self.dim());
  int64_t num_indices = indices.size();
  std::vector<Tensor> splits(num_indices + 1);
  T start_idx(0);
  for (const auto split_idx : c10::irange(num_indices)) {
    auto end_idx = indices[split_idx];
    splits[split_idx] = at::symint::slice<T>(self, dim_, start_idx, end_idx);
    start_idx = end_idx;
  }
  splits[num_indices] = at::symint::slice<T>(
      self, dim_, start_idx, at::symint::size<T>(self, dim_));
  return splits;
}

std::vector<Tensor> tensor_split(
    const Tensor& self,
    IntArrayRef indices,
    int64_t dim) {
  return _tensor_split_indices(self, indices, dim);
}

std::vector<Tensor> tensor_split_indices_symint(
    const Tensor& self,
    SymIntArrayRef indices,
    int64_t dim) {
  return _tensor_split_indices(self, indices, dim);
}

std::vector<Tensor> tensor_split(
    const Tensor& self,
    const Tensor& tensor_indices_or_sections,
    int64_t dim) {
  TORCH_CHECK(
      self.dim() > 0,
      "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ",
      self.dim(),
      " dims");
  auto split_device = tensor_indices_or_sections.device();
  TORCH_CHECK(
      split_device == kCPU,
      "tensor_split expected tensor_indices_or_sections to be on cpu, but it's on ",
      split_device);
  auto split_dtype = tensor_indices_or_sections.scalar_type();
  TORCH_CHECK(
      split_dtype == at::kLong,
      "tensor_split expected tensor_indices_or_sections to have dtype of long, but got ",
      split_dtype);
  auto split_dim = tensor_indices_or_sections.dim();
  TORCH_CHECK(
      split_dim == 1 || split_dim == 0,
      "tensor_split expected tensor_indices_or_sections to be a zero-dimensional or one-dimensional tensor, but got a tensor with ",
      split_dim,
      " dims");

  if (split_dim == 0) {
    int64_t sections = tensor_indices_or_sections.item<int64_t>();
    return self.tensor_split(sections, dim);
  } else {
    auto indices_data = tensor_indices_or_sections.const_data_ptr<int64_t>();
    auto stride = tensor_indices_or_sections.stride(0);
    auto numel = tensor_indices_or_sections.numel();
    std::vector<int64_t> indices(numel);
    for (const auto offset : c10::irange(numel)) {
      // indices tensor could be non-contiguous
      indices[offset] = *(indices_data + offset * stride);
    }
    return self.tensor_split(indices, dim);
  }
}

std::vector<Tensor> unsafe_chunk(
    const Tensor& self,
    int64_t chunks,
    int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "chunk expects at least a 1-dimensional tensor");
  TORCH_CHECK(
      chunks > 0, "chunk expects `chunks` to be greater than 0, got: ", chunks);

  const auto dim_size = self.size(dim);
  int64_t split_size = (dim_size + chunks - 1) / chunks;

  // See the comment above in chunk(...)
  if (split_size == 0 && dim_size == 0) {
    std::vector<int64_t> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
    return self.unsafe_split_with_sizes(split_sizes, dim);
  } else {
    return self.unsafe_split(split_size, dim);
  }
}

Tensor diagflat(const Tensor& self, int64_t offset) {
  return self.contiguous().view(-1).diag(offset);
}

Tensor diagonal(
    const Tensor& self,
    int64_t offset,
    int64_t dim1_,
    int64_t dim2_) {
  int64_t nDims = self.dim();
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  TORCH_CHECK(
      dim1 != dim2,
      "diagonal dimensions cannot be identical ",
      dim1_,
      ", ",
      dim2_);
  auto outnames = namedinference::compute_diagonal_outnames(self, dim1, dim2);
  NoNamesGuard no_names_guard;

  int64_t diag_size = 0;
  int64_t storage_offset = self.storage_offset();
  // compute storage offset and size for the diagonal
  // for positive values of offset (above the main diagonal)
  // "leftmost columns" (along dim2) are dropped
  // for negative values of offset (below the main diagonal)
  // "topmost rows" (along dim1) are dropped.
  // Note that we invert +/- in the second to absorb the negative
  // sign in the offset.
  if (offset >= 0) {
    diag_size = std::max<int64_t>(
        std::min(self.size(dim1), self.size(dim2) - offset), 0);
  } else {
    diag_size = std::max<int64_t>(
        std::min(self.size(dim1) + offset, self.size(dim2)), 0);
  }

  // NumPy allows you to specify offsets "off the end"; let's just be careful
  // not to set a ridiculous storage_offset in that case (technically it
  // shouldn't matter because there are no elements in the tensor, but let's be
  // kosher).
  if (diag_size == 0) {
    // skip
  } else if (offset >= 0) {
    storage_offset += offset * self.stride(dim2);
  } else {
    storage_offset -= offset * self.stride(dim1);
  }

  // construct new size and stride: we drop dim1 and dim2 (maximum first for not
  // changing the index of the minimum) the new ("joint") dimension is appended
  // to the end of the shape / stride to match numpy semantics
  DimVector sizes(self.sizes().begin(), self.sizes().end());
  DimVector strides(self.strides().begin(), self.strides().end());
  sizes.erase(sizes.begin() + std::max(dim1, dim2));
  strides.erase(strides.begin() + std::max(dim1, dim2));
  sizes.erase(sizes.begin() + std::min(dim1, dim2));
  strides.erase(strides.begin() + std::min(dim1, dim2));
  sizes.push_back(diag_size);
  strides.push_back(self.stride(dim1) + self.stride(dim2));

  // return view with new parameters
  auto result = self.as_strided(sizes, strides, storage_offset);

  no_names_guard.reset();
  namedinference::propagate_names_if_nonempty(result, outnames);
  return result;
}

Tensor diagonal(
    const Tensor& self,
    Dimname outdim,
    Dimname dim1,
    Dimname dim2,
    int64_t offset) {
  auto result = at::diagonal(
      self,
      offset,
      dimname_to_position(self, dim1),
      dimname_to_position(self, dim2));
  // This is slower than it needs to be because there is no way to modify
  // the names of a tensor in-place right now. In the future we should consider
  // offering that functionality.
  std::vector<Dimname> new_names = result.names().vec();
  new_names[new_names.size() - 1] = outdim;
  return result.refine_names(new_names);
}

Tensor diag_embed(
    const Tensor& self,
    int64_t offset,
    int64_t dim1_,
    int64_t dim2_) {
  int64_t nDims = self.dim() + 1;
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  TORCH_CHECK(
      dim1 != dim2,
      "diagonal dimensions cannot be identical ",
      dim1_,
      ", ",
      dim2_);
  int64_t new_dim_len = std::abs(offset) + self.size(-1);
  auto sizes = self.sizes().vec();
  sizes.pop_back();
  sizes.insert(sizes.begin() + std::min(dim1, dim2), new_dim_len);
  sizes.insert(sizes.begin() + std::max(dim1, dim2), new_dim_len);
  auto result = at::zeros(sizes, self.options());
  auto diag = result.diagonal(offset, dim1, dim2);
  diag.copy_(self);
  return result;
}

Tensor expand(const Tensor& self, c10::IntArrayRef size, bool /*unused*/) {
  TORCH_CHECK(
      size.size() >= (size_t)self.dim(),
      "expand(",
      self.toString(),
      "{",
      self.sizes(),
      "}, size=",
      size,
      "): the number of sizes provided (",
      size.size(),
      ") ",
      "must be greater or equal to the number of dimensions in the tensor (",
      self.dim(),
      ")");
  TORCH_CHECK(
      !self.is_sparse() && !at::sparse_csr::is_sparse_compressed(self),
      "expand is unsupported for ",
      self.layout(),
      " tensors");

  auto expandedSizesAndStrides =
      inferExpandGeometry_dimvector(self.sizes(), self.strides(), size);

  auto result = self.as_strided(
      expandedSizesAndStrides.sizes, expandedSizesAndStrides.strides);
  namedinference::propagate_names_for_expand(result, self);
  return result;
}

Tensor expand_as(const Tensor& self, const Tensor& other) {
  return self.expand_symint(other.sym_sizes());
}

Tensor sum_to_size_symint(const Tensor& self, SymIntArrayRef size) {
  TORCH_CHECK(
      is_expandable_to(size, self.sym_sizes()),
      "size {",
      size,
      "} is not expandable to size {",
      self.sizes(),
      "}.");

  return sum_to(self, size);
}

// We currently do not support per-channel quant for unfold, diagonal, expand,
// permute.
// TODO: Make this an aten function and replace as_strided_qtensorimpl once that
// is done.
static Tensor make_qtensor(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    QuantizerPtr quantizer) {
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW,
      Storage(self.storage()),
      self.key_set(),
      self.dtype(),
      quantizer);
  setStrided(result, size, stride, self.storage_offset());
  return result;
}

Tensor as_strided_tensorimpl(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW,
      Storage(self.storage()),
      self.key_set(),
      self.dtype());
  setStrided(result, size, stride, storage_offset);
  return result;
}

template <typename T>
static inline void setStridedUnchecked(
    const Tensor& self,
    ArrayRef<T> size,
    ArrayRef<T> stride,
    T&& storage_offset) {
  au
```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 282 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `void`, `at`

**Classes/Structs**: `new`, `TransposeDim`, `the`, `InferUnsqueezeGeometryResult`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ATen_fwd.h`
- `c10/core/ScalarType.h`
- `c10/core/SymInt.h`
- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/ExpandUtils.h`
- `ATen/InferSize.h`
- `ATen/MemoryOverlap.h`
- `ATen/NamedTensorUtils.h`
- `ATen/SparseCsrTensorUtils.h`
- `ATen/TensorOperators.h`
- `ATen/TensorSubclassLikeUtils.h`
- `ATen/WrapDimUtils.h`
- `ATen/core/DimVector.h`
- `ATen/core/IListRef.h`
- `ATen/core/Tensor.h`
- `ATen/core/functional.h`
- `ATen/native/Copy.h`
- `ATen/native/NonSymbolicBC.h`
- `ATen/native/Resize.h`
- `ATen/native/SparseTensorUtils.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/TensorShape.h`
- `ATen/native/TypeProperties.h`
- `ATen/native/cpu/CatKernel.h`
- `ATen/native/cpu/SerialStackImpl.h`
- `ATen/native/cpu/StackKernel.h`
- `ATen/quantized/QTensorImpl.h`
- `c10/core/Contiguity.h`
- `c10/core/GradMode.h`


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

- **File Documentation**: `TensorShape.cpp_docs.md`
- **Keyword Index**: `TensorShape.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
