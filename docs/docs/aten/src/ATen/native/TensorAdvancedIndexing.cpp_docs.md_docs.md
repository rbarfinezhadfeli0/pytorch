# Documentation: `docs/aten/src/ATen/native/TensorAdvancedIndexing.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/TensorAdvancedIndexing.cpp_docs.md`
- **Size**: 53,389 bytes (52.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/TensorAdvancedIndexing.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/TensorAdvancedIndexing.cpp`
- **Size**: 105,889 bytes (103.41 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// Indexing tensors by by tensors
//
// This corresponds to "advanced indexing" in NumPy. The two operations are:
//
//  index(Tensor self, indices) -> Tensor
//  index_put_(Tensor self, indices, value, accumulate=false)
//
// The index is a TensorList containing kLong, kBool or kByte tensors or nulls.
// Byte tensors (boolean masks) are expanded to long tensors via nonzero(). Null
// tensors signify that the dimension is not indexed.
//
// All indexes are broadcast together and iterated as *one*. From NumPy:
//
// result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
//                           ..., ind_N[i_1, ..., i_M]]
//
// Note 1: ByteTensors expand to index as many dimensions as there are in the
// mask.
//
// Note 2: The behavior is more complicated when the index tensors are not all
// adjacent (e.g. x[[0, 1], :, [2, 3]]). In this case, self and the index
// tensors are transposed to the front: x.transpose(1, 2)[[0, 1], [2, 3]]
//
// The code contains two implementations of indexing. The more efficient
// implementation treats indexing like an elementwise operation over the
// tensors `result`, `x`, `ind_1`, `ind_2`, etc. This implementation does
// not work for index_put_ with accumulate=True. The other implementation
// combines the indexed tensors into a single linear index that is used
// with Tensor.put_. This is used for index_put_ with accumulate=True.
//
// The more efficient implementation takes the following steps for the
// above operation:
//
// 1) Broadcast ind_1, ind_2, ind_3 together to a common shape
// 2) Record x.stride(i) for each indexed dimension `i`
// 3) Replace the indexed subspace of `x` with the shape of the corresponding
//    subspace of `result` but with stride 0
// 4) Add dimensions of size 1 to the index tensors (ind_1, ind_2, etc.) so
//    that their shape is compatible with the result shape
//
// The CPU or CUDA kernel then computes element-wise over the broadcasted
// and restrided result, x, ind_1,  ind_2, etc.:
//
//   result[...] = *(&x[...] +
//                   ind_1[...] * x.stride(1) +
//                   ind_2[...] * x.stride(2) +
//                   ...)
//
// where & and * represent the C-style address-of and indirection operations.
// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ATen.h>

#include <ATen/native/IndexKernel.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorAdvancedIndexing.h>

#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NumericUtils.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ScatterGatherChecks.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_gather_sparse_backward.h>
#include <ATen/ops/_gather_sparse_backward_native.h>
#include <ATen/ops/_index_put_impl.h>
#include <ATen/ops/_index_put_impl_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/_unsafe_index_native.h>
#include <ATen/ops/_unsafe_index_put_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/argwhere_native.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/broadcast_to.h>
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/count_nonzero_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/gather_backward_native.h>
#include <ATen/ops/gather_meta.h>
#include <ATen/ops/gather_native.h>
#include <ATen/ops/index.h>
#include <ATen/ops/index_add_meta.h>
#include <ATen/ops/index_add_native.h>
#include <ATen/ops/index_copy_meta.h>
#include <ATen/ops/index_copy_native.h>
#include <ATen/ops/index_fill_native.h>
#include <ATen/ops/index_meta.h>
#include <ATen/ops/index_native.h>
#include <ATen/ops/index_put_native.h>
#include <ATen/ops/index_reduce_meta.h>
#include <ATen/ops/index_reduce_native.h>
#include <ATen/ops/index_select_backward_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/masked_scatter_native.h>
#include <ATen/ops/masked_select_backward_native.h>
#include <ATen/ops/masked_select_native.h>
#include <ATen/ops/nested_to_padded_tensor_native.h>
#include <ATen/ops/nonzero_native.h>
#include <ATen/ops/nonzero_numpy_native.h>
#include <ATen/ops/nonzero_static_native.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/put_native.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/scatter_add_meta.h>
#include <ATen/ops/scatter_add_native.h>
#include <ATen/ops/scatter_meta.h>
#include <ATen/ops/scatter_native.h>
#include <ATen/ops/scatter_reduce_meta.h>
#include <ATen/ops/scatter_reduce_native.h>
#include <ATen/ops/take_along_dim_native.h>
#include <ATen/ops/take_native.h>
#include <ATen/ops/zeros_like.h>
#endif

#ifdef USE_FBGEMM
#include <fbgemm/Utils.h>
#endif

#include <c10/util/Unroll.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace at::meta {

TORCH_META_FUNC(gather)
(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad) {
  const Tensor& result = maybe_get_output(0);
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());

  // Memory overlap checks need to be done after resizing (if required) is done.
  // But it only makes sense to do these checks when result was defined, hence
  // the boolean variable `check_result` here.
  // For more details, see:
  // https://github.com/pytorch/pytorch/pull/63312#discussion_r694794832 and
  // https://github.com/pytorch/pytorch/issues/63837
  bool check_result = result.defined();
  set_output_raw_strided(0, index.sizes(), {}, self.options());
  if (check_result) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, self);
    at::assert_no_partial_overlap(result, index);
  }

  auto is_index_empty = index.numel() == 0;
  if (!is_index_empty) {
    TORCH_CHECK(
        index.scalar_type() == ScalarType::Long ||
            index.scalar_type() == ScalarType::Int,
        "gather",
        "(): Expected dtype int32/int64 for index");
  }
  if (is_index_empty)
    return;
  at::native::gather_shape_check(self, wrapped_dim, index);
}

template <bool use_new_options = false, typename Meta>
static void scatter_meta_impl(
    Meta& meta,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const std::optional<Tensor>& src = std::nullopt,
    const std::optional<std::string_view> reduce = std::nullopt) {
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  at::native::scatter_gather_dtype_check("scatter", self, index, src);
  at::native::scatter_shape_check(self, wrapped_dim, index, src);
  auto output = meta.maybe_get_output(0);

  if (output.defined()) {
    at::assert_no_internal_overlap(output);
    at::assert_no_overlap(output, index);
    if (src.has_value()) {
      at::assert_no_overlap(output, src.value());
    }
  }

  meta.set_output_raw_strided(0, self.sizes(), {}, self.options());
  if (reduce.has_value()) {
    // Check if we have a valid reduce operator.
    at::native::get_operator_enum(reduce.value(), use_new_options);
  }
}

TORCH_META_FUNC2(scatter, src)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  scatter_meta_impl(*this, self, dim, index, src);
}

TORCH_META_FUNC2(scatter, value)
(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
  scatter_meta_impl(*this, self, dim, index);
}

TORCH_META_FUNC2(scatter, reduce)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const std::string_view reduce) {
  TORCH_WARN_ONCE(
      "The reduce argument of torch.scatter with Tensor src is deprecated and will be removed ",
      "in a future PyTorch release. Use torch.scatter_reduce instead for more reduction options.");
  scatter_meta_impl(*this, self, dim, index, src, reduce);
}

TORCH_META_FUNC2(scatter, value_reduce)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& src,
 const std::string_view reduce) {
  scatter_meta_impl(*this, self, dim, index, std::nullopt, reduce);
}

TORCH_META_FUNC(scatter_add)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  scatter_meta_impl(*this, self, dim, index, src, "add");
}

TORCH_META_FUNC2(scatter_reduce, two)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const std::string_view reduce,
 bool include_self) {
  (void)include_self;
  scatter_meta_impl</*use_new_options=*/true>(
      *this, self, dim, index, src, reduce);
}

TORCH_PRECOMPUTE_META_FUNC(index_copy)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
  dim = maybe_wrap_dim(dim, self.dim());

  const Tensor& result = maybe_get_output(0);

  // Memory overlap checks need to be done after resizing (if required) is done.
  // But it only makes sense to do these checks when result was defined, hence
  // the boolean variable `check_result` here.
  // For more details, see:
  // https://github.com/pytorch/pytorch/pull/63312#discussion_r694794832 and
  // https://github.com/pytorch/pytorch/issues/63837
  bool check_result = result.defined();
  set_output_raw_strided(0, self.sizes(), {}, self.options());
  if (check_result) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, index);
    at::assert_no_overlap(result, source);
  }

  TORCH_CHECK_INDEX(
      index.dim() < 2,
      "index_copy_(): Index should have dimension 1 or 0 (got ",
      index.dim(),
      ")");

  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
    TORCH_CHECK_INDEX(
        false,
        "index_copy_(): When source is scalar, index should have one element (got ",
        numIndices,
        ")");
  } else if (
      (source.dim() != self.dim()) && (source.dim() != 0 && self.dim() != 0)) {
    TORCH_CHECK_INDEX(
        false,
        "index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality (",
        source.dim(),
        "), destination dimensionality (",
        self.dim(),
        ")");
  }

  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long,
      "index_copy_(): Expected a long tensor for index, but got ",
      index.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "index_copy_(): self and source expected to have the same dtype, but got (self) ",
      self.scalar_type(),
      " and (source) ",
      source.scalar_type());
  TORCH_CHECK(
      self.device() == source.device() && self.device() == index.device(),
      "index_copy_(): self, index and source expected to be in the same device, but got (self) ",
      self.device(),
      ", (index) ",
      index.device(),
      ", and (source) ",
      source.device());

  // Check that source and destination slices have the same size
  auto selfSlicedSizes = self.sizes().vec();
  if (!selfSlicedSizes.empty()) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
  }
  auto sourceSlicedSizes = source.sizes().vec();
  if (!sourceSlicedSizes.empty()) {
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);
  }
  if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
      !std::equal(
          selfSlicedSizes.begin(),
          selfSlicedSizes.end(),
          sourceSlicedSizes.begin())) {
    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
    ss << "Destination slice shape: " << selfSlicedSizes << " at dimension "
       << dim;
    ss << " and source slice shape: " << sourceSlicedSizes
       << " at dimension 0.";
    TORCH_CHECK(false, ss.str());
  }
  TORCH_CHECK_INDEX(
      source.dim() == 0 || numIndices == source.size(dim),
      "index_copy_(): Number of indices (",
      numIndices,
      ") should be equal to source.size(dim) (",
      source.size(dim),
      ")");

  return TORCH_PRECOMPUTE_STRUCT(index_copy)().set_dim(dim);
}

template <typename Meta>
static void index_func_meta_impl(
    Meta& meta,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    std::string_view func) {
  auto numel = index.numel();

  TORCH_CHECK_INDEX(
      index.dim() <= 1,
      func,
      "_(): Index is supposed to be a vector, but got dim: ",
      index.dim(),
      " with type: ",
      index.scalar_type(),
      " and size: ",
      index.sizes());
  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long ||
          index.scalar_type() == ScalarType::Int,
      func,
      "_(): Expected dtype int32/int64 for index but got: ",
      index.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      func,
      "_(): self (",
      self.scalar_type(),
      ") and source (",
      source.scalar_type(),
      ") must have the same scalar type");
  TORCH_CHECK(
      dim == 0 || dim < source.dim(),
      func,
      "_(): Indexing dim ",
      dim,
      " is out of bounds of the source tensor with dim ",
      source.dim());
  TORCH_CHECK(
      numel == (source.dim() == 0 ? 1 : source.size(dim)),
      func,
      "_(): Number of indices (",
      numel,
      ") should be equal to source.size(dim): (",
      source.size(dim),
      "), for dim: ",
      dim);

  auto self_sizes = self.sizes().vec();
  auto source_sizes = source.sizes().vec();
  if (source.dim() != 0 && self.dim() != 0) {
    self_sizes.erase(self_sizes.begin() + dim);
    source_sizes.erase(source_sizes.begin() + dim);
  }
  TORCH_CHECK(
      self_sizes == source_sizes,
      "source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = ",
      self.sizes(),
      " source.shape = ",
      source.sizes());

  auto& result = meta.maybe_get_output(0);
  bool is_defined = result.defined();
  meta.set_output_raw_strided(0, self.sizes(), {}, self.options());
  if (is_defined) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, index);
    at::assert_no_overlap(result, source);
  }

  // A hack to run TensorIterator checks in the meta function.
  // See comment:
  // https://github.com/pytorch/pytorch/pull/65993#discussion_r760307417
  // TODO: (@krshrimali) Try inheriting from TensorIteratorBase instead.
  if (result.device() == kMeta && result.dim() > 0) {
    auto selfSlice = result.select(dim, 0);
    auto sourceSlice = source.select(dim, 0);
    auto iter =
        TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);
  }
}

TORCH_PRECOMPUTE_META_FUNC(index_add)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const Scalar& alpha) {
  dim = maybe_wrap_dim(dim, self.dim());
  index_func_meta_impl(*this, self, dim, index, source, "index_add");
  return TORCH_PRECOMPUTE_STRUCT(index_add)().set_dim(dim);
}

TORCH_PRECOMPUTE_META_FUNC(index_reduce)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const std::string_view reduce,
 bool include_self) {
  (void)include_self;
  TORCH_CHECK(
      reduce == "prod" || reduce == "mean" || reduce == "amax" ||
          reduce == "amin",
      "index_reduce(): Expected reduce to be one of prod, mean, amax or amin but got ",
      reduce,
      ".");
  dim = maybe_wrap_dim(dim, self.dim());
  index_func_meta_impl(*this, self, dim, index, source, "index_reduce");
  return TORCH_PRECOMPUTE_STRUCT(index_reduce)().set_dim(dim);
}

static void build_index_op(
    TensorIteratorBase& iter,
    const at::native::AdvancedIndex& info,
    const Tensor& result) {
  // 'TensorIterator' needs to own the things coming from 'info', since
  // 'info' will be destroyed after the META function.
  TensorIteratorConfig config;
  // info.src is a restrided view of result
  config.set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .add_output(result)
      .add_owned_const_input(info.src);
  for (auto& index : info.indices) {
    config.add_owned_const_input(index);
  }
  if (!result.defined()) {
    config.declare_static_dtype_and_device(
        info.src.scalar_type(), info.src.device());
  }
  iter.build(config);
}

static void check_indices_on_cpu_or_selfdevice(
    const Tensor& self,
    const at::MaterializedIOptTensorListRef& indices) {
  auto dev = self.device();
  bool indices_on_cpu_or_dev = std::all_of(
      indices.begin(), indices.end(), [=](const at::OptionalTensorRef& opt) {
        return opt.has_value() ? (opt->is_cpu() || opt->device() == dev) : true;
      });
  TORCH_CHECK(
      indices_on_cpu_or_dev,
      "indices should be either on ",
      kCPU,
      " or on the same device as the indexed tensor (",
      dev,
      ")");
}

TORCH_PRECOMPUTE_META_FUNC2(index, Tensor)
(const Tensor& self, at::IOptTensorListRef indices) {
  auto materialized = indices.materialize();

  TORCH_CHECK_INDEX(
      materialized.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      materialized.size(),
      ")");

  // Only allow: `dev_tensor[{cpu,dev}_tensor]`.
  // See: https://github.com/pytorch/pytorch/pull/69607
  check_indices_on_cpu_or_selfdevice(self, materialized);

  const auto& result = maybe_get_output();

  if (result.defined()) {
    TORCH_CHECK(
        self.scalar_type() == result.scalar_type(),
        "index_out: self (",
        self.scalar_type(),
        ") and result (",
        result.scalar_type(),
        ") must have the same scalar type");
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, self);
    for (const at::OptionalTensorRef& index : materialized) {
      if (index.has_value()) {
        at::assert_no_overlap(result, *index);
      }
    }
  }

  auto info = at::native::make_info(self, std::move(indices));
  build_index_op(*this, info, result);
  return TORCH_PRECOMPUTE_STRUCT2(index, Tensor)()
      .set_sizes(std::move(info.indexed_sizes))
      .set_strides(std::move(info.indexed_strides));
}

} // namespace at::meta

namespace at::native {

DEFINE_DISPATCH(index_stub);
DEFINE_DISPATCH(index_fill_stub);
DEFINE_DISPATCH(index_copy_stub);
DEFINE_DISPATCH(index_put_stub);
DEFINE_DISPATCH(index_put_with_sort_stub);
DEFINE_DISPATCH(put_stub);
DEFINE_DISPATCH(take_stub);
DEFINE_DISPATCH(masked_fill_stub);
REGISTER_NO_CPU_DISPATCH(index_put_with_sort_stub)
REGISTER_NO_CPU_DISPATCH(index_put_with_sort_quantized_stub)
DEFINE_DISPATCH(masked_select_serial_stub);
DEFINE_DISPATCH(masked_select_stub);
DEFINE_DISPATCH(masked_scatter_stub);

DEFINE_DISPATCH(gather_stub);
DEFINE_DISPATCH(scatter_stub);
DEFINE_DISPATCH(scatter_fill_stub);
DEFINE_DISPATCH(scatter_add_stub);
DEFINE_DISPATCH(scatter_reduce_stub);
DEFINE_DISPATCH(scatter_scalar_reduce_stub);
DEFINE_DISPATCH(scatter_reduce_two_stub);

DEFINE_DISPATCH(scatter_add_expanded_index_stub);
DEFINE_DISPATCH(scatter_reduce_expanded_index_stub);
DEFINE_DISPATCH(gather_expanded_index_stub);

static bool all_strides_match(TensorList tensors) {
  TORCH_CHECK(!tensors.empty());
  auto strides = tensors[0].strides();
  for (auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

// Replace indexed dimensions in src with stride 0 and the size of the result
// tensor. The offset in these dimensions is computed by the kernel using the
// index tensor's values and the stride of src. The new shape is not meaningful.
// It's used to make the shape compatible with the result tensor.
static Tensor restride_src(
    const Tensor& src,
    int64_t dims_before,
    int64_t dims_indexed,
    IntArrayRef replacement_shape) {
  auto shape = DimVector(src.sizes());
  auto strides = DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(
      shape.begin() + dims_before,
      replacement_shape.begin(),
      replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

// Add dimensions of size 1 to an index tensor so that it can be broadcast to
// the result shape and iterated over element-wise like the result tensor and
// the restrided src.
static Tensor reshape_indexer(
    const Tensor& index,
    int64_t dims_before,
    int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

AdvancedIndex::AdvancedIndex(const Tensor& src, TensorList indices_list) {
  int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  IntArrayRef replacement_shape;
  for (const auto dim : c10::irange(indices_list.size())) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) !=
          indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) ==
          replacement_shape.end()) {
    TORCH_CHECK_INDEX(
        false, "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }

  // For CUDA/MPS/XPU tensors, force all index tensors to have the same striding
  // to simplify the CUDA/MPS/XPU kernel.
  if (indices.size() >= 2 &&
      (this->src.device().type() == kCUDA ||
       this->src.device().type() == kMPS ||
       this->src.device().type() == kXPU)) {
    if (!all_strides_match(indices)) {
      for (auto& indice : indices) {
        indice = indice.contiguous();
      }
    }
  }
}

static TensorIterator make_index_put_iterator(
    const AdvancedIndex& info,
    const Tensor& value) {
  TORCH_CHECK(
      is_expandable_to(value.sizes(), info.src.sizes()),
      "shape mismatch: value tensor of shape ",
      value.sizes(),
      " cannot be broadcast to indexing result of shape ",
      info.src.sizes());
  TORCH_CHECK(
      value.scalar_type() == info.src.scalar_type(),
      "Index put requires the source and destination dtypes match, "
      "got ",
      info.src.scalar_type(),
      " for the destination "
      "and ",
      value.scalar_type(),
      " for the source.");
  TensorIteratorConfig config;
  // info.src is restrided by restride_src with 0 strided dimensions
  config.set_check_mem_overlap(false);
  config.resize_outputs(false);
  config.check_all_same_dtype(false);
  config.add_output(info.src);
  config.add_const_input(value);
  for (auto& index : info.indices) {
    config.add_const_input(index);
  }
  return config.build();
}

TORCH_IMPL_FUNC(index_out)
(const Tensor& self, DimVector sizes, DimVector strides, const Tensor& result) {
  index_stub(device_type(), *this, sizes, strides);
}

Tensor quantized_index(
    const Tensor& self,
    const torch::List<std::optional<Tensor>>& indices) {
  TORCH_INTERNAL_ASSERT(
      self.qscheme() == c10::kPerTensorAffine ||
          self.qscheme() == c10::kPerTensorSymmetric,
      "Indexing is only supported for per-Tensor quantized Tensors.");

  // For now, this is a naive implementation which does dq -> index -> q.
  // TODO(future PR): improve performance by removing the copies.
  const auto& self_dq = self.dequantize();
  auto result = at::index(self_dq, indices);
  return at::quantize_per_tensor(
      result, self.q_scale(), self.q_zero_point(), self.scalar_type());
}

Tensor _unsafe_index(
    const Tensor& self,
    const torch::List<std::optional<Tensor>>& indices) {
  // Disallow boolean indexing since it leads to dynamic output shapes
  for (auto i : c10::irange(indices.size())) {
    auto index = indices.get(i);
    if (index.has_value()) {
      auto dtype = index->scalar_type();
      TORCH_CHECK(
          dtype == kLong || dtype == kInt,
          "_unsafe_index found unexpected index type ",
          dtype);
    }
  }
  return at::index(self, indices);
}

Tensor _unsafe_masked_index(
    const Tensor& self,
    const Tensor& mask,
    const torch::List<std::optional<Tensor>>& indices,
    const Scalar& fill) {
  // Unsafe masked index is equivalent to
  //   where(mask, self[indices], fill)
  // with the main difference being that the when the `mask` is false, the
  // tensor `self` is not indexed using `indices`. This allows `indices` to be
  // out-of-bounds when `mask` is false. When `mask` is true, the `indices` are
  // expected to be in bounds and is not checked. We also assume that the
  // `indices` are non-negative
  //
  // This function is not meant to be executed on eager mode. An unoptimized
  // version is provided here.
  //
  // compiler backends should implement this op such that `self[indices]` is not
  // loaded when `mask` is true. See inductor for a reference.
  auto clamp = [](const std::optional<Tensor>& index,
                  auto size) -> std::optional<Tensor> {
    if (!index) {
      return index;
    }
    // Disallow bool
    auto dtype = index->scalar_type();
    TORCH_CHECK(
        dtype == kLong || dtype == kInt,
        "_unsafe_masked_index found unexpected index type ",
        dtype);
    return at::clamp(*index, -size, size - 1);
  };

  torch::List<std::optional<Tensor>> clamped_indices(indices);
  std::transform(
      indices.begin(),
      indices.end(),
      self.sizes().begin(),
      clamped_indices.begin(),
      clamp);

  if (self.numel() == 0) {
    // Returns a tensor filled with `fill` value
    // We use a hack here since we do not have a method to get the
    // correct size of the tensor. (except with meta impl which is
    // not available on mobile builds)
    std::vector<int64_t> new_sizes(self.dim());
    auto compute_new_size = [](const std::optional<Tensor>& index,
                               auto size) -> int64_t {
      if (index && size == 0) {
        return 1;
      } else {
        return size;
      }
    };
    std::transform(
        indices.begin(),
        indices.end(),
        self.sizes().begin(),
        new_sizes.begin(),
        compute_new_size);
    auto result = self.new_full(new_sizes, fill);
    return at::_unsafe_index(result, clamped_indices);
  }

  auto result = at::_unsafe_index(self, clamped_indices);
  return result.masked_fill(at::logical_not(mask), fill);
}

Tensor _unsafe_masked_index_put_accumulate(
    const Tensor& self,
    const Tensor& mask,
    const torch::List<std::optional<Tensor>>& indices,
    const Tensor& values) {
  // This is the backward of _unsafe_masked_index.
  // This function is not meant to be executed on eager mode.

  if (self.numel() == 0) {
    return self.clone();
  }

  // We recompute the clamped indices and rely on inductor to CSE the
  // computation
  auto clamp = [](const std::optional<Tensor>& index,
                  auto size) -> std::optional<Tensor> {
    if (!index) {
      return index;
    }
    // Disallow bool
    auto dtype = index->scalar_type();
    TORCH_CHECK(
        dtype == kLong || dtype == kInt,
        "_unsafe_masked_index found unexpected index type ",
        dtype);
    return at::clamp(*index, -size, size - 1);
  };

  torch::List<std::optional<Tensor>> clamped_indices(indices);
  std::transform(
      indices.begin(),
      indices.end(),
      self.sizes().begin(),
      clamped_indices.begin(),
      clamp);

  auto masked_value = values.masked_fill(at::logical_not(mask), 0);
  return at::_unsafe_index_put(self, clamped_indices, masked_value, true);
}

Tensor& put_(
    Tensor& self,
    const Tensor& index,
    const Tensor& source,
    const bool accumulate) {
  // See note [Writing Nondeterministic Operations]
  // Nondeterministic when index contains duplicate entries and we do not
  // accumulate If we accumulate on GPU, we use atomicGPUAdd, which is
  // non-deterministic
  if (!accumulate || (accumulate && self.device().type() == DeviceType::CUDA)) {
    at::globalContext().alertNotDeterministic("put_");
  }

  // Type and device checks
  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long,
      "put_(): Expected a long tensor for index, but got ",
      index.scalar_type())
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "put_(): self and source expected to have the same dtype, but got self.dtype = ",
      self.scalar_type(),
      " and source.dtype = ",
      source.scalar_type());
  TORCH_CHECK(
      self.device() == source.device() && self.device() == index.device(),
      "put_(): self, index and source expected to be in the same device, but got self.device = ",
      self.device(),
      ", index.device = ",
      index.device(),
      ", and source.device = ",
      source.device());

  // index checks
  TORCH_CHECK_INDEX(
      source.numel() == index.numel(),
      "put_(): Expected source and index to have the same number of elements, but got source.numel() = ",
      source.numel(),
      ", index.numel() = ",
      index.numel());
  TORCH_CHECK_INDEX(
      !(self.numel() == 0 && index.numel() != 0),
      "put_(): Tried to put elements into an empty tensor");

  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  at::assert_no_overlap(self, source);

  // Early return
  if (index.numel() == 0) {
    return self;
  }

  auto index_reshaped = index.reshape(source.sizes());
  // Do not iterate over self, we will compute the offsets manually
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .add_const_input(source)
                  .add_const_input(index_reshaped)
                  .build();

  put_stub(iter.device_type(), iter, self, accumulate);

  return self;
}

Tensor put(
    const Tensor& self,
    const Tensor& index,
    const Tensor& source,
    const bool accumulate) {
  return self.clone(at::MemoryFormat::Preserve).put_(index, source, accumulate);
}

Tensor index_put(
    const Tensor& self,
    const torch::List<std::optional<Tensor>>& indices,
    const Tensor& value,
    bool accumulate) {
  return self.clone(at::MemoryFormat::Preserve)
      .index_put_(indices, value, accumulate);
}

Tensor _unsafe_index_put(
    const Tensor& self,
    const torch::List<std::optional<Tensor>>& indices,
    const Tensor& value,
    bool accumulate) {
  return at::index_put(self, indices, value, accumulate);
}

Tensor& _index_put_impl_(
    Tensor& self,
    const torch::List<std::optional<Tensor>>& indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  TORCH_CHECK_INDEX(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
        "Use of index_put_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  if (!accumulate) {
    auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
    if (std::get<0>(masked_fill_dispatch)) {
      return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
    }
  }
  auto value_ = value;
  if (value.device() != self.device() && value.numel() == 1 &&
      value.dim() == 0) {
    value_ = value.to(self.device());
  }
  at::assert_no_overlap(self, value);
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const std::optional<Tensor>& index : indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }
  if ((self.device().type() == DeviceType::CUDA ||
       self.device().type() == DeviceType::XPU) &&
      (accumulate ||
       (globalContext().deterministicAlgorithms() && value_.numel() > 1))) {
    TORCH_CHECK(
        value_.device() == self.device(),
        "expected device ",
        self.device(),
        " but got device ",
        value_.device(),
        " for value tensor");
    index_put_with_sort_stub(
        self.device().type(), self, indices, value_, accumulate, unsafe);
    return self;
  }

  auto info = make_info(self, indices);
  auto iter = make_index_put_iterator(info, value_);
  index_put_stub(
      iter.device_type(),
      iter,
      info.indexed_sizes,
      info.indexed_strides,
      accumulate);
  return self;
}

Tensor& take_out(const Tensor& self, const Tensor& index, Tensor& out) {
  // Type and device checks
  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long,
      "take(): Expected a long tensor for index, but got ",
      index.scalar_type())
  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      "take(): self and out expected to have the same dtype, but got self.dtype = ",
      self.scalar_type(),
      " and out.dtype = ",
      out.scalar_type());
  TORCH_CHECK(
      self.device() == out.device() && self.device() == index.device(),
      "take(): self, index and out expected to be in the same device, but got self.device = ",
      self.device(),
      ", index.device = ",
      index.device(),
      ", and out.device = ",
      out.device());

  // index checks
  TORCH_CHECK_INDEX(
      !(self.numel() == 0 && index.numel() != 0),
      "take(): tried to take from an empty tensor");

  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  at::assert_no_overlap(out, self);

  // Do not iterate over self, we will compute the offsets manually
  // out is resized inside tensor_iterator
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .add_output(out)
                  .add_const_input(index)
                  .build();

  // Early return after out has been resized
  if (index.numel() == 0) {
    return out;
  }

  take_stub(iter.device_type(), iter, self);

  return out;
}

Tensor take(const Tensor& self, const Tensor& index) {
  auto out = at::empty(index.sizes(), self.options());
  at::native::take_out(self, index, out);
  return out;
}

Tensor& index_put_(
    Tensor& self,
    const torch::List<std::optional<Tensor>>& indices,
    const Tensor& value,
    const bool accumulate) {
  return at::_index_put_impl_(
      self, indices, value, accumulate, /*unsafe=*/false);
}

TORCH_IMPL_FUNC(index_copy_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const Tensor& result) {
  if (!result.is_same(self))
    result.copy_(self);

  // See Note [Enabling Deterministic Operations]
  if (result.is_cuda() && globalContext().deterministicAlgorithms()) {
    torch::List<std::optional<Tensor>> indices;
    indices.resize(dim + 1);
    indices.set(dim, index);
    result.index_put_(indices, source, false);
    return;
  }

  // Handle the case when self / source is 0-dim
  Tensor result_nonzero = result.dim() == 0 ? result.unsqueeze(0) : result;
  Tensor source_nonzero = source.dim() == 0 ? source.unsqueeze(0) : source;

  // The only difference between the following  tensor iterator and that of
  // index_fill_ is that this one has also source as an input. We should
  // refactor it when if constexpr is available (C++17)

  // Prepare `index` for TensorIterator.
  // It is restrided to be broadcastable over `self` in TensorIterator.
  auto index_sizes = std::vector<int64_t>(result_nonzero.dim(), 1);
  auto index_strides = std::vector<int64_t>(result_nonzero.dim(), 0);
  index_sizes[dim] = index.numel();
  index_strides[dim] =
      (index.dim() > 0) ? index.stride(0) : 1; // `index` is 1d or scalar
  auto index_restrided = index.as_strided(index_sizes, index_strides);

  // Prepare `result` for TensorIterator.
  // Restride `result` to not advance in dimension `dim`.
  // We do not use squash_dim here because `index` will
  // need to advance in this dimension.
  // Note that self_sizes[dim] is set to index.numel().
  // This is done so that self_sizes[dim] and index_sizes[dim]
  // match as required by TensorIterator (input shape should
  // strictly broadcast over output shape, i.e.
  // output.shape[i] >= input.shape[i] for i in range(dims)).
  auto result_sizes = result_nonzero.sizes().vec();
  auto result_strides = result_nonzero.strides().vec();
  result_sizes[dim] = index.numel();
  result_strides[dim] = 0;
  auto result_restrided =
      result_nonzero.as_strided(result_sizes, result_strides);

  auto iter = TensorIteratorConfig()
                  // We do not check for overlap because `result` is restrided
                  // with zero stride. Zero strides trigger memory overlap
                  // assert within TensorIterator.
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_output(result_restrided)
                  .add_const_input(index_restrided)
                  .add_const_input(source_nonzero)
                  .build();

  auto result_dim_size = result_nonzero.size(dim);
  auto result_dim_stride = result_nonzero.stride(dim);
  index_copy_stub(
      iter.device_type(), iter, dim, result_dim_size, result_dim_stride);
}

// Not calling into index_reduce_func_impl because of a different dtype dispatch
TORCH_IMPL_FUNC(index_add_cpu_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const Scalar& alpha,
 const Tensor& result) {
  if (!result.is_same(self)) {
    result.copy_(self);
  }
  auto numel = index.numel();

  auto index_contig = index.contiguous();

  if (result.dim() > 1) {
    // Equivalent to:
    //   for (const auto i : c10::irange(numel)) {
    //     auto selfSlice = self.select(dim, index_data[i]);
    //     auto sourceSlice = source.select(dim, i);
    //     selfSlice.add_(sourceSlice);
    //   }
    // But much faster as this reuses the iterator from add_
    if (numel == 0 || self.numel() == 0) {
      return;
    }

    dim = maybe_wrap_dim(dim, self.dim());

    // When the slice of source or result is noncontiguous,
    // original index_add is slow as it uses add for the sliced tensor,
    // which is serial on index and parallel on sliced tensor to avoid write
    // conflict. Doing parallel on the sliced tensor is not optimal as the size
    // of sliced tensor may be not big enough to parallel and also causes
    // multiple parallelizations. scatter_add is used to speedup for this case
    // as scatter_add parallels on the outer dimension of input and is serial on
    // the inner dimension to avoid write conflict. scatter_add only need one
    // parallel and the size of outer dimensions is bigger to do parallel.

    if ((dim == 0 || dim == self.dim() - 1) &&
        // Data type of index should be long and alpha should be 1 to use
        // scatter_add.
        alpha.equal(1.0) && index_contig.scalar_type() == ScalarType::Long &&
        // scatter_add does not support ComplexHalf
        source.scalar_type() != ScalarType::ComplexHalf &&
        result.scalar_type() != ScalarType::ComplexHalf) {
      std::vector<int64_t> ep_sizes(result.sizes().size());
      std::vector<int64_t> ep_strides(source.sizes().size());

      // Check whether result and source are matched apart from the dimension
      // dim. Note that the broadcast case: source.select(dim, i) is broadcast
      // for result.select(dim, index_data[i]) The broadcast case is not
      // applicable for scatter_add
      auto check_sizes =
          [&ep_sizes, &ep_strides, &numel](
              IntArrayRef a, IntArrayRef b, int64_t dim) -> bool {
        ep_sizes[dim] = numel;
        ep_strides[dim] = 1;
        for (const int64_t i : c10::irange(a.size())) {
          if (i == dim) {
            continue;
          }

          if (a[i] != b[i]) {
            return false;
          }
          ep_sizes[i] = a[i];
          ep_strides[i] = 0;
        }
        return true;
      };

      if (check_sizes(result.sizes(), source.sizes(), dim)) {
        auto ep_index = index_contig.as_strided(ep_sizes, ep_strides);
        result.scatter_add_(dim, ep_index, source);
        return;
      }
    }

    auto selfSlice = result.select(dim, 0);
    auto sourceSlice = source.select(dim, 0);
    auto self_stride_bytes =
        result.stride(dim) * elementSize(result.scalar_type());
    auto source_stride_bytes =
        source.stride(dim) * elementSize(source.scalar_type());
    auto self_dim_size = result.size(dim);
    auto iter =
        TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);

    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cpu_", [&]() {
      auto index_data = index_contig.const_data_ptr<index_t>();
      for (const auto i : c10::irange(numel)) {
        auto self_i = index_data[i];
        TORCH_CHECK_INDEX(
            (self_i >= 0) && (self_i < self_dim_size),
            "index out of range in self");
        auto self_data = static_cast<char*>(selfSlice.data_ptr()) +
            self_i * self_stride_bytes;
        auto source_data =
            static_cast<const char*>(sourceSlice.const_data_ptr()) +
            i * source_stride_bytes;
        iter.unsafe_replace_operand(0, self_data);
        iter.unsafe_replace_operand(1, self_data);
        iter.unsafe_replace_operand(2, const_cast<char*>(source_data));
        add_stub(iter.device_type(), iter, alpha);
      }
    });
  } else {
    TORCH_CHECK(
        source.dim() <= 1,
        "source.dim() (",
        source.dim(),
        ") must one or zero for given self.dim() (",
        self.dim(),
        ")");

    // explicitly capture all required variables to work around windows build
    // TODO: fix this when windows can correctly capture variables in nested
    // lambda
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        ScalarType::Half,
        ScalarType::Bool,
        ScalarType::BFloat16,
        ScalarType::ComplexHalf,
        result.scalar_type(),
        "index_add_",
        [&result, &source, &dim, &index_contig, &numel, &alpha] {
          auto alpha_value = alpha.to<scalar_t>();
          auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);
          auto source_stride = source.dim() == 0 ? 1 : source.stride(dim);
          // TODO: Maybe TensorAccessor can be used here?
          auto* result_ptr = result.data_ptr<scalar_t>();
          auto* source_ptr = source.const_data_ptr<scalar_t>();
          AT_DISPATCH_INDEX_TYPES(
              index_contig.scalar_type(),
              "index_add_cpu_",
              [&index_contig,
               &numel,
               &result,
               &result_ptr,
               &result_stride,
               &source_ptr,
               &source_stride,
               &alpha_value] {
                auto index_data = index_contig.const_data_ptr<index_t>();
                for (const auto i : c10::irange(numel)) {
                  auto self_i = index_data[i];
                  TORCH_CHECK_INDEX(
                      (self_i >= 0) && (self_i < result.numel()),
                      "index out of range in self");
                  scalar_t* self_ip = result_ptr + self_i * result_stride;
                  *self_ip +=
                      c10::load(source_ptr + i * source_stride) * alpha_value;
                }
              });
        });
  }
}

static void index_reduce_func_impl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const Tensor& result,
    const ReductionType& op) {
  if (!result.is_same(self))
    result.copy_(self);
  if (!include_self) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "index_reduce_func_exclude_input_init",
        [&] {
          scalar_t init_val;
          switch (op) {
            case ReductionType::PROD:
              init_val = (scalar_t)1;
              break;
            case ReductionType::MAX:
              init_val = std::numeric_limits<scalar_t>::has_infinity
                  ? -std::numeric_limits<scalar_t>::infinity()
                  : std::numeric_limits<scalar_t>::lowest();
              break;
            case ReductionType::MIN:
              init_val = std::numeric_limits<scalar_t>::has_infinity
                  ? std::numeric_limits<scalar_t>::infinity()
                  : std::numeric_limits<scalar_t>::max();
              break;
            default:
              init_val = (scalar_t)0;
              break;
          }
          // index_fill_ requires index to be a LongTensor
          result.index_fill_(dim, index.to(at::ScalarType::Long), init_val);
        });
  }

  auto numel = index.numel();

  auto index_contig = index.contiguous();

  if (result.dim() > 1) {
    // Equivalent to:
    //   for (const auto i : c10::irange(numel)) {
    //     auto selfSlice = self.select(dim, index_data[i]);
    //     auto sourceSlice = source.select(dim, i);
    //     selfSlice.op_(sourceSlice);
    //   }
    // But much faster as this reuses the iterator from the binary op
    if (numel == 0) {
      return;
    }
    auto selfSlice = result.select(dim, 0);
    auto sourceSlice = source.select(dim, 0);
    auto self_stride_bytes =
        result.stride(dim) * elementSize(result.scalar_type());
    auto source_stride_bytes =
        source.stride(dim) * elementSize(source.scalar_type());
    auto self_dim_size = result.size(dim);
    auto iter =
        TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);

    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_func_cpu_", [&]() {
      auto index_data = index_contig.const_data_ptr<index_t>();
      for (const auto i : c10::irange(numel)) {
        auto self_i = index_data[i];
        TORCH_CHECK_INDEX(
            (self_i >= 0) && (self_i < self_dim_size),
            "index out of range in self");
        auto self_data = static_cast<char*>(selfSlice.data_ptr()) +
            self_i * self_stride_bytes;
        auto source_data =
            static_cast<const char*>(sourceSlice.const_data_ptr()) +
            i * source_stride_bytes;
        iter.unsafe_replace_operand(0, self_data);
        iter.unsafe_replace_operand(1, self_data);
        iter.unsafe_replace_operand(2, const_cast<char*>(source_data));

        switch (op) {
          case ReductionType::PROD:
            mul_stub(iter.device_type(), iter);
            break;
          case ReductionType::MIN:
            minimum_stub(iter.device_type(), iter);
            break;
          case ReductionType::MAX:
            maximum_stub(iter.device_type(), iter);
            break;
          default:
            add_stub(iter.device_type(), iter, 1);
            break;
        }
      }
    });

    if (op == ReductionType::MEAN) {
      auto counts =
          include_self ? at::ones_like(result) : at::zeros_like(result);
      counts.index_add_(dim, index, at::ones_like(source));
      counts.masked_fill_(counts == 0, 1);
      if (result.is_floating_point() || result.is_complex()) {
        result.div_(counts);
      } else {
        result.div_(counts, "floor");
      }
    }
  } else {
    TORCH_CHECK(
        source.dim() <= 1,
        "source.dim() (",
        source.dim(),
        ") must one or zero for given self.dim() (",
        self.dim(),
        ")");
    auto counts = include_self ? at::ones_like(result) : at::zeros_like(result);
    // explicitly capture all required variables to work around windows build
    // TODO: fix this when windows can correctly capture variables in nested
    // lambda
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        result.scalar_type(),
        "index_func_",
        [&result, &source, &dim, &index_contig, &numel, &op, &counts] {
          auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);
          auto source_stride = source.dim() == 0 ? 1 : source.stride(dim);
          auto counts_stride = counts.dim() == 0 ? 1 : counts.stride(dim);
          // TODO: Maybe TensorAccessor can be used he
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `TensorAdvancedIndexing.cpp_docs.md_docs.md`
- **Keyword Index**: `TensorAdvancedIndexing.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
