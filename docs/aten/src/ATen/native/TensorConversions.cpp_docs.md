# Documentation: `aten/src/ATen/native/TensorConversions.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/TensorConversions.cpp`
- **Size**: 87,888 bytes (85.83 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/quantized/Quantizer.h>
#include <optional>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_autocast_to_full_precision_native.h>
#include <ATen/ops/_autocast_to_reduced_precision_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_bsc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_bsr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_native.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/_to_copy_native.h>
#include <ATen/ops/_to_cpu_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/_to_sparse_bsc_native.h>
#include <ATen/ops/_to_sparse_bsr_native.h>
#include <ATen/ops/_to_sparse_csc_native.h>
#include <ATen/ops/_to_sparse_csr_native.h>
#include <ATen/ops/_to_sparse_native.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/ops/to_dense_backward_native.h>
#include <ATen/ops/to_dense_native.h>
#include <ATen/ops/to_mkldnn_backward_native.h>
#include <ATen/ops/to_native.h>
#include <ATen/ops/to_sparse_bsc_native.h>
#include <ATen/ops/to_sparse_bsr_native.h>
#include <ATen/ops/to_sparse_csc_native.h>
#include <ATen/ops/to_sparse_csr_native.h>
#include <ATen/ops/to_sparse_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorConversions.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <algorithm>
#include <numeric>

namespace at::native {

namespace {
// dense_to_sparse_{csr,bsr,csc,bsc} common helpers

// Preparation for the N-D dense -> sparse compressed conversion.
// The N-D input is converted to 3-D (single batch dim) where we check that the
// product of batch dims is nonzero and for each batch the sparse matrix
// contained within has the same number of non-zero elements.
// The batches are joined along the compressed axis. The generation of indices
// for this matrix can be performed in a single step followed by a single step
// conversion to restore the batch dimension.
void dense_to_sparse_compressed_prepare_check_mask_values_batched(
    const Layout& target_layout,
    Tensor& values,
    Tensor& mask,
    const int64_t& n_batch_dim) {
  if (n_batch_dim > 1) {
    // For inputs with more than 1 batch dim we flatten them out.
    // Input shape (b0, b1 ..., bn, r, c) -> (b0 * b1 * ... * bn, r ,c)
    values = values.flatten(0, n_batch_dim - 1);
    mask = mask.flatten(0, n_batch_dim - 1);
  }

  // For informative messaging form the name of the function
  // to_sparse_{csr,csc,bsr,bsc}.
  TORCH_CHECK(
      mask.size(0) > 0,
      "to_sparse_",
      // We want the message to match the function name so generate the
      // lowercase acronym for the layout
      sparse_csr::layoutToString(target_layout, false, true),
      ": Expected product of batch dimensions to be non-zero.");

  // Compute the number of non-zero elements in the first batch, expand to full
  // size
  auto nse_per_batch = mask.select(0, 0).sum().expand(mask.size(0));
  TORCH_CHECK(
      mask.sum({-2, -1}).equal(nse_per_batch),
      "Expect the same number of specified elements per batch.");

  // We need to join batches into a matrix increasing the length of the
  // compressed axis. This allows us to create indices for a compressed matrix
  // and de-batch them later (two kernels). Otherwise we would have to create
  // indices for each batch individually requiring n_batch kernels. For csr/bsr,
  // we already have the batch dim adjacent to the compressed axis and can
  // flatten them together. For csc/bsc, we need to transpose first.
  // For BSR/CSR (b, r, c) -> (b*r, c)
  // For BSC/CSC (b, c, r) -> (r, b*c)
  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      target_layout,
      "dense_to_sparse_compressed",
      [&]() {
        values = values.flatten(0, 1);
        mask = mask.flatten(0, 1);
      },
      [&]() {
        values = values.transpose(0, 1).flatten(1, 2);
        mask = mask.transpose(0, 1).flatten(1, 2);
      });
}

// This function unfolds the compressed indices of a compressed sparse matrix
// into a batched compressed sparse tensor.
// This is analogous to an unflatten-like operation:
// unflatten(0, {b, r}) for csr/bsr with input shape (r*b, c)
//          (output shape (b, r, c))
// unflatten(1, {b, c}).transpose(0,1) for csc/bsc with input shape (r, c*b)
//          (output shape (r, b, c) unflatten, (b, r, c) unflatten + transpose)
// This only operates on the compressed indices as the plain indices and values
// can be manipulated as described above without special handling.
// It is a prerequisite for the conversion that the sparsity pattern is sane for
// the batched shape. That is each batch has the same number of nonzero
// elements.
Tensor compressed_to_batched_compressed_indices(
    const Tensor& compressed_in,
    const int64_t& n_batch,
    bool out_int32) {
  auto n_compressed_per_batch = (compressed_in.size(0) - 1) / n_batch;
  ScalarType out_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  auto batched_out = at::zeros(
      {n_batch, n_compressed_per_batch + 1},
      compressed_in.options().dtype(out_type));

  // If the compressed dimension has length zero there is 1 element in each
  // batch and it is zero we already have this result formed
  if (n_compressed_per_batch > 0) {
    // Slice the compressed indices ignoring the leading 0 element and reshape
    // to n-batch rows
    auto trailing_slice =
        compressed_in.slice(0, 1, std::nullopt, 1).reshape({n_batch, -1});
    // Slice the compressed indices again selecting the elements corresponding
    // to the batch boundary. The values here will be increasing multiples of
    // nnz per batch. Reshape to n-batch rows (1 col) for broadcasting.
    // This is equivalent to arange(n_batch) * nnz_per_batch with the same
    // reshape
    auto offsets = compressed_in.slice(0, 0, -1, n_compressed_per_batch)
                       .reshape({n_batch, -1});
    // Subtracting the offsets from each row of the reshaped compressed indices
    // gives us the compressed indices within the batch. The leading element of
    // each row is not computed as it is always zero.  We copy into the view on
    // the output buffer.
    batched_out.narrow(-1, 1, n_compressed_per_batch)
        .copy_(trailing_slice - offsets);
  }
  return batched_out;
}

// After generating member tensors for sparse_compressed matrix, if the target
// shape is N-D we must reform the batch dimensions.
// Single kernel is used to restore one batch dimension in the compressed
// indices. From there full batch shape is restored by reshape. No special
// handling is needed for restoring batch dimensions of the values or
// plain_indices it can be done with reshape/unflatten.
void reshape_2d_sparse_compressed_members_to_nd_batched(
    const IntArrayRef full_sizes,
    const int64_t& n_batch_dim,
    Tensor& compressed_indices,
    Tensor& plain_indices,
    Tensor& values) {
  auto batch_shape = full_sizes.slice(0, n_batch_dim);
  auto n_batch = std::accumulate(
      batch_shape.begin(), batch_shape.end(), 1, std::multiplies<int64_t>());
  // NOTE: using this conversion requires the nnz per batch is the same for all
  // batches that will be formed. We ensured this was the case on the way in so
  // it is safe to use this conversion.
  compressed_indices = compressed_to_batched_compressed_indices(
      compressed_indices, n_batch, /*out_int32*/ false);

  // We can infer the last dim of the reshape targets, it will be nnz or
  // nrow/ncol+1 depending on the layout and member tensor targeted.
  auto batchsize_infer_last = DimVector(batch_shape);
  batchsize_infer_last.push_back(-1);

  // -1 will be nnz per batch
  plain_indices = plain_indices.reshape(batchsize_infer_last);
  // -1 will be ncols (bsc,csc) or nrows (bsr,csr) + 1
  compressed_indices = compressed_indices.reshape(batchsize_infer_last);
  // -1 will be nnz (per batch).
  // Note: Unflatten rather than reshape as it will work
  // for both blocked and unblocked layouts. reshape works for unblocked layouts
  // only
  values = values.unflatten(0, batchsize_infer_last);
}
} // namespace

// Take a Device that may not have device_index set (i.e., having it as -1
// representing the current device) and return the corresponding Device
// according to the actual device at the time of this function call.  No-op
// if the device_index is set.
static inline Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

static inline std::optional<Device> ensure_has_index(
    std::optional<Device> device) {
  if (!device.has_value()) {
    return std::nullopt;
  }
  return ensure_has_index(device.value());
}

Tensor _to_copy(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !layout.has_value() || self.layout() == layout.value(),
      "to(options) doesn't support converting to a different layout, "
      "but got self.layout being ",
      self.layout(),
      " and options.layout set as ",
      layout.value());
  auto options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  if (options.has_device()) {
    options = options.device(ensure_has_index(options.device()));
  }
  // memory_format is handled separately due to MemoryFormat::Preserve logic
  options = self.options().merge_in(options).memory_format(std::nullopt);
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  // TODO: Use the dispatcher for this.
  // Currently there are unenumerated extensibility issues preventing this.
  if (self.layout() == kSparse) {
    TORCH_CHECK(
        memory_format == MemoryFormat::Preserve,
        "to(options): COO only supports memory format Preserve, but got ",
        memory_format,
        " instead.");
    if (options.device().is_meta()) {
      return zeros_like(self, options);
    }
    auto indices = self._indices();
    const auto new_indices = at::native::to(
        indices,
        indices.scalar_type(),
        c10::kStrided,
        device,
        pin_memory,
        non_blocking,
        true, // force copy since we are in _to_copy
        memory_format);
    const auto new_values = at::native::to(
        self._values(),
        dtype,
        c10::kStrided,
        device,
        pin_memory,
        non_blocking,
        true, // force copy since we are in _to_copy
        memory_format);

    return at::_sparse_coo_tensor_unsafe(
        new_indices, new_values, self.sizes(), options, self.is_coalesced());
  } else if (at::sparse_csr::is_sparse_compressed(self)) {
    TORCH_CHECK(
        memory_format == MemoryFormat::Preserve,
        "to(options): ",
        at::sparse_csr::layoutToString(self.layout()),
        " only supports memory format Preserve, but got ",
        memory_format,
        " instead.");

    if (options.device().is_meta()) {
      return zeros_like(self, options);
    }

    auto [compressed_indices, plain_indices] =
        at::sparse_csr::getCompressedPlainIndices(self);

    const auto new_values = at::native::to(
        self.values(),
        dtype,
        c10::kStrided,
        device,
        pin_memory,
        non_blocking,
        true, // force copy since we are in _to_copy
        memory_format);

    const auto new_compressed_indices = at::native::to(
        compressed_indices,
        compressed_indices.scalar_type(),
        c10::kStrided,
        device,
        pin_memory,
        non_blocking,
        true, // force copy since we are in _to_copy
        memory_format);

    const auto new_plain_indices = at::native::to(
        plain_indices,
        plain_indices.scalar_type(),
        c10::kStrided,
        device,
        pin_memory,
        non_blocking,
        true, // force copy since we are in _to_copy
        memory_format);

    return at::_sparse_compressed_tensor_unsafe(
        new_compressed_indices,
        new_plain_indices,
        new_values,
        self.sizes(),
        options);
  }

  bool pin_out =
      (non_blocking &&
       at::accelerator::isAcceleratorExcluded(self.device().type(), at::kMPS) &&
       options.device().is_cpu() && (options.layout() == c10::kStrided));

  if (memory_format == MemoryFormat::Preserve) {
    if (options.device().supports_as_strided()) {
      if (self.is_non_overlapping_and_dense()) {
        Tensor r;
        if (self.is_quantized()) {
          r = at::empty_quantized(self.sizes(), self, options);
          at::QuantizerPtr quantizer = r.quantizer();
          r.copy_(self, non_blocking);
          set_quantizer_(r, quantizer);
        } else {
          r = at::empty_strided(
              self.sizes(), self.strides(), options.pinned_memory(pin_out));
          r.copy_(self, non_blocking);
        }
        return r;
      } else if (!self.is_quantized() && self.layout() == kStrided) {
        Tensor r;
        auto strides = infer_dense_strides(self.sizes(), self.strides());
        r = at::empty_strided(
            self.sizes(), strides, options.pinned_memory(pin_out));
        r.copy_(self, non_blocking);
        return r;
      } else {
        memory_format = self.suggest_memory_format();
      }
    } else {
      memory_format = self.suggest_memory_format();
    }
  }
  // See Note [Explicit nullopt MemoryFormat argument]
  // TODO: empty_quantized does not work here. It raises an exception in
  // CheckMemoryFormat.h prior to
  // empty_affine_quantized/_empty_per_channel_affine_quantized calls
  // at::empty also does not work here because there is no proper at::empty
  // support for quantized tensors as it would return a quantized tensor with an
  // UnknownQuantizer
  auto r = self.is_quantized()
      ? at::empty_like(self, memory_format)
      : at::empty_symint(
            self.sym_sizes(),
            options.memory_format(memory_format).pinned_memory(pin_out),
            std::nullopt);
  r.copy_(self, non_blocking);
  return r;
}

template <typename T>
static inline bool is_null_or_equal_to(
    const std::optional<T>& test,
    const T& value) {
  if (!test.has_value()) {
    return true;
  }
  return test.value() == value;
}

// NOTE: static runtime's to_maybe_copy_out relies on details of this
// check; if you change how it works, please update static runtime as
// well.
bool to_will_alias(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  return is_null_or_equal_to(dtype, self.dtype().toScalarType()) &&
      is_null_or_equal_to(layout, self.layout()) &&
      is_null_or_equal_to(device, self.device()) && !copy &&
      (memory_format == MemoryFormat::Preserve ||
       self.suggest_memory_format() == memory_format);
}

static inline Tensor to_impl(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // fast path
  if (to_will_alias(
          self, dtype, layout, device, copy, optional_memory_format)) {
    return self;
  }
  return at::_to_copy(
      self,
      dtype,
      layout,
      device,
      pin_memory,
      non_blocking,
      optional_memory_format);
}

// If input tensor is fp32, cast it to fp16, otherwise leave it alone.
// (this is intended to be used internally by the JIT autocast implementation)
Tensor _autocast_to_reduced_precision(
    const Tensor& self,
    bool cuda_enabled,
    bool cpu_enabled,
    ScalarType cuda_dtype,
    ScalarType cpu_dtype) {
  if (self.dtype() == at::ScalarType::Float &&
      ((self.device().is_cuda() && cuda_enabled) ||
       (self.device().is_cpu() && cpu_enabled))) {
    at::ScalarType target = at::ScalarType::Undefined;
    if (self.device().is_cuda()) {
      target = cuda_dtype;
    } else if (self.device().is_cpu()) {
      target = cpu_dtype;
    }

    TORCH_INTERNAL_ASSERT(
        target != at::ScalarType::Undefined,
        "_autocast_to_reduced_precision requires legit ScalarType argument for given device");

    return to_impl(
        self,
        target,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        false,
        false,
        std::nullopt);
  } else {
    return self;
  }
}

// If input tensor is fp16, cast it to fp32, otherwise leave it alone.
// (this is intended to be used internally by the JIT autocast implementation)
Tensor _autocast_to_full_precision(
    const Tensor& self,
    bool cuda_enabled,
    bool cpu_enabled) {
  if ((self.dtype() == at::ScalarType::Half ||
       self.dtype() == at::ScalarType::BFloat16) &&
      ((self.device().is_cuda() && cuda_enabled) ||
       (self.device().is_cpu() && cpu_enabled))) {
    return to_impl(
        self,
        at::ScalarType::Float,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        false,
        false,
        std::nullopt);
  } else {
    return self;
  }
}

Tensor to(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self,
      dtype,
      layout,
      ensure_has_index(device),
      pin_memory,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(
    const Tensor& self,
    Device device,
    ScalarType dtype,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self,
      dtype,
      std::nullopt,
      ensure_has_index(device),
      std::nullopt,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(
    const Tensor& self,
    ScalarType dtype,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self,
      dtype,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(
    const Tensor& self,
    const Tensor& other,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  auto options = other.options();
  return to_impl(
      self,
      options.dtype().toScalarType(),
      options.layout(),
      options.device(),
      options.pinned_memory(),
      non_blocking,
      copy,
      optional_memory_format);
}

// This op is important primarily for lazy / graph-based backends.
// While this vanilla implementation loops through each tensor and independently
// converts it to cpu, a lazy backend like XLA might need to tell sync updates
// across tensors.
std::vector<Tensor> _to_cpu(TensorList tensors) {
  std::vector<Tensor> cpu_tensors;
  for (const auto& t : tensors) {
    cpu_tensors.push_back(t.cpu());
  }
  return cpu_tensors;
}

Tensor to_dense_backward(
    const Tensor& grad,
    const Tensor& input_,
    std::optional<bool> masked_grad_) {
  /*
    For historical reasons, to_dense backward implements masked
    semantics for sparse tensors, that is, gradients with respect to
    unspecified elements are ignored.  The masked_grad kw argument of
    to_dense is introduced to allow to_dense to be used in the
    non-masked semantics context. However, for BC reasons, the default
    value to masked_grad kw argument is set True as a first instance.
    Eventually, we should eliminate the masked_grad kw argument and
    let to_dense backward to behave according to non-masked
    semantics. Masked semantics of tensors is implemented in the
    framework of masked tensors.
  */
  const auto input_layout = input_.layout();
  const bool masked_grad = masked_grad_.value_or(true);
  switch (input_layout) {
    case kStrided:
      // TODO: return grad as it is
      return grad.to_dense(input_.scalar_type(), masked_grad_);
    case kSparse:
      // Autograd operates on the coalesced assumption, i.e. no duplicate
      // values.
      if (masked_grad) {
        return grad.sparse_mask(input_.coalesce());
      } else {
        // TODO: return grad as it is
        return grad.to_sparse(input_.sparse_dim());
      }
    case kSparseCsr:
    case kSparseCsc:
      // TODO: add efficient CSR/CSC support for sparse_mask
      if (masked_grad) {
        return grad.sparse_mask(input_.to_sparse(input_.sparse_dim()))
            .to_sparse(input_layout);
      } else {
        // TODO: return grad as it is
        return grad.to_sparse(
            input_layout,
            /*blocksize=*/std::nullopt,
            /*dense_dim=*/input_.dense_dim());
      }
    case kSparseBsr:
    case kSparseBsc: {
      // TODO: add efficient BSR/BSC support for sparse_mask
      const auto blocksize = at::sparse_csr::getBlockSize(input_);
      if (masked_grad) {
        return grad.sparse_mask(input_.to_sparse(input_.sparse_dim()))
            .to_sparse(input_layout, blocksize);
      } else {
        // TODO: return grad as it is
        return grad.to_sparse(input_layout, blocksize, input_.dense_dim());
      }
    }
    case kMkldnn:
      return grad.to_mkldnn(input_.scalar_type());
    default:
      TORCH_CHECK(
          false, "to_dense_backward: Unsupported input layout: ", input_layout);
      return Tensor{};
  }
}

Tensor to_mkldnn_backward(const Tensor& grad, const Tensor& input_) {
  AT_ASSERT(input_.layout() == c10::kStrided);
  return grad.to_dense(input_.scalar_type());
}

Tensor to_dense(
    const Tensor& tensor,
    std::optional<c10::ScalarType> dtype,
    std::optional<bool> masked_grad) {
  if (tensor.layout() == c10::kSparse) {
    return tensor._to_dense(dtype, masked_grad);
  }
  if (tensor.layout() == c10::kSparseCsr ||
      tensor.layout() == c10::kSparseCsc ||
      tensor.layout() == c10::kSparseBsr ||
      tensor.layout() == c10::kSparseBsc) {
    return tensor._to_dense(dtype, masked_grad);
  }
  if (tensor.layout() == c10::kMkldnn) {
    return tensor._to_dense(dtype, masked_grad);
  }
  TORCH_CHECK(
      tensor.layout() == c10::kStrided,
      "to_dense does not support layout ",
      tensor.layout());
  if (dtype) {
    return tensor.to(*dtype);
  }
  return tensor;
}

Tensor sparse_to_dense(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<bool> masked) {
  TORCH_CHECK(
      !dtype.has_value(), "dtype argument is not supported by sparse_to_dense");
  Tensor dst = at::zeros(self.sizes(), self.options().layout(kStrided));
  return dst.add_(self);
}

Tensor sparse_compressed_to_dense(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<bool> masked_grad) {
  TORCH_CHECK(
      !dtype.has_value(),
      "dtype argument is not supported by sparse_csr_to_dense");

  if (self.numel() == 0) {
    return at::zeros(self.sizes(), self.options().layout(kStrided));
  }

  auto batch_ndim = sparse_csr::numBatchDimensions(self);

  auto compressed_rows =
      self.layout() == kSparseCsr || self.layout() == kSparseBsr;
  auto block_sparse =
      self.layout() == kSparseBsr || self.layout() == kSparseBsc;

  auto [compressed_indices, plain_indices] =
      sparse_csr::getCompressedPlainIndices(self);

  auto values = self.values();
  Tensor dense = at::zeros(self.sizes(), self.options().layout(kStrided));

  if (batch_ndim == 0) {
    // Pad shape so we can treat non-batched like batched, we will
    // squeeze out the phantom batch dim at the end.
    compressed_indices.unsqueeze_(0);
    plain_indices.unsqueeze_(0);
    values.unsqueeze_(0);
    dense.unsqueeze_(0);
  }
  if (batch_ndim > 1) {
    // Flatten batch dims
    compressed_indices = compressed_indices.flatten(0, batch_ndim - 1);
    plain_indices = plain_indices.flatten(0, batch_ndim - 1);
    values = values.flatten(0, batch_ndim - 1);
    dense = dense.flatten(0, batch_ndim - 1);
  }

  // At this point there is only one batch dim, existed already or was
  // flattened from multiple batch dims.  Now, reshape the resulting
  // dense matrix so that this single batch dim is joined with sparse
  // dims into a single dim, so that the remaining dims are only block
  // dims eventually, and then dense dims.
  auto n_batch = values.size(0);
  int64_t nrows = 0, ncols = 0;
  auto dense_reshaped_sizes = dense.sizes().vec();
  if (!block_sparse) {
    nrows = self.size(batch_ndim);
    ncols = self.size(batch_ndim + 1);
    dense_reshaped_sizes.erase(
        dense_reshaped_sizes.begin(), dense_reshaped_sizes.begin() + 2);
  } else {
    std::array<int64_t, 2> blocksize = {values.size(2), values.size(3)};
    nrows = self.size(batch_ndim) / blocksize[0];
    ncols = self.size(batch_ndim + 1) / blocksize[1];
    dense_reshaped_sizes[1] = blocksize[0];
    dense_reshaped_sizes[2] = blocksize[1];
  }
  dense_reshaped_sizes[0] = n_batch * nrows * ncols;
  dense = dense.reshape(dense_reshaped_sizes);

  // Calculate batch, row and column indices for non-zeros in the
  // sparse matrix, and use these to calculate corresponding indices
  // into the dense matrix reshaped as above.  Then, update dense
  // matrix by adding sparse matrix values into elements with indices
  // calculated this way.
  auto options = compressed_indices.options();
  auto nnz_per_batch = values.size(1);
  auto batch_indices =
      at::arange(0, n_batch, options).repeat_interleave(nnz_per_batch);
  auto ncompressed = compressed_rows ? nrows : ncols;
  auto compressed_indices_over_all_batches = at::cat(
      {compressed_indices.slice(1, 0, ncompressed).flatten() +
           nnz_per_batch *
               at::arange(0, n_batch, options).repeat_interleave(ncompressed),
       n_batch * nnz_per_batch * at::ones({1}, options)});
  Tensor indices = at::_convert_indices_from_csr_to_coo(
      compressed_indices_over_all_batches,
      plain_indices.flatten(),
      false,
      !compressed_rows);
  auto row_indices = indices.select(0, 0);
  auto col_indices = indices.select(0, 1);
  if (compressed_rows) {
    row_indices -= batch_indices * nrows;
  } else {
    col_indices -= batch_indices * ncols;
  }
  auto offsets =
      col_indices + row_indices * ncols + batch_indices * nrows * ncols;
  dense.index_add_(0, offsets, values.flatten(0, 1));

  // Un-tile the result.  The final reshape uses the original
  // self.sizes() which will squeeze out the extra batch dim if we put
  // one in.
  if (!block_sparse) {
    return dense.reshape(self.sizes());
  } else {
    return dense.unflatten(0, {-1, nrows, ncols})
        .transpose(2, 3)
        .reshape(self.sizes());
  }
}

// Computes the strides for view_dtype output when the view dtype is
// smaller than the original dtype
static inline SymDimVector compute_strides_for_view_dtype_downsize(
    SymIntArrayRef old_strides,
    int64_t size_ratio,
    ScalarType old_dtype,
    ScalarType new_dtype) {
  const int64_t ndim = old_strides.size();

  TORCH_CHECK(
      old_strides[ndim - 1] == 1,
      "self.stride(-1) must be 1 to view ",
      old_dtype,
      " as ",
      new_dtype,
      " (different element sizes), but got ",
      old_strides[ndim - 1]);

  SymDimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    new_strides[dim_idx] = old_strides[dim_idx] * size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

// Computes the strides for view_dtype output when the view dtype is
// larger than the original dtype
static inline SymDimVector compute_strides_for_view_dtype_upsize(
    SymIntArrayRef old_strides,
    int64_t size_ratio,
    ScalarType old_dtype,
    ScalarType new_dtype) {
  const int64_t ndim = old_strides.size();
  TORCH_CHECK(
      old_strides[ndim - 1] == 1,
      "self.stride(-1) must be 1 to view ",
      old_dtype,
      " as ",
      new_dtype,
      " (different element sizes), but got ",
      old_strides[ndim - 1]);

  SymDimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    TORCH_CHECK(
        (old_strides[dim_idx] % size_ratio) == 0,
        "self.stride(",
        dim_idx,
        ") must be divisible by ",
        size_ratio,
        " to view ",
        old_dtype,
        " as ",
        new_dtype,
        " (different element sizes), ",
        "but got ",
        old_strides[dim_idx]);

    new_strides[dim_idx] = old_strides[dim_idx] / size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

Tensor view_dtype(const Tensor& self, ScalarType dtype) {
  const auto type_meta = c10::scalarTypeToTypeMeta(dtype);
  TORCH_CHECK(
      !self.is_conj(),
      "torch.Tensor.view is not supported for conjugate view tensors when converting to a different dtype.");
  TORCH_CHECK(
      !self.is_neg(),
      "torch.Tensor.view is not supported for tensors with negative bit set when converting to a different dtype.");

  int64_t self_element_size = self.element_size();
  int64_t new_element_size = static_cast<int64_t>(type_meta.itemsize());

  Storage storage = self.storage();
  auto new_tensor = detail::make_tensor<TensorImpl>(
      std::move(storage), self.key_set(), type_meta);
  auto* impl = new_tensor.unsafeGetTensorImpl();

  if (self_element_size == new_element_size) {
    impl->set_sizes_and_strides(
        self.sym_sizes(), self.sym_strides(), self.sym_storage_offset());

  } else if (self.dim() == 0) {
    TORCH_CHECK(
        false,
        "self.dim() cannot be 0 to view ",
        self.scalar_type(),
        " as ",
        dtype,
        " (different element sizes)");

  } else if (self_element_size > new_element_size) {
    // Downsizing element size

    int64_t size_ratio = self_element_size / new_element_size;
    auto new_strides = compute_strides_for_view_dtype_downsize(
        self.sym_strides(), size_ratio, self.scalar_type(), dtype);

    auto old_sizes = self.sym_sizes();
    SymDimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes[self.dim() - 1] *= size_ratio;

    auto new_storage_offset = size_ratio * self.sym_storage_offset();

    impl->set_sizes_and_strides(new_sizes, new_strides, new_storage_offset);

  } else {
    // Upsizing element size

    int64_t size_ratio = new_element_size / self_element_size;

    TORCH_CHECK(
        (self.sym_size(-1) % size_ratio) == 0,
        "self.size(-1) must be divisible by ",
        size_ratio,
        " to view ",
        self.scalar_type(),
        " as ",
        dtype,
        " (different element sizes), ",
        "but got ",
        self.sym_size(-1));

    TORCH_CHECK(
        (self.sym_storage_offset() % size_ratio) == 0,
        "self.storage_offset() must be divisible by ",
        size_ratio,
        " to view ",
        self.scalar_type(),
        " as ",
        dtype,
        " (different element sizes), but got ",
        self.sym_storage_offset());

    auto new_strides = compute_strides_for_view_dtype_upsize(
        self.sym_strides(), size_ratio, self.scalar_type(), dtype);

    auto old_sizes = self.sym_sizes();
    SymDimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes[self.dim() - 1] /= size_ratio;

    auto new_storage_offset = self.sym_storage_offset() / size_ratio;

    impl->set_sizes_and_strides(new_sizes, new_strides, new_storage_offset);
  }

  return new_tensor;
}

static Tensor _tile_tensor(const Tensor& self, IntArrayRef blocksize) {
  // This code turns a matrix into a sequence of blocks
  //
  // Given matrix
  //
  //  1  2  3  4
  //  5  6  7  8
  //  9 10 11 12
  // 14 15 16 17
  //
  // _tile_tensor(matrix, {2, 2}) will yield the following 2 by 2 blocks
  //
  //  1  2 |  3  4 |  9 10 | 11 12
  //  5  6 |  7  8 | 14 15 | 16 17
  //
  //  via a 4D Tensor of shape (2, 2, 2, 2)
  //
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[0] > 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[1] > 0);
  auto block_size_0 = self.size(0) / blocksize[0];
  auto block_size_1 = self.size(1) / blocksize[1];

  auto new_shape =
      DimVector({block_size_0, blocksize[0], block_size_1, blocksize[1]});
  new_shape.append(DimVector(self.sizes().slice(2, self.dim() - 2)));
  return self.reshape(new_shape).transpose(1, 2).contiguous();
}

static Tensor _batch_tile_tensor(
    const Tensor& self,
    IntArrayRef blocksize,
    const int64_t dense_dim) {
  if (self.dim() == 2 + dense_dim) {
    return _tile_tensor(self, blocksize);
  }
  auto n_batch_dim = self.dim() - 2 - dense_dim;
  // Same as _tile_tensor, just per matrix entry of self, if self is 3D.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[0] > 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[1] > 0);
  auto block_size_0 = self.size(n_batch_dim) / blocksize[0];
  auto block_size_1 = self.size(n_batch_dim + 1) / blocksize[1];
  auto tiled_sizes = DimVector(self.sizes().slice(0, n_batch_dim));
  tiled_sizes.push_back(block_size_0);
  tiled_sizes.push_back(blocksize[0]);
  tiled_sizes.push_back(block_size_1);
  tiled_sizes.push_back(blocksize[1]);
  tiled_sizes.append(DimVector(self.sizes().slice(n_batch_dim + 2, dense_dim)));
  return self.reshape(tiled_sizes)
      .transpose(n_batch_dim + 1, n_batch_dim + 2)
      .contiguous();
}

static Tensor _mask_to_indices(const Tensor& mask) {
  // This function returns a vector of the indices at which given
  // boolean mask is True. Here at::nonzero performs test (time/mem).
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      mask.dim() == 1, "_mask_to_indices only supports 1-d masks.");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      mask.dtype() == at::kBool, "Expected mask to be of dtype bool.");
  return at::native::flatten(at::nonzero(mask));
}

static std::pair<Tensor, Tensor> _not_zero_mask_to_col_row_indices(
    Tensor not_zero_mask) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      not_zero_mask.dim() == 2,
      "_not_zero_mask_to_col_row_indices only supports 2-d masks.");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      not_zero_mask.dtype() == at::kBool, "Expected mask to be of dtype bool.");
  auto nz = not_zero_mask.nonzero();
  return {nz.select(1, 1), nz.select(1, 0)};
}

// Sparse layout conversions Start

static inline void _to_sparse_check_arguments(
    const std::string& funcname,
    const Tensor& self,
    const int64_t sparse_dim) {
  auto layout_from = self.layout();

  auto layout_from_valid = layout_from == kStrided || layout_from == kSparse ||
      at::sparse_csr::is_sparse_compressed(layout_from);
  if (!layout_from_valid) {
    TORCH_CHECK(false, funcname, ": unexpected source layout ", layout_from);
  }

  if (layout_from == kStrided) {
    if (sparse_dim == 0 && self.dim() > 0) {
      TORCH_CHECK(
          false,
          funcname,
          ": sparse_dim argument must be in >0 when self.dim()>0");
    }
    if (sparse_dim < 0 || sparse_dim > self.dim()) {
      TORCH_CHECK(
          false,
          funcname,
          ": sparse_dim argument must be in [0,",
          self.dim(),
          "] range, but ",
          sparse_dim,
          " is given");
    }
  } else if (layout_from == kSparse) {
    if (sparse_dim != self.sparse_dim()) {
      TORCH_CHECK(
          false,
          funcname,
          ": conversion from ",
          layout_from,
          " to ",
          kSparse,
          " with sparse_dim argument !=self.sparse_dim() is not supported");
    }
  } else if (at::sparse_csr::is_sparse_compressed(layout_from)) {
    if (sparse_dim != 2) {
      TORCH_CHECK(
          false,
          funcname,
          ": conversion from ",
          layout_from,
          " to ",
          kSparse,
          " with sparse_dim argument !=2 is not supported");
    }
  }
}

static inline void _to_sparse_check_arguments(
    const std::string& funcname,
    const Tensor& self,
    std::optional<c10::Layout> layout,
    OptionalIntArrayRef blocksize,
    std::optional<int64_t> dense_dim_opt) {
  auto layout_from = self.layout();
  auto layout_to = layout.value_or(kSparse);

  auto layout_from_valid = layout_from == kStrided || layout_from == kSparse ||
      at::sparse_csr::is_sparse_compressed(layout_from);
  if (!layout_from_valid) {
    TORCH_CHECK(false, funcname, ": unexpected source layout ", layout_from);
  }
  auto layout_to_valid = layout_to == kStrided || layout_to == kSparse ||
      at::sparse_csr::is_sparse_compressed(layout_to);
  if (!layout_to_valid) {
    TORCH_CHECK(false, funcname, ": unexpected source layout ", layout_from);
  }

  if (layout_from == kSparse && layout_to != kSparse) {
    if (self.sparse_dim() != 2) {
      TORCH_CHECK(
          false,
          funcname,
          ": conversion from ",
          layout_from,
          " to ",
          layout_to,
          " for input tensors with sparse_dim()!=2 is not supported");
    }
  }

  if ((layout_from == kSparseCsr || layout_from == kSparseCsc) &&
      (layout_to == kSparseBsr || layout_to == kSparseBsc)) {
    if (sparse_csr::numBatchDimensions(self) > 0) {
      TORCH_CHECK(
          false,
          funcname,
          ": conversion from ",
          layout_from,
          " to ",
          layout_to,
          " for batched inputs is not supported");
    }
  }

  if (blocksize.has_value()) {
    if (blocksize.value().size() != 2) {
      TORCH_CHECK(
          false,
          funcname,
          ": blocksize needs to be a tuple of size 2, but got ",
          blocksize.value().size());
    }
    auto blocksize_to = *blocksize;
    if (blocksize_to[0] <= 0 || blocksize_to[1] <= 0) {
      TORCH_CHECK(
          false,
          funcname,
          ": blocksize needs to be positive, but got ",
          blocksize_to);
    }

    if (layout_to == kSparseBsr || layout_to == kSparseBsc) {
      if (layout_from == kSparseBsr || layout_from == kSparseBsc) {
        auto blocksize_from = at::sparse_csr::getBlockSize(self);
        if (!(blocksize_to == blocksize_from)) {
          TORCH_CHECK(
              false,
              funcname,
              ": conversion from ",
              layout_from,
              " to ",
              layout_to,
              " with blocksize changed from ",
              blocksize_from,
              " to ",
              blocksize_to,
              " is not supported");
        }
      } else {
        auto dense_dim = (layout_from == kStrided) ? dense_dim_opt.value_or(0)
                                                   : self.dense_dim();
        auto sparse_row_dim = -(dense_dim + 2);
        auto sparse_col_dim = -(dense_dim + 1);
        if ((self.size(sparse_row_dim) % blocksize_to[0] != 0) ||
            (self.size(sparse_col_dim) % blocksize_to[1] != 0)) {
          TORCH_CHECK(
              false,
              funcname,
              ": tensor sparse size (",
              self.size(sparse_row_dim),
              ",",
              self.size(sparse_row_dim),
              ") must be divisible by given blocksize (",
              blocksize_to[0],
              ",",
              blocksize_to[1],
              ")");
        }
      }
    } else {
      TORCH_CHECK(
          false,
          funcname,
          ": conversion from ",
          layout_from,
          " to ",
          layout_to,
          " with blocksize argument given is not supported");
    }
  } else {
    if ((layout_to == kSparseBsr || layout_to == kSparseBsc) &&
        !(layout_from == kSparseBsr && layout_from == kSparseBsc)) {
      TORCH_CHECK(
          false,
          funcname,
          ": conversion from ",
          layout_from,
          " to ",
          layout_to,
          " without blocksize argument given is not supported");
    }
  }

  if (dense_dim_opt.has_value()) {
    if (layout_from != kStrided) {
      TORCH_CHECK(
          false,
          funcname,
          ": conversion from ",
          layout_from,
          " to ",
          layout_to,
          " with dense_dim argument given is not supported");
    }

    auto dense_dim = *dense_dim_opt;
    if (layout_to == kSparse) {
      if (dense_dim == self.dim() && self.dim() > 0) {
        TORCH_CHECK(
            false,
            funcname,
            ": dense_dim argument must be !=self.dim() when self.dim()>0");
      }
      if (dense_dim < 0 || dense_dim > self.dim()) {
        TORCH_CHECK(
            false,
            funcname,
            ": dense_dim argument must be in [0,",
            self.dim(),
            "] range, but ",
            dense_dim,
            " is given");
      }
    } else {
      if (dense_dim < 0 || dense_dim > self.dim() - 2) {
        TORCH_CHECK(
            false,
            funcname,
            ": dense_dim argument must be in [0,",
            self.dim() - 2,
            "] range, but ",
            dense_dim,
            " is given");
      }
    }
  }
}

template <Layout target_layout>
static Tensor dense_to_sparse_compressed(
    const Tensor& self,
    const Tensor& self_mask,
    IntArrayRef blocksize,
    std::optional<int64_t> dense_dim_opt) {
  static_assert(
      target_layout == Layout::SparseCsr ||
          target_layout == Layout::SparseCsc ||
          target_layout == Layout::SparseBsr ||
          target_layout == Layout::SparseBsc,
      "invalid layout template parameter for dense_to_sparse_compressed");
  constexpr auto compressed_rows_layout =
      target_layout == Layout::SparseCsr || target_layout == Layout::SparseBsr;
  constexpr auto blocked_layout =
      target_layout == Layout::SparseBsr || target_layout == Layout::SparseBsc;

  int64_t dense_dim = dense_dim_opt.value_or(0);

  // Reshape values so that the block dims are explicitly added, and
  // calculate a mask tensor that has only batch and sparse dims, and
  // value true whenever sparse matrix has a non-zero element over
  // corresponding block and dense dims, and false otherwise.
  auto n_batch_dim = self.dim() - 2 - dense_dim;
  auto is_batched = n_batch_dim > 0;
  auto values =
      blocked_layout ? _batch_tile_tensor(self, blocksize, dense_dim) : self;
  auto not_zero_mask = blocked_layout
      ? _batch_tile_tensor(self_mask, blocksize, dense_dim)
      : self_mask;
  if (blocked_layout || dense_dim > 0) {
    std::vector<int64_t> reduce_dim((blocked_layout ? 2 : 0) + dense_dim);
    std::iota(reduce_dim.begin(), reduce_dim.end(), n_batch_dim + 2);
    not_zero_mask = not_zero_mask.sum(reduce_dim) != 0;
  }

  if (is_batched) {
    // Prepare for the conversion, in particular join the batch dims
    // and the compressed dim into the single dim.
    dense_to_sparse_compressed_prepare_check_mask_values_batched(
        target_layout, values, not_zero_mask, n_batch_dim);
  }

  // Calculate sparse matrix row and col indices and then, depending
  // on the target layout, corresponding compressed and sparse
  // indices.  Use the mask tensor calculate above to generate sparse
  // matrix values tensor.
  Tensor row_indices;
  Tensor col_indices;
  Tensor compressed_indices;
  if (compressed_rows_layout) {
    std::tie(col_indices, row_indices) =
        _not_zero_mask_to_col_row_indices(not_zero_mask);
    compressed_indices = at::_convert_indices_from_coo_to_csr(
        row_indices, not_zero_mask.size(0), false /*out_int32*/);
    {
      auto mask_indices = _mask_to_indices(not_zero_mask.flatten());
      values = values.flatten(0, 1).index_select(0, mask_indices);
    }
  } else {
    std::tie(row_indices, col_indices) =
        _not_zero_mask_to_col_row_indices(not_zero_mask.transpose(1, 0));
    compressed_indices = at::_convert_indices_from_coo_to_csr(
        col_indices, not_zero_mask.size(-1), false /*out_int32*/);
    {
      auto mask_indices =
          _mask_to_indices(not_zero_mask.transpose(0, 1).flatten());
      values =
          values.transpose(0, 1).flatten(0, 1).index_select(0, mask_indices);
    }
  }
  Tensor& plain_indices = compressed_rows_layout ? col_indices : row_indices;

  if (is_batched) {
    // Restore the batch dims and compressed dim.
    reshape_2d_sparse_compressed_members_to_nd_batched(
        self.sizes(), n_batch_dim, compressed_indices, plain_indices, values);
  }

  // Create compressed sparse matrix with the target layout.
  return at::_sparse_compressed_tensor_unsafe(
      compressed_indices,
      plain_indices,
      values,
      self.sizes(),
      self.options().layout(target_layout));
}

Tensor dense_to_sparse_with_mask(
    const Tensor& self,
    const Tensor& mask,
    std::optional<c10::Layout> layout,
    OptionalIntArrayRef blocksize,
    std::optional<int64_t> dense_dim_opt) {
  auto layout_to = layout.value_or(kSparse);
  TORCH_INTERNAL_ASSERT(
      self.layout() != layout_to,
      "dense_to_sparse: unexpected same input and output layout");
  TORCH_INTERNAL_ASSERT(
      self.layout() == mask.layout(),
      "dense_to_sparse_with_mask: expected mask layout ",
      self.layout(),
      ", got ",
      mask.layout());
  TORCH_INTERNAL_ASSERT(
      self.sizes() == mask.sizes(),
      "dense_to_sparse_with_mask: expected mask size ",
      self.sizes(),
      ", got ",
      mask.sizes());
  _to_sparse_check_arguments(
      "dense_to_sparse_with_mask", self, layout, blocksize, dense_dim_opt);

  switch (layout_to) {
    case kSparse:
      return self.sparse_mask(
          mask.to_sparse(self.dim() - dense_dim_opt.value_or(0)));
    case kSparseCsr:
      return dense_to_sparse_compressed<Layout::SparseCsr>(
          self, mask, {}, dense_dim_opt);
    case kSparseCsc:
      return dense_to_sparse_compressed<Layout::SparseCsc>(
          self, mask, {}, dense_dim_opt);
    case kSparseBsr:
      return dense_to_sparse_compressed<Layout::SparseBsr>(
          self, mask, *blocksize, dense_dim_opt);
    case kSparseBsc:
      return dense_to_sparse_compressed<Layout::SparseBsc>(
          self, mask, *blocksize, dense_dim_opt);
    default:
      break;
  }

  TORCH_CHECK(
      false,
      "dense_to_sparse_with_mask: ",
      self.layout(),
      " to ",
      layout_to,
      " conversion not supported");
  return Tensor{};
}

Tensor dense_to_sparse_csr(
    const Tensor& self,
    std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseCsr;
  _to_sparse_check_arguments(
      "dense_to_sparse_csr", self, layout_to, {}, dense_dim_opt);

  return dense_to_sparse_compressed<Layout::SparseCsr>(
      self, self != 0, {}, dense_dim_opt);
}

Tensor dense_to_sparse_csc(
    const Tensor& self,
    std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseCsc;
  _to_sparse_check_arguments(
      "dense_to_sparse_csc", self, layout_to, {}, dense_dim_opt);

  return dense_to_sparse_compressed<Layout::SparseCsc>(
      self, self != 0, {}, dense_dim_opt);
}

Tensor dense_to_sparse_bsr(
    const Tensor& self,
    IntArrayRef blocksize,
    std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseBsr;
  _to_sparse_check_arguments(
      "dense_to_sparse_bsr", self, layout_to, blocksize, dense_dim_opt);

  return dense_to_sparse_compressed<Layout::SparseBsr>(
      self, self != 0, blocksize, dense_dim_opt);
}

Tensor dense_to_sparse_bsc(
    const Tensor& self,
    IntArrayRef blocksize,
    std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseBsc;
  _to_sparse_check_arguments(
      "dense_to_sparse_bsc", self, layout_to, blocksize, dense_dim_opt);

  return dense_to_sparse_compressed<Layout::SparseBsc>(
      self, self != 0, blocksize, dense_dim_opt);
}

Tensor dense_to_sparse(
    const Tensor& self,
    std::optional<c10::Layout> layout,
    OptionalIntArrayRef blocksize,
    std::optional<int64_t> dense_dim_opt) {
  auto layout_to = layout.value_or(kSparse);
  TORCH_INTERNAL_ASSERT(
      self.layout() != layout_to,
      "dense_to_sparse: unexpected same input and output layout");
  _to_sparse_check_arguments(
      "dense_to_sparse", self, layout, blocksize, dense_dim_opt);

  switch (layout_to) {
    case kSparse:
      return self.to_sparse(self.dim() - dense_dim_opt.value_or(0));
    case kSparseCsr:
      return self.to_sparse_csr(dense_dim_opt);
    case kSparseCsc:
      return self.to_sparse_csc(dense_dim_opt);
    case kSparseBsr:
      return self.to_sparse_bsr(*blocksize, dense_dim_opt);
    case kSparseBsc:
      return self.to_sparse_bsc(*blocksize, dense_dim_opt);
    default:
      break;
  }

  TORCH_CHECK(
      false,
      "dense_to_sparse: ",
      self.layout(),
      " to ",
      layout_to,
      " conversion not supported");
  
```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 139 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `TORCH_IMPL_FUNC`, `at`

**Classes/Structs**: `index_t`, `scalar_t`, `index_t`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/Dispatch.h`
- `ATen/Parallel.h`
- `ATen/TensorOperators.h`
- `ATen/core/Tensor.h`
- `ATen/quantized/Quantizer.h`
- `optional`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_autocast_to_full_precision_native.h`
- `ATen/ops/_autocast_to_reduced_precision_native.h`
- `ATen/ops/_convert_indices_from_coo_to_csr.h`
- `ATen/ops/_convert_indices_from_coo_to_csr_native.h`
- `ATen/ops/_convert_indices_from_csr_to_coo.h`
- `ATen/ops/_convert_indices_from_csr_to_coo_native.h`
- `ATen/ops/_sparse_bsc_tensor_unsafe_native.h`
- `ATen/ops/_sparse_bsr_tensor_unsafe_native.h`
- `ATen/ops/_sparse_compressed_tensor_unsafe_native.h`
- `ATen/ops/_sparse_coo_tensor_unsafe_native.h`
- `ATen/ops/_sparse_coo_tensor_with_dims_native.h`
- `ATen/ops/_sparse_csc_tensor_unsafe_native.h`
- `ATen/ops/_sparse_csr_tensor_unsafe_native.h`
- `ATen/ops/_to_copy.h`
- `ATen/ops/_to_copy_native.h`
- `ATen/ops/_to_cpu_native.h`
- `ATen/ops/_to_dense_native.h`
- `ATen/ops/_to_sparse_bsc_native.h`
- `ATen/ops/_to_sparse_bsr_native.h`
- `ATen/ops/_to_sparse_csc_native.h`
- `ATen/ops/_to_sparse_csr_native.h`


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

- **File Documentation**: `TensorConversions.cpp_docs.md`
- **Keyword Index**: `TensorConversions.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
