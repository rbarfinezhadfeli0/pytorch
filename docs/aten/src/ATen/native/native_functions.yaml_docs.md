# Documentation: native_functions.yaml

## File Metadata
- **Path**: `aten/src/ATen/native/native_functions.yaml`
- **Size**: 618005 bytes
- **Lines**: 16102
- **Extension**: .yaml
- **Type**: Regular file

## Original Source

```yaml
# See README.md in this directory for more guidance

# *********NB: _cast_* operators are DEPRECATED and will be removed
# eventually. These were previously used before TorchScript IR supported
# representing ScalarType's. They are now superseded by usage of
# `aten::to()`. The ops remain here for backward compatibility purposes.

# DEPRECATED. DO NOT USE
- func: _cast_Byte(Tensor self, bool non_blocking=False) -> Tensor
  variants: function

# DEPRECATED. DO NOT USE
- func: _cast_Char(Tensor self, bool non_blocking=False) -> Tensor
  variants: function

# DEPRECATED. DO NOT USE
- func: _cast_Double(Tensor self, bool non_blocking=False) -> Tensor
  variants: function

# DEPRECATED. DO NOT USE
- func: _cast_Float(Tensor self, bool non_blocking=False) -> Tensor
  variants: function

# DEPRECATED. DO NOT USE
- func: _cast_Int(Tensor self, bool non_blocking=False) -> Tensor
  variants: function

# DEPRECATED. DO NOT USE
- func: _cast_Long(Tensor self, bool non_blocking=False) -> Tensor
  variants: function

# DEPRECATED. DO NOT USE
- func: _cast_Short(Tensor self, bool non_blocking=False) -> Tensor
  variants: function

# DEPRECATED. DO NOT USE
- func: _cast_Half(Tensor self, bool non_blocking=False) -> Tensor
  variants: function

# Computes the gradient of current tensor w.r.t. graph leaves.
- func: _backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()
  manual_cpp_binding: True
  variants: method

# DEPRECATED. Sets the tensor data held by this `Variable` to be the same as
# `new_data`.  It requires that `new_data` and `Variable` have compatible tensor
# type, by checking `_has_compatible_shallow_copy_type(this, new_data)`.
#
# This function is deprecated because it doesn't really make sense in a world
# where Variables *are* Tensors (as opposed to them containing tensors, which
# is what the previous interpretation was.)
- func: set_data(Tensor(a!) self, Tensor new_data) -> ()
  manual_cpp_binding: True
  variants: method

- func: data(Tensor self) -> Tensor
  manual_cpp_binding: True
  variants: method

# True if this `Variable` is a leaf and thus does not have a `grad_fn`.
- func: is_leaf(Tensor self) -> bool
  manual_cpp_binding: True
  variants: method

# Returns the output index of this variable from the forward operation that
# produced it.  Conversely, it returns the input index of the gradient `Node` to
# which this `Variable` is connected (because in the gradient computation,
# inputs and outputs switch meaning).  For example:
#
#   y0, y1, y2 = f(x)
#   assert y0.output_nr == 0
#   assert y1.output_nr == 1
#   assert y2.output_nr == 2
#
- func: output_nr(Tensor self) -> int
  manual_cpp_binding: True
  variants: method

- func: _version(Tensor self) -> int
  manual_cpp_binding: True
  variants: method

- func: requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)
  manual_cpp_binding: True
  variants: method

# Enables .grad attribute for non-leaf Tensors.
- func: retain_grad(Tensor(a!) self) -> ()
  manual_cpp_binding: True
  variants: method

- func: retains_grad(Tensor self) -> bool
  manual_cpp_binding: True
  variants: method

- func: _fw_primal(Tensor(a) self, int level) -> Tensor(a)
  variants: method
  dispatch:
    CompositeExplicitAutograd: _fw_primal

- func: _make_dual(Tensor(a) primal, Tensor tangent, int level) -> Tensor(a)
  variants: function
  dispatch:
    CompositeExplicitAutograd: _make_dual

- func: _unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor tangent)
  variants: function

# NOTE: [_new_zeros_with_same_feature_meta]
# This function creates a new tensor with the layout and TensorOptions
# of `other` but also takes into account the batch dimensions of `self`
#
# This function has a couple extra constraints because it is also used for `jvp`
# in functorch.
# - is used for forward AD because there is the restriction
#   that the primal and tangent must have the same layout
# - We cannot assume that `self` and `other` have the same sizes or even dim
#   because in the inplace over view case, `other` is the base tensor, and
#   `self` is the forward grad with respect to the view, which can have an
#   entirely different shape
# - takes the number of batch dims for `self` because we also handle
#   some batching logic. We handle that here instead of a batching rule because
#   we'd like to avoid calling as_strided in the batching rule (as to enable
#   nested vmap in functorch).
# - needs to be CompositeExplicitAutograd for jvp support in functorch.
#   functorch currently relies on TensorWrapper which does not have storage
#   CompositeExplicitAutograd makes sure the TensorWrapper is unwrapped.
# - this function may eventually take on another int argument to store the
#   the number of batch dims for other once we support that use case
- func: _new_zeros_with_same_feature_meta(Tensor self, Tensor other, *, int self_num_batch_dims=0) -> Tensor
  variants: function
  dispatch:
    CompositeExplicitAutograd: _new_zeros_with_same_feature_meta
  autogen: _new_zeros_with_same_feature_meta.out

# This function compares the storage numel of self with that of other, where
# storage numel is computed as: `other.storage().nbytes() / other.itemsize()`.
# We create this function for composite compliance purposes. The batching rule
# always returns true because vmapped as_strided does not support accessing
# storage locations not indexable by the input tensor.
# See the note above for more information.
- func: _has_same_storage_numel(Tensor self, Tensor other) -> bool
  variants: function
  dispatch:
    CompositeExplicitAutograd: _has_same_storage_numel

- func: rename_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)
  variants: method
  tags: inplace_view

- func: rename(Tensor(a) self, Dimname[]? names) -> Tensor(a)
  variants: method

- func: align_to(Tensor(a) self, Dimname[] names) -> Tensor(a)
  variants: method

- func: align_to.ellipsis_idx(Tensor(a) self, Dimname[] order, int ellipsis_idx) -> Tensor(a)
  variants: method

- func: align_as(Tensor self, Tensor other) -> Tensor
  variants: method

- func: align_tensors(Tensor[] tensors) -> Tensor[]

# Not assert because it's a keyword; not Assert because FX already
# took that syntax
# TODO: need to specify this is side-effectful somehow
- func: _assert_async(Tensor self) -> ()
  dispatch:
    CPU: _assert_async_cpu
    CUDA: _assert_async_cuda

- func: _assert_async.msg(Tensor self, str assert_msg) -> ()
  dispatch:
    CPU: _assert_async_msg_cpu
    CUDA: _assert_async_msg_cuda

- func: _assert_scalar(Scalar self, str assert_msg) -> ()
  dispatch:
    CompositeExplicitAutograd: _assert_scalar

- func: _functional_assert_scalar(Scalar self, str assert_msg, Tensor dep_token) -> Tensor
  dispatch:
    CompositeExplicitAutograd: _functional_assert_scalar

- func: _functional_assert_async.msg(Tensor self, str assert_msg, Tensor dep_token) -> Tensor
  dispatch:
    CPU: _functional_assert_async_msg_cpu

- func: _assert_tensor_metadata(Tensor a, SymInt[]? size=None, SymInt[]? stride=None, ScalarType? dtype=None, *, Device? device=None, Layout? layout=None) -> ()
  dispatch:
    CompositeExplicitAutograd: _assert_tensor_metadata
    Meta: _assert_tensor_metadata_meta_symint

- func: _async_error(str msg) -> ()
  dispatch:
    CompositeExplicitAutograd: _async_error
    Meta: _async_error_meta

- func: _print(str s) -> ()
  dispatch:
    CompositeExplicitAutograd: _print

- func: sym_constrain_range(Scalar size, *, int? min=None, int? max=None) -> ()
  dispatch:
    CompositeExplicitAutograd: sym_constrain_range

- func: sym_constrain_range_for_size(Scalar size, *, int? min=None, int? max=None) -> ()
  dispatch:
    CompositeExplicitAutograd: sym_constrain_range_for_size

- func: _functional_sym_constrain_range(Scalar size, int? min, int? max, Tensor dep_token) -> Tensor
  dispatch:
    CompositeExplicitAutograd: _functional_sym_constrain_range

- func: _functional_sym_constrain_range_for_size(Scalar size, int? min, int? max, Tensor dep_token) -> Tensor
  dispatch:
    CompositeExplicitAutograd: _functional_sym_constrain_range_for_size

- func: _make_dep_token(*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  dispatch:
    CPU: _make_dep_token_cpu

- func: refine_names(Tensor(a) self, Dimname[] names) -> Tensor(a)
  variants: method

- func: _use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank) -> bool
  device_check: NoCheck  # Tensor arguments allowed to be on different devices, see also _cudnn_ctc_loss
  dispatch:
    CUDA: _use_cudnn_ctc_loss

- func: _use_cudnn_ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank) -> bool
  device_check: NoCheck  # Tensor arguments allowed to be on different devices, see also _cudnn_ctc_loss
  dispatch:
    CUDA: _use_cudnn_ctc_loss_tensor

- func: _cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
  device_check: NoCheck  # log_probs is expected to be on CUDA while targets is expected to be on CPU
  dispatch:
    CUDA: _cudnn_ctc_loss
  autogen: _cudnn_ctc_loss.out

- func: _cudnn_ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
  device_check: NoCheck  # log_probs is expected to be on CUDA while targets is expected to be on CPU
  dispatch:
    CUDA: _cudnn_ctc_loss_tensor

- func: _use_cudnn_rnn_flatten_weight() -> bool

- func: _cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, SymInt input_size, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor
  dispatch:
    CUDA: _cudnn_rnn_flatten_weight
  autogen: _cudnn_rnn_flatten_weight.out

- func: _cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
  # rnn_tanh may or may not redispatch to _cudnn_rnn based on algorithm and build. Thus it might hit dispatch or kernel device check.
  # Disable dispatch time device check for consistent behavior.
  device_check: NoCheck
  dispatch:
    CUDA: _cudnn_rnn
  autogen: _cudnn_rnn.out
  tags: nondeterministic_seeded

- func: _cudnn_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
  dispatch:
    CUDA: _cudnn_rnn_backward
  autogen: _cudnn_rnn_backward.out

- func: _cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
  dispatch:
    CUDA: _cudnn_init_dropout_state
  autogen: _cudnn_init_dropout_state.out
  tags: nondeterministic_seeded

- func: _debug_has_internal_overlap(Tensor self) -> int
  variants: function

- func: _fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)
  variants: function
  dispatch:
    CUDA: fused_dropout_cuda
  tags: nondeterministic_seeded
  autogen: _fused_dropout.out

- func: _masked_scale(Tensor self, Tensor mask, float scale) -> Tensor
  variants: function
  dispatch:
    CUDA: masked_scale_cuda
  autogen: _masked_scale.out

- func: native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)
  variants: function
  dispatch:
    CPU: native_dropout_cpu
    CUDA: native_dropout_cuda
    MPS: native_dropout_mps
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: native_dropout_nested
  tags: [nondeterministic_seeded, core]
  autogen: native_dropout.out

- func: native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor
  dispatch:
    CPU, NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: native_dropout_backward
    CUDA: native_dropout_backward_cuda
    MPS: native_dropout_backward_mps
  autogen: native_dropout_backward.out
  tags: pointwise

- func: _sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)

- func: _sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)

- func: _sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)

- func: _sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)

- func: _reshape_from_tensor(Tensor self, Tensor shape) -> Tensor

- func: _shape_as_tensor(Tensor self) -> Tensor

- func: dropout(Tensor input, float p, bool train) -> Tensor
  tags: [nondeterministic_seeded, maybe_aliasing_or_mutating]

- func: dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  tags: nondeterministic_seeded

- func: feature_dropout(Tensor input, float p, bool train) -> Tensor
  tags: [nondeterministic_seeded, maybe_aliasing_or_mutating]

- func: feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  tags: nondeterministic_seeded

- func: alpha_dropout(Tensor input, float p, bool train) -> Tensor
  tags: [nondeterministic_seeded, maybe_aliasing_or_mutating]

- func: alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  tags: nondeterministic_seeded

- func: feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor
  tags: [nondeterministic_seeded, maybe_aliasing_or_mutating]

- func: feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  tags: nondeterministic_seeded

- func: abs(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: abs
    SparseCPU, SparseCUDA, SparseMPS: abs_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMPS, SparseCsrMeta: abs_sparse_csr
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_abs
  tags: [core, pointwise]

- func: abs_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: abs_
    SparseCPU, SparseCUDA, SparseMPS: abs_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMPS, SparseCsrMeta: abs_sparse_csr_
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_abs_

- func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA, MPS, MTIA: abs_out
    SparseCPU, SparseCUDA, SparseMPS: abs_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMPS, SparseCsrMeta: abs_sparse_csr_out
  tags: pointwise

# Note [Adding an alias]
# To add an alias do the following:
#
# 1) Copy the original functions native_functions.yaml entry, but replace the
#      original function's name with their own and delete any dispatch
#      keys for the aliases. Specifying a dispatch key will prevent
#      autograd from recording the operations the alias performs, which
#      will stop it from "inheriting" the original operation's autograd behavior.
# 2) Implement the corresponding functions and have them redispatch to the
#      original function.
# 3) Add docstrings to the new function that reference the original function,
#      and document the method as usual (if it exists.)
#    (See torch/_torch_docs.py and docs/source/torch.rst if adding a function,
#     torch/_tensor_docs.py and docs/source/tensors.rst if adding a method,
#     or module-specific doc bindings (like torch/linalg/__init__.py) if
#     adding an alias in a namespace.)
# 4) Update torch/overrides.py consistent with the original function.
# 5) Update the alias_map in torch/csrc/jit/passes/normalize_ops.cpp.
# 6) Add aliases argument to existing OpInfo/UnaryUfuncInfo or create new OpInfo/UnaryUfuncInfo entry
# in op_db list in torch/testing/_internal/common_methods_invocations.py
#
# See torch.absolute, an alias for torch.abs, as an example.
# Absolute, alias for abs

- func: absolute(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method

- func: absolute_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method

- func: absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator

- func: angle(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CPU, CUDA, MPS: angle
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: angle_sparse_csr
  tags: pointwise

- func: angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA, MPS: angle_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: angle_sparse_csr_out
  tags: pointwise

- func: view_as_real(Tensor(a) self) -> Tensor(a)
  variants: function
  dispatch:
    CPU, CUDA, MPS, Meta: view_as_real

- func: view_as_complex(Tensor(a) self) -> Tensor(a)
  variants: function
  dispatch:
    CPU, CUDA, MPS, Meta: view_as_complex

- func: sgn(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: sgn.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: sgn_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: sgn_sparse_csr
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_sgn
  tags: pointwise

- func: sgn_(Tensor(a!) self) -> Tensor(a!)
  variants: method
  structured_delegate: sgn.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: sgn_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: sgn_sparse_csr_
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_sgn_
  tags: pointwise

- func: sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: sgn_out
    MPS: sgn_out_mps
    SparseCPU, SparseCUDA, SparseMPS: sgn_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: sgn_sparse_csr_out
  tags: pointwise

- func: chalf(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
  variants: method

- func: real(Tensor(a) self) -> Tensor(a)
  device_check: NoCheck   # TensorIterator
  variants: function

- func: imag(Tensor(a) self) -> Tensor(a)
  device_check: NoCheck   # TensorIterator
  variants: function

- func: _conj(Tensor(a) self) -> Tensor(a)
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: _conj

- func: conj(Tensor(a) self) -> Tensor(a)
  variants: function, method
  manual_cpp_binding: True

- func: _conj_physical(Tensor self) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: _conj_physical
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMPS, SparseCsrMeta: conj_physical_sparse_csr
  autogen: _conj_physical.out

- func: conj_physical(Tensor self) -> Tensor
  variants: function, method
  tags: [pointwise, maybe_aliasing_or_mutating]

- func: conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU, CUDA: conj_physical_out
    MPS: conj_physical_out_mps
    SparseCPU, SparseCUDA, SparseMPS: conj_physical_out_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMPS, SparseCsrMeta: conj_physical_sparse_csr_out
  tags: pointwise

- func: conj_physical_(Tensor(a!) self) -> Tensor(a!)
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: conj_physical_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: conj_physical_sparse_csr_
  tags: pointwise

- func: resolve_conj(Tensor(a) self) -> Tensor(a)
  variants: function, method

- func: resolve_neg(Tensor(a) self) -> Tensor(a)
  variants: function, method

- func: _neg_view(Tensor(a) self) -> Tensor(a)
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: _neg_view

- func: acos(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: acos.out
  tags: [core, pointwise]

- func: acos_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: acos.out
  tags: pointwise

- func: acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: acos_out
  tags: pointwise

# arccos, alias of acos
- func: arccos(Tensor self) -> Tensor
  variants: function, method

- func: arccos_(Tensor(a!) self) -> Tensor(a!)
  variants: function, method

- func: arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

- func: avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
  tags: core
  autogen: avg_pool1d.out

- func: adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor
  tags: core
  autogen: adaptive_avg_pool1d.out

# Return: (Tensor output, Tensor indices)
- func: adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)

- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: add.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS, SparseMeta: add_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: add_sparse_csr
    MkldnnCPU: mkldnn_add
    ZeroTensor: add_zerotensor
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_add_Tensor
  tags: [core, pointwise]

- func: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  structured_delegate: add.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS, SparseMeta: add_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: add_sparse_csr_
    MkldnnCPU: mkldnn_add_
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_add__Tensor
  tags: pointwise

- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  ufunc_inner_loop:
    Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf)
    ScalarOnly: add (Bool)
  dispatch:
    SparseCPU, SparseMeta: add_out_sparse_cpu
    SparseCUDA: add_out_sparse_cuda
    SparseMPS: add_out_sparse_mps
    SparseCsrCPU, SparseCsrMeta: add_out_sparse_compressed_cpu
    SparseCsrCUDA: add_out_sparse_compressed_cuda
    MkldnnCPU: mkldnn_add_out
    MPS: add_out_mps
    MTIA: add_out_mtia
  tags: pointwise

- func: _add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function
  dispatch:
    CPU: add_relu

- func: _add_relu_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
  variants: function
  dispatch:
    CPU: add_relu_

- func: _add_relu.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  variants: function
  dispatch:
    CPU: add_relu_out

- func: _add_relu.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  variants: function
  dispatch:
    CPU: add_relu

- func: _add_relu_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
  variants: function
  dispatch:
    CPU: add_relu_
  autogen: _add_relu.Scalar_out

# For C++ only, until we have conversion from C++ numbers to Tensor
- func: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: add
  tags: [core, pointwise]

- func: add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  dispatch:
    CompositeExplicitAutograd: add_
  autogen: add.Scalar_out
  tags: pointwise

- func: addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  structured_delegate: addmv.out
  variants: function, method

- func: addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
  structured_delegate: addmv.out
  variants: function, method

- func: addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  structured: True
  dispatch:
    CPU: addmv_out_cpu
    CUDA: addmv_out_cuda
    MPS: addmv_out_mps
    XPU: addmv_out_xpu
    SparseCsrCPU: addmv_out_sparse_compressed
    SparseCsrCUDA: addmv_out_sparse_compressed_cuda

- func: addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU, CUDA: addr
    MPS: addr_mps
    CompositeExplicitAutograd: math_addr

- func: addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
  variants: method
  dispatch:
    CompositeExplicitAutograd: addr_

- func: addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU, CUDA: addr_out
    MPS: addr_out_mps
    CompositeExplicitAutograd: math_addr_out

- func: affine_grid_generator(Tensor theta, SymInt[] size, bool align_corners) -> Tensor
  variants: function
  dispatch:
    CompositeExplicitAutograd: affine_grid_generator
  autogen: affine_grid_generator.out

- func: affine_grid_generator_backward(Tensor grad, SymInt[] size, bool align_corners) -> Tensor
  variants: function

- func: _is_all_true(Tensor self) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: _is_all_true

- func: _is_any_true(Tensor self) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: _is_any_true

# Note: this function is only for testing.
- func: _test_check_tensor(Tensor self) -> Tensor
  variants: function

# Note; this function is only for testing
- func: _test_functorch_fallback(Tensor self, Tensor other) -> Tensor
  variants: function
  dispatch:
    CPU: _test_functorch_fallback
  autogen: _test_functorch_fallback.out

- func: all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: all.out
  variants: function, method
  dispatch:
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_all
  tags: reduction


- func: all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: all.dims_out
  variants: function, method
  cpp_no_default_args: ['dim']
  dispatch:
    CompositeExplicitAutograd: all_dims_default
  tags: reduction

- func: all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  dispatch:
    CPU, CUDA: all_out
    MPS: all_out_mps
    MTIA: all_out_mtia
  tags: reduction

- func: all.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  dispatch:
    CPU, CUDA: all_dims_out
    CompositeExplicitAutograd: all_dims_out_default
  cpp_no_default_args: ['dim']
  tags: reduction

- func: all.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  tags: reduction

- func: all.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  tags: reduction

- func: allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
  variants: function, method
  tags: data_dependent_output
  dispatch:
    CompositeExplicitAutograd: allclose

- func: any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: any.out
  variants: function, method
  tags: [core, reduction]

- func: any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: any.dims_out
  variants: function, method
  cpp_no_default_args: ['dim']
  tags: [core, reduction]
  dispatch:
    CompositeExplicitAutograd: any_dims_default

- func: any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  dispatch:
    CPU, CUDA: any_out
    MPS: any_out_mps
  tags: reduction

- func: any.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  dispatch:
    CPU, CUDA: any_dims_out
    CompositeExplicitAutograd: any_dims_out_default
  cpp_no_default_args: ['dim']
  tags: reduction

- func: any.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  tags: reduction

- func: any.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  tags: reduction

- func: arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CompositeExplicitAutograd: arange

- func: arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CompositeExplicitAutograd: arange

# This operator should be named `arange.start_out` if following the naming convention. However that
# name is already taken. Disabled because of CI job failures.
# FIXME: enable this
#- func: arange.start_out_(Scalar start, Scalar end, *, Tensor(a!) out) -> Tensor(a!)
#  dispatch:
#    CompositeExplicitAutograd: arange_start_out

- func: arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CompositeExplicitAutograd: arange
  cpp_no_default_args: ['step']
  tags: core

- func: arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CompositeExplicitAutograd: arange_out

- func: arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU, Meta: arange_out
    CUDA: arange_cuda_out
    MPS: arange_mps_out
    MTIA: arange_mtia_out
  cpp_no_default_args: ['step']

# This function is a temporary hack to allow tracing of arange like constructs with dynamic
# bounds on arange.  Normal arange is not traceable because it does not take any tensor inputs;
# if the range you need is based on another tensor, calling this function directly will
# preserve tracing.  Get rid of this when arange can directly take tensors for bounds
# (so that it can be traced directly).
- func: _dim_arange(Tensor like, int dim) -> Tensor

- func: argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  structured_delegate: argmax.out
  device_check: NoCheck   # TensorIterator
  variants: function, method
  tags: [core, reduction]

- func: argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  dispatch:
    CPU, CUDA: argmax_out
    MPS: argmax_out_mps
  tags: reduction

- func: argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  structured_delegate: argmin.out
  device_check: NoCheck   # TensorIterator
  variants: function, method
  tags: [core, reduction]

- func: argmin.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  dispatch:
    CPU, CUDA: argmin_out
    MPS: argmin_out_mps
  tags: reduction

- func: acosh(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: acosh.out
  tags: [core, pointwise]

- func: acosh_(Tensor(a!) self) -> Tensor(a!)
  variants: function, method
  structured_delegate: acosh.out
  tags: pointwise

- func: acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: acosh_out
    MPS: acosh_out_mps
  tags: pointwise
# arccosh, alias for acosh

- func: arccosh(Tensor self) -> Tensor
  variants: function, method

- func: arccosh_(Tensor(a!) self) -> Tensor(a!)
  variants: function, method

- func: arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

- func: asinh(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: asinh.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: asinh_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: asinh_sparse_csr
  tags: [core, pointwise]

- func: asinh_(Tensor(a!) self) -> Tensor(a!)
  variants: function, method
  structured_delegate: asinh.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: asinh_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: asinh_sparse_csr_
  tags: pointwise

- func: asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: asinh_out
    MPS: asinh_out_mps
    SparseCPU, SparseCUDA, SparseMPS: asinh_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: asinh_sparse_csr_out
  tags: pointwise

# arcsinh, alias for asinh
- func: arcsinh(Tensor self) -> Tensor
  variants: function, method

- func: arcsinh_(Tensor(a!) self) -> Tensor(a!)
  variants: function, method

- func: arcsinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

- func: atanh(Tensor self) -> Tensor
  structured_delegate: atanh.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: atanh_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: atanh_sparse_csr
  tags: [core, pointwise]

- func: atanh_(Tensor(a!) self) -> Tensor(a!)
  structured_delegate: atanh.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: atanh_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: atanh_sparse_csr_
  tags: pointwise

- func: atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: atanh_out
    MPS: atanh_out_mps
    SparseCPU, SparseCUDA, SparseMPS: atanh_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: atanh_sparse_csr_out
  tags: pointwise
# arctanh, alias for atanh

- func: arctanh(Tensor self) -> Tensor
  variants: function, method

- func: arctanh_(Tensor(a!) self) -> Tensor(a!)
  variants: function, method

- func: arctanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

- func: as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
  variants: function, method
  dispatch:
    ZeroTensor, CPU, CUDA, MTIA, MPS: as_strided_tensorimpl
    Meta: as_strided_tensorimpl_meta_symint
    QuantizedCPU, QuantizedCUDA: as_strided_qtensorimpl
  device_check: NoCheck
  device_guard: False
  tags: core

- func: as_strided_(Tensor(a!) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a!)
  use_const_ref_for_mutable_tensors: True
  variants: function, method
  device_check: NoCheck
  device_guard: False
  tags: inplace_view
  dispatch:
    CompositeExplicitAutogradNonFunctional: as_strided__symint

- func: asin(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: asin.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: asin_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: asin_sparse_csr
  tags: [core, pointwise]

- func: asin_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: asin.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: asin_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: asin_sparse_csr_
  tags: pointwise

- func: asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: asin_out
    SparseCPU, SparseCUDA, SparseMPS: asin_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: asin_sparse_csr_out
  tags: pointwise

# arcsin, alias of asin
- func: arcsin(Tensor self) -> Tensor
  variants: function, method

- func: arcsin_(Tensor(a!) self) -> Tensor(a!)
  variants: function, method

- func: arcsin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

- func: atan(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: atan.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: atan_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: atan_sparse_csr
  tags: [core, pointwise]

- func: atan_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured_delegate: atan.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: atan_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: atan_sparse_csr_
  tags: pointwise

- func: atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: atan_out
    SparseCPU, SparseCUDA, SparseMPS: atan_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: atan_sparse_csr_out
  tags: pointwise

# arctan, alias of atan
- func: arctan(Tensor self) -> Tensor
  variants: function, method

- func: arctan_(Tensor(a!) self) -> Tensor(a!)
  variants: function, method

- func: arctan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

- func: atleast_1d(Tensor self) -> Tensor
  variants: function
  tags: maybe_aliasing_or_mutating

- func: atleast_1d.Sequence(Tensor[] tensors) -> Tensor[]

- func: atleast_2d(Tensor self) -> Tensor
  variants: function
  tags: maybe_aliasing_or_mutating

- func: atleast_2d.Sequence(Tensor[] tensors) -> Tensor[]
  variants: function

- func: atleast_3d(Tensor self) -> Tensor
  variants: function
  tags: maybe_aliasing_or_mutating

- func: atleast_3d.Sequence(Tensor[] tensors) -> Tensor[]
  variants: function

- func: baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  variants: function, method
  structured_delegate: baddbmm.out

- func: baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
  variants: method
  structured_delegate: baddbmm.out

- func: baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  structured: True
  variants: function
  dispatch:
    CPU: baddbmm_out_cpu
    CUDA: baddbmm_out_cuda
    MPS: baddbmm_out_mps
    XPU: baddbmm_out_xpu
    MTIA: baddbmm_out_mtia
    SparseCsrCUDA: baddbmm_out_sparse_csr_cuda

- func: baddbmm.dtype(Tensor self, Tensor batch1, Tensor batch2, ScalarType out_dtype, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  variants: function
  dispatch:
    CUDA: _baddbmm_dtype_cuda

- func: baddbmm.dtype_out(Tensor self, Tensor batch1, Tensor batch2, ScalarType out_dtype, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  variants: function
  dispatch:
    CUDA: _baddbmm_out_dtype_cuda

- func: bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CompositeExplicitAutograd: bartlett_window
  autogen: bartlett_window.out

- func: bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CompositeExplicitAutograd: bartlett_window
  autogen: bartlett_window.periodic_out

- func: batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
  tags: maybe_aliasing_or_mutating

- func: quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
  dispatch:
    QuantizedCPU: quantized_batch_norm
  autogen: quantized_batch_norm.out

- func: _batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)
  tags: maybe_aliasing_or_mutating

- func: _batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)

# Sample bernoulli with values in `self` as probability.
- func: bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: bernoulli
  tags: nondeterministic_seeded

- func: bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: function
  tags: nondeterministic_seeded
  dispatch:
    CPU, CUDA: bernoulli_out
    MPS: bernoulli_out_mps

- func: bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  tags: nondeterministic_seeded
  dispatch:
    CPU, CUDA: bernoulli_
    MPS: bernoulli_mps_
  autogen: bernoulli.Tensor, bernoulli.Tensor_out

- func: bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  tags: nondeterministic_seeded
  dispatch:
    CPU, CUDA: bernoulli_
    MPS: bernoulli_mps_
  autogen: bernoulli.float_out

# Note [bernoulli.p schema]
# We should probably just fix the overload ambiguity by appending a _functional to the C++ API name (BC breaking)
# This out-of-place version isn't used explicitly, but needed by jit.
# There is no default valid on `p` here because it would introduce ambiguity
# with `bernoulli(Tensor self, *, Generator? generator=None)` declaration.
- func: bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  tags: nondeterministic_seeded
  dispatch:
    CompositeExplicitAutogradNonFunctional: bernoulli

- func: bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> Tensor

- func: binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
  device_check: NoCheck   # TensorIterator
  python_module: nn
  variants: function
  dispatch:
    CPU: binary_cross_entropy_cpu
    CUDA: binary_cross_entropy_cuda
    MPS: binary_cross_entropy_mps

- func: binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  python_module: nn
  variants: function
  dispatch:
    CPU: binary_cross_entropy_out_cpu
    CUDA: binary_cross_entropy_out_cuda
    MPS: binary_cross_entropy_out_mps

- func: binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
  python_module: nn
  variants: function
  dispatch:
    CPU: binary_cross_entropy_backward_cpu
    CUDA: binary_cross_entropy_backward_cuda
    MPS: binary_cross_entropy_backward_mps

- func: binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
  python_module: nn
  variants: function
  dispatch:
    CPU: binary_cross_entropy_backward_out_cpu
    CUDA: binary_cross_entropy_backward_out_cuda
    MPS: binary_cross_entropy_backward_out_mps

- func: binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function
  dispatch:
    CompositeExplicitAutograd: binary_cross_entropy_with_logits
  autogen: binary_cross_entropy_with_logits.out

- func: bincount(Tensor self, Tensor? weights=None, SymInt minlength=0) -> Tensor
  variants: function, method
  dispatch:
    CPU: _bincount_cpu
    CUDA: _bincount_cuda
    MPS: _bincount_mps
  tags: dynamic_output_shape
  autogen: bincount.out

- func: bitwise_not(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: bitwise_not.out
  variants: function, method
  tags: [core, pointwise]

- func: bitwise_not_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured_delegate: bitwise_not.out
  variants: method
  tags: pointwise

- func: bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS, MTIA: bitwise_not_out
  tags: pointwise

- func: copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: copysign_out
  tags: pointwise

- func: copysign.Tensor(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: copysign.out
  tags: pointwise

- func: copysign_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  structured_delegate: copysign.out

- func: copysign.Scalar(Tensor self, Scalar other) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: copysign
  tags: pointwise

- func: copysign_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
  variants: method
  dispatch:
    CompositeExplicitAutograd: copysign_

- func: copysign.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CompositeExplicitAutograd: copysign_out
  tags: pointwise

- func: _lazy_clone(Tensor self) -> Tensor
  # Like clone, but the copy takes place lazily, only if either the
  # input or the output are written.
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: _lazy_clone

- func: logical_not(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: logical_not
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_logical_not
  tags: [core, pointwise]

- func: logical_not_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  dispatch:
    CompositeExplicitAutograd: logical_not_
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_logical_not_
  tags: pointwise

- func: logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA, MTIA: logical_not_out
    MPS: logical_not_out_mps
  tags: pointwise

- func: logical_xor(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: logical_xor
  tags: [core, pointwise]

- func: logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  dispatch:
    CompositeExplicitAutograd: logical_xor_
  tags: pointwise

- func: logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA: logical_xor_out
    MPS: logical_xor_out_mps
  tags: pointwise

- func: logical_and(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: logical_and
  tags: [core, pointwise]

- func: logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  dispatch:
    CompositeExplicitAutograd: logical_and_
  tags: pointwise

- func: logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA, MTIA: logical_and_out
    MPS: logical_and_out_mps
  tags: pointwise

- func: logical_or(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: logical_or
  tags: [core, pointwise]

- func: logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  dispatch:
    CompositeExplicitAutograd: logical_or_
  tags: pointwise

- func: logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA, MTIA: logical_or_out
    MPS: logical_or_out_mps
  tags: pointwise

- func: blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CompositeExplicitAutograd: blackman_window
  autogen: blackman_window.out

- func: blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CompositeExplicitAutograd: blackman_window
  autogen: blackman_window.periodic_out

- func: bmm(Tensor self, Tensor mat2) -> Tensor
  structured_delegate: bmm.out
  variants: function, method
  dispatch:
    SparseCPU: bmm_sparse_cpu
    SparseCUDA: bmm_sparse_cuda
    SparseMPS: bmm_sparse_mps
    NestedTensorCPU: bmm_nested
    NestedTensorCUDA: bmm_nested_cuda
  tags: core

- func: bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  variants: function
  dispatch:
    CPU: bmm_out_cpu
    CUDA: bmm_out_cuda
    MPS: bmm_out_mps
    XPU: bmm_out_xpu
    MTIA: bmm_out_mtia
    SparseCPU: bmm_out_sparse_cpu
    SparseCUDA: bmm_out_sparse_cuda
    SparseMPS: bmm_out_sparse_mps
    SparseCsrCUDA: bmm_out_sparse_csr_cuda

- func: bmm.dtype(Tensor self, Tensor mat2, ScalarType out_dtype) -> Tensor
  variants: function
  dispatch:
    CUDA: _bmm_dtype_cuda

- func: bmm.dtype_out(Tensor self, Tensor mat2, ScalarType out_dtype, *, Tensor(a!) out) -> Tensor(a!)
  variants: function
  dispatch:
    CUDA: _bmm_out_dtype_cuda

- func: broadcast_tensors(Tensor[] tensors) -> Tensor[]
  device_check: NoCheck
  device_guard: False

- func: broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)
  variants: function, method
  dispatch:
    CompositeImplicitAutograd: broadcast_to_symint

- func: _sparse_broadcast_to(Tensor(a) self, int[] size) -> Tensor(a)
  variants: function
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: sparse_broadcast_to

- func: cat(Tensor[] tensors, int dim=0) -> Tensor
  structured_delegate: cat.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: cat_sparse
    QuantizedCPU: cat_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: cat_nested
  tags: core

- func: cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  precomputed:
  - dim -> int dim, int valid, bool all_contiguous, bool all_same_dtype, bool all_same_sizes_and_stride, MemoryFormat memory_format
  dispatch:
    CPU: cat_out_cpu
    CUDA: cat_out_cuda
    MPS: cat_out_mps
    QuantizedCPU: cat_out_quantized_cpu

- func: cat.names(Tensor[] tensors, Dimname dim) -> Tensor

- func: cat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)

# alias for torch.cat
- func: concat(Tensor[] tensors, int dim=0) -> Tensor

- func: concat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)

- func: concat.names(Tensor[] tensors, Dimname dim) -> Tensor

- func: concat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)

# alias for torch.cat
- func: concatenate(Tensor[] tensors, int dim=0) -> Tensor

- func: concatenate.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)

- func: concatenate.names(Tensor[] tensors, Dimname dim) -> Tensor

- func: concatenate.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)

- func: block_diag(Tensor[] tensors) -> Tensor
  variants: function
  dispatch:
    CompositeExplicitAutograd: block_diag
  autogen: block_diag.out

- func: ceil(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: ceil.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: ceil_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: ceil_sparse_csr
  tags: [core, pointwise]

- func: ceil_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured_delegate: ceil.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: ceil_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: ceil_sparse_csr_
  tags: pointwise

- func: ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: ceil_out
    SparseCPU, SparseCUDA, SparseMPS: ceil_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: ceil_sparse_csr_out
  tags: pointwise

# alias for torch.linalg.multi_dot
- func: chain_matmul(Tensor[] matrices) -> Tensor
  variants: function

# alias for torch.linalg.multi_dot
- func: chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)

- func: unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]
  variants: function, method
  device_check: NoCheck
  device_guard: False
  tags: maybe_aliasing_or_mutating

- func: chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]
  variants: function, method
  device_check: NoCheck
  device_guard: False
  dispatch:
    CompositeImplicitAutograd: chunk
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: chunk_nested_tensor

- func: tensor_split.sections(Tensor(a -> *) self, SymInt sections, int dim=0) -> Tensor(a)[]
  variants: function, method
  dispatch:
    CompositeImplicitAutograd: tensor_split_sections_symint

- func: tensor_split.indices(Tensor(a -> *) self, SymInt[] indices, int dim=0) -> Tensor(a)[]
  variants: function, method
  dispatch:
    CompositeImplicitAutograd: tensor_split_indices_symint

- func: tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]
  variants: function, method

- func: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  cpp_no_default_args: ['min']
  structured_delegate: clamp.out
  dispatch:
    QuantizedCPU: clamp_quantized_cpu
  tags: [core, pointwise]

- func: clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
  variants: function, method
  structured_delegate: clamp.Tensor_out
  tags: [core, pointwise]

- func: clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  cpp_no_default_args: ['min']
  structured_delegate: clamp.out
  tags: pointwise

- func: clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
  variants: function, method
  structured_delegate: clamp.Tensor_out
  tags: pointwise

- func: clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  cpp_no_default_args: ['min']
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MTIA: clamp_out
    MPS: clamp_out_mps
  tags: pointwise

- func: clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: clamp_Tensor_out
    MPS: clamp_Tensor_out_mps
  tags: pointwise

- func: clamp_max(Tensor self, Scalar max) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: clamp_max.out
  tags: pointwise

- func: clamp_max.Tensor(Tensor self, Tensor max) -> Tensor
  variants: function, method
  structured_delegate: clamp_max.Tensor_out
  tags: pointwise

- func: clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: clamp_max.out
  tags: pointwise

- func: clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)
  variants: function, method
  structured_delegate: clamp_max.Tensor_out
  tags: pointwise

- func: clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MTIA: clamp_max_out
    MPS: clamp_max_out_mps
  tags: pointwise

- func: clamp_max.Tensor_out(Tensor self, Tensor max, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: clamp_max_Tensor_out
    MPS: clamp_max_Tensor_out_mps
  tags: pointwise

- func: clamp_min(Tensor self, Scalar min) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: clamp_min.out
  tags: pointwise

- func: clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
  variants: function, method
  structured_delegate: clamp_min.Tensor_out
  tags: pointwise

- func: clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: clamp_min.out
  tags: pointwise

- func: clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!)
  variants: function, method
  structured_delegate: clamp_min.Tensor_out
  tags: pointwise

- func: clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MTIA: clamp_min_out
    MPS: clamp_min_out_mps
  tags: pointwise

- func: clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: clamp_min_Tensor_out
    MPS: clamp_min_Tensor_out_mps
  tags: pointwise

# clip is an alias for clamp
- func: clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
  cpp_no_default_args: ['min']
  variants: function, method
  tags: pointwise

- func: clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
  variants: function, method
  tags: pointwise

- func: clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
  cpp_no_default_args: ['min']
  variants: function, method
  tags: pointwise

- func: clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
  variants: function, method
  tags: pointwise

- func: clip.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
  cpp_no_default_args: ['min']
  tags: pointwise

- func: clip.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)

- func: cudnn_is_acceptable(Tensor self) -> bool
  device_check: NoCheck
  device_guard: False

- func: complex(Tensor real, Tensor imag) -> Tensor
  variants: function
  dispatch:
    CompositeExplicitAutograd: complex

- func: complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU, CUDA, MPS: complex_out

- func: polar(Tensor abs, Tensor angle) -> Tensor
  variants: function
  dispatch:
    CompositeExplicitAutograd: polar

- func: polar.out(Tensor abs, Tensor angle, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU, CUDA, MPS: polar_out

- func: constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
  variants: function
  dispatch:
    CompositeExplicitAutograd: constant_pad_nd
    MPS: constant_pad_nd_mps
  autogen: constant_pad_nd.out
  tags: core

- func: contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)
  variants: method
  manual_cpp_binding: True

- func: convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
  dispatch:
    CompositeExplicitAutograd: convolution
  autogen: convolution.out
  tags: core

- func: convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
  dispatch:
    CompositeExplicitAutograd, CUDA: convolution_backward
  autogen: convolution_backward.out
  tags: core

- func: convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
  dispatch:
    CompositeExplicitAutograd: convolution_overrideable
  autogen: convolution_overrideable.out

- func: convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
  dispatch:
    CompositeExplicitAutograd: convolution_backward_overrideable
  autogen: convolution_backward_overrideable.out

- func: _convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor
  dispatch:
    CompositeExplicitAutograd: _convolution
  autogen: _convolution.out

- func: _convolution.deprecated(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, int[] output_padding, SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor

- func: _convolution_mode(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, str padding, SymInt[] dilation, SymInt groups) -> Tensor
  dispatch:
    CompositeImplicitAutograd: _convolution_mode_symint

- func: _convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

- func: conv1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] dilation=1, SymInt groups=1) -> Tensor
  dispatch:
    CompositeImplicitAutograd: conv1d_symint

- func: conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor
  dispatch:
    CompositeImplicitAutograd: conv2d_symint

- func: conv3d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] dilation=1, SymInt groups=1) -> Tensor
  dispatch:
    CompositeImplicitAutograd: conv3d_symint

- func: conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, str padding="valid", SymInt[1] dilation=1, SymInt groups=1) -> Tensor
  cpp_no_default_args: ['bias', 'stride', 'padding']
  dispatch:
    CompositeImplicitAutograd: conv1d_padding_symint

- func: conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, str padding="valid", SymInt[2] dilation=1, SymInt groups=1) -> Tensor
  cpp_no_default_args: ['bias', 'stride', 'padding']
  dispatch:
    CompositeImplicitAutograd: conv2d_padding_symint

- func: conv3d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, str padding="valid", SymInt[3] dilation=1, SymInt groups=1) -> Tensor
  cpp_no_default_args: ['bias', 'stride', 'padding']
  dispatch:
    CompositeImplicitAutograd: conv3d_padding_symint

- func: conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor
  dispatch:
    CompositeExplicitAutograd: conv_tbc
  autogen: conv_tbc.out

- func: conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)

# NB: we inherit the goofy argument order from PyTorch torch.nn.functional
- func: conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] output_padding=0, SymInt groups=1, SymInt[1] dilation=1) -> Tensor
  dispatch:
    CompositeImplicitAutograd: conv_transpose1d_symint

- func: conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, SymInt groups=1, SymInt[2] dilation=1) -> Tensor
  dispatch:
    CompositeImplicitAutograd: conv_transpose2d_symint

- func: conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] output_padding=0, SymInt groups=1, SymInt[3] dilation=1) -> Tensor
  dispatch:
    CompositeImplicitAutograd: conv_transpose3d_symint

- func: copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
  variants: function
  dispatch:
    Meta: copy_meta
    CompositeExplicitAutogradNonFunctional: copy
  tags: core

- func: copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
  variants: method
  device_check: NoCheck
  device_guard: False
  dispatch:
    MkldnnCPU: copy_mkldnn_
    SparseCPU, SparseCUDA, SparseMPS: copy_sparse_wrapper_
    CompositeExplicitAutograd: copy_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: copy_sparse_compressed_
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: copy_nested_
  autogen: copy.out

- func: _copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
  dispatch:
    MPS: _copy_from_mps
  autogen: _copy_from.out

# We need this to be able to properly copy from a CPU to an XLA tensor with different sizes.
# See https://github.com/pytorch/xla/issues/2881
- func: _copy_from_and_resize(Tensor self, Tensor dst) -> Tensor
  dispatch:
    MPS: _copy_from_and_resize_mps
  autogen: _copy_from_and_resize.out

- func: cos(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: cos.out
  dispatch:
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_cos
  tags: [core, pointwise]

- func: cos_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: cos.out
  tags: pointwise

- func: cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS, MTIA: cos_out
  tags: pointwise

- func: cosh(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: cosh.out
  tags: [core, pointwise]

- func: cosh_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: cosh.out
  tags: pointwise

- func: cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: cosh_out
  tags: pointwise

- func: cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor

- func: count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
  variants: function, method
  dispatch:
    CPU: count_nonzero_cpu
    CUDA: count_nonzero_cuda
    MPS: count_nonzero_mps
  autogen: count_nonzero.dim_IntList_out
  tags: reduction

- func: count_nonzero(Tensor self, int? dim=None) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: count_nonzero
  autogen: count_nonzero.out
  tags: reduction

- func: cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> Tensor
  variants: function, method

- func: corrcoef(Tensor self) -> Tensor
  variants: function, method

- func: cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid
  dispatch:
    CUDA: cudnn_affine_grid_generator_forward
  autogen: cudnn_affine_grid_generator.out

# TODO: Why do I have to call this grad?!
- func: cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor grad_theta
  dispatch:
    CUDA: cudnn_affine_grid_generator_backward
  autogen: cudnn_affine_grid_generator_backward.out

- func: cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)
  dispatch:
    CUDA: cudnn_batch_norm

- func: cudnn_batch_norm.out(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))
  dispatch:
    CUDA: cudnn_batch_norm_out

# NB: You can only use this if you used cudnn_batch_norm training=True
- func: cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace) -> (Tensor, Tensor, Tensor)
  dispatch:
    CUDA: cudnn_batch_norm_backward
  autogen: cudnn_batch_norm_backward.out

- func: cudnn_convolution(Tensor self, Tensor weight, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
  dispatch:
    CUDA: cudnn_convolution

- func: cudnn_convolution.out(Tensor self, Tensor weight, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CUDA: cudnn_convolution_out

- func: cudnn_convolution_transpose(Tensor self, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
  dispatch:
    CUDA: cudnn_convolution_transpose
  autogen: cudnn_convolution_transpose.out

- func: _mps_convolution_transpose(Tensor self, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups) -> Tensor
  dispatch:
    MPS: _mps_convolution_transpose
  autogen: _mps_convolution_transpose.out

- func: mps_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool[2] output_mask) -> (Tensor, Tensor)
  dispatch:
    MPS: mps_convolution_transpose_backward
  autogen: mps_convolution_transpose_backward.out

- func: cudnn_convolution_relu(Tensor self, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups) -> Tensor
  dispatch:
    CUDA: cudnn_convolution_relu
  autogen: cudnn_convolution_relu.out

- func: cudnn_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups) -> Tensor
  dispatch:
    CUDA: cudnn_convolution_add_relu
  autogen: cudnn_convolution_add_relu.out

# NB: input is special cased in a way I don't quite understand
- func: cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output
  dispatch:
    CUDA: cudnn_grid_sampler_forward
  autogen: cudnn_grid_sampler.out

- func: cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)
  dispatch:
    CUDA: cudnn_grid_sampler_backward
  autogen: cudnn_grid_sampler_backward.out

- func: cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: cummax

- func: cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CompositeExplicitAutograd: cummax_out

- func: cummax.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
  device_check: NoCheck   # TensorIterator
  variants: function, method

- func: cummax.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
  device_check: NoCheck   # TensorIterator

- func: _cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
  variants: function
  dispatch:
    CPU: cummax_helper_cpu
    CUDA: cummax_helper_cuda
    MPS: cummax_helper_mps

- func: cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: cummin

- func: cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CompositeExplicitAutograd: cummin_out

- func: cummin.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
  device_check: NoCheck   # TensorIterator
  variants: function, method

- func: cummin.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
  device_check: NoCheck   # TensorIterator

- func: _cummin_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
  variants: function
  dispatch:
    CPU: cummin_helper_cpu
    CUDA: cummin_helper_cuda
    MPS: cummin_helper_mps

- func: cummaxmin_backward(Tensor grad, Tensor input, Tensor indices, int dim) -> Tensor
  variants: function
  device_check: NoCheck
  device_guard: False

- func: cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
  structured_delegate: cumprod.out
  device_check: NoCheck   # TensorIterator
  variants: function, method

- func: cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
  structured_delegate: cumprod.out
  variants: method

- func: cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
  structured: True
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA: cumprod_out
    MPS: cumprod_out_mps

- func: cumprod.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method

- func: cumprod_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)
  variants: method

- func: cumprod.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator

- func: cumprod_backward(Tensor grad, Tensor input, int dim, Tensor output) -> Tensor
  variants: function
  device_check: NoCheck
  device_guard: False

- func: cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
  structured_delegate: cumsum.out
  device_check: NoCheck   # TensorIterator
  variants: function, method
  tags: core

- func: cumsum_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
  structured_delegate: cumsum.out
  variants: method

- func: cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
  structured: True
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA: cumsum_out
    MPS: cumsum_out_mps

- func: cumsum.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method

- func: cumsum_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)
  variants: method

- func: cumsum.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator

- func: cumulative_trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor

- func: cumulative_trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor

- func: ctc_loss.IntList(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor

# convenience function that converts to intlists for you
- func: ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor

- func: _ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
  dispatch:
    CPU: ctc_loss_cpu
    CUDA: ctc_loss_gpu
    Meta: ctc_loss_meta
  autogen: _ctc_loss.out
  tags: dynamic_output_shape  # the shape of second output is data dependent

- func: _ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
  dispatch:
    CPU, CUDA: ctc_loss_tensor
  autogen: _ctc_loss.Tensor_out
  tags: dynamic_output_shape  # the shape of second output is data dependent

- func: _ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor
  dispatch:
    CPU: ctc_loss_backward_cpu
    CUDA: ctc_loss_backward_gpu
  autogen: _ctc_loss_backward.out

- func: _ctc_loss_backward.Tensor(Tensor grad, Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor
  dispatch:
    CPU, CUDA: ctc_loss_backward_tensor

- func: diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutogradNonFunctional: diag_embed
  autogen: diag_embed.out

- func: diagflat(Tensor self, int offset=0) -> Tensor
  variants: function, method

- func: diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: diagonal
  tags: core

- func: linalg_diagonal(Tensor(a) A, *, int offset=0, int dim1=-2, int dim2=-1) -> Tensor(a)
  python_module: linalg
  variants: function

- func: diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)
  variants: function, method

- func: diagonal_backward(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2) -> Tensor
  variants: function
  device_check: NoCheck
  device_guard: False
  dispatch:
    CompositeExplicitAutograd: diagonal_backward_symint
  autogen: diagonal_backward.out

- func: fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)
  variants: method

- func: diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor
  variants: function, method

- func: diff.out(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None, *, Tensor(a!) out) -> Tensor(a!)
  variants: function

- func: gradient.scalarint(Tensor self, *, Scalar? spacing=None, int? dim=None, int edge_order=1) -> Tensor[]
  variants: function

- func: gradient.scalararray(Tensor self, *, Scalar spacing, int[] dim, int edge_order=1) -> Tensor[]
  variants: function

- func: gradient.array(Tensor self, *, int[] dim, int edge_order=1) -> Tensor[]
  variants: function

- func: gradient.scalarrayint(Tensor self, *, Scalar[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
  variants: function

- func: gradient.scalarrayarray(Tensor self, *, Scalar[] spacing, int[] dim, int edge_order=1) -> Tensor[]
  variants: function

- func: gradient.tensorarrayint(Tensor self, *, Tensor[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
  variants: function

- func: gradient.tensorarray(Tensor self, *, Tensor[] spacing, int[] dim, int edge_order=1) -> Tensor[]
  variants: function

- func: div.Tensor(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: div.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: div_sparse
    ZeroTensor: div_zerotensor
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_div_Tensor
  tags: [core, pointwise]

- func: div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  structured_delegate: div.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: div_sparse_
  tags: pointwise

- func: div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS, MTIA: div_out
    SparseCPU, SparseCUDA, SparseMPS: div_out_sparse_zerodim
  tags: pointwise

- func: div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: div.out_mode
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: div_sparse
  tags: [core, pointwise]

- func: div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  structured_delegate: div.out_mode
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: div_sparse_
  tags: pointwise

- func: div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: div_out_mode
    SparseCPU, SparseCUDA, SparseMPS: div_out_sparse_zerodim
  tags: pointwise

# For C++ only, until we have conversion from C++ numbers to Tensor
- func: div.Scalar(Tensor self, Scalar other) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: div
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_div_Scalar
  tags: [core, pointwise]

- func: div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  dispatch:
    CompositeExplicitAutograd: div_
  autogen: div.Scalar_out
  tags: pointwise

- func: div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: div
  tags: [core, pointwise]

- func: div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
  variants: method
  dispatch:
    CompositeExplicitAutograd: div_
  autogen: div.Scalar_mode_out
  tags: pointwise

# divide, alias for div
- func: divide.Tensor(Tensor self, Tensor other) -> Tensor
  variants: function, method

- func: divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  variants: method

- func: divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

- func: divide.Scalar(Tensor self, Scalar other) -> Tensor
  variants: function, method

- func: divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
  variants: method

- func: divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
  variants: function, method

- func: divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
  variants: method

- func: divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)

- func: divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
  variants: function, method

- func: divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
  variants: method

  # true_divide, an alias for div
- func: true_divide.Tensor(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  tags: pointwise

- func: true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method

- func: true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator

- func: true_divide.Scalar(Tensor self, Scalar other) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method

- func: true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method

- func: dot(Tensor self, Tensor tensor) -> Tensor
  variants: function, method
  dispatch:
    CPU: dot
    CUDA: dot_cuda
    MPS: dot_mps

- func: dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CompositeExplicitAutograd: dot_out

- func: vdot(Tensor self, Tensor other) -> Tensor
  variants: function, method
  dispatch:
    CPU: vdot
    CUDA: vdot_cuda

- func: vdot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CompositeExplicitAutograd: vdot_out

- func: einsum(str equation, Tensor[] tensors, *, int[]? path=None) -> Tensor

- func: embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
  dispatch:
    CompositeExplicitAutograd: embedding_symint
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_embedding
  autogen: embedding.out
  tags: core

- func: embedding_backward(Tensor grad, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor
  dispatch:
    CompositeImplicitAutograd: embedding_backward_symint

- func: embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor
  dispatch:
    CPU: embedding_dense_backward_cpu
    CUDA: embedding_dense_backward_cuda
    MPS: embedding_dense_backward_mps
  autogen: embedding_dense_backward.out
  tags: core

- func: embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)
  dispatch:
    CPU: embedding_renorm_cpu_
    CUDA: embedding_renorm_cuda_
  autogen: embedding_renorm, embedding_renorm.out

- func: embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor

# NOTE [ embedding_bag Native Functions ]
# The `_embedding_bag.*` variants assume that input tensors except for `weight`,
# e.g. `indices` and `offsets` (and `offset2bag`), are contiguous.
# We really only need to enforce this for `_embedding_bag` (the forward) because
# the backward inputs are the same as forward ones.
# The above `embedding_bag` wrapper is created to achieve this, e.g.,
# applying indices = indices.contiguous().
# The backward functions apply a check that these input tensors are contiguous.


- func: _embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
  dispatch:
    CPU: _embedding_bag_forward_only_cpu
    CUDA: _embedding_bag_forward_only_cuda
    MPS: _embedding_bag_forward_only_mps
  autogen: _embedding_bag_forward_only.out

- func: _rowwise_prune(Tensor weight, Tensor mask, ScalarType compressed_indices_dtype) -> (Tensor, Tensor)

# row_stack is the alias of vstack
- func: row_stack(Tensor[] tensors) -> Tensor

- func: row_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)

- func: embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)

# To keep backward and forward compatibility, and to avoid ambiguity with the
# original signature above, scale_grad_by_freq, mode, sparse,
# per_sample_weights, and include_last_offset parameters do not have default
# values. Once the original signature is removed, default values can be added.
- func: embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)

- func: _embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
  dispatch:
    CPU: _embedding_bag_cpu
    CUDA: _embedding_bag_cuda
    MPS: _embedding_bag_mps
  autogen: _embedding_bag.out
  tags: core

- func: _embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, SymInt num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
  dispatch:
    CPU, CUDA, MPS: _embedding_bag_backward_symint

- func: _embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, SymInt num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
  dispatch:
    CompositeImplicitAutograd: _embedding_bag_sparse_backward_symint

- func: _embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, SymInt num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
  dispatch:
    CPU: _embedding_bag_dense_backward_cpu
    CUDA: _embedding_bag_dense_backward_cuda
    MPS: _embedding_bag_dense_backward_mps
  autogen: _embedding_bag_dense_backward.out

- func: _embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1) -> Tensor
  dispatch:
    CPU: _embedding_bag_per_sample_weights_backward_cpu
    CUDA: _embedding_bag_per_sample_weights_backward_cuda
    MPS: _embedding_bag_per_sample_weights_backward_mps
  autogen: _embedding_bag_per_sample_weights_backward.out

- func: empty.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  device_check: NoCheck
  device_guard: False
  dispatch:
    CompositeExplicitAutograd: empty_names
  autogen: empty.names_out

- func: empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  dispatch:
    CPU: empty_cpu
    CUDA: empty_cuda
    MPS: empty_mps
    Meta: empty_meta_symint
    MkldnnCPU: empty_mkldnn
    SparseCPU, SparseCUDA, SparseMPS: empty_sparse
    SparseMeta: empty_sparse_symint
    SparseCsrCPU, SparseCsrCUDA: empty_sparse_compressed
    SparseCsrMeta: empty_sparse_compressed_symint
    QuantizedCPU, QuantizedCUDA, QuantizedMeta: empty_unknown_quantized
  tags: core

- func: empty_permuted(SymInt[] size, int[] physical_layout, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CompositeExplicitAutograd: empty_permuted_symint
  autogen: empty_permuted.out

# We do not make new_empty a composite that calls into new_empty_strided, as the strided version
# is significantly more difficult to implement by different backends
- func: new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  variants: method
  dispatch:
    CompositeExplicitAutograd: new_empty_symint
  autogen: new_empty.out

- func: new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  variants: method
  dispatch:
    CompositeExplicitAutogradNonFunctional: new_empty_strided_symint
  autogen: new_empty_strided.out

- func: new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  variants: method
  dispatch:
    # NB: Although this composite mutates on the inside, it is
    # non-differentiable so NonFunctional doesn't apply
    CompositeExplicitAutograd: new_full
  autogen: new_full.out

- func: new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  variants: method
  dispatch:
    # NB: Although this composite mutates on the inside, it is
    # non-differentiable so NonFunctional doesn't apply
    CompositeExplicitAutograd: new_zeros
  autogen: new_zeros.out

- func: new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  variants: method
  dispatch:
    # NB: Although this composite mutates on the inside, it is
    # non-differentiable so NonFunctional doesn't apply
    CompositeExplicitAutograd: new_ones
  autogen: new_ones.out

# other overrides are to provide a more helpful error message that dtype is required
- func: _empty_affine_quantized(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor
  dispatch:
    CPU: empty_affine_quantized_other_backends_stub
    QuantizedCPU, QuantizedCUDA: empty_affine_quantized
  autogen: _empty_affine_quantized.out

# it's a factory function receiving a tensor argument, thus overriding explicitly
# other overrides are to provide a more helpful error message that dtype is required
- func: _empty_per_channel_affine_quantized(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor
  category_override: factory
  dispatch:
    CPU: empty_per_channel_affine_quantized_other_backends_stub
    QuantizedCPU, QuantizedCUDA: empty_per_channel_affine_quantized
  autogen: _empty_per_channel_affine_quantized.out

- func: resize_(Tensor(a!) self, SymInt[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
  use_const_ref_for_mutable_tensors: True
  variants: method
  device_check: NoCheck
  device_guard: False
  tags: [core, inplace_view]
  dispatch:
    Meta: resize__symint
    CPU: resize_
    CUDA: resize_cuda_
    MPS: resize_mps_
    QuantizedCPU: quantized_resize_cpu_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: resize_sparse_csr_
  autogen: resize, resize.out

# This is a utility function to enable users to resize out tensor while registering kernels for out variants.
# Eventually, we can consider exposing `resize_output` as a public API to ship it with python op registration
# to make it easy to register out variants for ops.
- func: _resize_output_(Tensor(a!) self, SymInt[] size, Device device) -> Tensor(a!)
  use_const_ref_for_mutable_tensors: True
  variants: function
  dispatch:
    Meta: _resize_output_
  autogen: _resize_output, _resize_output.out

- func: empty_quantized(int[] size, Tensor qtensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  category_override: factory
  variants: function
  dispatch:
    QuantizedCPU, QuantizedCUDA: empty_quantized
  autogen: empty_quantized.out

- func: empty.out(SymInt[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck
  device_guard: False

- func: empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  device_check: NoCheck
  device_guard: False
  dispatch:
    CompositeExplicitAutograd: empty_like
    QuantizedCPU, QuantizedCUDA: empty_like_quantized
    SparseCPU, SparseCUDA, SparseMPS, SparseMeta: empty_like_sparse_coo
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: empty_like_sparse_csr
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: empty_like_nested
  autogen: empty_like.out

- func: empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CPU: empty_strided_cpu
    CUDA: empty_strided_cuda
    MPS: empty_strided_mps
    Meta: empty_strided_meta_symint
    QuantizedCPU, QuantizedCUDA: empty_strided_unknown_quantized
  autogen: empty_strided.out
  tags: core

- func: erf(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: erf.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: erf_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: erf_sparse_csr
  tags: [core, pointwise]

- func: erf_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured_delegate: erf.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: erf_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: erf_sparse_csr_
  tags: pointwise

- func: erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS, MTIA: erf_out
    SparseCPU, SparseCUDA, SparseMPS: erf_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: erf_sparse_csr_out
  tags: pointwise

- func: erfc(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: erfc.out
  variants: function, method
  tags: pointwise

- func: erfc_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured_delegate: erfc.out
  variants: function, method
  tags: pointwise

- func: erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: erfc_out
  tags: pointwise

- func: exp(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: exp.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: exp_sparse
  tags: [core, pointwise]

- func: exp_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured_delegate: exp.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: exp_sparse_
  tags: pointwise

- func: exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterato

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a configuration file.

## Detailed Walkthrough


## Key Components

The file contains 61271 words across 16102 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 618005 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
