# Documentation: `tools/autograd/gen_variable_type.py`

## File Metadata

- **Path**: `tools/autograd/gen_variable_type.py`
- **Size**: 84,498 bytes (82.52 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# Generates VariableType.h/cpp
#
# **If any changes are being made to the VariableType codegen please also check
# if updates are needed in torch/csrc/autograd/autograd_not_implemented_fallback.cpp
#
# VariableType is a subclass of at::Type that provides the binding code
# necessary to provide a differentiable version of ATen operators. There are a
# number of different things we could mean:
#
#   - Given a non-differentiable forward implementation, we might
#     directly associate it with a backward implementation to make
#     it differentiable.  This is the common case.
#
#   - Some functions don't need a backwards implementation, because
#     backpropagation will never propagate beyond them.  There are a
#     number of different reasons why this may be the case:
#
#       - The function has no differentiable inputs
#       - The function's output is not differentiable
#       - The function has no data dependency on its input
#
#   - Some function don't need a backwards implementation because they
#     are implemented as a composition of other (differentiable) ATen
#     functions.  These are dispatched directly to the Type superclass,
#     which will in turn dispatch back to VariableType for its
#     differentiable subcomponents.
#

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from torchgen.api import cpp
from torchgen.api.autograd import (
    DifferentiableInput,
    dispatch_strategy,
    ForwardDerivative,
    gen_differentiable_outputs,
    is_differentiable,
    NativeFunctionWithDifferentiabilityInfo,
    SavedAttribute,
)
from torchgen.api.types import (
    ArrayRefCType,
    BaseCppType,
    BaseCType,
    Binding,
    intArrayRefT,
    iTensorListRefT,
    ListCType,
    MutRefCType,
    OptionalCType,
    scalarT,
    SpecialArgName,
    stringT,
    symIntArrayRefT,
    TENSOR_LIST_LIKE_CTYPES,
    tensorListT,
    tensorT,
    TupleCType,
    VectorCType,
)
from torchgen.code_template import CodeTemplate
from torchgen.context import (
    native_function_manager,
    with_native_function,
    with_native_function_and,
)
from torchgen.model import (
    Argument,
    BaseType,
    ListType,
    NativeFunction,
    SchemaKind,
    SelfArgument,
    TensorOptionsArguments,
)
from torchgen.utils import FileManager, mapMaybe

from .context import with_native_function_with_differentiability_info_and_key
from .gen_inplace_or_view_type import (
    ALL_VIEW_FUNCTIONS,
    ASSIGN_RETURN_VALUE,
    AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION,
    gen_formals,
    get_base_name,
    get_view_info,
    is_tensor_list_type,
    is_tensor_type,
    METHOD_DEFINITION,
    modifies_arguments,
    TMP_VAR,
    unpack_args,
    unpacked_name,
    use_derived,
    WRAPPER_REGISTRATION,
)
from .gen_trace_type import (
    get_return_value,
    MANUAL_AUTOGRAD_AND_TRACER,
    MANUAL_BACKEND,
    tie_return_values,
    type_wrapper_name,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# We don't set or modify grad_fn on these methods. Generally, they return
# tensors that have requires_grad=False. In-place functions listed here will
# not examine or modify requires_grad or grad_fn.
# NB: this does NOT include overload name
DONT_REQUIRE_DERIVATIVE = {
    # These only depend on the input Tensor's shape and device, not the data
    "empty_like",
    "ones_like",
    "full_like",
    "zeros_like",
    "rand_like",
    "randn_like",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_zeros",
    "new_ones",
    # These are only implemented on integral types
    "__and__",
    "__iand__",
    "__ilshift__",
    "__ior__",
    "__irshift__",
    "__ixor__",
    "__lshift__",
    "__or__",
    "__rshift__",
    "__xor__",
    # These work on integral data types, and hence don't require derivative
    "_sobol_engine_draw",
    "_sobol_engine_ff",
    "_sobol_engine_scramble_",
    "_sobol_engine_initialize_state_",
    # This is an unsafe method that is meant to be out of reach of autograd.
    "_coalesced_",
    # Quantize functions should not record gradients
    "quantize_per_tensor",
    "quantize_per_channel",
    # Functions that return integers should not have output that require gradients
    "argmax",
    "argmin",
    "argsort",
    "searchsorted",
    "bucketize",
    # Functions that return booleans are not differentiable
    "isnan",
    "isposinf",
    "isneginf",
    "isinf",
    "signbit",
    "isin",
    "allclose",
    # Functions return none are not differentiable
    "record_stream",
    # These functions are not differentiable
    "logical_and",
    "logical_xor",
    "logical_not",
    "logical_or",
    # This function returns nested_tensor shape as a tensor that is non-differentiable
    "_nested_tensor_size",
    "_nested_tensor_strides",
    "_nested_tensor_storage_offsets",
}

# The C -> R functions at the time of adding this are still being audited and tested
# but will not error out.
# C -> C, R -> C functions for which backward is correctly implemented and tested
GRADIENT_IMPLEMENTED_FOR_COMPLEX = {
    "fill",
    "t",
    "t_copy",
    "view",
    "reshape",
    "reshape_as",
    "view_as",
    "view_copy",
    "roll",
    "clone",
    "block_diag",
    "diag_embed",
    "repeat",
    "expand",
    "expand_copy",
    "flip",
    "fliplr",
    "flipud",
    "rot90",
    "nanmean",
    "nansum",
    "transpose",
    "transpose_copy",
    "permute",
    "permute_copy",
    "squeeze",
    "squeeze_copy",
    "unsqueeze",
    "unsqueeze_copy",
    "resize",
    "resize_as",
    "tril",
    "triu",
    "chunk",
    "zero_",
    "eq_",
    "ne_",
    "add",
    "__radd__",
    "sum",
    "_conj",
    "sin",
    "cos",
    "mul",
    "sinc",
    "sinh",
    "cosh",
    "__rmul__",
    "sgn",
    "asin",
    "acos",
    "sub",
    "div",
    "cat",
    "view_as_complex",
    "index_put",
    "neg",
    "complex",
    "select",
    "where",
    "as_strided",
    "as_strided_copy",
    "as_strided_scatter",
    "slice",
    "constant_pad_nd",
    "unbind",
    "unbind_copy",
    "split",
    "split_with_sizes",
    "unsafe_split",
    "split_with_sizes_backward",
    "dot",
    "vdot",
    "cholesky",
    "triangular_solve",
    "mm",
    "_unsafe_view",
    "mv",
    "outer",
    "bmm",
    "diagonal",
    "alias",
    "atan",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logsumexp",
    "logcumsumexp",
    "reciprocal",
    "tan",
    "pow",
    "rsqrt",
    "tanh",
    "tanh_backward",
    "asinh",
    "acosh",
    "atanh",
    "take",
    "fill_",
    "exp",
    "exp2",
    "expm1",
    "nonzero",
    "mean",
    "std_mean",
    "var_mean",
    "inverse",
    "solve",
    "linalg_cholesky",
    "addcmul",
    "addcdiv",
    "matrix_exp",
    "linalg_matrix_exp",
    "_linalg_eigh",
    "cholesky_solve",
    "linalg_qr",
    "_linalg_svd",
    "_fft_c2c",
    "_fft_r2c",
    "linalg_solve",
    "sqrt",
    "stack",
    "gather",
    "index_select",
    "index_add_",
    "linalg_inv",
    "linalg_inv_ex",
    "baddbmm",
    "addbmm",
    "addmm",
    "addmv",
    "addr",
    "linalg_householder_product",
    "ormqr",
    "reflection_pad1d",
    "reflection_pad2d",
    "reflection_pad3d",
    "linalg_cholesky_ex",
    "linalg_eig",
    "diagonal_copy",
    "diagonal_scatter",
    "alias_copy",
    "select_backward",
    "diagonal_backward",
    "slice_backward",
    "reflection_pad1d_backward",
    "reflection_pad2d_backward",
    "reflection_pad3d_backward",
    "_sparse_sparse_matmul",
    "replication_pad1d",
    "replication_pad2d",
    "replication_pad3d",
    "put",
    "put_",
    "_to_copy",
    "replication_pad1d_backward",
    "replication_pad2d_backward",
    "replication_pad3d_backward",
    "diag",
    "masked_scatter",
    "masked_select",
    "index_add",
    "index_fill",
    "trace",
    "polar",
    "cumsum",
    "rsub",
    "eig",
    "lerp",
    "linalg_vector_norm",
    "cumprod",
    "prod",
    "index_copy",
    "lu",
    "unfold",
    "unfold_backward",
    "index",
    "masked_fill",
    "masked_scatter_backward",
    "linalg_cross",
    "lu_unpack",
    "renorm",
    "_conj_physical",
    "linalg_lu_factor_ex",
    "scatter",
    "scatter_add",
    "sigmoid",
    "sigmoid_backward",
    "sparse_mask",
    "trapezoid",
    "cumulative_trapezoid",
    "conj_physical_",
    "_neg_view",
    "_reshape_alias",
    "_reshape_copy",
    "_linalg_det",
    "lu_solve",
    "linalg_solve_triangular",
    "linalg_pinv",
    "linalg_lstsq",
    "unfold_copy",
    "col2im",
    "im2col",
    "cholesky_inverse",
    "to_sparse",
    "sparse_sampled_addmm",
    "linalg_lu",
    "pixel_shuffle",
    "pixel_unshuffle",
    "channel_shuffle",
    "linalg_lu_solve",
    "_linalg_slogdet",
    "_linalg_solve_ex",
    "_unsafe_index",
    "_unsafe_index_put",
    "_unsafe_masked_index",
    "_unsafe_masked_index_put_accumulate",
}

GRADIENT_IMPLEMENTED_FOR_SPARSE_COMPLEX = {
    "_to_dense",
    "_coalesce",
    "coalesce",
    "values",
    "_sparse_coo_tensor_with_dims_and_tensors",
    "_sparse_addmm",
}

GRADIENT_IMPLEMENTED_FOR_COMPLEX.update(GRADIENT_IMPLEMENTED_FOR_SPARSE_COMPLEX)

# Some operators invalidate the grad_accumulator. Let's reset it.
RESET_GRAD_ACCUMULATOR = {"set_", "resize_"}

# NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
#
# We check the following properties:
#   1) A function should never change the input tensors' underlying c10::TensorImpl
#      pointers or c10::Storage pointers, even if it modifies its input tensors (via
#      inplace or out-variants)
# If the function does not modify its arguments, we also check the following properties
# pertaining to its output:
#   2) Its TensorImpl has use_count of 1
#   3) If the function is a view function, it has the same StorageImpl as that of
#      the input it is aliased with. Otherwise, its StorageImpl has use_count of 1
#
# The following code templates implement the checks for this invariant:
SAVE_TENSOR_STORAGE = CodeTemplate(
    """\
auto ${tensor_name}_storage_saved =
  ${tensor_name}.has_storage() ? ::std::optional<Storage>(${tensor_name}.storage()) : ::std::nullopt;
"""
)


# If tensor_name == out_tensor_name, used to enforce (1), otherwise used for (2)
ENFORCE_SAME_TENSOR_STORAGE = CodeTemplate(
    """\
if (${tensor_name}_storage_saved.has_value() &&
    !at::impl::dispatch_mode_enabled() &&
    !at::impl::tensor_has_dispatch(${tensor_name}) &&
    !at::impl::tensor_has_dispatch(${out_tensor_name}))
  TORCH_INTERNAL_ASSERT(${tensor_name}_storage_saved.value().is_alias_of(${out_tensor_name}.storage()));
"""
)

SAVE_TENSORLIST_STORAGE = CodeTemplate(
    """\
std::vector<::std::optional<Storage>> ${tensorlist_name}_storage_saved(${tensorlist_name}.size());
for (const Tensor& tensor : ${tensorlist_name})
  ${tensorlist_name}_storage_saved.push_back(
    tensor.has_storage() ? ::std::optional<Storage>(tensor.storage()) : ::std::nullopt);
"""
)

ENFORCE_SAME_TENSORLIST_STORAGE = CodeTemplate(
    """\
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
  if (${tensorlist_name}_storage_saved[i].has_value() && !at::impl::tensorlist_has_dispatch(${tensorlist_name}))
    TORCH_INTERNAL_ASSERT(${tensorlist_name}_storage_saved[i].value().is_alias_of(${tensorlist_name}[i].storage()));
}
"""
)

SAVE_OPTIONALTENSORLIST_STORAGE = CodeTemplate(
    """\
std::vector<::std::optional<Storage>> ${tensorlist_name}_storage_saved(${tensorlist_name}.size());
for (const ::std::optional<Tensor>& tensor : ${tensorlist_name})
  ${tensorlist_name}_storage_saved.push_back(
    tensor.has_value() && tensor->has_storage() ? ::std::optional<Storage>(tensor->storage()) : ::std::nullopt);
"""
)

ENFORCE_SAME_OPTIONALTENSORLIST_STORAGE = CodeTemplate(
    """\
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
  if (${tensorlist_name}_storage_saved[i].has_value() && !at::impl::tensorlist_has_dispatch(${tensorlist_name}))
    TORCH_INTERNAL_ASSERT(${tensorlist_name}_storage_saved[i].value().is_alias_of(
        static_cast<::std::optional<Tensor>>(${tensorlist_name}[i])->storage()));
}
"""
)

SAVE_TENSOR_IMPL = CodeTemplate(
    """\
c10::intrusive_ptr<TensorImpl> ${tensor_name}_impl_saved;
if (${tensor_name}.defined()) ${tensor_name}_impl_saved = ${tensor_name}.getIntrusivePtr();
"""
)

ENFORCE_SAME_TENSOR_IMPL = CodeTemplate(
    """\
if (${tensor_name}_impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(${tensor_name}))
  TORCH_INTERNAL_ASSERT(${tensor_name}_impl_saved == ${tensor_name}.getIntrusivePtr());
"""
)

ENFORCE_TENSOR_IMPL_USE_COUNT_LT_OR_EQ_ONE = CodeTemplate(
    """\
if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(${tensor_name}))
  TORCH_INTERNAL_ASSERT(${tensor_name}.use_count() <= 1, "function: ${fn_name}");
"""
)

ENFORCE_TENSOR_STORAGE_USE_COUNT_EQUALS_ONE = CodeTemplate(
    """\
if (${tensor_name}.has_storage() && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(${tensor_name})) {
  TORCH_INTERNAL_ASSERT(${tensor_name}.storage().use_count() == 1, "function: ${fn_name}");
}
"""
)

SAVE_TENSORLIST_IMPL = CodeTemplate(
    """\
std::vector<c10::intrusive_ptr<TensorImpl>> ${tensorlist_name}_impl_saved(${tensorlist_name}.size());
for (size_t i=0; i<${tensorlist_name}.size(); i++)
  if (${tensorlist_name}[i].defined()) ${tensorlist_name}_impl_saved[i] = ${tensorlist_name}[i].getIntrusivePtr();
"""
)

ENFORCE_SAME_TENSORLIST_IMPL = CodeTemplate(
    """\
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
  if (${tensorlist_name}_impl_saved[i] && !at::impl::tensorlist_has_dispatch(${tensorlist_name}))
    TORCH_INTERNAL_ASSERT(${tensorlist_name}_impl_saved[i] == ${tensorlist_name}[i].getIntrusivePtr());
}
"""
)

SAVE_OPTIONALTENSORLIST_IMPL = CodeTemplate(
    """\
std::vector<c10::intrusive_ptr<TensorImpl>> ${tensorlist_name}_impl_saved(${tensorlist_name}.size());
for (size_t i=0; i<${tensorlist_name}.size(); i++) {
  ::std::optional<Tensor> t = ${tensorlist_name}[i];
  if (t.has_value() && t->defined()) ${tensorlist_name}_impl_saved[i] = t->getIntrusivePtr();
}
"""
)

ENFORCE_SAME_OPTIONALTENSORLIST_IMPL = CodeTemplate(
    """\
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
  if (${tensorlist_name}_impl_saved[i])
    TORCH_INTERNAL_ASSERT(
      ${tensorlist_name}_impl_saved[i] == static_cast<::std::optional<Tensor>>(${tensorlist_name}[i])->getIntrusivePtr());
}
"""
)

# The following list contains functions that we don't enforce the invariant on.
DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE = {
    # These functions are expected to change impl or storage of input tensors
    "set_",
    "_cudnn_rnn_flatten_weight",
    "_unsafe_masked_index",
    "_unsafe_masked_index_put_accumulate",
}
DONT_ENFORCE_TENSOR_IMPL_USE_COUNT = {
    # These non-inplace, non-out functions return tensors with use_count > 1
    # Therefore, they MAY (but not necessarily) return one of its inputs as-is
    # See https://github.com/pytorch/pytorch/issues/60426 for more information
    "_embedding_bag",
    "_embedding_bag_forward_only",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "lu_unpack",
    "_cudnn_rnn_backward",
    # The below failed StorageImpl use_count check but we skip tensor_impl check
    # just in case
    "_cudnn_rnn",
    "dequantize_self",
    # lift() should never actually be called with a requires_grad=True tensor,
    "lift",
    "lift_fresh",
    "lift_fresh_copy",
    # Nested Tensors related functions
    # _nested_tensor_size() should never actually be called with requires_grad=True tensor
    "_nested_tensor_size",
    "_nested_tensor_strides",
    "_nested_tensor_storage_offsets",
}

DONT_ENFORCE_STORAGE_IMPL_USE_COUNT = {
    # These non-view functions return tensors with storage use_count != 1
    "_slow_conv2d_forward",
    "slow_conv3d_forward",
    "channel_shuffle",
    # If an input is returned as-is in output, we cannot guarantee its storage_impl
    # use count to be 1 either.
    *DONT_ENFORCE_TENSOR_IMPL_USE_COUNT,
}
# END CHECKS FOR [ TensorImpl and Storage Pointer Sanity Checks ]

DECLARE_GRAD_FN = CodeTemplate(
    """\
std::shared_ptr<${op}> grad_fn;
"""
)

DECLARE_VECTOR_OF_GRAD_FN = CodeTemplate(
    """\
std::vector<std::shared_ptr<${op}>> grad_fns;
"""
)

SETUP_ANY_REQUIRES_GRAD = CodeTemplate(
    """\
[[maybe_unused]] auto _any_requires_grad = compute_requires_grad( ${args_with_derivatives} );
${extra_differentiability_conditions}
"""
)

SETUP_DERIVATIVE = CodeTemplate(
    """\
if (_any_requires_grad) {
  ${setup}
}
"""
)

SETUP_NONE_REQUIRES_GRAD = CodeTemplate(
    """\
if (compute_requires_grad( ${args_to_check} )) {
  throw_error_out_requires_grad("${base_name}");
}
"""
)

ASSIGN_GRAD_FN = CodeTemplate(
    """\
grad_fn = std::shared_ptr<${op}>(new ${op}(${op_ctor}), deleteNode);
grad_fn->set_next_edges(collect_next_edges( ${args_with_derivatives} ));
"""
)

# note(crcrpar): `compute_requires_grad` in the template below is supplied with arguments indexed with `i`
# while the `SETUP_ANY_REQUIRES_GRAD` above takes whole tensors and scalars.
ASSIGN_VECTOR_OF_GRAD_FN = CodeTemplate(
    """\
for (const auto& i : c10::irange( ${irange} )) {
  const auto ith_requires_grad = compute_requires_grad(${args_with_derivatives});
  check_inplace(self[i], ith_requires_grad);
  grad_fns.push_back([&]() -> std::shared_ptr<${op}> {
      if (!ith_requires_grad) {
          return nullptr;
      } else {
          auto grad_fn = std::shared_ptr<${op}>(new ${op}(${op_ctor}), deleteNode);
          grad_fn->set_next_edges(collect_next_edges( ${args_with_derivatives} ));
          return grad_fn;
      }
  }());
}
"""
)

CALL_REDISPATCH = CodeTemplate(
    """\
at::redispatch::${api_name}(${unpacked_args})"""
)
# If the non-variable operation has return values, we use the `tmp` variable to hold the
# values temporarily and pass the values to the return variables outside of the
# `at::AutoDispatchBelowAutograd` guard block.
DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES_JVP_DECOMP = CodeTemplate(
    """\
auto ${tmp_var} = ([&]() {
  if (${any_has_forward_grad}) {
    static c10::OperatorName full_name("aten::${op_name}", "${op_overload}");
    static ::std::optional<c10::OperatorHandle> opt_op = c10::Dispatcher::singleton().findSchema(full_name);
    return impl::run_jit_decomposition_with_args_for_jvp<${return_types}>("${op_name}", *opt_op, ks, ${arg_names});
  } else {
    ${guard}
    return ${base_type_call};
  }
})();
"""
)

DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES = CodeTemplate(
    """\
auto ${tmp_var} = ([&]() {
  ${guard}
  return ${base_type_call};
})();
"""
)

DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES = CodeTemplate(
    """\
{
  ${guard}
  ${base_type_call};
}
"""
)

SET_HISTORY = CodeTemplate(
    """\
if (grad_fn) {
    ${fn}_history(${differentiable_outputs}, grad_fn);
}
"""
)

LOOP_OVER_VECTOR_OF_GRAD_FNS = CodeTemplate(
    """\
if (!grad_fns.empty()) {
    ${preamble}
    for (const auto& i : c10::irange(grad_fns.size())) {
        auto grad_fn = grad_fns[i];
        if (grad_fn != nullptr) {
            ${statements}
        }
    }
}
"""
)

CONDITIONAL = CodeTemplate(
    """\
if (${cond}) {
  ${statements}
}
"""
)

RUN_ONLY_IN_DEBUG_MODE = CodeTemplate(
    """\
#ifndef NDEBUG
${statements}
#endif
"""
)

FW_DERIVATIVE_CHECK_TEMPLATE = CodeTemplate(
    """\
isFwGradDefined(${req_inp})\
"""
)
FW_DERIVATIVE_SIZE_CHECK_TEMPLATE = CodeTemplate(
    """\
TORCH_CHECK(
    self.size() == ${inp_name}.size(),
      "Tensor lists must have the same number of tensors, got ",
    self.size(),
      " and ",
    ${inp_name}.size());
"""
)

FW_DERIVATIVE_TENSORLIST_CHECK_TEMPLATE = CodeTemplate(
    """\
isFwGradDefinedTensorList(${req_inp})\
"""
)

FW_DERIVATIVE_DEFINED_GRAD_TEMPLATE = CodeTemplate(
    """\
auto ${inp_name}_t_raw = toNonOptFwGrad(${inp});
auto ${inp_name}_tensor = toNonOptTensor(${inp});
auto ${inp_name}_t = (${inp_name}_t_raw.defined() || !${inp_name}_tensor.defined())
  ? ${inp_name}_t_raw : at::${zeros_fn}(${inp_name}_tensor.sym_sizes(), ${inp_name}_tensor.options());
"""
)

FW_DERIVATIVE_UPDATE_WRAPPED_NUM_TEMPLATE = CodeTemplate(
    """\
update_wrapped_number(${inp_name}_tensor, ${inp_name}_t);
"""
)

FW_DERIVATIVE_DEFINED_PRIMAL_TEMPLATE = CodeTemplate(
    """\
auto ${inp_name}_p = toNonOptPrimal(${inp});
"""
)

FW_DERIVATIVE_SETTER_TENSOR = CodeTemplate(
    """\
if (${out_arg}_new_fw_grad_opt.has_value() && ${out_arg}_new_fw_grad_opt.value().defined() && ${out_arg}.defined()) {
  // The hardcoded 0 here will need to be updated once we support multiple levels.
  ${out_arg}._set_fw_grad(${out_arg}_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ ${is_inplace});
}
"""
)

FW_DERIVATIVE_SETTER_TENSOR_FOREACH = CodeTemplate(
    """\
for (const auto& i : c10::irange(${out_arg}_new_fw_grad_opts.size())) {
  auto& ${out_arg}_new_fw_grad_opt = ${out_arg}_new_fw_grad_opts[i];
  if (${out_arg}_new_fw_grad_opt.has_value() && ${out_arg}_new_fw_grad_opt.value().defined() && ${out_arg}[i].defined()) {
    // The hardcoded 0 here will need to be updated once we support multiple levels.
    ${out_arg}[i]._set_fw_grad(${out_arg}_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ ${is_inplace});
  }
}
"""
)

FW_DERIVATIVE_SETTER_MULTI_OUTPUT = CodeTemplate(
    """\
if (${all_res}_new_fw_grad_opt.has_value() && std::get<${idx}>(${all_res}_new_fw_grad_opt.value()).defined()
    && ${out_arg}.defined()) {
  ${out_arg}._set_fw_grad(std::get<${idx}>(${all_res}_new_fw_grad_opt.value()), /* level */ 0, /* is_inplace_op */ false);
}
"""
)

FW_DERIVATIVE_SETTER_TENSOR_LIST = CodeTemplate(
    """\
if (${out_arg}_new_fw_grad_opt.has_value()) {
  auto ${out_arg}_new_fw_grad = ${out_arg}_new_fw_grad_opt.value();
  TORCH_INTERNAL_ASSERT(${out_arg}.size() == ${out_arg}_new_fw_grad.size());
  for (const auto i : c10::irange(${out_arg}.size())) {
    if (${out_arg}_new_fw_grad[i].defined() && ${out_arg}[i].defined()) {
      // The hardcoded 0 here will need to be updated once we support multiple levels.
      ${out_arg}[i]._set_fw_grad(${out_arg}_new_fw_grad[i], /* level */ 0, /* is_inplace_op */ ${is_inplace});
    }
  }
}
"""
)

FW_DERIVATIVE_TEMPLATE = CodeTemplate(
    """\
${fw_grad_opt_definition}
if (${requires_fw_grad}) {
    ${unpacked_arguments}
    ${out_arg}_new_fw_grad_opt = ${formula};
}
"""
)

FW_DERIVATIVE_FOREACH_TEMPLATE = CodeTemplate(
    """\
${fw_grad_opt_definition}
for (const auto& i : c10::irange(${vector_of_optional_tensor}.size())) {
  if (${any_has_forward_grad_for_current_index}) {
      ${unpacked_arguments}
      ${vector_of_optional_tensor}[i] = ${formula};
  }
}
"""
)

FW_DERIVATIVE_FORBID_TEMPLATE = CodeTemplate(
    """\
TORCH_CHECK_NOT_IMPLEMENTED(!(${cond}), "Trying to use forward AD with ${name} that does not support it ${msg}");
"""
)

FW_DERIVATIVE_FORBID_LIST_TEMPLATE = CodeTemplate(
    """\
for (const auto& _t: ${arg}) {
    TORCH_CHECK_NOT_IMPLEMENTED(!(${cond}), "Trying to use forward AD with ${name} that does not support it ${msg}");
}
"""
)


def gen_variable_type(
    out: str,
    native_yaml_path: str,
    tags_yaml_path: str,
    fns_with_diff_infos: list[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
    used_keys: set[str],
) -> None:
    """VariableType.h and VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    """
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write(
        "VariableType.h",
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/VariableType.h"
        },
    )

    # helper that generates a TORCH_LIBRARY_IMPL macro for each
    # dispatch key that appears in derivatives.yaml
    def wrapper_registrations(used_keys: set[str]) -> str:
        library_impl_macro_list: list[str] = []
        for key in sorted(used_keys):
            dispatch_key = key
            if key == "Default":
                dispatch_key = "Autograd"
            library_impl_macro = (
                f"TORCH_LIBRARY_IMPL(aten, {dispatch_key}, m) "
                + "{\n"
                + "${"
                + f"wrapper_registrations_{key}"
                + "}\n}"
            )
            library_impl_macro_list += [library_impl_macro]
        return "\n\n".join(library_impl_macro_list)

    # Generate a new template from VariableType.cpp which replaces ${wrapper_registrations}
    # with per key TORCH_LIBRARY_IMPL macros for each key that appears in derivatives.yaml
    fm1 = FileManager(
        install_dir=out + "/templates", template_dir=template_path, dry_run=False
    )
    fm1.write(
        "VariableType.cpp",
        lambda: {
            "type_derived_method_definitions": "\n\n".join(
                [
                    "${" + f"type_derived_method_definitions_{key}" + "}"
                    for key in sorted(used_keys)
                ]
            ),
            "wrapper_registrations": wrapper_registrations(used_keys),
        },
    )

    # Generate final VariableType_*.cpp files from the generated template
    fm2 = FileManager(install_dir=out, template_dir=out + "/templates", dry_run=False)

    sharded_keys = set(
        [f"type_derived_method_definitions_{key}" for key in sorted(used_keys)]
        + [f"wrapper_registrations_{key}" for key in sorted(used_keys)]
    )
    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.
    fm2.write_sharded(
        "VariableType.cpp",
        [fn for fn in fns_with_diff_infos if use_derived(fn)],
        key_fn=lambda fn: cpp.name(fn.func.func),
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/VariableType.cpp",
        },
        env_callable=gen_variable_type_func,
        num_shards=5,
        sharded_keys=sharded_keys,
    )


@with_native_function_and
def gen_wrapper_registration(f: NativeFunction, key: str = "Default") -> str:
    return WRAPPER_REGISTRATION.substitute(
        unqual_operator_name_with_overload=f.func.name,
        type_wrapper_name=type_wrapper_name(f, key),
        class_type="VariableType",
    )


def gen_variable_type_func(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> dict[str, list[str]]:
    f = fn.func
    result = {}
    with native_function_manager(f):
        name = cpp.name(f.func)
        formals = gen_formals(f)

        if (
            fn.info is None
            and str(f.func.name.name) not in RESET_GRAD_ACCUMULATOR
            and get_base_name(f) not in DONT_REQUIRE_DERIVATIVE
            and len(gen_differentiable_outputs(fn)) > 0
            and cpp.name(f.func) not in DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE
            and type_wrapper_name(f) not in DONT_ENFORCE_STORAGE_IMPL_USE_COUNT
            and type_wrapper_name(f) not in DONT_ENFORCE_TENSOR_IMPL_USE_COUNT
        ):
            # NOTE: [ Registering AutogradNotImplemented boxed kernel ]
            #
            # When there is no derivatives.yaml entry, we register a generic boxed
            # NotImplemented kernel to set grad_fn to be NotImplemented, so that forward
            # proceeds as usual but an error is properly produced on backward.
            # TODO: it would be nice to not have these special cases
            #
            # There are several cases where still let codegen handle it:
            # 1) ops that need to reset grad accumulator (we let codegen handle this case
            #     because) the list is (currently) only accessible in Python.
            # 2) User explicitly specifies DONT_REQUIRE_DERIVATIVE. This basically makes
            #    autograd a fallthrough with NDEBUG checks. This can be useful for when all
            #    outputs are integral.
            # 3) When there are no differentiable outputs. This is similar to (2).
            # 4) There are certain ops where we skip certain NDEBUG checks. this is similar
            #    to (1).
            type_definition = ""
            wrapper_registration = AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION.substitute(
                unqual_operator_name_with_overload=f.func.name
            )
            result["type_derived_method_definitions_Default"] = [type_definition]
            result["wrapper_registrations_Default"] = [wrapper_registration]
        else:
            if not fn.info:
                key = "Default"
                type_definition = METHOD_DEFINITION.substitute(
                    return_type=cpp.returns_type(
                        f.func.returns, symint=True
                    ).cpp_type(),
                    type_wrapper_name=type_wrapper_name(f, key),
                    type_definition_body=emit_body(fn, key),
                    formals=formals,
                )
                wrapper_registration = gen_wrapper_registration(f, key)
                result[f"type_derived_method_definitions_{key}"] = [type_definition]
                result[f"wrapper_registrations_{key}"] = [wrapper_registration]
            else:
                for key in fn.info.keys():
                    type_definition = METHOD_DEFINITION.substitute(
                        return_type=cpp.returns_type(
                            f.func.returns, symint=True
                        ).cpp_type(),
                        type_wrapper_name=type_wrapper_name(f, key),
                        type_definition_body=emit_body(fn, key),
                        formals=formals,
                    )
                    wrapper_registration = gen_wrapper_registration(f, key)
                    result[f"type_derived_method_definitions_{key}"] = [type_definition]
                    result[f"wrapper_registrations_{key}"] = [wrapper_registration]
    # See Note [Manual Backend kernels]
    assert (name in MANUAL_BACKEND) == f.manual_kernel_registration
    # If you want to register a kernel to Autograd, you must make the op abstract.
    # In other words, this op must have dispatch section in native_functions.yaml.
    if name in MANUAL_AUTOGRAD_AND_TRACER or (
        fn.info and any(info.has_derivatives for info in fn.info.values())
    ):
        msg = (
            f"There's a formula for {name}(or its functional variant) in derivatives.yaml. "
            f"It's required to add a dispatch section for it with explicit supported backends e.g CPU/CUDA "
            f"or CompositeExplicitAutograd in native_functions.yaml. Please see "
            f"https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native#choosing-the-right-dispatch-keyword "
            f"for instructions to choose the right dispatch keyword."
        )
        assert f.is_abstract, msg

    return result


_foreach_ops_without_differentiability_info = {
    # No reference backward available due to the lack of `{maximum, minimum}(tensor, scalar)`.
    ("_foreach_maximum", "Scalar"),
    ("_foreach_maximum", "ScalarList"),
    ("_foreach_minimum", "Scalar"),
    ("_foreach_minimum", "ScalarList"),
    # No reference backward available as addcdiv/addcmul don't support Tensor as scaling factor.
    ("_foreach_addcdiv", "Tensor"),
    ("_foreach_addcmul", "Tensor"),
    ("_foreach_copy", ""),
}

_foreach_ops_with_different_arity = {
    # These ops lack `alpha` of scaling factor to applied to the right hand side argument.
    ("_foreach_add", "Scalar"),
    ("_foreach_add", "ScalarList"),
    ("_foreach_sub", "Scalar"),
    ("_foreach_sub", "ScalarList"),
}


@with_native_function_with_differentiability_info_and_key
def emit_body(
    fn: NativeFunctionWithDifferentiabilityInfo, key: str = "Default"
) -> list[str]:
    assert dispatch_strategy(fn) == "use_derived"
    f = fn.func
    info = fn.info[key] if fn.info else None
    fw_derivatives = fn.fw_derivatives.get(key, []) if fn.fw_derivatives else []

    name = cpp.name(f.func)
    inplace = f.func.kind() == SchemaKind.inplace
    is_out_fn = f.func.kind() == SchemaKind.out
    returns_void = len(f.func.returns) == 0
    base_name = get_base_name(f)
    view_info = get_view_info(f)

    is_foreach = name.startswith("_foreach")
    is_inplace_foreach = is_foreach and inplace
    if is_inplace_foreach:
        inplace_foreacharg2refarg: dict[Argument, Argument] = {}
        refargname2inplace_foreacharg: dict[str, Argument] = {}
        base_name_and_overload_name = (f.func.name.name.base, f.func.name.overload_name)
        if info is None:
            assert (
                base_name_and_overload_name
                in _foreach_ops_without_differentiability_info
            ), (
                f"{'.'.join(base_name_and_overload_name)} should have a differentiability info"
            )
        else:
            assert (
                len(f.func.arguments.flat_non_out)
                == len(info.func.func.arguments.flat_non_out)
            ) or (base_name_and_overload_name in _foreach_ops_with_different_arity), (
                f"{'.'.join(base_name_and_overload_name)} has {len(f.func.arguments.flat_non_out)} args "
                f"but the reference has {len(info.func.func.arguments.flat_non_out)}"
            )
            for foreach_arg, ref_arg in zip(
                f.func.arguments.flat_non_out, info.func.func.arguments.flat_non_out
            ):
                foreach_arg_type = foreach_arg.type
                if isinstance(foreach_arg_type, ListType):
                    foreach_arg_type = foreach_arg_type.elem
                assert foreach_arg_type == ref_arg.type
                inplace_foreacharg2refarg[foreach_arg] = ref_arg
                refargname2inplace_foreacharg[ref_arg.name] = foreach_arg

    def gen_differentiable_input(
        arg: Argument | SelfArgument | TensorOptionsArguments,
    ) -> DifferentiableInput | None:
        if isinstance(arg, TensorOptionsArguments):
            return None
        a: Argument = arg.argument if isinstance(arg, SelfArgument) else arg

        # TODO: `cpp_type` is only to keep it byte-for-byte compatible with the old codegen, should remove.
        # NB: This is not a clone of cpp.argument() - TensorOptionsArguments / faithful / binds are
        # not handled properly as they are irrelevant for this codegen.
        cpp_type = cpp.argument_type(a, binds=a.name, symint=True).cpp_type()

        if not is_differentiable(a.name, a.type, info):
            return None
        return DifferentiableInput(
            name=a.name,
            type=a.type,
            cpp_type=cpp_type,
        )

    @with_native_function
    def gen_differentiable_inputs(f: NativeFunction) -> list[DifferentiableInput]:
        arguments = list(f.func.arguments.non_out)
        if is_inplace_foreach and info is not None:
            for i, arg in enumerate(f.func.arguments.flat_non_out):
                if arg in inplace_foreacharg2refarg:
                    # note(crcrpar): From what I understand, what matters is only the name.
                    # Thus originally I only replace argument only when the names are different.
                    # TODO(crcrpar): Make it simpler.
                    mapped_arg = inplace_foreacharg2refarg[arg]
                    arguments[i] = Argument(
                        mapped_arg.name,
                        mapped_arg.type,
                        mapped_arg.default,
                        mapped_arg.annotation,
                    )
        return list(mapMaybe(gen_differentiable_input, arguments))

    def find_args_with_derivatives(
        differentiable_inputs: list[DifferentiableInput],
    ) -> list[DifferentiableInput]:
        """Find arguments that have derivative definitions"""
        if info is None or not info.has_derivatives:
            return differentiable_inputs
        names = {name for d in info.derivatives for name in d.var_names}
        differentiable = [arg for arg in differentiable_inputs if arg.name in names]
        if len(differentiable) != len(names):
            missing = names - {arg.name for arg in differentiable}
            raise RuntimeError(
                f"Missing arguments for derivatives: {missing} in {info.name}"
            )
        return differentiable

    differentiable_inputs = gen_differentiable_inputs(f)
    args_with_derivatives = find_args_with_derivatives(differentiable_inputs)
    differentiable_outputs = gen_differentiable_outputs(fn, key)

    undifferentiable = (base_name in DONT_REQUIRE_DERIVATIVE) or (
        name in DONT_REQUIRE_DERIVATIVE
    )

    requires_derivative = (
        (not undifferentiable)
        and (len(differentiable_inputs) > 0)
        and (
            (len(differentiable_outputs) > 0)
            # note(crcrpar): In-place foreach functions are a void function.
            or is_inplace_foreach
        )
    )

    if (
        info is not None
        and info.has_derivatives
        and not requires_derivative
        # out= ops are allowed to have zero returns which cause requires_derivative to be False
        # we shouldn't error out though (out= ops for autograd just redispatch)
        and len(f.func.returns) > 0
    ):
        raise RuntimeError(
            f"ERROR: derivative ignored for {name} -- specified an autograd function without derivative"
        )

    # note(crcrpar): In-place foreach functions do not support forward AD
    if requires_derivative and len(fw_derivatives) > 0 and not is_inplace_foreach:
        assert sum(len(derivative.var_names) for derivative in fw_derivatives) == len(
            differentiable_outputs
        ), (
            "Expected the number of forward derivatives implemented to match the "
            "number of differentiable outputs. NB: This only applies when at least "
            "one forward derivative is implemented. Not implementing any forward "
            "derivatives is also okay, and we would require inputs to the op to "
            "not have associated tangents in that case."
        )

    try_jit_decomposition = (
        requires_derivative
        and len(fw_derivatives) == 0
        and (not modifies_arguments(f))
        and (not returns_void)
    )

    def emit_save_inputs() -> list[str]:
        setup: list[str] = []
        if info is None or not info.has_derivatives:
            return setup

        has_tensorlist_arg = any(
            is_tensor_list_type(arg.type) for arg in args_with_derivatives
        )

        # We don't want to save tensors if we know that they will never be used
        # when computing the derivative, so we add guards to those statements
        def guard_for(arg: SavedAttribute) -> str | None:
            assert info is not None

            # It's hard to determine the edge offset if we have TensorLists
            # NOTE(crcrpar): in-place foreach functions' arguments include tensorlist
            # but their derivatives don't use it, so let them bypass this check.
            if has_tensorlist_arg and (not is_inplace_foreach):
                return None

            # Empirical evaluation of the cases where we insert those guards in
            # backward show that they are somewhat useless. E.g. there's no need
            # to guard on some values captured from forward, because they had to
            # require_grad if the backward function even gets executed. I don't
            # have any good ideas for detecting those cases, so I simply disabled the
            # checks.
            if "backward" in info.name:
                return None

            # If there's a single derivative we could compute, we already have
            # a requires_grad check that is sufficient
            if len(args_with_derivatives) <= 1:
                return None

            # We really only care about trimming down the amount of tensors we save
            if arg.nctype.type != BaseCType(tensorT):
                return None

            # We want to emit simple guards, so we only allow that if checking one
            # input is enough to determine whether we need that value
            used_in = [d for d in info.derivatives if arg in d.saved_inputs]
            assert len(used_in) > 0
            if len(used_in) != 1:
                return None
            derivative = used_in[0]

            # Case with multioutput formulas
            # TODO: process all derivative formulas!!!
            if len(derivative.var_names) != 1:
                wrap_opt_if_start = derivative.formula.find(
                    f"wrap_opt_if({arg.nctype.name}"
                )
                if wrap_opt_if_start == -1:
                    return None

                wrap_opt_if_match = re.match(
                    rf"wrap_opt_if\({arg.nctype.name},(.*?)\)",
                    derivative.formula[wrap_opt_if_start:],
                )
                assert wrap_opt_if_match is not None

                # Condition is between 'wrap_opt_if(var_name,' and ')'.
                condition_slice = slice(len(rf"wrap_opt_if\({arg.nctype.name},"), -1)
                wrap_opt_if_condition = wrap_opt_if_match.group(0)[
                    condition_slice
                ].strip()
                # replace 'grad_input_mask[num]' with 'grad_fn->should_compute_output(num)'
                wrap_opt_if_condition = re.sub(
                    r"grad_input_mask\[(\d+)\]",
                    r"grad_fn->should_compute_output(\1)",
                    wrap_opt_if_condition,
                )
                return f"{wrap_opt_if_condition}"

            # Figure out the offset of the edge that uses this variable
            derivative_var_name = derivative.var_names[0]
            for edge_off, a in enumerate(args_with_derivatives):
                if a.name == derivative_var_name:
                    break
            else:
                raise AssertionError
            return f"grad_fn->should_compute_output({edge_off})"

        if is_inplace_foreach:
            save_input_stmts = save_variables(info.all_saved_inputs, False, guard_for)
            if save_input_stmts:
                setup.append(
                    LOOP_OVER_VECTOR_OF_GRAD_FNS.substitute(
                        preamble="", statements=save_input_stmts
                    )
                )
        else:
            setup.extend(save_variables(info.all_saved_inputs, False, guard_for))
            for arg in args_with_derivatives:
                if is_tensor_list_type(arg.type):
                    setup.append(f"grad_fn->{arg.name}_size_ = {arg.name}.size();")
        return setup

    def setup_derivative(differentiable_inputs: list[DifferentiableInput]) -> list[str]:
        body: list[str] = []
        if is_out_fn:
            # For out functions, ensure that no input or output requires grad
            body.append(DECLARE_GRAD_FN.substitute(op="Node"))
            body.append(
                SETUP_NONE_REQUIRES_GRAD.substitute(
                    base_name=base_name,
                    args_to_check=[arg.name for arg in differentiable_inputs],
                )
            )
            body.append(
                SETUP_NONE_REQUIRES_GRAD.substitute(
                    base_name=base_name,
                    args_to_check=[arg.name for arg in differentiable_outputs],
                )
            )
            return body

        op = info.op if info is not None and info.has_derivatives else "NotImplemented"
        setup = []
        if not is_inplace_foreach:
            setup.extend(
                ASSIGN_GRAD_FN.substitute(
                    op=op,
                    op_ctor=""
                    if info is not None and info.has_derivatives
                    else f'"{cpp.name(f.func)}"',
                    args_with_derivatives=[arg.name for arg in args_with_derivatives],
                ).split("\n")
            )
        else:
            # note(crcrpar): Assuming in-place foreach function's self_arg is always TensorList.
            list_like_arg = "self"
            args = [arg.name for arg in args_with_derivatives]
            for i, arg in enumerate(args):
                if is_inplace_foreach and info is not None:
                    if arg in refargname2inplace_foreacharg:
                        foreach_arg = refargname2inplace_foreacharg[arg]
                        args[i] = foreach_arg.name + (
                            "[i]" if isinstance(foreach_arg.type, ListType) else ""
                        )
                else:
                    if arg == list_like_arg:
                        args[i] = arg + "[i]"
            setup.extend(
                ASSIGN_VECTOR_OF_GRAD_FN.substitute(
                    op=op,
                    op_ctor=""
                    if info is not None and info.has_derivatives
                    else f'"{cpp.name(f.func)}"',
                    args_with_derivatives=args,
                    irange=f"{list_like_arg}.size()",
                ).split("\n")
            )
        setup.extend(emit_save_inputs())

        body.extend(
            emit_check_no_requires_grad(differentiable_inputs, args_with_derivatives)
        )
        declare_grad_fn_template = (
            DECLARE_GRAD_FN if not is_inplace_foreach else DECLARE_VECTOR_OF_GRAD_FN
        )
        body.append(declare_grad_fn_template.substitute(op=op))
        body.append(SETUP_DERIVATIVE.substitute(setup=setup))
        return body

    def emit_check_if_in_complex_autograd_allowlist() -> list[str]:
        body: list[str] = []
        if base_name in GRADIENT_IMPLEMENTED_FOR_COMPLEX:
            return body
        for arg in differentiable_outputs:
            name = arg.name
            # TODO: should be `arg.type.is_tensor_like()`?
            if arg.cpp_type == "at::Tensor" or arg.cpp_type in TENSOR_LIST_LIKE_CTYPES:
                body.append(f'throw_error_for_complex_autograd({name}, "{base_name}");')
        return body

    def emit_check_no_requires_grad(
        tensor_args: list[DifferentiableInput],
        args_with_derivatives: list[DifferentiableInput],
    ) -> list[str]:
        """Checks that arguments without derivatives don't require grad"""
        body: list[str] = []
        for arg in tensor_args:
            if arg in args_with_derivatives:
                continue
            arg_name = arg.name
            if info and arg_name in info.non_differentiable_arg_names:
                continue
            if arg_name == "output":
                # Double-backwards definitions sometimes take in 'input' and
                # 'output', but only define the derivative for input.
                continue
            body.append(f'check_no_requires_grad({arg_name}, "{arg_name}", "{name}");')
        return body

    def emit_original_self_definition() -> list[str]:
        body: list[str] = []
        if inplace:
            if is_inplace_foreach:
                body.append(
                    "std::vector<::std::optional<at::Tensor>> original_selfs(self.size());"
                )
            else:
                body.append("::std::optional<at::Tensor> original_self;")

            all_forward_grad_cond = []
            for derivative in fw_derivatives:
                if derivative.required_original_self_value:
                    all_forward_grad_cond.append(
                        get_any_has_forward_grad_name(derivative.var_names)
                    )

            if all_forward_grad_cond:
                if not is_inplace_foreach:
                    body.append(f"if ({' || '.join(all_forward_grad_cond)}) {{")
                    body.append("  original_self = self.clone();")
                    body.append("}")
                else:
                    current_all_forward_grad_cond = [
                        f"{cond}[i]" for cond in all_forward_grad_cond
                    ]
                    body.append("for (const auto& i : c10::irange(self.size())) {")
                    body.append(
                        f"  if ({' || '.join(current_all_forward_grad_cond)}) {{"
                    )
                    body.append("    original_selfs[i] = self[i].clone();")
                    body.append("  }")
                    body.append("}")

        return body

    def save_variables(
        saved_variables: Sequence[SavedAttribute],
        is_output: bool,
        guard_for: Callable[[SavedAttribute], str | None] = lambda name: None,
    ) -> Sequence[str]:
        # assign the saved variables to the generated grad_fn
        stmts: list[str] = []
        for arg in sorted(saved_variables, key=lambda sa: str(sa.nctype.name)):
            name = (
                arg.nctype.name.name
                if isinstance(arg.nctype.name, SpecialArgName)
                else arg.nctype.name
            )
            foreacharg: Argument | None = None
            is_foreacharg_list_type: bool = False
            type = arg.nctype.type
            expr = arg.expr
            stmts_prepend = None
            if is_inplace_foreach and info is not None:
                # todo(crcrpar): See if we can add some check e.g. `assert foreacharg is not None`.
                # for now the example assert would fail.
                name_to_query = name.split("_scalar_type")[0]
                if name_to_query in refargname2inplace_foreacharg:
                    foreacharg = refargname2inplace_foreacharg[name_to_query]
                    is_foreacharg_list_type = isinstance(foreacharg.type, ListType)
                if foreacharg is not None:
                    name_in_expr = (
                        f"{foreacharg.name}{'[i]' if is_foreacharg_list_type else ''}"
                    )
                    src_name = name
    
```



## High-Level Overview


This Python file contains 2 class(es) and 29 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `gen_variable_type`, `wrapper_registrations`, `gen_wrapper_registration`, `gen_variable_type_func`, `emit_body`, `gen_differentiable_input`, `gen_differentiable_inputs`, `find_args_with_derivatives`, `emit_save_inputs`, `guard_for`, `setup_derivative`, `emit_check_if_in_complex_autograd_allowlist`, `emit_check_no_requires_grad`, `emit_original_self_definition`, `save_variables`, `emit_dispatch_call`, `wrap_output`, `check_tensorimpl_and_storage`, `emit_call`, `emit_history`

**Key imports**: annotations, re, TYPE_CHECKING, cpp, CodeTemplate, FileManager, mapMaybe, with_native_function_with_differentiability_info_and_key, Callable, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/autograd`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `re`
- `typing`: TYPE_CHECKING
- `torchgen.api`: cpp
- `torchgen.code_template`: CodeTemplate
- `torchgen.utils`: FileManager, mapMaybe
- `.context`: with_native_function_with_differentiability_info_and_key
- `collections.abc`: Callable, Sequence


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

Files in the same folder (`tools/autograd`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`derivatives.yaml_docs.md`](./derivatives.yaml_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`gen_autograd.py_docs.md`](./gen_autograd.py_docs.md)
- [`load_derivatives.py_docs.md`](./load_derivatives.py_docs.md)
- [`gen_view_funcs.py_docs.md`](./gen_view_funcs.py_docs.md)
- [`gen_inplace_or_view_type.py_docs.md`](./gen_inplace_or_view_type.py_docs.md)
- [`gen_python_functions.py_docs.md`](./gen_python_functions.py_docs.md)


## Cross-References

- **File Documentation**: `gen_variable_type.py_docs.md`
- **Keyword Index**: `gen_variable_type.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
