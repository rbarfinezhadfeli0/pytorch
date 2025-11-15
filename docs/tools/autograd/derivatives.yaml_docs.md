# Documentation: `tools/autograd/derivatives.yaml`

## File Metadata

- **Path**: `tools/autograd/derivatives.yaml`
- **Size**: 182,902 bytes (178.62 KB)
- **Type**: YAML Configuration
- **Extension**: `.yaml`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```yaml
# Defines derivative formulas and Python signatures of methods on Variable
#
# Note about possibly confusing nomenclature: An 'output gradient' is the
# gradient of an output of a forward function. Output gradients are used as
# the inputs to backward functions. `grads` is a vector of output gradients,
# and `grad == grads[0]`, in all the derivative formulas in this file.
# An 'input gradient' is the gradient of an input to a forward function.
# Input gradients are the outputs of backward functions, corresponding to the
# input names included in the derivative formulas defined in this file.
# Also, every time we talk computing "gradient" we actually mean computing
# the vector jacobian product using the given 'output gradient' as the vector.
#
# Each entry consists of:
#   - A 'name', which specifies the ATen name of the function you
#     are defining derivatives for, and an argument specification.
#   - An optional 'dispatch' entry which can be used to specify
#     per-autograd dispatch key derivatives. If this entry is not
#     specified, then the gradient entries will be taken as the
#     default gradients (i.e. registered for every backward dispatch
#     key). (see _test_autograd_multiple_dispatch for an example
#     of how to register separate derivates for different dispatch keys).
#     The list of allowed dispatch keys (in addition to 'Default' which
#     represents the Autograd alias key) is torchgen/model.py:AUTOGRAD_KEYS.
#   - One or more gradients entries, mapping differentiable input
#     names to a formula specifying how to compute its gradient.
#     Note that a single gradient entry can specify the gradient
#     formula for multiple input names, by specifying a key
#     "input1, input2" (see atan2 for an example).
#   - An argument can be flagged as 'non_differentiable'.
#   - Optional entry with key 'output_differentiability' and value a list of the
#     same length as the number of outputs from the forward function. The list
#     should contain only booleans, specifying whether each of the output Tensor
#     is differentiable.
#     If it is not specified for a function that returns multiple elements but
#     uses `grad` instead of `grads[idx]`, then all but the first output will
#     be marked as non-differentiable.
#     If None of the output is differentiable, you can also add the function
#     name to `gen_variable_type.py`'s `DONT_REQUIRE_DERIVATIVE` list.
#
# There are two cases for Tensor and TensorList arguments here:
#   - If that argument is differentiable, in the sense that a gradient with respect
#     to that argument could exist. You should either:
#       - Specify the formula for that gradient
#       - Specify not_implemented("function_name") as a formula to say that this is not
#         implemented yet (but might be in the future and the user can request that on an issue)
#   - If that argument is not differentiable, because it is not a floating point dtype or the
#     function is not differentiable with respect to that argument  for
#     example. You should either:
#       - Do not specify any formula for this argument
#       - Specify explicitly that this argument is "non_differentiable". Note that in this case,
#         we trust you that this argument will never have requires_grad=True and it will be silently
#         ignored if it does.
#
# If a function has out-of-place and in-place variants, then the derivative
# definition for the in-place variant is optional. It will default to the
# definition for the out-of-place variant. Note that _out variants are never
# differentiable.
#
# Gradient expressions are standard C++ expressions operating on ATen
# variables.  In a gradient expression, the following variables/functions
# are in scope:
#
#   - 'grad', the gradient of the output (often spelled grad_output
#     in Python) which we are going to left-multiply.
#
#     When a function returns multiple *differentiable* outputs,
#     you can refer to the gradients of each outputs using 'grads',
#     e.g., 'grads[0]', 'grads[1]'.
#
#     When a function returns multiple *differentiable* outputs that
#     are named, you can refer to the gradients of each outputs using
#     'grad_{name}', e.g., 'grad_x', 'grad_y'.
#
#     When a function returns *one* differentiable output (the
#     first output) and some more nondifferentiable outputs,
#     you MUST refer to the gradient of the differentiable output with
#     'grad' (this case is special-cased in our code generation).
#
#     Note that the number of differentiable outputs can be modified by the
#     'output_differentiability' entry (see above).
#
#     Across a differentiable function's derivatives set, it is not
#     permitted to mix the use of "grad", "grads", and
#     "grad_{name}". You must be consistent for that differentiable
#     function.
#
#   - Any of the input arguments, tensor or non-tensor, including
#     argument names that only appear in Declarations.yaml, e.g. 'output'.
#
#   - 'result', representing the result of evaluating the forward
#     expression for ATen native function declarations. If the forward
#     expression outputs a tuple, use 'resultX' instead to access the
#     X-th entry
#
#   - 'grad_input_mask', a std::array<bool, n>, specifies which input
#     gradients are actually needed.  For example, in the entry
#     `input0, input1: foo(grad_input_mask)`, `grad_input_mask` is a size
#     two array, where `grad_input_mask[0]` is true if `input0` requires
#     grad, and `grad_input_mask[1]` is true if `input1` requires grad.
#
#     (NB: if your function computes gradient for a list of tensors,
#     the `grad_input_mask` will only have a single entry for the list
#     specifying if either zero or at least one tensor from the list requires
#     grad.  If we want to support more fine-grained signalling,
#     we'll need some alternate variable which is not a std::array)
#
#   - 'retain_variables', a bool which is true if a user has specified
#     that saved variables should be retained in case the backwards is
#     run again later.  This allows an optimization where we can
#     destroy saved buffers if we know variables are not going to be retained,
#     e.g., it is used by _cudnn_rnn
#
#   - `wrap_opt_if`, is a 2-argument function that accepts a tensor
#     variable and a boolean condition that dictates whether to save that
#     variable in a graph. The result of this function is `std::optional<Tensor>`,
#     and it is `::std::nullopt` when the condition evaluates to `false`,
#     otherwise it is the variable wrapped in `std::optional<Tensor>`.
#     For example, wrap_opt_if(var_0, grad_input_mask[1] || grad_input_mask[2])
#     would mean that `var_0` is saved as long as the second (grad_input_mask[1])
#     or the third (grad_input_mask[2]) argument requires gradients.
#     Another interpretation of this expression would read as `var_0` is needed
#     in the backward computation of the second or the third argument.
#     NOTE: the usage of `var_i.requires_grad()` in the conditional expression
#     is not supported, use `grad_input_mask[i]` instead.
#     NOTE: `wrap_opt_if` could be used to prevent saving redundant variables
#     with multi-output backward formulas.
#     See https://github.com/pytorch/pytorch/issues/97575 for more details
#     on the issue.
#
# If you need a complex expression, e.g., with local variables,
# write a _backward function in torch/csrc/autograd/FunctionsManual.cpp
# and invoke it from here.  By the way, go read
# https://github.com/zdevito/ATen/issues/163; this describes an
# important hazard that occurs when porting backwards from Python to C++
#
# Double backwards gradient expressions can be somewhat confusing;
# the most important thing to remember is: (1) you need to define a
# derivative formula for every input, including inputs named things
# like 'grad_output', and (2) the gradient to multiply with is always
# called 'grad' (even though it really is a grad-grad).
#
# You can also add forward derivative definition by defining a formula for
# a returned value (in general "result" if the name is not specified). This
# formula works the same way as the backward one and advanced implementations
# should also be placed in the FunctionsManual file.
# This formula should compute a single Jacobian vector product using the (primal)
# value of the argument "foo_p", its forward grad "foo_t" and the result of the
# function as "result".
# Note that the forward derivative can be automatically generated in two cases:
#     - if your function is linear (NOT affine or multi-linear), then you can
#       specify so by just using the string "auto_linear" for the formula.
#     - if your function is applied element wise (and has a single input), you
#       can specify so by just using the string "auto_element_wise" for the formula.
#
# Note that to avoid unpacking overhead, functions taking TensorList as inputs
# will always have their forward grad formula called. This function is responsible
# to check if any computation is needed and should return an undefined Tensor when
# there is nothing to do. You can check "cat_forward" for a full example.
#
# NB: There are a number of gradient definitions in here which are bogus
# (implemented using zeros_like).  These gradients are (hopefully) not
# used by our frontend.  You MUST check the frontend code; search for
# OpName.apply to see if it's still using a legacy Python style API.
#
# Note: Returning views.
# The following cases exist:
#     - If a function returns no view, it can have arbitrary outputs.
#     - If a function return at least one Tensor that is a differentiable view
#       of one of its input:
#         - If there is only one differentiable output, this Tensor is marked as a
#           differentiable view. (alias or transpose for example)
#         - If there are more than one differentiable output, by default all the views are
#           marked as differentiable views and created with allow_rebase_history=false.
#           Meaning that any inplace operation on it will raise an error. (unbind for example)
#
#  Notes about undefined output gradients:
#     All backward functions must support all combinations of undefined output
#     gradient Tensors, where `grad[i].defined() == false`. Depending on the
#     number of input and output grads your derivative formula uses, code
#     generation may automatically add some level of undefined grad support,
#     according to these three cases:
#
#       * 1 input grad and 1 output grad:
#           Complete undefined grad support is automatically added, so you
#           shouldn't have to think about it, unless there is a bug in the code
#           generation.
#
#       * 1 input grad and multiple output grads:
#           Undefined grad support is automatically added ONLY in the case where
#           all output grads are undefined. You will have to add explicit support
#           for cases where a subset of output grads is undefined.
#
#       * multiple input grads:
#           No automatic support, so you will need to add it.
#
#     If your derivative formula uses more than one output grad, it is usually
#     preferable to add undefined grad support in the backward function itself
#     (if you're using one), rather than in the derivative formula in this file.
#
#     Undefined Tensors are created with the default constructor `at::Tensor()`.
#     It is an efficient way to represent a Tensor filled with zeros because
#     the Tensor holds no sizing information and no Storage data is allocated.
#     But consequently, Tensor operations cannot be performed on them.
#     Therefore, your backward function should treat an undefined output grad as
#     a zero, and it needs to be a special case.
#
#     If all output grads are undefined, then it should be correct for the
#     backward function to return undefined input grads. Since we use the chain
#     rule, output grads equal to zero should result in input grads equal to zero,
#     unless there is some rare special case.
#
#     If a subset of output grads is undefined, then it may be acceptable for
#     the backward function to return undefined input grads--it depends on the
#     specific function, so you'll have to determine that yourself. If returning
#     an undefined Tensor is correct for a given input grad, it is also logically
#     correct to return a defined grad full of zeros, but that would not be
#     preferable since it would be less efficient.
#
# NB: The parameter names here MUST be consistent with the parameter names
# in native_functions.yaml
- name: abs(Tensor self) -> Tensor
  self: grad * self.sgn()
  result: handle_r_to_c(result.scalar_type(), self_t.conj() * self_p.sgn())

- name: acos(Tensor self) -> Tensor
  self: grad * -((-self * self + 1).rsqrt()).conj()
  result: auto_element_wise

- name: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  other: handle_r_to_c(other.scalar_type(), maybe_multiply(grad, alpha.conj()))
  result: self_t + maybe_multiply(other_t, alpha)

- name: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  result: self_t.clone()

- name: addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: maybe_multiply(grad, beta.conj())
  batch1: maybe_multiply(grad.unsqueeze(0).expand_symint({ batch1.sym_size(0), batch1.sym_size(1), batch2.sym_size(2) }).bmm(batch2.transpose(1, 2).conj()), alpha.conj())
  batch2: maybe_multiply(batch1.transpose(1, 2).conj().bmm(grad.unsqueeze(0).expand_symint({ batch1.sym_size(0), batch1.sym_size(1), batch2.sym_size(2) })), alpha.conj())
  result: maybe_multiply(self_t, beta) + maybe_multiply(batch1_t.bmm(batch2_p).sum(0), alpha) + maybe_multiply(batch1_p.bmm(batch2_t).sum(0), alpha)

- name: addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  tensor1: handle_r_to_c(tensor1.scalar_type(), grad * (value / tensor2).conj())
  tensor2: handle_r_to_c(tensor2.scalar_type(), -grad * (value * tensor1 / (tensor2 * tensor2)).conj())
  result: self_t + maybe_multiply(tensor1_t / tensor2_p, value) - maybe_multiply(tensor2_t * (tensor1_p / tensor2_p) / tensor2_p, value)

- name: addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  tensor1: handle_r_to_c(tensor1.scalar_type(), grad * (tensor2 * value).conj())
  tensor2: handle_r_to_c(tensor2.scalar_type(), grad * (tensor1 * value).conj())
  result: self_t + maybe_multiply(tensor1_t * tensor2_p, value) + maybe_multiply(tensor2_t * tensor1_p, value)

- name: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: maybe_multiply(grad, beta.conj())
  mat1: mm_mat1_backward(grad, mat2, mat1.sym_sizes(), mat1.sym_strides(), mat1.layout(), alpha)
  mat2: mm_mat2_backward(grad, mat1, mat2.sym_sizes(), mat2.sym_strides(), mat2.layout(), alpha)
  result: maybe_multiply(self_t, beta) + maybe_multiply(mat1_t.mm(mat2_p), alpha) + maybe_multiply(mat1_p.mm(mat2_t), alpha)

- name: _sparse_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: maybe_multiply(grad, beta)
  mat1: mm_mat1_sparse_backward(grad, mat1, mat2, alpha)
  mat2: mm_mat2_backward(grad, mat1, mat2.sym_sizes(), mat2.sym_strides(), mat2.layout(), alpha)

- name: addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: maybe_multiply(grad, beta.conj())
  mat: maybe_multiply(grad.ger(vec.conj()), alpha.conj())
  vec: maybe_multiply(mat.t().conj().mv(grad), alpha.conj())
  result: maybe_multiply(self_t, beta) + maybe_multiply(mat_t.mv(vec_p), alpha) + maybe_multiply(mat_p.mv(vec_t), alpha)

- name: addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: maybe_multiply(grad, beta.conj())
  vec1: maybe_multiply(grad.mv(vec2.conj()), alpha.conj())
  vec2: maybe_multiply(grad.t().mv(vec1.conj()), alpha.conj())
  result: maybe_multiply(self_t, beta) + maybe_multiply(vec1_t.outer(vec2_p), alpha) + maybe_multiply(vec1_p.outer(vec2_t), alpha)

- name: affine_grid_generator(Tensor theta, SymInt[] size, bool align_corners) -> Tensor
  theta: affine_grid_generator_backward_symint(grad, size, align_corners)
  result: auto_linear

- name: alias(Tensor(a) self) -> Tensor(a)
  self: grad
  result: self_t

- name: angle(Tensor self) -> Tensor
  self: angle_backward(grad, self)
  result: handle_r_to_c(result.scalar_type(), angle_backward(self_t.conj(), self_p).conj())

# The four items below are necessary because TensorIterator doesn't work on
# Variables (codegen does not unwrap the input Tensor for all() and any() ).
- name: any(Tensor self) -> Tensor
  output_differentiability: [False]

- name: any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
  output_differentiability: [False]

- name: any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
  output_differentiability: [False]

- name: _is_all_true(Tensor self) -> Tensor
  self: non_differentiable

- name: _is_any_true(Tensor self) -> Tensor
  self: non_differentiable

- name: all(Tensor self) -> Tensor
  output_differentiability: [False]

- name: all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
  output_differentiability: [False]

- name: all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
  output_differentiability: [False]

- name: acosh(Tensor self) -> Tensor
# Save one rsqrt in the real case by using that for x real and positive sqrt(x*y) = sqrt(x)*sqrt(y) (not true in the complex case)
  self: "self.is_complex() ? grad * ((self + 1).rsqrt() * (self - 1).rsqrt()).conj() : grad * (self * self - 1).rsqrt()"
  result: auto_element_wise

- name: acosh_(Tensor(a!) self) -> Tensor(a!)
  self: not_implemented("inplace version of acosh")

- name: asinh(Tensor self) -> Tensor
  self: grad * (self.pow(2) + 1).rsqrt().conj()
  result: auto_element_wise

- name: asinh_(Tensor(a!) self) -> Tensor(a!)
  self: not_implemented("inplace version of asinh")

- name: atanh(Tensor self) -> Tensor
  self: grad * 1 / (1 - self.pow(2)).conj()
  result: auto_element_wise

- name: atanh_(Tensor(a!) self) -> Tensor(a!)
  self: not_implemented("inplace version of atanh")

- name: as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
  self: as_strided_backward(grad, TensorGeometry(self), size, stride, storage_offset)
  result: auto_linear

- name: as_strided_(Tensor(a!) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a!)
  self: as_strided_backward(grad, TensorGeometry(self), size, stride, storage_offset)
  result: auto_linear

- name: asin(Tensor self) -> Tensor
  self: grad * (-self * self + 1).rsqrt().conj()
  result: auto_element_wise

- name: atan(Tensor self) -> Tensor
  self: grad / (self * self + 1).conj()
  result: auto_element_wise

- name: atan2(Tensor self, Tensor other) -> Tensor
  self, other: atan2_backward(grad, self, other, grad_input_mask)
  result: (-self_p * other_t + other_p * self_t) / (self_p.pow(2) + other_p.pow(2))

- name: baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: maybe_multiply(grad, beta.conj())
  batch1: maybe_multiply(grad.bmm(batch2.transpose(1, 2).conj()), alpha.conj())
  batch2: maybe_multiply(batch1.transpose(1, 2).conj().bmm(grad), alpha.conj())
  result: maybe_multiply(self_t, beta) + maybe_multiply(batch1_t.bmm(batch2_p), alpha) + maybe_multiply(batch1_p.bmm(batch2_t), alpha)

- name: bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
  self: zeros_like(grad)
  result: auto_element_wise

- name: bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  p: zeros_like(p)
  result: self_t.zero_()

- name: bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.zero_()

- name: bmm(Tensor self, Tensor mat2) -> Tensor
  self: grad.bmm(mat2.transpose(1, 2).conj())
  mat2: self.transpose(1, 2).conj().bmm(grad)
  result: self_t.bmm(mat2_p) + self_p.bmm(mat2_t)

- name: matmul(Tensor self, Tensor other) -> Tensor
  self, other: matmul_backward(grad, self, other, grad_input_mask)

- name: cat(Tensor[] tensors, int dim=0) -> Tensor
  tensors: cat_tensors_backward(grad, to_args_sizes_symint(tensors), to_args_scalartypes(tensors), dim)
  result: cat_jvp(tensors, dim)

- name: cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.zero_()

- name: ceil(Tensor self) -> Tensor
  self: zeros_like(grad)
  result: auto_element_wise

- name: cholesky(Tensor self, bool upper=False) -> Tensor
  self: cholesky_backward(grad, upper, result)

- name: chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]
  dispatch:
    Default:
      # the default case will use the CompositeImplicitAutograd
      self: not_implemented("chunk")
    AutogradNestedTensor:
      self: chunk_backward_nested(grads, self, chunks, dim)

- name: linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)
  self: cholesky_backward(grad, upper, L)
  L: cholesky_jvp(self_t, L, upper)

- name: cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
  self, input2: cholesky_solve_backward(grad, self, input2, result, upper, grad_input_mask)
  result: cholesky_solve_jvp(result, input2_p, input2_t, self_t, upper)

- name: cholesky_inverse(Tensor self, bool upper=False) -> Tensor
  self: cholesky_inverse_backward(grad, self, upper, result)
  result: cholesky_inverse_jvp(self_p, self_t, result, upper)

# For clamp, gradient is not defined at the boundaries. But empirically it's helpful
# to be able to get gradient on min and max, so we return the subgradient 1 for these cases.
- name: clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
  self: clamp_backward(grad, self, min, max)
  min, max: clamp_backward_min_max(grad, self, min, max, grad_input_mask)
  result: clamp_jvp(self_p, self_t, min_p, min_t, max_p, max_t)

- name: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
  self: clamp_backward(grad, self, min, max)
  result: auto_element_wise

- name: clamp_min(Tensor self, Scalar min) -> Tensor
  self: where(self >= min, grad, at::scalar_tensor(0., grad.options()))
  result: auto_element_wise

- name: clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
  self: where(self >= min, grad, at::scalar_tensor(0., grad.options()))
  min: where(self < min, grad, at::scalar_tensor(0., grad.options()))
  result: where(self_p >= min_p, self_t, min_t)

- name: clamp_max(Tensor self, Scalar max) -> Tensor
  self: where(self <= max, grad, at::scalar_tensor(0., grad.options()))
  result: auto_element_wise

- name: clamp_max.Tensor(Tensor self, Tensor max) -> Tensor
  self: where(self <= max, grad, at::scalar_tensor(0., grad.options()))
  max: where(self > max, grad, at::scalar_tensor(0., grad.options()))
  result: where(self_p <= max_p, self_t, max_t)

- name: clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
  self: grad
  result: auto_linear

- name: _lazy_clone(Tensor self) -> Tensor
  self: grad
  result: auto_linear

- name: _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
  self: _to_copy_backward(grad, self.options())
  result: _to_copy(self_t, dtype, layout, device, pin_memory, non_blocking, memory_format)
  # The condition is: if dtype is not nullopt, then isDifferentiableType(*dtype)
  # (If dtype IS nullopt, we rely on the regular check that any input requires grad).
  output_differentiability: ["!dtype || isDifferentiableType(*dtype)"]

- name: _coalesce(Tensor self) -> Tensor
  self: grad

- name: complex(Tensor real, Tensor imag) -> Tensor
  real: at::real(grad)
  imag: at::imag(grad)
  result: at::complex(real_t, imag_t)

- name: polar(Tensor abs, Tensor angle) -> Tensor
  abs, angle: polar_backward(grad, result)
  result: at::complex(abs_t*angle_p.cos() - angle_t*abs_p*angle_p.sin(), abs_t*angle_p.sin() + angle_t*abs_p*angle_p.cos())

- name: _conj(Tensor(a) self) -> Tensor(a)
  self: grad.conj()
  result: self_t.conj()

- name: _neg_view(Tensor(a) self) -> Tensor(a)
  self: grad.neg()
  result: self_t._neg_view()

- name: _conj_physical(Tensor self) -> Tensor
  self: grad.conj_physical()
  result: self_t.conj_physical()

- name: conj_physical_(Tensor(a!) self) -> Tensor(a!)
  self: grad.conj_physical()
  result: self_t.conj_physical_()

- name: copysign.Tensor(Tensor self, Tensor other) -> Tensor
  self: copysign_tensor_self_backward(grad, self, result)
  other: zeros_like(other)
  result: copysign_tensor_self_backward(self_t, self_p, result)

- name: copysign.Scalar(Tensor self, Scalar other) -> Tensor
  self: copysign_tensor_self_backward(grad, self, result)
  result: auto_element_wise

- name: cos(Tensor self) -> Tensor
  self: grad * -self.sin().conj()
  result: auto_element_wise

- name: cosh(Tensor self) -> Tensor
  self: grad * self.sinh().conj()
  result: auto_element_wise

- name: count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
  output_differentiability: [False]

- name: count_nonzero(Tensor self, int? dim=None) -> Tensor
  output_differentiability: [False]

- name: linalg_cross(Tensor self, Tensor other, *, int dim=-1) -> Tensor
  self: at::linalg_cross(other.conj(), grad, dim)
  other: at::linalg_cross(grad, self.conj(), dim)
  result: "at::linalg_cross(self_t, other_p, dim) + at::linalg_cross(self_p, other_t, dim)"

- name: logcumsumexp(Tensor self, int dim) -> Tensor
  self: logcumsumexp_backward(grad, self, result, dim)
  result: logcumsumexp_jvp(self_p, self_t, dim)

- name: cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
  self: cumprod_backward(grad.to(self.scalar_type()), self, dim, result)
  result: "cumprod_jvp(self_t, self_p, result, dim).to(dtype.has_value() ? *dtype : self_p.scalar_type())"

- name: cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
  self: cumsum_backward(grad.to(self.scalar_type()), dim)
  result: auto_linear

- name: cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
  self: cummaxmin_backward(grad, self, indices, dim)
  values: self_t.gather(dim, indices)

- name: cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
  self: cummaxmin_backward(grad, self, indices, dim)
  values: self_t.gather(dim, indices)

- name: conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor
  self, weight, bias: "grad.defined() ? conv_tbc_backward(grad, self, weight, bias, pad) : std::tuple<Tensor, Tensor, Tensor>()"

- name: _ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
  log_probs: _ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, result0, result1, blank, zero_infinity)

- name: _ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
  log_probs: _ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, result0, result1, blank, zero_infinity)

- name: deg2rad(Tensor self) -> Tensor
  self: deg2rad_backward(grad)
  result: auto_element_wise

- name: _linalg_det(Tensor A) -> (Tensor result, Tensor LU, Tensor pivots)
  A: linalg_det_backward(grad, result, A, LU, pivots)
  result: linalg_det_jvp(A_t, result, LU, pivots, A_p.is_contiguous() && !A_p.is_complex())
  output_differentiability: [True, False, False]

- name: _linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet, Tensor LU, Tensor pivots)
  A: slogdet_backward(grad_sign, grad_logabsdet, A, sign, LU, pivots)
  sign, logabsdet: slogdet_jvp(LU, pivots, A_t, sign, A_p.is_contiguous() && !A_p.is_complex())
  output_differentiability: [True, True, False, False]

- name: block_diag(Tensor[] tensors) -> Tensor
  tensors: block_diag_backward(grad, to_args_sizes(tensors), to_args_scalartypes(tensors))
  result: block_diag_jvp(tensors)

- name: diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
  self: grad.diagonal(offset, dim1, dim2)
  result: auto_linear

- name: diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
  self: diagonal_backward_symint(grad, self.sym_sizes(), offset, dim1, dim2)
  result: auto_linear

- name: diagonal_backward(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2) -> Tensor
  grad_output: grad.diagonal(offset, dim1, dim2)
  result: auto_linear

- name: dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
  self: norm_backward(grad, self - other, p, result)
  other: -norm_backward(grad, self - other, p, result)
  result: norm_jvp(self_p - other_p, self_t - other_t, p, result, {}, false)

# The backward formula is done in this order to improve numerical stability
# of the higher order derivatives, see https://github.com/pytorch/pytorch/issues/43414
# Note that we don't use "result" because saving it would be BC-breaking when it is used in an inplace operation later
- name: div.Tensor(Tensor self, Tensor other) -> Tensor
  self: div_tensor_self_backward(grad, other, self.scalar_type())
  other: div_tensor_other_backward(grad, self, other)
  result: (self_t - other_t * result) / other_p

- name: div.Scalar(Tensor self, Scalar other) -> Tensor
  self: div_tensor_self_backward(grad, other, self.scalar_type())
  result: self_t / other

- name: div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
  self: div_tensor_self_backward(grad, other, self.scalar_type(), rounding_mode)
  other: div_tensor_other_backward(grad, self, other, rounding_mode)
  result: "rounding_mode.has_value() ? result.new_zeros_symint(result.sym_sizes()) : self_t / other_p - other_t * (self_p / other_p) / other_p"

- name: div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
  self: div_tensor_self_backward(grad, other, self.scalar_type(), rounding_mode)
  result: "rounding_mode.has_value() ? result.new_zeros_symint(result.sym_sizes()) : self_t / other"

- name: dot(Tensor self, Tensor tensor) -> Tensor
  self: grad * tensor.conj()
  tensor: grad * self.conj()
  result: at::dot(self_t, tensor_p) + at::dot(self_p, tensor_t)

- name: vdot(Tensor self, Tensor other) -> Tensor
  self: grad.conj() * other
  other: grad * self
  result: at::vdot(self_t, other_p) + at::vdot(self_p, other_t)

- name: _fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)
  self: _fused_dropout_backward(grad, result1, p)

- name: native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)
  input: "GradMode::is_enabled() ? infinitely_differentiable_native_dropout_backward(grad, result1, (!train.has_value() || !train.value() ? 1 : (p == 1 ? 0.0 : 1.0 / (1.0 - p)))) : native_dropout_backward(grad, result1, (!train.has_value() || !train.value() ? 1 : (p == 1 ? 0.0 : 1.0 / (1.0 - p))))"
  result0: "(!train.has_value() || train.value()) ? (p == 1 ? 0.0 : 1.0 / (1.0 - p)) * input_t * result1 : input_t"

- name: native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor
  grad_output: "native_dropout_double_backward(grad, grad_output, mask, scale)"
  mask: 'not_implemented("native_dropout_backward: mask")'

- name: eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
  self: zeros_like(self)
  result: self_t.zero_()

- name: eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  self: zeros_like(self)
  other: zeros_like(other)
  result: self_t.zero_()

- name: erf(Tensor self) -> Tensor
  self: 2.0 / sqrt(M_PI) * exp(-(self.pow(2))) * grad
  result: auto_element_wise

- name: erfc(Tensor self) -> Tensor
  self: -2.0 / sqrt(M_PI) * exp(-(self.pow(2))) * grad
  result: auto_element_wise

- name: special_erfcx(Tensor self) -> Tensor
  self: (2.0 * self * result - 2.0 / sqrt(M_PI)) * grad
  result: auto_element_wise

- name: erfinv(Tensor self) -> Tensor
  self: 0.5 * sqrt(M_PI) * exp(self.erfinv().pow(2)) * grad
  result: auto_element_wise

- name: exp(Tensor self) -> Tensor
  self: grad * result.conj()
  result: auto_element_wise

- name: exp2(Tensor self) -> Tensor
  self: grad * result.conj() * M_LN2
  result: auto_element_wise

- name: expm1(Tensor self) -> Tensor
  self: grad * (result.conj() + 1)
  result: auto_element_wise

# TODO: this derivative is not SymInt safe, need sum_to support
- name: expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
  self: at::sum_to(grad, self.sym_sizes())
  result: auto_linear

- name: exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.zero_()

- name: fake_quantize_per_tensor_affine_cachemask(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
  self: fake_quantize_per_tensor_affine_cachemask_backward(grad, mask)

- name: _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, Tensor fake_quant_enabled, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
  self: fake_quantize_per_tensor_affine_cachemask_backward(grad, mask)

- name: _fake_quantize_learnable_per_tensor_affine(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
  self, scale, zero_point: "grad.defined() ? _fake_quantize_learnable_per_tensor_affine_backward(grad, self, scale, zero_point, quant_min, quant_max, grad_factor) : std::tuple<Tensor, Tensor, Tensor>()"

- name: fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
  self: fake_quantize_per_channel_affine_cachemask_backward(grad, mask)

- name: _fake_quantize_learnable_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
  self, scale, zero_point: "grad.defined() ? _fake_quantize_learnable_per_channel_affine_backward(grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor) : std::tuple<Tensor, Tensor, Tensor>()"

- name: _fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)
  self: fake_quantize_per_tensor_affine_cachemask_backward(grad, mask)

- name: fill.Scalar(Tensor self, Scalar value) -> Tensor
  self: zeros_like(grad)
  result: at::fill(self_t, 0)

- name: fill.Tensor(Tensor self, Tensor value) -> Tensor
  self: zeros_like(grad)
  value: grad.sum()
  result: at::fill(self_t, value_t)

- name: fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.fill_(0)

- name: fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
  self: zeros_like(grad)
  value: grad.sum()
  result: self_t.fill_(value_t)

- name: floor(Tensor self) -> Tensor
  self: zeros_like(grad)
  result: auto_element_wise

- name: fmod.Scalar(Tensor self, Scalar other) -> Tensor
  self: grad
  result: auto_element_wise

- name: fmod.Tensor(Tensor self, Tensor other) -> Tensor
  self: grad
  other: -grad * self.div(other, /*rounding_mode=*/"trunc")
  result: self_t - other_t * self_p.div(other_p, /*rounding_mode=*/"trunc")

- name: frac(Tensor self) -> Tensor
  self: grad
  result: self_t

- name: frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)
  self: grad / exponent.exp2()
  mantissa: self_t / exponent.exp2()

- name: gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
  self: gather_backward(grad, self, dim, index, sparse_grad)
  index: non_differentiable
  result: auto_linear

- name: ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
  self: zeros_like(self)
  result: self_t.zero_()

- name: ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  self: zeros_like(self)
  other: zeros_like(other)
  result: self_t.zero_()

- name: geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.zero_()

- name: geqrf(Tensor self) -> (Tensor a, Tensor tau)
  self: not_implemented("geqrf")

- name: indices(Tensor(a) self) -> Tensor(a)
  output_differentiability: [False]

- name: _indices(Tensor(a) self) -> Tensor(a)
  output_differentiability: [False]

- name: crow_indices(Tensor(a) self) -> Tensor(a)
  output_differentiability: [False]

- name: col_indices(Tensor(a) self) -> Tensor(a)
  output_differentiability: [False]

- name: ccol_indices(Tensor(a) self) -> Tensor(a)
  output_differentiability: [False]

- name: row_indices(Tensor(a) self) -> Tensor(a)
  output_differentiability: [False]

- name: grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
  input, grid: "grad.defined() ? grid_sampler_2d_backward(grad, input, grid, interpolation_mode, padding_mode, align_corners, grad_input_mask) : std::tuple<Tensor, Tensor>()"

- name: grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
  input, grid: "grad.defined() ? grid_sampler_3d_backward(grad, input, grid, interpolation_mode, padding_mode, align_corners, grad_input_mask) : std::tuple<Tensor, Tensor>()"

# See NOTE [ grid_sample CPU fallback ]
- name: _grid_sampler_2d_cpu_fallback(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
  input, grid: "grad.defined() ? _grid_sampler_2d_cpu_fallback_backward(grad, input, grid, interpolation_mode, padding_mode, align_corners) : std::tuple<Tensor, Tensor>()"

- name: gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
  self: zeros_like(self)
  result: self_t.zero_()

- name: gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  self: zeros_like(self)
  other: zeros_like(other)
  result: self_t.zero_()

- name: hardsigmoid(Tensor self) -> Tensor
  self: hardsigmoid_backward(grad, self)
  result: auto_element_wise

- name: histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
  output_differentiability: [False]

- name: hardswish(Tensor self) -> Tensor
  self: hardswish_backward(grad, self)
  result: auto_element_wise

- name: hardswish_backward(Tensor grad_output, Tensor self) -> Tensor
  grad_output: hardswish_backward(grad, self)
  self: at::where(at::logical_and(-3.0 < self, self < 3.0), grad * grad_output / 3.0, at::zeros({}, self.options()))
  result: "hardswish_backward(grad_output_t, self_p)
         + at::where(at::logical_and(-3.0 < self_p, self_p < 3.0), self_t * grad_output_p / 3.0, at::zeros({}, self_p.options()))"

- name: hypot(Tensor self, Tensor other) -> Tensor
  self: grad * self / result
  other: grad * other / result
  result: self_t * self_p / result + other_t * other_p / result

- name: i0(Tensor self) -> Tensor
  self: grad * at::special_i1(self)
  result: auto_element_wise

- name: special_i0e(Tensor self) -> Tensor
  self: grad * (at::special_i1e(self) - self.sgn() * result)
  result: auto_element_wise

- name: special_i1(Tensor self) -> Tensor
  self: i1_backward(grad, self, result)
  result: auto_element_wise

- name: special_i1e(Tensor self) -> Tensor
  self: i1e_backward(grad, self, result)
  result: auto_element_wise

- name: igamma(Tensor self, Tensor other) -> Tensor
  self: 'not_implemented("igamma: input")'
  other: grad * exp((self - 1) * log(other) - other - lgamma(self))

- name: igammac(Tensor self, Tensor other) -> Tensor
  self: 'not_implemented("igammac: input")'
  other: -grad * exp((self - 1) * log(other) - other - lgamma(self))

- name: index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
  self: index_backward(grad.new_zeros_symint(self.sym_sizes(), self.options()), indices, grad)
  result: auto_linear

- name: _unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
  self: at::_unsafe_index_put(grad.new_zeros_symint(self.sym_sizes(), self.options()), indices, grad, true)
  result: auto_linear

- name: _unsafe_masked_index(Tensor self, Tensor mask, Tensor?[] indices, Scalar fill) -> Tensor
  self: at::_unsafe_masked_index_put_accumulate(grad.new_zeros_symint(self.sym_sizes(), self.options()), mask, indices, grad)
  mask: non_differentiable
  result: _unsafe_masked_index(self_t, mask, indices, 0)

- name: _unsafe_masked_index_put_accumulate(Tensor self, Tensor mask, Tensor?[] indices, Tensor values) -> Tensor
  self: grad
  mask: non_differentiable
  values: at::_unsafe_masked_index(grad, mask, indices, 0)
  result: at::_unsafe_masked_index_put_accumulate(self_t, mask, indices, values_t)

- name: index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
  self: grad
  # The case source.dim() == 0  is necessary to support scalar tensors of the form
  # source.dim() == 0 and index.dim() == 1 and index.size() == (1,),
  # This is because source is not broadcastable to index, as source.dim() < index.dim()
  source: "maybe_multiply(source.dim() > 0 ? grad.index_select(dim, index).expand_as(source) : grad.index_select(dim, index.squeeze(0)), alpha)"
  index: non_differentiable
  result: at::index_add(self_t, dim, index, maybe_multiply(source_t, alpha))

- name: index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor
  self, source: index_reduce_backward(grad, self, dim, index, source, reduce, include_self, result)
  index: non_differentiable

- name: index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
  self: grad.index_fill(dim, index, 0)
  # The case source.dim() == 0 is necessary to support scalar tensors of the form
  # source.dim() == 0 and index.dim() == 1 and index.size() == (1,),
  # This is because source is not broadcastable to index, as source.dim() < index.dim()
  source: "source.dim() > 0 ? grad.index_select(dim, index).expand_as(source) : grad.index_select(dim, index.squeeze(0))"
  index: non_differentiable
  result: self_t.index_copy(dim, index, source_t)

- name: index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
  self: grad.index_fill(dim, index, 0)
  index: non_differentiable
  result: self_t.index_fill(dim, index, 0)

- name: index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
  self: grad.index_fill(dim, index, 0)
  value: grad.index_select(dim, std::get<0>(at::_unique(index, /*sorted=*/false))).sum()
  index: non_differentiable
  result: self_t.index_fill(dim, index, value_t)

- name: index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
  self: "accumulate ? grad : grad.index_put(indices, zeros_like(values), false)"
  values: grad.index(indices)
  result: self_t.index_put(indices, values_t, accumulate)

- name: _unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
  self: "accumulate ? grad : at::_unsafe_index_put(grad, indices, zeros_like(values), false)"
  values: at::_unsafe_index(grad, indices)
  result: at::_unsafe_index_put(self_t, indices, values_t, accumulate)

- name: _index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)
  self: "accumulate ? grad : grad.index_put(indices, zeros_like(values), false)"
  values: grad.index(indices)
  result: at::_index_put_impl_(self_t, indices, values_t, accumulate, unsafe)

- name: index_select(Tensor self, int dim, Tensor index) -> Tensor
  self: index_select_backward_symint(grad, self.sym_sizes(), dim, index)
  index: non_differentiable
  result: auto_linear

- name: linalg_inv_ex(Tensor A, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
  A: -at::matmul(inverse.mH(), at::matmul(grad, inverse.mH()))
  inverse: -at::matmul(at::matmul(inverse, A_t), inverse)
  output_differentiability: [True, False]

- name: linalg_pinv.atol_rtol_tensor(Tensor self, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False) -> Tensor
  self: pinv_backward(grad, result, self)
  result: pinv_jvp(self_p, result, self_t)

- name: isnan(Tensor self) -> Tensor
  self: non_differentiable

- name: kthvalue(Tensor self, SymInt k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), keepdim)
  values: gather_with_keepdimed_indices(self_t, dim, indices, keepdim)

- name: le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
  self: zeros_like(self)
  result: self_t.zero_()

- name: le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  self: zeros_like(self)
  other: zeros_like(other)
  result: self_t.zero_()

- name: lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
  self: "weight.isComplex() ? grad * (1 - weight.conj().toComplexDouble()) : grad * (1 - weight.toDouble())"
  end: grad * weight.conj()
  result: at::lerp(self_t, end_t, weight)

- name: lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
  self: grad * (1 - weight).conj()
  end: grad * weight.conj()
  weight: grad * (end - self).conj()
  result: at::lerp(self_t, end_t, weight_p) + weight_t * (end_p - self_p)

- name: lgamma(Tensor self) -> Tensor
  self: grad * digamma(self)
  result: auto_element_wise

- name: digamma(Tensor self) -> Tensor
  self: grad * polygamma(1, self)
  result: auto_element_wise

- name: polygamma(int n, Tensor self) -> Tensor
  self: grad * polygamma(n + 1, self)
  result: auto_element_wise

- name: polygamma_(Tensor(a!) self, int n) -> Tensor(a!)
  self: grad * polygamma(n + 1, self)
  result: self_t.mul_(polygamma(n + 1, original_self_p))

- name: log(Tensor self) -> Tensor
  self: grad.div(self.conj())
  result: auto_element_wise

- name: log10(Tensor self) -> Tensor
  self: grad / (self.conj() * 2.3025850929940456)
  result: auto_element_wise

- name: log1p(Tensor self) -> Tensor
  self: log1p_backward(grad, self)
  result: auto_element_wise

- name: log2(Tensor self) -> Tensor
  self: grad / (self.conj() * 0.6931471805599453)
  result: auto_element_wise

- name: logaddexp(Tensor self, Tensor other) -> Tensor
  self: grad / (1 + exp(other - self)).conj()
  other: grad / (1 + exp(self - other)).conj()
  result: self_t / (1 + exp(other_p - self_p)) + other_t / (1 + exp(self_p - other_p))

- name: logaddexp2(Tensor self, Tensor other) -> Tensor
  self: grad / (1 + pow(2, other - self))
  other: grad / (1 + pow(2, self - other))
  result: self_t / (1 + pow(2, other_p - self_p)) + other_t / (1 + pow(2, self_p - other_p))

# Note [Gradient formula for xlogy at x = 0, y <= 0]
# x * log(y) is not defined at y <= 0, so we cannot even talk about differentiability
# Now, xlogy(0, y) = 0 by definition.
# This does not make it differentiable as it's not defined in a neighbourhood of a point
# (0, y) when y <= 0.
# Now, when a function is non-differentiable, sometimes we return "a relatively sensible value"
# In this case, as per the discussion in https://github.com/pytorch/pytorch/issues/80770, we choose
# this value to be zero, which is the directional derivative along the line {x = 0}.
- name: xlogy.Tensor(Tensor self, Tensor other) -> Tensor
  self: at::xlogy(grad, other).masked_fill((self == 0.) & (other <= 0.), 0.)
  other: grad * self / other
  result: at::xlogy(self_t, other_p).masked_fill((self_p == 0.) & (other_p <= 0.), 0.) + other_t * self_p / other_p

- name: xlogy.Scalar_Self(Scalar self, Tensor other) -> Tensor
  other: grad * self / other
  result: auto_element_wise

- name: xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor
  self: "other.toDouble() > 0.
          ? at::xlogy(grad,  other)
          : at::xlogy(grad,  other).masked_fill(self == 0., 0.)"
  result: auto_element_wise

# See Note [Gradient formula for xlogy at x = 0, y <= 0]
# Same here but with y <= -1
- name: special_xlog1py(Tensor self, Tensor other) -> Tensor
  self: at::special_xlog1py(grad,  other).masked_fill((self == 0.) & (other <= -1.), 0.)
  other: grad * self / (other + 1)
  result: at::special_xlog1py(self_t,  other_p).masked_fill((self_p == 0.) & (other_p <= -1.), 0.) + other_t * self_p / (other_p + 1)

- name: special_xlog1py.self_scalar(Scalar self, Tensor other) -> Tensor
  other: grad * self / (other + 1)
  result: auto_element_wise

- name: special_xlog1py.other_scalar(Tensor self, Scalar other) -> Tensor
  self: "other.toDouble() > -1.
          ? at::special_xlog1py(grad,  other)
          : at::special_xlog1py(grad,  other).masked_fill(self == 0., 0.)"
  result: auto_element_wise

- name: special_zeta(Tensor self, Tensor other) -> Tensor
  self: not_implemented("zeta")
  other:  grad * -self * special_zeta(self + 1., other)

- name: special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor
  other:  grad * -self * special_zeta(self.toDouble() + 1., other)

- name: special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor
  self: not_implemented("zeta")

- name: log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.zero_()

- name: logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
  self: logsumexp_backward(grad, self, result, dim, keepdim)
  result: logsumexp_jvp(self_p, self_t, dim, keepdim)

```



## High-Level Overview

This file is part of the PyTorch framework located at `tools/autograd`.

## Detailed Analysis

### Code Structure

This is a configuration file. See the original source for structure.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/autograd`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`tools/autograd`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`gen_variable_type.py_docs.md`](./gen_variable_type.py_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`gen_autograd.py_docs.md`](./gen_autograd.py_docs.md)
- [`load_derivatives.py_docs.md`](./load_derivatives.py_docs.md)
- [`gen_view_funcs.py_docs.md`](./gen_view_funcs.py_docs.md)
- [`gen_inplace_or_view_type.py_docs.md`](./gen_inplace_or_view_type.py_docs.md)
- [`gen_python_functions.py_docs.md`](./gen_python_functions.py_docs.md)


## Cross-References

- **File Documentation**: `derivatives.yaml_docs.md`
- **Keyword Index**: `derivatives.yaml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
