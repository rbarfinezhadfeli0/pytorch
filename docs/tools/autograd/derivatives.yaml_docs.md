# Documentation: derivatives.yaml

## File Metadata
- **Path**: `tools/autograd/derivatives.yaml`
- **Size**: 182902 bytes
- **Lines**: 3242
- **Extension**: .yaml
- **Type**: Regular file

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

- name: linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
  self, b: linalg_lstsq_backward(grads[0], grads[1], self, b, solution, grad_input_mask)
  solution: linalg_lstsq_solution_jvp(self_p, b_p, self_t, b_t)
  residuals: linalg_lstsq_residuals_jvp(self_p, b_p, self_t, b_t, solution, residuals)
  output_differentiability: [True, True, False, False]

- name: lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
  self: zeros_like(self)
  result: self_t.zero_()

- name: lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  self: zeros_like(self)
  other: zeros_like(other)
  result: self_t.zero_()

- name: linalg_lu_factor_ex(Tensor A, *, bool pivot=True, bool check_errors=False) -> (Tensor LU, Tensor pivots, Tensor info)
  A: lu_factor_ex_backward(grad, LU, pivots, pivot)
  LU: lu_factor_ex_jvp(A_t, LU, pivots, pivot)
  output_differentiability: [True, False, False]

- name: linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U)
  A: linalg_lu_backward(grad_L, grad_U, P, L, U, pivot)
  L: std::get<0>(linalg_lu_jvp(A_t, P, L, U, pivot))
  U: std::get<1>(linalg_lu_jvp(A_t, P, L, U, pivot))
  output_differentiability: [False, True, True]

- name: linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False) -> Tensor
  LU: linalg_lu_solve_LU(grad, LU, pivots, result, left, adjoint)
  B: "at::linalg_lu_solve(LU, pivots, grad, left, !adjoint)"
  result: linalg_lu_solve_jvp(result, LU_p, pivots, LU_t, B_t, left, adjoint)

- name: lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)
  LU_data: lu_unpack_backward(grad_L, grad_U, LU_data.sym_size(-2), LU_data.sym_size(-1))
  LU_pivots: non_differentiable
  L: "LU_data_t.sym_size(-2) >= LU_data_t.sym_size(-1) ? LU_data_t.tril_symint(-1) : LU_data_t.narrow_symint(-1, 0, LU_data_t.sym_size(-2)).tril_symint(-1)"
  U: "LU_data_t.sym_size(-1) >= LU_data_t.sym_size(-2) ? LU_data_t.triu_symint() : LU_data_t.narrow_symint(-2, 0, LU_data_t.sym_size(-1)).triu_symint()"
  output_differentiability: [False, True, True]

- name: masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
  self: grad.masked_fill(mask, 0)
  mask: non_differentiable
  result: self_t.masked_fill(mask, 0)

- name: masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
  self: grad.masked_fill(mask, 0)
  value: masked_fill_backward(grad, mask)
  mask: non_differentiable
  result: self_t.masked_fill(mask, value_t)

- name: masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
  self: grad.masked_fill(mask, 0)
  source: masked_scatter_backward_symint(grad, mask, source.sym_sizes())
  mask: non_differentiable
  result: self_t.masked_scatter(mask, source_t)

- name: masked_scatter_backward(Tensor grad_output, Tensor mask, SymInt[] sizes) -> Tensor
  grad_output: zeros_like(grad_output).masked_scatter(mask, grad)
  mask: non_differentiable
  result: masked_scatter_backward(grad_output_t, mask, grad_output_t.sizes())

- name: masked_select(Tensor self, Tensor mask) -> Tensor
  self: masked_select_backward(grad, self, mask)
  mask: non_differentiable
  result: auto_linear

- name: linalg_matrix_exp(Tensor self) -> Tensor
  self: linalg_matrix_exp_differential(self, grad, /*adjoint*/ true)
  result: linalg_matrix_exp_differential(self_p, self_t, /*adjoint*/ false)

- name: max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), keepdim)
  values: gather_with_keepdimed_indices(self_t, dim, indices, keepdim)

- name: max(Tensor self) -> Tensor
  self: evenly_distribute_backward(grad, self, result)
  result: evenly_read_jvp(self_t, self_p, result)

- name: maximum(Tensor self, Tensor other) -> Tensor
  self: at::where(self == other, grad / 2, grad).masked_fill_(self < other, 0)
  other: at::where(self == other, grad / 2, grad).masked_fill_(self > other, 0)
  result: other_t + at::where(self_p == other_p, at::scalar_tensor(0.5, result.options()), (self_p > other_p).to(result.scalar_type())) * (self_t - other_t)

- name: fmax(Tensor self, Tensor other) -> Tensor
  self: grad.masked_fill((self >= other).logical_or_(other.isnan()).logical_not_(), 0)
  other: grad.masked_fill((self >= other).logical_or_(other.isnan()), 0)
  result: other_t + (self_p > other_p).logical_or_(other_p.isnan()) * (self_t - other_t)

- name: mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
  dispatch:
    Default:
      self: grad.expand_symint(self.sym_sizes()) / self.sym_numel()
      result: auto_linear
    AutogradNestedTensor:
      # TODO: replace this with grad.expand_as(self) / self.sym_numel() when that is supported
      self: (ones_like(self) * grad) / self.sym_numel()
      result: auto_linear

- name: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  self: mean_backward(grad, self.sym_sizes(), dim, self.sym_numel(), keepdim)
  result: auto_linear

- name: median(Tensor self) -> Tensor
  self: evenly_distribute_backward(grad, self, result)
  result: evenly_read_jvp(self_t, self_p, result)

- name: nanmedian(Tensor self) -> Tensor
  self: evenly_distribute_backward(grad, self, result)
  result: evenly_read_jvp(self_t, self_p, result)

# This is in theory incorrect in the following case:
#   sorted list: [..., a, b, b, ..., b, b, c, ...] with median = b and the value
#                            |                     at middle position of the
#                            |                     list between two `b`s. E.g.,
#                            |
#                            ^the middle position
# The gradient exists and is essentially 0 in this case.
#
# In case where the middle position is at the boundary of `b` range, e.g.,
#   sorted list: [..., a, b, b, ..., b, b, c, ...]
#                                       |
#                                       ^the middle position
# The backward implementation is correct in the sense that it returns the
# subgradient on one side.
- name: median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), keepdim)
  values: gather_with_keepdimed_indices(self_t, dim, indices, keepdim)

- name: nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), keepdim)
  values: gather_with_keepdimed_indices(self_t, dim, indices, keepdim)

- name: min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), keepdim)
  values: gather_with_keepdimed_indices(self_t, dim, indices, keepdim)

- name: min(Tensor self) -> Tensor
  self: evenly_distribute_backward(grad, self, result)
  result: evenly_read_jvp(self_t, self_p, result)

- name: minimum(Tensor self, Tensor other) -> Tensor
  self: at::where(self == other, grad / 2, grad).masked_fill_(self > other, 0)
  other: at::where(self == other, grad / 2, grad).masked_fill_(self < other, 0)
  result: other_t + at::where(self_p == other_p, at::scalar_tensor(0.5, result.options()), (self_p < other_p).to(result.scalar_type())) * (self_t - other_t)

- name: fmin(Tensor self, Tensor other) -> Tensor
  self: grad.masked_fill((self <= other).logical_or_(other.isnan()).logical_not_(), 0)
  other: grad.masked_fill((self <= other).logical_or_(other.isnan()), 0)
  result: other_t + (self_p <= other_p).logical_or_(other_p.isnan()) * (self_t - other_t)

- name: amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
  self: scale_grad_by_count(restore_reduced_dims(grad, dim, keepdim), restore_reduced_dims(result, dim, keepdim) == self, dim)
  result: amaxamin_jvp(self_p, self_t, result, dim, keepdim)

- name: amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
  self: scale_grad_by_count(restore_reduced_dims(grad, dim, keepdim), restore_reduced_dims(result, dim, keepdim) == self, dim)
  result: amaxamin_jvp(self_p, self_t, result, dim, keepdim)

- name: mm(Tensor self, Tensor mat2) -> Tensor
  self: mm_mat1_backward(grad, mat2, self.sym_sizes(), self.sym_strides(), self.layout(), 1)
  mat2: mm_mat2_backward(grad, self, mat2.sym_sizes(), mat2.sym_strides(), mat2.layout(), 1)
  result: at::mm(self_t, mat2_p) + at::mm(self_p, mat2_t)

- name: _grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor
  self: _grouped_mm_mat1_backward(grad, mat2, self.sym_sizes(), self.sym_strides(), self.layout(), offs, 1)
  mat2: _grouped_mm_mat2_backward(grad, self, mat2.sym_sizes(), mat2.sym_strides(), mat2.layout(), offs, 1)

- name: mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), keepdim)
  values: gather_with_keepdimed_indices(self_t, dim, indices, keepdim)

- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: mul_tensor_backward(grad, other, self.scalar_type())
  other: mul_tensor_backward(grad, self, other.scalar_type())
  result: other_t * self_p + self_t * other_p

- name: mul.Scalar(Tensor self, Scalar other) -> Tensor
  self: mul_tensor_backward(grad, other, self.scalar_type())
  result: self_t * other

- name: mv(Tensor self, Tensor vec) -> Tensor
  self: grad.ger(vec.conj())
  vec: self.conj().t().mv(grad)
  result: mv(self_t, vec_p) + mv(self_p, vec_t)

- name: mvlgamma(Tensor self, int p) -> Tensor
  self: mvlgamma_backward(grad, self, p)
  result: auto_element_wise

- name: nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
  self: grad * at::isfinite(self)
  result: auto_element_wise

- name: native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
  input, weight, bias: "grad.defined() ? native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, training, eps, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>()"
  result0: batch_norm_jvp(input_p, input_t, weight_p, weight_t, bias_p, bias_t, running_mean, running_var, result1, result2, training, eps)

- name: _native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
  input, weight, bias: "grad.defined() ? native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, training, eps, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>()"
  result0: batch_norm_jvp(input_p, input_t, weight_p, weight_t, bias_p, bias_t, running_mean, running_var, result1, result2, training, eps)

- name: _native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)
  input, weight, bias: "grad.defined() ? native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, /*training=*/false, eps, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>()"
  result0: batch_norm_jvp(input_p, input_t, weight_p, weight_t, bias_p, bias_t, running_mean, running_var, result1, result2, /*training=*/false, eps)

- name: _native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
  input, weight, bias: "grad.defined() ? native_batch_norm_backward(grad, input, weight, Tensor(), Tensor(), result1, result2, training, eps, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>()"
  result0: batch_norm_jvp(input_p, input_t, weight_p, weight_t, bias_p, bias_t, Tensor(), Tensor(), result1, result2, training, eps)

- name: native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
  input, weight, grad_out: batchnorm_double_backward(input, weight, grads[0], grads[1], grads[2], grad_out, running_mean, running_var, train, eps, save_mean, save_invstd, grad_input_mask)
  save_mean: not_implemented("native_batch_norm_backward save_mean")
  save_invstd: not_implemented("native_batch_norm_backward save_invstd")

- name: native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
  input, weight, bias: "grad.defined() ? native_layer_norm_backward_symint(grad, input, normalized_shape, result1, result2, weight, bias, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>()"
  result0: layer_norm_jvp(input_p, input_t, weight_p, weight_t, bias_p, bias_t, result1, result2, normalized_shape)

- name: native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
  input, weight, grad_out: layer_norm_double_backward(input, weight, grads[0], grads[1], grads[2], grad_out, mean, rstd, normalized_shape, grad_input_mask)
  bias: Tensor()
  mean: not_implemented("native_layer_norm_backward mean")
  rstd: not_implemented("native_layer_norm_backward rstd")

- name: _fused_rms_norm(Tensor input, int[] normalized_shape, Tensor? weight, float? eps) -> (Tensor, Tensor)
  input, weight: "GradMode::is_enabled() || grads[1].defined() ? infinitely_differentiable_native_rms_norm_backward(grads[0], grads[1], input, normalized_shape, result1, weight, grad_input_mask) : (grads[0].defined() ? _fused_rms_norm_backward(grads[0], input, normalized_shape, result1, weight, grad_input_mask) : std::tuple<Tensor, Tensor>())"
  result0: rms_norm_jvp(input_p, input_t, weight_p, weight_t, result1, normalized_shape)
  result1: rms_norm_rstd_jvp(input_p, input_t, result1, normalized_shape)

- name: native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
  input, weight, bias: "GradMode::is_enabled() || grads[1].defined() || grads[2].defined() ? infinitely_differentiable_native_group_norm_backward(grads[0], grads[1], grads[2], input, result1, result2, weight, N, C, HxW, group, eps, grad_input_mask) : (grads[0].defined() ? native_group_norm_backward_symint(grads[0].device().is_xpu() ? grads[0] : grads[0].contiguous(grads[0].device().is_cpu() ? input.suggest_memory_format() : c10::MemoryFormat::Contiguous), input.device().is_xpu() ? input : input.contiguous(input.device().is_cpu() ? input.suggest_memory_format() : c10::MemoryFormat::Contiguous), result1, result2, weight, N, C, HxW, group, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>())"
  result0: group_norm_jvp(input_p, input_t, weight_p, weight_t, bias_p, bias_t, result1, result2, group)
  result1: group_norm_mean_jvp(input_t, result1, group)
  result2: group_norm_invstd_jvp(input_p, input_t, result1, result2, group)

- name: ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
  self: zeros_like(self)
  result: self_t.zero_()

- name: ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  self: zeros_like(self)
  other: zeros_like(other)
  result: self_t.zero_()

- name: neg(Tensor self) -> Tensor
  self: grad.neg()
  result: auto_element_wise

- name: _batch_norm_with_update(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)
  input, weight, bias: "grad.defined() ? batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, /*update*/true, eps, grad_input_mask, retain_variables ? result3.clone() : result3) : std::tuple<Tensor, Tensor, Tensor>()"
  result0: batch_norm_jvp(input_p, input_t, weight_p, weight_t, bias_p, bias_t, running_mean, running_var, result1, result2, true, eps)

- name: _batch_norm_no_update(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)
  input, weight, bias: "grad.defined() ? batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, /*update*/false, eps, grad_input_mask, retain_variables ? result3.clone() : result3) : std::tuple<Tensor, Tensor, Tensor>()"
  result0: batch_norm_jvp(input_p, input_t, weight_p, weight_t, bias_p, bias_t, running_mean, running_var, result1, result2, false, eps)

- name: batch_norm_backward(Tensor grad_out, Tensor input, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, bool update, float eps, bool[3] output_mask, Tensor reserve) -> (Tensor, Tensor, Tensor)
  input, weight, grad_out: batchnorm_double_backward(input, weight, grads[0], grads[1], grads[2], grad_out, running_mean, running_var, update, eps, save_mean, save_var, grad_input_mask)
  save_mean: not_implemented("batch_norm_backward save_mean")
  save_var: not_implemented("batch_norm_backward save_var")
  reserve: not_implemented("batch_norm_backward reserve")

- name: nextafter(Tensor self, Tensor other) -> Tensor
  self: not_implemented("nextafter")
  other: not_implemented("nextafter")

- name: norm.Scalar(Tensor self, Scalar p=2) -> Tensor
  self: norm_backward(grad, self, p, result)
  result: norm_jvp(self_p, self_t, p, result)

- name: norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
  self: norm_backward(grad, self, p, result, dim, keepdim)
  result: norm_jvp(self_p, self_t, p, result, dim, keepdim)

- name: norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
  self: norm_backward(grad, self.to(grad.scalar_type()), p, result)
  result: norm_jvp(self_p, self_t, p, result)

- name: norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
  self: norm_backward(grad, self.to(grad.scalar_type()), p, result, dim, keepdim)
  result: norm_jvp(self_p, self_t, p, result, dim, keepdim)

- name: linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  self: linalg_vector_norm_backward(grad, self, ord, result, dim, keepdim)
  result: linalg_vector_norm_jvp(self_p, self_t, ord, result, dim, keepdim)

- name: _pdist_forward(Tensor self, float p=2) -> Tensor
  self: _pdist_backward(grad, self, p, result)

- name: _pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor
  grad: not_implemented("_pdist_backward")
  self: not_implemented("_pdist_backward")
  pdist: not_implemented("_pdist_backward")

- name: _euclidean_dist(Tensor x1, Tensor x2) -> Tensor
  x1, x2: _euclidean_dist_backward(grad, x1, x2, result)

- name: _cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor
  x1: _cdist_backward(grad.contiguous(), x1, x2, p, result)
  x2: _cdist_backward(grad.mT().contiguous(), x2, x1, p, result.mT().contiguous())

- name: _cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor
  grad: not_implemented("_cdist_backward")
  x1: not_implemented("_cdist_backward")
  x2: not_implemented("_cdist_backward")
  cdist: not_implemented("_cdist_backward")

- name: normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.zero_()

- name: normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor
  mean: at::zeros_symint(mean.sym_sizes(), grad.options())
  result: auto_element_wise

- name: normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor
  std: at::zeros_symint(std.sym_sizes(), grad.options())
  result: auto_element_wise

- name: normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor
  mean: at::zeros_symint(mean.sym_sizes(), grad.options())
  std: at::zeros_symint(std.sym_sizes(), grad.options())
  result: zeros_like(mean_t)

- name: linalg_householder_product(Tensor input, Tensor tau) -> Tensor
  input, tau: householder_product_backward(grad, result, input, tau)
  result: householder_product_jvp(input_t, tau_t, result, input_p, tau_p)

- name: ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
  self, input2, input3: ormqr_backward(grad, result, self, input2, input3, left, transpose, grad_input_mask)

- name: permute(Tensor(a) self, int[] dims) -> Tensor(a)
  self: permute_backwards(grad, dims)
  result: auto_linear

- name: poisson(Tensor self, Generator? generator=None) -> Tensor
  self: zeros_like(self)
  result: auto_element_wise

- name: pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
  self: pow_backward(grad, self, exponent)
  result: auto_element_wise

- name: pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
  self: pow_backward_self(grad, self, exponent)
  exponent: pow_backward_exponent(grad, self, exponent, result)
  result: (pow_backward_self(self_t.conj(), self_p, exponent_p) + pow_backward_exponent(exponent_t.conj(), self_p, exponent_p, result)).conj()

- name: pow.Scalar(Scalar self, Tensor exponent) -> Tensor
  exponent: pow_backward_exponent(grad, self, exponent, result)
  result: auto_element_wise

- name: prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
  self: prod_backward(grad, self.to(grad.scalar_type()), result)
  result: (prod_backward(at::ones({}, result.options()).expand_as(result), self_p.to(result.scalar_type()), result) * self_t.conj()).sum().conj()

- name: prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  self: prod_backward(grad, self.to(grad.scalar_type()), result, dim, keepdim)
  result: (prod_backward(at::ones({}, result.options()).expand_as(result), self_p.to(result.scalar_type()), result, dim, keepdim) * self_t.conj()).sum(dim, keepdim).conj()

- name: put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor
  self: "accumulate ? grad : grad.put(index, zeros_like(source), false)"
  index: non_differentiable
  source: grad.take(index).reshape_as(source)
  result: self_t.put(index, source_t, accumulate)

- name: linalg_qr(Tensor A, str mode='reduced') -> (Tensor Q, Tensor R)
  A: linalg_qr_backward(grad_Q, grad_R, Q, R, mode)
  Q, R: linalg_qr_jvp(A_t, Q, R, mode)

- name: rad2deg(Tensor self) -> Tensor
  self: rad2deg_backward(grad)
  result: auto_element_wise

- name: random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.zero_()

- name: random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.zero_()

- name: random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.zero_()

- name: reciprocal(Tensor self) -> Tensor
  self: -grad * (result * result).conj()
  result: auto_element_wise

- name: remainder.Scalar(Tensor self, Scalar other) -> Tensor
  self: grad
  result: auto_element_wise

- name: remainder.Tensor(Tensor self, Tensor other) -> Tensor
  self: grad
  other: -grad * self.div(other, /*rounding_mode=*/"floor")
  result: self_t - other_t * self_p.div(other_p, /*rounding_mode=*/"floor")

- name: renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
  self: renorm_backward(grad, self, p, dim, maxnorm)
  result: renorm_jvp(self_p, self_t, p, dim, maxnorm)

- name: repeat(Tensor self, SymInt[] repeats) -> Tensor
  self: repeat_backward(grad, repeats, self.sym_sizes())
  result: auto_linear

- name: special_entr(Tensor self) -> Tensor
  self: grad * (-(1 + self.log()))
  result: auto_element_wise

- name: special_ndtri(Tensor self) -> Tensor
  self: grad * std::sqrt(2 * M_PI) * (result.square() / 2).exp()
  result: auto_element_wise

- name: special_log_ndtr(Tensor self) -> Tensor
  self: grad / std::sqrt(2 * M_PI) * (result + self.pow(2) / 2).neg().exp()
  result: auto_element_wise

# [Note: Sometimes view derivatives]
# The following situation applies to other operations as well.
# TODO: This note is only referenced by to_dense and to_sparse*. Make
# this more generic if it's been referenced more than once.
#
# DO NOT define a backward for reshape!
# reshape is special in that it sometimes returns a view, and sometimes not.
# Defining a backward will make codegen spit out the forward call as
#     as_variable(baseType->reshape(self)),
# making it impossible (hard) to detect when it is actually a view.
# - name: reshape(Tensor self, IntArrayRef shape)

- name: _reshape_alias(Tensor(a) self, SymInt[] size, SymInt[] stride) -> Tensor(a)
  self: grad.reshape_symint(self.sym_sizes())
  result: auto_linear

- name: round(Tensor self) -> Tensor
  self: zeros_like(grad)
  result: auto_element_wise

- name: round.decimals(Tensor self, *, int decimals) -> Tensor
  self: zeros_like(grad)
  result: auto_element_wise

- name: rsqrt(Tensor self) -> Tensor
  self: -0.5 * grad * result.pow(3).conj()
  result: auto_element_wise

- name: scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
  self: grad.scatter(dim, index, 0)
  index: non_differentiable
  src: grad.gather(dim, index)
  result: self_t.scatter(dim, index, src_t)

- name: scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
  self: grad.scatter(dim, index, 0)
  index: non_differentiable
  result: self_t.scatter(dim, index, 0)

- name: scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
  self: grad
  index: non_differentiable
  src: grad.gather(dim, index)
  result: scatter_add(self_t, dim, index, src_t)

- name: select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
  dispatch:
    Default:
      self: select_backward_symint(grad, self.sym_sizes(), dim, index)
      result: auto_linear
    AutogradNestedTensor:
      self: _nested_select_backward_symint(grad, self, dim, index)

- name: select_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt index) -> Tensor
  grad_output: grad.select_symint(dim, index)
  result: auto_linear

- name: sigmoid(Tensor self) -> Tensor
  self: sigmoid_backward(grad, result)
  result: auto_element_wise

- name: logit(Tensor self, float? eps=None) -> Tensor
  self: "GradMode::is_enabled() ? infinitely_differentiable_logit_backward(grad, self, eps) : logit_backward(grad, self, eps)"
  result: auto_element_wise

- name: sign(Tensor self) -> Tensor
  self: zeros_like(grad)
  result: auto_element_wise

- name: sgn(Tensor self) -> Tensor
  self: sgn_backward(self, grad, result)
  # Cannot use auto_element_wise here because the Jacobian is *not* Hermitian (in fact, it is symmetric)
  # The function is not holomorphic, so there's no reason for its Jacobian to be Hermitian
  # auto_element_wise has a name that's a bit deceiving in the complex case
  result: sgn_backward(self_p, self_t, result)

- name: sin(Tensor self) -> Tensor
  self: grad * self.cos().conj()
  result: auto_element_wise

- name: sinc(Tensor self) -> Tensor
  self: sinc_backward(grad, self)
  result: auto_element_wise

- name: sinh(Tensor self) -> Tensor
  self: grad * self.cosh().conj()
  result: auto_element_wise

- name: slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
  self: slice_backward_wrapper(grad, self.sym_sizes(), dim, start, end, step)
  result: auto_linear

- name: slice_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step) -> Tensor
  grad_output: grad.slice_symint(dim, start, end, step)
  result: auto_linear

- name: slice_inverse(Tensor(a) self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
  self: grad.slice_symint(dim, start, end, step)
  src: slice_scatter_symint(grad, zeros_like(self), dim, start, end, step)
  result: auto_linear

- name: slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
  self: slice_scatter_symint(grad, zeros_like(src), dim, start, end, step)
  src: grad.slice_symint(dim, start, end, step)
  result: auto_linear

- name: select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor
  self: select_scatter_symint(grad, zeros_like(src), dim, index)
  src: grad.select_symint(dim, index)
  result: auto_linear

- name: diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> Tensor
  self: diagonal_scatter(grad, zeros_like(src), offset, dim1, dim2)
  src: grad.diagonal(offset, dim1, dim2)
  result: auto_linear

- name: as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor
  self: as_strided_scatter_backward(grad, TensorGeometry(self), TensorGeometry(src), size, stride, storage_offset)
  # See Note [as_strided_scatter backward support]
  src: grad.contiguous().as_strided_symint(size, stride, storage_offset)
  result: auto_linear

- name: _linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False) -> (Tensor result, Tensor LU, Tensor pivots, Tensor info)
  A, B: linalg_solve_backward(grad, result, A, LU, pivots, left, grad_input_mask[1])
  result: "linalg_solve_jvp(A_t, B_t, result, LU, pivots, left, A_p.is_contiguous() && !A_p.is_complex())"
  output_differentiability: [True, False, False, False]  # LU is an auxiliary tensor not exposed to the user

- name: sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), true)
  output_differentiability: [True, False]
  values: gather_with_keepdimed_indices(self_t, dim, indices, true)

- name: sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), true)
  output_differentiability: [True, False]
  values: gather_with_keepdimed_indices(self_t, dim, indices, true)

- name: split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]
  self: split_backward(grads, split_size, dim, self.sym_sizes(), self.options())
  result: auto_linear

- name: unsafe_split.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]
  self: split_backward(grads, split_size, dim, self.sym_sizes(), self.options())
  result: auto_linear

- name: split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
  dispatch:
    Default:
      self: split_with_sizes_backward(grads, split_sizes, dim, self.sym_sizes(), self.options())
      result: auto_linear
    AutogradNestedTensor:
      self: _nested_split_with_sizes_backward(grads, split_sizes, dim, at::native::get_nested_tensor_impl(self)->get_nested_sizes(), self.options())

- name: unsafe_split_with_sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]
  self: split_with_sizes_backward(grads, split_sizes, dim, self.sym_sizes(), self.options())
  result: auto_linear

- name: sqrt(Tensor self) -> Tensor
  self: grad / (2 * result.conj())
  result: auto_element_wise

- name: squeeze(Tensor(a) self) -> Tensor(a)
  self: unsqueeze_to(grad, self.sym_sizes())
  result: auto_linear

- name: squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
  dispatch:
    Default:
      self: unsqueeze_to(grad, dim, self.sym_sizes())
      result: auto_linear
    AutogradNestedTensor:
      self: grad.unsqueeze(dim)

- name: squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
  dispatch:
    Default:
      self: unsqueeze_to(grad, dim, self.sym_sizes())
      result: auto_linear
    AutogradNestedTensor:
      self: unsqueeze_multiple(grad, dim, self.dim())

- name: squeeze_(Tensor(a!) self) -> Tensor(a!)
  self: unsqueeze_to(grad, self.sym_sizes())
  result: auto_linear

- name: squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)
  self: unsqueeze_to(grad, dim, self.sym_sizes())
  result: auto_linear

- name: squeeze_.dims(Tensor(a!) self, int[] dim) -> Tensor(a!)
  self: unsqueeze_to(grad, dim, self.sym_sizes())
  result: auto_linear

- name: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
  self: std_backward(result, grad, self, dim, correction, keepdim)
  # pointwise (variance) + sum + sqrt
  result: (at::real(var_backward(self_t.conj(), self_p, dim, correction, true).sum(dim.value_or(IntArrayRef({})), keepdim)) / (2. * result)).masked_fill_(result == 0, 0)

- name: std_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
  self: std_mean_backward(grads[0], grads[1], self, result0, dim, correction, keepdim)
  result0: (at::real(var_backward(self_t.conj(), self_p, dim, correction, true).sum(dim.value_or(IntArrayRef({})), keepdim)) / (2. * result0)).masked_fill_(result0 == 0, 0)
  # linear
  result1: mean(self_t, dim.value_or(IntArrayRef({})), keepdim)

- name: sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  other: handle_r_to_c(other.scalar_type(), maybe_multiply(-grad, alpha.conj()))
  result: self_t - maybe_multiply(other_t, alpha)

- name: sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  result: auto_element_wise

- name: rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), maybe_multiply(-grad, alpha.conj()))
  other: handle_r_to_c(other.scalar_type(), grad)
  result: -maybe_multiply(self_t, alpha) + other_t

- name: rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), maybe_multiply(-grad, alpha.conj()))
  result: auto_element_wise

- name: sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
  dispatch:
    Default:
      self: grad.expand_symint(self.sym_sizes())
      result: auto_linear
    AutogradNestedTensor:
      # TODO: replace this with grad.expand_as(self) when that is supported
      self: ones_like(self) * grad
      result: auto_linear

- name: sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  dispatch:
    Default:
      self: sum_backward(grad, self.sym_sizes(), dim, keepdim)
      result: auto_linear
    AutogradNestedTensor:
      # TODO: replace this function once semantics for nested tensor expand have been settled on
      self: _nested_sum_backward(grad, self, dim, keepdim)

- name: nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  self: nansum_backward(grad.to(self.scalar_type()), self, dim, keepdim)
  result: at::where(self_p.isnan(), 0, self_t).sum(dim, keepdim, dtype)

# We never call _linalg_svd with compute_uv=False in an autograd context, so we don't even consider it here
- name: _linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)
  A: "svd_backward(full_matrices && grad_U.defined() ? grad_U.narrow_symint(-1, 0, S.sym_size(-1)) : grad_U,
                   grad_S,
                   full_matrices && grad_Vh.defined() ? grad_Vh.narrow_symint(-2, 0, S.sym_size(-1)) : grad_Vh,
                   full_matrices ? U.narrow_symint(-1, 0, S.sym_size(-1)) : U,
                   S,
                   full_matrices ? Vh.narrow_symint(-2, 0, S.sym_size(-1)) : Vh)"
  U, S, Vh: linalg_svd_jvp(A_t, U, S, Vh, full_matrices)

- name: _linalg_eigh(Tensor A, str UPLO="L", bool compute_v=True) -> (Tensor eigenvalues, Tensor eigenvectors)
  A: linalg_eig_backward(grads[0], grads[1], eigenvalues, eigenvectors, /*is_hermitian=*/true)
  eigenvalues, eigenvectors: linalg_eig_jvp(A_t, eigenvalues, eigenvectors, /*is_hermitian=*/true)

- name: linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
  self: handle_r_to_c(self.scalar_type(), linalg_eig_backward(grads[0], grads[1], eigenvalues, eigenvectors, /*is_hermitian=*/false))
  eigenvalues, eigenvectors: linalg_eig_jvp(self_t, eigenvalues, eigenvectors, /*is_hermitian=*/false)

- name: t(Tensor(a) self) -> Tensor(a)
  self: grad.t()
  result: auto_linear

- name: t_(Tensor(a!) self) -> Tensor(a!)
  self: grad.t()
  result: auto_linear

- name: one_hot(Tensor self, int num_classes=-1) -> Tensor
  self: non_differentiable

- name: flip(Tensor self, int[] dims) -> Tensor
  self: grad.flip(dims)
  result: auto_linear

- name: roll(Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor
  self: grad.roll_symint(fmap(reverse_list_symint(shifts), [](c10::SymInt i){return -i;}), reverse_list(dims))
  result: auto_linear

- name: rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
  self: grad.rot90(-k, dims)
  result: auto_linear

- name: take(Tensor self, Tensor index) -> Tensor
  self: take_backward(grad, self, index)
  index: non_differentiable
  result: auto_linear

- name: tan(Tensor self) -> Tensor
  self: grad * (1 + result.pow(2)).conj()
  result: auto_element_wise

- name: tanh(Tensor self) -> Tensor
  self: tanh_backward(grad, result)
  result: auto_element_wise

- name: topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), true)
  output_differentiability: [True, False]
  values: gather(self_t, dim, indices)

- name: trace(Tensor self) -> Tensor
  self: trace_backward_symint(grad, self.sym_sizes())
  result: auto_linear

- name: transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
  self: grad.transpose(dim0, dim1)
  result: auto_linear

- name: transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
  self: grad.transpose(dim0, dim1)
  result: auto_linear

- name: triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
  self, A: triangular_solve_backward(grad_solution, grad_cloned_coefficient, self, A, solution, upper, transpose, unitriangular, grad_input_mask)
  solution: triangular_solve_jvp(solution, A_p, A_t, self_t, upper, transpose, unitriangular)
  cloned_coefficient: A_t

- name: linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor
  self, B: linalg_solve_triangular_backward(grad, self, result, upper, left, unitriangular, grad_input_mask)
  result: linalg_solve_triangular_forward_AD(self_t, B_t, self_p, result, upper, left, unitriangular)

- name: tril(Tensor self, SymInt diagonal=0) -> Tensor
  self: grad.tril_symint(diagonal)
  result: auto_linear

- name: triu(Tensor self, SymInt diagonal=0) -> Tensor
  self: grad.triu_symint(diagonal)
  result: auto_linear

- name: trunc(Tensor self) -> Tensor
  self: zeros_like(grad)
  result: auto_element_wise

- name: hash_tensor(Tensor self, int[1] dim=[], *, bool keepdim=False, int mode=0) -> Tensor
  output_differentiability: [False]

# DO NOT define a backward for to_dense
# See [Note: Sometimes view derivatives]
# - name: to_dense(Tensor self, ScalarType? dtype=None, *, bool? masked_grad=None) -> Tensor
#
- name: _to_dense(Tensor self, ScalarType? dtype=None, bool? masked_grad=None) -> Tensor
  self: to_dense_backward(grad, self, masked_grad)

# DO NOT define a backward for to_sparse.sparse_dim
# See [Note: Sometimes view derivatives]
# - name: to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
#
- name: _to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
  self: to_sparse_backward(grad, self.layout(), self.sym_blocksize())

# DO NOT define a backward for to_sparse
# See [Note: Sometimes view derivatives]
# - name: to_sparse(Tensor self, *, Layout? layout=None, int[2]? blocksize=None, int? dense_dim=None) -> Tensor
#
- name: _to_sparse(Tensor self, *, Layout? layout=None, int[2]? blocksize=None, int? dense_dim=None) -> Tensor
  self: to_sparse_backward(grad, self.layout(), self.sym_blocksize())

# DO NOT define a backward for to_sparse_csr
# See [Note: Sometimes view derivatives]
# - name: to_sparse_csr(Tensor self, int? dense_dim=None) -> Tensor
#
- name: _to_sparse_csr(Tensor self, int? dense_dim=None) -> Tensor
  self: to_sparse_backward(grad, self.layout(), self.sym_blocksize())

# DO NOT define a backward for to_sparse_csc
# See [Note: Sometimes view derivatives]
# - name: to_sparse_csc(Tensor self, int? dense_dim=None) -> Tensor
#
- name: _to_sparse_csc(Tensor self, int? dense_dim=None) -> Tensor
  self: to_sparse_backward(grad, self.layout(), self.sym_blocksize())

# DO NOT define a backward for to_sparse_bsr
# See [Note: Sometimes view derivatives]
# - name: to_sparse_bsr(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
#
- name: _to_sparse_bsr(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
  self: to_sparse_backward(grad, self.layout(), self.sym_blocksize())

# DO NOT define a backward for to_sparse_bsc
# See [Note: Sometimes view derivatives]
# - name: to_sparse_bsc(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
#
- name: _to_sparse_bsc(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
  self: to_sparse_backward(grad, self.layout(), self.sym_blocksize())

- name: to_mkldnn(Tensor self, ScalarType? dtype=None) -> Tensor
  self: to_mkldnn_backward(grad, self)

- name: unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
  self: unfold_backward_symint(grad, self.sym_sizes(), dimension, size, step)
  result: auto_linear

- name: unfold_backward(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step) -> Tensor
  grad_in: grad.unfold(dim, size, step)
  result: auto_linear

- name: uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)
  self: zeros_like(grad)
  result: self_t.zero_()

- name: _unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)
  output_differentiability: [True, False]
  self: not_implemented("_unique")

- name: unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
  output_differentiability: [True, False, False]
  self: not_implemented("unique_dim")

- name: unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)
  output_differentiability: [True, False, False]
  self: not_implemented("unique_consecutive")

- name: unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
  output_differentiability: [True, False, False]
  self: not_implemented("unique_dim_consecutive")

- name: _unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
  output_differentiability: [True, False, False]
  self: not_implemented("_unique2")

- name: _unsafe_view(Tensor self, SymInt[] size) -> Tensor
  self: grad.reshape_symint(self.sym_sizes())
  result: auto_linear

- name: lift(Tensor self) -> Tensor
  self: grad
  result: auto_linear

- name: lift_fresh(Tensor(a) self) -> Tensor(a)
  self: grad
  result: auto_linear

- name: unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
  self: grad.squeeze(dim)
  result: auto_linear

- name: unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
  self: grad.squeeze(dim)
  result: auto_linear

- name: var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
  self: var_backward(grad, self, dim, correction, keepdim)
  # pointwise + sum
  result: at::real(var_backward(self_t.conj(), self_p, dim, correction, true).sum(dim.value_or(IntArrayRef({})), keepdim))

- name: var_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
  self: var_mean_backward(grads[0], grads[1], self, dim, correction, keepdim)
  result0: at::real(var_backward(self_t.conj(), self_p, dim, correction, true).sum(dim.value_or(IntArrayRef({})), keepdim))
  # linear
  result1: mean(self_t, dim.value_or(IntArrayRef({})), keepdim)

- name: view(Tensor(a) self, SymInt[] size) -> Tensor(a)
  dispatch:
    Default:
      self: grad.reshape_symint(self.sym_sizes())
      result: auto_linear
    AutogradNestedTensor:
      self: grad.reshape_as(self)
      result: auto_linear

- name: view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)
  output_differentiability: [False]

- name: view_as_real(Tensor(a) self) -> Tensor(a)
  self: at::view_as_complex(grad.contiguous()) # gx0 + 1j * gx1
  result: at::view_as_real(self_t)

- name: view_as_complex(Tensor(a) self) -> Tensor(a)
  self: at::view_as_real(grad.contiguous().resolve_conj()) # [gx, gy]
  result: at::view_as_complex(self_t)

- name: where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
  condition: non_differentiable
  self: where(condition, grad, 0)
  other: where(condition, 0, grad)
  result: where(condition, self_t, other_t)

# weight_norm_cuda_interface_backward does not have an explicitly defined derivative, so if we do happen
# to be running backward with create_graph=True, fall back to a backward function that uses
# differentiable ops.
- name: _weight_norm_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)
  v, g: "grad.defined() ? (GradMode::is_enabled() ? _weight_norm_differentiable_backward(grad.contiguous(), v, g, result1, dim) : _weight_norm_interface_backward(grad.contiguous(), v, g, result1, dim)) : std::tuple<Tensor, Tensor>()"

- name: zero_(Tensor(a!) self) -> Tensor(a!)
  self: zeros_like(grad)
  result: auto_linear

- name: sparse_mask(Tensor self, Tensor mask) -> Tensor
  self: sparse_mask_backward(grad, mask, self.layout())
  mask: non_differentiable

- name: _sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, SymInt[] size, Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? is_coalesced=None) -> Tensor
  indices: non_differentiable
  values: grad.sparse_mask(result)._values()

- name: sparse_compressed_tensor.comp_plain_value_size(Tensor compressed_indices, Tensor plain_indices, Tensor values, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
  compressed_indices: non_differentiable
  plain_indices: non_differentiable
  # TODO: remove to_dense after gh-107381 is fixed
  values: grad.to_dense().sparse_mask(result).values()

- name: _sparse_sum.dim(Tensor self, int[1] dim) -> Tensor
  self: at::_sparse_sum_backward(grad, self, dim)

- name: _standard_gamma(Tensor self, Generator? generator=None) -> Tensor
  self: grad * _standard_gamma_grad(self, result)

- name: _standard_gamma_grad(Tensor self, Tensor output) -> Tensor
  self: not_implemented("_standard_gamma_grad")

- name: values(Tensor(a) self) -> Tensor(a)
  dispatch:
    Default:
      self: values_backward(grad, self)
    AutogradNestedTensor:
      self: at::_nested_view_from_buffer(grad.contiguous(), self._nested_tensor_size(), self._nested_tensor_strides(), self._nested_tensor_storage_offsets())

# Why is _values() not differentiable?
# See NOTE [ Sparse: autograd and API ]
- name: _values(Tensor(a) self) -> Tensor(a)
  output_differentiability: [False]

# NN
- name: _trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor
  i1, i2, i3: "_trilinear_backward(grad,
               wrap_opt_if(i1, grad_input_mask[1] || grad_input_mask[2]),
               wrap_opt_if(i2, grad_input_mask[0] || grad_input_mask[2]),
               wrap_opt_if(i3, grad_input_mask[0] || grad_input_mask[1]),
               expand1, expand2, expand3, sumdim, grad_input_mask)"
  result: "_trilinear(i1_t, i2_p, i3_p, expand1, expand2, expand3, sumdim, unroll_dim) +
           _trilinear(i1_p, i2_t, i3_p, expand1, expand2, expand3, sumdim, unroll_dim) +
           _trilinear(i1_p, i2_p, i3_t, expand1, expand2, expand3, sumdim, unroll_dim)"

- name: constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
  self: constant_pad_nd_backward(grad, pad)
  result: constant_pad_nd_symint(self_t, pad, 0)

- name: binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
  self: binary_cross_entropy_backward(grad, self, target, weight, reduction)
  target: binary_cross_entropy_target_backward(grad, self, target, weight, reduction)
  result: "apply_loss_reduction(
               binary_cross_entropy_backward(self_t, self_p, target_p, weight, at::Reduction::None)
             + binary_cross_entropy_target_backward(target_t, self_p, target_p, weight, at::Reduction::None),
           reduction)"

- name: binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
  self: binary_cross_entropy_double_backward(grad_output, grad, self, target, weight, reduction)
  target: binary_cross_entropy_double_backward_target(grad, grad_output, self, target, weight, reduction)
  grad_output: binary_cross_entropy_double_backward_grad_output(grad, self, target, weight, reduction)
  result: " binary_cross_entropy_double_backward(grad_output_p, self_t, self_p, target_p, weight, reduction)
          + binary_cross_entropy_double_backward_target(target_t, grad_output_p, self_p, target_p, weight, reduction)
          + binary_cross_entropy_double_backward_grad_output(grad_output_t, self_p, target_p, weight,

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a configuration file.

## Detailed Walkthrough


## Key Components

The file contains 18919 words across 3242 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 182902 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
