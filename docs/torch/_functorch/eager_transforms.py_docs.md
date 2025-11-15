# Documentation: `torch/_functorch/eager_transforms.py`

## File Metadata

- **Path**: `torch/_functorch/eager_transforms.py`
- **Size**: 71,071 bytes (69.41 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from collections.abc import Callable
from functools import partial, wraps
from typing import Any, Optional, Union

import torch
import torch.autograd.forward_ad as fwAD
from torch._C._functorch import (
    _assert_wrapped_functional,
    _func_decrement_nesting,
    _func_increment_nesting,
    _grad_decrement_nesting,
    _grad_increment_nesting,
    _jvp_decrement_nesting,
    _jvp_increment_nesting,
    _propagate_functional_input_mutation,
    _unwrap_for_grad,
    _unwrap_functional_tensor,
    _wrap_for_grad,
    _wrap_functional_tensor,
    get_inplace_requires_grad_allowed,
    get_unwrapped,
    is_functorch_wrapped_tensor,
    set_inplace_requires_grad_allowed,
)
from torch._functorch.utils import argnums_t, exposed_in
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import (
    tree_flatten,
    tree_map,
    tree_map_,
    tree_map_only,
    tree_unflatten,
    treespec_pprint,
)

from .apis import vmap
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes


def lazy_dynamo_disallow(func):
    import torch._dynamo

    return torch._dynamo.disallow_in_graph(func)


@contextlib.contextmanager
def enable_inplace_requires_grad(enabled):
    prev_state = get_inplace_requires_grad_allowed()
    set_inplace_requires_grad_allowed(enabled)
    try:
        yield
    finally:
        set_inplace_requires_grad_allowed(prev_state)


def _set_tensor_requires_grad(x):
    # avoid graph-break on x.requires_grad_()
    # https://github.com/pytorch/pytorch/pull/110053
    return x.requires_grad_()


def _create_differentiable(inps, level=None):
    def create_differentiable(x):
        if isinstance(x, torch.Tensor):
            with enable_inplace_requires_grad(True):
                return _set_tensor_requires_grad(x)
        raise ValueError(f"Thing passed to transform API must be Tensor, got {type(x)}")

    return tree_map(create_differentiable, inps)


def _undo_create_differentiable(inps, level=None):
    def unwrap_tensors(x):
        if isinstance(x, torch.Tensor):
            return _unwrap_for_grad(x, level)
        # TODO: Remove the following hack for namedtuples
        if isinstance(x, tuple):
            return tree_map(unwrap_tensors, tuple(x))

        raise RuntimeError(f"Expected tensors, got unsupported type {type(x)}")

    return tree_map(unwrap_tensors, inps)


def _is_differentiable(maybe_tensor):
    if not isinstance(maybe_tensor, torch.Tensor):
        return False
    return maybe_tensor.requires_grad


def _any_differentiable(tensor_or_tuple_of_tensors):
    flat_args, _ = tree_unflatten(tensor_or_tuple_of_tensors)
    return any(tuple(map(_is_differentiable, flat_args)))


def _wrap_tensor_for_grad(maybe_tensor, level):
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    return _wrap_for_grad(maybe_tensor, level)


def _wrap_all_tensors(tensor_pytree, level):
    return tree_map(partial(_wrap_tensor_for_grad, level=level), tensor_pytree)


def _as_tuple(val):
    if isinstance(val, tuple):
        return val
    return (val,)


# Version of autograd.grad that handles outputs that don't depend on inputs


def _autograd_grad(
    outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
):
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        result = tuple(
            (out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad
        )
        if len(result) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*result)
    if len(diff_outputs) == 0:
        return tuple(torch.zeros_like(inp) for inp in inputs)
    with torch._dynamo.compiled_autograd._disable():
        grad_inputs = torch.autograd.grad(
            diff_outputs,
            inputs,
            grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
    grad_inputs = tuple(
        torch.zeros_like(inp) if gi is None else gi
        for gi, inp in zip(grad_inputs, inputs)
    )
    return grad_inputs


# NOTE [grad and vjp interaction with no_grad]
#
# def f(x):
#   with torch.no_grad():
#     c = x ** 2
#   return x - c
#
# The thing to consider is if enable_grad is on/off before grad gets called.
#
# Case 1: enable_grad is on.
# grad(f)(x)
# In this case, `grad` should respect the inner torch.no_grad.
#
# Case 2: enable_grad is off
# with torch.no_grad():
#   grad(f)(x)
# In this case, `grad` should respect the inner torch.no_grad, but not the
# outer one. This is because `grad` is a "function transform": its result
# should not depend on the result of a context manager outside of `f`.
#
# This gives us the following desired behavior:
# - (nested) grad transforms must obey torch.no_grad inside them
# - (nested) grad transforms should not obey torch.no_grad outside them
#
# To achieve this behavior, upon entering grad/vjp:
# - we save the current ("previous") is_grad_enabled (*)
# - we unconditionally enable grad.
#
# Inside DynamicLayerBackFallback, when we're temporarily popping `grad` layer
# off the stack:
# - if grad_mode is disabled, then we do nothing. (there is a torch.no_grad
#   active, all subsequent grad transforms must obey it).
# - if grad_mode is enabled, and the previous is_grad_enabled (*) is False,
#   then we temporarily restore the previous `is_grad_enabled`. This is
#   because we're crossing the boundary from a `grad` outside the
#   no_grad to a `grad` inside the no_grad.
#
# NB: vjp has some interesting behavior because the vjp's callable can be called
# under a different grad_mode than the forward computation...
#
# NB: forward-mode AD: forward-mode AD doesn't respect torch.no_grad, but
# it respects c10::AutoFwGradMode. We've implemented the same logic for
# our jvp transform (it will have special handling if FwGradMode is disabled).


# How do we increment and decrement the nesting? I don't think we can.
@exposed_in("torch.func")
def vjp(func: Callable, *primals, has_aux: bool = False):
    """
    Standing for the vector-Jacobian product, returns a tuple containing the
    results of ``func`` applied to ``primals`` and a function that, when
    given ``cotangents``, computes the reverse-mode Jacobian of ``func`` with
    respect to ``primals`` times ``cotangents``.

    Args:
        func (Callable): A Python function that takes one or more arguments. Must
            return one or more Tensors.
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, vjp_fn)`` tuple containing the output of ``func``
        applied to ``primals`` and a function that computes the vjp of
        ``func`` with respect to all ``primals`` using the cotangents passed
        to the returned function. If ``has_aux is True``, then instead returns a
        ``(output, vjp_fn, aux)`` tuple.
        The returned ``vjp_fn`` function will return a tuple of each VJP.

    When used in simple cases, :func:`vjp` behaves the same as :func:`grad`

        >>> x = torch.randn([5])
        >>> f = lambda x: x.sin().sum()
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> grad = vjpfunc(torch.tensor(1.0))[0]
        >>> assert torch.allclose(grad, torch.func.grad(f)(x))

    However, :func:`vjp` can support functions with multiple outputs by
    passing in the cotangents for each of the outputs

        >>> x = torch.randn([5])
        >>> f = lambda x: (x.sin(), x.cos())
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> vjps = vjpfunc((torch.ones([5]), torch.ones([5])))
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    :func:`vjp` can even support outputs being Python structs

        >>> x = torch.randn([5])
        >>> f = lambda x: {"first": x.sin(), "second": x.cos()}
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> cotangents = {"first": torch.ones([5]), "second": torch.ones([5])}
        >>> vjps = vjpfunc(cotangents)
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    The function returned by :func:`vjp` will compute the partials with
    respect to each of the ``primals``

        >>> x, y = torch.randn([5, 4]), torch.randn([4, 5])
        >>> (_, vjpfunc) = torch.func.vjp(torch.matmul, x, y)
        >>> cotangents = torch.randn([5, 5])
        >>> vjps = vjpfunc(cotangents)
        >>> assert len(vjps) == 2
        >>> assert torch.allclose(vjps[0], torch.matmul(cotangents, y.transpose(0, 1)))
        >>> assert torch.allclose(vjps[1], torch.matmul(x.transpose(0, 1), cotangents))

    ``primals`` are the positional arguments for ``f``. All kwargs use their
    default value

        >>> x = torch.randn([5])
        >>> def f(x, scale=4.):
        >>>   return x * scale
        >>>
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> vjps = vjpfunc(torch.ones_like(x))
        >>> assert torch.allclose(vjps[0], torch.full(x.shape, 4.0))

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``vjp``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``vjp(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``vjp`` inside ``torch.no_grad`` context manager:

            >>> # xdoctest: +SKIP(failing)
            >>> with torch.no_grad():
            >>>     vjp(f)(x)

        In this case, ``vjp`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``vjp`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    return _vjp_with_argnums(func, *primals, has_aux=has_aux)


@contextlib.contextmanager
def grad_increment_nesting():
    try:
        grad_level = _grad_increment_nesting()
        yield grad_level
    finally:
        _grad_decrement_nesting()


def enter_jvp_nesting():
    global JVP_NESTING
    jvp_level = _jvp_increment_nesting()
    JVP_NESTING += 1
    return jvp_level


def exit_jvp_nesting():
    global JVP_NESTING
    _jvp_decrement_nesting()
    JVP_NESTING -= 1


@contextlib.contextmanager
def jvp_increment_nesting():
    try:
        yield enter_jvp_nesting()
    finally:
        exit_jvp_nesting()


@doesnt_support_saved_tensors_hooks
def _vjp_with_argnums(
    func: Callable, *primals, argnums: Optional[argnums_t] = None, has_aux: bool = False
):
    # This is the same function as vjp but also accepts an argnums argument
    # All args are the same as vjp except for the added argument
    # argnums (Optional[int or tuple[int]]): Optional, specifies the argument(s) to compute gradients with respect to.
    #         If None, computes the gradients with respect to all inputs (used for vjp). Default: None
    #
    # WARN: Users should NOT call this function directly and should just be calling vjp.
    # It is only separated so that inputs passed to jacrev but not differentiated get the correct wrappers.
    #
    # NOTE: All error messages are produced as if vjp was being called, even if this was called by jacrev
    #
    # Returns the same two elements as :func:`vjp` but the function returned, vjp_fn, returns a tuple of VJPs
    # for only the primal elements given by argnums.
    with grad_increment_nesting() as level:
        # See NOTE [grad and vjp interaction with no_grad]
        with torch.enable_grad():
            primals = _wrap_all_tensors(primals, level)
            if argnums is None:
                diff_primals = _create_differentiable(primals, level)
            else:
                diff_primals = _slice_argnums(primals, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_primals)
            primals_out = func(*primals)

            if has_aux:
                if not (isinstance(primals_out, tuple) and len(primals_out) == 2):
                    raise RuntimeError(
                        "vjp(f, *primals): output of function f should be a tuple: (output, aux) "
                        "if has_aux is True"
                    )
                primals_out, aux = primals_out
                aux = _undo_create_differentiable(aux, level)

            flat_primals_out, primals_out_spec = tree_flatten(primals_out)
            assert_non_empty_tensor_output(flat_primals_out, "vjp(f, *primals)")
            flat_diff_primals, primals_spec = tree_flatten(diff_primals)
            results = _undo_create_differentiable(primals_out, level)

            for primal_out in flat_primals_out:
                assert isinstance(primal_out, torch.Tensor)
                if primal_out.is_floating_point() or primal_out.is_complex():
                    continue
                raise RuntimeError(
                    "vjp(f, ...): All outputs of f must be "
                    "floating-point or complex Tensors, got Tensor "
                    f"with dtype {primal_out.dtype}"
                )

        def wrapper(cotangents, retain_graph=True, create_graph=None):
            if create_graph is None:
                create_graph = torch.is_grad_enabled()
            flat_cotangents, cotangents_spec = tree_flatten(cotangents)
            if primals_out_spec != cotangents_spec:
                raise RuntimeError(
                    f"Expected pytree structure of cotangents to be the same "
                    f"as pytree structure of outputs to the function. "
                    f"cotangents: {treespec_pprint(cotangents_spec)}, "
                    f"primal output: {treespec_pprint(primals_out_spec)}"
                )
            result = _autograd_grad(
                flat_primals_out,
                flat_diff_primals,
                flat_cotangents,
                retain_graph=retain_graph,
                create_graph=create_graph,
            )
            return tree_unflatten(result, primals_spec)

    if has_aux:
        return results, wrapper, aux
    else:
        return results, wrapper


def _safe_zero_index(x):
    assert len(x) == 1
    return x[0]


# jacrev and jacfwd don't support complex functions
# Helper function to throw appropriate error.
def error_if_complex(func_name, args, is_input):
    flat_args = pytree.tree_leaves(args)
    for idx, arg in enumerate(flat_args):
        if isinstance(arg, torch.Tensor) and arg.dtype.is_complex:
            input_or_output = "inputs" if is_input else "outputs"
            err_msg = (
                f"{func_name}: Expected all {input_or_output} "
                f"to be real but received complex tensor at flattened input idx: {idx}"
            )
            raise RuntimeError(err_msg)


@exposed_in("torch.func")
def jacrev(
    func: Callable,
    argnums: Union[int, tuple[int]] = 0,
    *,
    has_aux=False,
    chunk_size: Optional[int] = None,
    _preallocate_and_copy=False,
):
    """
    Computes the Jacobian of ``func`` with respect to the arg(s) at index
    ``argnum`` using reverse mode autodiff

    .. note::
        Using :attr:`chunk_size=1` is equivalent to computing the jacobian
        row-by-row with a for-loop i.e. the constraints of :func:`vmap` are
        not applicable.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        chunk_size (None or int): If None (default), use the maximum chunk size
            (equivalent to doing a single vmap over vjp to compute the jacobian).
            If 1, then compute the jacobian row-by-row with a for-loop.
            If not None, then compute the jacobian :attr:`chunk_size` rows at a time
            (equivalent to doing multiple vmap over vjp). If you run into memory issues computing
            the jacobian, please try to specify a non-None chunk_size.

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>> jacobian = jacrev(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacrev(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    :func:`jacrev` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacrev, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacrev(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    Additionally, :func:`jacrev` can be composed with itself to produce
    Hessians

        >>> from torch.func import jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacrev(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacrev` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``jacrev``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``jacrev(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``jacrev`` inside ``torch.no_grad`` context manager:

            >>> with torch.no_grad():
            >>>     jacrev(f)(x)

        In this case, ``jacrev`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``jacrev`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError("jacrev: `chunk_size` should be greater than 0.")

    @wraps(func)
    def wrapper_fn(*args):
        error_if_complex("jacrev", args, is_input=True)
        vjp_out = _vjp_with_argnums(func, *args, argnums=argnums, has_aux=has_aux)
        if has_aux:
            output, vjp_fn, aux = vjp_out
        else:
            output, vjp_fn = vjp_out

        # See NOTE: [Computing jacobian with vmap and vjp for multiple outputs]
        flat_output, output_spec = tree_flatten(output)

        error_if_complex("jacrev", flat_output, is_input=False)

        # NB: vjp already checks that all outputs are tensors
        # Step 1: Construct grad_outputs by splitting the standard basis
        flat_output_numels = tuple(out.numel() for out in flat_output)

        primals = _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)

        def compute_jacobian_stacked():
            # Helper function to compute chunked Jacobian
            # The intermediate chunked calculation are only
            # scoped at this function level.
            chunked_results = []
            for flat_basis_chunk in _chunked_standard_basis_for_(
                flat_output, flat_output_numels, chunk_size=chunk_size
            ):
                if chunk_size == 1:
                    # sanity check.
                    for t in flat_basis_chunk:
                        assert t.size(0) == 1

                    flat_basis_chunk = tree_map(
                        lambda t: torch.squeeze(t, 0), flat_basis_chunk
                    )

                basis = tree_unflatten(flat_basis_chunk, output_spec)

                if chunk_size == 1:
                    # Behaviour with `chunk_size=1` is same as `for-loop`
                    # i.e. user shouldn't deal with the limitations of vmap.
                    chunked_result = vjp_fn(basis)
                else:  # chunk_size is None or chunk_size != 1
                    chunked_result = vmap(vjp_fn)(basis)

                flat_results = pytree.tree_leaves(chunked_result)

                if chunk_size == 1:
                    flat_results = tree_map(
                        lambda t: torch.unsqueeze(t, 0), flat_results
                    )

                chunked_results.append(flat_results)

            if len(chunked_results) == 1:
                # Short-circuit if we used a single chunk
                return chunked_results[0]

            # Concatenate chunks.
            flat_results = []
            # Iterate and concat the jacobians of different
            # inputs.
            for idx in range(len(flat_primals)):
                r = tuple(r_[idx] for r_ in chunked_results)
                flat_results.append(torch.cat(r, 0))

            return flat_results

        def compute_jacobian_preallocate_and_copy():
            # Helper function to compute chunked Jacobian
            # The intermediate chunked calculation are only
            # scoped at this function level.
            out_vec_size = sum(flat_output_numels)

            # Don't pre-allocate if we have a single chunk.
            if not (chunk_size is None or chunk_size >= out_vec_size):
                stacked_results = [
                    primal.new_zeros(out_vec_size, *primal.shape)
                    for primal in flat_primals
                ]

            for idx, flat_basis_chunk in enumerate(
                _chunked_standard_basis_for_(
                    flat_output, flat_output_numels, chunk_size=chunk_size
                )
            ):
                if chunk_size == 1:
                    # sanity check.
                    for t in flat_basis_chunk:
                        assert t.size(0) == 1

                    flat_basis_chunk = [torch.squeeze(t, 0) for t in flat_basis_chunk]

                basis = tree_unflatten(flat_basis_chunk, output_spec)

                if chunk_size == 1:
                    # Behaviour with `chunk_size=1` is same as `for-loop`
                    # i.e. user shouldn't deal with the limitations of vmap.
                    chunked_result = vjp_fn(basis)
                else:  # chunk_size is None or chunk_size != 1
                    chunked_result = vmap(vjp_fn)(basis)

                flat_results = pytree.tree_leaves(chunked_result)

                # Short-circuit if we have a single chunk.
                if chunk_size is None or chunk_size >= out_vec_size:
                    if chunk_size == 1:  # and out_vec_size == 1
                        # Since we squeezed the output dim
                        flat_results = tree_map(
                            lambda t: torch.unsqueeze(t, 0), flat_results
                        )
                    return flat_results

                for r, sr in zip(flat_results, stacked_results):
                    sr[idx * chunk_size : (idx + 1) * chunk_size].copy_(r)

            return stacked_results

        if _preallocate_and_copy:
            flat_jacobians_per_input = compute_jacobian_preallocate_and_copy()
        else:
            flat_jacobians_per_input = compute_jacobian_stacked()

        # Step 2: The returned jacobian is one big tensor per input. In this step,
        # we split each Tensor by output.
        flat_jacobians_per_input = [
            result.split(flat_output_numels, dim=0)
            for result in flat_jacobians_per_input
        ]
        flat_input_flat_output = [
            tuple(
                split.view(out.shape + primal.shape)
                for split, out in zip(splits, flat_output)
            )
            for splits, primal in zip(flat_jacobians_per_input, flat_primals)
        ]

        # Step 3: Right now, `jacobian` is a List[List[Tensor]].
        # The outer List corresponds to the number of primals,
        # the inner List corresponds to the number of outputs.
        # We need to:
        # a. Exchange the order of the outer List and inner List
        # b. tree_unflatten the inner Lists (which correspond to the primals)
        # c. handle the argnums=int case
        # d. tree_unflatten the outer List (which corresponds to the outputs)
        flat_output_flat_input = tuple(zip(*flat_input_flat_output))

        flat_output_input = tuple(
            tree_unflatten(flat_input, primals_spec)
            for flat_input in flat_output_flat_input
        )

        if isinstance(argnums, int):
            flat_output_input = tuple(
                _safe_zero_index(flat_input) for flat_input in flat_output_input
            )
        output_input = tree_unflatten(flat_output_input, output_spec)
        if has_aux:
            return output_input, aux
        return output_input

    return wrapper_fn


# NOTE: [Computing jacobian with vmap and vjp for multiple outputs]
#
# Let's consider f(x) = (x**2, x.sum()) and let x = torch.randn(3).
# It turns out we can compute the jacobian of this function with a single
# call to autograd.grad by using vmap over the correct grad_outputs.
#
# Firstly, one way to compute the jacobian is to stack x**2 and x.sum()
# into a 4D vector. E.g., use g(x) = torch.stack([x**2, x.sum()])
#
# To get the first row of the jacobian, we call
# >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([1, 0, 0, 0]))
# To get the 2nd row of the jacobian, we call
# >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([0, 1, 0, 0]))
# and so on.
#
# Using vmap, we can vectorize all 4 of these computations into one by
# passing the standard basis for R^4 as the grad_output.
# vmap(partial(autograd.grad, g(x), x))(torch.eye(4)).
#
# Now, how do we compute the jacobian *without stacking the output*?
# We can just split the standard basis across the outputs. So to
# compute the jacobian of f(x), we'd use
# >>> autograd.grad(f(x), x, grad_outputs=_construct_standard_basis_for(...))
# The grad_outputs looks like the following:
# ( torch.tensor([[1, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, 1],
#                 [0, 0, 0]]),
#   torch.tensor([[0],
#                 [0],
#                 [0],
#                 [1]]) )
#
# But we're not done yet!
# >>> vmap(partial(autograd.grad(f(x), x, grad_outputs=...)))
# returns a Tensor of shape [4, 3]. We have to remember to split the
# jacobian of shape [4, 3] into two:
# - one of shape [3, 3] for the first output
# - one of shape [   3] for the second output


def _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
    # This function:
    # - constructs a N=sum(tensor_numels) standard basis. i.e. an NxN identity matrix.
    # - Splits the identity matrix into chunks with each chunk size determined by `tensor_numels`.
    # - Each chunk corresponds to one tensor. The chunk has the same dtype and
    #   device as the tensor
    #
    # For example, with tensor_numels = [1, 2, 1], this function returns:
    # ( tensor([[1],     tensor([[0, 0],      tensor([[0],
    #           [0],             [1, 0],              [0],
    #           [0],             [0, 1],              [0],
    #           [0]])  ,         [0, 0]])  ,          [1]])  )
    #
    # Precondition: tensor_numels == tuple(tensor.numel() for tensor in tensors)
    # Precondition: tensors always has at least one element.
    #
    # See NOTE: [Computing jacobian with vmap and grad for multiple tensors]
    # for context behind this function.
    # NOTE: Argument `chunk_size` is used to generate chunked basis instead of
    #       one huge basis matrix. `chunk_size` dictates the maximum size of the
    #       basis matrix along dim=0.
    assert len(tensors) == len(tensor_numels)
    assert len(tensors) > 0
    assert chunk_size is None or chunk_size > 0
    total_numel = sum(tensor_numels)
    if chunk_size and chunk_size < total_numel:
        chunk_numels = get_chunk_sizes(total_numel, chunk_size)
    else:  # chunk_size is None or chunk_size >= total_numel
        chunk_size = total_numel
        chunk_numels = [total_numel]

    diag_start_indices = (
        0,
        *torch.tensor(tensor_numels).cumsum(dim=0)[:-1].neg().unbind(),
    )

    for chunk_idx, total_numel in enumerate(chunk_numels):
        chunks = tuple(
            tensor.new_zeros(total_numel, tensor_numel)
            for tensor, tensor_numel in zip(tensors, tensor_numels)
        )

        for chunk, diag_start_idx in zip(chunks, diag_start_indices):
            chunk.diagonal(diag_start_idx + chunk_idx * chunk_size).fill_(1)
        chunks = tuple(
            chunk.view(total_numel, *tensor.shape)
            for chunk, tensor in zip(chunks, tensors)
        )
        yield chunks


def _construct_standard_basis_for(tensors, tensor_numels):
    for basis in _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
        return basis


def _validate_and_wrap_argnum(argnum, num_args):
    if not isinstance(argnum, int):
        raise RuntimeError(f"argnum must be int, got: {type(argnum)}")
    if argnum >= 0 and argnum < num_args:
        return argnum
    if argnum < 0 and argnum >= -num_args:
        return argnum + num_args
    raise RuntimeError(f"Got argnum={argnum}, but only {num_args} positional inputs")


def _check_unique_non_empty(argnums):
    if isinstance(argnums, tuple):
        if len(argnums) == 0:
            raise RuntimeError("argnums must be non-empty")
        if len(set(argnums)) != len(argnums):
            raise RuntimeError(f"argnums elements must be unique, got {argnums}")


def _replace_args(old_args, new_args, argnums):
    if isinstance(argnums, int):
        if len(new_args) != 1:
            raise RuntimeError(
                f"new_args should be of size 1, was of size {len(new_args)}"
            )
        return tuple(
            new_args[0] if i == argnums else old_args[i] for i in range(len(old_args))
        )
    if isinstance(argnums, tuple):
        if len(new_args) != len(argnums):
            raise RuntimeError(
                "new_args should have the same size as argnums. "
                f"Argnums size {len(argnums)}, new_args size {len(new_args)}"
            )

        def get_right_elem(i):
            return new_args[argnums.index(i)] if i in argnums else old_args[i]

        return tuple(get_right_elem(i) for i in range(len(old_args)))
    raise RuntimeError(f"argnums must be int or Tuple[int, ...], got: {type(argnums)}")


def _validate_and_wrap_argnums(argnums, num_args):
    if isinstance(argnums, int):
        return _validate_and_wrap_argnum(argnums, num_args)
    if isinstance(argnums, tuple):
        return tuple(_validate_and_wrap_argnum(argnum, num_args) for argnum in argnums)
    raise AssertionError("Should never get here")


def _slice_argnums(args, argnums, as_tuple=True):
    if not isinstance(argnums, int) and not isinstance(argnums, tuple):
        raise RuntimeError(
            f"argnums must be int or Tuple[int, ...], got: {type(argnums)}"
        )
    argnums = _validate_and_wrap_argnums(argnums, len(args))
    _check_unique_non_empty(argnums)
    if isinstance(argnums, int):
        if as_tuple:
            return (args[argnums],)
        else:
            return args[argnums]
    return tuple(args[i] for i in argnums)


JVP_NESTING = 0


def assert_flat_tuple_of_tensors(elts: Any, api: str, argname: str) -> None:
    if not isinstance(elts, tuple):
        raise RuntimeError(
            f"{api}: Expected {argname} to be a tuple of Tensors, got {type(elts)}"
        )
    for elt in elts:
        if isinstance(elt, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected {argname} to be a tuple of Tensors, got "
            f"a tuple with an element of type {type(elt)}"
        )
    if len(elts) == 0:
        raise RuntimeError(
            f"{api}: Expected {argname} to be a non-empty tuple of Tensors."
        )


def assert_non_empty_tensor_output(output: list[Any], api: str) -> None:
    if (len(output) == 1 and output[0] is None) or len(output) < 1:
        raise RuntimeError(
            f"{api}: Expected f to be a function that has non-empty output (got output = {output})"
        )
    for o in output:
        if not isinstance(o, torch.Tensor):
            raise RuntimeError(
                f"{api}: expected f(*primals) to return only tensors"
                f", got unsupported type {type(o)}"
            )


def assert_output_is_tensor_or_tensors(output: Any, api: str) -> None:
    if isinstance(output, torch.Tensor):
        return
    if not isinstance(output, tuple):
        raise RuntimeError(
            f"{api}: Expected output of f to be a Tensor or Tensors, got {type(output)}"
        )
    if len(output) == 0:
        raise RuntimeError(
            f"{api}: Expected output of f to be a non-empty tuple of Tensors."
        )
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected output of f to be a Tensor or Tensors, got "
            f"{type(out)} as an output"
        )


def assert_non_empty_list_of_tensors(
    output: list[torch.Tensor], api: str, argname: str
) -> None:
    if len(output) == 0:
        raise RuntimeError(f"{api}: Expected {argname} to contain at least one Tensor.")
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected {argname} to only contain Tensors, got {type(out)}"
        )


jvp_str = "jvp(f, primals, tangents)"


def safe_unpack_dual(dual, strict):
    if not isinstance(dual, torch.Tensor):
        raise RuntimeError(
            f"{jvp_str}: expected f(*args) to return only tensors"
            f", got unsupported type {type(dual)}"
        )

    primal, tangent = fwAD.unpack_dual(dual)
    if tangent is None:
        if strict:
            raise RuntimeError(
                "jvp(f, primals, tangents, strict=True): "
                "The output of f is independent of "
                "the inputs. This is not allowed with strict=True."
            )
        tangent = torch.zeros_like(primal)
    return primal, tangent


@exposed_in("torch.func")
def jvp(
    func: Callable,
    primals: Any,
    tangents: Any,
    *,
    strict: bool = False,
    has_aux: bool = False,
):
    """
    Standing for the Jacobian-vector product, returns a tuple containing
    the output of `func(*primals)` and the "Jacobian of ``func`` evaluated at
    ``primals``" times ``tangents``. This is also known as forward-mode autodiff.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        tangents (Tensors): The "vector" for which Jacobian-vector-product is
            computed. Must be the same structure and sizes as the inputs to
            ``func``.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, jvp_out)`` tuple containing the output of ``func``
        evaluated at ``primals`` and the Jacobian-vector product.
        If ``has_aux is True``, then instead returns a ``(output, jvp_out, aux)`` tuple.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.

    jvp is useful when you wish to compute gradients of a function R^1 -> R^N

        >>> from torch.func import jvp
        >>> x = torch.randn([])
        >>> f = lambda x: x * torch.tensor([1.0, 2.0, 3])
        >>> value, grad = jvp(f, (x,), (torch.tensor(1.0),))
        >>> assert torch.allclose(value, f(x))
        >>> assert torch.allclose(grad, torch.tensor([1.0, 2, 3]))

    :func:`jvp` can support functions with multiple inputs by passing in the
    tangents for each of the inputs

         >>> from torch.func import jvp
         >>> x = torch.randn(5)
         >>> y = torch.randn(5)
         >>> f = lambda x, y: (x * y)
         >>> _, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
         >>> assert torch.allclose(output, x + y)

    """

    return _jvp_with_argnums(
        func, primals, tangents, argnums=None, strict=strict, has_aux=has_aux
    )


def _jvp_with_argnums(
    func: Callable,
    primals: Any,
    tangents: Any,
    argnums: Optional[argnums_t],
    *,
    strict: bool = False,
    has_aux: bool,
):
    # This is the same function as jvp but also accepts an argnums argument
    # Most args are the same as jvp except for the added argument
    # argnums (Optional[int or tuple[int]]): Optional, specifies the argument(s) to compute gradients with respect to.
    #         If None, computes the gradients with respect to all inputs (used for jvp). Default: None
    # Because of this, tangents must be of length argnums and matches up to the corresponding primal whose index is
    # given by argnums
    #
    # WARN: Users should NOT call this function directly and should just be calling jvp.
    # It is only separated so that inputs passed to jacfwd but not differentiated get the correct wrappers.
    #
    # NOTE: All error messages are produced as if jvp was being called, even if this was called by jacfwd
    #
    # Returns the same two elements as :func:`jvp` but the returned tuple, ``jvp_out``, only has JVPs with respect to
    # the primals given by argnums
    if not isinstance(primals, tuple):
        raise RuntimeError(
            f"{jvp_str}: Expected primals to be a tuple. "
            f"E.g. it should be valid to call f(*primals)."
        )
    diff_args = primals if argnums is None else _slice_argnums(primals, argnums)
    flat_primals, primals_spec = tree_flatten(diff_args)
    flat_tangents, tangents_spec = tree_flatten(tangents)
    if primals_spec != tangents_spec:
        raise RuntimeError(
            f"{jvp_str}: Expected primals and tangents to have the same python "
            f"structure. For example, if primals is a tuple of 3 tensors, "
            f"tangents also must be. Got primals with structure {primals_spec} "
            f"and tangents with structure {tangents_spec}"
        )
    assert_non_empty_list_of_tensors(flat_primals, jvp_str, "primals")
    assert_non_empty_list_of_tensors(flat_tangents, jvp_str, "tangents")

    global JVP_NESTING

    with jvp_increment_nesting() as level:
        with fwAD._set_fwd_grad_enabled(True):
            ctx = fwAD.dual_level if JVP_NESTING == 1 else contextlib.nullcontext
            with ctx():
                flat_duals = tuple(
                    fwAD.make_dual(p, t) for p, t in zip(flat_primals, flat_tangents)
                )
                duals = tree_unflatten(flat_duals, primals_spec)
                if argnums is not None:
                    primals = _wrap_all_tensors(primals, level)
                    duals = _replace_args(primals, duals, argnums)
                result_duals = func(*duals)
                if has_aux:
                    if not (isinstance(result_duals, tuple) and len(result_duals) == 2):
                        raise RuntimeError(
                            f"{jvp_str}: output of function f should be a tuple: (output, aux) "
                            "if has_aux is True"
                        )
                    result_duals, aux = result_duals
                    aux = _undo_create_differentiable(aux, level)

                result_duals, spec = tree_flatten(result_duals)
                assert_non_empty_tensor_output(result_duals, jvp_str)

                primals_out, tangents_out = zip(
                    *[safe_unpack_dual(dual, strict) for dual in result_duals]
                )
                primals_out = tree_map(
                    partial(_undo_create_differentiable, level=level), primals_out
                )
                tangents_out = tree_map(
                    partial(_undo_create_differentiable, level=level), tangents_out
                )

                primals_out_unflatten = tree_unflatten(primals_out, spec)
                tangents_out_unflatten = tree_unflatten(tangents_out, spec)
                if has_aux:
                    return primals_out_unflatten, tangents_out_unflatten, aux

                return primals_out_unflatten, tangents_out_unflatten


def safe_unflatten(tensor, dim, shape):
    if len(shape) == 0:
        assert tensor.shape[dim] == 1
        return tensor.squeeze(dim)
    return tensor.unflatten(dim, shape)


@exposed_in("torch.func")
def jacfwd(
    func: Callable,
    argnums: argnums_t = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
):
    """
    Computes the Jacobian of ``func`` with respect to the arg(s) at index
    ``argnum`` using forward-mode autodiff

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        randomness(str): Flag indicating what type of randomness to use.
            See :func:`vmap` for more detail. Allowed: "different", "same", "error".
            Default: "error"

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.
        An alternative is to use :func:`jacrev`, which has better operator coverage.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>> jacobian = jacfwd(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    :func:`jacfwd` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacfwd, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacfwd(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacfwd(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    Additionally, :func:`jacrev` can be composed with itself or :func:`jacrev`
    to produce Hessians

        >>> from torch.func import jacfwd, jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacfwd` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    """

    @wraps(func)
    def wrapper_fn(*args):
        error_if_complex("jacfwd", args, is_input=True)
        primals = args if argnums is None else _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)
        flat_primals_numels = tuple(p.numel() for p in flat_primals)
        flat_basis = _construct_standard_basis_for(flat_primals, flat_primals_numels)
        basis = tree_unflatten(flat_basis, primals_spec)

        def push_jvp(basis):
            output = _jvp_with_argnums(
                func, args, basis, argnums=argnums, has_aux=has_aux
            )
            # output[0] is the output of `func(*args)`
            error_if_complex("jacfwd", output[0], is_input=False)
            if has_aux:
                _, jvp_out, aux = output
                return jvp_out, aux
            _, jvp_out = output
            return jvp_out

        results = vmap(push_jvp, randomness=randomness)(basis)
        if has_aux:
            results, aux = results
            # aux is in the standard basis format, e.g. NxN matrix
            # We need to fetch the first element as original `func` output
            flat_aux, aux_spec = tree_flatten(aux)
            flat_aux = [value[0] for value in flat_aux]
            aux = tree_unflatten(flat_aux, aux_spec)

        jac_outs, spec = tree_flatten(results)
        # Most probably below output check can never raise an error
        # as jvp should test the output before
        # assert_non_empty_output(jac_outs, 'jacfwd(f, ...)(*args)')

        jac_outs_ins = tuple(
            tuple(
                safe_unflatten(jac_out_in, -1, primal.shape)
                for primal, jac_out_in in zip(
                    flat_primals,
                    jac_out.movedim(0, -1).split(flat_primals_numels, dim=-1),
                )
  
```



## High-Level Overview


This Python file contains 0 class(es) and 81 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `lazy_dynamo_disallow`, `enable_inplace_requires_grad`, `_set_tensor_requires_grad`, `_create_differentiable`, `create_differentiable`, `_undo_create_differentiable`, `unwrap_tensors`, `_is_differentiable`, `_any_differentiable`, `_wrap_tensor_for_grad`, `_wrap_all_tensors`, `_as_tuple`, `_autograd_grad`, `f`, `vjp`, `f`, `f`, `grad_increment_nesting`, `enter_jvp_nesting`, `exit_jvp_nesting`

**Key imports**: contextlib, Callable, partial, wraps, Any, Optional, Union, torch, torch.autograd.forward_ad as fwAD, argnums_t, exposed_in, FunctionalTensor, const_fold, make_fx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `collections.abc`: Callable
- `functools`: partial, wraps
- `typing`: Any, Optional, Union
- `torch`
- `torch.autograd.forward_ad as fwAD`
- `torch._functorch.utils`: argnums_t, exposed_in
- `torch._subclasses.functional_tensor`: FunctionalTensor
- `torch.fx.experimental`: const_fold
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.utils`: _pytree as pytree
- `.apis`: vmap
- `.vmap`: doesnt_support_saved_tensors_hooks, get_chunk_sizes
- `torch._dynamo`
- `torch.func`: jacrev


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/_functorch`):

- [`predispatch.py_docs.md`](./predispatch.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`batch_norm_replacement.py_docs.md`](./batch_norm_replacement.py_docs.md)
- [`pytree_hacks.py_docs.md`](./pytree_hacks.py_docs.md)
- [`python_key.py_docs.md`](./python_key.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`partitioners.py_docs.md`](./partitioners.py_docs.md)
- [`vmap.py_docs.md`](./vmap.py_docs.md)
- [`pyfunctorch.py_docs.md`](./pyfunctorch.py_docs.md)


## Cross-References

- **File Documentation**: `eager_transforms.py_docs.md`
- **Keyword Index**: `eager_transforms.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
