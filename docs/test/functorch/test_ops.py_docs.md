# Documentation: `test/functorch/test_ops.py`

## File Metadata

- **Path**: `test/functorch/test_ops.py`
- **Size**: 126,065 bytes (123.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: functorch"]
# ruff: noqa: F841

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import unittest

from common_utils import (
    check_vmap_fallback,
    decorate,
    expectedFailureIf,
    generate_vmap_inputs,
    get_fallback_and_vmap_exhaustive,
    is_batch_norm_training,
    is_valid_inplace_sample_input,
    loop,
    loop2,
    opsToleranceOverride,
    skip,
    skipOps,
    tol1,
    tol2,
    xfail,
)
from functorch_additional_op_db import additional_op_db

import torch
import torch.autograd.forward_ad as fwAD
from functorch import grad, jacfwd, jacrev, vjp, vmap
from torch import Tensor
from torch._functorch.eager_transforms import _as_tuple, jvp
from torch.testing._internal.autograd_function_db import autograd_function_db
from torch.testing._internal.common_cuda import with_tf32_off
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    is_iterable_of_tensors,
    noncontiguous_like,
    parametrize,
    run_tests,
    runOnRocm,
    skipIfRocm,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase,
    unMarkDynamoStrictTest,
)
from torch.testing._internal.opinfo.core import SampleInput
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


aten = torch.ops.aten


# Version of autograd.grad with some differences:
#   - pytree inputs is allowed (but leaves of the pytree have to all
#     be tensors)
#   - if an input is not used as part of derivatives, we will return a
#     zero-filled tensor for the result
def _autograd_grad(
    outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
):
    inputs, inputs_spec = tree_flatten(inputs)
    diff_inputs = tuple(inp for inp in inputs if inp.requires_grad)
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        diff_grad_outputs = [
            (out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad
        ]
        if len(diff_grad_outputs) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*diff_grad_outputs)
    grad_inputs = torch.autograd.grad(
        diff_outputs,
        diff_inputs,
        grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    result = []
    grad_inputs_iter = iter(grad_inputs)
    for inp in inputs:
        if inp.requires_grad:
            grad_input = next(grad_inputs_iter)
            if grad_input is None:
                result.append(torch.zeros_like(inp))
            else:
                result.append(grad_input)
        else:
            result.append(torch.zeros_like(inp))
    return tree_unflatten(result, inputs_spec)


def diff_arg(arg, requires_grad=True):
    def is_differentiable_arg(arg):
        if requires_grad:
            return arg.requires_grad
        else:
            return arg.is_floating_point() or arg.is_complex()

    if is_iterable_of_tensors(arg):
        if all(is_differentiable_arg(a) for a in arg):
            return True
        if all(not is_differentiable_arg(a) for a in arg):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)


# Given f, returns an f' such that:
# - f' takes only positional arguments
# - All arguments to f' are floating-point Tensors
# - All outputs of f' are floating-point Tensors
def normalize_op_input_output2(
    f, args, kwargs, output_process_fn_grad=None, requires_grad=True
):
    flat_args, args_spec = tree_flatten(args)
    diff_argnums = tuple(
        i
        for i, arg in enumerate(flat_args)
        if diff_arg(arg, requires_grad=requires_grad)
    )
    assert len(diff_argnums) > 0
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result

    return wrapped, primals


# TODO: consolidate with normalize_op_input_output2
def normalize_op_input_output3(
    f, args, kwargs, sample_args, output_process_fn_grad=None
):
    flat_args, args_spec = tree_flatten(args)
    flat_sample_args = pytree.tree_leaves(sample_args)
    diff_argnums = tuple(
        i
        for i, (arg, sample) in enumerate(zip(flat_args, flat_sample_args))
        if diff_arg(sample, requires_grad=True)
    )
    assert len(diff_argnums) > 0
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result

    return wrapped, primals


def normalize_op_input_output(f, sample, requires_grad=True):
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_input_output2(
        f,
        args,
        sample.kwargs,
        sample.output_process_fn_grad,
        requires_grad=requires_grad,
    )


def ref_vjp(f, *primals):
    result = f(*primals)

    def wrapped(cotangents):
        return _autograd_grad(_as_tuple(result), primals, _as_tuple(cotangents))

    return result, wrapped


def simulate_jvp(f, primals, tangents):
    primals_out, tangents_out = torch.autograd.functional.jvp(f, primals, tangents)
    return primals_out, tangents_out


def ref_jvp(f, primals, tangents):
    with fwAD.dual_level():
        duals = tuple(fwAD.make_dual(p, t) for p, t in zip(primals, tangents))
        result_duals = f(*duals)
        result_duals, spec = tree_flatten(result_duals)
        primals_out, tangents_out = zip(*(fwAD.unpack_dual(d) for d in result_duals))
        return tree_unflatten(primals_out, spec), tree_unflatten(tangents_out, spec)


def get_sample_cotangents(f, sample):
    fn, primals = normalize_op_input_output(f, sample)
    output = fn(*primals)
    return tree_map(torch.randn_like, output)


# returns a new function g(*args, *cotangents)
# that computes vjps and (*args, cotangents)
def get_vjp_fn_and_args_with_cotangents(f, sample, cotangents):
    args = tuple([sample.input] + list(sample.args))
    kwargs = sample.kwargs
    flat_args, args_spec = tree_flatten(args)
    flat_cotangents, cotangents_spec = tree_flatten(cotangents)

    @functools.wraps(f)
    def wrapped(*args):
        assert len(args) == len(flat_args) + len(flat_cotangents)
        actual_args = args[: len(flat_args)]
        cotangents = args[len(flat_args) :]
        actual_args = tree_unflatten(actual_args, args_spec)
        cotangents = tree_unflatten(cotangents, cotangents_spec)

        fn, primals = normalize_op_input_output3(
            f, actual_args, kwargs, flat_args, sample.output_process_fn_grad
        )
        _, vjp_fn = vjp(fn, *primals)
        return vjp_fn(cotangents)

    return wrapped, tuple(flat_args + flat_cotangents)


# Returns a new function g(*args, *cotangents) that computes vjps and
# sample (*args, *cotangents)
def get_vjpfull_variant(f, sample):
    fn, primals = normalize_op_input_output(f, sample)
    return _get_vjpfull_variant(fn, primals)


def get_vjpfull_variant2(f, args, kwargs):
    fn, primals = normalize_op_input_output2(f, args, kwargs)
    return _get_vjpfull_variant(fn, primals)


def _get_vjpfull_variant(fn, primals):
    result = fn(*primals)
    cotangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x, requires_grad=True), result)
    )
    num_primals = len(primals)
    args = (*primals, *cotangents)

    @functools.wraps(fn)
    def wrapped(*args):
        primals = args[:num_primals]
        cotangents = args[num_primals:]
        result, vjp_fn = vjp(fn, *primals)
        if isinstance(result, torch.Tensor):
            assert len(cotangents) == 1
            cotangents = cotangents[0]
        return vjp_fn(cotangents)

    return wrapped, args


def get_jvp_variant(f, sample):
    # We want this higher-order variant of jvp, so that it can
    # be used to wrap vmap
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    tangents = _as_tuple(tree_map(lambda x: torch.randn_like(x), primals))

    @functools.wraps(f)
    def wrapped(*args):
        tangents = args
        primals_out, tangents_out = jvp(fn, primals, tangents)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            flat_primals_out = pytree.tree_leaves(primals_out)
            flat_tangents_out = pytree.tree_leaves(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, tangents


def get_jvp_variant_primals_tangents2(
    f, args, kwargs, output_process_fn_grad=None, requires_grad=False
):
    fn, primals = normalize_op_input_output2(
        f, args, kwargs, output_process_fn_grad, requires_grad
    )
    tangents = _as_tuple(tree_map(lambda x: torch.randn_like(x), primals))
    return _get_jvp_variant(fn, primals, tangents)


def get_jvp_variant_primals_tangents(f, sample):
    # We want this higher-order variant of jvp, so that it can
    # be used to wrap vmap
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    tangents = _as_tuple(tree_map(lambda x: torch.randn_like(x), primals))
    return _get_jvp_variant(fn, primals, tangents)


def _get_jvp_variant(fn, primals, tangents):
    @functools.wraps(fn)
    def wrapped(*args):
        primals_in = args[: len(primals)]
        tangents_in = args[len(primals) :]
        primals_out, tangents_out = jvp(fn, primals_in, tangents_in)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            flat_primals_out = pytree.tree_leaves(primals_out)
            flat_tangents_out = pytree.tree_leaves(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, primals + tangents


def is_inplace(op, variant):
    if hasattr(variant, "__wrapped__"):
        return variant.__wrapped__ is op.get_inplace()
    return variant is op.get_inplace()


vjp_fail = {
    xfail("tensor_split"),  # data_ptr composite compliance
    # Very minor accuracy issue on ROCm
    decorate("nn.functional.scaled_dot_product_attention", decorator=skipIfRocm),
}

aliasing_ops = {
    "T",
    "broadcast_to",
    "conj",
    "contiguous",
    "diagonal",  # linalg.diagonal is an alias
    "expand",
    "flatten",
    "imag",
    "mH",  # adjoint is an alias
    "mT",
    "movedim",  # moveaxis is an alias
    "narrow",
    "permute",
    "positive",
    # 'ravel', is composite implicit autograd and may call clone
    "real",
    "reshape",
    "resolve_conj",
    "resolve_neg",
    "select",
    "squeeze",
    "transpose",  # swapdims and swapaxes are aliases
    "unflatten",
    "unfold",
    "unsqueeze",
    "view",
    "view_as",
    "view_as_complex",
    "view_as_real",
}

aliasing_ops_list_return = {
    "chunks",
    "dsplit",
    "hsplit",
    "split",
    "unbind",
    "vsplit",
    # 'tensor_split' not composite compliant, see vjp_fail
}

skip_noncontig = {
    "_batch_norm_with_update",
    "as_strided_copy",
}

bool_unsupported_ordered_ops = {
    "topk",
    "argmin",
    "ceil",
    "argmax",
    "floor",
}
bool_ordered_op_db = tuple(
    filter(lambda op: op.name in bool_unsupported_ordered_ops, op_db)
)

complex_unsupported_ordered_ops = {
    "sort",
    "topk",
    "lt",
    "argmin",
    "le",
    "ge",
    "amax",
    "maximum",
    "minimum",
    "clamp",
    "amin",
    "gt",
    "ceil",
    "argmax",
    "floor",
}
complex_ordered_op_db = tuple(
    filter(lambda op: op.name in complex_unsupported_ordered_ops, op_db)
)


@unittest.skipIf(TEST_WITH_ASAN, "tests time out with asan, are probably redundant")
@unMarkDynamoStrictTest
class TestOperators(TestCase):
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_grad",
        vjp_fail.union(
            {
                xfail(
                    "chalf", "", device_type="cpu"
                ),  # RuntimeError: "sum_cpu" not implemented for 'ComplexHalf'
                xfail(
                    "sparse.sampled_addmm", ""
                ),  # RuntimeError: Sparse CSR tensors do not have strides
                xfail(
                    "sparse.mm", "reduce"
                ),  # RuntimeError: Sparse CSR tensors do not have strides
                # Non-contiguous Bugs
                #
                # AssertionError: Tensor-likes are not close!
                xfail("as_strided"),
                xfail("as_strided", "partial_views"),
                # RuntimeError: !self.requires_grad() || self.is_contiguous()
                xfail("as_strided_scatter"),
                # RuntimeError: Tensor must have a last dimension with stride 1
                xfail("view_as_complex"),
                # query: last dimension must be contiguous
                # Fused attention kernels require last dim to be contiguous
                decorate(
                    "nn.functional.scaled_dot_product_attention",
                    decorator=expectedFailureIf(not TEST_WITH_ROCM),
                ),  # Works on ROCm
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_grad",
        (
            tol1(
                "nn.functional.binary_cross_entropy_with_logits",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
            ),
            tol1("masked.cumprod", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),
            tol1("svd_lowrank", {torch.float32: tol(atol=3e-04, rtol=3e-04)}),
            tol1(
                "linalg.multi_dot",
                {torch.float32: tol(atol=1e-05, rtol=8e-04)},
                device_type="cuda",
            ),
            tol1(
                "linalg.tensorsolve",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            tol1(
                "nn.functional.multi_head_attention_forward",
                {torch.float32: tol(atol=8e-04, rtol=1e-03)},
            ),
            tol1(
                "__rmatmul__",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            tol1(
                "matmul",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            tol1(
                "pca_lowrank",
                {torch.float32: tol(atol=3e-05, rtol=4e-06)},
                device_type="cpu",
            ),
        ),
    )
    def test_grad(self, device, dtype, op):
        if op.name in vjp_fail:
            self.skipTest("Skipped; Expected failures")
            return

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped for redundancy. test_vjp handles in-place testing.")
            return

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            if op.name not in skip_noncontig:
                noncontig_sample = sample.noncontiguous()
                noncontig_args = [noncontig_sample.input] + list(noncontig_sample.args)
                noncontig_kwargs = noncontig_sample.kwargs

            diff_argnums = tuple(i for i, arg in enumerate(args) if diff_arg(arg))
            assert len(diff_argnums) > 0
            diff_args = tuple(args[i] for i in diff_argnums)

            def wrapped_fn(*args, **kwargs):
                result = op(*args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    result = sample.output_process_fn_grad(result)

                def abs_if_complex(t):
                    if t.dtype.is_complex:
                        return t.abs()
                    return t

                # Reduce into single value for grad
                if isinstance(result, torch.Tensor):
                    return abs_if_complex(result.sum())
                result = sum(abs_if_complex(res.sum()) for res in result)
                return result

            result = grad(wrapped_fn, diff_argnums)(*args, **kwargs)
            expected = _autograd_grad(_as_tuple(wrapped_fn(*args, **kwargs)), diff_args)
            self.assertEqual(result, expected)

            if op.name not in skip_noncontig:
                result_noncontig = grad(wrapped_fn, diff_argnums)(
                    *noncontig_args, **noncontig_kwargs
                )
                self.assertEqual(result_noncontig, expected)

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_jvp",
        set(
            {
                # Composite ops that do bad things. Need to be fixed in PyTorch core.
                # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
                xfail("tensor_split"),
                # BUG: silent incorrectness: runs and produces numerical differences
                skip("nn.functional.max_unpool1d"),  # fails everywhere except on mac
                skip(
                    "nn.functional.max_unpool2d"
                ),  # fails everywhere except on windows
                skip("nn.functional.max_unpool3d"),  # fails everywhere except on mac
                xfail(
                    "native_batch_norm"
                ),  # TODO: fails comparing None to tensor of 0s for saved_mean/var tangents
                xfail(
                    "_native_batch_norm_legit"
                ),  # TODO: fails comparing None to tensor of 0s for saved_mean/var tangents
                xfail(
                    "_batch_norm_with_update"
                ),  # TODO: fails comparing None to tensor of 0s for saved_mean/var tangents
                xfail("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
                xfail(
                    "nn.functional.rrelu"
                ),  # in-place test errors out with no formula implemented
                xfail(
                    "NumpyExpMarkDirtyAutogradFunction"
                ),  # TODO: https://github.com/pytorch/pytorch/issues/91280
                # --- Non-Contiguous Failures! ---
                # This is expected to fail as the operator
                # expects last dim to have stride=1
                xfail("view_as_complex"),
                # BUG
                # AssertionError: Tensor-likes are not close!
                xfail("as_strided"),
                xfail("as_strided", "partial_views"),
                xfail("as_strided_scatter"),
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_jvp",
        (
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=1e-04, rtol=1.3e-06)},
                device_type="cuda",
            ),
            tol1(
                "linalg.tensorsolve",
                {torch.float32: tol(atol=1e-04, rtol=1.3e-05)},
                device_type="cuda",
            ),
            tol1(
                "masked.prod",
                {torch.float32: tol(atol=1e-05, rtol=1.3e-05)},
                device_type="cuda",
            ),
            tol1(
                "nn.functional.binary_cross_entropy_with_logits",
                {torch.float32: tol(atol=4e-04, rtol=4e-04)},
            ),
            tol1(
                "nn.functional.batch_norm", {torch.float32: tol(atol=4e-05, rtol=5e-05)}
            ),
            tol1("nn.functional.conv2d", {torch.float32: tol(atol=4e-05, rtol=5e-05)}),
            tol1("svd_lowrank", {torch.float32: tol(atol=5e-05, rtol=5e-05)}),
            tol1("pca_lowrank", {torch.float32: tol(atol=5e-05, rtol=5e-05)}),
            tol1(
                "nn.functional.multi_head_attention_forward",
                {torch.float32: tol(atol=6e-05, rtol=2e-05)},
            ),
            tol2(
                "linalg.pinv", "hermitian", {torch.float32: tol(atol=5e-5, rtol=2e-5)}
            ),
        ),
    )
    def test_jvp(self, device, dtype, op):
        # TODO: get rid of vjp_decomp when we add decomposition support to
        # PyTorch's forward-mode ad. Currently the decomposition support only
        # works for functorch.jvp
        VJP_DECOMP = {
            "nn.functional.logsigmoid",
        }
        if op.name in VJP_DECOMP:
            fixme_ref_jvp_local = simulate_jvp
        else:
            fixme_ref_jvp_local = ref_jvp

        if not op.supports_forward_ad and op.name not in VJP_DECOMP:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        outplace_variant = op if not is_inplace(op, op.get_op()) else None
        inplace_variant = op.inplace_variant if op.supports_inplace_autograd else None

        for sample in samples:
            if outplace_variant:
                self.jvp_opinfo_test(
                    outplace_variant,
                    sample,
                    sample.output_process_fn_grad,
                    clone_inputs=False,
                    fixme_ref_jvp_local=fixme_ref_jvp_local,
                    test_noncontig=op.name not in skip_noncontig,
                )
            if is_valid_inplace_sample_input(sample, op, inplace_variant):
                self.jvp_opinfo_test(
                    inplace_variant,
                    sample,
                    sample.output_process_fn_grad,
                    clone_inputs=True,
                    fixme_ref_jvp_local=fixme_ref_jvp_local,
                    test_noncontig=op.name not in skip_noncontig,
                )

    def jvp_opinfo_test(
        self,
        fn,
        sample,
        output_process_fn,
        clone_inputs,
        fixme_ref_jvp_local,
        test_noncontig,
    ):
        # NB: we used requires_grad=True to determine where the primals are,
        # but don't need that information otherwise
        args = (sample.input,) + sample.args
        kwargs = sample.kwargs
        contig_fn, primals = normalize_op_input_output2(
            fn, args, kwargs, output_process_fn, requires_grad=True
        )
        orig_primals = tree_map(lambda x: x.detach(), primals)
        orig_tangents = tree_map(lambda x: torch.randn_like(x), primals)

        def maybe_clone_inputs():
            if clone_inputs:
                primals = tree_map(torch.clone, orig_primals)
                tangents = tree_map(torch.clone, orig_tangents)
                return primals, tangents
            return orig_primals, orig_tangents

        primals, tangents = maybe_clone_inputs()
        expected_primal_outs, expected_tangent_outs = fixme_ref_jvp_local(
            contig_fn, primals, tangents
        )

        primals, tangents = maybe_clone_inputs()
        primal_outs, tangent_outs = jvp(contig_fn, primals, tangents)

        self.assertEqual(primal_outs, expected_primal_outs)
        self.assertEqual(tangent_outs, expected_tangent_outs)

        if test_noncontig:
            noncontig_sample = sample.noncontiguous()
            noncontig_args = (noncontig_sample.input,) + noncontig_sample.args
            noncontig_kwargs = sample.kwargs
            noncontig_fn, primals = normalize_op_input_output2(
                fn,
                noncontig_args,
                noncontig_kwargs,
                output_process_fn,
                requires_grad=True,
            )
            noncontig_primals = tree_map(lambda x: x.detach(), primals)
            noncontig_tangents = tree_map(
                lambda x: noncontiguous_like(x), orig_tangents
            )
            noncontig_primal_outs, noncontig_tangent_outs = jvp(
                noncontig_fn, noncontig_primals, noncontig_tangents
            )

            self.assertEqual(noncontig_primal_outs, expected_primal_outs)
            self.assertEqual(noncontig_tangent_outs, expected_tangent_outs)

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_vjp",
        vjp_fail.union(
            {
                xfail("sparse.sampled_addmm", ""),
                xfail("sparse.mm", "reduce"),
                # ---- Non-Contiguous Failures ----
                # This is expected to fail as the operator
                # expects last dim to have stride=1
                xfail("view_as_complex"),
                # RuntimeError: query: last dimension must be contiguous
                # The fused attention kernels require the last dim to be contiguous
                decorate(
                    "nn.functional.scaled_dot_product_attention",
                    decorator=expectedFailureIf(not TEST_WITH_ROCM),
                ),  # Works on ROCm
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
                # BUG
                # AssertionError: Tensor-likes are not close!
                xfail("as_strided"),
                xfail("as_strided_scatter"),
                xfail("as_strided", "partial_views"),
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_vjp",
        (
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=5e-05, rtol=9e-05)},
                device_type="cuda",
            ),
            tol1(
                "nn.functional.binary_cross_entropy_with_logits",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
            ),
            tol1(
                "nn.functional.multi_head_attention_forward",
                {torch.float32: tol(atol=2e-03, rtol=2e-04)},
            ),
            tol1("__rmatmul__", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),
            tol1("matmul", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),
            tol2(
                "linalg.pinv", "hermitian", {torch.float32: tol(atol=1e-05, rtol=1e-05)}
            ),
            tol1("linalg.tensorsolve", {torch.float32: tol(atol=9e-03, rtol=2e-04)}),
            tol1("linalg.multi_dot", {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
            tol1("svd_lowrank", {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
            tol1("pca_lowrank", {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
        ),
    )
    def test_vjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        def _test(_op, inplace=False):
            for sample in samples:
                if inplace and not is_valid_inplace_sample_input(
                    sample, op, op.inplace_variant
                ):
                    continue
                fn, primals = normalize_op_input_output(_op, sample)
                result = fn(*primals)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                out, vjp_fn = vjp(fn, *primals)
                self.assertEqual(out, result)
                result_vjps = vjp_fn(cotangents)

                _, vjp_fn = ref_vjp(fn, *primals)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)

                if op.name not in skip_noncontig:
                    noncontig_fn, noncontig_primals = normalize_op_input_output(
                        _op, sample.noncontiguous()
                    )
                    noncontig_cotangents = tree_map(
                        lambda x: noncontiguous_like(x), cotangents
                    )
                    out_noncontig, vjp_fn = vjp(noncontig_fn, *noncontig_primals)
                    self.assertEqual(out_noncontig, result)
                    noncontig_result_vjps = vjp_fn(noncontig_cotangents)
                    self.assertEqual(noncontig_result_vjps, expected_vjps)

        _test(op)
        for a_op in op.aliases:
            _test(a_op)
        if op.inplace_variant:

            def f(inp, *args, **kwargs):
                return op.inplace_variant(inp.clone(), *args, **kwargs)

            _test(f, inplace=True)

    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_vjpvjp",
        vjp_fail.union(
            {
                skip("nn.functional.max_unpool1d"),  # silent incorrectness; Flaky
                skip("nn.functional.max_unpool2d"),  # silent incorrectness; Flaky
                xfail("nn.functional.ctc_loss"),  # Not Implemented
                xfail(
                    "native_layer_norm", ""
                ),  # Expected a proper Tensor but got None for argument #1 'other'
                xfail("sparse.sampled_addmm", ""),  # sparse tensors have no strides
                xfail("sparse.mm", "reduce"),  # sparse tensors have no strides
                skip("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
                # AssertionError: Tensor-likes are not close!
                # Mismatched elements: 1 / 15 (6.7%)
                # Greatest absolute difference: 24.0 at index (2, 4) (up to 1e-05 allowed)
                # Greatest relative difference: 1.7933241714393998e-06 at index (2, 4) (up to 1.3e-06 allowed)
                # The failure occurred for item [0]
                xfail("masked.prod"),
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_vjpvjp",
        (
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=5e-05, rtol=9e-05)},
                device_type="cuda",
            ),
            tol1("prod", {torch.float32: tol(atol=2e-05, rtol=1e-04)}),
            tol1("masked.cumprod", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
            tol1("cumprod", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
            tol1("linalg.vander", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
        ),
    )
    def test_vjpvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        def test(_op, inplace=False):
            for sample in samples:
                if inplace and not is_valid_inplace_sample_input(
                    sample, op, op.inplace_variant
                ):
                    continue
                fn, args = get_vjpfull_variant(_op, sample)
                result = fn(*args)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                # Compute vjp of vjp
                _, vjp_fn = vjp(fn, *args)
                result_vjps = vjp_fn(cotangents)

                # Compute ref_vjp of vjp. We could have done ref_vjp of ref_vjp,
                # but since we're confident that vjp works by itself, this is
                # an equivalent way to test that.
                _, vjp_fn = ref_vjp(fn, *args)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)

        test(op)
        if op.inplace_variant:

            def fn(inp, *args, **kwargs):
                return op.inplace_variant(inp.clone(), *args, **kwargs)

            test(fn, inplace=True)

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @skipOps(
        "TestOperators",
        "test_vmapvjpvjp",
        vjp_fail.union(
            {
                skip("atleast_1d"),  # Takes too long
                skip("atleast_2d"),  # Takes too long
                skip("atleast_3d"),  # Takes too long
                skip("ormqr"),  # Takes too long
                xfail("as_strided"),  # incorrect output
                xfail("as_strided", "partial_views"),  # incorrect output
                xfail("as_strided_scatter"),  # incorrect output
                skip("bernoulli"),  # calls random op
                xfail("bfloat16"),  # rank 4 tensor for channels_last
                xfail("cdouble"),  # rank 4 tensor for channels_last
                xfail("cfloat"),  # rank 4 tensor for channels_last
                xfail("chalf"),  # rank 4 tensor for channels_last
                xfail("double"),  # rank 4 tensor for channels_last
                xfail("float"),  # rank 4 tensor for channels_last
                xfail("half"),  # rank 4 tensor for channels_last
                xfail(
                    "NumpyCubeNotComposableAutogradFunction"
                ),  # Not composable autograd.Function
                # It looks like you're either (1) calling .item() on a Tensor or
                # (2) attempting to use a Tensor in some data-dependent control flow or
                # (3) encountering this error in PyTorch internals.
                xfail("index_reduce", "prod"),
                decorate(
                    "linalg.householder_product", decorator=runOnRocm
                ),  # works on ROCm
                xfail(
                    # nans
                    "masked.softmax",
                    device_type="cpu",
                ),
                xfail(
                    "nanquantile", device_type="cpu"
                ),  # vmap not implemented for at::equal.
                xfail("native_layer_norm"),  # vmap: inplace into a regular tensor
                # got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("nn.functional.batch_norm"),
                xfail(
                    "nn.functional.binary_cross_entropy"
                ),  # vmap: inplace into a regular tensor
                xfail(
                    "nn.functional.ctc_loss"
                ),  # derivate not implemented for _ctc_loss_backward
                # flaky on ROCM needs investigation
                decorate("nn.functional.conv_transpose2d", decorator=skipIfRocm),
                skip("nn.functional.dropout"),  # calls random op
                skip("nn.functional.dropout2d"),  # calls random op
                skip("nn.functional.dropout3d"),  # calls random op
                skip("nn.functional.alpha_dropout"),  # calls random op
                skip(
                    "nn.functional.feature_alpha_dropout", "with_train"
                ),  # calls random op
                skip("nn.functional.fractional_max_pool2d"),  # calls random op
                skip("nn.functional.fractional_max_pool3d"),  # calls random op
                xfail("nn.functional.scaled_dot_product_attention"),  # randomness
                xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
                xfail("nn.functional.multi_head_attention_forward"),  # randomness
                # It looks like you're either (1) calling .item() on a Tensor or
                # (2) attempting to use a Tensor in some data-dependent control flow or
                # (3) encountering this error in PyTorch internals.
                xfail("nn.functional.gaussian_nll_loss"),
                # got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("nn.functional.instance_norm"),
                xfail(
                    "nn.functional.layer_norm"
                ),  # vmap: inplace into a regular tensor
                # RuntimeError: NYI: querying is_contiguous inside of vmap
                # for memory_format other than torch.contiguous_formats
                xfail("nn.functional.max_pool2d"),
                # RuntimeError: NYI: Tensor.clone(memory_format) inside vmap is only
                # supported with memory_format torch.preserve_format or
                # torch.contiguous_format (got ChannelsLast)
                xfail("nn.functional.max_unpool2d"),
                # RuntimeError: NYI: Tensor.clone(memory_format) inside vmap is only
                # supported with memory_format torch.preserve_format
                # or torch.contiguous_format (got ChannelsLast)s
                xfail("nn.functional.max_unpool2d", "grad"),
                xfail(
                    "nn.functional.rrelu"
                ),  # RuntimeError: vmap: we do not yet support aten::rrelu_with_noise.
                xfail("normal"),  # calls random op
                xfail("normal", "number_mean"),  # calls random op
                xfail("pca_lowrank"),  # calls random op
                xfail(
                    "quantile", device_type="cpu"
                ),  # Batching rule not implemented for `at::equal`
                xfail(
                    "scatter_reduce", "prod"
                ),  # vmap (looks like you are calling item/data-dependent)
                xfail(
                    "sparse.sampled_addmm"
                ),  # RuntimeError: Sparse CSR tensors do not have strides
                xfail(
                    "sparse.mm", "reduce"
                ),  # RuntimeError: Sparse CSR tensors do not have strides
                xfail("svd_lowrank"),  # calls random op
                xfail("to"),  # rank 4 tensor for channels_last
                xfail(
                    "view_as_complex"
                ),  # RuntimeError: Tensor must have a last dimension with stride 1
                # got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("nn.functional.batch_norm", "without_cudnn"),
                # view doesn't work on sparse
                xfail("to_sparse"),
                xfail("native_batch_norm"),
                xfail("_native_batch_norm_legit"),
                # TODO: implement batching rule
                xfail("_batch_norm_with_update"),
                xfail(
                    "unbind_copy"
                ),  # Batching rule not implemented for aten::unbind_copy.int.
            }
        ),
    )
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @opsToleranceOverride(
        "TestOperators",
        "test_vmapvjpvjp",
        (
            tol1("linalg.svd", {torch.float32: tol(atol=1e-03, rtol=5e-04)}),
            tol1("linalg.lu", {torch.float32: tol(atol=5e-04, rtol=7e-04)}),
            tol1("linalg.lu_factor", {torch.float32: tol(atol=2e-03, rtol=2e-02)}),
            tol1("linalg.multi_dot", {torch.float32: tol(atol=2e-03, rtol=2e-04)}),
            tol1("svd", {torch.float32: tol(atol=1e-03, rtol=5e-04)}),
            tol1("matrix_exp", {torch.float32: tol(atol=1e-03, rtol=5e-04)}),
            tol1("masked.prod", {torch.float32: tol(atol=2e-03, rtol=2e-04)}),
        ),
    )
    @skipOps(
        "TestOperators",
        "test_vmapvjpvjp",
        {
            xfail("as_strided", "partial_views"),
            xfail("as_strided_copy"),
        },
    )
    def test_vmapvjpvjp(self, device, dtype, op):
        # Since, we test `vjpvjp` independently,
        # for this test, we just verify that vmap
        # of `vjpvjp` is correct.
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, args = get_vjpfull_variant(op, sample)
            result = fn(*args)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)
            cotangents = pytree.tree_leaves(cotangents)
            num_args = len(args)

            args_and_cotangents = tuple(args) + tuple(cotangents)

            def vjp_of_vjp(*args_and_cotangents):
                args = args_and_cotangents[:num_args]
                cotangents = args_and_cotangents[num_args:]
                result, vjp_fn = vjp(fn, *args)
                result_vjps = vjp_fn(cotangents)
                result = pytree.tree_leaves(result)
                result_vjps = pytree.tree_leaves(result_vjps)
                return (*result, *result_vjps)

            is_batch_norm_and_training = is_batch_norm_training(op.name, sample.kwargs)
            generator = get_fallback_and_vmap_exhaustive(
                vjp_of_vjp,
                args_and_cotangents,
                {},
                is_batch_norm_and_training=is_batch_norm_and_training,
            )
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    vmapvjp_fail = vjp_fail.union(
        {
            # -------------------- ALLOWED FAILURES --------------------------------
            # The following are not bugs and are expected behavior
            xfail("masked_select"),  # Not possible due to dynamic shapes
            skip("bernoulli"),  # randomness
            skip("normal", ""),  # randomness
            skip("normal", "number_mean"),  # randomness
            skip("nn.functional.rrelu"),  # randomness
            skip("nn.functional.feature_alpha_dropout", "with_train"),  # randomness
            skip("nn.functional.feature_alpha_dropout", "without_train"),  # randomness
            skip("nn.functional.dropout"),  # randomness
            skip("nn.functional.dropout2d"),  # randomness
            skip("nn.functional.dropout3d", ""),  # randomness
            skip("nn.functional.alpha_dropout"),  # randomness
            skip("nn.functional.scaled_dot_product_attention"),  # randomness
            xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
            skip("nn.functional.multi_head_attention_forward"),  # randomness
            xfail(
                "index_put", ""
            ),  # not possible due to dynamic shapes; we support a subset
            xfail("nn.functional.fractional_max_pool2d"),  # random
            xfail("nn.functional.fractional_max_pool3d"),  # random
            xfail("pca_lowrank", ""),  # randomness
            xfail("svd_lowrank", ""),  # randomness
            xfail("to_sparse", ""),  # non-dense output
            skip(
                "to"
            ),  # RuntimeError: required rank 4 tensor to use channels_last format
            xfail("as_strided", "partial_views"),
            xfail(
                "NumpyCubeNotComposableAutogradFunction"
            ),  # Not composable autograd.Function
            # ----------------------------------------------------------------------
            # ---------------------------- BUGS ------------------------------------
            # All of the following are bugs and need to be fixed
            skip(
                "linalg.svdvals"
            ),  # # really annoying thing where it passes correctness check but not has_batch_rule
            skip("native_batch_norm"),
            skip("_native_batch_norm_legit"),
            # TODO: implement batching rule
            skip("_batch_norm_with_update"),
            xfail("__getitem__", ""),  # dynamic error
            xfail("nanquantile", device_type="cpu"),  # checks q via a .item() call
            xfail("nn.functional.gaussian_nll_loss"),  # checks var for if any value < 0
            xfail("narrow"),  # .item() call
            xfail("quantile", device_type="cpu"),  # checks q via a .item() call
            xfail("view_as_complex"),  # Tensor must have a last dimension with stride 1
            # required rank 4 tensor to use channels_last format
            xfail("bfloat16"),
            xfail("double"),
            xfail("float"),
            xfail("half"),
            xfail("cdouble", ""),
            xfail("cfloat", ""),
            xfail("chalf", ""),
            xfail("scatter_reduce", "prod"),  # item call
            # Batching rule not implemented for aten::_use_cudnn_ctc_loss.Tensor
            xfail("nn.functional.ctc_loss", device_type="cuda"),
            # NYI: querying is_contiguous inside of vmap for memory_format other than torch.contiguous_format
            xfail("nn.functional.max_unpool2d"),
            xfail("nn.functional.max_unpool2d", "grad"),
            xfail("sparse.sampled_addmm", ""),
            xfail("sparse.mm", "reduce"),
            xfail("as_strided_scatter", ""),  # calls as_strided
            xfail("index_reduce", "prod"),  # .item() call
            xfail(
                "unbind_copy"
            ),  # Batching rule not implemented for aten::unbind_copy.int.
            # ---------------------------------------------------------------------
        }
    )

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @opsToleranceOverride(
        "TestOperators",
        "test_vmapvjp",
        (
            tol1(
                "linalg.svd",
                {torch.float32: tol(atol=5e-04, rtol=1e-04)},
                device_type="cuda",
            ),
            tol1(
                "svd", {torch.float32: tol(atol=5e-04, rtol=1e-04)}, device_type="cuda"
            ),
            tol1(
                "linalg.householder_product",
                {torch.float32: tol(atol=3e-04, rtol=9e-04)},
            ),
            tol1(
                "matrix_exp",
                {torch.float32: tol(atol=5e-04, rtol=1e-04)},
                device_type="cuda",
            ),
            tol1(
                "nn.functional.layer_norm",
                {torch.float32: tol(atol=3e-4, rtol=1e-4)},
                device_type="cpu",
            ),
            tol1(
                "native_layer_norm",
                {torch.float32: tol(atol=3e-4, rtol=1e-4)},
                device_type="cpu",
            ),
        ),
    )
    @skipOps(
        "TestOperators",
        "test_vmapvjp",
        vmapvjp_fail.union(
            {
                xfail("as_strided"),
                xfail("as_strided_copy"),
                xfail("as_strided", "partial_views"),
            }
        ),
    )
    def test_vmapvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return
        for sample in samples:
            cotangents = get_sample_cotangents(op, sample)
            fn, args = get_vjp_fn_and_args_with_cotangents(op, sample, cotangents)
            is_batch_norm_and_training = is_batch_norm_training(op.name, sample.kwargs)
            generator = get_fallback_and_vmap_exhaustive(
                fn, args, {}, is_batch_norm_and_training=is_batch_norm_and_training
            )
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    vmapjvpall_fail = {
        # -------------------- ALLOWED FAILURES --------------------------------
        # The following are expected (not a bug)
        skip("bernoulli", ""),  # randomness
        skip("nn.functional.dropout"),  # randomness
        skip("nn.functional.rrelu"),  # randomness
        skip("nn.functional.dropout2d", ""),
        skip("nn.functional.dropout3d", ""),
        skip("nn.functional.scaled_dot_produ
```



## High-Level Overview


This Python file contains 1 class(es) and 97 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestOperators`

**Functions defined**: `_autograd_grad`, `diff_arg`, `is_differentiable_arg`, `normalize_op_input_output2`, `wrapped`, `normalize_op_input_output3`, `wrapped`, `normalize_op_input_output`, `ref_vjp`, `wrapped`, `simulate_jvp`, `ref_jvp`, `get_sample_cotangents`, `get_vjp_fn_and_args_with_cotangents`, `wrapped`, `get_vjpfull_variant`, `get_vjpfull_variant2`, `_get_vjpfull_variant`, `wrapped`, `get_jvp_variant`

**Key imports**: functools, itertools, unittest, additional_op_db, torch, torch.autograd.forward_ad as fwAD, grad, jacfwd, jacrev, vjp, vmap, Tensor, _as_tuple, jvp, autograd_function_db


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `itertools`
- `unittest`
- `functorch_additional_op_db`: additional_op_db
- `torch`
- `torch.autograd.forward_ad as fwAD`
- `functorch`: grad, jacfwd, jacrev, vjp, vmap
- `torch._functorch.eager_transforms`: _as_tuple, jvp
- `torch.testing._internal.autograd_function_db`: autograd_function_db
- `torch.testing._internal.common_cuda`: with_tf32_off
- `torch.testing._internal.common_methods_invocations`: op_db
- `torch.testing._internal.opinfo.core`: SampleInput
- `torch.utils`: _pytree as pytree
- `torch.utils._pytree`: tree_flatten, tree_map, tree_unflatten


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/functorch/test_ops.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/functorch`):

- [`test_vmap.py_docs.md`](./test_vmap.py_docs.md)
- [`test_rearrange.py_docs.md`](./test_rearrange.py_docs.md)
- [`test_aot_joint_with_descriptors.py_docs.md`](./test_aot_joint_with_descriptors.py_docs.md)
- [`functorch_additional_op_db.py_docs.md`](./functorch_additional_op_db.py_docs.md)
- [`xfail_suggester.py_docs.md`](./xfail_suggester.py_docs.md)
- [`discover_coverage.py_docs.md`](./discover_coverage.py_docs.md)
- [`test_eager_transforms.py_docs.md`](./test_eager_transforms.py_docs.md)
- [`test_ac.py_docs.md`](./test_ac.py_docs.md)
- [`common_utils.py_docs.md`](./common_utils.py_docs.md)
- [`test_logging.py_docs.md`](./test_logging.py_docs.md)


## Cross-References

- **File Documentation**: `test_ops.py_docs.md`
- **Keyword Index**: `test_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
