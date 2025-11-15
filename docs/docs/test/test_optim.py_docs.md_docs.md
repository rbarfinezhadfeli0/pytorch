# Documentation: `docs/test/test_optim.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_optim.py_docs.md`
- **Size**: 53,775 bytes (52.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_optim.py`

## File Metadata

- **Path**: `test/test_optim.py`
- **Size**: 101,733 bytes (99.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: optimizer"]
import functools
import math
import tempfile
import unittest
from copy import deepcopy
from itertools import product
from typing import Any
from unittest.mock import patch

from optim.test_lrscheduler import TestLRScheduler  # noqa: F401
from optim.test_optim import TestDifferentiableOptimizer  # noqa: F401
from optim.test_swa_utils import TestSWAUtils  # noqa: F401

import torch
from torch.nn import Parameter
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipMPS,
    TEST_WITH_ROCM,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_optimizers import (
    _get_device_type,
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    OptimizerErrorEnum,
    optims,
    TensorTracker,
)
from torch.testing._internal.common_utils import (
    markDynamoStrictTest,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


FP16_REDUCED_PRECISION = {"atol": 1e-5, "rtol": 1e-4}


def rosenbrock(tensor):
    assert tensor.size() == torch.Size([2]), (
        f"Requires tensor with 2 scalars but got {tensor.size()}"
    )
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def drosenbrock(tensor):
    assert tensor.size() == torch.Size([2]), (
        f"Requires tensor with 2 scalars but got {tensor.size()}"
    )
    x, y = tensor
    return torch.stack((-400 * x * (y - x**2) - 2 * (1 - x), 200 * (y - x**2)))


@markDynamoStrictTest
class TestOptimRenewed(TestCase):
    """
    This test class validates the core optimizers and is structured as the correctness of:
    - The update algorithms (forloop implementation)
        * Every optimizer's algorithm is most readably implemented through a big for-loop
          over all the parameters, which is what we refer to as the forloop or single tensor
          implementation. These algorithms are manually validated by comparing to the paper
          and systematically validated by assuring that the loss goes the right direction
          when the optimizer has been applied.
        * This implementation should compose with optimizer hyperparameters well, such as
          supporting Tensor LRs, the capturable API, and sparse and complex parameters.
    - Each varying implementation
        * We then have implementations that improve upon the performance of the forloop
          implementation by leveraging fusion, namely our foreach (mult_tensor) and fused
          implementations.
        * These variations are validated numerically by comparing with the forloop version
          of the optimizer. In fact, we test most variations this way--we see the forloop
          implementation as the ground truth and expect that improvements to it in any way
          should be just as correct.
        * Both params and optimizer states should be validated numerically.
    - state_dict APIs
        * The optimizer instance should be serializable
        * Calling save and load should be deterministic
        * Moving between devices should be seamless
        * BC - load_state_dict should be able to handle older optimizer states
    - Hook APIs (everything should fire in the right order)
    - LR Scheduler integration (composing should not error + should go the right direction)
    - Parameter groups (should be equivalent to having multiple optimizers)
    - Erroring (what should error should error)

    We also cover different ways of generating parameters and grads:
    - With parameters, we either generate them randomly given specific shapes or we take
      them from a sample NN module.
        * Variety is important here because NN modules have type Parameter and randomly
          generated tensors have type Tensor.
        * Parameters can be sparse for a subset of the optimizers (check out OptimizerInfo)
        * Complex parameters should be handled using view_as_real
        * Parameters can be spread across different devices and different dtypes for any
          given optimizer
        * Parameters can be contiguous and noncontiguous
    - With grads, we follow suit from the parameters.
        * Grads can also be None, empty, or zero-valued, and this should not disrupt training.
    """

    @onlyCPU
    @optims(optim_db)
    def test_optim_infos_do_not_specify_global_cliquey_kwargs(
        self, device, dtype, optim_info
    ):
        global_cliquey_flags = ["foreach", "fused", "differentiable"]
        for optim_input in optim_info.optim_inputs_func(device=device):
            self.assertFalse(
                any(f for f in global_cliquey_flags if f in optim_input.kwargs)
            )

    @optims([optim for optim in optim_db if optim.optim_error_inputs_func is not None])
    def test_errors(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        error_inputs = optim_info.optim_error_inputs_func(device=device, dtype=dtype)

        for error_input in error_inputs:
            optim_input = error_input.optimizer_error_input
            params, kwargs = optim_input.params, optim_input.kwargs
            if error_input.error_on == OptimizerErrorEnum.CONSTRUCTION_ERROR:
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim_cls(params, **kwargs)
                else:
                    with self.assertRaisesRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim_cls(params, **kwargs)
            elif error_input.error_on == OptimizerErrorEnum.STEP_ERROR:
                optim = optim_cls(params, **kwargs)
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim.step()
                else:
                    with self.assertRaisesRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim.step()
            else:
                raise NotImplementedError(f"Unknown error type {error_input.error_on}")

    @parametrize("contiguous", [True, False])
    @parametrize("with_lrsched", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_forloop_goes_right_direction(
        self, device, dtype, optim_info, contiguous, with_lrsched
    ):
        optim_cls = optim_info.optim_cls
        schedulers_constructors = (
            optim_info.scheduler_inputs if with_lrsched else [None]
        )

        for schedulers_constructor in schedulers_constructors:
            # with tensor LR we need fresh inputs for each scheduler
            # or mutating it will carry across iters
            optim_inputs = optim_info.optim_inputs_func(device=device)
            for optim_input in optim_inputs:
                if "foreach" in optim_info.supported_impls:
                    optim_input.kwargs["foreach"] = False  # force forloop
                if contiguous:
                    weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
                    bias = Parameter(torch.randn((10), device=device, dtype=dtype))
                else:
                    weight = Parameter(
                        torch.randn((10, 5, 2), device=device, dtype=dtype)[..., 0]
                    )
                    bias = Parameter(
                        torch.randn((10, 2), device=device, dtype=dtype)[..., 0]
                    )
                input = torch.randn(5, device=device, dtype=dtype)

                params = [weight, bias] if optim_cls.__name__ != "Muon" else [weight]
                optimizer = optim_cls(params, **optim_input.kwargs)
                schedulers = [
                    s(optimizer)
                    for s in (schedulers_constructor if schedulers_constructor else [])
                ]

                def closure():
                    optimizer.zero_grad()
                    wo = (
                        weight.mv(input)
                        if optim_cls.__name__ == "Muon"
                        else weight.mv(input) + bias
                    )
                    loss = wo.pow(2).sum()
                    loss.backward()
                    if optim_info.only_supports_sparse_grads:
                        # For this test, we naively convert the Tensor layout, which we know does
                        # NOT represent the expected use case for optims like SparseAdam!
                        weight.grad = weight.grad.to_sparse()
                        bias.grad = bias.grad.to_sparse()
                    return loss

                initial_value = closure().item()
                for _ in range(20):
                    if optim_info.step_requires_closure:
                        loss = optimizer.step(closure)
                    else:
                        loss = closure()
                        optimizer.step()

                    for scheduler in schedulers:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                if optim_input.kwargs.get("maximize", False):
                    self.assertGreater(closure().item(), initial_value)
                else:
                    self.assertLess(closure().item(), initial_value)

    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @parametrize("with_lrsched", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_forloop_goes_right_direction_multigpu(
        self, device, dtype, optim_info, with_lrsched
    ):
        optim_cls = optim_info.optim_cls
        schedulers_constructors = (
            optim_info.scheduler_inputs if with_lrsched else [None]
        )
        for schedulers_constructor in schedulers_constructors:
            # We need a fresh set of inputs if we have a tensor LR
            # to not carry mutations across iterations.
            optim_inputs = optim_info.optim_inputs_func(device=device)
            for optim_input in optim_inputs:
                if "foreach" in optim_info.supported_impls:
                    optim_input.kwargs["foreach"] = False  # force forloop

                weight = Parameter(torch.randn((10, 5), device="cuda:0", dtype=dtype))
                bias = Parameter(torch.randn((10), device="cuda:1", dtype=dtype))
                inpt = torch.randn(5, device="cuda:0", dtype=dtype)

                params = [weight, bias] if optim_cls.__name__ != "Muon" else [weight]
                optimizer = optim_cls(params, **optim_input.kwargs)
                schedulers = [
                    s(optimizer)
                    for s in (schedulers_constructor if schedulers_constructor else [])
                ]

                def closure():
                    optimizer.zero_grad()
                    wo = (
                        weight.mv(inpt).cuda(1)
                        if optim_cls.__name__ == "Muon"
                        else weight.mv(inpt).cuda(1) + bias
                    )
                    loss = wo.pow(2).sum()
                    loss.backward()
                    if optim_info.only_supports_sparse_grads:
                        # For this test, we naively convert the Tensor layout, which we know does
                        # NOT represent the expected use case for optims like SparseAdam!
                        weight.grad = weight.grad.to_sparse()
                        bias.grad = bias.grad.to_sparse()
                    return loss

                initial_value = closure().item()
                for _ in range(20):
                    loss = optimizer.step(closure)
                    for scheduler in schedulers:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                if optim_input.kwargs.get("maximize", False):
                    self.assertGreater(closure().item(), initial_value)
                else:
                    self.assertLess(closure().item(), initial_value)

    @optims(optim_db, dtypes=[torch.float32])
    def test_param_group_with_lrscheduler_goes_right_direction(
        self, device, dtype, optim_info
    ):
        optim_cls = optim_info.optim_cls

        for schedulers_c in optim_info.scheduler_inputs:
            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            weight2 = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            inpt = torch.randn(5, device=device, dtype=dtype)

            # avoid endless recompiles by wrapping LR in a tensor if we're compiling
            lr = torch.tensor(0.01) if torch.compiler.is_compiling() else 0.01
            optimizer = optim_cls(
                [{"params": [weight]}, {"params": [weight2], "lr": lr}]
            )
            schedulers = [scheduler_c(optimizer) for scheduler_c in schedulers_c]

            def closure():
                optimizer.zero_grad()
                loss = (weight.mv(inpt) + weight2.mv(inpt)).pow(2).sum()
                loss.backward()
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
                    weight.grad = weight.grad.to_sparse()
                    weight2.grad = weight2.grad.to_sparse()
                return loss

            initial_value = closure().item()
            for _ in range(20):
                loss = optimizer.step(closure)
                for scheduler in schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss)
                    else:
                        scheduler.step()

            self.assertLess(closure().item(), initial_value)

    @parametrize("num_dim", [0, 1, 2])
    @optims(optim_db, dtypes=[torch.float32])
    def test_tensor_lr(self, device, dtype, optim_info, num_dim):
        optim_cls = optim_info.optim_cls

        lr_devices = [device]
        if _get_device_type(device) != "cpu":
            lr_devices.append("cpu")

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        for optim_input, lr_device in product(all_optim_inputs, lr_devices):
            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            weight_c = weight.detach().clone().requires_grad_(True)
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            bias_c = bias.detach().clone().requires_grad_(True)
            inpt = torch.randn(5, device=device, dtype=dtype)

            kwargs = optim_input.kwargs
            if "lr" in kwargs:
                del kwargs["lr"]

            params = [weight, bias] if optim_cls.__name__ != "Muon" else [weight]
            kwargs["lr"] = 1.0 if optim_info.step_requires_closure else 1e-3
            optimizer_r = optim_cls(params, **kwargs)

            try:
                kwargs["lr"] = (
                    torch.tensor(kwargs["lr"]).reshape([1] * num_dim).to(lr_device)
                )
                params_c = [weight_c, bias_c]
                if optim_cls.__name__ == "Muon":
                    params_c = [weight_c]
                optimizer = optim_cls(params_c, **kwargs)
            except ValueError as e:
                self.assertRegex(str(e), ".*lr as a Tensor is not supported.*")
                continue

            def closure(optim, w, b, i):
                optim.zero_grad()
                wo = w.mv(i) if optim_cls.__name__ == "Muon" else w.mv(i) + b
                loss = wo.pow(2).sum()
                loss.backward()
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
                    w.grad = w.grad.to_sparse()
                    b.grad = b.grad.to_sparse()
                return loss

            for _ in range(5):
                if optim_info.step_requires_closure:
                    optimizer_r.step(
                        functools.partial(closure, optimizer_r, weight, bias, inpt)
                    )
                    optimizer.step(
                        functools.partial(closure, optimizer, weight_c, bias_c, inpt)
                    )
                else:
                    closure(optimizer_r, weight, bias, inpt)
                    optimizer_r.step()
                    closure(optimizer, weight_c, bias_c, inpt)
                    optimizer.step()

                self.assertEqual(weight, weight_c)
                if optim_cls.__name__ != "Muon":
                    self.assertEqual(bias, bias_c)

    @parametrize("with_lrsched", [True, False])
    @optims(
        [o for o in optim_db if o.supports_sparse or o.only_supports_sparse_grads],
        dtypes=[torch.float64],
    )
    def test_rosenbrock_sparse(self, device, dtype, optim_info, with_lrsched):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # Fused impls do not support sparse gradients
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable", "fused")
        )
        kwarg_updates, schedulers_constructors = optim_info.metadata_for_sparse

        if with_lrsched and len(schedulers_constructors) == 0:
            return

        supported_inputs = []
        if len(kwarg_updates) != 0:
            seen = set()
            for i in all_optim_inputs:
                for k in kwarg_updates:
                    if k in i.kwargs:
                        del i.kwargs[k]
                hashable_kwargs = tuple(sorted(i.kwargs.items()))
                if len(i.kwargs) > 0 and hashable_kwargs not in seen:
                    supported_inputs.append(i)
                    seen.add(hashable_kwargs)
                    if "lr" in kwarg_updates:
                        i.kwargs["lr"] = kwarg_updates["lr"]
        else:
            supported_inputs = all_optim_inputs

        for optim_input in supported_inputs:
            kwargs = optim_input.kwargs
            multi_tensor = kwargs.get("foreach", False)

            # For rosenbrock tests, it is mandated that the param is a tensor with 2 numbers
            if multi_tensor:
                params_t = [
                    torch.tensor([1.5, 1.5]),
                    torch.tensor([1.5, 1.5], dtype=dtype),
                ]
            else:
                params_t = [torch.tensor([1.5, 1.5])]

            params = [Parameter(param_t) for param_t in params_t]
            optimizer = optim_cls(params, **kwargs)
            schedulers = [
                s(optimizer) for s in (schedulers_constructors if with_lrsched else [])
            ]

            if not optim_info.only_supports_sparse_grads:
                params_c = [Parameter(param_t.clone()) for param_t in params_t]
                optimizer_c = optim_cls(params_c, **kwargs)
                schedulers_c = [
                    s(optimizer_c)
                    for s in (schedulers_constructors if with_lrsched else [])
                ]

            solution = torch.tensor([1, 1])
            with torch.no_grad():
                initial_dist = sum(param.dist(solution) for param in params)

            def get_grad(param, sparse_grad, w):
                grad = drosenbrock(param)
                # NB: We torture test the optimizer by returning an
                # uncoalesced sparse tensor

                # Depending on w, provide only the x or y gradient
                if sparse_grad:
                    if w:
                        i = torch.tensor([[0, 0]], dtype=torch.int64)
                        x = grad[0]
                        v = torch.tensor([x / 4.0, x - x / 4.0])
                    else:
                        i = torch.tensor([[1, 1]], dtype=torch.int64)
                        y = grad[1]
                        v = torch.tensor([y - y / 4.0, y / 4.0])
                    grad_out = torch.sparse_coo_tensor(i, v, (2,), dtype=v.dtype)
                else:
                    if w:
                        grad_out = torch.tensor([grad[0], 0], dtype=param.dtype)
                    else:
                        grad_out = torch.tensor([0, grad[1]], dtype=param.dtype)
                return grad_out

            def eval(params, sparse_grad, w):
                optimizer.zero_grad()
                if multi_tensor:
                    loss = sum(rosenbrock(param) for param in params)
                else:
                    loss = rosenbrock(params[0])
                loss.backward()

                grads_out = [get_grad(param, sparse_grad, w) for param in params]
                with torch.no_grad():
                    params[0].grad = grads_out[0]
                    if multi_tensor:
                        params[1].grad = grads_out[1].to(dtype=dtype)
                return loss

            for i in range(1800):
                # Do cyclic coordinate descent
                w = i % 2
                optimizer.step(functools.partial(eval, params, True, w))
                for scheduler in schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(rosenbrock(params[0]))
                    else:
                        scheduler.step()
                if not optim_info.only_supports_sparse_grads:
                    optimizer_c.step(functools.partial(eval, params_c, False, w))
                    for scheduler in schedulers_c:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(rosenbrock(params_c[0]))
                        else:
                            scheduler.step()
                    # Tolerance is increased due to floating point error from different
                    # code path for dense case: x v.s. x - x / 4.0 + x / 4.0
                    self.assertEqual(params, params_c, atol=5e-6, rtol=5e-6)

            if not kwargs.get("maximize", False):
                self.assertLessEqual(
                    sum(param.dist(solution) for param in params), initial_dist
                )
            else:
                self.assertGreaterEqual(
                    sum(rosenbrock(param) for param in params),
                    sum(rosenbrock(param_t) for param_t in params_t),
                )

    @skipMPS
    @optims([o for o in optim_db if o.supports_complex], dtypes=[torch.complex64])
    def test_complex(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # Also skip fused, since our fused kernels do not support complex
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable", "fused")
        )
        for optim_input in all_optim_inputs:
            # Last param is intentionally real to test that we can mix real and complex
            complex_params = [
                torch.randn(10, 5, device=device, dtype=dtype, requires_grad=True),
                torch.randn(10, device=device, dtype=dtype, requires_grad=True),
                torch.randn(
                    10, 5, device=device, dtype=torch.float32, requires_grad=True
                ),
            ]
            real_params = [
                (
                    torch.view_as_real(param).detach().clone().requires_grad_()
                    if param.is_complex()
                    else param.detach().clone().requires_grad_()
                )
                for param in complex_params
            ]

            complex_optimizer = optim_cls(complex_params, **optim_input.kwargs)
            real_optimizer = optim_cls(real_params, **optim_input.kwargs)
            real_steps = []
            complex_steps = []
            grads_losses = []

            def real_closure():
                for param in real_params:
                    grad = torch.randn_like(param)
                    param.grad = grad
                    real_steps.append(param.detach().clone())
                    grads_losses.append(grad.clone())
                loss = torch.randn(1)
                grads_losses.append(loss.clone())
                return loss

            def complex_closure():
                for param in complex_params:
                    if torch.is_complex(param):
                        grad = torch.view_as_complex(grads_losses.pop(0))
                        complex_steps.append(torch.view_as_real_copy(param.detach()))
                    else:
                        grad = grads_losses.pop(0)
                        complex_steps.append(param.detach().clone())
                    param.grad = grad
                return grads_losses.pop(0)

            for _ in range(3):
                if optim_info.step_requires_closure:
                    # LBFGS, for example, requires closure and calls it internally
                    real_optimizer.step(real_closure)
                    complex_optimizer.step(complex_closure)
                else:
                    # For other optimizers, we call closure explicitly to set the gradients
                    real_closure()
                    complex_closure()
                    real_optimizer.step()
                    complex_optimizer.step()

            # Final Parameters should be the same
            complex_params_asreal = [
                torch.view_as_real(param) if param.is_complex() else param
                for param in complex_params
            ]
            self.assertEqual(real_params, complex_params_asreal)

            # All intermediate steps should also be the same
            # also checks steps taken within for example a line search
            self.assertEqual(complex_steps, real_steps)

    @skipMPS
    @optims([o for o in optim_db if o.supports_complex], dtypes=[torch.complex64])
    def test_complex_2d(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # Also skip fused, since our fused kernels do not support complex
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable", "fused")
        )
        for optim_input in all_optim_inputs:
            if optim_info.step_requires_closure:
                # Why? The way we implement complex is by turning complex params into view_as_real
                # alternatives. For example, an size (M,N) tensor will become (M,N,2). In this test,
                # we break apart a tensor into its real and imaginary parts, which would be 2x(M,N).
                # For other pointwise optimizers, this distinction is trivial, but for LBFGS where
                # there are reductions across all parameters (and all the grads get flattened into
                # one long Tensor), this ordering matters. Why? Reductions are not deterministic
                # because addition between floating point numbers is not associative, i.e.,
                # a + b + c != a + c + b. Thus, we add a seed here to control the discrepancy that
                # will happen with LBFGS. Note that in test_complex above, there is no need for a seed
                # nor for increased tolerance, because results should be bitwise equivalent.
                torch.manual_seed(2024)

            a1 = torch.randn(2, device=device, dtype=dtype, requires_grad=True)
            a1_real = a1.real.detach().clone()
            a1_imag = a1.imag.detach().clone()
            a1_real.requires_grad_()
            a1_imag.requires_grad_()
            optim1 = optim_cls([a1], **optim_input.kwargs)
            optim2 = optim_cls([a1_real, a1_imag], **optim_input.kwargs)

            a1_reals = TensorTracker()
            a1_imags = TensorTracker()
            a1_grad_reals = TensorTracker()
            a1_grad_imags = TensorTracker()
            losses = TensorTracker()

            def closure1():
                optim1.zero_grad()
                loss = rosenbrock(a1).abs()
                loss.backward()

                # Track clones to best test accuracy
                a1_reals.add(a1.real)
                a1_imags.add(a1.imag)
                a1_grad_reals.add(a1.grad.real)
                a1_grad_imags.add(a1.grad.imag)

                losses.add(loss)

                return loss

            def closure2():
                optim2.zero_grad()
                a1_reals.pop_check_set(a1_real, self)
                a1_imags.pop_check_set(a1_imag, self)
                a2 = torch.complex(a1_real, a1_imag)
                loss = rosenbrock(a2).abs()
                losses.pop_check_set(loss, self)
                loss.backward()
                a1_grad_reals.pop_check_set(a1_real.grad, self)
                a1_grad_imags.pop_check_set(a1_imag.grad, self)
                return loss

            for _ in range(3):
                if optim_info.step_requires_closure:
                    # LBFGS, for example, requires closure and calls it internally
                    optim1.step(closure1)
                    optim2.step(closure2)
                else:
                    closure1()
                    closure2()
                    optim1.step()
                    optim2.step()

                self.assertEqual(a1.real, a1_real)
                self.assertEqual(a1.imag, a1_imag)

            self.assertTrue(a1_reals.all_popped())
            self.assertTrue(a1_imags.all_popped())
            self.assertTrue(a1_grad_reals.all_popped())
            self.assertTrue(a1_grad_imags.all_popped())
            self.assertTrue(losses.all_popped())

    def test_adamw_serialization(self, device):
        model = torch.nn.Linear(5, 5).to(device)
        optim = torch.optim.AdamW(model.parameters())

        loaded_dict = optim.state_dict()

        # Test that Adam respects the decoupled_weight_decay key
        new_optim = torch.optim.Adam(model.parameters())
        new_optim.load_state_dict(loaded_dict)
        self.assertTrue(new_optim.param_groups[0]["decoupled_weight_decay"])

        # Test that decoupled_weight_decay is always True for AdamW
        adam_optim = torch.optim.Adam(model.parameters())
        adam_state_dict = adam_optim.state_dict()
        self.assertFalse(adam_state_dict["param_groups"][0]["decoupled_weight_decay"])

        new_optim = torch.optim.AdamW(model.parameters())
        new_optim.load_state_dict(adam_state_dict)
        self.assertTrue(new_optim.param_groups[0]["decoupled_weight_decay"])

        # Test that state_dicts from the old AdamW (with no decoupled_weight_decay key)
        # will have decoupled_weight_decay=True in new AdamW:
        old_adamw_dict = deepcopy(loaded_dict)
        del old_adamw_dict["param_groups"][0]["decoupled_weight_decay"]
        self.assertFalse("decoupled_weight_decay" in old_adamw_dict["param_groups"][0])

        new_optim = torch.optim.AdamW(model.parameters())
        new_optim.load_state_dict(old_adamw_dict)
        self.assertTrue(new_optim.param_groups[0]["decoupled_weight_decay"])

    def _compare_between(
        self, inputs, models, optimizers, assert_eq_kwargs=None, assert_step_dtype=None
    ):
        # why 7? iteration 7 is where we start to see differences for RAdam
        # params interacting with the small eps value, because that's right
        # after rho_t becomes greater than 5 in step 6.
        if assert_eq_kwargs is None:
            assert_eq_kwargs = {}
        kIterations = 7
        tracker = TensorTracker(assert_eq_kwargs)
        for i in range(kIterations):
            state, updated_params = [], []
            if not isinstance(inputs, list):
                inputs = [inputs, inputs]
            for input, model, optimizer in zip(inputs, models, optimizers):
                optimizer.zero_grad()

                if i == 3:
                    # Freeze a layer to test if the step of this layer in 'fused' or 'foreach'
                    # is same as the step in 'forloop'.
                    model[2].requires_grad_(False)
                if i == 5:
                    # Unfreeze the layer after 2 iters.
                    model[2].requires_grad_(True)

                # Test that step behaves as expected (a no-op) when grads are set to None
                if i != 2:
                    output = model(input)
                    loss = output.sum()
                    loss.backward()

                optimizer.step()
                state.append(optimizer.state)
                updated_params.append(model.parameters())

            og_state, new_state = state
            for og_p, new_p in zip(updated_params[0], updated_params[1]):
                tracker.add(og_p)
                tracker.pop_check_set(new_p, self)

                # check that optimizer states are the same
                og_p_state = og_state[og_p]
                new_p_state = new_state[new_p]
                if assert_step_dtype is not None:
                    if torch.is_tensor(og_p_state.get("step", None)):
                        self.assertEqual(og_p_state["step"].dtype, assert_step_dtype)
                    if torch.is_tensor(new_p_state.get("step", None)):
                        self.assertEqual(new_p_state["step"].dtype, assert_step_dtype)
                for k in og_p_state:
                    tracker.add(og_p_state[k])
                    tracker.pop_check_set(new_p_state[k], self)

            self.assertTrue(tracker.all_popped())

    def _test_derived_optimizers(
        self,
        device,
        dtype,
        optim_info,
        flag,
        reduced_precision=False,
        assert_step_dtype=None,
    ):
        """
        Given a flag 'fused' or 'foreach', test for parity of optimizer state
        and updated parameters between when the flag is set to True and False
        for provided optimizer configurations.
        """
        assert flag in ("foreach", "fused")
        assert_eq_kwargs = {} if not reduced_precision else FP16_REDUCED_PRECISION

        optim_inputs = optim_info.optim_inputs_func(device=device, dtype=dtype)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            models, optimizers = [], []
            kwargs = deepcopy(optim_input.kwargs)
            if kwargs.get("capturable", False) and _get_device_type(device) == "cpu":
                # capturable is not supported on CPU
                continue
            for flag_value in (False, True):
                kwargs[flag] = flag_value
                input = torch.tensor(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype, device=device
                ).reshape(3, 2)

                torch.manual_seed(1)
                model = torch.nn.Sequential(
                    torch.nn.Linear(2, 3),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(3, 1),
                    torch.nn.Sigmoid(),
                )
                model.to(dtype=dtype, device=device)

                # foreach/fused optimizers should be tested with a
                # zero_size tensor as its last param.
                # ref: https://github.com/pytorch/pytorch/issues/100701
                empty_param = torch.empty(
                    (), device=device, dtype=dtype, requires_grad=True
                )
                empty_param.grad = torch.rand_like(empty_param)
                params = list(model.parameters()) + [empty_param]

                optimizer = optim_cls(params, **kwargs)
                models.append(model)
                optimizers.append(optimizer)

            self._compare_between(
                input, models, optimizers, assert_eq_kwargs, assert_step_dtype
            )

    @skipMPS  # MPS doesn't support torch.float64, see https://github.com/pytorch/pytorch/issues/115350
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float64],
    )
    def test_foreach_matches_forloop(self, device, dtype, optim_info):
        self._test_derived_optimizers(device, dtype, optim_info, "foreach")

    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @parametrize("impl", ["foreach", "fused"])
    @optims(
        [
            optim
            for optim in optim_db
            if "foreach" in optim.supported_impls or "fused" in optim.supported_impls
        ]
    )
    def test_mixed_device_dtype(self, device, dtype, optim_info, impl):
        """
        Similar in essence to _test_derived_optimizers above. The main difference is that
        _test_derived_optimizers uses model parameters whereas we randomly pass in
        parameters of different dtypes and devices here. We need multiple GPUs (vs just a
        CPU and GPU) because fused adam only works on GPUs. (Thus we only run the tests
        that call into this helper when TEST_MULTIGPU.)
        """
        assert impl in ("foreach", "fused")
        if impl == "foreach" and "foreach" not in optim_info.supported_impls:
            return unittest.skip(
                f"foreach not supported for {optim_info.optim_cls.__name__}"
            )
        elif impl == "fused" and "cuda" not in optim_info.supports_fused_on:
            return unittest.skip(
                f"fused not supported for {optim_info.optim_cls.__name__} on cuda"
            )

        params = [
            torch.rand(2, 3, dtype=torch.float64, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float32, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float16, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.bfloat16, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float64, device="cuda:1", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float32, device="cuda:1", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float16, device="cuda:1", requires_grad=True),
            torch.rand(2, 3, dtype=torch.bfloat16, device="cuda:1", requires_grad=True),
            torch.randint(
                1024, (2, 3), dtype=torch.int64, device="cuda:1", requires_grad=False
            ),
        ]

        for p in params:
            if p.requires_grad:
                p.grad = torch.rand_like(p, device=p.device, dtype=p.dtype)

        kIterations = 7 if impl == "foreach" else 1
        optim_inputs = optim_info.optim_inputs_func(device=device)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            updated_params, state = [], []
            kwargs = deepcopy(optim_input.kwargs)
            if kwargs.get("capturable", False) and _get_device_type(device) == "cpu":
                # capturable is not supported on CPU
                continue
            for use_impl in (False, True):
                kwargs[impl] = use_impl
                params_clone = []
                for p in params:
                    p_clone = p.detach().clone()
                    if p.requires_grad:
                        p_clone.requires_grad = True
                        p_clone.grad = p.grad.detach().clone()
                        params_clone.append(p_clone)

                optimizer = optim_cls(params_clone, **kwargs)
                for _ in range(kIterations):
                    optimizer.step()

                state.append(optimizer.state)
                updated_params.append(params_clone)

            og_state, new_state = state
            for og_p, new_p in zip(updated_params[0], updated_params[1]):
                # Increasing the tolerance as we are collating lots of ops together for optimizers and
                # the designated tolerances are for single op only.
                single_rtol, single_atol = torch.testing._comparison.get_tolerances(
                    new_p.dtype, rtol=None, atol=None
                )
                rtol = 5 * single_rtol
                atol = 5 * single_atol

                self.assertEqual(og_p, new_p, rtol=rtol, atol=atol)

                # check that optimizer states are the same
                og_p_state = og_state[og_p]
                new_p_state = new_state[new_p]

                for k in og_p_state:
                    actual = new_p_state[k]
                    self.assertEqual(og_p_state[k], actual, rtol=rtol, atol=atol)

    @onlyCUDA
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float64],
    )
    def test_set_default_dtype_works_with_foreach(self, device, dtype, optim_info):
        # https://github.com/pytorch/pytorch/issues/110940
        # We coerce step to always be float32 unless the
        # default dtype is higher prec float64
        old_default_dtype = torch.get_default_dtype()
        for default_dtype in [torch.float64, torch.float16]:
            try:
                torch.set_default_dtype(default_dtype)
                self._test_derived_optimizers(
                    device,
                    dtype,
                    optim_info,
                    "foreach",
                    reduced_precision=default_dtype == torch.float16,
                    assert_step_dtype=(
                        torch.float64
                        if default_dtype == torch.float64
                        else torch.float32
                    ),
                )
            finally:
                torch.set_default_dtype(old_default_dtype)

    @onlyCUDA
    @largeTensorTest("72GB", "cuda")
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float16],
    )
    def test_foreach_large_tensor(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        for optim_input in optim_inputs:
            params = [torch.ones(2**32, device=device, dtype=dtype)]
            params[0].grad = torch.zeros_like(params[0])
            optimizer = optim_cls(params, foreach=True, **optim_input.kwargs)
            optimizer.step()

    @onlyCUDA
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    def test_peak_memory_foreach(self, device, dtype, optim_info):
        nparams = 10
        optim_inputs = optim_info.optim_inputs_func(device=device)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            kwargs = deepcopy(optim_input.kwargs)
            max_mems = []
            for flag_value in (False, True):
                kwargs["foreach"] = flag_value
                # The 16 * 8 = 128 is critical here! Our CUDACachingAllocator allocates in blocks
                # of 512, meaning any tensor that occupies <512 bytes of memory will allocate a
                # whole 512 bytes anyway. We use 128 (cuz datasize would be 4 bytes) so that param
                # is size 512 exactly, making our later calculations for intermediate_size easy.
                param = torch.rand(16, 8, device=device, dtype=dtype)
                params = [torch.rand_like(param) for _ in range(nparams)]

                optimizer = optim_cls(params, **kwargs)

                for p in params:
                    p.grad = torch.rand_like(p)

                optimizer.step()
                import gc

                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                optimizer.step()
                gc.collect()
                max_mems.append(torch.cuda.max_memory_allocated())

            st_max_mem, mt_max_mem = max_mems
            intermediate_size = nparams * param.nelement() * param.element_size()
            nintermediates = 1  # we expect a budget of 1 intermediate most of the time

            # Check the param group directly to handle if the compiler set capturable
            if optimizer.param_groups[0].get(
                "capturable", False
            ) or optim_cls.__name__ in ["Adadelta", "ASGD", "RAdam"]:
                # with capturable in Adam(W), we have 2 extra intermediates for the bias_corrections
                # with Adadelta, we have 2 extra for (acc_delta + eps) and (square_avg + eps)
                # ASGD allocates axs, 2x mus, 2x etas, and grads at the same time
                nintermediates = 3
                if optim_cls.__name__ == "NAdam":
                    # with capturable in NAdam, we have 3 extra intermediates for the
                    # bias_correction, mus, and mu_nexts
                    if TEST_WITH_TORCHDYNAMO:
                        # With dynamo, the eager/FX backend appears to hold memory longer than
                        # vanilla eager: https://github.com/pytorch/pytorch/issues/125511
                        nintermediates = 8
                    else:
                        nintermediates = 5

                if optim_cls.__name__ == "RAdam":
                    # RAdam has four intermediates with capturable
                    # num, unrect_step_size, buffer, grouped_grads
                    if TEST_WITH_TORCHDYNAMO:
                        # With dynamo, the eager/FX backend appears to hold memory than
                        # vanilla eager: https://github.com/pytorch/pytorch/issues/125511
                        nintermediates = 6
                    else:
                        nintermediates = 4

            elif optim_cls.__name__ in ["NAdam", "Adagrad", "RMSprop", "Adafactor"]:
                # NAdam uses two intermediates at the same time (grads & exp_avg_sq_sqrt)
                # Adagrad uses std and grads at the same time
                # RMSprop uses avg and grads
                # Adafactor uses row/col var and its mean
                nintermediates = 2

                if optim_cls.__name__ == "Adafactor" and kwargs.get("maximize", False):
                    # When maximize is True, Adafactor also tracks device_grad
                    nintermediates = 3

            # Dynamo ST uses less mem than eager in the case of Adam/Adagrad/Nadam/RAdam
            # which makes the foreach memory check fail
            if TEST_WITH_TORCHDYNAMO:
                st_max_mem += 6000

            expected_max_mem = st_max_mem + intermediate_size * nintermediates
            # hipcc currently can't generate efficient code for the small buffer optimization
            # code path (see Note [small buffer optimization] for details), thus we always
            # dynamically allocate the tensor metadata for ROCM. Adjusting the expected max
            # memory usage to account for this.
            if TEST_WITH_ROCM:
                expected_max_mem *= 1.02

            self.assertLessEqual(mt_max_mem, expected_max_mem)

    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=floating_types_and(
            torch.bfloat16,
            torch.float16,
        ),
    )
    def test_fused_matches_forloop(self, device, dtype, optim_info):
        if _get_device_type(device) not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        if _get_device_type(device) == "mps" and dtype not in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ):
            self.skipTest(
                "MPS supports only torch.float16, torch.float32 and torch.bfloat16"
            )
        self._test_derived_optimizers(device, dtype, optim_info, "fused")

    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=(torch.float32,),
    )
    def test_fused_error_on_params_on_meta(self, device, dtype, optim_info):
        if _get_device_type(device) not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )

        with torch.device("meta"):
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 3),
                torch.nn.Sigmoid(),
                torch.nn.Linear(3, 1),
                torc
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_optim.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_optim.py_docs.md_docs.md`
- **Keyword Index**: `test_optim.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
