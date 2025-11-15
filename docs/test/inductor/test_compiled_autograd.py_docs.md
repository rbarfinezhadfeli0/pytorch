# Documentation: `test/inductor/test_compiled_autograd.py`

## File Metadata

- **Path**: `test/inductor/test_compiled_autograd.py`
- **Size**: 206,204 bytes (201.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import contextlib
import dataclasses
import functools
import io
import itertools
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from importlib.machinery import SourceFileLoader
from pathlib import Path
from string import Template
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd, config
from torch._dynamo.backends.debugging import aot_eager
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.cpp_builder import is_msvc_cl
from torch._inductor.test_case import run_tests, TestCase
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.overrides import BaseTorchFunctionMode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_S390X,
    IS_WINDOWS,
    parametrize,
    scoped_load_inline,
    skipIfWindows,
)
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
)
from torch.testing._internal.logging_utils import logs_to_string
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.utils._python_dispatch import TorchDispatchMode


# note: these tests are not run on windows due to inductor_utils.HAS_CPU


def make_compiler_fn(
    fullgraph=True, dynamic=True, backend="inductor", gm_hook=lambda gm: None
):
    assert backend in ["inductor", "aot_eager", "eager", "ca_eager"]

    def _compiler_fn(gm):
        """Same as torch.compile() but counts number of compiles"""
        gm_hook(gm)

        _backend = backend
        if backend == "ca_eager":
            return gm
        elif backend != "eager":

            def _inner_compiler(gm_, example_inputs_):
                counters["compiled_autograd"]["compiles"] += 1
                if backend == "inductor":
                    return inductor.compile(gm_, example_inputs_)
                elif backend == "aot_eager":
                    return aot_eager(gm_, example_inputs_)

            _backend = _inner_compiler

        return torch.compile(gm, backend=_backend, fullgraph=fullgraph, dynamic=dynamic)

    return _compiler_fn


compiler_fn = make_compiler_fn()


# TODO(jansel): hooks as lambdas creates recompiles in dynamo, we should fix that
def hook1(grad):
    return grad * 2


def hook2(grads):
    return (grads[0] + 1,)


def hook3(gI, gO):
    return (torch.sin(gI[0]) + gO[0],)


def reset():
    torch._logging.set_logs(compiled_autograd_verbose=False)
    config.compiled_autograd = False
    compiled_autograd.reset()
    torch._dynamo.utils.counters.clear()


class BaseCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("must override")


class TestCompiledAutograd(TestCase):
    def setUp(self) -> None:
        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(config.patch("record_runtime_overhead", False))
        super().setUp()
        reset()

    def tearDown(self) -> None:
        self.exit_stack.close()
        super().tearDown()
        reset()

    def check_output_and_recompiles(
        self, fn, count=1, compiler_fn=compiler_fn, compile_fn=False
    ):
        if isinstance(count, list):
            captures, compiles = count
        else:
            captures, compiles = count, count
        with torch.autograd.set_multithreading_enabled(False):
            torch._dynamo.reset()
            counters["compiled_autograd"].clear()
            torch.manual_seed(123)
            expected = list(fn())
            torch.manual_seed(123)
            with (
                compiled_autograd._enable(compiler_fn),
                mock.patch(
                    "torch._functorch.aot_autograd.AOT_COUNTER",
                    new_callable=itertools.count,
                ),
            ):
                opt_fn = torch.compile(fn) if compile_fn else fn
                actual = list(opt_fn())
            self.assertEqual(expected, actual)
            self.assertEqual(counters["compiled_autograd"]["captures"], captures)
            self.assertEqual(counters["compiled_autograd"]["compiles"], compiles)

    def run_as_subprocess(self, script) -> bytes:
        try:
            return subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            self.fail(f"Subprocess exited with return code: {e.returncode}")

    def test_hipify_not_loaded_with_import_torch(self):
        script = """
import torch
assert globals().get("hipify", False) is False
"""
        self.run_as_subprocess(script)

    def test_hipify_not_loaded_with_import_cpp_extension(self):
        script = """
import torch.utils.cpp_extension
assert globals().get("hipify", False) is False
"""
        self.run_as_subprocess(script)

    def test_dynamo_flaky_segfault(self):
        script = """
import torch

def main():
    def compiler_fn(gm):
        return torch.compile(gm, backend="eager")

    def inner():
        x = torch.randn(1000, 3000)
        w = torch.randn(1000, 3000, requires_grad=True)
        def model(i):
            return torch.nn.functional.linear(i, w)
        out = model(x)
        loss = out.sum()
        with torch._dynamo.compiled_autograd._enable(compiler_fn):
            loss.backward()
        assert(w.grad is not None)

    inner()
    torch._dynamo.reset()
    inner()

main()
        """
        # Run it three times to catch bad dynamo state resets
        for _ in range(3):
            self.run_as_subprocess(script)

    def gen_cache_miss_log_prefix(self):
        if IS_WINDOWS:
            if is_msvc_cl():
                return "Cache miss due to new autograd node: struct "
            else:
                self.fail(
                    "Compilers other than msvc have not yet been verified on Windows."
                )
                return ""
        else:
            return "Cache miss due to new autograd node: "

    def test_reset(self):
        compiled_autograd.compiled_autograd_enabled = True
        torch._C._dynamo.compiled_autograd.set_autograd_compiler(lambda: None, True)
        # TODO: return prior verbose logger
        # torch._C._dynamo.compiled_autograd.set_verbose_logger(dummy)
        compiled_autograd.COMPILE_COUNTER = None

        # state should be clean after reset
        compiled_autograd.reset()

        assert compiled_autograd.compiled_autograd_enabled is False
        (
            prior_compiler,
            prior_dynamic,
        ) = torch._C._dynamo.compiled_autograd.set_autograd_compiler(None, False)
        assert prior_compiler is None
        assert prior_dynamic is False
        assert (
            compiled_autograd.COMPILE_COUNTER is not None
            and next(compiled_autograd.COMPILE_COUNTER) == 0
        )

    def test_basic(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])
            result = model(x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        self.check_output_and_recompiles(fn)

    def test_cache_hit(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([2, 4])
                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad

        self.check_output_and_recompiles(fn)

    def test_graph_break_custom_op(self):
        @torch.library.custom_op("mylib::sin", mutates_args={})
        def sin(x: torch.Tensor) -> torch.Tensor:
            return x.sin()

        def setup_context(ctx, inputs, output):
            (x,) = inputs
            ctx.save_for_backward(x)

        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * x.cos()

        sin.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        y = sin(x.clone()).sum()
        with compiled_autograd._enable(compiler_fn):
            y.backward()

    def test_tensor_grad_hook1(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([2, 4])

                model[0].weight.register_hook(hook1)

                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook2(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.grad_fn.register_prehook(hook2)
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook3(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.grad_fn.register_hook(hook3)
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_reorder_acc_grad(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3, bias=True),
            torch.nn.Conv2d(4, 4, 3, bias=True),
        )
        compiled_model = torch.compile(model)
        x = torch.randn([1, 4, 32, 32])

        model(x).sum().backward()
        ref_res = [
            model[0].weight.grad,
            model[0].bias.grad,
            model[1].weight.grad,
            model[1].bias.grad,
        ]

        model[0].weight.grad = None
        model[0].bias.grad = None
        model[1].weight.grad = None
        model[1].bias.grad = None
        with compiled_autograd._enable(compiler_fn):
            compiled_model(x).sum().backward(retain_graph=True)
        res = [
            model[0].weight.grad,
            model[0].bias.grad,
            model[1].weight.grad,
            model[1].bias.grad,
        ]

        self.assertEqual(res[0], ref_res[0])
        self.assertEqual(res[1], ref_res[1])
        self.assertEqual(res[2], ref_res[2])
        self.assertEqual(res[3], ref_res[3])

    def test_reorder_post_hook1(self):
        def grad_div(param):
            param.grad = param.grad / 4.0

        class Module(torch.nn.Module):
            def __init__(self, ioc):
                super().__init__()
                self.fc1 = torch.nn.Linear(ioc, ioc, bias=False)
                self.fc2 = torch.nn.Linear(ioc, ioc, bias=False)

                self.grad_acc_hooks = []
                self.grad_acc = []
                self.params = [self.fc1.weight, self.fc2.weight]
                for param in self.params:

                    def wrapper(param):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def grad_acc_hook(*notneeded):
                            grad_div(param)

                        self.grad_acc.append(grad_acc)
                        self.grad_acc_hooks.append(
                            grad_acc.register_hook(grad_acc_hook)
                        )

                    wrapper(param)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x.sum()

        bs = 8
        ioc = 16
        model = Module(ioc)
        input = torch.randn([bs, ioc])

        # eager ref
        model(input).backward()
        ref_res = [model.fc1.weight.grad, model.fc2.weight.grad]

        # cag
        model.fc1.weight.grad = None
        model.fc2.weight.grad = None
        model_to_train = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            model_to_train(input).backward()
        res = [model_to_train.fc1.weight.grad, model_to_train.fc2.weight.grad]

        self.assertEqual(res[0], ref_res[0])
        self.assertEqual(res[1], ref_res[1])

    def test_reorder_post_hook2(self):
        x = torch.randn([1, 4, 32, 32], requires_grad=True)
        y = torch.sigmoid(x)
        z = torch.tanh(y)

        assert isinstance(z.grad_fn, torch.autograd.graph.Node)
        assert isinstance(y.grad_fn, torch.autograd.graph.Node)
        handle_z = z.grad_fn.register_hook(lambda gI, gO: (gO[0] * 2,))
        handle_y = y.grad_fn.register_hook(lambda gI, gO: (gI[0] * 2,))
        z.sum().backward(retain_graph=True)
        ref_res = x.grad

        x.grad = None
        with compiled_autograd._enable(compiler_fn):
            z.sum().backward(retain_graph=True)
        res = x.grad

        self.assertEqual(res, ref_res)

    def test_reorder_post_hook3(self):
        conv = torch.nn.Conv2d(4, 4, 3, bias=False)
        x = torch.randn([1, 4, 32, 32])
        y = conv(x)

        assert isinstance(y.grad_fn, torch.autograd.graph.Node)
        # this hook will mul 2.0 to the conv weight gradient
        handle_y = y.grad_fn.register_hook(lambda gI, gO: (gI[0], gI[1] * 2, gI[2]))
        y.sum().backward(retain_graph=True)
        ref_res = x.grad

        x.grad = None
        with compiled_autograd._enable(compiler_fn):
            y.sum().backward(retain_graph=True)
        res = x.grad

        self.assertEqual(res, ref_res)

    def test_reorder_all_bwd_hooks(self):
        def tensor_hook(grad):
            return grad.sub(2.0)

        def acc_grad_node_pre_hook(grad_out):
            return (grad_out[0].div(5.0),)

        def post_acc_grad_hook(tensor):
            tensor.grad.add_(3.0)

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]
                self.conv1.weight.register_hook(tensor_hook)
                self.conv1.weight.register_post_accumulate_grad_hook(post_acc_grad_hook)
                self.acc_grad1.register_prehook(acc_grad_node_pre_hook)

                def acc_grad_node_post_hook1(grad_in, grad_out):
                    self.conv1.weight.grad.mul_(0.5)

                self.acc_grad1.register_hook(acc_grad_node_post_hook1)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]
                self.conv2.weight.register_hook(tensor_hook)
                self.conv2.weight.register_post_accumulate_grad_hook(post_acc_grad_hook)
                self.acc_grad2.register_prehook(acc_grad_node_pre_hook)

                def acc_grad_node_post_hook2(grad_in, grad_out):
                    self.conv2.weight.grad.mul_(0.5)

                self.acc_grad2.register_hook(acc_grad_node_post_hook2)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_reorder_multi_post_hooks(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]

                def acc_grad_node1_post_hook1(grad_in, grad_out):
                    self.conv1.weight.grad.mul_(0.5)

                def acc_grad_node1_post_hook2(grad_in, grad_out):
                    self.conv1.weight.grad.sub_(0.3)

                self.acc_grad1.register_hook(acc_grad_node1_post_hook1)
                self.acc_grad1.register_hook(acc_grad_node1_post_hook2)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]

                def acc_grad_node2_post_hook1(grad_in, grad_out):
                    self.conv2.weight.grad.mul_(0.3)

                def acc_grad_node2_post_hook2(grad_in, grad_out):
                    self.conv2.weight.grad.sub_(0.5)

                self.acc_grad2.register_hook(acc_grad_node2_post_hook1)
                self.acc_grad2.register_hook(acc_grad_node2_post_hook2)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_reorder_multi_pre_hooks(self):
        def acc_grad_node_pre_hook1(grad_out):
            return (grad_out[0].div(5.0),)

        def acc_grad_node_pre_hook2(grad_out):
            return (grad_out[0].sub(0.3),)

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]
                self.acc_grad1.register_prehook(acc_grad_node_pre_hook1)
                self.acc_grad1.register_prehook(acc_grad_node_pre_hook2)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]
                self.acc_grad2.register_prehook(acc_grad_node_pre_hook1)
                self.acc_grad2.register_prehook(acc_grad_node_pre_hook2)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_reorder_multi_tensor_pre_hooks(self):
        def tensor_hook1(grad):
            return grad.sub(2.0)

        def tensor_hook2(grad):
            return grad.mul(0.5)

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]
                self.conv1.weight.register_hook(tensor_hook1)
                self.conv1.weight.register_hook(tensor_hook2)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]
                self.conv2.weight.register_hook(tensor_hook1)
                self.conv2.weight.register_hook(tensor_hook2)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_torch_compile(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.Sigmoid(),
            )
            opt_model = torch.compile(model, fullgraph=True)

            for _ in range(3):
                x = torch.randn([1, 4])

                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                model.zero_grad()

        self.check_output_and_recompiles(fn)

    @parametrize("api", ("compile", "optimize"))
    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    def test_compile_api(self, api, backend):
        def wrap(fn, backend):
            if api == "compile":
                return torch.compile(fn, backend=backend)
            elif api == "optimize":
                return torch._dynamo.optimize(backend)(fn)

        def fn(model, inputs):
            res = []
            for inp in inputs:
                result = model(inp).sum()
                result.backward()
                res.append(model[0].weight.grad)
                res.append(model[0].bias.grad)
                model.zero_grad()
            return res

        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        inputs = [
            torch.randn([1, 4]),
            torch.randn([2, 4]),
            torch.randn([3, 4]),
        ]

        expected = fn(model, inputs)
        with config.patch(compiled_autograd=True):
            compiled_fn = wrap(fn, backend)
        actual = compiled_fn(model, inputs)
        self.assertEqual(expected, actual)
        self.assertEqual(counters["compiled_autograd"]["captures"], 2)

    @parametrize("api", ("compile", "optimize"))
    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    def test_compile_api_disable(self, api, backend):
        def wrap(fn, backend):
            if api == "compile":
                return torch.compile(fn, backend=backend)
            elif api == "optimize":
                return torch._dynamo.optimize(backend)(fn)

        def fn(model, inputs):
            res = []
            for inp in inputs:
                result = model(inp).sum()
                result.backward()
                res.append(model[0].weight.grad)
                res.append(model[0].bias.grad)
                model.zero_grad()
            return res

        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        inputs = [
            torch.randn([1, 4]),
            torch.randn([2, 4]),
            torch.randn([3, 4]),
        ]

        expected = fn(model, inputs)
        with config.patch(compiled_autograd=True):
            compiled_fn = wrap(fn, backend)
        with torch._dynamo.compiled_autograd._disable():
            actual = compiled_fn(model, inputs)
        self.assertEqual(expected, actual)
        self.assertTrue("compiled_autograd" not in counters)

    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    def test_optimize_assert(self, backend):
        # can be merged into the test above once we support
        # no graph break on .backward

        def fn(model, inp):
            # NOTE: not calling .backward in the compiled fn
            return model(inp).sum()

        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        inp = torch.randn([1, 4])

        out = fn(model, inp)
        out.backward()
        expected = [p.grad for p in model.parameters()]
        model.zero_grad()
        with config.patch(compiled_autograd=True):
            compiled_fn = torch._dynamo.optimize_assert(backend)(fn)

        # should not error due to undefined `rebuild_ctx`
        out = compiled_fn(model, inp)
        out.backward()
        actual = [p.grad for p in model.parameters()]
        self.assertEqual(expected, actual)
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)

    @config.patch(compiled_autograd=True)
    def test_nested_context_manager(self):
        def ctx():
            return compiled_autograd._enable(torch.compile)

        # ok
        outer = ctx()
        inner = ctx()
        outer.__enter__()
        inner.__enter__()
        inner.__exit__(None, None, None)
        outer.__exit__(None, None, None)

        # not ok
        outer = ctx()
        inner = ctx()
        outer.__enter__()
        inner.__enter__()
        with self.assertRaisesRegex(
            AssertionError,
            "Nested Compiled Autograd Contexts must return before their parent context",
        ):
            outer.__exit__(None, None, None)

    @config.patch(compiled_autograd=True)
    def test_nested_compile(self):
        with torch.library._scoped_library("testlib", "FRAGMENT") as lib:
            lib.define("square(Tensor x) -> Tensor")

            @torch.library.impl("testlib::square", "CPU")
            def square_impl(x: torch.Tensor) -> torch.Tensor:
                # nested inference graph compile
                @torch.compile(backend="eager")
                def fn(x):
                    return x**2

                return fn(x)

            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, x):
                    return torch.ops.testlib.square(x)

            x = torch.tensor([2.0, 3.0], requires_grad=True)

            @torch.compile
            def fn(x):
                return MyFn.apply(x)

            fn(x).sum().backward()

    @config.patch(compiled_autograd=True)
    def test_no_nested_compiled_autograd(self):
        # We disable CA before entering the CA graph
        # So re-entrants should be running with the eager autograd engine

        def unrelated_autograd_call():
            x = torch.randn(20, 20, requires_grad=True)
            y = torch.randn(20, 20, requires_grad=True)
            loss = torch.matmul(x, y).sum()
            loss.backward()

        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gO):
                unrelated_autograd_call()
                return gO

        x = torch.randn(10, 10, requires_grad=True)
        loss = MyFn.apply(x).sum()

        torch.compile(lambda: loss.backward(create_graph=True))()
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_multiple_torch_compile(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        def fn():
            result = model(x).sum()
            result.backward()

        model2 = torch.nn.Linear(4, 4)
        x2 = torch.randn([1, 4])

        def fn2():
            result = model2(x2).sum()
            result.backward()

        no_ca1 = torch.compile(fn)
        no_ca1()
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        counters.clear()

        with config.patch(compiled_autograd=True):
            with_ca = torch.compile(fn2)
            with_ca()
            self.assertEqual(counters["compiled_autograd"]["captures"], 1)
            counters.clear()

        no_ca2 = torch.compile(fn)
        no_ca2()
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)

    def test_torch_compile_graph_break(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        @torch._dynamo.disable()
        def fn():
            result = model(x).sum()
            result.backward()

        with config.patch(compiled_autograd=True):
            opt_fn = torch.compile(fn)
            opt_fn()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_graph_break2(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        @torch._dynamo.disable()
        def inner_fn(loss):
            loss.backward()

        def fn():
            result = model(x).sum()
            inner_fn(result)

        with config.patch(compiled_autograd=True):
            opt_fn = torch.compile(fn)
            opt_fn()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_only_backward_call(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        result = model(x).sum()
        with config.patch(compiled_autograd=True):
            opt_bwd = torch.compile(lambda: result.backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_dynamo_boxed(self):
        def get_placeholders(gm_):
            placeholders = []
            for node in gm_.graph.nodes:
                if node.op == "placeholder":
                    placeholders.append(node)
            return placeholders

        def eager_with_check(gm, is_bwd):
            def inner_compiler(gm_, example_inputs_):
                placeholders = get_placeholders(gm_)
                if is_bwd:
                    # boxed inputs
                    assert isinstance(placeholders[0].meta["example_value"], list)
                else:
                    # not boxed inputs
                    assert not isinstance(placeholders[0].meta["example_value"], list)

                return gm_

            return torch.compile(gm, backend=inner_compiler)

        bwd_compiler_fn = functools.partial(eager_with_check, is_bwd=True)

        def fn(inputs):
            args_0, args_1, args_2 = inputs
            out = torch.mm(args_0, args_1)
            out = torch.mm(out, args_2)
            loss = out.sum()
            with compiled_autograd._enable(bwd_compiler_fn):
                loss.backward()
            yield args_0.grad
            yield args_1.grad
            yield args_2.grad

        inputs = [
            torch.randn([1, 2], requires_grad=True),
            torch.randn([2, 3], requires_grad=True),
            torch.randn([3, 4], requires_grad=True),
        ]

        compiled_fn = eager_with_check(fn, is_bwd=False)
        grads = list(compiled_fn(inputs))
        self.assertEqual(len(grads), 3)
        self.assertNotEqual(grads[0], None)
        self.assertNotEqual(grads[1], None)
        self.assertNotEqual(grads[2], None)

    def test_inputs_aliasing_bytecode_attr_mutations(self):
        # Freeze compiled autograd graph
        compiler = torch._dynamo.compiled_autograd.AutogradCompilerInstance(compiler_fn)
        param = torch.ones(100)
        active = torch.ones(100) * 2
        inputs = [param, active]
        _, proxies, _, _ = compiler.begin_capture(
            inputs=inputs,
            sizes=[],
            scalars=[],
            origins=[[], [], []],
            accumulate_grad=False,
            check_nans=False,
        )
        param_proxy, activ_proxy = proxies
        buf = activ_proxy * 2
        torch.ops.inductor.accumulate_grad_.default(param_proxy, buf)
        runtime_wrapper, compiled_fn = compiler.end_capture(buf)

        def bytecode_hook(code, out_code):
            import dis
            import sys

            if sys.version_info < (3, 11):
                call_op = "CALL_FUNCTION"
            else:
                call_op = "CALL"

            insts = list(dis.get_instructions(out_code))
            call_graph_idx = next(
                i for i, inst in enumerate(insts) if inst.opname == call_op
            )
            # pre-graph should alias: inputs_ref_0 = inputs[0]
            matches = [
                inst
                for inst in insts[:call_graph_idx]
                if inst.opname == "STORE_FAST" and inst.argval == "inputs_ref_0"
            ]
            self.assertTrue(len(matches) == 1)
            # post-graph should access inputs_ref_0 instead of inputs
            matches = [
                inst for inst in insts[call_graph_idx:] if inst.argval == "inputs"
            ]
            self.assertTrue(len(matches) == 0)
            matches = [
                inst
                for inst in insts[call_graph_idx:]
                if inst.opname == "LOAD_FAST" and inst.argval == "inputs_ref_0"
            ]
            self.assertTrue(len(matches) == 1)

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)
        try:
            runtime_wrapper(
                compiled_fn=compiled_fn,
                inputs=[param, active],
                sizes=(),
                scalars=(),
                hooks=[],
                packed_inputs=[],
            )
        finally:
            handle.remove()

    def test_inputs_aliasing_bytecode_stack_restore(self):
        logging.getLogger().setLevel(logging.WARNING)
        from torch.testing._internal.logging_tensor import LoggingTensor

        # Create a graph that allows inputs stealing
        def forward(inputs):
            add = inputs[0] + 1
            add_1 = add + inputs[1]  # handled in suffix for tensor subclass
            out = add_1.cpu()
            return (out,)

        gm = torch.fx.symbolic_trace(forward)
        torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
        compiled_fn = torch.compile(gm)

        inputs = [
            torch.ones(1000000, dtype=torch.float32),
            LoggingTensor(torch.ones(1)),
        ]
        match_done = False

        def bytecode_hook(code, out_code):
            import dis
            import sys

            nonlocal match_done

            # test is sensitive to what Dynamo traces. So as soon as the main
            # graph is tested, we skip the bytecode hook checks for future
            # frames.
            if not match_done:
                if sys.version_info < (3, 11):
                    call_op = "CALL_FUNCTION"
                else:
                    call_op = "CALL"

                insts = list(dis.get_instructions(out_code))
                call_graph_idx = next(
                    i for i, inst in enumerate(insts) if inst.opname == call_op
                )
                # pre-graph should alias: inputs_ref_0 = inputs[0]
                matches = [
                    inst
                    for inst in insts[:call_graph_idx]
                    if inst.opname == "STORE_FAST" and inst.argval == "inputs_ref_0"
                ]
                self.assertTrue(len(matches) == 1)
                # post-graph should access inputs_ref_0 instead of inputs
                matches = [
                    inst for inst in insts[call_graph_idx:] if inst.argval == "inputs"
                ]
                self.assertTrue(len(matches) == 0)
                matches = [
                    inst
                    for inst in insts[call_graph_idx:]
                    if inst.opname == "LOAD_FAST" and inst.argval == "inputs_ref_0"
                ]
                self.assertTrue(len(matches) == 1)
                match_done = True

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)
        try:
            compiled_fn(inputs)
            self.assertTrue(len(inputs) == 0)
        finally:
            handle.remove()

    def test_implicit_add(self):
        def fn():
            y = torch.randn(1, 4, requires_grad=True)

            def model(x):
                # y is used multiple times, gradients get added
                return torch.sigmoid(x * y + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.backward()
                yield result
                yield y.grad
                y.grad = None

        self.check_output_and_recompiles(fn)

    def test_output_nodes_all_leaves(self):
        def fn():
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                return torch.sigmoid(x * z + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                gy, gz = torch.autograd.grad(result, inputs=[y, z])
                assert y.grad is None
                assert z.grad is None
                yield gy
                yield gz

        self.check_output_and_recompiles(fn)

    def test_output_nodes_some_leaves(self):
        def fn():
            class UnreachableBwd(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    raise RuntimeError

            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                return torch.sigmoid(UnreachableBwd.apply(y) * z)

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                gz = torch.autograd.grad(result, inputs=[z])
                assert y.grad is None
                assert z.grad is None
                yield gz

        self.check_output_and_recompiles(fn)

    def test_no_output_nodes_all_leaves(self):
        def fn():
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                return torch.sigmoid(x * z + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])
                result = model(x).sum()
                out = result.backward()
                assert out is None
                assert y.grad is not None
                assert z.grad is not None
                yield y.grad
                yield z.grad
                y.grad = None
                z.grad = None

        self.check_output_and_recompiles(fn)

    def test_no_output_nodes_some_leaves(self):
        def fn():
            class UnreachableBwd(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    raise RuntimeError

            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)
            a = torch.randn(1, 4, requires_grad=True)

            def model(x):
                return torch.sigmoid(x * y * z * UnreachableBwd.apply(a))

            for _ in range(3):
                x = torch.randn([1, 4])
                result = model(x).sum()
                out = result.backward(inputs=[y, z])
                assert out is None
                assert y.grad is not None
                assert z.grad is not None
                assert a.grad is None
                yield y.grad
                yield z.grad
                y.grad = None
                z.grad = None

        self.check_output_and_recompiles(fn)

    def test_no_output_nodes_different_leaves_will_recompile(self):
        def fn():
            def fwd(x, y, z):
                out = x * y  # MulBackward0
                out2 = out * z  # MulBackward0
                return out2.sum()  # SumBackward0

            x = torch.randn(5, requires_grad=True)
            y = torch.randn(5, requires_grad=True)
            z = torch.randn(5, requires_grad=True)
            loss = fwd(x, y, z)
            torch.compile(lambda: torch.autograd.backward(loss, inputs=[x]))()
            yield x.grad
            x.grad = None

            loss = fwd(x, y, z)
            torch.compile(lambda: torch.autograd.backward(loss, inputs=[y]))()
            yield y.grad

        # Guarded by TensorArg id, mismatch on last MulBackward0
        self.check_output_and_recompiles(fn, 2)

    def test_dynamic_shapes(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for b in range(10, 100, 10):
                x = torch.randn([b, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()

        self.check_output_and_recompiles(fn)

    def test_dynamic_shapes_from_forward(self):
        class ToyModel(nn.Module):
            def __init__(self, in_feat=10, hidden_feat=50, out_feat=5):
                super().__init__()
                self.linear1 = nn.Linear(in_feat, hidden_feat)
                self.linear2 = nn.Linear(hidden_feat, hidden_feat)
                self.linear3 = nn.Linear(hidden_feat, out_feat)
                self.mse_loss = torch.nn.MSELoss()

            def forward(self, inputs, output):
                out1 = self.linear1(inputs)
                out2 = self.linear2(out1)
                out3 = self.linear3(out2)
                return self.mse_loss(out3, output)

        m = ToyModel()
        m = torch.compile(m)

        def run(i):
            torch._dynamo.utils.counters.clear()
            inp = torch.randn(i, 10)
            target = torch.randn(i, 5)
            loss = m(inp, target)
            with compiled_autograd._enable(make_compiler_fn(dynamic=None)):
                loss.backward()

        counters = torch._dynamo.utils.counters
        run(3)
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 1)
        run(4)
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 1)
        run(5)
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 0)
        run(6)
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 0)

    def test_dynamic_shapes_eager_node(self):
        # Here, we have no way of marking the symbolic sizes using in SumBackward as dynamic
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for b, s in zip([10, 20, 30], [2, 4, 8]):
                x = torch.randn([b, 4])
                result = opt_model(x)
                view = result.view(s, -1)
                # sum will save dynamic sizes
                loss = view.sum()
                loss.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()

        self.check_output_and_recompiles(fn)

    def test_dynamic_shapes_annotations(self):
        @torch.compile
        def f(x):
            return x.sin().sin()

        with torch._dynamo.compiled_autograd._enable(torch.compile):
            x = torch.randn(2, 3, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            out = f(x)
            out.sum().backward()

            x = torch.randn(4, 3, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            out = f(x)
            out.sum().backward()

        # mark_dynamic should not cause ConstraintViolationError
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_api_dynamic_shapes(self):
        # Here, we have no way of marking the symbolic sizes using in SumBackward as dynamic
        def fn(call_backward):
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )

            for b, s in zip([10, 20, 30], [2, 4, 8]):
                x = torch.randn([b, 4])
                result = model(x)
                view = result.view(s, -1)
                # sum will save dynamic sizes
                loss = view.sum()
                call_backward(loss)
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()

        def call_backward(loss):
            loss.backward()

        eager_out = list(fn(call_backward))
        with config.patch(compiled_autograd=True):
            compiled_out = list(fn(torch.compile(call_backward, dynamic=True)))
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_accumulate_without_zero(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for _ in range(10):
                x = torch.randn([10, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad.clone()
                yield model[0].bias.grad.clone()
                yield model[2].weight.grad.clone()
                yield model[2].bias.grad.clone()

        self.check_output_and_recompiles(fn, count=2)

    def test_inplace_grad_update(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compi
```



## High-Level Overview


This Python file contains 67 class(es) and 457 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BaseCustomOp`, `TestCompiledAutograd`, `Module`, `TestModel`, `TestModel`, `TestModel`, `TestModel`, `MyFn`, `MyFn`, `UnreachableBwd`, `UnreachableBwd`, `ToyModel`, `ModuleWithJit`, `Model`, `MySin`, `MyFn`, `MyFn`, `MyFn`, `MyFn`, `MyFn`

**Functions defined**: `make_compiler_fn`, `_compiler_fn`, `_inner_compiler`, `hook1`, `hook2`, `hook3`, `reset`, `forward`, `backward`, `setUp`, `tearDown`, `check_output_and_recompiles`, `run_as_subprocess`, `test_hipify_not_loaded_with_import_torch`, `test_hipify_not_loaded_with_import_cpp_extension`, `test_dynamo_flaky_segfault`, `main`, `compiler_fn`, `inner`, `model`

**Key imports**: contextlib, dataclasses, functools, io, itertools, logging, os, re, subprocess, sys


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `dataclasses`
- `functools`
- `io`
- `itertools`
- `logging`
- `os`
- `re`
- `subprocess`
- `sys`
- `tempfile`
- `unittest`
- `copy`: deepcopy
- `importlib.machinery`: SourceFileLoader
- `pathlib`: Path
- `string`: Template
- `torch`
- `torch.distributed as dist`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch._dynamo`: compiled_autograd, config
- `torch._dynamo.backends.debugging`: aot_eager
- `torch._dynamo.device_interface`: get_interface_for_device
- `torch._dynamo.testing`: normalize_gm
- `torch._dynamo.utils`: counters
- `torch._inductor`: config as inductor_config
- `torch._inductor.cpp_builder`: is_msvc_cl
- `torch._inductor.test_case`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_compiled_autograd.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_compiled_autograd.py_docs.md`
- **Keyword Index**: `test_compiled_autograd.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
