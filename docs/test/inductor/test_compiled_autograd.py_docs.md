# Documentation: test_compiled_autograd.py

## File Metadata
- **Path**: `test/inductor/test_compiled_autograd.py`
- **Size**: 206204 bytes
- **Lines**: 5427
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
            opt_model = torch.compile(model, dynamic=True)

            for _ in range(10):
                w_grad = torch.rand_like(model[0].weight)
                b_grad = torch.rand_like(model[0].bias)
                model[0].weight.grad = w_grad
                model[0].bias.grad = b_grad

                x = torch.randn([10, 4])
                result = opt_model(x).sum()
                result.backward()
                assert model[0].weight.grad is w_grad
                assert model[0].bias.grad is b_grad
                yield w_grad.clone()
                yield b_grad.clone()

        self.check_output_and_recompiles(fn, count=1)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_issue106555(self):
        DEVICE = torch.device(GPU_TYPE, 0)
        NUM_FEATURES = 256

        def bias_sigmoid_mul(x1, x2, bias):
            x2 = torch.sigmoid(x2 + bias)
            y = x1 * x2
            return y

        bias_sigmoid_mul_jit = torch.compile(bias_sigmoid_mul)

        class ModuleWithJit(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear_1 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=True)
                self.linear_2 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=False)
                self.linear_2_bias = nn.Parameter(torch.zeros(NUM_FEATURES))

            def forward(self, input_tensor):
                x1 = self.linear_1(input_tensor)
                x2 = self.linear_2(input_tensor)
                output = bias_sigmoid_mul_jit(x1, x2, self.linear_2_bias)
                return output

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.module_with_jit_1 = ModuleWithJit()
                self.module_with_jit_2 = ModuleWithJit()

            def forward(self, x, gradient_checkpointing: bool):
                if gradient_checkpointing:
                    y = torch.utils.checkpoint.checkpoint(
                        self._forward, x, use_reentrant=True
                    )
                else:
                    y = self._forward(x)
                return y

            def _forward(self, x):
                x = x + self.module_with_jit_1(x)
                x = x + self.module_with_jit_2(x.transpose(-2, -3)).transpose(-2, -3)
                return x

        device_interface = get_interface_for_device(GPU_TYPE)
        device_interface.set_device(device=DEVICE)
        torch.manual_seed(1234567890)
        model = Model()
        model.train()
        model.to(device=DEVICE)
        model_parameters = list(model.parameters())

        torch.manual_seed(1234567890)
        input_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(device=DEVICE)
        input_tensor.requires_grad = True
        target_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(
            dtype=input_tensor.dtype, device=DEVICE
        )

        for _ in range(10):
            for param in model_parameters:
                param.grad = None
            output_tensor = model(
                x=input_tensor.clone(),
                gradient_checkpointing=True,
            )
            loss = torch.mean(torch.abs(target_tensor - output_tensor))
            loss.backward()

    def test_keep_graph_simple(self):
        x = torch.tensor([2.0], requires_grad=True)
        y = x**2

        # First backward pass; keep the computation graph
        y.backward(retain_graph=True)
        self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4

        # Note - this will run under both the eager and compiled regime.
        def fn():
            # Reset the gradients
            x.grad = torch.tensor([0.0])
            # Second and Third backward pass; keep the computation graph
            y.backward(retain_graph=True)
            self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4
            return x.grad

        self.check_output_and_recompiles(fn, count=1)

    def test_keep_graph_usage_after_compiled(self):
        x = torch.tensor([2.0], requires_grad=True)
        y = x**2

        # First backward pass; keep the computation graph
        def eager_check():
            y.backward(retain_graph=True)
            self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4
            x.grad = torch.tensor([0.0])

        eager_check()

        for _ in range(5):
            with compiled_autograd._enable(compiler_fn):
                eager_check()

            eager_check()

    def test_custom_fn_saved_tensors(self):
        def fn():
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MySin.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_multiple_tensors(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y):
                    ctx.save_for_backward(x, y)
                    return torch.sin(x), torch.sin(y)

                @staticmethod
                def backward(ctx, gO_x, gO_y):
                    (x, y) = ctx.saved_tensors
                    return gO_x * torch.cos(x), gO_y * torch.cos(y)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                y = torch.arange(0.0, i, requires_grad=True)
                out1, out2 = MyFn.apply(x, y)
                loss = (out1 * out2).sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_multiple_tensors_dedup(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x, x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    (x1, x2) = ctx.saved_tensors
                    return gO * torch.cos(x1) * torch.cos(x2)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MyFn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_shape_tensor(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    (x,) = ctx.saved_tensors
                    return gO * x.shape[0]

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MyFn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_attr(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.shape = x.shape
                    return x

                @staticmethod
                def backward(ctx, gO):
                    x_shape = ctx.shape[0]
                    return gO * x_shape

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MyFn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, compiler_fn=make_compiler_fn(fullgraph=False)
        )

    def test_custom_fn_multiple_grads(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y):
                    return x + y, y

                @staticmethod
                def backward(ctx, gO_1, gO_2):
                    return gO_1, gO_2

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                y = torch.arange(0.0, i, requires_grad=True)
                out1, out2 = MyFn.apply(x, y)
                loss = (out1 + out2).sum()
                loss.backward()
                yield x.grad
                yield y.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_non_variable_input(self):
        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y, z):
                    return x * 2, y * 3, z * 4

                @staticmethod
                def backward(ctx, gO_1, gO_2, gO_3):
                    return gO_1, gO_2, gO_3

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                y = 1
                z = torch.arange(0.0, i, requires_grad=True)
                out1, out2, out3 = MyFn.apply(x, y, z)
                loss = (out1 + out2 + out3).sum()
                loss.backward()
                yield x
                yield y
                yield z

        self.check_output_and_recompiles(fn)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_logging_tensor_flaky(self) -> None:
        # when you first run some test using triton and then run test_inputs_aliasing_bytecode_stack_restore
        # resulting in:
        #   - pytest: `TypeError: unsupported operand type(s) for +: 'Tensor' and 'LoggingTensor'`
        #   - python: `TypeError: not all arguments converted during string formatting`

        # 1. some triton involving test
        def fn():
            def _fn(x):
                return x

            x = torch.arange(
                1, 10, requires_grad=True, dtype=torch.float16, device=GPU_TYPE
            )
            out = _fn(x)
            loss = out.sum()
            loss.backward()

        with compiled_autograd._enable(compiler_fn):
            fn()

        logging.getLogger().setLevel(
            logging.WARNING
        )  # triton setup overwrote it to INFO
        # 2. test_inputs_aliasing_bytecode_stack_restore
        from torch.testing._internal.logging_tensor import LoggingTensor

        def forward(inputs):
            add = inputs[0] + 1
            add_1 = add + inputs[1]
            out = add_1.cpu()
            return (out,)

        gm = torch.fx.symbolic_trace(forward)
        print(gm.print_readable())
        torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
        compiled_fn = torch.compile(gm)

        inputs = [
            torch.ones(1000000, dtype=torch.float32),
            LoggingTensor(torch.ones(1)),
        ]

        compiled_fn(inputs)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_custom_fn_output_metadata(self):
        def my_compiler_fn(gm):
            for node in gm.graph.nodes:
                if isinstance(node.target, torch._ops.OpOverload):
                    assert node.target._name != "aten::_to_copy", (
                        "there should be no implicit copies (e.g. dtype casting)"
                    )

            def inner_compiler(gm_, example_inputs_):
                counters["compiled_autograd"]["compiles"] += 1
                return inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        def fn():
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO

            x = torch.arange(
                1, 10, requires_grad=True, dtype=torch.float16, device=GPU_TYPE
            )
            x_view = x.view(3, 3)
            out = MyFn.apply(x_view)
            loss = out.sum()
            loss.backward()
            yield x.dtype
            yield x.device
            yield x.grad

        self.check_output_and_recompiles(fn, count=1)

    def test_custom_fn_with_same_graph(self):
        def fn():
            class MyFn1(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO

            # same as MyFn1, but different autograd function id
            # should not be using same graph as MyFn1
            class MyFn2(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO

            for myfn in [MyFn1, MyFn2, MyFn1, MyFn2]:
                x = torch.arange(0.0, 10, requires_grad=True)
                out = myfn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=2
        )  # should compile once for MyFn1 and once for MyFn2

    def test_custom_fn_dynamically_defined_class(self):
        def fn():
            def create_class(multiplier: int):
                class DynamicFn(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x):
                        return x * multiplier

                    @staticmethod
                    def backward(ctx, gO):
                        return gO * multiplier

                return DynamicFn

            for multiplier in [10, 20, 30]:
                x = torch.arange(0.0, 10, requires_grad=True)
                out = create_class(multiplier).apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn, count=3)

    def test_custom_fn_bw_graph_break(self):
        def fn():
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    print("graph break")
                    (x,) = ctx.saved_tensors
                    print("graph break")
                    return gO * torch.cos(x)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MySin.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=[1, 3], compiler_fn=make_compiler_fn(fullgraph=False)
        )

    def test_custom_fn_compiled_fw_graph_break(self):
        def fn():
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    print("graph break")
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            opt_model = torch.compile(MySin.apply)
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = opt_model(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=1, compiler_fn=make_compiler_fn(fullgraph=False)
        )
        self.assertEqual(counters["stats"]["unique_graphs"], 4)  # 3 fw, 1 bw

    def test_custom_fn_compiled_fw_bw_graph_break(self):
        def fn():
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    print("graph break")
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    print("graph break")
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            opt_model = torch.compile(MySin.apply)
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = opt_model(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=[1, 3], compiler_fn=make_compiler_fn(fullgraph=False)
        )
        self.assertEqual(counters["stats"]["unique_graphs"], 6)  # 3 fw, 3 bw

    def test_mismatch_fake_tensor_mode(self, dynamic_shape=False):
        """
        Repro the failure of training nanogpt with both compiled-autograd
        and _LazyGraphModule. Check https://github.com/pytorch/pytorch/pull/118981
        for more context.
        """
        B = 8
        x = torch.rand(B, 16)
        y = torch.rand(B, 16, requires_grad=True)

        if dynamic_shape:
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(y, 0)

        def f():
            y.grad = None
            out = x + y

            # make sure the backward call does not trigger any error when
            # compiling the backward graph
            out.sum().backward()
            return out, y.grad

        self.check_output_and_recompiles(f, compile_fn=True)

    def test_mismatch_fake_tensor_mode_dynamic_shape(self):
        self.test_mismatch_fake_tensor_mode(dynamic_shape=True)

    def test_accumulate_grad_accuracy(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 1, bias=False),
                torch.nn.Linear(1, 2, bias=False),
            )
            x = torch.randn(2, 2)

            out = model(x)
            loss = out.sum()
            torch.manual_seed(0)
            loss.backward()

            yield model[0].weight.grad
            yield model[1].weight.grad

        self.check_output_and_recompiles(fn, 1)

    def test_trace_run_with_rng_state(self):
        def sdpa(xq, xk):
            return F.scaled_dot_product_attention(xq, xk, xk, is_causal=True)

        def g(xq_1, xk_1, xq_2, xk_2):
            # xq: (bs, n_local_heads, seqlen, head_dim)
            # xk: (bs, n_local_heads, cache_len + seqlen, head_dim)
            y1 = sdpa(xq_1, xk_1)
            y2 = torch.utils.checkpoint.checkpoint(
                sdpa, xq_2, xk_2, use_reentrant=False
            )
            y = torch.mul(y1, y2)
            z = torch.matmul(y, y)
            return z

        def f():
            bs = 1
            n_local_heads = 1
            seqlen = 2
            head_dim = 2
            cache_len = 2
            xq_list = [
                torch.ones(
                    (bs, n_local_heads, seqlen, head_dim),
                    requires_grad=True,
                    device="cpu",
                )
                for _ in range(2)
            ]
            xk_list = [
                torch.ones(
                    (bs, n_local_heads, cache_len + seqlen, head_dim),
                    requires_grad=True,
                    device="cpu",
                )
                for _ in range(2)
            ]
            out = torch.compile(g, fullgraph=True)(
                xq_list[0], xk_list[0], xq_list[1], xk_list[1]
            )
            out.sum().backward()
            return out, *[x.grad for x in xq_list + xk_list]

        """
        Walkthrough of what happens with `run_with_rng_state`:
        1. `run_with_rng_state` only shows up in the backward graph (this op is inserted by the partitioner).
        2. The Dynamo graph captured by Compiled Autograd looks like:
        ```
        ===== __compiled_fn_3 =====
        torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
            def forward(self, L_inputs_ : list):
                ...
                run_with_rng_state = torch.ops.higher_order.run_with_rng_state(
                    getitem_8,
                    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
                    getitem_3, getitem_4, getitem_4, 0.0, True,
                )
                ...
        ```
        3. We want to preserve this `run_with_rng_state` op when going through AOTAutograd. We do it by having special handling
        in `run_with_rng_state` op's py_functionalize_impl.
        """

        def _run_with_rng_state_op_check(inductor_post_grad_graph):
            # Checks that `run_with_rng_state` op exists in Compiled Autograd's Inductor post-grad graph.
            op_set = {node.target for node in inductor_post_grad_graph.nodes}
            if torch.ops.higher_order.run_and_save_rng_state not in op_set:
                # This is backward graph, so check existence of `run_with_rng_state` op
                self.assertTrue(torch.ops.higher_order.run_with_rng_state in op_set)

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=_run_with_rng_state_op_check
        ):
            compiler_fn = make_compiler_fn(fullgraph=True)

            def make_compiler_fn_with_op_check():
                def _compiler_fn(gm):
                    # Checks that `run_with_rng_state` op exists in Compiled Autograd's Dynamo graph.
                    self.assertTrue(
                        any(
                            node.target is torch.ops.higher_order.run_with_rng_state
                            for node in gm.graph.nodes
                        )
                    )
                    return compiler_fn(gm)

                return _compiler_fn

            compiler_fn_with_op_check = make_compiler_fn_with_op_check()
            self.check_output_and_recompiles(
                f, compiler_fn=compiler_fn_with_op_check, compile_fn=False
            )

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_trace_auto_functionalized_v2(self):
        self.trace_auto_functionalized_base()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_trace_auto_functionalized(self):
        self.trace_auto_functionalized_base()

    def trace_auto_functionalized_base(self):
        with torch.library._scoped_library("testlib", "FRAGMENT") as lib:
            torch.library.define(
                "testlib::foo",
                "(Tensor(a!) x) -> (Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "testlib::foo_mutated",
                "(Tensor(a!) x) -> (Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("testlib::foo", "cpu", lib=lib)
            def foo(x):
                x.add_(5)
                return x

            @torch.library.impl("testlib::foo", "Meta", lib=lib)
            def foo_meta(x):
                return x

            @torch.library.impl(
                "testlib::foo_mutated", "CompositeImplicitAutograd", lib=lib
            )
            def foo_mutated(x):
                return torch.ops.testlib.foo(x)

            def _get_custom_policy(must_recompute_list=None):
                def _custom_policy(ctx, func, *args, **kwargs):
                    if must_recompute_list is not None and func in must_recompute_list:
                        return torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE
                    else:
                        return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

                return _custom_policy

            def context_fn():
                must_recompute_list = [
                    torch.ops.higher_order.auto_functionalized,
                ]
                return torch.utils.checkpoint.create_selective_checkpoint_contexts(
                    _get_custom_policy(
                        must_recompute_list=must_recompute_list,
                    ),
                )

            def g(x):
                x = torch.matmul(x, x)
                torch.ops.testlib.foo_mutated(x)
                return torch.matmul(x, x)

            def g_cp(x):
                return torch.utils.checkpoint.checkpoint(
                    g, x, use_reentrant=False, context_fn=context_fn
                )

            def f():
                inps = (torch.randn(4, 4, requires_grad=True),)
                output = torch.compile(g_cp, backend="aot_eager", fullgraph=True)(*inps)
                output.sum().backward()
                return output, inps[0].grad

            """
            Walkthrough of what happens with `auto_functionalized`:
            1. `auto_functionalized` op is inserted into the graph during AOTAutograd functionalization.
            We force the op to be recomputed (by using SAC), so it appears in the backward graph.
            2. The AOT backward graph looks like:
            ```
            ===== Backward graph 0 =====
            def forward(self, primals_1: "f32[4, 4][4, 1]cpu", tangents_1: "f32[4, 4][4, 1]cpu"):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = mm)
                ...
                return (add_1,)
            ```
            3. The Compiled Autograd graph looks like:
            ```
            ===== Compiled autograd graph =====
            def forward(self, inputs, sizes, scalars, hooks):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = aot0_mm)
                ...
                return []
            ```
            4. The Dynamo graph captured by Compiled Autograd looks like:
            ```
            ===== __compiled_fn_3 =====
            def forward(self, L_inputs_ : list):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = aot0_mm)
                ...
                return (new_grad,)
            ```
            5. The Compiled Autograd's AOT "forward-only" graph looks like:
            ```
            ===== Forward graph 1 =====
            def forward(self, arg0_1: "f32[][]cpu", arg1_1: "f32[4, 4][4, 1]cpu"):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = mm)
                ...
                return (clone_1,)
            ```
            6. The `auto_functionalized` op should then be lowered using the normal lowering path in Inductor.
            """

            compiler_fn = make_compiler_fn(fullgraph=True, backend="aot_eager")

            def make_compiler_fn_with_op_check():
                def _compiler_fn(gm):
                    auto_functionalize_func = (
                        torch.ops.higher_order.auto_functionalized
                        if not torch._inductor.config.enable_auto_functionalized_v2
                        else torch.ops.higher_order.auto_functionalized_v2
                    )

                    # Checks that `auto_functionalized` op exists in Compiled Autograd's Dynamo graph.
                    self.assertTrue(
                        any(
                            node.target is auto_functionalize_func
                            for node in gm.graph.nodes
                        ),
                        f"{auto_functionalize_func} op not found in {gm.graph}",
                    )
                    return compiler_fn(gm)

                return _compiler_fn

            compiler_fn_with_op_check = make_compiler_fn_with_op_check()
            self.check_output_and_recompiles(
                f, compiler_fn=compiler_fn_with_op_check, compile_fn=False
            )

    @scoped_load_inline
    def test_autograd_cpp_node_non_traceable(self, load_inline):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = false;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_non_traceable_autograd_cpp_node, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = load_inline(
            name="test_non_traceable_autograd_cpp_node",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            x = torch.ones(10, 10, requires_grad=True)
            out = module.custom_op_backed_by_autograd_fn(x)
            loss = out.sum()
            loss.backward()
            yield x.grad

        # should not raise
        self.check_output_and_recompiles(
            fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
        )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_basic(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_basic_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_basic",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            for i in [10, 100, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn, 1)
        else:
            # compiles for 10 (static) and 100 (dynamic), each with a graph break
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_id(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

struct CustomOpAutogradFunction2 : public torch::autograd::Function<CustomOpAutogradFunction2> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

torch::Tensor custom_op_backed_by_autograd_fn2(torch::Tensor x) {
  return CustomOpAutogradFunction2::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_id_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    m.def("custom_op_backed_by_autograd_fn2", custom_op_backed_by_autograd_fn2);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_id",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions=[
                "custom_op_backed_by_autograd_fn",
                "custom_op_backed_by_autograd_fn2",
            ],
            verbose=True,
        )

        def same_autograd_fn():
            def fn():
                x = torch.ones(10, 10, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

            yield from fn()  # compile
            yield from fn()  # reuse
            yield from fn()  # reuse
            yield from fn()  # reuse

        if is_traceable:
            self.check_output_and_recompiles(same_autograd_fn, 1)
        else:
            self.check_output_and_recompiles(
                same_autograd_fn,
                count=[1, 2],
                compiler_fn=make_compiler_fn(fullgraph=False),
            )

        def different_autograd_fn():
            def fn(op):
                x = torch.ones(10, 10, requires_grad=True)
                out = op(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

            op1 = module.custom_op_backed_by_autograd_fn
            op2 = module.custom_op_backed_by_autograd_fn2
            yield from fn(op1)  # compile
            yield from fn(op2)  # compile
            yield from fn(op1)  # reuse
            yield from fn(op2)  # reuse

        if is_traceable:
            self.check_output_and_recompiles(different_autograd_fn, 2)
        else:
            # ????
            self.check_output_and_recompiles(
                same_autograd_fn,
                count=[1, 2],
                compiler_fn=make_compiler_fn(fullgraph=False),
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_basic(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      const torch::Tensor& y,
      const torch::Tensor& fixed) {
    ctx->save_for_backward({x, y});
    ctx->saved_data["fixed_tensor"] = fixed;
    ctx->saved_data["bool"] = true;
    ctx->saved_data["int"] = 1;
    c10::List<std::string> list({"string"});
    ctx->saved_data["list"] = std::move(list);
    c10::Dict<std::string, double> dict;
    dict.insert("string", 1.0);
    ctx->saved_data["dict"] = std::move(dict);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor y = saved_variables[1];
    torch::Tensor fixed = ctx->saved_data["fixed_tensor"].toTensor();
    assert(ctx->saved_data["bool"].isBool());
    c10::SymInt i = ctx->saved_data["int"].toSymInt();
    c10::List<c10::IValue> list = ctx->saved_data["list"].toList();
    assert(list.size() == 1);
    assert(list.get(0).toStringRef() == "string");
    c10::Dict<c10::IValue, c10::IValue> dict = ctx->saved_data["dict"].toGenericDict();
    assert(dict.size() == 1);
    assert(dict.at("string") == 1.0);

    torch::autograd::variable_list grad_inputs(3);
    grad_inputs[0] = x + y + torch::sum(fixed) + i;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& fixed) {
  return CustomOpAutogradFunction::apply(x, y, fixed);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_basic_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_saved_basic",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            fixed = torch.ones(2, 2)
            for i in [10, 100, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                y = torch.randn(i, i)
                out = module.custom_op_backed_by_autograd_fn(x, y, fixed)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn, 1)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_dynamic(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    ctx->save_for_backward({x});
    ctx->saved_data["dynamic"] = x.view(-1);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    torch::Tensor z = ctx->saved_data["dynamic"].toTensor();

    torch::autograd::variable_list grad_inputs(1);
    grad_inputs[0] = x + torch::sum(z);
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_dynamic_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_saved_dynamic",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            for i in [10, 30, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        # compiles for 10 (static) and 100 (dynamic)
        if is_traceable:
            self.check_output_and_recompiles(fn, 1)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_int(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      int64_t y) {
    ctx->save_for_backward({x});
    ctx->saved_data["int"] = y;
    ctx->saved_data["symint"] = c10::SymInt(y);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    c10::SymInt y = ctx->saved_data["int"].toSymInt();
    c10::SymInt ys = ctx->saved_data["symint"].toSymInt();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + y + ys;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, int64_t y) {
  return CustomOpAutogradFunction::apply(x, y);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_int_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_saved_int",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            for y in [1, 2, 3, 1]:
                x = torch.ones(10, 10, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x, y)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_float(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      double z) {
    ctx->save_for_backward({x});
    ctx->saved_data["float"] = z;
    ctx->saved_data["symfloat"] = c10::SymFloat(z);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    c10::SymFloat z = ctx->saved_data["float"].toSymFloat();
    c10::SymFloat zs = ctx->saved_data["symfloat"].toSymFloat();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + z + zs;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, double z) {
  return CustomOpAutogradFunction::apply(x, z);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_float_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_saved_float",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            for z in [1.1, 2.2, 3.3, 1.1]:
                x = torch.ones(10, 10, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x, z)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            # compiled autograd and dynamo both support symfloat, but not backend
            self.check_output_and_recompiles(fn, [1, 4])
            # 1 restart analysis due to specialize_float=False
            self.assertEqual(counters["stats"]["unique_graphs"], 3)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 3], compiler_fn=make_compiler_fn(fullgraph=False)
            )
            self.assertEqual(counters["stats"]["unique_graphs"], 2)

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_data_dependent(self, load_inline, is_traceable):
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;
  static int iteration;

  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      const torch::Tensor& y) {
    ctx->save_for_backward({x, y});
    ctx->saved_data["bool"] = true;
    ctx->saved_data["int"] = 1;

    switch (iteration) {
        case 0: {
            break;
        }
        case 1: {
            // recompile
            ctx->saved_data["forces_recompile"] = iteration;
            break;
        }
        case 2: {
            // recompile
            ctx->set_materialize_grads(false);
            break;
        }
        case 3: {
            // reuse
            break;
        }
        default: {
            throw std::runtime_error("unexpected iteration");
        }
    }
    iteration++;
    return {x, y};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor y = saved_variables[1];
    c10::SymInt i = ctx->saved_data["int"].toSymInt();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + y + i;
    return grad_inputs;
  }
};

int CustomOpAutogradFunction::iteration = 0;

torch::autograd::variable_list custom_op_backed_by_autograd_fn(const torch::Tensor& x, const torch::Tensor& y) {
  return CustomOpAutogradFunction::apply(x, y);
}

void reset() {
    CustomOpAutogradFunction::iteration = 0;
}

TORCH_LIBRARY(test_autograd_cpp_node_data_dependent_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    m.def("reset", reset);
}
        """
        )

        module = load_inline(
            name="test_autograd_cpp_node_data_dependent",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions=["custom_op_backed_by_autograd_fn", "reset"],
            verbose=True,
        )

        def fn():
            module.reset()
            for i in [10, 10, 10, 10]:
                x = torch.ones(i, i, requires_grad=True)
                y = torch.randn(i, i)
                (
                    out1,
                    out2,
                ) = module.custom_op_backed_by_autograd_fn(x, y)
                loss = (out1 + out2).sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn, 3)
        else:
            self.check_output_and_recompiles(
                fn, count=[3, 6], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_free_activation_memory(self):
        script = """
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.testing._internal.inductor_utils import GPU_TYPE

def main():
    device_interface = get_interface_for_device(GPU_TYPE)
    assert(device_interface.memory_allocated() == 0)

    # Use an op to check that the memory is freed by the time the op is executed
    def assertion_impl(to_clone):
        mem_allocated = device_interface.memory_allocated()
        assert mem_allocated < 4000000  # some activations should be freed
        return to_clone.clone()

    with torch.library._scoped_library("test_compiled_autograd", "FRAGMENT") as lib:
        lib.define(
            "assertion_op(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,)
        )
        lib.impl("assertion_op", assertion_impl, "CPU")
        lib.impl("assertion_op", lambda x: x.clone(), "Meta")

        # Create a graph that allows inputs stealing
        def forward(activations):
            add = activations[0] + 1
            out = add.cpu()
            cloned_out = torch.ops.test_compiled_autograd.assertion_op(out)
            return (cloned_out,)

        gm = torch.fx.symbolic_trace(forward)
        torch._dynamo.utils.set_locals_to_steal(gm, ["activations"])
        compiled_fn = torch.compile(gm)

        # allocate at least 4,000,000 bytes (1,000,000 * 4 bytes)
        activations = [torch.ones(1000000, dtype=torch.floa

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 63 class(es): BaseCustomOp, TestCompiledAutograd, Module, TestModel, TestModel, TestModel, TestModel, MyFn, MyFn, UnreachableBwd, UnreachableBwd, ToyModel, ModuleWithJit, Model, MySin, MyFn, MyFn, MyFn, MyFn, MyFn

### Functions
This file defines 457 function(s): make_compiler_fn, _compiler_fn, _inner_compiler, hook1, hook2, hook3, reset, forward, backward, setUp, tearDown, check_output_and_recompiles, run_as_subprocess, test_hipify_not_loaded_with_import_torch, test_hipify_not_loaded_with_import_cpp_extension, test_dynamo_flaky_segfault, main, compiler_fn, inner, model, gen_cache_miss_log_prefix, test_reset, test_basic, fn, test_cache_hit, fn, test_graph_break_custom_op, sin, setup_context, backward


## Key Components

The file contains 14616 words across 5427 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 206204 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
