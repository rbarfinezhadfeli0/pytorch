# Documentation: `test/inductor/test_torchinductor.py`

## File Metadata

- **Path**: `test/inductor/test_torchinductor.py`
- **Size**: 545,214 bytes (532.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import contextlib
import copy
import dataclasses
import functools
import gc
import importlib
import itertools
import math
import operator
import os
import random
import re
import subprocess
import sys
import threading
import time
import unittest
import unittest.mock
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from typing_extensions import ParamSpec
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.aoti_eager
import torch.fx.traceback as fx_traceback
import torch.nn as nn
from torch._C._dynamo.guards import assert_alignment, assert_size_stride
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.debug_utils import aot_graph_input_parser
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import (
    CompileCounterWithBackend,
    expectedFailureCodegenDynamic,
    rand_strided,
    reset_rng_state,
    same,
    skipIfPy312,
)
from torch._dynamo.utils import ifdynstaticdefault
from torch._guards import CompileContext, CompileId
from torch._inductor import lowering
from torch._inductor.aoti_eager import (
    aoti_compile_with_persistent_cache,
    aoti_eager_cache_dir,
    load_aoti_eager_cache,
)
from torch._inductor.codegen.common import DataTypePropagation, OptimizationContext
from torch._inductor.fx_passes import pad_mm
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    add_scheduler_init_hook,
    run_and_get_code,
    run_and_get_cpp_code,
    run_and_get_kernels,
    run_and_get_triton_code,
    run_fw_bw_and_get_code,
    triton_version_uses_attrs_dict,
)
from torch._inductor.virtualized import V
from torch._prims_common import is_integer_dtype
from torch.fx.experimental.proxy_tensor import make_fx
from torch.library import _scoped_library
from torch.nn import functional as F
from torch.testing import FileCheck, make_tensor
from torch.testing._internal.common_cuda import (
    IS_SM90,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    SM80OrLater,
    SM90OrLater,
    TEST_CUDNN,
    tf32_on_and_off,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    expectedFailureXPU,
    largeTensorTest,
)
from torch.testing._internal.common_dtype import all_types, get_all_dtypes
from torch.testing._internal.common_quantization import (
    _dynamically_quantize_per_channel,
    _group_quantize_tensor_symmetric,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    IS_X86,
    MACOS_VERSION,
    parametrize,
    serialTest,
    skipIfMPS,
    skipIfRocm,
    skipIfWindows,
    skipIfXpu,
    subtest,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    xfailIfS390X,
)
from torch.testing._internal.logging_utils import logs_to_string
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.weak import WeakTensorKeyDictionary


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"

importlib.import_module("functorch")
importlib.import_module("filelock")

from torch._inductor import config, cpu_vec_isa, test_operators
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
from torch._inductor.utils import has_torchvision_roi_align
from torch.testing._internal.common_utils import slowTest
from torch.testing._internal.inductor_utils import (
    clone_preserve_strides_offset,
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    HAS_MPS,
    HAS_MULTIGPU,
    IS_BIG_GPU,
    requires_gpu,
    RUN_CPU,
    RUN_GPU,
    skipCPUIf,
    skipCUDAIf,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


_T = TypeVar("_T")
_P = ParamSpec("_P")


HAS_AVX2 = "fbgemm" in torch.backends.quantized.supported_engines

if TEST_WITH_ROCM:
    torch._inductor.config.force_layout_optimization = 1
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"

aten = torch.ops.aten

requires_multigpu = functools.partial(
    unittest.skipIf, not HAS_MULTIGPU, f"requires multiple {GPU_TYPE} devices"
)
skip_if_x86_mac = functools.partial(
    unittest.skipIf, IS_MACOS and IS_X86, "Does not work on x86 Mac"
)
vec_dtypes = [torch.float, torch.bfloat16, torch.float16]

libtest = torch.library.Library("test", "FRAGMENT")  # noqa: TOR901
ids = set()

f32 = torch.float32
i64 = torch.int64
i32 = torch.int32

test_dtypes = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]

test_int_dtypes = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]

if SM80OrLater or MACOS_VERSION >= 14.0:
    test_dtypes.append(torch.bfloat16)


def _large_cumprod_input(shape, dim, dtype, device):
    # Construct a cumprod input which guarantees not to overflow or underflow
    if is_integer_dtype(dtype):
        # Large products don't fit in integers, the best we can do
        # is random +/-1 values to test the sign of the result
        x = torch.randint(0, 1, shape, dtype=dtype, device=device)
        return x * 2 - 1

    comp_dtype = torch._prims_common.get_computation_dtype(dtype)
    batch_size = 256
    if comp_dtype != dtype:
        batch_size = math.floor(math.log2(torch.finfo(dtype).max) / 3)

    # Create random values with a uniform magnitude and uniform exponent
    num_batches = (shape[dim] + 2 * batch_size - 1) // (2 * batch_size)
    batch_shape = (
        shape[:dim]
        + (
            num_batches,
            batch_size,
        )
        + shape[dim + 1 :]
    )
    magnitude = 1 + torch.rand(batch_shape, dtype=comp_dtype, device=device)
    exponent = torch.randint(-1, 1, batch_shape, device=device).to(comp_dtype)
    batch = magnitude * exponent.exp2()

    # Alternate each batch of values with their reciprocals so the product
    # never gets too far away from 1
    t = torch.cat((batch, batch.reciprocal()), dim=dim + 1)
    t = t.flatten(dim, dim + 1)
    t = aten.slice(t, dim=dim, start=0, end=shape[dim])

    # Randomize sign
    sign = torch.randint(0, 1, shape, device=device) * 2 - 1
    return (t * sign).to(dtype)


def define_custom_op_for_test(id_, fn, fn_meta, tags=()):
    global libtest
    global ids
    if id_ not in ids:
        libtest.define(f"{id_}(Tensor self) -> Tensor", tags=tags)
        libtest.impl(id_, fn, "CPU")
        libtest.impl(id_, fn, "CUDA")
        libtest.impl(id_, fn, "XPU")
        libtest.impl(id_, fn, "MPS")
        libtest.impl(id_, fn_meta, "Meta")
        ids.add(id_)


def define_custom_op_2_for_test(id_, fn, fn_meta, tags=()):
    global libtest
    global ids
    if id_ not in ids:
        libtest.define(
            f"{id_}(Tensor self, float scale) -> (Tensor, Tensor)", tags=tags
        )
        libtest.impl(id_, fn, "CPU")
        libtest.impl(id_, fn, "CUDA")
        libtest.impl(id_, fn, "XPU")
        libtest.impl(id_, fn, "MPS")
        libtest.impl(id_, fn_meta, "Meta")
        ids.add(id_)


def define_custom_op_3_for_test(id_, fn, fn_meta, tags=()):
    global libtest
    global ids
    if id_ not in ids:
        libtest.define(f"{id_}(Tensor[] x) -> Tensor", tags=tags)
        libtest.impl(id_, fn, "CPU")
        libtest.impl(id_, fn, "CUDA")
        libtest.impl(id_, fn, "XPU")
        libtest.impl(id_, fn, "MPS")
        libtest.impl(id_, fn_meta, "Meta")
        ids.add(id_)


f32 = torch.float32


def register_ops_with_aoti_compile(ns, op_set, dispatch_key, torch_compile_op_lib_impl):
    for _op_name in op_set:
        qualified_op_name = f"{ns}::{_op_name}"
        _, overload_names = torch._C._jit_get_operation(qualified_op_name)
        for overload_name in overload_names:
            try:
                reg_op_name = qualified_op_name
                schema = torch._C._get_schema(qualified_op_name, overload_name)
                if schema.overload_name:
                    reg_op_name = f"{qualified_op_name}.{schema.overload_name}"
                torch_compile_op_lib_impl._impl_with_aoti_compile(  # noqa: F821
                    reg_op_name, dispatch_key
                )
            except Exception as e:
                continue


def get_divisible_by_16(cfg):
    # attribute was renamed between triton versions, from "divisible_by_16" to "divisibility_16"
    if hasattr(cfg, "divisibility_16"):
        return cfg.divisibility_16
    elif hasattr(cfg, "divisible_by_16"):
        return cfg.divisible_by_16
    # `cfg` example:
    # {(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}
    return [
        key[0]
        for key, value in cfg.items()
        if len(key) == 1 and value[0] == ["tt.divisibility", 16]
    ]


def get_post_grad_graph(f, inputs):
    log_stream, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
    with ctx():
        f(*inputs)
    post_grad_graph = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()
    return post_grad_graph


class TestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "debug_index_asserts": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                    "generate_intermediate_hooks": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        super().setUp()
        self._start = time.perf_counter()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        if os.environ.get("ERROR_ON_SLOW") == "1":
            elapsed = time.perf_counter() - self._start
            assert elapsed < 120


class ToTuple(torch.nn.Module):
    def forward(self, x):
        return (x,)


@dataclasses.dataclass
class InputGen:
    n: int
    device: str

    def dense(self):
        return torch.randn((self.n, self.n), device=self.device)

    def transposed(self):
        return self.dense().transpose(0, 1)

    def strided(self):
        return torch.randn((self.n * 2, self.n * 3), device=self.device)[
            self.n :, self.n :: 2
        ]

    def broadcast1(self):
        return torch.randn((self.n,), device=self.device)

    def broadcast2(self):
        return torch.randn((1, self.n, 1), device=self.device)

    def broadcast3(self):
        return torch.randn((1,), device=self.device)

    def double(self):
        if self.device == "mps":
            raise unittest.SkipTest("MPS does not support torch.float64")
        return torch.randn((self.n, self.n), device=self.device, dtype=torch.double)

    def int(self):
        return torch.arange(self.n, device=self.device, dtype=torch.int32)


def compute_grads(args, kwrags, results, grads):
    def gather_leaf_tensors(args, kwargs):
        args = pytree.arg_tree_leaves(*args, **kwargs)
        leaf_tensors = [
            arg for arg in args if isinstance(arg, torch.Tensor) and arg.requires_grad
        ]
        return leaf_tensors

    flat_results = pytree.tree_leaves(results)
    flat_diff_results = [
        r for r in flat_results if isinstance(r, torch.Tensor) and r.requires_grad
    ]
    assert len(flat_diff_results) > 0

    leaf_tensors = gather_leaf_tensors(args, kwrags)
    assert len(leaf_tensors) > 0
    return torch.autograd.grad(
        flat_diff_results,
        leaf_tensors,
        grads,
        allow_unused=True,
        retain_graph=True,
    )


def check_model(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    grad_atol=None,
    grad_rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_gpu=True,
    reference_in_float=True,
    assert_equal=True,
    check_gradient=False,
    check_has_compiled=True,
    output_process_fn_grad=lambda x: x,
    # TODO: enable this for all tests
    exact_stride=False,
):
    kwargs = kwargs or {}
    torch._dynamo.reset()

    ref_inputs = [clone_preserve_strides_offset(x) for x in example_inputs]
    ref_kwargs = kwargs
    has_lowp_args = False

    if reference_in_float and exact_dtype:
        # Store expected dtypes so we can check actual result gives the correct types
        torch.manual_seed(0)
        try:
            eager_result = model(*ref_inputs, **ref_kwargs)
        except RuntimeError:
            # Eager model may fail if the dtype is not supported
            eager_result = None

        ref_inputs = [clone_preserve_strides_offset(x) for x in example_inputs]
        expect_dtypes = [
            x.dtype if isinstance(x, torch.Tensor) else None
            for x in pytree.tree_leaves(eager_result)
        ]
        del eager_result

    ref_model = model
    if reference_in_float:
        # check_lowp is ignored here, it's kept just to be able to call `common` with extra arg
        def upcast_fn(x):
            nonlocal has_lowp_args
            if isinstance(x, torch.Tensor) and (
                x.dtype == torch.float16 or x.dtype == torch.bfloat16
            ):
                has_lowp_args = True
                # Preserve strides when casting
                result = torch.empty_strided(
                    x.size(), x.stride(), device=x.device, dtype=torch.float
                )
                result.copy_(x)
                return result
            else:
                return x

        # We previously call upcast_fn on example_inputs. It's incorrect
        # if example_inputs is already fp32 and get inplace updated in the model.
        # Call on the cloned tensors instead
        ref_inputs = list(map(upcast_fn, ref_inputs))
        ref_kwargs = {k: upcast_fn(v) for k, v in kwargs.items()}
        if has_lowp_args and hasattr(model, "to"):
            ref_model = copy.deepcopy(model).to(torch.float)

    torch.manual_seed(0)

    correct = ref_model(*ref_inputs, **ref_kwargs)

    torch._inductor.metrics.reset()

    called = False

    def compile_fx_wrapper(model_, example_inputs_):
        nonlocal called
        called = True
        return compile_fx(model_, example_inputs_)

    def run(*ex, **kwargs):
        return model(*ex, **kwargs)

    run = torch.compile(run, backend=compile_fx_wrapper, fullgraph=nopython)

    torch.manual_seed(0)
    actual = run(*example_inputs, **kwargs)
    # if not called:
    #     exp = torch._dynamo.explain(run)(*example_inputs)
    #     print("Explain:", exp[0])
    #     for graph in exp[2]:
    #         print("Graph", graph)
    if check_has_compiled:
        assert called, "Ran graph without calling compile_fx"
    assert type(actual) is type(correct)
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(correct)
        assert all(
            type(actual_item) is type(correct_item)
            for actual_item, correct_item in zip(actual, correct)
        )

    correct_flat, correct_spec = tree_flatten(correct)
    actual_flat = pytree.tree_leaves(actual)

    def reference_to_expect(actual_flat, correct_flat):
        return tuple(
            (
                y.to(x.dtype)
                if isinstance(y, torch.Tensor) and y.dtype.is_floating_point
                else y
            )
            for x, y in zip(actual_flat, correct_flat)
        )

    if reference_in_float and exact_dtype:
        for expect_dtype, actual_result in zip(expect_dtypes, actual_flat):
            if expect_dtype is not None:
                assert actual_result.dtype == expect_dtype, (
                    f"dtype mismatch, expected {expect_dtype} but got {actual_result.dtype}"
                )

    if reference_in_float:
        correct_flat = reference_to_expect(actual_flat, correct_flat)
        correct = tree_unflatten(correct_flat, correct_spec)

    # Allow assert_equal to be a custom function, instead of True or False, for
    # cases where differences may not indicate incorrectness.
    if assert_equal:
        if callable(assert_equal):

            def custom_assert_with_self(*args, **kwargs):
                assert_equal(self, *args, **kwargs)

            assert_equal_fn = custom_assert_with_self
        else:
            assert_equal_fn = self.assertEqual

        assert_equal_fn(
            actual,
            correct,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
            exact_dtype=exact_dtype,
            exact_stride=exact_stride,
        )
        # In case of input mutations, check that inputs are the same
        # (This never uses a custom assert_equal fn.)
        self.assertEqual(
            ref_inputs,
            example_inputs,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
            # our testing sometimes uses higher precision inputs for the reference
            exact_dtype=False,
            exact_stride=exact_stride,
        )
    else:
        for correct_val, actual_val in zip(correct_flat, actual_flat):
            if isinstance(correct_val, torch.Tensor):
                assert correct_val.device == actual_val.device
                assert correct_val.size() == actual_val.size()
                strides_equal, _ = torch._prims_common.check_significant_strides(
                    correct_val, actual_val
                )
                assert strides_equal
                assert correct_val.layout == actual_val.layout
                if exact_dtype:
                    assert correct_val.dtype == actual_val.dtype
                if exact_stride:
                    assert correct_val.stride() == actual_val.stride()

    if check_gradient:
        actual = output_process_fn_grad(actual)
        correct = output_process_fn_grad(correct)
        actual_flat = pytree.tree_leaves(actual)
        correct_flat = pytree.tree_leaves(correct)

        # generate random unit norm gradients
        grads = [
            torch.randn_like(r)
            for r in correct_flat
            if isinstance(r, torch.Tensor) and r.requires_grad
        ]
        for g in grads:
            g /= g.norm()

        correct_grad = compute_grads(ref_inputs, ref_kwargs, correct, grads)
        all_none_grads = all(x is None for x in correct_grad)
        tensor_args = [
            x
            for x in pytree.tree_flatten(example_inputs)[0]
            if isinstance(x, torch.Tensor)
        ]
        any_non_leaves = any(x.grad_fn is not None for x in tensor_args)
        if all_none_grads and any_non_leaves:
            # See Note [Detaching inputs that never need gradients]
            # There are a handful of ops that can return None gradients, into of zero gradients.
            # If all inputs to an AOTAutograd graph are supposed to get None gradients,
            # AOTAutograd will end up forcing all of the outputs of the forward to not require grad.
            # There's no easy fix to this (see the note above), although one option is to
            # force any derivative formulas in core to return tensors of zeros instead of None.
            flat_results = pytree.tree_leaves(actual)
            results_that_require_grad = [
                x
                for x in flat_results
                if isinstance(x, torch.Tensor) and x.requires_grad
            ]
            self.assertEqual(len(results_that_require_grad), 0)
        else:
            actual_grad = compute_grads(example_inputs, kwargs, actual, grads)

            if reference_in_float:
                expect_grad = reference_to_expect(actual_grad, correct_grad)
            else:
                expect_grad = correct_grad

            self.assertEqual(
                actual_grad,
                expect_grad,
                atol=grad_atol or atol,
                rtol=grad_rtol or rtol,
                equal_nan=True,
                exact_dtype=exact_dtype,
                exact_stride=exact_stride,
            )

    torch._dynamo.reset()


@torch._inductor.config.patch("triton.cudagraphs", False)
def check_model_gpu(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    grad_atol=None,
    grad_rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_gpu=True,
    reference_in_float=True,
    assert_equal=True,
    check_gradient=False,
    check_has_compiled=True,
    output_process_fn_grad=lambda x: x,
    # TODO: enable this for all tests
    exact_stride=False,
):
    kwargs = kwargs or {}
    if hasattr(model, "to"):
        model = model.to(device=GPU_TYPE)

    if copy_to_gpu:
        example_inputs = tuple(
            clone_preserve_strides_offset(x, device=GPU_TYPE) for x in example_inputs
        )

    check_model(
        self,
        model,
        example_inputs,
        kwargs,
        atol=atol,
        rtol=rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
        exact_dtype=exact_dtype,
        nopython=nopython,
        reference_in_float=reference_in_float,
        assert_equal=assert_equal,
        check_gradient=check_gradient,
        check_has_compiled=check_has_compiled,
        output_process_fn_grad=output_process_fn_grad,
        exact_stride=exact_stride,
    )

    if check_lowp:

        def downcast_fn(x):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.float:
                return x
            return torch.empty_strided(
                x.size(), x.stride(), device=GPU_TYPE, dtype=torch.half
            ).copy_(x)

        example_inputs = list(map(downcast_fn, example_inputs))
        if hasattr(model, "to"):
            model = model.to(torch.half)
        if rtol is not None:
            rtol = max(2e-3, rtol)
        check_model(
            self,
            model,
            example_inputs,
            kwargs,
            atol=atol,
            rtol=rtol,
            grad_atol=grad_atol,
            grad_rtol=grad_rtol,
            exact_dtype=exact_dtype,
            nopython=nopython,
            reference_in_float=reference_in_float,
            assert_equal=assert_equal,
            check_gradient=check_gradient,
            check_has_compiled=check_has_compiled,
            output_process_fn_grad=output_process_fn_grad,
            exact_stride=exact_stride,
        )


check_model_cuda = check_model_gpu


def _run_and_assert_no_indirect_indexing(
    test_case, func, *args, has_wrapping=None, has_assert=False, **kwargs
):
    result, source_codes = run_and_get_code(func, *args, **kwargs)

    for code in source_codes:
        for line in code.split("\n"):
            stmt = None
            # Find indexing expressions
            if ".load(" in line:
                stmt = line.split(".load")[-1]
            elif "tl.store" in line:
                stmt = line.split(".store")[-1]
                stmt = ",".join(stmt.split(",")[:-2])  # Remove store value and mask
            elif ".store" in line:
                stmt = line.split(".store")[-1]
            elif "[" in line:
                stmt = line.split("[")[-1].split("]")[0]
            if "tl.make_block_ptr(" in line:
                continue

            if stmt is None:
                continue

            # indirect indexing involves a `tmp` variable
            test_case.assertTrue(
                "tmp" not in stmt,
                msg=f"Found indirect indexing in statement '{stmt}' from code:\n{code}",
            )
        if has_wrapping is not None:
            test_case.assertTrue(
                ("where" in code or ") ? (" in code) is has_wrapping,
                msg=f"Wanted {has_wrapping=} but got\n{code}",
            )
    test_case.assertTrue(
        any(
            ("device_assert" in code or "TORCH_CHECK" in code) is has_assert
            for code in source_codes
        )
    )
    return result


def assertGeneratedKernelCountEqual(self: TestCase, expected: int):
    if config.triton.multi_kernel:
        # when multi_kernel is enabled, we generated both persistent reduction
        # and non-persistent reduction kernels for the same node schedule.
        # That will mess up with the kernel count. Just don't check it.
        return
    self.assertEqual(torch._inductor.metrics.generated_kernel_count, expected)


class SweepInputs2:
    input_gen_types1 = [
        "dense",
        "transposed",
        "strided",
        "broadcast1",
        "broadcast2",
        "broadcast3",
        "double",
        "int",
    ]
    input_gen_types2 = input_gen_types1
    gen = None

    @staticmethod
    def kernel(a, b):
        return (a + b,)

    @classmethod
    def gen_template(cls, name1, name2):
        def test(self):
            check_model(
                self,
                cls.kernel,
                (
                    getattr(cls.gen, name1)(),
                    getattr(cls.gen, name2)(),
                ),
            )

        test.__name__ = f"test_{cls.gen.device}_{name1}_{name2}"
        setattr(cls, test.__name__, test)

    @classmethod
    def populate(cls):
        for name1 in cls.input_gen_types1:
            for name2 in cls.input_gen_types2:
                cls.gen_template(name1, name2)


def is_cpp_backend(device):
    return getattr(device, "type", device) == "cpu" and config.cpu_backend == "cpp"


def skip_if_cpu(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self.device == "cpu":
            raise unittest.SkipTest("cpu not supported")
        return fn(self, *args, **kwargs)

    return wrapper


def skip_if_halide(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if is_halide_backend(self.device):
            raise unittest.SkipTest("halide not supported")
        return fn(self, *args, **kwargs)

    return wrapper


def xfail_if_mps(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not is_mps_backend(self.device):
            return fn(self, *args, **kwargs)
        with self.assertRaises(Exception):
            return fn(self, *args, **kwargs)

    return wrapper


# Just an alias to track failures due to the missing eager ops
xfail_if_mps_unimplemented = xfail_if_mps


def skip_if_triton(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if is_triton_backend(self.device):
            raise unittest.SkipTest("triton not supported")
        return fn(self, *args, **kwargs)

    return wrapper


def skip_if_not_triton(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not is_triton_backend(self.device):
            raise unittest.SkipTest(f"triton backend is required for {self.device}")
        return fn(self, *args, **kwargs)

    return wrapper


def skip_if_dynamic(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if ifdynstaticdefault(True, False):
            raise unittest.SkipTest("associtaive_scan doesn's support lifted SymInts.")
        return fn(self, *args, **kwargs)

    return wrapper


def is_halide_backend(device):
    if getattr(device, "type", device) == "cpu":
        return config.cpu_backend == "halide"
    return config.cuda_backend == "halide"


def is_mps_backend(device):
    return getattr(device, "type", device) == "mps"


def is_triton_backend(device):
    device_type = getattr(device, "type", device)
    if device_type == "cpu":
        return config.cpu_backend == "triton"
    if device_type == "mps":
        return False
    return config.cuda_backend == "triton"


def is_triton_cpu_backend(device):
    return getattr(device, "type", device) == "cpu" and config.cpu_backend == "triton"


def skip_if_triton_cpu(fn):
    import types

    reason = "Triton CPU not supported"

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if is_triton_cpu_backend(self.device):
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)

        return wrapper

    if isinstance(fn, types.FunctionType):
        return decorator(fn)
    else:
        reason = fn
        return decorator


def xfail_if_triton_cpu(fn):
    fn._expected_failure_triton_cpu = True
    return fn


def skip_if_gpu_halide(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if (
            is_halide_backend(self.device)
            and getattr(self.device, "type", self.device) == "cuda"
        ):
            raise unittest.SkipTest("halide not supported")
        return fn(self, *args, **kwargs)

    return wrapper


class skip_if_cpp_wrapper:
    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def __call__(self, fn, *args, **kwargs):
        @functools.wraps(fn)
        def wrapper(test_self):
            if config.cpp_wrapper:
                raise unittest.SkipTest(f"cpp wrapper bug to be fixed: {self.reason}")
            return fn(test_self, *args, **kwargs)

        return wrapper


def is_dynamic_shape_enabled():
    # What's the best way to decide this?
    return not torch._dynamo.config.assume_static_by_default


@instantiate_parametrized_tests
class CommonTemplate:
    def is_dtype_supported(self, dtype: torch.dtype) -> bool:
        device_interface = get_interface_for_device(self.device)
        return device_interface.is_dtype_supported(dtype)

    def test_bool(self):
        def fn(a, b):
            return (
                a + b,
                a * b,
                a & b,
                a | b,
                a ^ b,
                torch.logical_and(a, b),
                torch.logical_or(a, b),
                torch.logical_not(a),
                torch.sign(b),
            )

        self.common(
            fn,
            (
                torch.tensor([True, False, True, False]),
                torch.tensor([False, False, True, True]),
            ),
        )

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skip_if_triton_cpu  # aoti
    @skipIfWindows(msg="aoti not support on Windows")
    def test_aoti_eager_dtype_device_layout(self):
        ns = "aten"
        op_name = "tril_indices"
        dispatch_key = "CPU"
        device = "cpu"
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = "cuda"

        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            row = 128
            col = 256
            offset = 1
            dtype = torch.int32
            layout = torch.strided
            pin_memory = False
            ref = torch.tril_indices(
                row=row,
                col=col,
                offset=offset,
                dtype=dtype,
                layout=layout,
                pin_memory=pin_memory,
                device=device,
            )
            register_ops_with_aoti_compile(
                ns, [op_name], dispatch_key, torch_compile_op_lib_impl
            )
            res = torch.tril_indices(
                row=row,
                col=col,
                offset=offset,
                dtype=dtype,
                layout=layout,
                pin_memory=pin_memory,
                device=device,
            )
            self.assertEqual(ref, res)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skip_if_triton_cpu  # aoti
    @skipIfWindows(msg="aoti not support on Windows")
    def test_aoti_eager_support_out(self):
        ns = "aten"
        op_name = "clamp"
        dispatch_key = "CPU"
        device = "cpu"
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = "cuda"

        inp_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(1.0)
        min_tensor = inp_tensor - 0.05
        max_tensor = inp_tensor + 0.05
        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            ref_out_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(
                -1
            )
            ref_tensor = torch.clamp(
                max=max_tensor, min=min_tensor, input=inp_tensor, out=ref_out_tensor
            )

            ref_out_tensor1 = torch.randn(128, dtype=torch.float, device=device).fill_(
                -1
            )
            ref_tensor1 = torch.clamp(
                max=max_tensor, out=ref_out_tensor1, min=min_tensor, input=inp_tensor
            )

            register_ops_with_aoti_compile(
                ns, [op_name], dispatch_key, torch_compile_op_lib_impl
            )

            res_out_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(
                -1
            )
            res_tensor = torch.clamp(
                max=max_tensor, min=min_tensor, input=inp_tensor, out=res_out_tensor
            )

            self.assertEqual(ref_tensor, res_tensor)
            self.assertEqual(ref_out_tensor, res_out_tensor)

            res_out_tensor1 = torch.randn(128, dtype=torch.float, device=device).fill_(
                -1
            )
            res_tensor1 = torch.clamp(
                max=max_tensor, out=res_out_tensor1, min=min_tensor, input=inp_tensor
            )

            self.assertEqual(ref_tensor1, res_tensor1)
            self.assertEqual(ref_out_tensor1, res_out_tensor1)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skip_if_triton_cpu  # aoti
    @skipIfWindows(msg="aoti not support on Windows")
    def test_aoti_eager_support_str(self):
        ns = "aten"
        op_name = "div"
        dispatch_key = "CPU"
        device = "cpu"
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = "cuda"

        a = torch.randn(128, dtype=torch.float, device=device)
        b = torch.randn(128, dtype=torch.float, device=device)
        rounding_mode_list = ["trunc", "floor"]
        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            # Get ref result from eager
            ref_value_list = []
            for rounding_mode in rounding_mode_list:
                ref_value = getattr(torch.ops.aten, op_name)(
                    a, b, rounding_mode=rounding_mode
                )
                ref_value_list.append(ref_value)

            register_ops_with_aoti_compile(
                ns, [op_name], dispatch_key, torch_compile_op_lib_impl
            )

            # Invoke the pre-compiled kernel and get result.
            res_value_list = []
            for rounding_mode in rounding_mode_list:
                res_value = getattr(torch.ops.aten, op_name)(
                    a, b, rounding_mode=rounding_mode
                )
                res_value_list.append(res_value)

            for ref_value, res_value in zip(ref_value_list, res_value_list):
                self.assertEqual(ref_value, res_value)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skip_if_triton_cpu  # aoti
    @skipIfWindows(msg="aoti not support on Windows")
    def test_aoti_eager_cache_hit(self):
        ns = "aten"
        op_name = "abs"
        dispatch_key = "CPU"
        device = "cpu"
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = "cuda"

        input_tensor = torch.randn(128, dtype=torch.float, device=device)
        kernel_lib_path = aoti_compile_with_persistent_cache(
            ns,
            op_name,
            device,
            False,
            getattr(torch.ops.aten, op_name),
            (input_tensor,),
            {},
        )
        self.assertTrue(Path(kernel_lib_path).exists())

        from unittest import mock

        # Patch the aoti_compile_with_persistent_cache as None to ensure no new kernel is generated
        with mock.patch(
            "torch._inductor.aoti_eager.aoti_compile_with_persistent_cache", None
        ):
            with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
                # Get ref result from eager
                ref_value = getattr(torch.ops.aten, op_name)(input_tensor)

                register_ops_with_aoti_compile(
                    ns, [op_name], dispatch_key, torch_compile_op_lib_impl
                )

                # Invoke the pre-compiled kernel and get result.
                res_value = getattr(torch.ops.aten, op_name)(input_tensor)

                self.assertEqual(ref_value, res_value)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skip_if_triton_cpu  # aoti
    @skipIfWindows(msg="aoti not support on Windows")
    def test_aoti_eager_with_persistent_cache(self):
        def fn(a):
            return torch.abs(a)

        ns = "aten"
        op_name = "abs"

        device = "cpu"
        if self.device.lower() == "cuda":
            device = "cuda"

        input_tensor = torch.randn(128, dtype=torch.float, device=device)
        kernel_lib_path = aoti_compile_with_persistent_cache(
            ns,
            op_name,
            input_tensor.device.type,
            False,
            fn,
            args=(input_tensor,),
            kwargs={},
        )
        self.assertTrue(len(kernel_lib_path) > 0)

        device_kernel_cache = aoti_eager_cache_dir(ns, device)
        kernel_conf = device_kernel_cache / f"{op_name}.json"
        self.assertTrue(kernel_conf.exists())

        json_data = load_aoti_eager_cache("aten", "abs", input_tensor.device.type)
        self.assertTrue(json_data is not None)
        self.assertTrue(isinstance(json_data, list))
        self.assertTrue(len(json_data) > 0)

        op_info = json_data[0]
        self.assertTrue(isinstance(op_info, dict))
        self.assertTrue("meta_info" in op_info)
        self.assertTrue("kernel_path" in op_info)
        kernel_libs_abs_path = []
        for item in json_data:
            kernel_path = device_kernel_cache / item["kernel_path"]
            kernel_libs_abs_path.append(kernel_path.as_posix())

        self.assertTrue(kernel_lib_path in kernel_libs_abs_path)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skip_if_triton_cpu  # aoti
    @skipIfWindows(msg="aoti not support on Windows")
    def test_aoti_eager_with_scalar(self):
        namespace_name = "aten"
        op_name = "add"
        op_overload_name = "Tensor"
        op_name_with_overload = f"{op_name}.{op_overload_name}"

        dispatch_key = "CPU"
        device = torch.device("cpu")
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = torch.device("cuda")

        # Test the difference between scalar tensor and scalar
        a = torch.scalar_tensor(1.0, device=device)
        b = torch.scalar_tensor(2.0, device=device)

        kernel_lib_path = aoti_compile_with_persistent_cache(
            namespace_name,
            op_name_with_overload,
            a.device.type,
            False,
            torch.ops.aten.add,
            args=(a, b),
            kwargs={"alpha": 3.0},
        )
        self.assertTrue(Path(kernel_lib_path).exists())
        device_kernel_cache = aoti_eager_cache_dir(namespace_name, device.type)
        kernel_conf = device_kernel_cache / f"{op_name_with_overload}.json"
        self.assertTrue(kernel_conf.exists())
        json_data = load_aoti_eager_cache(
            namespace_name, op_name_with_overload, a.device.type
        )
        op_info = json_data[0]
        self.assertTrue(isinstance(op_info, dict))
        self.assertTrue("meta_info" in op_info)
        self.assertTrue(len(op_info["meta_info"]) == 3)
        # Scalar Tensor
        self.assertTrue("scalar_value" not in op_info["meta_info"][0])
        self.assertTrue(op_info["meta_info"][0]["sizes"] == [])
        self.assertTrue(op_info["meta_info"][0]["strides"] == [])
        # Scalar Tensor
        self.assertTrue("scalar_value" not in op_info["meta_info"][1])
        self.assertTrue(op_info["meta_info"][1]["sizes"] == [])
        self.assertTrue(op_info["meta_info"][1]["strides"] == [])
        # Scalar
        self.assertTrue("scalar_value" in op_info["meta_info"][2])
        self.assertTrue("sizes" not in op_info["meta_info"][2])
        self.assertTrue("strides" not in op_info["meta_info"][2])

        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            a = torch.randn(128, device=device)
            b = torch.randn(128, device=device)

            scalar_values = [1.0, 2.0, 3.0]
            ref_values = []
            for scalar_value in scalar_values:
                ref_values.append(torch.add(a, b, alpha=scalar_value))

            register_ops_with_aoti_compile(
                namespace_name, [op_name], dispatch_key, torch_compile_op_lib_impl
            )

            res_values = []
            for scalar_value in scalar_values:
                res_values.append(torch.add(a, b, alpha=scalar_value))

            self.assertEqual(len(ref_values), len(res_values))
            self.assertEqual(ref_values, res_values)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_halide  # aoti
    @skip_if_triton_cpu  # aoti
    @skipIfWindows(msg="aoti not support on Windows")
    def test_aoti_eager_override_registration(self):
        namespace_name = "aten"
        dispatch_key = "CPU"
        device = torch.device("cpu")
        if self.device.lower() == "cuda":
            dispatch_key = "CUDA"
            device = torch.device("cuda")

        unary_op_set = ["abs", "acos"]

        def fn(x, op_name=""):
            return getattr(torch, op_name)(x)

        # Invoke torch.compile directly to get referent results
        x = torch.randn(3, 4, device=device)

        ref_array = []
        for unary_op_name in unary_op_set:
            opt_fn = torch.compile(functools.partial(fn, op_name=unary_op_name))
            ref = opt_fn(x)
            ref_array.append(ref)

        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            register_ops_with_aoti_compile(
                namespace_name, unary_op_set, dispatch_key, torch_compile_op_lib_impl
            )

            res_array = []
            for unary_op_name in unary_op_set:
                res_array.append(getattr(torch, unary_op_name)(x))

            for ref, res in zip(ref_array, res_array):
                self.assertEqual(ref, res)

        a = torch.randn(128, device=device)
        min_tensor = torch.randn(128, device=device)
        max_tensor = min_tensor + 0.5

        ref_with_min = torch.ops.aten.clamp(a, min_tensor)
        ref_with_min_max = torch.ops.aten.clamp(a, min_tensor, max_tensor)

        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            register_ops_with_aoti_compile(
                namespace_name, ["clamp"], dispatch_key, torch_compile_op_lib_impl
            )
            res_with_min = torch.ops.aten.clamp(a, min_tensor)
            res_with_min_max = torch.ops.aten.clamp(a, min_tensor, max_tensor)
            self.assertEqual(ref_with_min, res_with_min)
            self.assertEqual(ref_with_min_max, res_with_min_max)

    def test_add_const_int(self):
        def fn(a):
            return (a + 1, torch.add(a, 1, alpha=2))

        for dtype in [torch.float32, torch.int32, torch.int64]:
            self.common(fn, (torch.arange(32, dtype=dtype),))

    def test_add_const_float(self):
        def fn(a):
            return (a + 1.5,)

        self.common(fn, (torch.randn(32),))

    def test_add_inplace_permuted(self):
        if config.cpu_backend == "halide":
            raise unittest.SkipTest(
                "Halide cpu backend does not work for this test case: https://github.com/pytorch/pytorch/issues/140344"
            )

        def fn(x, y):
            return x.add_(y)

        x = torch.ones([2, 12, 13, 17]).transpose(1, 2)
        y = torch.randn([2, 13, 1, 17])

        self.common(fn, (x, y))

    def test_add_complex(self):
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        x = torch.tensor([1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1])
        y = torch.tensor([1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1])

        self.common(fn, (x, y, 2))

    def test_add_complex3(self):
        # fix https://github.com/pytorch/pytorch/issues/115071
        @torch.compile
        def fn(*args):
            a = torch.neg(args[0])
            b = torch.add(args[0], args[0])
            return (a, b)

        x = torch.randn(41, dtype=torch.complex64, device=self.device)
        y = x.clone()
        # should not inplace write to the input
        fn(x)
        self.assertEqual(x, y)

    def test_add_complex4(self):
        @torch.compile
        def fn(a, b):
            c = a + b
            d = a + b
            return c + d

        for dtype in [torch.complex32, torch.complex64, torch.complex128]:
            if not self.is_dtype_supported(dtype):
                continue
            x = torch.tensor(
                [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1],
                dtype=dtype,
                device=self.device,
            )
            y = torch.tensor(
                [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1],
                dtype=dtype,
                device=self.device,
            )
            _, code = run_and_get_code(fn, x, y)
            code = " ".join(code)
            assert_keywords = ["assert_size_stride", "assert_alignment"]
            filtered_lines = [
                line
                for line in code.splitlines()
                if not any(assert_key in line for assert_key in assert_keywords)
            ]
            code = "\n".join(filtered_lines)
            self.assertGreaterEqual(
                code.count("view_dtype" if config.cpp_wrapper else "aten.view"), 3
            )

    def test_add_complex_strided_fallback(self):
        @torch.compile
        def fn(a, b):
            return a + b

        if not self.is_dtype_supported(torch.complex64):
            raise unittest.SkipTest("complex64 not supported on device")

        base = torch.randn(3, 4, dtype=torch.complex64, device=self.device)
        x = base.transpose(0, 1)
        y = base.transpose(0, 1)

        torch._inductor.metrics.reset()
        _, code = run_and_get_code(fn, x, y)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 0)

        code = " ".join(code)
        fallback_markers = [
            "extern_kernels.add",
            "torch.ops.aten.add.Tensor",
        ]
        if config.cpp_wrapper:
            fallback_markers.extend(
                [
                    "aoti_torch_cuda_add_Tensor",
                    "aoti_torch_cpu_add_Tensor",
                ]
            )
        self.assertTrue(
            any(code.count(marker) >= 1 for marker in fallback_markers),
            msg=f"Expected complex add with strided inputs to fall back to extern kernels, got:\n{code}",
        )

    def test_add_complex5(self):
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        x = torch.tensor([[1 + 1j, -1 + 1j], [-2 + 2j, 3 - 3j]])
        y = torch.tensor([[1 + 1j, -1 + 1j], [-2 + 2j, 3 - 3j]])

        self.common(fn, (x, y, 2))

    def test_add_complex6(self):
        # Fix https://github.com/pytorch/pytorch/issues/125745.
        # Add complex tensors with broadcasting.
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        x = torch.tensor([[1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j]])
        y = torch.tensor([[1 + 1j]])

        self.common(fn, (x, y, 2))

    def test_add_complex7(self):
        # Fix https://github.com/pytorch/pytorch/issues/160495
        # Test scalar (0-dimensional) complex tensor addition: 0D + 0D
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        x = torch.rand((), dtype=torch.complex64, device=self.device)
        y = torch.rand((), dtype=torch.complex64, device=self.device)

        self.common(fn, (x, y, 2))

    def test_add_complex8(self):
        # Fix https://github.com/pytorch/pytorch/issues/160495
        # Test scalar complex addition: 1D + 0D
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        x = torch.rand(1, dtype=torch.complex64, device=self.device)
        y = torch.rand((), dtype=torch.complex64, device=self.device)

        self.common(fn, (x, y, 2))

    def test_add_complex9(self):
        # Fix https://github.com/pytorch/pytorch/issues/160495
        # Test scalar complex addition: 0D + 1D
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        x = torch.rand((), dtype=torch.complex64, device=self.device)
        y = torch.rand(1, dtype=torch.complex64, device=self.device)

        self.common(fn, (x, y, 2))

    def test_add_complex10(self):
        # Fix https://github.com/pytorch/pytorch/issues/160495
        # Test scalar complex broadcasting
        def fn(a, b, alpha):
            return torch.add(a, b, alpha=alpha)

        x = torch.randn(2, 3, dtype=torch.complex64, device=self.device)
        y = torch.rand((), dtype=torch.complex64, device=self.device)
        self.common(fn, (x, y, 2))

    def test_concat_add_inplace(self):
        def fn(x, y, z):
            return torch.cat([x, y], dim=1).add_(z)

        x = torch.randn([2, 12, 14, 14])
        y = torch.randn([2, 12, 14, 14])
        z = torch.randn([2, 24, 14, 14])

        self.common(fn, (x, y, z))

    def test_abs(self):
        def fn(a):
            return (a / (torch.abs(a) + 1),)

        self.common(fn, (torch.randn(17),))

    @xfail_if_triton_cpu
    def test_angle(self):
        def fn(a, b, c):
            return torch.angle(a), torch.angle(b), torch.angle(c)

        complex_input = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1, float("nan")]
        )
        real_input = torch.tensor([-1.0, 0.0, 1.0, float("nan")])
        interger_real_input = torch.tensor([-1, 0, 1])
        self.common(fn, (complex_input, real_input, interger_real_input))

    def test_sgn(self):
        def fn(a):
            return torch.sgn(a), torch.sgn(a + 1) - 1

        self.common(fn, [torch.linspace(-10, 10, 41)])

    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    def test_scatter_bf16(self):
        def fn(inp, src, index):
            return inp.scatter_add(0, index, src)

        for dtype in [torch.int64, torch.bool, torch.bfloat16]:
            if not self.is_dtype_supported(dtype):
                continue
            self.common(
                fn,
                [
                    torch.zeros(3, 5, dtype=dtype),
                    torch.ones((2, 5), dtype=dtype),
                    torch.tensor([[0, 1, 2, 0, 0]]),
                ],
            )

    def test_randn_generator(self):
        def fn(a, generator):
            return torch.randn([20, 20], generator=generator, device=a.device)

        self.common(fn, (torch.linspace(-10, 10, 41), None), assert_equal=False)

        # generator not yet supported in dynamo
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "Generator"):
            self.common(fn, (torch.linspace(-10, 10, 41), torch.Generator(self.device)))

    def test_sgn_extremal(self):
        def fn(a):
            return (torch.sgn(a),)

        self.common(fn, [torch.tensor([np.nan, np.inf, -np.inf, 0])])

    def test_max_min(self):
        def fn(a, b):
            return (torch.maximum(a, b), torch.minimum(a, b))

        self.common(fn, (torch.randn(8), torch.randn(8)))
        t1 = torch.randn(8)
        t1[0] = float("nan")
        t2 = torch.randn(8)
        t2[1] = float("nan")
        self.common(fn, (t1, t2))

    def test_neg_max_uint8(self):
        # https://github.com/pytorch/pytorch/issues/93380
        def fn(a, b):
            c = torch.neg(a)
            return torch.maximum(b, c)

        a = torch.randint(256, (1,), dtype=torch.uint8)
        b = torch.randint(256, (8390,), dtype=torch.uint8)
        self.common(fn, (a, b))

    def test_compar(self):
        def fn(x):
            return x.gt(3.5), x.ge(3.5), x.eq(3.5), x.le(2.5), x.lt(3.5), x.ne(3.5)

        a = torch.tensor([3])
        self.common(fn, (a,))

    def test_horizonal_fusion1(self):
        def fn(a, b, c):
            return (a + b, a - c, b * c)

        self.common(
            fn, (torch.randn(8, 16, 16), torch.randn(8, 16, 16), torch.randn(1, 16, 1))
        )

    def test_horizonal_fusion2(self):
        def fn(a, b, c):
            return a + 1, b + 2, c + 3

        self.common(fn, (torch.randn(8, 16, 8), torch.randn(8, 16), torch.randn(16, 8)))

    def test_vertical_fusion1(self):
        def fn(sa, ct, p):
            # From torchbench.pyhpc_equation_of_state
            v17 = -3.087032500374211e-7
            v18 = -1.988366587925593e-8
            v19 = -1.061519070296458e-11
            v20 = 1.550932729220080e-10
            t15 = v19 * ct
            t19 = v17 + ct * (v18 + t15) + v20 * sa
            t20 = 1.0 / t19
            t128 = t19 * p
            return t20 + t128

        self.common(
            fn,
            (
                torch.randn(204, 204, 26),
                torch.randn(204, 204, 26),
                torch.randn(26),
            ),
        )
        assertGeneratedKernelCountEqual(self, 1)

    @config.patch({"fx_graph_cache": False})
    @skipIfWindows(msg="torch._dynamo.exc.Unsupported")
    def test_forced_buffer_realize(self):
        # Test torch._test_inductor_realize forces a buffer to be realized
        def fn(a):
            b = test_operators.realize(a * 2)
            return (b * 2,)

        self.common(fn, (torch.randn(10),))
        self.assertEqual(torch._inductor.metrics.ir_nodes_pre_fusion, 2)

    @config.patch({"fx_graph_cache": False})
    @skipIfWindows(msg="torch._dynamo.exc.Unsupported")
    def test_scheduler_vertical_fusion1(self):
        realize = test_operators.realize

        def fn(sa, ct, p):
            # From torchbench.pyhpc_equation_of_state
            v17 = -3.087032500374211e-7
            v18 = -1.988366587925593e-8
            v19 = -1.061519070296458e-11
            v20 = 1.550932729220080e-10
            t15 = realize(v19 * ct)
            t19 = realize(v17 + ct * (v18 + t15) + v20 * sa)
            t20 = realize(1.0 / t19)
            t128 = realize(t19 * p)
            return t20 + t128

        self.common(
            fn,
            (
                torch.randn(204, 204, 26),
                torch.randn(204, 204, 26),
                torch.randn(26),
            ),
        )
        self.assertEqual(torch._inductor.metrics.ir_nodes_pre_fusion, 5)
        assertGeneratedKernelCountEqual(
            self, 1 if not is_cpp_backend(self.device) else 2
        )

    def test_index_propagation(self):
        def copy(x):
            i = torch.arange(x.size(0), device=x.device)
            return x[i]

        x = torch.randn(8, device=self.device)
        copy_opt = torch.compile(copy, backend="inductor")

        expect = copy(x)
        actual = _run_and_assert_no_indirect_indexing(self, copy_opt, x)
        self.assertEqual(expect, actual)

    @dynamo_config.patch("capture_dynamic_output_shape_ops", True)
    # https://github.com/halide/Halide/issues/8308
    @config.patch("halide.scheduler_cpu", "Mullapudi2016")
    @config.patch("halide.scheduler_cuda", "Li2018")
    @config.patch(implicit_fallbacks=True)
    def test_index_propagation_nested_indirect_indexing(self):
        def nested(x, repeats):
            rank = torch.arange(repeats.numel(), device=x.device)
            index = rank.repeat_interleave(repeats, dim=0)
            return torch.index_select(x, index=index, dim=0)

        example_inputs = (
            torch.randn((32, 64), device=self.device),
            repeats := torch.tensor([5, 10, 15], device=self.device),
        )
        torch._dynamo.mark_dynamic(repeats, 0)  # create backed symint

        nested_opt = torch.compile(nested, backend="inductor")

        expect = nested(*example_inputs)
        actual = nested_opt(*example_inputs)
        self.assertEqual(expect, actual)

    def test_index_propagation_flip(self):
        def flip(x):
            i = torch.arange(x.size(0) - 1, -1, -1, device=x.device)
            return x[i]

        x = torch.randn(8, device=self.device)
        flip_opt = torch.compile(flip, backend="inductor")

        expect = flip(x)
        actual = _run_and_assert_no_indirect_indexing(self, flip_opt, x)
        self.assertEqual(expect, actual)

    def test_index_propagation_floordiv(self):
        def repeat_interleave(x, n):
            # e.g. x=[1, 2, 3], n=2 => returns [1, 1, 2, 2, 3, 3]
            i = torch.arange(x.shape[0] * n, device=x.device)
            return x[i // n]

        x = torch.randn(8, 16, device=self.device)
        repeat_interleave_opt = torch.compile(repeat_interleave, backend="inductor")
        # With static shapes we can prove the bound, our dynamic shapes reasoning is not good enough
        has_assert = ifdynstaticdefault(False, True)
        # this should be collapsed to direct indexing
        actual = _run_and_assert_no_indirect_indexing(
            self, repeat_interleave_opt, x, 3, has_assert=has_assert
        )
        expect = torch.repeat_interleave(x, 3, dim=0)
        self.assertEqual(expect, actual)
        self.assertEqual(actual, repeat_interleave(x, 3))

    def test_index_propagation_remainder(self):
        def repeat(x, n):
            # e.g. x=[1, 2, 3], n=2 => returns [1, 2, 3, 1, 2, 3]
            i = torch.arange(x.shape[0] * n, device=x.device)
            return x[i % x.shape[0]]

        x = torch.randn(8, 16, device=self.device)
        repeat_opt = torch.compile(repeat, backend="inductor")

        # With static shapes we can prove the bound, our dynamic shapes reasoning is not good enough
        has_assert = ifdynstaticdefault(False, True)
        # this should be collapsed to direct indexing
        actual = _run_and_assert_no_indirect_indexing(
            self, repeat_opt, x, 3, has_wrapping=False, has_assert=has_assert
        )
        expect = x.repeat(3, 1)
        self.assertEqual(expect, actual)
        self.assertEqual(actual, repeat(x, 3))

    def test_index_propagation_abs(self):
        def reflection_pad_left(x, n):
            # e.g. x=[1, 2, 3], n=2 => returns [3, 2, 1, 2, 3]
            i = torch.arange(x.shape[0] + n, device=x.device)
            return x[(i - n).abs()]

        x = torch.randn(8, device=self.device)
        opt_fn = torch.compile(reflection_pad_left, backend="inductor")

        # With static shapes we can prove the bound, our dynamic shapes reasoning is not good enough
        has_assert = ifdynstaticdefault(False, True)
        # this should be collapsed to direct indexing
        actual = _run_and_assert_no_indirect_indexing(
            self, opt_fn, x, 3, has_wrapping=False, has_assert=has_assert
        )
        expect = reflection_pad_left(x, 3)
        self.assertEqual(expect, actual)

    def test_index_propagation_device_assert_masked(self):
        def fn(a):
            idx = torch.arange(a.size(0), device=a.device)
            padded_idx = torch.constant_pad_nd(idx, (1050, 0))
            padded_idx = torch.where(padded_idx >= 0, padded_idx, padded_idx)
            return a[padded_idx]

        self.common(fn, (torch.randn(1024),))

    def test_index_remainder(self):
        def fn(x, y):
            return x[y % 12]

        self.common(fn, (torch.rand(1024), torch.randint(50, (50,))))

    @xfailIfS390X
    @config.patch(debug_index_asserts=False)
    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_neg_index(self):
        def test(
            fn, inps, has_assert: bool, has_wrapping: bool, vectorize: bool = True
        ):
            fn_opt = torch.compile(fn)
            if is_halide_backend(self.device):
                pass  # no device asserts in halide
            # TODO: remove once https://github.com/pytorch/pytorch/issues/144634
            # is fixed.
            elif is_mps_backend(self.device):
                pass  # no device asserts in MPS
            elif self.device == "cpu" and not is_triton_cpu_backend(self.device):
                _, code = run_and_get_cpp_code(fn_opt, *inps)
                self.assertTrue(("TORCH_CHECK" in code) is has_assert)
                if (
                    cpu_vec_isa.valid_vec_isa_list()
                    and os.getenv("ATEN_CPU_CAPABILITY") != "default"
                ):
                    self.assertTrue(
                        (") ? (" in code or "blendv" in code) is has_wrapping
                    )
                    # Assert that we always vectorize the kernel regardless of wrapping / checks
                    self.assertTrue(("loadu" in code) is vectorize)
            else:
                code = run_and_get_triton_code(fn_opt, *inps)
                self.assertTrue(("tl.where" in code) is has_wrapping)
                self.assertTrue(("device_assert" in code) is has_assert)

        def indirect(a, b):
            return a[b - 1]

        a = torch.rand(1024, device=self.device)
        b = torch.zeros(256, dtype=torch.long, device=self.device)
        test(indirect, (a, b), has_assert=True, has_wrapping=True)

        def direct(x):
            return x[:, -1]

        a = torch.rand(1, 64, 32, device=self.device)
        # Does not even generate a kernel as it's a view
        test(direct, (a,), has_assert=False, has_wrapping=False, vectorize=False)

        def flip(a, b):
            return a[b]

        a = torch.rand(1024, device=self.device)
        b = torch.arange(start=-1, end=-a.numel() - 1, step=-1, device=self.device)
        test(flip, (a, b), has_assert=True, has_wrapping=True)

        # Constant propagate a constant that's negative
        def flip_with_index_constant(a):
            b = torch.arange(start=-1, end=-a.numel() - 1, step=-1, device=a.device)
            return a[b]

        # Wrapping is constant-folded
        test(flip_with_index_constant, (a,), has_assert=False, has_wrapping=False)

        # Operation where we can't prove that the index is always positive or negative
        def pos_and_neg(a):
            b = torch.arange(start=1, end=-a.numel() - 1, step=-1, device=a.device)
            return a[b]

        # It has wrapping but no assert
        test(pos_and_neg, (a,), has_assert=False, has_wrapping=True)

        # We currently don't do constant propagation with float constants
        # We cannot prove this kind of asserts just with bounds. We would need
        # to lift IndexPropagation.shape_env to be accessible in all of Inductor
        def flip_with_index(a):
            b = 1.0 * torch.arange(
                start=-1, end=-a.numel() - 1, step=-1, device=a.device
            )
            b = b.int()
            return a[b]

        test(
            flip_with_index,
            (a,),
            has_assert=ifdynstaticdefault(False, True),
            has_wrapping=False,
            vectorize=True,
        )

        def unsafe_index(a, b):
            return aten._unsafe_index(a, (b,))

        test(unsafe_index, (a, b), has_assert=False, has_wrapping=True)

        def constant_propagation(a):
            b = torch.tensor([2], device=a.device)
            return a[b]

        test(
            constant_propagation,
            (a,),
            has_assert=ifdynstaticdefault(False, True),
            has_wrapping=False,
            vectorize=False,  # There's no loop to vectorize!
        )

        def constant_propagation_neg(a):
            b = torch.tensor([-2], device=a.device)
            return a[b]

        # In symbolic shapes, we know that we can access -2, so no assert is necessary!
        test(
            constant_propagation_neg,
            (a,),
            has_assert=False,
            has_wrapping=False,
            vectorize=False,  # There's no loop to vectorize!
        )

    def test_computed_buffer_inlining(self):
        def flip(x):
            idx = torch.arange(x.size(0) - 1, -1, -1, device=x.device)
            return x[idx], idx

        flip_opt = torch.compile(flip, backend="inductor")
        x = torch.randn(8, device=self.device)

        expect = flip(x)
        actual = _run_and_assert_no_indirect_indexing(self, flip_opt, x)
        self.assertEqual(expect, actual)

    def test__unsafe_masked_index(self):
        def fn(a, mask, idx):
            return aten._unsafe_masked_index(a, mask, idx, 1)

        self.common(
            fn,
            (
                torch.randn(8, device=self.device),
                torch.tensor([True, False, True], device=self.device),
                [torch.tensor([3, 9, 2], device=self.device)],
            ),
        )

    def test__unsafe_masked_index_put_accumulate(self):
        def fn(a, mask, idx, values):
            return aten._unsafe_masked_index_put_accumulate(a, mask, idx, values)

        self.common(
            fn,
            (
                torch.randn(8, device=self.device),
                torch.tensor([True, False, True], device=self.device),
                [torch.tensor([3, 9, 2], device=self.device)],
                torch.randn(3, device=self.device),
            ),
        )

    def test_sum1(self):
        def fn(a, b):
            return ((a + b).sum(-1),)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_sum2(self):
        def fn(a, b):
            return ((a + b).sum([1, 2]), (a + b).sum(-1))

        self.common(fn, (torch.randn(8, 9, 3, 21), torch.randn(8, 9, 3, 21)))

    def test_sum3(self):
        def fn(a, b):
            r1 = a + b
            r2 = r1.sum(-1)
            r3 = torch.squeeze(b) + 10
            return (r1, r2, r3)

        # Mismatched elements: 2 / 10 (20.0%)
        # Greatest absolute difference: 0.0029296875 at index (8,) (up to 1e-05 allowed)
        # Greatest relative difference: 0.0017482517482517483 at index (6,) (up to 0.001 allowed)
        self.common(fn, (torch.randn(10, 10), torch.randn(1, 10)), atol=1e-5, rtol=2e-3)

    def test_sum4(self):
        def fn(a):
            b = a + 1
            c = b.sum(-1)
            d = c + 3
            e = d.sum(-1)
            f = e + 5
            return (f, e, d, c, b)

        self.common(fn, (torch.randn(1, 16, 8, 8),))

    def test_sum5(self):
        def fn(a):
            b = a + 1
            c = b.sum(-1)
            d = c + 3
            e = d.sum(-1)
            f = e + 5
            return (f,)

        self.common(fn, (torch.randn(1, 17, 8, 9),))

    def test_reduction1(self):
        def fn(a):
            return (a.sum(), a.max(), a.min(), a.argmax(), a.argmin())

        self.common(fn, (torch.tensor([float("-inf"), 0.0, float("inf")]),))

    @skip_if_x86_mac()
    def test_reduction2(self):
        def fn(a):
            # FIXME: a.argmax
            return (a.sum(), a.max(), a.min(), a.argmin())

        self.common(fn, (torch.full((4,), float("inf")),))

    @skip_if_x86_mac()
    def test_reduction3(self):
        def fn(a):
            # FIXME: a.argmin
            return (a.sum(), a.max(), a.min(), a.argmax())

        self.common(fn, (torch.full((4,), float("-inf")),))

    def test_reduction4(self):
        if self.device == "cpu":
            raise unittest.SkipTest("Non-deterministic CPU results")

        def fn(a):
            return (a.argmax(-1), a.argmin(-1))

        inputs = (torch.ones(128), torch.ones(4, 4, 1))
        for i in inputs:
            self.common(fn, (i,), check_lowp=not is_halide_backend(self.device))

    @config.patch(unroll_reductions_threshold=1)
    def test_reduction5(self):
        if self.device == "cpu":
            raise unittest.SkipTest("Non-deterministic CPU results")

        def fn(a):
            return (a.sum(), a.max(), a.min(), a.argmax())

        self.common(fn, (torch.full((4,), float("-inf")),))

    @skip_if_not_triton
    def test_reduction_config_limit(self):
        """
        This unit-test tests whether we exceed cudaDeviceProperties.maxGridSize in
        triton reduction configs for large size hints. #128826 introduced a scaling XBLOCK
        feature to resolve the issue in reduction configs which may exceed the maxGridSize
        """
        from torch._inductor.runtime.runtime_utils import next_power_of_2
        from torch._inductor.runtime.triton_heuristics import triton_config_reduction

        size_hints = {"x": 67108864, "r0_": 8192}
        for _ in range(4):
            size_hints["x"] = next_power_of_2(size_hints["x"])
            triton_config_reduction(size_hints, 1, 2048, 1, 8)

    def test_prod(self):
        def fn(a):
            return a.prod(0), a.prod(1), a.prod()

        self.common(fn, (torch.rand((10, 10)),))
        self.common(fn, (torch.rand((1, 2050)),))

    def test_unroll_small_reduction(self):
        def fn(x):
            val1, index1 = x.min(-1)
            val2, index2 = x.max(-1)
            return (
                val1,
                index1,
                val2,
                index2,
                x.sum(-1),
                (x > 1).any(-1),
                (x > 0).all(-1),
                x.argmin(-1),
                x.argmax(-1),
                x.amin(-1),
                x.amax(-1),
                x.aminmax(),
            )

        with config.patch(unroll_reductions_threshold=8):
            # small sized reductions will get unrolled
            self.common(fn, (torch.randn(8, 3),))
        torch._dynamo.reset()
        with config.patch(unroll_reductions_threshold=1):
            # make sure things also work if they aren't unrolled
            self.common(fn, (torch.randn(8, 3),))

    def test_multilayer_sum_low_prec(self):
        # fp16 nyi for cpu
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        def fn(a):
            return torch.mean(a)

        self.common(fn, ((torch.rand((10, 3, 352, 352), dtype=torch.float16),)))

    def test_multilayer_prime_size(self):
        def fn(a):
            return torch.max(a), torch.sum(a)

        # Requires masked loading for the intermediate reduction
        sample = torch.full((3999971,), 0, dtype=torch.int64)
        sample[-1] = 1
        self.common(fn, (sample,))

    @skip_if_gpu_halide
    @skipCPUIf(IS_MACOS, "fails on macos")
    def test_multilayer_var(self):
        def fn(a):
            return torch.var(a)

        self.common(
            fn,
            ((torch.rand((10, 3, 352, 352), dtype=torch.float32),)),
            atol=1e-3,
            rtol=1e-3,
        )
        self.common(
            fn,
            ((torch.rand((14923), dtype=torch.float32),)),
            atol=1e-3,
            rtol=1e-3,
        )

    @skipCPUIf(IS_MACOS, "fails on macos")
    @skip_if_halide  # accuracy 4.7% off
    @xfailIfS390X  # accuracy failure
    def test_multilayer_var_lowp(self):
        def fn(a):
            return torch.var(a)

        atol = None
        rtol = None
        if self.device == "cpu" and os.getenv("ATEN_CPU_CAPABILITY") == "default":
            atol = 1e-3
            rtol = 1e-3
        self.common(
            fn,
            (torch.rand((16, 16, 352, 352), dtype=torch.float16),),
            atol=atol,
            rtol=rtol,
        )
        self.common(
            fn, (torch.rand((14923), dtype=torch.float16),), atol=atol, rtol=rtol
        )

    def test_split_cumsum(self):
        def fn(a):
            return torch.cumsum(a, -1)

        for dtype in get_all_dtypes(
            include_bfloat16=False,
            include_bool=True,
            include_complex=False,
            include_half=False,
        ):
            if not self.is_dtype_supported(dtype):
                continue
            # Use low=0 since when the mean value is 0, cumsum at all points
            # tends towards zero which makes the relative error term blow up
            inp = make_tensor(10, 3, 352, 352, low=0, dtype=dtype, device=self.device)
            self.common(fn, (inp.view(-1),), rtol=1e-4, atol=1e-5, check_lowp=False)
            self.common(fn, (inp.view(10, -1),), rtol=1e-4, atol=1e-5, check_lowp=False)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_gpu_halide  # accuracy issue
    def test_split_cumsum_low_prec(self):
        if is_cpp_backend(self.device):
            raise unittest.SkipTest("ir.Scan nyi on CPU")

        def fn(a):
            return torch.cumsum(a.view(-1), 0)

        self.common(
            fn,
            (torch.rand((10, 3, 352, 352), dtype=torch.float16),),
            reference_in_float=True,
            check_lowp=False,
        )

    def test_consecutive_split_cumsum(self):
        def fn(a, b):
            a = a.view(-1)
            b = b.view(-1)
            return torch.cumsum(a, 0) + torch.cumsum(b, 0)

        dtype_a = torch.float32
        dtype_b = torch.float64

        ctx = (
            contextlib.nullcontext()
            if self.is_dtype_supported(dtype_a) and self.is_dtype_supported(dtype_b)
            else self.assertRaises(TypeError)
        )

        with ctx:
            a = make_tensor(10, 3, 352, 352, low=0, dtype=dtype_a, device=self.device)
            b = make_tensor(10, 3, 352, 352, low=0, dtype=dtype_b, device=self.device)

            self.common(fn, (a, b), rtol=1e-4, atol=1e-5, check_lowp=False)

    @config.patch(max_autotune_pointwise=True)
    def test_split_cumsum_index(self):
        # Split scan uses a workspace that needs to be zeroed before use.
        # data[index] does indirect indexing that should catch issues if the
        # workspace is not zeroed.
        def fn(lengths, data):
            offsets = torch.cumsum(lengths, 0)
            return data[offsets]

        lengths = torch.full((2**14,), 2**2, dtype=torch.int64, device=self.device)
        lengths[-2] = 3
        lengths[-1] = 3
        data = make_tensor((2**16,), dtype=torch.float32, device=self.device)
        self.common(fn, (lengths, data))

    def test_split_cumprod(self):
        def fn(a):
            return torch.cumprod(a, -1)

        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            if not self.is_dtype_supported(dtype):
                continue
            inp = _large_cumprod_input(
                (10, 10000), dim=1, dtype=dtype, device=self.device
            )
            self.common(fn, (inp,), atol=1e-5, rtol=1e-4, check_lowp=False)

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_gpu_halide  # accuracy issue
    def test_split_cumprod_low_prec(self):
        if is_cpp_backend(self.device):
            raise unittest.SkipTest("ir.Scan nyi on CPU")

        def fn(a):
            return torch.cumprod(a.view(-1), 0)

        for dtype in [torch.float16, torch.bfloat16]:
            if not self.is_dtype_supported(dtype):
                continue
            inp = _large_cumprod_input(
                (10, 10000), dim=1, dtype=dtype, device=self.device
            )
            self.common(
                fn,
                (inp,),
                reference_in_float=True,
                check_lowp=False,
            )

    def test_consecutive_split_cumprod(self):
        def fn(a, b):
            return torch.cumprod(a, 0) + torch.cumprod(b, 0)

        dtype_a = torch.float32
        dtype_b = torch.float64

        ctx = (
            contextlib.nullcontext()
            if self.is_dtype_supported(dtype_a) and self.is_dtype_supported(dtype_b)
            else self.assertRaises(TypeError)
        )

        with ctx:
            a = _large_cumprod_input((10000,), dim=0, dtype=dtype_a, device=self.device)
            b = _large_cumprod_input((10000,), dim=0, dtype=dtype_b, device=self.device)

            self.common(fn, (a, b), atol=1e-5, rtol=1e-5, check_lowp=False)

    @skip_if_halide  # scan ops
    # TODO: support lifted symints when dynamic
    @torch._dynamo.config.patch(
        {"dynamic_shapes": False, "assume_static_by_default": True}
    )
    def test_custom_scan_op(self):
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        def sum_combine(a, b):
            return a + b

        from torch._higher_order_ops.associative_scan import associative_scan

        a = torch.randn(100, 100, device=self.device)
        expect = torch.cumsum(a, 0)
        actual = associative_scan(sum_combine, a, 0)
        self.assertEqual(expect, actual)

        def logcumsum_combine(a, b):
            min_v = torch.minimum(a, b)
            max_v = torch.maximum(a, b)
            mask = (min_v != max_v) | ~min_v.isinf()
            return torch.where(mask, max_v + (min_v - max_v).exp().log1p(), a)

        expect = torch.logcumsumexp(a, 0)
        actual = associative_scan(logcumsum_combine, a, 0)
        self.assertEqual(expect, actual)

    @skip_if_halide  # scan ops
    # TODO: support lifted symints when dynamic
    @torch._dynamo.config.patch(
        {"dynamic_shapes": False, "assume_static_by_default": True}
    )
    def test_custom_scan_op_compiled(self):
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        from torch._higher_order_ops.associative_scan import associative_scan

        def sum_combine(a, b):
            return a + b

        def fn(a, b, dim):
            diff = (a - b).abs()
            sad = associative_scan(sum_combine, diff, dim)
            return sad.sum(dim)

        a = torch.randn(100, 100, device=self.device)
        b = torch.randn(100, 100, device=self.device)
        self.common(fn, (a, b, 0))
        cfn = torch.compile(fn)
        _, code = run_and_get_code(cfn, a, b, 0)

        # Check everything is fused into a single kernel
        FileCheck().check_not("run(").check_regex(
            r"triton_.*\.run\(arg[01]_1, arg[12]_1, buf1,"
        ).check_not("run(").run(code[0])

    @skip_if_halide  # scan ops
    # TODO: support lifted symints when dynamic
    @torch._dynamo.config.patch(
        {"dynamic_shapes": False, "assume_static_by_default": True}
    )
    def test_custom_scan_op_multi_input(self):
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        def argmax_combine(a, b):
            a_value, a_index = a
            b_value, b_index = b
            mask = (a_value > b_value) | ((a_value == b_value) & (a_index > b_index))
            return (
                torch.where(mask, a_value, b_value),
                torch.where(mask, a_index, b_index),
            )

        from torch._higher_order_ops.associative_scan import associative_scan

        a = torch.randn(100, 100, device=self.device)
        expect = torch.cummax(a, 0)

        idx = torch.arange(100, device=self.device).view(100, 1).expand(100, 100)
        actual = associative_scan(argmax_combine, (a, idx), 0)
        self.assertEqual(expect, actual)

    @skip_if_halide  # scan ops
    # TODO: support lifted symints when dynamic
    @torch._dynamo.config.patch(
        {"dynamic_shapes": False, "assume_static_by_default": True}
    )
    def test_custom_scan_would_split(self):
        if self.device != "cuda":
            raise unittest.SkipTest("associative_scan only supported on GPU")

        def combine_linear_recurrence(left, right):
            xl, fl = left
            xr, fr = right
            x = xl * fr + xr
            f = fl * fr
            return x, f

        def eager_scan(x, g):
            x, g = x.to(torch.float64), g.to(torch.float64)
            x_out = torch.empty_like(x)
            g_out = torch.empty_like(g)
            x_out[:, 0] = x[:, 0]
            g_out[:, 0] = g[:, 0]
            for i in range(1, x.shape[1]):
                x_out[:, i], g_out[:, i] = combine_linear_recurrence(
                    (x_out[:, i - 1], g_out[:, i - 1]),
                    (x[:, i], g[:, i]),
                )
            return x_out.float(), g_out.float()

        @torch.compile
        def compiled_scan(x, f):
            from torch._higher_order_ops.associative_scan import associative_scan

            x, f = associative_scan(combine_linear_recurrence, (x, f), dim=1)
            return x, f

        x = torch.randn(1, 129, 2, device=self.device)
        f = torch.randn(1, 129, 2, device=self.device)
        expect = eager_scan(x, f)
        actual = compiled_scan(x, f)
        self.assertEqual(expect, actual)

    def test_embedding_bag_byte_unpack(self):
        if self.device != "cpu":
            raise unittest.SkipTest(f"No {GPU_TYPE} implementation (it returns empty)")

        def fn(a):
            return torch.ops.quantized.embedding_bag_byte_unpack(a)

        M, N = 32, 64
        scales = torch.randn(M, 1).view(torch.uint8)
        offsets = torch.randn(M, 1).view(torch.uint8)
        data = torch.randint(0, 255, (M, N), dtype=torch.uint8)
        packed = torch.cat([data, scales, offsets], dim=-1)
        self.common(fn, [packed])

    @xfail_if_mps_unimplemented
    @skipIfXpu(msg="No _weight_int8pack_mm implementation on XPU")
    def test_int8_weight_only_quant(self):
        def convert_weight_to_int8pack(b):
            b_int8pack, b_scales, _ = _dynamically_quantize_per_channel(
                b, -128, 127, torch.int8
            )
            return b_int8pack, b_scales

        def fn(a, b_int8pack, b_scales, c):
            res = torch._weight_int8pack_mm(a, b_int8pack, b_scales)
            res = res + c
            return res

        m = 32
        k = 32
        n = 48
        a = torch.rand((m, k), dtype=torch.bfloat16)
        b = torch.rand((n, k), dtype=torch.bfloat16)
        c = torch.rand((m, n), dtype=torch.bfloat16)
        b_int8pack, b_scales = convert_weight_to_int8pack(b)
        self.common(fn, (a, b_int8pack, b_scales, c))

    @xfail_if_mps_unimplemented
    @xfail_if_triton_cpu
    @skipCUDAIf(True, "No _dyn_quant_pack_4bit_weight implementation on CUDA")
    @skipIfRocm
    @skipIfXpu(msg="No _dyn_quant_pack_4bit_weight implementation on XPU")
    def test__dyn_quant_pack_4bit_weight(self):
        q_group = 32
        k = 128
        n = 128

        torch.manual_seed(1)
        b = torch.rand((k, n), dtype=torch.float32)
        in_features = b.size(0)
        out_features = b.size(1)

        def dyn_quant_pack_4bit_weight(b, in_features, out_features):
            b_uint8, b_scales_and_zeros = _group_quantize_tensor_symmetric(
                b, n_bit=4, groupsize=q_group
            )

            if q_group == in_features:
                b_scales_and_zeros = b_scales_and_zeros.to(torch.float)
            else:
                b_scales_and_zeros = b_scales_and_zeros.to(torch.bfloat16)
            b_int4pack = torch._dyn_quant_pack_4bit_weight(
                b_uint8, b_scales_and_zeros, None, q_group, in_features, out_features
            )

            return b_int4pack, b_scales_and_zeros

        def fn(b, in_features, out_features):
            b_int4pack, _ = dyn_quant_pack_4bit_weight(b, in_features, out_features)
            return b_int4pack

        self.common(fn, (b, in_features, out_features))

    @xfail_if_mps_unimplemented
    @xfail_if_triton_cpu
    @skipCUDAIf(True, "No _dyn_quant_matmul_4bit implementation on CUDA")
    @skipIfRocm
    @skipIfXpu(msg="No _dyn_quant_matmul_4bit implementation on XPU")
    def test__dyn_quant_matmul_4bit(self):
        q_group = 32
        m = 32
        k = 128
        n = 128

        torch.manual_seed(1)
        a = torch.rand((m, k), dtype=torch.float32)
        b = torch.rand((k, n), dtype=torch.float32)
        in_features = b.size(0)
        out_features = b.size(1)

        def dyn_quant_pack_4bit_weight(b, in_features, out_features):
            b_uint8, b_scales_and_zeros = _group_quantize_tensor_symmetric(
                b, n_bit=4, groupsize=q_group
            )

            if q_group == in_features:
                b_scales_and_zeros = b_scales_and_zeros.to(torch.float)
            else:
                b_scales_and_zeros = b_scales_and_zeros.to(torch.bfloat16)
            b_int4pack = torch._dyn_quant_pack_4bit_weight(
                b_uint8, b_scales_and_zeros, None, q_group, in_features, out_features
            )

            return b_int4pack, b_scales_and_zeros

        def fn(a, q_group, in_features, out_features):
            b_int4pack, _ = dyn_quant_pack_4bit_weight(b, in_features, out_features)
            res = torch._dyn_quant_matmul_4bit(
                a,
                b_int4pack,
                q_group,
                in_features,
                out_features,
            )
            return res

        self.common(fn, (a, q_group, in_features, out_features))

    def test_expanded_reduction(self):
        def fn(x, y):
            z = x * y
            return z.sum((0, 1))

        atol = 1e-3
        rtol = 1e-3
        self.common(
            fn, (torch.randn(2, 197, 256), torch.randn(2, 1, 256)), atol=atol, rtol=rtol
        )

    @skip_if_gpu_halide
    def test_min_max_reduction(self):
        def fn(a, b):
            return (
                (a + b).max(),
                (a + b).min(),
                torch.amax(a + 1, keepdim=True),
                torch.amin(b + 1, keepdim=True),
            )

        dtypes = [torch.float, torch.float16]
        if self.is_dtype_supported(torch.bfloat16):
            dtypes += [torch.bfloat16]
        for dtype in dtypes:
            self.common(fn, (torch.randn(8, 8).to(dtype), torch.randn(8, 8).to(dtype)))

    @skip_if_halide  # bug in nan handling
    def test_min_max_reduction_nan(self):
        def fn(a):
            return (torch.max(a), torch.min(a))

        t1 = torch.randn(32)
        t1[16] = float("nan")
        self.common(fn, (t1,))

    @skip_if_halide  # bug in nan handling
    def test_fmin_fmax(self):
        def fn(a, b):
            return (
                torch.fmin(a, b),
                torch.fmax(a, b),
                torch.fmax(a + 1, torch.tensor(0.0)),
            )

        self.common(
            fn,
            (
                torch.tensor(
                    [-10.0, 10.0, float("nan"), float("nan"), float("nan"), 3, 4]
                ),
                torch.tensor(
                    [float("nan"), float("nan"), -10.0, 10.0, float("nan"), 4, 3]
                ),
            ),
        )

    def test_sum_int(self):
        def fn(x):
            return 2 * x.sum(-1) + x.sum()

        dtypes = torch.bool, torch.uint8, torch.int
        inps = [torch.randint(2, (64,), dtype=dtype) for dtype in dtypes]

        for i in inps:
            self.common(fn, (i,), check_lowp=False)

    def test_sum_dtype(self):
        sum_dtype = torch.double if self.device != "mps" else torch.bfloat16

        def fn(x):
            return x * x.sum(-1, dtype=sum_dtype) + x.sum(dtype=sum_dtype)

        self.common(fn, (torch.ones(32, 32) * 70,))

    @skip_if_halide
    def test_cummin(self):
        def fn(x):
            return x.cummin(0)

        self.common(
            fn, (torch.rand(16, 32),), check_lowp=not is_halide_backend(self.device)
        )
        self.common(fn, (torch.rand(1),), check_lowp=not is_halide_backend(self.device))
        self.common(fn, (torch.rand(0),), check_lowp=not is_halide_backend(self.device))

    def test_cumsum(self):
        def fn(x):
            return x.cumsum(0), x.cumsum(1)

        # Persistent reductions
        self.common(
            fn, (torch.rand(16, 32),), check_lowp=not is_halide_backend(self.device)
        )
        self.common(
            fn, (torch.rand(20, 30),), check_lowp=not is_halide_backend(self.device)
        )

        # Non-persistent reduction
        self.common(
            fn,
            (torch.rand(100, 4000),),
            check_lowp=not is_halide_backend(self.device),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cumsum_zero_dim(self):
        def fn(x):
            return x.cumsum(0), x.cumsum(-1)

        a = torch.rand(())
        self.common(fn, (a,))

    def test_cumsum_no_mask(self):
        def fn(x):
            return x.cumsum(-1)

        # Persistent reduction
        a = torch.rand((1, 1024))
        self.common(
            fn, (a,), check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device))
        )

        # Non-persistent reduction
        b = torch.rand((1, 8192))
        self.common(
            fn,
            (b,),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cumprod_zero_dim(self):
        def fn(x):
            return x.cumprod(0), x.cumprod(-1)

        a = torch.rand(())
        self.common(fn, (a,))

    def test_cumsum_inf(self):
        def fn(x):
            return x.cumsum(-1)

        _dtype = torch.float64

        def make_tensor(shape):
            return torch.full(shape, float("inf"), device=self.device, dtype=_dtype)

        ctx = (
            contextlib.nullcontext()
            if self.is_dtype_supported(_dtype)
            else self.assertRaises(TypeError)
        )
        with ctx:
            cfn = torch.compile(fn)

            for n in [100, 10, 100]:
                inp = torch.full((2, n), float("inf"), device=self.device, dtype=_dtype)
                self.assertEqual(cfn(inp), fn(inp))

    @xfail_if_triton_cpu
    def test_logcumsumexp(self):
        def fn(x):
            return x.logcumsumexp(0), x.logcumsumexp(1)

        # Persistent reductions
        self.common(
            fn,
            (torch.rand(16, 32),),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
        )
        self.common(
            fn,
            (torch.rand(20, 30),),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
        )

        # Non-persistent reduction
        self.common(
            fn,
            (torch.rand(100, 4000),),
            check_lowp=not (TEST_WITH_ROCM or is_halide_backend(self.device)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_logcumsumexp_zero_dim(self):
        def fn(x):
            return x.logcumsumexp(0), x.logcumsumexp(-1)

        a = torch.rand(())
        self.common(fn, (a,))

    def test_clamp(self):
        def fn(a, b):
            return (a.clamp(-0.1, 0.1), b.clamp(0), torch.clamp(a + b, max=0))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_clamp_type_promotion(self):
        tgt_dtype = torch.double if self.device != "mps" else torch.half

        def fn(a):
            b = torch.tensor(1.0, dtype=tgt_dtype, device=self.device)
            c = torch.full((4,), 2, device=self.device)
            return a.clamp(min=b, max=c)

        self.common(fn, (torch.randint(4, (4,)),))

    def test_clamp_type_promotion_non_tensor(self):
        def fn(a):
            return a.clamp(min=1.5), a.clamp(min=2)

        self.common(fn, (torch.randint(4, (4,)),))

    @skip_if_gpu_halide
    @xfail_if_triton_cpu
    def test_dist(self):
        def fn(a, b):
            return (
                torch.dist(a, b),
                torch.dist(a, b, p=1.2),
            )

        self.common(fn, (torch.randn(4, 4), torch.randn(4, 4)))

    @xfail_if_mps
    @skip_if_halide  # different pow accuracies
    @xfail_if_triton_cpu
    def test_norm_constant_overflow(self):
        def fn(a):
            return (
                torch.norm(a, p=-41.0, dim=1),
                torch.norm(a, p=-41.0, dim=0),
            )

        self.common(fn, (torch.randn(4, 1, 4),))

    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8311
    def test_dist_bf16(self):
        def fn(a, b):
            return torch.dist(a.to(torch.bfloat16), b.to(torch.bfloat16))

        if not self.is_dtype_supported(torch.bfloat16):
            raise unittest.SkipTest(
                f"torch.bfloat16 not supported for device {self.device}"
            )
        self.common(fn, (torch.randn(4, 4), torch.randn(4, 4)))

    def test_arange1(self):
        def fn(x):
            rng1 = torch.arange(8 * 8, dtype=torch.float32, device=x.device).view(8, 8)
            rng2 = torch.arange(10, 18, device=x.device)
            tmp = x * rng1
            return tmp, tmp + rng2

        self.common(fn, (torch.randn(8, 8),))

    def test_arange2(self):
        def fn(x):
            rng1 = torch.arange(8, device=x.device)
            return (x + rng1,)

        self.common(fn, (torch.randint(4, (8, 8)),), check_lowp=False)

    def test_arange3(self):
        def fn(x):
            return x + torch.ops.aten.arange.start_step(
                0, 53, 4, dtype=torch.int64, device=x.device
            )

        self.common(fn, (torch.randn(14),))

    def test_arange4(self):
        def fn(x):
            return x - torch.arange(512, -512, -1.0, device=x.device)

        self.common(fn, (torch.randn(1024),))

    def test_arange5(self):
        def fn(step, device):
            return torch.arange(512, -512, step, device=device)

        compiled_fn = torch.compile(fn)

        # NOTE: use assertEqual to check dtypes which self.common doesn't do
        for step in (-1, -1.0):
            expect = fn(step, self.device)
            actual = compiled_fn(step, self.device)
            self.assertEqual(expect, actual)
        self.assertEqual(expect, actual)

    def test_arange6(self):
        def fn(x):
            return torch.arange(0.1, 8.0001, 1, dtype=x.dtype, device=x.device)

        # Test that float arguments are truncated to int when dtype is set explicitly
        make_arg = functools.partial(
            make_tensor, device=self.device, requires_grad=False
        )
        self.common(fn, (make_arg(1, dtype=torch.float32),))
        self.common(fn, (make_arg(1, dtype=torch.int64),))

    def test_linspace1(self):
        def fn(x):
            return torch.linspace(0.125, 0.875, 7, device=x.device) + x

        self.common(fn, (torch.randn(1, 7),))

    def test_linspace2(self):
        def fn(x):
            return torch.linspace(0, 2, 1, device=x.device) + x

        self.common(fn, (torch.randn(1, 1),))

    def test_linspace3(self):
        def fn(x):
            return torch.linspace(0, 2, 0, device=x.device)

        self.common(fn, (torch.Tensor([]),))

    @requires_multigpu()
    def test_linspace4(self):
        def fn(x):
            return torch.linspace(0, 2, 0, device=f"{GPU_TYPE}:1")

        self.common(fn, (torch.Tensor([]),))

    def test_tensor1(self):
        def fn(x):
            return torch.tensor([1], device=x.device) + x, torch.tensor(
                5, device=x.device
            )

        self.common(fn, (torch.randn(10),))

    def test_tensor2(self):
        def fn(x):
            return torch.tensor(list(range(2, 40, 2)), device=x.device) + x

        self.common(fn, (torch.randn(1),))

    def test_tensor3(self):
        def fn(x):
            return (
                torch.tensor([], device=x.device),
                torch.tensor([1, 2], device=x.device) + 1,
                torch.tensor([1, 2, 3], device=x.device) + 2,
                torch.tensor([1, 2, 3, 4], device=x.device) + x,
            )

        self.common(fn, [torch.randn(4)])

    def test_views1(self):
        def fn1(x, y):
            return (x.view(size2) + y,)

        def fn2(x, y):
            return ((x + 1).view(size2) + y,)

        views = [
            ([5 * 7], [5, 7]),
            ([2 * 3 * 4 * 5 * 6 * 7], [2, 3, 4, 5, 6, 7]),
            ([2 * 3, 4, 5, 6 * 7], [2, 3, 4, 5, 6, 7]),
            ([10 * 5, 20], [10, 5, 20]),
            ([1, 10, 1], [10]),
            ([10, 1, 10, 1, 10], [10, 100]),
            ([2, 2, 2, 2], [4, 4]),
        ]
        for size1, size2 in views:
            self.common(fn1, (torch.randn(size1), torch.randn(size2)))
            self.common(fn2, (torch.randn(size1), torch.randn(size2)))

        for size2, size1 in views:
            self.common(fn1, (torch.randn(size1), torch.randn(size2)))
            self.common(fn2, (torch.randn(size1), torch.randn(size2)))

    def test_views2(self):
        for size1, size2 in [
            ([2, 2, 2, 2], [4, -1]),
            ([10, 1, 10, 1, 10], [-1, 100]),
            ([10 * 5, 20], [10, -1, 20]),
        ]:

            def fn1(x):
                return (x.view(size2) + 1,)

            def fn2(x):
                return ((x * 2).view(size2) + 1,)

            self.common(fn1, (torch.randn(size1),))
            self.common(fn2, (torch.randn(size1),))

    def test_views3(self):
        # example taken from hf_BigBird
        def forward(arg1, arg2):
            index = torch.ops.aten.index(arg1, [arg2])
            view_1 = torch.ops.aten.view(index, [1, 2232, 64])
            view_2 = torch.ops.aten.view(view_1, [1, 12, 62, 192])
            return view_2

        self.common(
            forward,
            (
                rand_strided((64, 64), (64, 1), torch.float32),
                rand_strided((2232,), (1,), torch.int64),
            ),
        )

    def test_views4(self):
        # example taken from hf_BigBird
        def forward(arg1, arg2):
            arg1 = arg1.index_select(0, arg2)
            arg1 = torch.ops.aten.view(arg1, [2, 3, 4, 5, 5])
            arg1 = torch.ops.aten.view(arg1, [2, 3, 2, 10, -1])
            return arg1

        self.common(
            forward,
            (
                torch.randn(12, 5, 5),
                torch.randint(0, 11, (24,)),
            ),
        )

    def test_views5(self):
        # tensor with shape 0 in any dimension
        def forward(x):
            y = x[:, 4:]
            return y.view(len(y), -1, 4)

        self.common(
            forward,
            (torch.randn(4, 4, 4, 4),),
        )

    def test_views6(self):
        def forward(x):
            x = torch.ops.aten.relu(x)
            s = torch.ops.aten.slice(x, 0, 0, 9223372036854775807)
            s = torch.ops.aten.slice(s, 1, 0, 9223372036854775807)
            s = torch.ops.aten.slice(s, 3, 0, 0)
            y = torch.ops.aten.view(s, [4, 2, -1])
            return y

        self.common(
            forward,
            (torch.randn(4, 2, 4, 4),),
        )

    def test_views7(self):
        # x.view(dtype)
        def forward(x, y):
            x = (x + 1).to(torch.float32)
            y = (y + 1).to(torch.int32)
            return x.view(torch.int32), y.view(torch.float32)

        self.common(
            forward,
            (
                torch.rand(2, 3, dtype=torch.float32),
                torch.randint(10, (2, 3), dtype=torch.int32),
            ),
        )

    def test_torch_device_split(self):
        def fn(x):
            return x.split(2)

        x = torch.rand(10)

        with x.device:
            out = torch.compile(fn, backend=lambda gm, _: gm)(x)
            ref = fn(x)
            for a, b in zip(out, ref):
                self.assertTrue(torch.allclose(a, b))

    def test_relu(self):
        def fn(a, b):
            return (torch.relu(a), torch.relu(a + b) / 10)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_exp(self):
        def fn(a, b):
            return (torch.exp(a), torch.exp(a + b))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_exp2(self):
        def fn(a, b):
            return (torch.exp2(a), torch.exp2(a + b), torch.pow(2, -torch.abs(a - b)))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    @skipIfXpu(msg="logaddexp_xpu not implemented for ComplexFloat")
    @skipCUDAIf(True, "Not implemented for CUDA")
    def test_logaddexp(self):
        self.common(
            torch.logaddexp,
            (
                torch.randn(8, 8).to(dtype=torch.complex64),
                torch.randn(8, 8).to(dtype=torch.complex64),
            ),
        )

    def test_sigmoid(self):
        def fn(a, b):
            return (torch.sigmoid(a), torch.sigmoid(a + b))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    @xfail_if_triton_cpu
    def test_round(self):
        def fn(a, b):
            return torch.round(a), torch.round(b + 1), torch.round(a, decimals=2)

        # without manual_seed, there is some chance this test fails due to:
        # https://github.com/triton-lang/triton/issues/530
        torch.manual_seed(0)

        # with *100 we are always getting a number exactly at .5 which we don't do right in half
        self.common(fn, (torch.randn(8, 8) * 100, torch.randn(8, 8) * 10))

    @xfail_if_triton_cpu
    def test_round_correctness(self):
        if self.device == "cuda":
            raise unittest.SkipTest("need to debug tl.libdevice on A100/V100")

        def fn(a):
            return torch.round(a)

        dtype = torch.float64 if self.device != "mps" else torch.float32
        self.common(
            fn,
            [torch.arange(-10, 10, 0.1, dtype=dtype)],
            check_lowp=False,
        )

    @xfail_if_triton_cpu
    def test_builtins_round(self):
        def fn(x, i):
            return x[: round(i / 2 + 1)] + round(i / 2)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(5, dtype=torch.int, device=self.device)
        with torch.no_grad():
            for i in range(1, 6):
                self.assertEqual(cfn(x, i), fn(x, i))

    @xfail_if_triton_cpu
    def test_builtins_round_float_ndigits_pos(self):
        def fn(x, i):
            return x + round(i / 2 * 123.4567, 1)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2, device=self.device)
        i = 2

        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    @xfail_if_triton_cpu
    def test_builtins_round_float_ndigits_zero(self):
        def fn(x, i):
            return x + round(i / 2 * 123.4567, 0)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2, device=self.device)
        i = 2

        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    @xfail_if_triton_cpu
    def test_builtins_round_float_ndigits_neg(self):
        def fn(x, i):
            return x + round(i / 2 * 123.4567, -1)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2, device=self.device)
        i = 2

        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    def test_builtins_round_int_ndigits_pos(self):
        def fn(x, i):
            return x + round(i, 1)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2, device=self.device)
        i = 123

        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    def test_builtins_round_int_ndigits_zero(self):
        def fn(x, i):
            return x + round(i, 0)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2, device=self.device)
        i = 123

        with torch.no_grad():
            self.assertEqual(cfn(x, i), fn(x, i))

    def test_silu(self):
        def fn(a):
            return (torch.nn.functional.silu(a),)

        self.common(fn, (torch.randn(8, 8),))

    @skip_if_halide  # halide has buggy nan handling
    def test_nan_to_num(self):
        def fn(a):
            return (
                torch.nan_to_num(a),
                torch.nan_to_num(a, nan=3.0),
                torch.nan_to_num(a, nan=None),
                torch.nan_to_num(a, posinf=4.0),
                torch.nan_to_num(a, neginf=5.0),
                torch.nan_to_num(a, nan=3.0, posinf=4.0, neginf=5.0),
            )

        self.common(
            fn,
            (torch.tensor((float("nan"), float("inf"), float("-inf"), 1.0)),),
            check_lowp=False,  # a much more elaborate test is required to match finfo max's for float and half
        )

    def test_one_hot(self):
        def fn(a):
            return torch.nn.functional.one_hot(a, 8) + 1

        self.common(
            fn,
            (torch.arange(100).view(4, 5, 5) % 8,),
            check_lowp=False,
        )

    def test_div1(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(fn, (torch.randn(8, 8) * 100, torch.randn(8, 8) * 100))

    def test_div2(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(fn, (torch.randint(-100, 100, [8, 8]), 100 * torch.randn(8, 8)))

    def test_div3(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        a = torch.randint(1, 100, [8, 8])
        self.common(fn, (a * 2, a))

    def test_div4(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(
            fn,
            (torch.randint(-100, 0, [8, 8]), torch.randint(1, 10, [8, 8])),
        )

    def test_div5(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        # divide a scalar
        self.common(fn, (torch.randint(-100, 0, [8, 8]), 16))

    def test_div6(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        # treat boolean as integer
        self.common(
            fn,
            (torch.ones([8, 8], dtype=torch.bool), torch.randint(-100, -1, [8, 8])),
        )

    @skip_if_triton_cpu  # divide by zero; cannot xfail because it crashes process
    def test_div7(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(
            fn,
            (
                torch.randint(2**32, 2**40, [100, 100]),
                torch.randint(-10, -1, [100, 100]),
            ),
        )

    def test_div8(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a * 0.5, b, rounding_mode=None),
                aten.div(a, b * 1.0, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(fn, (1024, 100))

    def test_div9(self):
        def fn(x):
            return (torch.div(42, x), aten.true_divide(42, x), aten.div.Tensor(42, x))

        self.common(fn, (torch.randn(8),))

    @skip_if_triton_cpu  # divide by zero; cannot xfail because it crashes process
    def test_div_zero_dim(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        for dtype in (torch.float32, torch.int64):
            self.common(
                fn,
                (
                    make_tensor(10, device=self.device, dtype=dtype),
                    make_tensor((), device=self.device, dtype=dtype, exclude_zero=True),
                ),
            )
            self.common(
                fn,
                (
                    make_tensor((), device=self.device, dtype=dtype),
                    make_tensor(10, device=self.device, dtype=dtype, exclude_zero=True),
                ),
            )

    @skip_if_triton_cpu  # divide by zero; cannot xfail because it crashes process
    def test_div_prim(self):
        def fn(a, b):
            return (torch.ops.prims.div(a, b),)

        for dtype in (torch.float32, torch.int64):
            self.common(
                fn,
                (
                    make_tensor(100, device=self.device, dtype=dtype),
                    make_tensor(
                        100, device=self.device, dtype=dtype, exclude_zero=True
                    ),
                ),
            )

    def test_floordiv(self):
        def fn_floor_input(a, i):
            n = (i * 1.234) // 8.234
            return a + n

        self.common(
            fn_floor_input,
            (make_tensor(10, device=self.device, dtype=torch.float32), 33),
        )

        def fn_int_input(a, i):
            n = i // 8
            return a + n

        self.common(
            fn_int_input, (make_tensor(10, device=self.device, dtype=torch.float32), 33)
        )

    def test_div_precision(self):
        # Reproducer for https://github.com/pytorch/pytorch/issues/101039

        def forward(x, y):
            z = x.div(y)
            return F.softmax(z, dim=-1)

        query = torch.randn(1, 10, 40)
        key = torch.randn(1, 2, 40)
        x = torch.matmul(query, key.transpose(-2, -1))
        self.common(forward, (x, 1e-6))

        x = torch.tensor(
            [
                [
                    [
                        [-16.1649, 5.6846, -5.1022, -9.1134],
                        [-11.5552, -2.2615, -12.8913, 10.6538],
                        [-7.1666, -5.3333, 2.0776, -9.7984],
                        [7.4469, -2.3948, 2.7371, 0.9201],
                    ],
                    [
                        [-8.0361, -16.3771, 22.7741, 4.4685],
                        [20.8047, -0.7771, -2.4355, -2.2299],
                        [3.8343, -2.0914, -2.4077, 2.2740],
                        [-15.8663, -2.7015, -12.5241, -3.0040],
                    ],
                    [
                        [-2.5139, 14.4393, -3.7186, 1.2255],
                        [5.6742, 14.1842, -8.5976, 16.8366],
                        [-9.7358, -3.0279, 11.8164, -4.0787],
                        [-9.0621, 8.2580, 29.9486, -2.4107],
                    ],
                    [
                        [7.3622, 12.5640, -20.5592, 13.6237],
                        [-11.5640, 0.8832, 16.7275, -2.5009],
                        [-2.0953, -12.2276, -26.2633, 4.5268],
                        [15.3329, -11.7492, 6.5650, -9.2483],
                    ],
                ],
                [
                    [
                        [7.9980, -4.9369, 3.1508, 5.2994],
                        [3.8052, 3.9514, 8.4987, -10.5045],
                        [-2.6827, -4.0010, -4.0611, 6.4091],
                        [-19.0318, 6.4073, 2.8923, 8.0250],
                    ],
                    [
                        [7.1650, -3.4585, 5.7720, -5.0305],
                        [-0.9765, -3.0086, 11.7114, 8.0555],
                        [-3.1027, -3.5514, 9.6182, -8.8526],
                        [-9.2348, -6.0239, 6.2528, -6.7221],
                    ],
                    [
                        [11.5936, 22.4139, -0.4089, -4.9889],
                        [14.8217, -2.3426, -17.6189, 3.7427],
                        [1.9546, -13.0902, 8.6293, -7.2457],
                        [-7.6900, -4.5796, 9.6332, -10.2631],
                    ],
                    [
                        [0.8027, -1.0955, 14.8404, -0.2673],
                        [3.2143, -1.8640, -2.9678, 6.5165],
                        [-3.9865, 6.5230, 6.3019, -0.4247],
                        [8.3185, -13.5076, 27.0986, -1.6792],
                    ],
                ],
            ]
        )
        x = torch.matmul(x, x)
        y = torch.tensor([[[0.6331]], [[1.6358]], [[-0.3459]], [[1.0196]]])
        self.common(forward, (x, y))

    def test_div_softmax_symfloat(self):
        def forward(x, y):
            z = x.div(y * x.shape[-1])
            return F.softmax(z, dim=-1)

        query = torch.randn(1, 10, 40)
        key = torch.randn(1, 2, 40)
        x = torch.matmul(query, key.transpose(-2, -1))

        cf = torch.compile(forward, dynamic=True)
        cf(x, 1e-5)
        cf(x, 1e-6)

    def test_div_presicion_accuracy(self):
        # fix https://github.com/pytorch/pytorch/issues/157959
        def forward(x, y):
            return (x / y).sum()

        x = torch.rand((5, 5))
        y = 101
        self.common(forward, (x, y))

    def test_mul_softmax_symfloat(self):
        def forward(x, y):
            z = x.mul(y * x.shape[-1])
            return F.softmax(z, dim=-1)

        query = torch.randn(1, 10, 40)
        key = torch.randn(1, 2, 40)
        x = torch.matmul(query, key.transpose(-2, -1))

        cf = torch.compile(forward, dynamic=True)
        cf(x, 1e-5)
        cf(x, 1e-6)

    def test_div_by_zero(self):
        def fn(x, runtime_zero, runtime_neg_zero):
            zero = torch.zeros_like(x)
            return (
                x / 0.0,
                x / -0.0,
                zero / 0.0,
                x / zero,
                x / -zero,
                zero / zero,
                x / runtime_zero,
                # NOTE: -runtime_zero doesn't work as -(0.0) is broken in triton
                x / runtime_neg_zero,
                runtime_zero / runtime_neg_zero,
            )

        a = torch.randn(10)
        zero = torch.zeros(10)
        neg_zero = -zero
        self.common(fn, (a, zero, neg_zero))

    def test_both_scalars(self):
        def fn(a, b):
            return (
                aten.add(a, b),
                aten.add(b, a),
                aten.sub(a, b),
                aten.sub(b, a),
                aten.mul(a, b),
                aten.mul(b, a),
            )

        self.common(fn, (4, 3.3), reference_in_float=False)

    def test_sum_keepdims(self):
        def fn(a, b):
            return (torch.sum(a + b, -1, keepdim=True),)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    @skip_if_cpu
    @skip_if_halide  # only 32-bit indexing
    @largeTensorTest("4GB", inductor=True)
    def test_large_tensor_reduction(self):
        # Test 64-bit indexing works correctly
        def fn(a):
            return torch.max(a)

        t = torch.ones(2**32, dtype=torch.int8, device=self.device)
        t[-1] = 2

        # self.common OOMs here because it copies inputs to check for mutations
        compiled_fn = torch.compile(fn)
        actual = compiled_fn(t)
        expect = torch.tensor(2, dtype=torch.int8, device=self.device)
        self.assertEqual(actual, expect)

    @skip_if_cpu
    @skip_if_gpu_halide  # only 32-bit indexing
    def test_large_broadcast_reduction(self):
        # Test 64-bit indexing works correctly when inputs are less than 32-bit
        # but intermediate tensors require 64-bit indexing
        def fn(a, b):
            return torch.max(a + b)

        t1 = torch.ones(1, 2**16, dtype=torch.int8, device=self.device)
        t2 = torch.ones(2**16, 1, dtype=torch.int8, device=self.device)

        t1[-1, -1] = 2
        t2[-1, -1] = 2

        # self.common OOMs here because it copies inputs to check for mutations
        compiled_fn = torch.compile(fn)
        actual = compiled_fn(t1, t2)
        expect = torch.tensor(4, dtype=torch.int8, device=self.device)
        self.assertEqual(actual, expect)

    @skip_if_halide  # only 32-bit indexing
    @largeTensorTest("4GB", inductor=True)
    def test_large_pointwise(self):
        def fn(a):
            return a + 1

        t = torch.ones(2**31 + 1, dtype=torch.int8, device=self.device)
        compiled_fn = torch.compile(fn)
        actual = compiled_fn(t)

        # Can't use assertEqual as it expands broadcasted inputs
        del t
        if torch.device(self.device).type == GPU_TYPE:
            getattr(torch, GPU_TYPE).empty_cache()

        self.assertTrue((actual == 2).all())

    @skip_if_halide  # only 32-bit indexing
    @largeTensorTest("3GB", inductor=True)
    def test_large_offset_pointwise(self):
        # Test 64-bit indexing is used when input views a tensor that can be
        # indexed with 32-bit strides but the storage offset pushes it over
        # INT_MAX
        def fn(a):
            return a + 4

        t = torch.ones(2**31 + 1, dtype=torch.int8, device=self.device)
        t[2**30 :] = 0
        compiled_fn = torch.compile(fn)
        actual = compiled_fn(t[2**30 :])
        self.assertTrue((actual == 4).all())

    @skip_if_halide  # only 32-bit indexing
    @largeTensorTest("2GB", inductor=True)
    def test_large_strided_reduction(self):
        # Test 64-bit indexing is used when input numel is less than INT_MAX
        # but stride calculations go above INT_MAX
        def fn(a):
            return torch.max(a)

        storage = torch.ones(2**31 + 1, dtype=torch.int8, device=self.device)
        view = storage[::32]
        view[-1] = 2

        compiled_fn = torch.compile(fn)
        actual = compiled_fn(view)
        expect = torch.tensor(2, dtype=torch.int8, device=self.device)
        self.assertEqual(actual, expect)

    def test_softmax(self):
        def fn(a, b):
            return (torch.softmax(a + b, -1), torch.softmax(a, 0), torch.softmax(b, 1))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_log_softmax(self):
        def fn(a, b):
            return (F.log_softmax(a + b, -1), F.log_softmax(a, 0), F.log_softmax(b, 1))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_transpose(self):
        def fn(a, b):
            return (
                torch.t(a) + b,
                torch.transpose(b * 2, 0, 1) + 10,
            )

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_permute1(self):
        def fn(a):
            return (
                torch.permute(a + 1, [2, 1, 4, 0, 3]) + 2,
                torch.permute(a, [2, 1, 4, 0, 3]) + 2,
            )

        self.common(fn, (torch.randn(2, 2, 2, 2, 2),))

    def test_permute2(self):
        def fn(a):
            a = a.unfold(0, 2, 1)
            a = torch.unsqueeze(a, 1)
            a = torch.permute(a, [0, 2, 3, -3])
            return (a,)

        self.common(fn, (torch.randn(4, 4),))

    def test_expand(self):
        def fn(a):
            return (
                (a + 1).expand(3, 4, 2, 3, 2) + 2,
                a.expand(2, 1, 2, 3, 2) + 2,
            ), a.expand(2, -1, 5, -1)

        self.common(fn, (torch.randn(2, 1, 2),))

    def test_squeeze1(self):
        def fn(a):
            return ((a + 1).squeeze() + 2, a.squeeze() + 2)

        self.common(fn, (torch.randn(1, 2, 1, 2, 2, 1, 1),))

    def test_squeeze2(self):
        def fn(a):
            return ((a + 1).squeeze(-1).squeeze(2) + 2, a.squeeze(0) + 2)

        self.common(fn, (torch.randn(1, 2, 1, 2, 2, 2, 1),))

    def test_squeeze_varargs(self):
        def fn(x):
            return x.squeeze(1, 2).clone()

        a = torch.randn(1024, 1, 1)
        self.common(fn, (a,))

    def test_simplify_loops(self):
        def fn(a, b):
            return a + b

        self.common(
            fn,
            (
                torch.randn(2, 3, 4, 5, 6),
                torch.randn(4, 2, 3, 5, 6).permute(1, 2, 0, 3, 4),
            ),
        )

    def test_unsqueeze(self):
        def fn(a):
            return (
                torch.unsqueeze(a + 1, -1) + 2,
                torch.unsqueeze(a, 2) + 2,
                torch.unsqueeze(a + 1, 0) + 2,
                torch.unsqueeze(a, -2) + 2,
            )

        self.common(
            fn,
            (
                torch.randn(
                    2,
                    2,
                    2,
                    2,
                ),
            ),
        )

    def test_unsqueeze_inplace(self):
        def fn(a):
            tmp1 = a + 1
            aten.unsqueeze_(tmp1, 2)
            tmp2 = aten.unsqueeze_(a + 1, 0) + 2
            return (tmp1, tmp2)

        self.common(
            fn,
            (
                torch.randn(
                    2,
                    2,
                    2,
                    2,
                ),
            ),
        )

    def test_addmm(self):
        def fn(a, b, c):
            return (torch.addmm(a + 1, b + 2, c + 3) + 4,)

        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randn(8, 8),
                torch.randn(8, 8),
            ),
        )

    def test_addmv(self):
        def fn(a, b, c):
            return torch.addmv(a, b, c)

        cfn = torch.compile(backend="inductor")(fn)
        input = torch.tensor([2], dtype=torch.int32)
        mat = torch.tensor(np.random.randn(0, 0), dtype=torch.int32)
        vec = torch.tensor([])
        with torch.no_grad():
            self.assertEqual(cfn(input, mat, vec), fn(input, mat, vec))

    # https://github.com/pytorch/pytorch/issues/98979
    @skipCUDAIf(True, "cuda failed for float64 linear")
    @skipIfXpu(msg="Double and complex datatype matmul is not supported in oneDNN")
    def test_linear_float64(self):
        _dtype = torch.float64
        ctx = (
            contextlib.nullcontext()
            if self.is_dtype_supported(_dtype)
            else self.assertRaises(TypeError)
        )
        with ctx:
            mod = torch.nn.Sequential(torch.nn.Linear(8, 16).to(_dtype)).eval()
            with torch.no_grad():
                self.common(mod, (torch.randn(2, 8).to(_dtype),))

    def test_linear1(self):
        mod = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.Sigmoid(),
            ToTuple(),
        )
        self.common(mod, (torch.randn(2, 8),))

    def test_linear2(self):
        mod = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
        )
        self.common(
            mod,
            (torch.randn(2, 8),),
            atol=1e-3,
            rtol=0.01,
        )

    def test_bmm1(self):
        def fn(a, b):
            return (
                torch.bmm(a, b),
                torch.bmm(a + 1, b + 2) + 3,
            )

        self.common(
            fn,
            (
                torch.randn(2, 8, 8),
                torch.randn(2, 8, 8),
            ),
            check_lowp=False,
        )
        self.common(
            fn,
            (
                torch.randn(1, 16, 8),
                torch.randn(1, 8, 10),
            ),
            check_lowp=False,
        )

    def test_bmm2(self):
        def fn(a, b):
            return torch.bmm(a.permute(0, 2, 1), b)

        self.common(
            fn,
            (
                torch.randn(1, 8, 8),
                torch.randn(1, 8, 8),
            ),
            check_lowp=False,
        )

    @skipIfPy312  # segfaults
    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    def test_mixed_mm(self):
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))

        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randint(-128, 127, (8, 8), dtype=torch.int8),
            ),
            check_lowp=True,
        )

    @skipIfPy312  # segfaults
    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    def test_mixed_mm2(self):
        def fn(a, b, scale, bias):
            return torch.mm(a, b.to(a.dtype)) * scale + bias

        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randint(-128, 127, (8, 8), dtype=torch.int8),
                torch.randn(8),
                torch.randn(8),
            ),
            check_lowp=True,
        )

    @skipIfPy312  # segfaults
    @skipCUDAIf(not SM80OrLater, "Requires sm80")
    def test_mixed_mm3(self):
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))

        # (256, 256) @ (256, 256) so different block sizes are tried out during autotuning
        self.common(
            fn,
            (
                torch.randn(256, 256),
                torch.randint(-128, 127, (256, 256), dtype=torch.int8),
            ),
            rtol=0.01,
            atol=0.1,
        )

    @with_tf32_off
    def test_uint4x2_mixed_mm(self):
        def fn(a, b):
            return torch.mm(
                a,
                torch.cat((b & 0xF, b >> 4), 1)
                .reshape(-1, b.shape[1])
                .to(a.dtype)
                .sub(8),
            )

        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randint(0, 255, (4, 8), dtype=torch.uint8),
            ),
            check_lowp=True,
        )

    @skipIfXpu
    def test_mm_mixed_dtype(self):
        def fn(a, b):
            return torch.mm(a, b)

        t1 = torch.arange(6, dtype=torch.float, device=self.device).view(2, 3)
        t2 = torch.arange(9, dtype=torch.int64, device=self.device).view(3, 3)

        msg = "expected .* and .* to have the same dtype, but got: .* != .*"
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(t1, t2)
        if config.cpp_wrapper:
            msg = "aoti_torch_.* API call failed at .*"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.compile(fn)(t1, t2)

    @skipIfXpu
    @xfail_if_mps_unimplemented  # linear for non-float inputs
    def test_linear_mixed_dtype(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super(Net, self).__init__()  # noqa: UP008
                self.fc1 = nn.Linear(3, 3)

            def forward(self, x):
                x = self.fc1(x.permute(1, 2, 0))
                return x

        fn = Net().to(self.device)
        t = torch.arange(27, device=self.device).view(3, 3, 3)

        msg = "expected .* and .* to have the same dtype, but got: .* != .*"
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(t)
        if config.cpp_wrapper:
            msg = "aoti_torch_.* API call failed at .*"
        with self.assertRaisesRegex(RuntimeError, msg):
            with torch.no_grad():
                torch.compile(fn)(t)
        with self.assertRaisesRegex(RuntimeError, "Autograd not support dtype:.*"):
            torch.compile(fn)(t)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_linear_dynamic_maxautotune(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        @torch.compile(dynamic=True)
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x)

        x = torch.randn(10, 1)
        torch._dynamo.mark_dynamic(x, 0)
        self.common(Model(), (x,))

    def test_scalar_input(self):
        def fn(x, y):
            a = torch.div(x, y, rounding_mode="floor")
            return a

        self.common(fn, [torch.randint(5, (1, 8)), 5400])

    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_scalar_output(self):
        def fn(arg0_1, arg2_1):
            arg1_1 = arg2_1.size(1)
            view = torch.ops.aten.view.default(arg2_1, [-1, arg1_1])
            embedding = torch.ops.aten.embedding.default(arg0_1, view)
            full = torch.ops.aten.full.default([1, arg1_1], 1, dtype=torch.float32)
            return (full, arg1_1, embedding)

        arg0_1 = rand_strided((32128, 768), (768, 1), device="cpu", dtype=torch.float32)
        arg2_1 = rand_strided((1, 22), (22, 1), device="cpu", dtype=torch.int64)
        self.common(fn, [arg0_1, arg2_1])

    def test_shape_prop_torch_ones(self):
        class Model(torch.nn.Module):
            def forward(self, attention_scores):
                extended_attention_mask = torch.ones(
                    8, 1, 1, 512, device=attention_scores.device
                )
                attention_scores = attention_scores + extended_attention_mask

                return attention_scores

        mod = Model().eval()
        with torch.no_grad():
            self.common(
                mod,
                (torch.randn(8, 12, 512, 512),),
            )

    @slowTest
    @expectedFailureCodegenDynamic
    @config.patch({"freezing": True})
    def test_conv_bn_fuse(self):
        # For gpu path, there is an accuracy issue
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv bn test")

        # fails dynamic check which bn is fused, and there will not have loops vars.
        input_shapes = {1: (112,), 2: (112, 112), 3: (55, 55, 55)}
        conv_modules = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        bn_modules = {
            1: torch.nn.BatchNorm1d,
            2: torch.nn.BatchNorm2d,
            3: torch.nn.BatchNorm3d,
        }
        options = itertools.product(
            [1, 2, 3],
            [True, False],
            [1, 3],
            [1, 2],
            [1, 4],
        )

        for (
            dim,
            bias,
            kernel_size,
            dilation,
            groups,
        ) in options:
            oC = 32 * groups
            iC = 3 * groups
            x_shape = (1, iC) + input_shapes[dim]
            mod = torch.nn.Sequential(
                conv_modules[dim](
                    iC,
                    oC,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                ),
                bn_modules[dim](oC),
            ).eval()
            test_memory_format = [torch.contiguous_format]
            # TODO: GPU path doesn't support channels_last now.
            if not HAS_GPU and dim > 1:
                channels_last = (
                    torch.channels_last if dim == 2 else torch.channels_last_3d
                )
                test_memory_format.append(channels_last)
            for memory_format in test_memory_format:
                v = torch.randn(x_shape, dtype=torch.float32).to(
                    memory_format=memory_format
                )
                with torch.no_grad():
                    self.common(
                        mod,
                        (v,),
                    )

    def test_conv_functional_bn_fuse(self):
        # For gpu path, there is an accuracy issue
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv bn test")

        # Define a BatchNorm using functional BN.
        class BatchNorm(torch.nn.BatchNorm2d):
            def __init__(
                self,
                num_features,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
                device=None,
                dtype=None,
            ):
                factory_kwargs = {"device": device, "dtype": dtype}
                super().__init__(
                    num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats,
                    **factory_kwargs,
                )

            def forward(self, x):
                if self.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = self.momentum

                if self.training and self.track_running_stats:
                    # TODO: if statement only here to tell the jit to skip emitting this when it is None
                    if self.num_batches_tracked is not None:  # type: ignore[has-type]
                        self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                        if self.momentum is None:  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(
                                self.num_batches_tracked
                            )
                        else:  # use exponential moving average
                            exponential_average_factor = self.momentum
                if self.training:
                    bn_training = True
                else:
                    bn_training = (self.running_mean is None) and (
                        self.running_var is None
                    )
                x = F.batch_norm(
                    x,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    (
                        self.running_mean
                        if not self.training or self.track_running_stats
                        else None
                    ),
                    (
                        self.running_var
                        if not self.training or self.track_running_stats
                        else None
                    ),
                    self.weight,
                    self.bias,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )
                return x

        v = torch.randn(1, 3, 556, 56, dtype=torch.float32)
        mod = torch.nn.Sequential(
            torch.nn.Conv2d(
                3,
                64,
                kernel_size=3,
                dilation=1,
                groups=1,
                bias=True,
            ),
            BatchNorm(64),
        ).eval()
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    @skipIfRocm
    @xfail_if_mps  # Expected to find .run(
    def test_conv_inference_heuristics(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest(f"{GPU_TYPE} only test")

        in_channels = 6
        out_channels = 6
        kernel_size = 3
        groups = 3

        grouped_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, groups=groups
        ).to(self.device)

        input_tensor = torch.randn(1, in_channels, 10, 10).to(self.device)

        # Perform the forward pass
        @torch.compile()
        def foo(m, inp):
            return m(inp)

        if self.device != "xpu":
            with torch.no_grad():
                _, code = run_and_get_code(foo, grouped_conv, input_tensor)
                # no to channels last permuting before kernel
                if config.cpp_wrapper:
                    FileCheck().check_not("  call_triton").check("_convolution(").run(
                        code[0]
                    )
                else:
                    FileCheck().check_not(".run(").check(".convolution(").run(code[0])

        # in out should do channels last in inference
        in_channels = 8
        out_channels = 4
        kernel_size = 3

        # Create the convolution layer
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size).to(self.device)

        input_tensor = torch.randn(1, in_channels, 10, 10).to(self.device)

        with torch.no_grad():
            _, code = run_and_get_code(foo, conv_layer, input_tensor)
            # should be channels last permuting before kernel
            if is_halide_backend(self.device):
                FileCheck().check("halide_kernel_0(").check(".convolution(").run(
                    code[0]
                )
            else:
                FileCheck().check(".run(").check("convolution(").run(code[0])

    def test_upsample_cat_conv(self):
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu upsample_cat_conv test")

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
                self.conv = torch.nn.Conv2d(
                    8,
                    5,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    dilation=1,
                    **kwargs,
                )

            def forward(self, x, y):
                x = self.upsample(x)
                z = torch.cat([x, y], dim=1)
                z = self.conv(z)
                return z

        v1 = torch.randn([8, 2, 12, 26])
        v2 = torch.randn([8, 6, 24, 52])

        with torch.no_grad():
            self.common(
                M().eval(),
                (v1, v2),
            )

    def test_aliased_buffer_reuse(self):
        def fn(x, y):
            x = 2 * x
            y = 2 * y
            c = torch.cat([x, y], dim=-1)
            d = 1 + c
            m = torch.mm(d, d)
            return m[:, :2] + x

        self.common(fn, (torch.randn(4, 2), torch.randn(4, 2)), check_lowp=False)

    def test_slice_view_with_graph_break(self):
        def fn():
            a = torch.tensor([1], device=self.device)
            a = a[0:1]
            b = a.squeeze()
            a[0] = 0
            if a[0] < 1e5:
                pass
            a[0] = 2
            return b

        expect = fn()
        opt_fn = torch.compile(fn)
        actual = opt_fn()
        self.assertEqual(expect, actual)

    def test_view_detach(self):
        def fn(a):
            return a[0].detach()

        self.common(
            fn,
            (torch.randn([4, 4], requires_grad=True),),
        )

    def test_gather1(self):
        def fn(a, b):
            return (
                torch.gather(a.expand([4, 5, 10, 6]), 3, b + 1),
                torch.gather(a.expand([4, 5, 10, 6]), -1, b + 1),
            )

        self.common(
            fn,
            (
                torch.randn([1, 1, 10, 6]),
                torch.randint(5, [4, 5, 10, 1], dtype=torch.int64),
            ),
        )

    def test_gather2(self):
        # 0d tensor
        def fn(a, b):
            return torch.gather(a, 0, b) + torch.gather(a, -1, b)

        x = torch.tensor(123)
        y = torch.tensor(0)
        self.assertEqual(fn(x, y), x + x)

    @xfail_if_mps_unimplemented  # Sparse not supported
    def test_gather3(self):
        def fn(a, b):
            return torch.gather(a, 1, b, sparse_grad=True)

        self.common(
            fn,
            (
                torch.randn([4, 5, 10, 6], requires_grad=True),
                torch.randint(5, [4, 5, 10, 1], dtype=torch.int64),
            ),
        )

    def test_device_assert(self):
        def fn(x, y):
            x = torch.sum(x.view(int(x.shape[0] / 6), 6), dim=1)
            return torch.gather(x, 0, torch.trunc(y).to(torch.int64))

        x1 = torch.randn(30, device=self.device)
        x2 = torch.randn(36, device=self.device)
        dtype = torch.float64 if self.device != "mps" else torch.float32
        y = torch.ones(1, dtype=dtype, device=self.device)

        self.assertEqual(torch.compile(fn)(x1, y), fn(x1, y))
        self.assertEqual(torch.compile(fn)(x2, y), fn(x2, y))

    def test_slice1(self):
        def fn(a):
            return (
                a[:, :10, 0] + a[:, 10:, 0],
                (a + 1)[:, :10, 0] + (a + 1)[:, 10:, 0],
                a[:, -30:, 0],  # negative index out of range
                a[:, :-30, 0],  # negative index out of range
            )

        self.common(
            fn,
            (torch.randn([2, 20, 2]),),
        )

    def test_slice2(self):
        def fn(a):
            return (
                a[:-1, ::2, -1] + a[-1:, 1::2, -2],
                (a + 1)[:-1, ::2, -1] + (a + 2)[-1:, 1::2, -2],
            )

        self.common(
            fn,
            (torch.randn([2, 20, 2]),),
        )

    # It's a view so it doesn't generate a kernel
    @expectedFailureCodegenDynamic
    def test_slice3(self):
        def fn(a, b):
            return torch.ops.aten.slice.Tensor(a, 0, 0, -b)

        x = torch.rand(48, 3, 512, 512)
        self.common(fn, (x, 2))

    @expectedFailureCodegenDynamic
    def test_slice4(self):
        # empty slices that require clamping the start or end
        def fn(a):
            return (
                aten.slice.Tensor(a, 0, 2, 0, 1),
                aten.slice.Tensor(a, 0, a.shape[0], a.shape[0] + 10, 1),
                aten.slice.Tensor(a, 0, -20, 0, 1),
                aten.slice.Tensor(a, 0, -20, -16, 1),
            )

        x = torch.rand(10)
        self.common(fn, (x,))

    def test_split_with_list(self):
        def fn(a, sizes):
            return [t + 1.0 for t in torch.split(a * 2.0, sizes, -1)]

        self.common(fn, (torch.randn(2, 2, 10), [3, 3, 4]))
        self.common(fn, (torch.randn(2, 2, 10), [4, 3, 3]))
        self.common(fn, (torch.randn(2, 2, 10), [1, 2, 3, 4]))

    def test_split_with_integer(self):
        # argument `split_size_or_sections` is integer
        @torch.compile(dynamic=True)
        def f(x, sizes):
            return torch.split(x, sizes, -1)

        # split into equally sized chunks, 10 = 5 + 5
        r1, r2 = f(torch.randn(2, 10), 5)
        self.assertTrue(r1.size() == (2, 5))
        self.assertTrue(r2.size() == (2, 5))

        # split into equally sized chunks, 12 = 4 + 4 + 4
        r1, r2, r3 = f(torch.randn(2, 12), 4)
        self.assertTrue(r1.size() == (2, 4))
        self.assertTrue(r2.size() == (2, 4))
        self.assertTrue(r3.size() == (2, 4))

        # split unevenly, 10 = 3 + 3 + 3 + 1
        r1, r2, r3, r4 = f(torch.randn(2, 10), 3)
        self.assertTrue(r1.size() == (2, 3))
        self.assertTrue(r2.size() == (2, 3))
        self.assertTrue(r3.size() == (2, 3))
        self.assertTrue(r4.size() == (2, 1))

    def test_split_failed(self):
        @torch.compile(backend="inductor")
        def fn(a):
            return torch.split(a, [2, 1, 1], dim=1)

        with self.assertRaisesRegex(RuntimeError, ""):
            fn(torch.randn(1, 5))

    def test_inductor_assert(self):
        @torch.compile(backend="inductor", dynamic=True)
        def fn(a):
            assert a.shape[0] >= 2 and a.shape[1] >= 4
            return a.cos()

        inp = torch.randn(2, 4, 6)
        torch._dynamo.mark_dynamic(inp, 0)
        torch._dynamo.mark_dynamic(inp, 1)
        self.assertEqual(fn(inp), inp.cos())

    def test_split(self):
        def fn(a):
            t = torch.split(a, 3, -1)
            return (t[0], t[1], t[2], t[3])

        def fn2(a):
            return fn(a + 1)

        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
        )

        self.common(
            fn2,
            (torch.randn([2, 2, 10]),),
        )

    @parametrize("dilation", (1, 2))
    @parametrize("dim", (subtest(2), subtest(3)))
    def test_low_memory_max_pool(self, dilation: int, dim: int):
        prims = torch.ops.prims

        def fn(x):
            kernel_size = [3, 3] if dim == 2 else [3, 3, 2]
            stride = [2] * dim
            padding = [1] * dim
            ceil_mode = False

            vals, offsets = prims._low_memory_max_pool_with_offsets(
                x,
                kernel_size,
                stride,
                padding,
                [dilation] * dim,
                ceil_mode,
            )
            indices = prims._low_memory_max_pool_offsets_to_indices(
                offsets,
                kernel_size,
                x.shape[-dim:],
                stride,
                padding,
                dilation=[dilation] * dim,
            )
            return vals, indices, offsets

        self.common(fn, (torch.randn(1, 3, *[10] * dim),))

    def test_to_dtype(self):
        new_dtype = torch.float64 if self.device != "mps" else torch.bfloat16

        def fn(a, b):
            return (
                aten._to_copy(a, dtype=6),
                aten._to_copy(b + 1, dtype=6),
                aten.to(b, new_dtype),
                aten.to(b, torch.bool),
            )

        self.common(
            fn,
            (
                torch.randn([2, 2, 10]),
                torch.randn([2, 2, 10], dtype=new_dtype),
            ),
        )

    @requires_gpu()
    def test_to_device(self):
        def fn(a):
            if a.device.type == "cpu":
                return aten._to_copy(
                    a, device=torch.device(GPU_TYPE), dtype=6, layout=0
                )
            else:
                return aten._to_copy(a, device=torch.device("cpu"), dtype=6, layout=0)

        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
        )

    def test_to_memory_format(self):
        def fn(a, memory_format):
            return a.to(memory_format=memory_format)

        self.common(
            fn,
            (torch.randn([2, 2, 10, 10]), torch.channels_last),
        )
        self.common(
            fn,
            (
                torch.randn([2, 2, 10, 10]).to(memory_format=torch.channels_last),
                torch.contiguous_format,
            ),
        )

    @requires_gpu()
    def test_to_device_constant(self):
        def fn(a):
            d1 = a.device.type
            if d1 == "cpu":
                d2 = GPU_TYPE
            else:
                d2 = "cpu"

            const1 = torch.as_tensor(list(range(64)), device=d2)
            return (
                torch.arange(10, device=d2).to(d1) + a,
                const1.to(d1),
                (const1 + 1).to(d1),
            )

        self.common(
            fn,
            (torch.randn([10]),),
        )

    @requires_gpu()
    @xfail_if_triton_cpu
    def test_multi_device(self):
        def fn(x):
            x = x + 1
            x = x + 2
            x = x.to(device=GPU_TYPE)
            x = x + 3
            x = x + 4
            x = x.cpu()
            x = x + 5
            x = x + 6
            x = x.to(device=GPU_TYPE)
            x = x + 7
            x = x + 8
            x = x.cpu()
            x = x + 9
            x = x + 10
            return x

        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
            check_lowp=False,  # cpu doesn't understand fp16, and there are explicit .cpu() calls
        )

    @skipIfRocm
    @requires_multigpu()
    def test_multi_gpu_device(self):
        # TODO: https://github.com/pytorch/pytorch/issues/92627
        x = torch.rand([4], device=GPU_TYPE)

        def fn(x, y):
            r = torch.ops.aten.div(x, y)
            r = r.to(f"{GPU_TYPE}:1")
            return 2 * r

        self.common(fn, (torch.randn(4), torch.randn(4)), check_lowp=False)

    @requires_multigpu()
    def test_multi_gpu_recompile_on_index(self):
        torch.set_float32_matmul_precision("high")

        def gemm(x, y):
            return x @ y

        failed_guard = None

        def fail(guard):
            nonlocal failed_guard
            failed_guard = guard

        gemm_opt = torch._dynamo.optimize("inductor", guard_fail_fn=fail)(gemm)

        x0 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:0")
        y0 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:0")

        gemm_opt(x0, y0)

        x1 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")
        y1 = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")

        gemm_opt(x1, y1)
        self.assertTrue(failed_guard is not None)
        self.assertTrue(
            "tensor 'x' Tensor device index mismatch. Expected device index to be"
            in failed_guard.reason
        )

    def test_unbind(self):
        def fn(a):
            return torch.unbind(a), torch.unbind(a, -1)

        self.common(
            fn,
            (torch.randn([4, 4, 4]),),
        )

    @skipIfXpu(msg="Incorrect reference on XPU, see issue #165392")
    def test_conv1d_with_permute(self):
        # fix https://github.com/pytorch/pytorch/issues/159462
        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 64, kernel_size=3, padding=1)

            def forward(self, x):
                x = x.permute(0, 2, 1)
                return self.conv(x)

        self.common(ConvModel(), (torch.randn([32, 100, 1]),), check_lowp=False)

    def test_conv1d_depthwise(self):
        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(
                    768,
                    768,
                    kernel_size=(9,),
                    stride=(1,),
                    padding=(4,),
                    groups=768,
                    bias=False,
                )

            def forward(self, x):
                return self.conv(x)

        input_tensor = torch.randn([1, 768, 512]).as_strided(
            (1, 768, 512), (393216, 1, 768)
        )
        self.common(ConvModel(), (input_tensor,), check_lowp=False)

    def test_convolution1(self):
        m = torch.nn.Sequential(
            torch.nn.Conv2d(5, 6, [3, 3]),
            torch.nn.ReLU(),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randn([2, 5, 16, 16]),),
            # Mismatched elements: 10 / 2352 (0.4%)
            # Greatest absolute difference: 5.7220458984375e-05 at index (0, 3, 12, 12) (up to 1e-05 allowed)
            # Greatest relative difference: 0.06512477175897748 at index (0, 4, 11, 9) (up to 0.001 allowed)
            atol=6e-5,
            rtol=0.001,
            # Make sure we compute also with fp16 in the reference. Otherwise,
            # the reference will compute with fp32 and cast back to fp16, which
            # causes numeric differences beyond tolerance.
            reference_in_float=not torch.version.hip,
        )

    def test_convolution2(self):
        def fn(x, w, b):
            # transposed conv
            return (aten.convolution(x, w, b, [4], [0], [1], True, [0], 1),)

        self.common(
            fn,
            (
                torch.randn([2, 32, 90]),
                torch.randn([32, 16, 8]),
                torch.randn([16]),
            ),
            check_lowp=False,
        )

    def test_convolution3(self):
        # Test stride or padding or dilation is 1 element list.
        m = torch.nn.Sequential(
            torch.nn.Conv2d(5, 6, [3, 3], stride=[1], padding=[0], dilation=[1]),
            torch.nn.ReLU(),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randn([2, 5, 16, 16]),),
            atol=6e-5,
            rtol=0.001,
            # Make sure we compute also with fp16 in the reference. Otherwise,
            # the reference will compute with fp32 and cast back to fp16, which
            # causes numeric differences beyond tolerance.
            reference_in_float=not torch.version.hip,
        )

    @skip_if_gpu_halide
    def test_convolution4(self):
        def fn(x, w):
            x = F.conv2d(x, w, groups=w.shape[0])
            return x.sum()

        self.common(
            fn,
            (
                torch.randn([2, 3, 16, 20]),
                torch.randn([3, 1, 5, 5]),
            ),
        )

    def test_convolution5(self):
        def fn(x, w):
            x = F.conv2d(x, w, dilation=[x.size(0)])
            return x.sum()

        x = torch.randn([2, 1, 16, 20])
        w = torch.randn([1, 1, 5, 5])

        torch._dynamo.mark_dynamic(x, 0)

        atol = None
        rtol = None
        if self.device == "xpu":
            # set to float32 default tolerance,
            # check_model_gpu with update rotl to 2e-3 for fp16.
            # fix issue #129974
            atol = 1e-05
            rtol = 1.3e-06
        self.common(fn, (x, w), atol=atol, rtol=rtol)

    def test_conv3d(self):
        m = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, kernel_size=7),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randn([1, 3, 8, 16, 32]),),
            atol=1e-3,
            rtol=0.001,
            # Make sure we compute also with fp16 in the reference. Otherwise,
            # the reference will compute with fp32 and cast back to fp16, which
            # causes numeric differences beyond tolerance.
            reference_in_float=not torch.version.hip,
        )

    def test_conv2d_channels_last(self):
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv2d channels_last")

        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),
            ToTuple(),
        )
        # only weight is channels_last
        self.common(
            m.to(memory_format=torch.channels_last),
            (torch.randn([2, 3, 16, 16]),),
            check_lowp=False,
        )
        # only activation is channels_last
        self.common(
            m,
            (torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last),),
            check_lowp=False,
        )
        # activation and weight are all channels_last
        self.common(
            m.to(memory_format=torch.channels_last),
            (torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last),),
            check_lowp=False,
        )

    def test_conv2d_backward_channels_last(self):
        def fn(grad_output, inp, weight):
            convolution_backward_8 = torch.ops.aten.convolution_backward.default(
                grad_output,
                inp,
                weight,
                [320],
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            )
            return convolution_backward_8

        # only weight is channels_last
        self.common(
            fn,
            (
                torch.randn([2, 320, 8, 8]),
                torch.randn([2, 2048, 8, 8]),
                torch.randn([320, 2048, 1, 1]).to(memory_format=torch.channels_last),
            ),
            check_lowp=False,
        )

    @parametrize(
        "use_block_ptr",
        [subtest(False), subtest(True, decorators=[skip_if_not_triton])],
    )
    def test_conv3d_channels_last(self, use_block_ptr: bool):
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu conv3d channels_last")

        m = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, 1, 1),
            ToTuple(),
        )
        with config.patch({"triton.use_block_ptr": use_block_ptr}):
            # only weight is channels_last
            self.common(
                m.to(memory_format=torch.channels_last_3d),
                (torch.randn([2, 3, 16, 16, 16]),),
            )
            # only activation is channels_last
            self.common(
                m,
                (
                    torch.randn([2, 3, 16, 16, 16]).to(
                        memory_format=torch.channels_last_3d
                    ),
                ),
            )
            # activation and weight are all channels_last
            self.common(
                m.to(memory_format=torch.channels_last_3d),
                (
                    torch.randn([2, 3, 16, 16, 16]).to(
                        memory_format=torch.channels_last_3d
                    ),
                ),
            )

    @skip_if_gpu_halide  # slow
    @xfail_if_mps  # Non-divisible input sizes are not implemented on MPS device
    def test_adaptive_avg_pool2d1(self):
        def fn(x):
            return aten._adaptive_avg_pool2d(x, (6, 6)), aten._adaptive_avg_pool2d(
                x + 1, (2, 5)
            )

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
            check_lowp=False,
        )

        # lowering to avg_pool2d case
        self.common(
            fn,
            (torch.randn(2, 4, 3, 3),),
        )

        # no-op case
        self.common(
            fn,
            (torch.randn(2, 4, 6, 6),),
        )

    @xfail_if_mps  # Non-divisible input sizes are not implemented on MPS device
    def test_adaptive_avg_pool2d2(self):
        # Big kernel size, use fallback
        def fn(x):
            return aten._adaptive_avg_pool2d(x, (4, 4))

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            (torch.randn(2, 4, 21, 21),),
            check_lowp=False,
        )
        assertGeneratedKernelCountEqual(self, 0)

    @xfail_if_mps
    @skip_if_gpu_halide  # slow
    def test_adaptive_max_pool2d1(self):
        def fn(x):
            return aten.adaptive_max_pool2d(x, (6, 6))

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
            check_lowp=False,
        )

        self.common(
            fn,
            (torch.randn(2, 4, 3, 3),),
        )

        # no-op case
        self.common(
            fn,
            (torch.randn(2, 4, 6, 6),),
        )

    @skip_if_gpu_halide  # slow
    def test_adaptive_max_pool2d2(self):
        # Big kernel size, use fallback
        def fn(x):
            return aten.adaptive_max_pool2d(x, (4, 4))

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            (torch.randn(2, 4, 21, 21),),
            check_lowp=False,
        )
        assertGeneratedKernelCountEqual(self, 0)

    @skip_if_gpu_halide  # slow
    def test_adaptive_max_pool2d3(self):
        # test when adaptive_max_pool2d fallbacks to max_pool2d
        def fn(x):
            return aten.adaptive_max_pool2d(x, (2, 2))

        # Big kernel (12 / 2 * 12 / 2 > 25)
        self.common(
            fn,
            (torch.randn(2, 4, 12, 12),),
        )

        # Small kernel
        self.common(
            fn,
            (torch.randn(2, 4, 4, 4),),
        )

    @xfail_if_mps_unimplemented
    def test_fractional_max_pool2d1(self):
        def fn(x, samples):
            return aten.fractional_max_pool2d(x, (3, 3), (2, 2), samples)

        self.common(
            fn, (torch.randn(1, 4, 16, 16), torch.rand(1, 4, 2)), check_lowp=False
        )

    @xfail_if_mps_unimplemented
    def test_fractional_max_pool2d2(self):
        # large kernel size without unrolling

        def fn(x, samples):
            return aten.fractional_max_pool2d(x, (6, 5), (3, 3), samples)

        self.common(
            fn,
            (torch.randn(2, 4, 36, 36), torch.rand(2, 4, 2)),
            check_lowp=False,
        )

    @xfail_if_mps_unimplemented
    def test_fractional_max_pool2d3(self):
        def fn(x, samples):
            return aten.fractional_max_pool2d(x, (1, 1), (16, 16), samples)

        self.common(
            fn, (torch.randn(2, 4, 16, 16), torch.rand(2, 4, 2)), check_lowp=False
        )

    @xfail_if_mps_unimplemented
    @config.patch(fallback_random=True)
    @skip_if_halide  # Can only unroll for loops over a constant extent
    def test_fractional_max_pool2d4(self):
        random.seed(1234)
        torch.manual_seed(1234)

        # check rectangular kernel/output size

        def fn(x):
            return torch.nn.functional.fractional_max_pool2d_with_indices(
                x, (4, 3), (3, 2)
            )

        self.common(fn, (torch.randn(1, 4, 16, 16),), check_lowp=False)

    @xfail_if_mps_unimplemented
    def test_fractional_max_pool2d5(self):
        def fn(x, samples):
            return aten.fractional_max_pool2d(x, (3, 3), (1, 1), samples)

        self.common(
            fn, (torch.randn(2, 4, 6, 6), torch.rand(2, 4, 2)), check_lowp=False
        )

    def test_multi_threading(self):
        model = torch.nn.Linear(2, 3).eval()
        inp = torch.randn(4, 2)

        num_run = 3

        def run_weights_sharing_model(m, inp):
            with torch.no_grad():
                for _ in range(num_run):
                    y = m(inp)

        numb_instance = 2
        threads = []
        compiled_m = torch.compile(model)
        for _ in range(1, numb_instance + 1):
            thread = threading.Thread(
                target=run_weights_sharing_model, args=(compiled_m, inp)
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    @unittest.skipIf(config.is_fbcode(), "fbcode triton error, needs debugging")
    @skip_if_triton_cpu("Flaky on Triton CPU")
    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8311
    def test_adaptive_avg_pool2d_low_prec(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.avgpool(x)
                return x

        mod = Model().to(self.device)
        for dtype in [torch.half, torch.bfloat16]:
            # Skip bfloat16 on MacOS-13 for MPS tests
            if not self.is_dtype_supported(dtype):
                continue
            x = torch.randn(4, 3, 7, 7, device=self.device).to(dtype=dtype)
            opt_mod = torch.compile(mod)
            res = opt_mod(x)
            expected = mod(x)
            self.assertTrue(torch.allclose(res, expected))

    def test_buffer_copied_in_graph(self):
        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))
                self.w1 = torch.nn.Parameter(torch.zeros(1))
                self.w2 = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                self.buf.add_(1)
                return (self.w1 * x * self.w2).sum() + self.buf.sum()

        model_for_eager = MyModel().to(self.device)
        model_for_compile = copy.deepcopy(model_for_eager)

        eager_version_counters = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        compile_version_counters = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        compiled_f = torch.compile(model_for_compile, backend="inductor")

        inp_ref = torch.ones(1, requires_grad=True, device=self.device)
        inp_test = torch.ones(1, requires_grad=True, device=self.device)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        eager_version_counters_after = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        compile_version_counters_after = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        eager_delta = list(
            map(operator.sub, eager_version_counters_after, eager_version_counters)
        )
        compile_delta = list(
            map(operator.sub, compile_version_counters_after, compile_version_counters)
        )

        self.assertEqual(eager_delta, compile_delta)

    @skip_if_gpu_halide
    def test_buffer_copied_in_graph_with_different_shapes(self):
        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(4, 4))
                self.w = torch.nn.Parameter(
                    torch.Tensor([[4, 5], [1, 2], [6, 7], [8, 9]])
                )

            def forward(self, x):
                self.buf.add_(1)
                return (self.w @ x).sum() + self.buf.sum()

        model_for_eager = MyModel().to(self.device)
        model_for_compile = copy.deepcopy(model_for_eager)

        eager_version_counters = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        compile_version_counters = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        compiled_f = torch.compile(model_for_compile, backend="inductor")

        inp_ref = torch.ones(2, 4, requires_grad=True, device=self.device)
        inp_test = torch.ones(2, 4, requires_grad=True, device=self.device)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        eager_version_counters_after = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        compile_version_counters_after = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        eager_delta = list(
            map(operator.sub, eager_version_counters_after, eager_version_counters)
        )
        compile_delta = list(
            map(operator.sub, compile_version_counters_after, compile_version_counters)
        )

        self.assertEqual(eager_delta, compile_delta)

    def test_buffer_batch_norm(self):
        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = torch.nn.BatchNorm1d(100)

            def forward(self, x):
                return self.m(x)

        model_for_eager = MyModel().to(self.device)
        model_for_compile = copy.deepcopy(model_for_eager)

        eager_version_counters = [
            buffer._version for _, buffer in model_for_eager.named_buffers()
        ]
        compile_version_counters = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        compiled_f = torch.compile(model_for_compile, backend="inductor")

        inp_ref = torch.ones(20, 100, requires_grad=True, device=self.device)
        inp_test = torch.ones(20, 100, requires_grad=True, device=self.device)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        eager_version_counters_after = [
            # TODO: remove the + 1 after https://github.com/pytorch/pytorch/issues/120622 is fixed
            (
                buffer._version + 1
                if k in ["m.running_mean", "m.running_var"]
                else buffer._version
            )
            for k, buffer in model_for_eager.named_buffers()
        ]

        compile_version_counters_after = [
            buffer._version for _, buffer in model_for_compile.named_buffers()
        ]

        eager_delta = list(
            map(operator.sub, eager_version_counters_after, eager_version_counters)
        )
        compile_delta = list(
            map(operator.sub, compile_version_counters_after, compile_version_counters)
        )

        self.assertEqual(eager_delta, compile_delta)

    @xfail_if_mps  # Non-divisible input sizes are not implemented on MPS device
    def test_adaptive_avg_pool_with_output_size_0(self):
        m1 = nn.AdaptiveAvgPool1d(0)
        self.common(m1, (torch.randn(1, 2),))
        m2 = nn.AdaptiveAvgPool2d(0)
        self.common(m2, (torch.randn(1, 2, 3),))

    def test_max_pool2d1(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
        )

    @skip_if_gpu_halide  # slow
    def test_max_pool2d2(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    @skip_if_gpu_halide  # slow
    def test_max_pool2d3(self):
        def fn(x):
            # with padding
            return (
                aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [1, 1]),
                aten.max_pool2d_with_indices(
                    x,
                    [
                        3,
                    ],
                    [
                        2,
                    ],
                    [
                        1,
                    ],
                ),
            )

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
        )

    @skip_if_halide  # Can only unroll for loops over a constant extent
    def test_max_pool2d4(self):
        def fn(x):
            # with padding
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [0, 0], [1, 1], True)

        self.common(
            fn,
            (torch.randn([2, 8, 111, 111]),),
        )

    @skip_if_gpu_halide  # slow
    def test_max_pool2d5(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [])

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    @skip_if_gpu_halide  # slow
    @parametrize("dilation", (1, 2))
    def test_max_pool2d6(self, dilation: int):
        # Big kernel size
        def fn(x):
            return aten.max_pool2d_with_indices(
                x, [13, 13], [], dilation=[dilation] * 2
            )

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    # From https://github.com/pytorch/pytorch/issues/94775
    def test_max_pool2d7(self):
        # ceil mode turns on
        def fn(x):
            return torch.nn.functional.max_pool2d(
                x, 1, stride=(2, 2), padding=0, ceil_mode=True
            )

        self.common(
            fn,
            (torch.randn([1, 1, 6, 7]),),
        )

    # From https://github.com/pytorch/pytorch/issues/93384
    def test_max_pool2d8(self):
        # dilation is not 1
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 2], [2, 1], [1, 1], [1, 2])

        self.common(
            fn,
            (torch.randn([2, 2, 3, 6]),),
        )

    def test_avg_pool2d1(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
        )

    def test_avg_pool2d2(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    def test_avg_pool2d3(self):
        def fn(x):
            return (
                aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1]),
                aten.avg_pool2d(
                    x,
                    [
                        3,
                    ],
                    [
                        2,
                    ],
                    [
                        1,
                    ],
                ),
            )

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
            check_lowp=not is_halide_backend(self.device),  # misaligned addr fp16
        )

    def test_avg_pool2d4(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [0, 0], True)

        self.common(
            fn,
            (torch.randn([2, 8, 111, 111]),),
        )

    def test_avg_pool2d5(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1], count_include_pad=False)

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
            check_lowp=not is_halide_backend(self.device),  # misaligned addr fp16
        )

    def test_avg_pool2d6(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1], divisor_override=3)

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
            check_lowp=not is_halide_backend(self.device),  # misaligned addr fp16
        )

    def test_avg_pool2d7(self):
        # Large kernel size, use fallback
        def fn(x):
            return aten.avg_pool2d(x, [13, 13], [1, 1], [0, 0])

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            (-torch.arange(1 * 24 * 24, dtype=torch.float32).view(1, 1, 24, 24),),
        )
        assertGeneratedKernelCountEqual(self, 0)

    def test_avg_pool2d8(self):
        # https://github.com/pytorch/pytorch/issues/100987
        def fn(x):
            return aten.avg_pool2d(
                x, kernel_size=3, stride=2, padding=1, ceil_mode=True
            )

        self.common(
            fn,
            (torch.randn(1, 3, 6, 6),),
            check_lowp=not is_halide_backend(self.device),  # misaligned addr fp16
        )

    @tf32_on_and_off(0.006)
    @skip_if_gpu_halide  # slow
    def test_alexnet_prefix(self):
        def forward(arg6, arg7, arg16):
            convolution = torch.ops.aten.convolution(
                arg16, arg7, arg6, [4, 4], [2, 2], [1, 1], False, [0, 0], 1
            )
            relu = torch.ops.aten.relu(convolution)
            max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices(
                relu, [3, 3], [2, 2]
            )
            getitem = max_pool2d_with_indices[0]
            return (getitem,)

        self.common(
            forward,
            (
                rand_strided((64,), (1,), torch.float32, "cpu"),
                rand_strided((64, 3, 11, 11), (363, 121, 11, 1), torch.float32, "cpu"),
                rand_strided(
                    (16, 3, 224, 224), (150528, 50176, 224, 1), torch.float32, "cpu"
                ),
            ),
            # Mismatched elements: 127 / 746496 (0.0%)
            # Greatest absolute difference: 0.0009765625 at index (1, 62, 7, 16) (up to 1e-05 allowed)
            # Greatest relative difference: 0.05187467899332306 at index (14, 18, 11, 0) (up to 0.001 allowed)
            atol=3e-3,
            rtol=2,
        )

    def test_elu(self):
        def fn(x):
            return aten.elu(x, 1.6732632423543772, 1.0507009873554805) + 2, aten.elu(
                x + 1, 2, 3, 4
            )

        self.common(
            fn,
            (torch.randn([16, 16]),),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_tan(self):
        def fn(x):
            return aten.tan(x) + 2, aten.tan(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_tanh(self):
        def fn(x):
            return aten.tanh(x) + 2, aten.tanh(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    @skip_if_halide  # lgamma not implemented
    @xfail_if_triton_cpu
    def test_lgamma(self):
        def fn(x):
            return aten.lgamma(x) + 2, aten.cos(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_cos(self):
        def fn(x):
            return aten.cos(x) + 2, aten.cos(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_sin(self):
        def fn(x):
            return aten.sin(x) + 2, aten.sin(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_repeat(self):
        def fn(x):
            return (
                x.repeat(0, 1, 1, 1),
                x.repeat(2, 2, 3, 1),
                x.repeat(8, 1, 1, 1),
                x.repeat(2, 1, 1, 1, 1, 1),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    def test_repeat_as_strided(self):
        # Reproducer for #127474

        def fn(x):
            view_size = (3, 2)
            full = x.repeat((3, 2))
            view = torch.as_strided(full, view_size, full.stride())
            result = view + view

            return result

        self.common(fn, (torch.randn(1, 1),))

    def test_as_strided_on_views(self):
        # https://github.com/pytorch/pytorch/issues/163286
        def fn(a):
            c = a.view(-1)
            # convert to float16
            d = c.view(torch.float16)
            e = d.as_strided((2, 5), (1, 1))
            # convert back to bfloat16
            f = e.view(torch.bfloat16)
            g = f.as_strided((10, 10), (1, 1))
            return g

        a = torch.randn(10, 10, dtype=torch.bfloat16)
        self.common(fn, (a,), reference_in_float=False)

        # test dtype separately
        out = fn(a)
        assert out.dtype == torch.bfloat16

        out = torch.compile(fn)(a)
        assert out.dtype == torch.bfloat16

    def test_repeat_interleave(self):
        def fn(x):
            return (
                x.repeat_interleave(2),
                x.repeat_interleave(3, dim=0),
                x.repeat_interleave(x.size(1), dim=1),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    @config.patch(implicit_fallbacks=True)
    def test_repeat_interleave_2(self):
        def fn(x):
            return torch.ops.aten.repeat_interleave.Tensor(x, output_size=12)

        self.common(
            fn,
            (torch.tensor([2, 4, 6]),),
        )

    @config.patch(fallback_random=True)
    def test_randn_with_dtype_and_device(self):
        if self.device == GPU_TYPE:
            raise unittest.SkipTest("only support cpu randn_with_dtype_and_device test")

        def fn(vectors):
            rotations_shape = (12, vectors.shape[-1], 1, 64)
            random_rotations = torch.randn(
                rotations_shape, device=vectors.device, dtype=vectors.dtype
            )
            random_rotations += 1
            return random_rotations

        self.common(
            fn,
            (torch.randn([4, 12, 2, 64]),),
        )

    def test_embedding(self):
        m = torch.nn.Sequential(
            torch.nn.Embedding(10, 4, padding_idx=0),
            torch.nn.ReLU(),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randint(10, [2, 8]),),
        )

    def test_embedding_sparse(self):
        # Fix https://github.com/pytorch/pytorch/issues/150656
        def fn(weight, indices):
            return F.embedding(indices, weight, sparse=True)

        indices = torch.randint(10, (2, 3))
        weight = torch.randn(10, 3, requires_grad=True)

        self.common(
            fn,
            (weight, indices),
        )

    def test_mean(self):
        def fn(x):
            return (
                x.mean(),
                x.mean(-1),
                torch.mean(x, -2, keepdim=True),
                x.mean([0, 1]),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    @parametrize("tile_reduction", (False, True))
    def test_var_mean(self, tile_reduction: bool):
        def fn(x):
            return (
                *torch.var_mean(x, -1),
                *torch.var_mean(x, [1, 3]),
            )

        with config.patch(
            {
                "triton.prefer_nd_tiling": tile_reduction,
                "triton.tile_reductions": tile_reduction,
            }
        ):
            self.common(
                fn,
                (torch.randn([1, 2, 4, 8]),),
            )

    def test_var_mean_div_by(self):
        def fn(x):
            var, mean = torch.var_mean(x, dim=2, keepdim=True)
            return x / var, var, mean

        self.common(fn, (torch.rand([1, 17, 2048]),))

    def test_var_correction(self):
        def fn(x):
            dim = -1
            return (
                torch.var(x, dim=dim, correction=1.3),
                torch.var(x, dim=dim, correction=3),
                torch.var(x, dim=dim, correction=10),
            )

        self.common(fn, (torch.randn([2, 8]),))
        # Unrolled reduction
        self.common(fn, (torch.randn([2, 4]),))

    @config.patch(pick_loop_orders=True)
    def test_transposed_propagates(self):
        @torch.compile(backend="inductor", fullgraph=True)
        def fn(x, y):
            return x + y

        a = torch.randn(1, 4, 4, 4, device=self.device).permute(0, 2, 3, 1)
        b = torch.randn(4, 4, 4, device=self.device).permute(1, 2, 0)
        c = fn(a, b)
        self.assertEqual(a.stride(), c.stride())
        self.assertEqual(c.stride()[2], 1)

    @skip_if_gpu_halide
    def test_std(self):
        def fn(x):
            return (
                torch.var(x, True),
                torch.var(x, False),
                torch.var(x, -1, True),
                torch.var(x, -1, False),
                torch.std(x, False),
                torch.std(x, [0, 1], True),
                torch.std(x, [0, 1], False),
                torch.std(x, -2, True, keepdim=True),
            )

        self.common(
            fn,
            (torch.randn([2, 4, 4, 8]),),
        )

    def test_embedding_bag(self):
        def fn(w, i, o):
            return aten._embedding_bag(w, i, o, False, 0, False, None)

        self.common(
            fn,
            (torch.randn([10, 4]), torch.randint(10, [8]), torch.tensor([0, 2, 6])),
        )

    def test_batch_norm_2d(self):
        m = torch.nn.Sequential(
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
        )
        m.eval()
        self.common(m, (torch.randn([2, 10, 8, 8]),), check_lowp=False)
        self.common(
            m,
            (torch.randn([3, 10, 16, 16]),),
            check_lowp=False,  # too painful to match types of bn model
        )

    # From yolov3
    @with_tf32_off
    def test_batch_norm_2d_2(self):
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_0 = torch.nn.Conv2d(
                    64,
                    128,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
                self.self_1 = torch.nn.BatchNorm2d(
                    128,
                    eps=0.0001,
                    momentum=0.03,
                    affine=True,
                    track_running_stats=True,
                )
                self.self_2 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

            def forward(self, l_input_: torch.Tensor):
                self_0 = self.self_0(l_input_)
                self_1 = self.self_1(self_0)
                self_2 = self.self_2(self_1)
                return (self_2,)

        inp = torch.randn((4, 64, 192, 256), dtype=torch.float32, device=GPU_TYPE)
        mod = Repro().to(device=GPU_TYPE)
        o1 = mod(inp)
        o2 = torch.compile(mod)(inp)
        self.assertEqual(o1, o2, rtol=1e-3, atol=1e-3)

    @patch.object(config.trace, "enabled", True)
    def test_layer_norm(self):
        m = torch.nn.Sequential(
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
        )
        m.eval()
        with torch.no_grad():
            self.common(m, (torch.randn([16, 32]),), check_lowp=False)
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    @torch._functorch.config.patch("donated_buffer", True)
    def test_matmul_layer_norm(self):
        batch_size = 32
        seq_length = 50
        hidden_size = 256

        inp = torch.randn(
            batch_size,
            seq_length,
            hidden_size,
            requires_grad=True,
            device=self.device,
        )
        weight = torch.randn(
            hidden_size, hidden_size, requires_grad=True, device=self.device
        )

        layer_norm = torch.nn.LayerNorm(hidden_size, device=self.device)

        def foo(inp, weight):
            matmul_output = inp @ weight
            final_output = layer_norm(matmul_output)
            return final_output

        self.common(foo, (inp, weight), check_lowp=False)

    def test_transpose_add(self):
        def fn(a, b):
            return a.t() + b

        self.common(
            fn, (torch.randn([16, 32]), torch.randn([32, 16])), check_lowp=False
        )
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    @patch.object(config.triton, "persistent_reductions", True)
    def test_softmax_one_kernel_persist(self):
        def fn(x):
            dim = 1
            x_max = torch.amax(x, dim, keepdim=True)
            unnormalized = torch.exp(x - x_max)
            result = unnormalized / torch.sum(unnormalized, dim, keepdim=True)
            return result

        self.common(fn, (torch.randn([16, 32]),), check_lowp=False)
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    @patch.object(config.triton, "persistent_reductions", False)
    def test_softmax_one_kernel_loop(self):
        def fn(x):
            x_max = torch.amax(x, 1, keepdim=True)
            unnormalized = torch.exp(x - x_max)
            result = unnormalized / torch.sum(unnormalized, 1, keepdim=True)
            return result

        self.common(fn, (torch.randn([16, 32]),), check_lowp=False)
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    def test_complex_fallback(self):
        def fn(x):
            return x * x + 10

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]).to(dtype=torch.complex64),),
        )
        assertGeneratedKernelCountEqual(self, 0)

        class ToComplex(nn.Module):
            def forward(self, x):
                return (x + x + 12).to(torch.complex64)

        self.common(ToComplex(), (torch.rand([1, 2, 4, 8]),), check_lowp=False)

        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    def test_complex_from_real_imag(self):
        def fn(x, y):
            return aten.complex.default(x, y)

        a = torch.randn([5, 3]).permute(1, 0)

        self.common(
            fn,
            (a, a),
            exact_stride=True,
            reference_in_float=False,
        )

    @skipIfMPS
    def test_linalg_eig_stride_consistency(self):
        def fn(x):
            eigenvals, eigenvecs = torch.linalg.eig(x)
            return eigenvecs

        x = torch.randn(5, 5, device=self.device, dtype=torch.float32)

        self.common(
            fn,
            [x],
            exact_stride=True,
            exact_dtype=True,
            check_lowp=False,
        )

    def test_view_as_complex(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, view_2):
                clone = torch.ops.aten.clone.default(
                    view_2, memory_format=torch.contiguous_format
                )
                view_2 = None
                view_as_complex = torch.ops.aten.view_as_complex.default(clone)
                clone = None
                return (view_as_complex,)

        inp = torch.empty_strided((128, 64, 12, 32, 2), (1, 98304, 8192, 256, 128)).to(
            self.device
        )
        mod = Repro()

        o1 = mod(inp)
        o2 = torch.compile(mod)(inp)

        self.assertEqual(o1, o2)

    def test_view_as_real(self):
        def fn(x):
            y = torch.view_as_real(x)
            return y + 1

        x = torch.randn(4, dtype=torch.complex64)

        self.common(fn, (x,))

    def test_polar(self):
        def fn(dist, angle):
            return torch.polar(dist, angle)

        dtype = torch.float64 if self.device != "mps" else torch.float32
        inp = (
            torch.tensor([1, 2], dtype=dtype),
            torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=dtype),
        )
        self.common(fn, (*inp,), reference_in_float=self.device != "mps")

    @skip_if_gpu_halide  # incorrect result on CUDA
    def test_cauchy(self):
        def fn(x, y):
            return torch.sum(1 / (torch.unsqueeze(x, -1) - y))

        self.common(
            fn,
            (
                torch.randn(32),
                torch.randn(32),
            ),
            # Absolute difference: 0.0003662109375 (up to 0.0001 allowed)
            # Relative difference: 1.8804297408767818e-05 (up to 1e-05 allowed)
            atol=5 * 1e-4,
            rtol=5 * 1e-5,
            check_lowp=False,
        )
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 1)

    @skip_if_gpu_halide  # misaligned address error
    def test_fusing_write_into_disjoint_read(self):
        def test_flip(a):
            return a.copy_(torch.flip(a, (0,)))

        self.common(test_flip, (torch.rand([20]),))

        assertGeneratedKernelCountEqual(self, 2)

        # issue only manifests on cuda with large tensors
        if self.device != "cpu":

            def f(a):
                a[:, 20:40] = a[:, 20:40] + 1
                a[:, 2:900025] = a[:, 1:900024] + 2

            a = torch.rand((1, 1000000), device=self.device)
            self.common(f, (a,))

    def test_inplace_flip(self):
        def f(x, y):
            x.copy_(x.flip(1))
            y = y.sum(dim=1, keepdim=True) + y
            return x + y

        x = torch.randn(20, 1024 * 1024)
        y = torch.randn(20, 1024 * 1024)
        self.common(f, (x, y), atol=1e-3, rtol=1e-3)

    def test_gather_scatter(self):
        def fn(node_feat, edge_index):
            src_node_feat = node_feat[edge_index[0]]
            dst_node_feat = node_feat[edge_index[1]]
            edge_feat = src_node_feat - dst_node_feat + 1
            new_node_feat = torch.zeros_like(node_feat)
            new_node_feat.scatter_add_(
                0, edge_index[1].unsqueeze(-1).expand_as(edge_feat), edge_feat
            )
            return new_node_feat

        num_nodes = 16
        num_features = 32
        node_feat = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, size=(2, num_nodes * 5))
        self.common(
            fn,
            (
                node_feat,
                edge_index,
            ),
            check_lowp=False,
        )
        if self.device != "cpu":
            assertGeneratedKernelCountEqual(self, 2)

    @config.patch(max_fusion_size=1)
    def test_no_mega_fusion_during_lowering(self):
        n = 50

        def fn(*args):
            x = args[0]
            for i in range(n):
                x = torch.add(x, args[i])
            return x

        self.common(
            fn,
            [torch.randn(64) for _ in range(n)],
            check_lowp=False,
        )
        print("-->", torch._inductor.metrics.generated_kernel_count)
        if self.device != "cpu":
            self.assertTrue(torch._inductor.metrics.generated_kernel_count > 1)

    def test_move_arange(self):
        def fn(x):
            return torch.arange(len(x), device="cpu").to(x.device) + x

        self.common(fn, (torch.randn([32]),), check_lowp=False)
        # if we have a copy there will be more than 1 kernel
        assertGeneratedKernelCountEqual(self, 1)

    def test_leaky_relu(self):
        def fn(x):
            return aten.leaky_relu(x, 0.2) + 2, aten.leaky_relu(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_gelu(self):
        def fn(x):
            return aten.gelu(x) + 2, aten.gelu(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_clone(self):
        def fn(x):
            return aten.clone(x) + 2, aten.clone(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_masked_fill(self):
        def fn(mask, value):
            return aten.masked_fill(value, mask, -10000.0) + 2, aten.masked_fill(
                value / 2.0, torch.logical_not(mask), 667
            )

        self.common(
            fn,
            (
                torch.randint(0, 1, [1, 16], dtype=torch.bool),
                torch.randn([16, 16]),
            ),
        )

    def test_masked_fill_promotion(self):
        def fn(mask, value):
            return aten.masked_fill(value, mask, torch.tensor(3.5))

        opt_fn = torch.compile(fn, backend="inductor")
        for inp in (
            torch.randn(
                [16, 16],
                dtype=torch.float16 if self.device == GPU_TYPE else torch.float32,
                device=self.device,
            ),
            torch.randint(16, (16, 16), device=self.device),
        ):
            inputs = (
                torch.randint(0, 1, [1, 16], dtype=torch.bool, device=self.device),
                inp,
            )
            self.assertEqual(fn(*inputs), opt_fn(*inputs))

    @xfail_if_mps  # 'NullHandler' object has no attribute 'wrapper_code'
    def test_masked_scatter(self):
        def fn(value, mask, source):
            return torch.masked_scatter(value, mask, source)

        value = make_tensor(10, 10, dtype=torch.float32, device=self.device)
        mask = make_tensor(10, 10, dtype=torch.bool, device=self.device)
        source = make_tensor(
            mask.count_nonzero(), dtype=torch.float32, device=self.device
        )

        self.common(fn, (value, mask, source))

    def test_fill1(self):
        def fn(x):
            tmp = torch.ones_like(x)
            return tmp, aten.fill.Scalar(tmp, 2)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_fill2(self):
        def fn(x):
            tmp = torch.ones_like(x)
            return tmp, aten.fill.Tensor(tmp, torch.tensor(3.0))

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_pow1(self):
        def fn(x):
            return [aten.pow(x, e) for e in range(-8, 9)]

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    @xfail_if_triton_cpu
    def test_pow2(self):
        def fn(x):
            return aten.pow(1000, x), aten.pow(x, 1000)

        self.common(
            fn,
            (
                torch.randn(
                    [16, 16],
                    dtype=torch.float32,
                ),
            ),
            # Mismatched elements: 9 / 256 (3.5%)
            # Greatest absolute difference: 2.491354329061828e+28 at index (6, 6) (up to 1e-05 allowed)
            # Greatest relative difference: 2.9793410720160818e-05 at index (4, 5) (up to 1.3e-06 allowed)
            atol=1e-5,
            rtol=3e-05,
        )

    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8318
    @config.patch("halide.scheduler_cuda", "Li2018")
    def test_pow3(self):
        # power of 0.5 is special-cased, arbitrary power would still produce triton codegen error
        def fn(x):
            z = torch.tensor(0.123, device=self.device)
            w = z + x
            return torch.pow(w, 0.5)

        opt = torch.compile(fn, backend="inductor")
        input = torch.rand((), device=self.device)
        self.assertTrue(same(opt(input), fn(input)))

    def test_pow_int(self):
        def fn(x, y):
            return torch.pow(x, 0x57), torch.pow(x, y)

        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            intmax = torch.iinfo(dtype).max
            make_arg = functools.partial(
                make_tensor, dtype=dtype, device=self.device, requires_grad=False
            )
            self.common(
                fn,
                (
                    make_arg(16, 16),
                    make_arg(16, 16, high=intmax),
                ),
            )

    @xfail_if_triton_cpu
    def test_pow_symfloat(self):
        def fn(x):
            r = math.sqrt(x.size(0))
            r = r**10
            return x * r

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)
        x = torch.randn([16, 16], device=self.device)
        self.assertEqual(cfn(x), fn(x))

    def test_glu(self):
        def fn(x):
            return aten.glu(x, -1), aten.glu(x, 1), aten.glu(x, 2)

        self.common(
            fn,
            (torch.randn([8, 16, 8, 8]),),
        )

    def test_unsigned_constant_tensors(self):
        def fn(x):
            c = torch.tensor(7, dtype=torch.uint8)
            return c + x, torch.neg(c), torch.neg(c) + x

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    # Disable size_asserts for this test due to https://github.com/pytorch/pytorch/issues/145963
    @config.patch(size_asserts=os.environ.get("TORCHINDUCTOR_SIZE_ASSERTS") == "1")
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_unbacked_refinement(self):
        def fn(x):
            z = x.nonzero()
            torch._check(z.size(0) == 4)
            return z + 3

        self.common(
            fn,
            (torch.tensor([0, 1, 3, 4, 2, 0, 0]),),
        )

        with self.assertRaises(RuntimeError):
            torch.compile(fn)(torch.tensor([0, 0, 0, 0]))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_floordiv_simplify(self):
        def fn(x, y):
            z = y.item()
            torch._check(z // 2 == 3)
            return x + x.new_ones(z)

        self.common(
            fn,
            (
                torch.randn(6),
                torch.tensor([6]),
            ),
        )

        self.common(
            fn,
            (
                torch.randn(7),
                torch.tensor([7]),
            ),
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_floordiv_simplify_errors(self):
        def fn(x, y):
            z = y.item()
            torch._check(z // 2 == 3)
            return x + x.new_zeros(z)

        # This is a little suboptimal: we actually fail /in the compiler/ but
        # not in a way that causes Dynamo to graph break
        with self.assertRaises(RuntimeError):
            torch.compile(fn)(torch.randn(8), torch.tensor(8))

    def test_cat(self):
        tgt_dtype = torch.double if self.device != "mps" else torch.half

        def fn(a):
            tmp = a * 2
            return (
                torch.cat((a, a[:, :4] + 1, a + 2), -1),
                torch.cat((tmp, tmp), 0),
                torch.cat((tmp, tmp.to(dtype=tgt_dtype)), 0),
            )

        self.common(
            fn,
            (torch.randn([8, 16]),),
        )
        self.common(
            fn,
            (torch.randn([1, 3, 3, 16]).to(memory_format=torch.channels_last),),
        )

    def test_cat_uint8(self):
        def fn(x):
            batch_shape = x.shape[:1]
            out = torch.cat([x.new_zeros(1).expand(batch_shape + (1,)), x], dim=-1)
            return out

        self.common(
            fn,
            (torch.randint(0, 256, size=(3, 255), dtype=torch.uint8),),
        )

    def test_cat_empty(self):
        def fn_2(*tensors):
            return torch.cat(tensors)

        self.common(
            fn_2,
            (
                torch.randn([1, 3, 3, 16]),
                torch.ones([0]),
            ),
        )
        self.common(
            fn_2,
            (
                torch.randn([1, 3, 3, 16]),
                torch.ones([0]),
                torch.randn([1, 3, 3, 16]),
            ),
        )
        self.common(
            fn_2,
            (
                torch.ones([0]),
                torch.randn([1, 3, 3, 16]),
            ),
        )

    def test_cat_empty_index(self):
        def fn(out, x):
            return torch.cat([out[0], x], dim=0)

        self.common(fn, (torch.randn(1, 0, 64), torch.randn(128, 64)))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked_legacy_empty(self):
        def fn(x, y):
            z = y.item()
            return torch.cat([x, x.new_ones(z)])

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected 2-D tensors, but got 1-D for tensor number 1 in the list",
        ):
            self.common(
                fn,
                (
                    torch.randn([2, 3]),
                    torch.tensor([0]),
                ),
            )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked_empty_1d(self):
        def fn(x, y):
            z = y.item()
            return torch.cat([x, x.new_ones(z)])

        self.common(
            fn,
            (
                torch.randn([2]),
                torch.tensor([0]),
            ),
        )

        self.common(
            fn,
            (
                torch.randn([2]),
                torch.tensor([3]),
            ),
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked_2d(self):
        def fn(x, y):
            z = y.item()
            return torch.cat([x, x.new_ones(z, x.shape[1])])

        self.common(
            fn,
            (
                torch.randn([2, 3]),
                torch.tensor([0]),
            ),
        )

        self.common(
            fn,
            (
                torch.randn([2, 3]),
                torch.tensor([4]),
            ),
        )

    def test_cat_negative_dim(self):
        def fn(*tensors):
            return torch.cat(tensors, dim=-1)

        self.common(
            fn,
            (
                torch.randn([2, 3]),
                torch.randn([2, 4]),
            ),
        )

        self.common(
            fn,
            (
                torch.randn([2, 3]),
                torch.randn([0]),
                torch.randn([2, 4]),
            ),
        )

        self.common(
            fn,
            (
                torch.randn([0]),
                torch.randn([2, 3]),
                torch.randn([2, 4]),
            ),
        )

    @expectedFailureCodegenDynamic
    def test_cat_single_empty(self):
        # fails dynamic check for 'has a dynamic dimension'
        def fn_2(*tensors):
            return torch.cat(tensors)

        self.common(
            fn_2,
            (torch.ones([0]),),
        )

    def test_cat_upcasting(self):
        def fn(arg4_1, slice_7):
            cat_1 = aten.cat.default([arg4_1, slice_7], 1)
            return (cat_1,)

        self.common(
            fn,
            (
                torch.randn([8, 16], dtype=torch.float32),
                torch.randn([8, 20], dtype=torch.float16),
            ),
        )

    def test_cat_extern_kernel(self):
        def fn(x1, x2, x3, x4):
            x = torch.mm(x2, x3)
            s = torch.narrow(x, 1, 0, 100)
            x = torch.mm(s, x4)
            c = torch.cat((x, x1), 1)
            return (c,)

        if self.device == "xpu":
            atol = 3e-4
            rtol = 1e-4
        else:
            atol = 5e-4
            rtol = 3e-4

        # MPS has correctness problem before MacOS15
        with (
            contextlib.nullcontext()
            if self.device != "mps" or MACOS_VERSION >= 15.0
            else self.assertRaises(AssertionError)
        ):
            self.common(
                fn,
                (
                    torch.randn(256, 256),
                    torch.randn(256, 1024),
                    torch.randn(1024, 1600),
                    torch.randn(100, 256),
                ),
                atol=atol,
                rtol=rtol,
                check_lowp=False,  # accuracy issues with relatively large matmuls
            )

    @skip_if_gpu_halide
    # Constant folding was explicitly turned off due to issue #108388
    # Turn it back on for test
    @unittest.skipIf(config.triton.native_matmul, "native matmul has better precision")
    @torch._inductor.config.patch(
        joint_graph_constant_folding=True,
        # Numerical accuracy failure for triton fp16
        max_autotune_gemm_backends="ATEN",
    )
    def test_remove_no_ops(self):
        def matmul_with_op(x, y, fn):
            return fn(x @ y)

        foo_opt = torch.compile(matmul_with_op)

        # test no-op
        fns = (
            lambda x: x + torch.zeros([256, 256], dtype=torch.float32, device=x.device),  # noqa: E731
            lambda x: x - torch.zeros([256, 256], dtype=torch.float32, device=x.device),  # noqa: E731
            lambda x: x * torch.ones([256, 256], dtype=torch.float32, device=x.device),  # noqa: E731
            lambda x: x / torch.ones([256, 256], dtype=torch.float32, device=x.device),  # noqa: E731
        )

        inps = [torch.rand([256, 256], device=self.device) for _ in range(2)]

        for fn in fns:
            out, source_codes = run_and_get_code(foo_opt, inps[0], inps[1], fn)
            self.assertEqual(out, matmul_with_op(inps[0], inps[1], fn))

            atol, rtol = None, None
            if self.device == "cpu":
                FileCheck().check_not("cpp_fused").run(source_codes[0])
            else:
                FileCheck().check_not("triton.jit").run(source_codes[0])

        # test dtype conversion
        for lowp_dtype in [torch.float16, torch.bfloat16]:
            if not self.is_dtype_supported(lowp_dtype):
                continue
            inps = [
                torch.rand([256, 256], device=self.device, dtype=lowp_dtype)
                for _ in range(2)
            ]
            for fn in fns:
                out, source_codes = run_and_get_code(foo_opt, inps[0], inps[1], fn)
                self.assertEqual(
                    out, matmul_with_op(inps[0], inps[1], fn), atol=atol, rtol=rtol
                )

            # test broadcasted shape bail
            fn = lambda x: x + torch.zeros(  # noqa: E731
                [256, 256, 256], dtype=lowp_dtype, device=self.device
            )
            out, source_codes = run_and_get_code(foo_opt, inps[0], inps[1], fn)
            self.assertEqual(
                out, matmul_with_op(inps[0], inps[1], fn), atol=atol, rtol=rtol
            )

    def test_remove_noop_copy(self):
        def fn(x, y):
            x = x.cos()
            a = x.copy_(y)
            return a.sin()

        self.common(fn, (torch.randn(8, 8), torch.randn(8)))

        def fn2(a, b):
            abs_max = torch.abs(a).max()
            b[0] = abs_max.to(a.dtype)
            return b

        self.common(
            fn2,
            (
                torch.randn(8, 8, dtype=torch.float16),
                torch.randn(8, dtype=torch.float32),
            ),
        )

    def test_remove_noop_clone(self):
        def fn(x):
            y = x.clone().reshape(-1, 4)
            y[:, [2, 0]] = y[:, [0, 2]]
            return y + x

        self.common(fn, (torch.randn(2, 4),))

    def test_remove_noop_slice(self):
        def f(x):
            x = x + 1
            size = x.shape[-1]
            y = torch.ops.aten.slice(x, -1, 0, size)  # noop
            return y + 1

        f = torch.compile(f)

        x = torch.ones((2, 3, 2), device=self.device)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.mark_dynamic(x, 2)

        post_grad_graph = get_post_grad_graph(f, (x,))
        expected_graph = f"""\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "Sym(s27)", arg2_1: "Sym(s53)", arg3_1: "f32[s77, s27, s53][s27*s53, s53, 1]{str(x.device)}"):
        add: "f32[s77, s27, s53][s27*s53, s53, 1]{str(x.device)}" = torch.ops.aten.add.Tensor(arg3_1, 1);  arg3_1 = None
        add_9: "f32[s77, s27, s53][s27*s53, s53, 1]{str(x.device)}" = torch.ops.aten.add.Tensor(add, 1);  add = None
        return (add_9,)"""  # noqa: B950
        self.assertExpectedInline(
            post_grad_graph,
            expected_graph,
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_remove_noop_slice1(self):
        def f(x):
            x = x + 1
            y = torch.ops.aten.slice(x, -1, 0, -1)  # not a noop
            return y + 1

        f = torch.compile(f)
        x = torch.ones((2, 3, 2), device=self.device)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        post_grad_graph = get_post_grad_graph(f, (x,))
        expected_graph = f"""\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "Sym(s27)", arg2_1: "f32[s77, s27, 2][2*s27, 2, 1]{str(x.device)}"):
        add: "f32[s77, s27, 2][2*s27, 2, 1]{str(x.device)}" = torch.ops.aten.add.Tensor(arg2_1, 1);  arg2_1 = None
        slice_1: "f32[s77, s27, 1][2*s27, 2, 1]{str(x.device)}" = torch.ops.aten.slice.Tensor(add, -1, 0, -1);  add = None
        add_9: "f32[s77, s27, 1][s27, 1, 1]{str(x.device)}" = torch.ops.aten.add.Tensor(slice_1, 1);  slice_1 = None
        return (add_9,)"""  # noqa: B950
        self.assertExpectedInline(
            post_grad_graph,
            expected_graph,
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_remove_noop_slice_scatter(self):
        def f(x):
            x = x + 1
            y = torch.empty_like(x)
            size = x.shape[-1]
            out = torch.ops.aten.slice_scatter(y, x, -1, 0, size)  # noop
            return out + 1

        f = torch.compile(f)

        x = torch.ones((2, 3, 2), device=self.device)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.mark_dynamic(x, 2)

        post_grad_graph = get_post_grad_graph(f, (x,))
        expected_graph = f"""\
def forward(self, arg0_1: "Sym(s77)", arg1_1: "Sym(s27)", arg2_1: "Sym(s53)", arg3_1: "f32[s77, s27, s53][s27*s53, s53, 1]{str(x.device)}"):
        empty: "f32[s77, s27, s53][s27*s53, s53, 1]{str(x.device)}" = torch.ops.aten.empty.memory_format([arg0_1, arg1_1, arg2_1], dtype = torch.float32, layout = torch.strided, device = {repr(x.device)}, pin_memory = False);  arg0_1 = arg1_1 = arg2_1 = None
        permute: "f32[s77, s27, s53][s27*s53, s53, 1]{str(x.device)}" = torch.ops.aten.permute.default(empty, [0, 1, 2]);  empty = permute = None
        add: "f32[s77, s27, s53][s27*s53, s53, 1]{str(x.device)}" = torch.ops.aten.add.Tensor(arg3_1, 1);  arg3_1 = None
        add_13: "f32[s77, s27, s53][s27*s53, s53, 1]{str(x.device)}" = torch.ops.aten.add.Tensor(add, 1);  add = None
        return (add_13,)"""  # noqa: B950
        self.assertExpectedInline(
            post_grad_graph,
            expected_graph,
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_cat_of_loops_and_extern_kernel(self):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    64,
                    5,
                    1,
                    **kwargs,
                )
                self.max_pool2d = torch.nn.MaxPool2d(2)

            def forward(self, x, y):
                x1 = self.conv(x)
                y1 = self.max_pool2d(y)
                return torch.cat([x1, y1], 1)

        mod = M()
        opt_mod = torch.compile(mod, backend="inductor")
        memory_format = torch.channels_last
        inputs = (
            torch.randn([1, 64, 16, 16]).to(memory_format=memory_format),
            torch.randn([1, 64, 32, 32]).to(memory_format=memory_format),
        )
        y = mod(*inputs)
        opt_y = opt_mod(*inputs)
        self.assertEqual(y, opt_y)
        self.assertEqual(y.stride(), opt_y.stride())

    def test_cat_inplace(self):
        def fn(x):
            rt = torch.cat([x])
            v = x.sin_()
            return rt

        # can't use self.common because input is modified inplace
        inp = torch.ones(2)
        opt_fn = torch.compile(fn)
        res = opt_fn(inp.clone())
        expected = fn(inp.clone())
        self.assertEqual(res, expected)

    def test_stack(self):
        def fn(a, b):
            return torch.stack(
                [
                    a.expand(12, 16),
                    b.expand(12, 16),
                ],
                2,
            )

        self.common(fn, (torch.randn([1, 16]), torch.randn([12, 1])))

    def test_hardtanh(self):
        def fn(x):
            return F.hardtanh(x), F.hardtanh(x + 1), F.hardtanh(x - 1)

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_hardsigmoid(self):
        def fn(x):
            return F.hardsigmoid(x), F.hardsigmoid(x + 3), F.hardsigmoid(x - 3)

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_hardswish(self):
        def fn(x):
            return F.hardswish(x), F.hardswish(x + 3), F.hardswish(x - 3)

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_rsqrt(self):
        def fn(x):
            return torch.rsqrt(x), torch.rsqrt(x + 1) - 2

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_expm1(self):
        def fn(x):
            return torch.expm1(x), torch.expm1(x) * 2

        for dtype in (torch.float16, torch.float, torch.double, torch.int, torch.int64):
            if not self.is_dtype_supported(dtype):
                continue

            self.common(
                fn,
                (torch.randn([64]).to(dtype=dtype),),
            )
            self.common(
                fn,
                (torch.arange(-1e-5, 1e-5, 1e-7).to(dtype=dtype),),
            )

    @xfail_if_mps_unimplemented
    def test_adaptive_pool_errors_with_long(self):
        class Model(torch.nn.Module):
            def __init__(self, pool_operator):
                super().__init__()
                self.pool = pool_operator

            def forward(self, x):
                x = torch.argmax(x, dim=1)
                x = self.pool(x)
                return x

        for dim in (1, 2, 3):
            op_inst = eval(f"torch.nn.AdaptiveMaxPool{dim}d(5)")
            model = Model(op_inst).to(self.device)
            x = torch.randn([1] * (dim + 2)).to(self.device)
            model = torch.compile(model, fullgraph=True)
            with self.assertRaisesRegex(
                RuntimeError, r".*(not implemented|aoti_torch_).*"
            ):
                model(x)

    @xfail_if_mps_unimplemented
    def test_adaptive_avg_pool_errors_with_long(self):
        class Model(torch.nn.Module):
            def __init__(self, pool_operator):
                super().__init__()
                self.pool = pool_operator

            def forward(self, x):
                x = torch.argmax(x, dim=1)
                x = self.pool(x)
                return x

        for dim in (1, 2, 3):
            op_inst = eval(f"torch.nn.AdaptiveAvgPool{dim}d(5)")
            model = Model(op_inst).to(self.device)
            x = torch.randn([1] * (dim + 2)).to(self.device)
            model = torch.compile(model, fullgraph=True)
            with self.assertRaisesRegex(
                RuntimeError, r".*(not implemented|aoti_torch_).*"
            ):
                model(x)

    @torch._dynamo.config.patch(recompile_limit=12)
    def test_avg_pool_errors_with_uint(self):
        for dim in (1, 2, 3):
            for dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
                x = torch.randn([2] * (dim + 2)).to(dtype)
                op = eval(f"torch.nn.functional.avg_pool{dim}d")
                c_op = torch.compile(op)
                with self.assertRaisesRegex(
                    RuntimeError, r".*(not implemented|aoti_torch_).*"
                ):
                    c_op(x, kernel_size=2, stride=2)

    def test_replication_pad_errors_with_bool(self):
        for dim in (1, 2, 3):

            def fn(x):
                x = torch.signbit(x)
                x = eval(f"nn.ReplicationPad{dim}d(padding=1)")(x)
                return x

            c_fn = torch.compile(fn)
            x = torch.randn([1] * (dim + 2))
            with self.assertRaisesRegex(
                RuntimeError, r".*(not implemented|aoti_torch_).*"
            ):
                c_fn(x)

    def test_log1p(self):
        def fn(x):
            return torch.log1p(x), torch.log1p(x) * 2

        for dtype in (torch.float16, torch.float, torch.double, torch.int, torch.int64):
            if not self.is_dtype_supported(dtype):
                continue

            self.common(
                fn,
                (torch.randn([64]).to(dtype=dtype),),
            )
            self.common(
                fn,
                (torch.arange(-1e-5, 1e-5, 1e-7).to(dtype=dtype),),
            )

    @config.patch(force_disable_caches=True)
    @skip_if_cpp_wrapper("run_and_get_kernels issue")
    def test_deterministic_codegen(self):
        if "cpu" in str(self.device) and config.is_fbcode():
            raise unittest.SkipTest("cpp packaging is wacky in fbcode")

        @torch.compile(fullgraph=True)
        def a(x):
            return x.cos().sin().softmax(-1)

        @torch.compile(fullgraph=True)
        def b(x):
            return x.sin().cos().softmax(-1)

        @torch.compile(fullgraph=True)
        def c(x):
            return x.cos().sin().softmax(-1)

        x = torch.randn(16, 256, device=self.device)
        _, (coda_a0,) = _run_and_get_stripped_kernels(a, x)
        _, (coda_b0,) = _run_and_get_stripped_kernels(b, x)
        _, (coda_c0,) = _run_and_get_stripped_kernels(c, x)
        self.assertEqual(coda_a0, coda_c0)

        # compile in a different order
        torch.compiler.reset()
        _, (coda_c1,) = _run_and_get_stripped_kernels(c, x)
        _, (coda_a1,) = _run_and_get_stripped_kernels(a, x)
        _, (coda_b1,) = _run_and_get_stripped_kernels(b, x)
        self.assertEqual(coda_a0, coda_a1)
        self.assertEqual(coda_b0, coda_b1)
        self.assertEqual(coda_c0, coda_c1)

        # force a different CompileId
        torch.compiler.reset()
        CompileContext_init = CompileContext.__init__
        with patch.object(
            CompileContext,
            "__init__",
            lambda self, _: CompileContext_init(
                self, CompileId(frame_id=999, frame_compile_id=999)
            ),
        ):
            _, (coda_a2,) = _run_and_get_stripped_kernels(a, x)
            _, (coda_c2,) = _run_and_get_stripped_kernels(c, x)
            _, (coda_b2,) = _run_and_get_stripped_kernels(b, x)
        self.assertEqual(coda_a0, coda_a2)
        self.assertEqual(coda_b0, coda_b2)
        self.assertEqual(coda_c0, coda_c2)

    @config.patch(force_disable_caches=True)
    @skip_if_cpp_wrapper("run_and_get_kernels issue")
    def test_deterministic_codegen_on_graph_break(self):
        if "cpu" in str(self.device) and config.is_fbcode():
            raise unittest.SkipTest("cpp packaging is wacky in fbcode")

        def a(x):
            return x.cos().sin().softmax(-1)

        @torch.compile()
        def b(x):
            x = a(x)
            torch._dynamo.graph_break()
            x = a(x)
            return x

        x = torch.randn(16, 256, device=self.device)
        _, (code0, code1) = _run_and_get_stripped_kernels(b, x)
        self.assertEqual(code0, code1)

    @config.patch(
        force_disable_caches=True,
        # Test expects a single (fused) kernel to be generated
        max_autotune_gemm_backends="ATEN",
    )
    @skip_if_cpp_wrapper("run_and_get_kernels issue")
    @unittest.skipIf(config.triton.native_matmul, "matmul is now generated")
    def test_deterministic_codegen_with_suffix(self):
        if "cpu" in str(self.device) and config.is_fbcode():
            raise unittest.SkipTest("cpp packaging is wacky in fbcode")

        @torch.compile(fullgraph=True)
        def a(x):
            return x.cos().sin().softmax(-1)

        @torch.compile(fullgraph=True)
        def b(x, y):
            x = x.cos().sin().softmax(-1)
            x = torch.matmul(x, y)
            return x

        x = torch.randn(16, 256, device=self.device)
        y = torch.randn(256, 256, device=self.device)
        _, (code0,) = _run_and_get_stripped_kernels(a, x)
        _, (code1,) = _run_and_get_stripped_kernels(b, x, y)
        self.assertEqual(code0, code1)

    def test_flip(self):
        def fn(x):
            return torch.flip(x, (-1,)), torch.flip(x, (0, 2)) - 2

        self.common(
            fn,
            (torch.randn([1, 2, 6, 6]),),
        )

    def test_signbit(self):
        def fn(x):
            return torch.signbit(x), ~torch.signbit(-x) & 1

        self.common(
            fn,
            (torch.randn([1, 2, 6, 6]),),
        )

    def test_sign_dtype(self):
        def fn(x):
            y = torch.sign(x)
            return torch.tanh(y)

        self.common(fn, (torch.randn([1, 2, 6, 6]),))

    @xfail_if_triton_cpu
    def test_fmod(self):
        def fn(a, b):
            return torch.fmod(a, b), torch.fmod(3.0 * a, b) - 2.0

        shape = [1, 2, 6, 6]
        self.common(fn, (torch.randn(shape), torch.randn(shape)))

    @xfail_if_triton_cpu
    def test_fmod_zero_dim(self):
        def fn(a, b):
            return (torch.fmod(a, b),)

        self.common(
            fn,
            (
                make_tensor(10, device=self.device, dtype=torch.float32),
                make_tensor((), device=self.device, dtype=torch.float32),
            ),
        )
        self.common(
            fn,
            (
                make_tensor((), device=self.device, dtype=torch.float32),
                make_tensor(10, device=self.device, dtype=torch.float32),
            ),
        )

    @skip_if_halide  # log2 not implemented for halide
    def test_log2(self):
        def fn(x):
            return torch.log2(x), torch.log2(x + 1) - 2

        self.common(
            fn,
            (torch.randn([64]) + 10,),
        )

    def test_logsumexp(self):
        def fn(x):
            return torch.logsumexp(x, -1), torch.logsumexp(x, 0) - 2

        self.common(
            fn,
            (torch.randn([8, 8]) + 10,),
        )

    @skip_if_halide  # log2 not implemented for halide
    def test_log_fp64(self):
        def fn(x):
            return torch.log(x), torch.log2(x)

        _dtype = torch.float64
        ctx = (
            contextlib.nullcontext()
            if self.is_dtype_supported(_dtype)
            else self.assertRaises(TypeError)
        )
        with ctx:
            self.common(
                fn,
                (torch.randn([1024], dtype=_dtype) + 10,),
            )

    def test_bitwise(self):
        def fn(x, y):
            return (
                torch.bitwise_not(x),
                torch.bitwise_or(x, y),
                torch.bitwise_xor(x, y),
                torch.bitwise_and(x, y),
            )

        self.common(
            fn,
            (
                torch.randint(0, 2**30, [64], dtype=torch.int32),
                torch.randint(0, 2**30, [64], dtype=torch.int32),
            ),
        )

    def test_bitwise2(self):
        # again with bool types
        def fn(x, y):
            return (
                torch.bitwise_not(x),
                torch.bitwise_or(x, y),
                torch.bitwise_xor(x, y),
                torch.bitwise_and(x, y),
            )

        self.common(
            fn,
            (
                torch.randint(0, 2, (2, 20), dtype=torch.bool),
                torch.randint(0, 2, (2, 20), dtype=torch.bool),
            ),
        )

    def test_bitwise3(self):
        # Repro for https://github.com/pytorch/pytorch/issues/97968
        def fn(x, y):
            return (
                torch.max(torch.bitwise_and(x, y), y),
                torch.clamp_max(torch.bitwise_or(x, y), y),
                torch.clamp_min(torch.bitwise_xor(x, y), y),
            )

        self.common(
            fn,
            (
                torch.rand([5, 10, 1]).to(torch.int8),
                torch.rand([10, 1]).to(torch.int8),
            ),
        )

    def test_inf(self):
        def fn(a):
            return a + float("inf"), a + float("-inf"), a * -float("inf")

        self.common(fn, (torch.randn(8),))

    def test_remainder(self):
        def fn(a, b):
            return (
                torch.remainder(a, b),
                torch.remainder(a + 1, b - 1),
                torch.remainder(a - 1, b + 1),
            )

        self.common(fn, (torch.randn(64), torch.randn(64)))

    def test_zeros(self):
        def fn(a):
            return (
                a + 1,
                torch.zeros(
                    (1, 8, 64, 64),
                    dtype=torch.float32,
                    device=a.device,
                ),
                torch.zeros(
                    1,
                    8,
                    64,
                    64,
                    dtype=torch.float32,
                    device=a.device,
                ),
                torch.zeros(2, 3),
                a + torch.ones(8, device=a.device),
                torch.full((2, 3), 3.1416, device=a.device),
            )

        self.common(fn, (torch.randn(8),))

    def test_new_ones(self):
        def fn(a):
            return (
                aten.new_ones(
                    a, [], device=a.device, dtype=6, layout=0, pin_memory=False
                ),
                aten.new_zeros(
                    a, [], device=a.device, dtype=6, layout=0, pin_memory=False
                ),
            )

        self.common(fn, (torch.randn(8),))

    def test_full_like(self):
        def fn(a):
            return torch.full_like(a, 7.777) - 1

        self.common(fn, (torch.randn(8),))

    def test_full_like_transposed(self):
        def fn(a):
            return torch.full_like(a, 3)

        self.common(fn, (torch.randn(4, 5, 6).transpose(1, -1),), exact_stride=True)

    def test_full_like_sliced(self):
        def fn(a):
            return torch.full_like(a, 3)

        self.common(fn, (torch.rand(3, 4)[:, ::2],), exact_stride=True)

    def test_full_truncation(self):
        def fn(a):
            return a + torch.full_like(a, 7.777)

        for dtype in all_types():
            ctx = (
                contextlib.nullcontext()
                if self.is_dtype_supported(dtype)
                else self.assertRaises(TypeError)
            )
            with ctx:
                self.common(
                    fn,
                    (make_tensor(8, dtype=dtype, device=self.device),),
                    check_lowp=False,
                )

    def test_full_boolean(self):
        def fn(n):
            x = torch.full((1,), n >= 1024, device=self.device)
            return x, x + 1

        self.common(fn, (1024,))
        self.common(fn, (1023,))

    def test_index1(self):
        def fn(a, b, c):
            return aten.index(a, [b, c])

        self.common(
            fn,
            (
                torch.randn(8, 8, 12),
                torch.tensor([0, 0, 2, 2], dtype=torch.int64),
                torch.tensor([3, 4, 4, 3], dtype=torch.int64),
            ),
        )
        self.common(
            fn,
            (
                torch.randn(8, 8, 12),
                torch.tensor([[0, 0, 2, 2]], dtype=torch.int64),
                torch.tensor([[3], [4], [4], [3]], dtype=torch.int64),
            ),
        )

    def test_index2(self):
        def fn(a, b):
            return (
                aten.index(a, [b]),
                aten.index(a, [None, b]),
            )

        self.common(
            fn,
            (
                torch.randn(8, 8, 8),
                torch.tensor([[0, 0, 2, 2]], dtype=torch.int64),
            ),
        )

    def test_index3(self):
        def fn(x, ia, ib):
            return (x[:, ia, None, ib, 0],)

        self.common(
            fn,
            (
                torch.randn(3, 4, 4, 4, 3),
                torch.tensor([0, 2, 1], dtype=torch.int64),
                torch.tensor([0, 2, 1], dtype=torch.int64),
            ),
        )

    def test_output_strides(self):
        def fn(x):
            y = x.permute(0, 2, 3, 1).contiguous()
            torch._dynamo.graph_break()
            return y.view(-1, 4)

        inp = torch.rand([4, 4, 4, 4], device=self.device)
        fn_opt = torch.compile(fn, backend="inductor")

        self.assertEqual(fn(inp), fn_opt(inp))
        self.assertEqual(fn(inp).stride(), fn_opt(inp).stride())

        # no redundant copy
        def foo(x):
            return x[0:2:2].T[3:].squeeze(0)

        foo_opt = torch.compile(foo, backend="inductor")
        out = foo_opt(inp)
        self.assertEqual(inp.storage(), out.storage())

    def test_index_select(self):
        def fn(a, b):
            return (
                torch.index_select(a, 0, b),
                torch.index_select(a, 1, b),
                torch.index_select(torch.index_select(a, 2, b), 1, b),
            )

        for ind_dtype in (torch.int32, torch.int64):
            self.common(
                fn,
                (
                    torch.randn(8, 8, 8),
                    torch.tensor([0, 0, 2, 1], dtype=ind_dtype),
                ),
            )

    @xfail_if_mps_unimplemented
    @skipCUDAIf(not TEST_CUDNN, "CUDNN not available")
    @skipIfXpu
    @skipIfRocm
    def test_cudnn_rnn(self):
        if self.device == "cpu":
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        def fn(
            a0,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
            a3,
            a4,
            a5,
        ):
            a1 = [
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b8,
                b9,
                b10,
                b11,
                b12,
                b13,
                b14,
                b15,
            ]
            return aten._cudnn_rnn(
                a0,
                a1,
                4,
                a3,
                a4,
                a5,
                2,
                2048,
                0,
                2,
                False,
                0.0,
                False,
                True,
                [],
                None,
            )

        self.common(
            fn,
            (
                torch.randn([92, 8, 2048]),
                torch.randn([8192, 2048]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([8192, 2048]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([8192, 4096]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([8192, 4096]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([167837696]),
                torch.randn([4, 8, 2048]),
                torch.randn([4, 8, 2048]),
            ),
            check_lowp=False,  # difference in rnn is too large between half and float inputs
        )

    def test_upsample_nearest1d(self):
        def fn(a):
            return (
                aten.upsample_nearest1d(a, [74], None),
                aten.upsample_nearest1d(a, [70], None),
                aten.upsample_nearest1d(a, [45], None),
                aten.upsample_nearest1d(a, [36], None),
                aten.upsample_nearest1d(a, None, [2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37]),))

    def test_upsample_nearest2d(self):
        def fn(a):
            return (
                aten.upsample_nearest2d(a, [74, 76]),
                aten.upsample_nearest2d(a, [70, 75]),
                aten.upsample_nearest2d(a, [45, 74]),
                aten.upsample_nearest2d(a, [36, 39]),
                aten.upsample_nearest2d(a, None, [2.0, 2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37, 38]),))

    def test_upsample_nearest3d(self):
        def fn(a):
            return (
                aten.upsample_nearest3d(a, [74, 76, 78], None),
                aten.upsample_nearest3d(a, [70, 75, 80], None),
                aten.upsample_nearest3d(a, [45, 74, 103], None),
                aten.upsample_nearest3d(a, [36, 39, 40], None),
                aten.upsample_nearest3d(a, None, [2.0, 2.0, 2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37, 38, 39]),))

    def test_upsample_nearest2d_backward(self):
        func = torch.ops.aten.upsample_nearest2d_backward

        def fn(a):
            return (
                func(a, output_size=[6, 12], input_size=[3, 3, 3, 6]),
                func(a, output_size=[6, 12], input_size=[3, 3, 4, 5]),
                func(a, output_size=[6, 12], input_size=[3, 3, 2, 8]),
                func(a, output_size=[6, 12], input_size=[3, 3, 2, 8]),
                func(a, output_size=[6, 12], input_size=[3, 3, 4, 7]),
            )

        self.common(fn, (torch.randn([3, 3, 6, 12]),))

    @skip_if_x86_mac()
    def test_upsample_bilinear2d_a(self):
        def fn(a):
            return (
                aten.upsample_bilinear2d(a, [45, 45], False, None),
                aten.upsample_bilinear2d(a, None, True, [2.0, 2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37, 38]),), atol=2.5e-5, rtol=1.3e-6)

    def test_upsample_bilinear2d_b(self):
        def fn(a):
            return aten.upsample_bilinear2d(a, None, True, [2.0, 2.0])

        self.common(
            fn,
            [
                torch.randn([1, 2, 40, 59]),
            ],
            atol=2.5e-5,
            rtol=1.3e-6,
        )

    @skip_if_gpu_halide  # accuracy issue
    def test_reflection_pad2d(self):
        def fn(a, pad):
            return (
                aten.reflection_pad2d(a, [1, 1, 1, 1]),
                aten.reflection_pad2d(a, pad),
            )

        self.common(
            fn,
            (
                torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),
                [5, 2, 3, 4],
            ),
        )

    @xfail_if_mps
    def test_reflection_pad2d_backward(self):
        def template(size, padding):
            def fn(grad_output, x):
                return aten.reflection_pad2d_backward(grad_output, x, padding)

            x = torch.randint(0, 999, size=size, dtype=torch.float32)
            result = aten.reflection_pad2d(x, padding)
            grad_output = torch.randn_like(result)

            self.common(
                fn, (grad_output, x), check_lowp=not is_halide_backend(self.device)
            )

        template([1, 1, 8, 8], [0, 0, 0, 0])
        template([1, 1, 8, 8], [1, 1, 1, 1])
        template([1, 1, 8, 8], [1, 2, 3, 4])
        template([1, 1, 8, 8], [0, -1, 2, 2])
        template([1, 1, 8, 8], [-1, 0, 2, 2])
        template([1, 1, 8, 8], [2, 2, 0, -1])
        template([1, 1, 8, 8], [2, 2, -1, 0])

    @xfail_if_mps_unimplemented  # Unsupported Border padding mode
    def test_grid_sampler_2d(self):
        def fn(a, b):
            return (
                aten.grid_sampler_2d(a, b, 0, 0, True),
                aten.grid_sampler_2d(a, b, 0, 1, False),
            )

        self.common(
            fn,
            (
                torch.randn([4, 3, 352, 352], dtype=torch.float32),
                torch.rand([4, 352, 352, 2], dtype=torch.float32) * 2 - 1,
            ),
            check_lowp=False,
            # Mismatched elements: 154697 / 1486848 (10.4%)
            # Greatest absolute difference: 0.0001976490020751953 at index (0, 0, 101, 243) (up to 1e-05 allowed)
            # Greatest relative difference: 7.332530120481928 at index (1, 1, 258, 301) (up to 1.3e-06 allowed)
            atol=0.0002,
            rtol=1.3e-06,
        )

    @requires_gpu()
    def test_grid_sampler_expand_preserves_view(self):
        if not self.device.startswith("cuda"):
            self.skipTest("requires CUDA")

        torch.manual_seed(0)
        torch._dynamo.reset()

        repeats = 9000
        batch = 48
        channels = 3
        img = 224
        grid_size = 13
        device = self.device

        class ExpandGridSampler(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.grid = torch.nn.Parameter(
                    torch.randn(repeats, grid_size, grid_size, 2, device=device)
                )
                self.fc = torch.nn.Linear(grid_size * grid_size * channels, 16).to(
                    device
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                per_channel = []
                for i in range(channels):
                    channel = x[:, i, ...].expand(repeats, -1, -1, -1)
                    patch = torch.nn.functional.grid_sample(
                        channel,
                        self.grid,
                        mode="bilinear",
                        align_corners=False,
                        padding_mode="border",
                    )
                    patch = patch.transpose(0, 1).flatten(start_dim=2)
                    per_channel.append(patch)
                x = torch.cat(per_channel, dim=2)
                return self.fc(x)

        model = ExpandGridSampler().to(device)
        compiled = torch.compile(model, backend="inductor")
        inp = torch.randn(batch, channels, img, img, device=device)

        out = compiled(inp)
        out.sum().backward()

        self.assertIsNotNone(model.grid.grad)

    def test_upsample_bicubic2d(self):
        def fn(a):
            return (
                aten.upsample_bicubic2d(a, (128, 128), True),
                aten.upsample_bicubic2d(a, (128, 256), False),
            )

        # Mismatched elements: 10 / 196608 (0.0%)
        # Greatest absolute difference: 1.3869255781173706e-05 at index (2, 1, 88, 65) (up to 1e-05 allowed)
        # Greatest relative difference: 0.0033082996811011046 at index (3, 1, 88, 91) (up to 1.3e-06 allowed)
        self.common(
            fn,
            (torch.randn([4, 3, 64, 32], dtype=torch.float32),),
            atol=2e-5,
            rtol=1e-3,
        )

    def test_float_index_expression(self):
        # Test that index propagation doesn't generate bad index_expr calls like
        # ops.index_expr(0.5*x, dtype) where the expression is not integral
        def fn(x):
            return aten.upsample_bicubic2d(x, (256, 256), False)

        x = torch.randn(1, 1, 128, 128, dtype=torch.float32, device=self.device)
        _, source_codes = run_and_get_code(fn, x)

        pattern = r"0\.50*\*[ix][\d]"
        for code in source_codes:
            self.assertIsNone(
                re.search(pattern, code), msg="Found bad index_expr in code:\n" + code
            )

    def test_float_index_expression_type_promotion(self):
        # Test that float indexing expressions participate in type promotion
        def fn(x):
            return x + 1.0 / x.size(0)

        x = torch.arange(10)
        self.common(fn, (x,))

    def test_sort(self):
        def fn(a, descending):
            return torch.sort(a)

        inp = torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32)
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    @parametrize("stable", (True, False))
    @parametrize("descending", (True, False))
    def test_nan_sort(self, descending, stable):
        def test_sort(x, descending, stable):
            out = torch.sort(x, descending=descending, stable=stable)
            if stable:
                return out
            else:
                # non stable idx may not be equal
                return out[0]

        tensor = torch.tensor(
            [
                0.7308,
                0.7053,
                0.3349,
                -0.7158,
                torch.nan,
                0.1234,
                1.0284,
                torch.nan,
                -1.8767,
                -0.4369,
            ],
            device=self.device,
        )
        inps = (tensor, descending, stable)
        a = torch.compile(test_sort)(*inps)
        b = test_sort(*inps)
        self.assertEqual(a, b, equal_nan=True)

    def test_sort_stable(self):
        def fn(a, descending):
            return a.sort(dim=-1, stable=True, descending=descending)

        # Duplicates give deterministic indices when stable sorting
        inp = torch.rand(10, 128, dtype=torch.float32)
        inp[:, 10:20] = 1.0
        inp[:, 30:40] = 1.0
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

        # Non-power of two
        inp = inp[:, :120]
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    def test_sort_bool(self):
        def fn(a, descending):
            return torch.sort(a.to(torch.int8), stable=True, descending=descending)

        inp = torch.randint(0, 2, size=[10, 128], dtype=torch.bool)
        self.common(fn, (inp, False))
        self.common(fn, (inp, True))

    @skipIfWindows(msg="Crash UT")
    def test_sort_transpose(self):
        def fn(a, descending):
            return torch.sort(a, stable=True, descending=descending)

        # MPS has correctness problem for transposed sort before MacOS15
        ctx = (
            contextlib.nullcontext()
            if self.device != "mps" or MACOS_VERSION >= 15.0
            else self.assertRaises(AssertionError)
        )
        inp = torch.randn(128, 10).transpose(0, 1)
        with ctx:
            self.common(fn, (inp, False))
            self.common(fn, (inp, True))

    def test_topk(self):
        def fn(a):
            return torch.topk(a, 2, -1)

        self.common(
            fn, (torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),)
        )

    def test_long_tensor(self):
        def fn(a):
            return (
                torch.LongTensor([294]).to(a.device) - a,
                torch.as_tensor([295]).to(a.device) + a,
            )

        self.common(fn, (torch.randint(0, 999, size=[8, 8]),))

    @skip_if_gpu_halide  # correctness issue
    def test_constant_pad_1d(self):
        def fn(a):
            return (
                aten.constant_pad_nd(a, [0, 1], 6.0),
                aten.constant_pad_nd(a, [2, 3], 99.0),
            )

        self.common(fn, (torch.randint(0, 999, size=[2, 16, 31], dtype=torch.float32),))

    def test_constant_pad_fill_dtype(self):
        def fn(a, b):
            return (
                aten.constant_pad_nd(a, (1, 1), 1.0) & b,
                aten.constant_pad_nd(a, (1, 1), 0.0) & b,
            )

        self.common(
            fn,
            (torch.randint(2, (4,), dtype=torch.bool), torch.ones(6, dtype=torch.bool)),
        )

    @skip_if_gpu_halide  # misaligned address
    def test_constant_pad_2d(self):
        def fn(a):
            return (
                aten.constant_pad_nd(a, [1, 1, 1, 1], 6.0),
                aten.constant_pad_nd(a, [1, 2, 3, 4], 99.0),
            )

        self.common(
            fn, (torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),)
        )

    def test_constant_pad_2d_strides_nonpositive(self):
        def fn(a):
            return torch.constant_pad_nd(a, [0, 0, 0, -2, 0, 0])

        self.common(
            fn, (torch.empty_strided((2, 4, 5), (20, 1, 4), dtype=torch.float32),)
        )

    @skip_if_gpu_halide  # misaligned address
    def test_constant_pad_3d(self):
        def fn(a):
            return (
                aten.constant_pad_nd(a, [1, 2, 3, 4, 5, 6], 6.0),
                aten.constant_pad_nd(a, [0, 0, 3, 4, 0, 0], 6.0),
            )

        self.common(
            fn, (torch.randint(0, 999, size=[2, 4, 4, 4], dtype=torch.float32),)
        )

    def test_constant_pad_float64(self):
        # Repro for https://github.com/pytorch/pytorch/issues/93351
        def fn(input):
            v1 = torch.nn.functional.pad(input, pad=(1, 0))
            return torch.gt(v1, input)

        _dtype = torch.float64

        ctx = (
            contextlib.nullcontext()
            if self.is_dtype_supported(_dtype)
            else self.assertRaises(TypeError)
        )
        x = torch.rand([1, 2, 2, 1], dtype=_dtype)
        with ctx:
            self.common(fn, (x,))

    def test_constant_pad_nd_inplace(self):
        def fn(a):
            return aten.constant_pad_nd(a, [0, 0])

        x = torch.randn([2], device=self.device)
        fn_compiled = torch.compile(fn)
        y = fn_compiled(x)
        self.assertTrue(y is not x)

    def test_l1_loss(self):
        def fn(a, b):
            return torch.nn.functional.l1_loss(a, b), torch.nn.functional.mse_loss(a, b)

        self.common(
            fn,
            (
                torch.randn([2, 3, 16, 16]),
                torch.randn([2, 3, 16, 16]),
            ),
            check_lowp=False,
        )

    def test_triu(self):
        def fn(a):
            return aten.triu(a, 1), aten.triu(a, 0), aten.triu(a, 2)

        self.common(fn, (torch.randn([2, 10, 10]),))

    def test_no_op_reduction(self):
        def fn(a):
            return a.sum(-1), torch.amax(a + 1, 1, keepdim=True)

        self.common(fn, (torch.randn([8, 1, 1]),))

    def test_inplace_add(self):
        @torch.compile(backend="inductor")
        def fn(x, y):
            return x.add_(y)

        inputs = (
            rand_strided((4, 4), (4, 1), device=self.device),
            rand_strided((4, 4), (4, 1), device=self.device),
        )
        inp_clone = inputs[0].clone()
        out = fn(*inputs)
        self.assertTrue(same(out, inp_clone + inputs[1]))
        self.assertTrue(out is inputs[0])

    # The following 2 tests are meant to check the logic that drops
    # xmask from triton load/store if xnumel = 1
    @requires_gpu()
    def test_single_elem(self):
        def fn(a):
            b = a + 1
            return (b,)

        self.common(fn, (torch.randn(1),))

    @requires_gpu()
    def test_single_elem_indirect(self):
        def fn(a, b):
            c = a[b] + 1
            return (c,)

        a = torch.randn(1)
        b = (torch.tensor([0], dtype=torch.int64),)

        self.common(fn, (a, b))

    # This test is meant to check for issues from the logic
    # that drops xmask from trito load/store if XBLOCK divides xnumel

    @requires_gpu()
    def test_xblock_divides_xnumel(self):
        def fn(a):
            b = a + 1
            return (b,)

        # assumption is that XBLOCK is always a divisor of 1024
        # so xmask will be dropped iff xnumel is multiple of 1024
        self.common(fn, (torch.randn(1024),))
        self.common(fn, (torch.randn(1025),))

    def test_inplace_mixed_dtype_ops(self):
        @torch.compile(backend="inductor")
        def fn(x, y):
            z = x + y.float()
            w = z.add_(y)
            return w.mul_(y)

        tgt_dtype = torch.double if self.device != "mps" else torch.half
        inputs = (
            rand_strided((4, 4), (4, 1), device=self.device, dtype=torch.float),
            rand_strided((4, 4), (4, 1), device=self.device, dtype=tgt_dtype),
        )
        out = fn(*inputs)
        out_eager = (inputs[0] + inputs[1].float()).add_(inputs[1]).mul_(inputs[1])
        self.assertTrue(same(out, out_eager))

    @config.patch(
        {"triton.unique_kernel_names": True, "triton.descriptive_names": False}
    )
    def test_kernel_names(self):
        @torch.compile(backend="inductor")
        def fn(x):
            return 2 * x

        inputs = (rand_strided((8,), (1,), device=self.device),)
        self.assertTrue(same(fn(*inputs), 2 * inputs[0]))

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_strided_inputs(self):
        @torch.compile(backend="inductor")
        def fn(x, y):
            return x + y

        inputs = (
            rand_strided((8, 16), (32, 2), device=self.device),
            rand_strided((8, 16), (16, 1), device=self.device),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_input_mutation1(self):
        def fn(a):
            b = a + 1
            a.copy_(b)
            c = a + 2
            return a * b / c

        arg1 = torch.randn(64, device=self.device)
        arg2 = arg1.clone()
        arg3 = torch.randn(64, device=self.device)
        arg4 = arg3.clone()
        correct1 = fn(arg1)
        correct2 = fn(arg3)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)
        actual2 = opt_fn(arg4)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(actual2, correct2))
        self.assertTrue(same(arg1, arg2))
        self.assertTrue(same(arg3, arg4))

    def test_input_mutation2(self):
        def fn(a):
            b = a + 1
            a.view(64).copy_(torch.tensor([66.0], device=a.device))
            c = a + 2
            return b, c

        # NOTE: this test fails when none of the inputs require grad.
        # That seems like an inductor bug.
        arg1 = torch.randn([1, 64], device=self.device).requires_grad_(True).add(1)
        arg2 = arg1.clone()
        correct1 = fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(arg1, arg2))

    def test_input_mutation3(self):
        def fn(a):
            a += 1
            a *= 2
            aten.sigmoid_(a)
            a = a.view(64)
            a += 3
            a *= 4
            aten.relu_(a)
            return a

        arg1 = torch.randn([1, 64], device=self.device)
        arg2 = arg1.clone()
        correct1 = fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(arg1, arg2))

    def test_input_mutation4(self):
        def fn(a):
            torch.relu_(a)
            return a

        arg1 = torch.randn([1, 64], device=self.device)
        arg2 = arg1.clone()
        correct1 = fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(arg1, arg2))

    def test_input_mutation5(self):
        def fn(x):
            tmp = x.ceil()
            x.add_(10)
            return tmp

        opt_fn = torch.compile(fn)

        a = torch.zeros((), dtype=torch.int64, device=self.device)
        a_expect = a.clone()
        expect = fn(a_expect)

        a_actual = a.clone()
        actual = opt_fn(a_actual)

        self.assertEqual(a_expect, a_actual)
        self.assertEqual(expect, actual)

    def test_slice_mutation1(self):
        def fn(a):
            x = torch.zeros_like(a)
            b = x + 1
            x[:, 3] = 3.0
            c = torch.clone(x)
            x[4, :] = 4.0
            d = x + 1
            return x, b, c, d

        self.common(fn, (torch.randn([8, 8]),))

    @skip_if_gpu_halide  # accuracy issue
    def test_slice_mutation2(self):
        def fn(a):
            a[:, 20:40] = a[:, 20:40] + 1
            a[:, 2:11] = a[:, 1:10] + 2

        arg1 = torch.randn([1, 64], device=self.device)
        arg2 = arg1.clone()
        fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        opt_fn(arg2)
        self.assertTrue(same(arg1, arg2))

    def test_slice_mutation3(self):
        def fn(a):
            a[:2, :2].fill_(10)

        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)

        x1 = torch.randn(8, 8, device=self.device)
        x2 = x1.clone()
        fn(x1)
        opt_fn(x2)
        self.assertEqual(x1, x2)

    def test_tensor_index_slice(self):
        def fn(a):
            x = torch.tensor([1, 2], device=self.device)
            y = torch.tensor([2, 3], device=self.device)
            xx = torch.tensor([1, 2], device=self.device).view(1, 2)
            yy = torch.tensor([1, 2, 3], device=self.device).view(3, 1)
            return [
                a[x, y],
                a[:, x, y],
                a[:, x, y, :],
                a[x, :, y],
                a[:, x, :, y, :],
                a[xx, yy],
                a[:, xx, yy],
                a[xx, :, yy],
                a[xx, yy, :],
                a[:, xx, :, yy],
            ]

        a = torch.arange(3 * 4 * 5 * 6 * 7, device=self.device).view(3, 4, 5, 6, 7)
        refs = fn(a)
        tests = torch.compile(fn)(a)
        for ref, test in zip(refs, tests):
            torch.testing.assert_close(ref, test)

    @torch._dynamo.config.patch(recompile_limit=10)
    def test_tensor_index_put_slice(self):
        def fn(a, version):
            x = torch.tensor([1, 2], device=self.device, dtype=torch.int32)
            y = torch.tensor([2, 3], device=self.device, dtype=torch.int32)

            xx = torch.tensor([1, 2], device=self.device).view(1, 2)
            yy = torch.tensor([1, 2, 3], device=self.device).view(3, 1)

            if version == 0:
                a[x, y] = torch.zeros_like(a[x, y])
            elif version == 1:
                a[:, x, y] = torch.zeros_like(a[:, x, y])
            elif version == 2:
                a[:, x, y, :] = torch.zeros_like(a[:, x, y, :])
            elif version == 3:
                a[x, :, y] = torch.zeros_like(a[x, :, y])
            elif version == 4:
                a[:, x, :, y, :] = torch.zeros_like(a[:, x, :, y, :])
            elif version == 5:
                a[xx, yy] = torch.zeros_like(a[xx, yy])
            elif version == 6:
                a[:, xx, yy] = torch.zeros_like(a[:, xx, yy])
            elif version == 7:
                a[xx, :, yy] = torch.zeros_like(a[xx, :, yy])
            elif version == 8:
                a[xx, yy, :] = torch.zeros_like(a[xx, yy, :])
            elif version == 9:
                a[:, xx, :, yy] = torch.zeros_like(a[:, xx, :, yy])

            return a

        a = torch.arange(3 * 4 * 5 * 6 * 7, device=self.device, dtype=torch.int32).view(
            3, 4, 5, 6, 7
        )
        for i in range(10):
            ref = fn(torch.clone(a), i)
            test = torch.compile(fn)(torch.clone(a), i)
            torch.testing.assert_close(ref, test)

    def test_indirect_load_broadcast(self):
        def fn(in_ptr0, in_ptr1, in_ptr2):
            return torch.gather(in_ptr1, 0, in_ptr2) + in_ptr0

        arg190 = rand_strided((32, 21), (1, 32), device=self.device, dtype=torch.int64)
        arg190.fill_(0)
        arg111 = rand_strided(
            (9521, 512), (512, 1), device=self.device, dtype=torch.float32
        )
        self.common(
            fn,
            (
                torch.randn(32, 1),
                arg111,
                arg190,
            ),
        )

    def test_roi_align(self):
        if not has_torchvision_roi_align():
            raise unittest.SkipTest("requires torchvision")

        def fn(a, b):
            return torch.ops.torchvision.roi_align(a, b, 0.25, 7, 7, 2, False)

        self.common(fn, (torch.zeros([4, 256, 296, 304]), torch.zeros([2292, 5])))

    # https://github.com/halide/Halide/issues/8256
    @config.patch("halide.scheduler_cuda", "Li2018")
    def test_nll_loss_forward(self):
        def fn(a, b):
            return aten.nll_loss_forward(a, b, None, 1, -100)

        labels = (
            torch.zeros([5], dtype=torch.int64),
            torch.tensor([-100, -100, 3, -100, -100], dtype=torch.int64),
        )
        inps = (torch.randn(5, 5), torch.randn(5, 5))
        for a, b in zip(inps, labels):
            self.common(
                fn,
                (a, b),
            )

    @xfail_if_mps  # dtypes mismatch
    def test_nll_loss_backward(self):
        def fn(a, b, c):
            return aten.nll_loss_backward(
                a, b, c, None, 1, -100, torch.tensor(1.0, device=self.device)
            )

        labels = (
            torch.zeros([5], dtype=torch.int64),
            torch.tensor([-100, -100, 3, -100, -100], dtype=torch.int64),
        )
        inps = (torch.randn(5, 5), torch.randn(5, 5))
        grad_outs = (torch.randn(()), torch.randn(()))
        for a, b, c in zip(grad_outs, inps, labels):
            self.common(
                fn,
                (a, b, c),
            )

    def test_isinf(self):
        def fn(x):
            return x.isinf(), x.isnan()

        values = [1, float("inf"), 2, float("-inf"), float("nan")]
        for dtype in [torch.float32, torch.float64, torch.half, torch.bfloat16]:
            ctx = (
                contextlib.nullcontext()
                if self.is_dtype_supported(dtype)
                else self.assertRaises(TypeError)
            )
            with ctx:
                self.common(fn, [torch.tensor(values, dtype=dtype)], check_lowp=False)

    @skip_if_halide  # different nan behavior in ==
    def test_isinf2(self):
        def fn(x):
            y = torch.tensor(
                [1, float("inf"), 2, float("-inf"), float("nan")], device=self.device
            )
            return x == y

        self.common(
            fn, (torch.tensor([1, float("inf"), 2, float("-inf"), float("nan")]),)
        )

    def test_any(self):
        def fn(x):
            return (
                x.any(-1),
                x.isinf().any(),
                torch.all(x.isinf(), dim=0),
                torch.all(torch.logical_not(x.isinf())),
            )

        self.common(fn, [-torch.rand(64)])
        tmp = torch.randn(16, 8)
        tmp[1, 1] = float("inf")
        self.common(fn, [tmp])

    @skip_if_gpu_halide
    def test_multilayer_any(self):
        def fn(x):
            return (x.isinf().any(), x.isfinite().all())

        sample = torch.rand(9, 3, 353, 353)
        self.common(fn, [sample])

        sample.view(-1)[-1] = float("inf")
        self.common(fn, [sample])

    def test_inplace_activations(self):
        def fn(x):
            a = aten.hardswish_(x + 1)
            b = aten.hardtanh_(x + 1)
            c = aten.leaky_relu_(x + 1)
            d = aten.silu_(x + 1)
            e = aten.log1p(x + 1)
            f = aten.masked_fill_(x + 1, torch.zeros_like(x, dtype=torch.bool), 99.0)
            h = aten.masked_fill_(x + 1, torch.ones_like(x, dtype=torch.bool), 99.0)
            return (a, b, c, d, e, f, h)

        self.common(fn, [torch.randn(64) * 10])

    def test_baddbmm(self):
        def fn(a, b, c, beta):
            return aten.baddbmm(a, b, c, beta=beta)

        b = torch.randn(6, 128, 64)
        c = torch.randn(6, 64, 100)
        options = itertools.product(
            [torch.randn(6, 1, 100), torch.randn(6, 1, 100).fill_(torch.nan)],
            [0.0, 1.0],
        )
        for a, beta in options:
            self.common(
                fn,
                [a, b, c, beta],
                # Mismatched elements: 1212 / 76800 (1.6%)
                # Greatest absolute difference: 0.001953125 at index (0, 0, 93) (up to 1e-05 allowed)
                # Greatest relative difference: 1.0 at index (3, 19, 4) (up to 0.001 allowed)
                atol=0.002,
                rtol=0.001,
            )

    @config.patch({"triton.max_tiles": 2})
    def test_fuse_tiled(self):
        def fn(a, b, c):
            return a + b, c + 1

        self.common(
            fn, [torch.randn(128, 1), torch.randn(1, 128), torch.randn(128, 128)]
        )

    def test_expand_as(self):
        def fn(a, b):
            return aten.expand_as(a, b), aten.expand_as(a + 1, b + 1) + 1

        self.common(
            fn,
            [
                torch.randn(6, 1, 100),
                torch.randn(6, 128, 100),
            ],
        )

    def test_index_put1(self):
        def fn(a, b, c):
            return (
                torch.index_put(a, [b], c),
                torch.index_put_(a + 1, [b + 1], c + 1) + 1,
            )

        self.common(
            fn,
            [
                torch.randn([800, 256, 7, 7]),
                torch.randperm(601),
                torch.randn([601, 256, 7, 7]),
            ],
        )
        self.common(
            fn, [torch.randn(1024, 4, 2), torch.arange(4), torch.randn(4, 1, 1)]
        )

    def test_index_put2(self):
        def fn(a, b, c):
            return torch.index_put(a, [b], c, True)

        self.common(
            fn,
            [
                torch.randn([100, 256, 7, 7]),
                torch.randint(0, 100, size=[600], dtype=torch.int64),
                torch.randn([600, 256, 7, 7]),
            ],
            # workaround for https://github.com/triton-lang/triton/issues/558
            check_lowp=False,
        )

    def test_index_put3(self):
        def fn(a, b, c):
            torch.ops.aten.index_put_(a, (None, b, None), c)
            a1 = a + 1
            torch.ops.aten.index_put_(a1, (None, b + 1, None), c + 1)
            return (a, a1)

        self.common(
            fn,
            [
                torch.randn([1024, 4, 2]),
                torch.arange(3),
                torch.randn([1024, 1, 2]),
            ],
        )

    def test_index_put4(self):
        # a, b[0] are not broadcastable
        # https://github.com/pytorch/pytorch/issues/97104
        def fn(a, b, c):
            return torch.index_put(a, [b], c)

        self.common(
            fn,
            [
                torch.rand([8, 2]),
                torch.rand([8]) > 0.5,
                torch.rand([]),
            ],
        )

    def test_index_put_as_masked_fill(self):
        def fn(a, b, c, d):
            a = a.clone()
            torch.ops.aten.index_put_(a, [b], c, d)
            return a

        self.common(
            fn,
            (
                torch.randn([1024, 4, 2]),
                torch.randn([1024, 4, 2]) > 0,
                torch.randn([]),
                False,
            ),
        )

        self.common(
            fn,
            (
                torch.randn([1024, 4, 2]),
                torch.randn([1024, 4, 2]) > 0,
                torch.randn([]),
                True,
            ),
        )

    def test_index_put_fallback1(self):
        def fn(a, b, c, d):
            a = a.clone()
            torch.ops.aten.index_put_(a, [b], c, d)
            return a

        self.common(
            fn,
            (
                torch.randn([3]),
                torch.as_tensor([True, True, False]),
                torch.randn([2]),
                False,
            ),
        )

        self.common(
            fn,
            (
                torch.randn([3]),
                torch.as_tensor([True, True, False]),
                torch.randn([2]),
                True,
            ),
        )

    def test_index_put_fallback2(self):
        def fn(a, b, c, d, e):
            a = a.clone()
            torch.ops.aten.index_put_(a, [None, b, c], d, e)
            return a

        self.common(
            fn,
            (
                torch.randn([1, 2, 3]),
                torch.as_tensor([0, 1]),
                torch.as_tensor([True, True, False]),
                torch.randn([]),
                False,
            ),
        )
        self.common(
            fn,
            (
                torch.randn([1, 2, 3]),
                torch.as_tensor([0, 1]),
                torch.as_tensor([True, True, False]),
                torch.randn([]),
                True,
            ),
        )

    def test_index_put_deterministic_fallback(self):
        with DeterministicGuard(True):

            def fn(a, b, c):
                return torch.index_put(a, [b], c, True)

            self.common(
                fn,
                [
                    torch.randn([100, 32]),
                    torch.randint(0, 100, size=[600], dtype=torch.int64),
                    torch.randn([600, 32]),
                ],
                check_lowp=False,
            )

    @skip_if_gpu_halide  # https://github.com/halide/Halide/issues/8312
    def test_index_put_index(self):
        def fn(ind, x, src):
            y = torch.ops.aten.index_put.default(x, [ind], src)
            return torch.ops.aten.index.Tensor(y, [ind])

        args = [torch.tensor([1], dtype=torch.int64), torch.randn(8, 4), torch.randn(4)]
        self.common(fn, args)

    def test_index_put_reinplace(self):
        def fn(x, idx):
            src = torch.ones(idx.size(0), device=x.device)
            x.index_put_((idx,), src)
            return x.expand((2, x.shape[0]))

        a = torch.randn(1024)
        idx = torch.arange(10)
        torch._inductor.metrics.generated_kernel_count = 0
        self.common(fn, (a, idx))
        assertGeneratedKernelCountEqual(self, 1)

    def test_index_put_failed_reinplace(self):
        def fn(x, idx):
            src = torch.ones(idx.size(0), device=x.device)
            y = x.index_put((idx,), src)
            return x, y

        a = torch.randn(1024)
        idx = torch.arange(10)
        torch._inductor.metrics.generated_kernel_count = 0
        self.common(fn, (a, idx))
        assertGeneratedKernelCountEqual(self, 2)

    def test_adding_tensor_offsets(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x[16:32]

        with torch.no_grad():
            x = torch.randn(1024, device=self.device)
            self.assertEqual(fn(x[0:]), x[16:][:16])
            self.assertEqual(fn(x[128:]), x[128 + 16 :][:16])

    def test_index_float_zero(self):
        def fn(arg0, arg1, arg2):
            t1 = torch.tanh(arg0)
            t2 = t1.clone()
            t2.fill_(arg1.item())
            t3 = torch.clamp(t2, 0, arg2.size(0) - 1).to(torch.long)
            return torch.nn.functional.embedding(t3, arg2)

        arg0 = torch.randint(0, 1000, [47], dtype=torch.int64, device=self.device)
        arg1 = torch.randint(0, 1000, [], dtype=torch.int64, device=self.device)
        arg2 = torch.rand([256, 88], dtype=torch.float16, device=self.device)

        cfn = torch.compile(fullgraph=True, dynamic=True)(fn)

        self.assertEqual(fn(arg0, arg1, arg2), cfn(arg0, arg1, arg2))

    # from GPT2ForSequenceClassification
    @skip_if_gpu_halide
    def test_index_tensor(self):
        def fn(x, y):
            ne = torch.ops.aten.ne.Scalar(x, 0)
            sum = torch.ops.aten.sum.dim_IntList(ne, [-1])
            sub = torch.ops.aten.sub.Tensor(sum, 1)
            iota = torch.ops.prims.iota.default(
                1,
                start=0,
                step=1,
                dtype=torch.int64,
                device=x.device,
                requires_grad=False,
            )
            return torch.ops.aten.index.Tensor(y, [iota, sub])

        self.common(fn, [torch.randn(1, 1024), torch.randn(1, 1024, 2)])

    @config.patch(fallback_random=True)
    def test_bernoulli1(self):
        def fn(a):
            b = a.clone()
            # aten.bernoulli_() uses aten.bernoulli.p() behind the scene, so it will be decomposed.
            return aten.bernoulli_(b).sum() / torch.prod(torch.tensor(a.size()))

        p = 0.3
        self.common(
            fn,
            [
                torch.ones(200, 200) * p,
            ],
            atol=p * 0.06,
            rtol=0.06,
        )

    @skip_if_triton_cpu
    def test_bernoulli2(self):
        def fn(a):
            return aten.bernoulli(a).sum() / torch.prod(torch.tensor(a.size()))

        p = 0.3
        self.common(
            fn,
            [torch.ones(200, 200) * p],
            atol=p * 0.06,
            rtol=0.06,
        )

    def test_narrow(self):
        def fn(x):
            return (
                aten.narrow(x, 1, 10, 16),
                aten.narrow(x + 2, 0, 10, 16) + 1,
                aten.narrow_copy(x, 1, 10, 16),
            )

        self.common(fn, [torch.randn(64, 64)])

    def test_as_strided(self):
        def fn(x):
            return (
                aten.as_strided(x, (8, 8, 64), (8 * 64, 64, 1), 0),
                aten.as_strided(x + 1, (8, 8, 64), (8 * 64, 64, 1), 0) + 2,
            )

        def fn_channels_last(x):
            return (
                aten.as_strided(
                    x, (8, 384, 2, 20, 12), (153600, 1, 61440, 384, 7680), 0
                ),
                aten.as_strided(
                    x + 1, (8, 384, 2, 20, 12), (153600, 1, 61440, 384, 7680), 0
                )
                + 2,
            )

        self.common(fn, [torch.randn(64, 64)])
        self.common(
            fn_channels_last,
            [torch.randn(8, 384, 20, 20).to(memory_format=torch.channels_last)],
        )

    def test_exact_stride(self):
        full = torch.randn((16, 16), device=self.device)
        view = torch.as_strided(full, (16, 8), full.stride())

        def fn(x):
            result = x + x
            result_strided = torch.empty_strided(
                x.size(), x.stride(), device=self.device
            )
            result_strided[:] = result
            return result_strided

        self.common(fn, [view])
        reference_out = fn(view)
        compiled_fn = torch.compile(fn)
        actual_out = compiled_fn(view)
        self.assertEqual(reference_out.stride(), actual_out.stride())

    def test_like_channels_last(self):
        def foo():
            randn = torch.randn((4, 3, 8, 8), device=self.device, dtype=torch.float32)
            xc = randn.contiguous(memory_format=torch.channels_last)
            clone = torch.zeros_like(xc, memory_format=torch.preserve_format)
            rand_like = torch.rand_like(randn)
            return (xc, clone, rand_like)

        out = foo()
        out_comp = torch.compile()(foo)()

        for t, t_comp in zip(out, out_comp):
            self.assertEqual(t.stride(), t_comp.stride())

    def test_as_strided_scatter(self):
        def fn(a, b):
            return aten.as_strided_scatter(
                a * 8 + 10,
                b * 2 - 4,
                size=(a.shape[0], a.shape[1] // 2),
                stride=(a.shape[1], 2),
                storage_offset=0,
            )

        self.common(fn, [torch.randn(10, 1024), torch.randn(10, 512)])

    def test_select_scatter(self):
        def fn(x, a, b):
            return (
                aten.select_scatter(x, a, 1, 0),
                aten.select_scatter(x, b, 0, 1),
            )

        self.common(
            fn,
            [
                torch.randn(8, 197, 38),
                torch.randn(8, 38),
                torch.randn(197, 38),
            ],
        )

    @skip_if_gpu_halide  # accuracy issue
    def test_slice_scatter(self):
        def fn(x, a):
            return (
                aten.slice_scatter(x, a, 2, 10, -10),
                aten.slice_scatter(x, a[:, :, :40], 2, 10, -10, 2),
            )

        self.common(
            fn,
            [
                torch.randn(4, 8, 100),
                torch.randn(4, 8, 80),
            ],
        )

    def test_slice_scatter2(self):
        def fn(a, b):
            return aten.slice_scatter(a, b, 0, 0, 9223372036854775807)

        self.common(
            fn,
            [
                torch.randn([8, 197, 384]),
                torch.randn([8, 197, 384]),
            ],
        )

    def test_slice_scatter3(self):
        def fn(a, b):
            return aten.slice_scatter.default(a, b, 1, 1, 9223372036854775807, 2)

        self.common(
            fn,
            [
                torch.randn([1, 4]),
                torch.randn([1, 2]),
            ],
        )

    def test_slice_scatter4(self):
        def fn(a, b):
            return aten.slice_scatter.default(a, b, 1, 2, 9223372036854775807, 3)

        self.common(
            fn,
            [
                torch.randn([1, 9]),
                torch.randn([1, 3]),
            ],
        )

    def test_slice_scatter5(self):
        # empty slices that require clamping the start or end
        def fn(a, b):
            return (
                aten.slice_scatter.default(a, b, 0, 2, 0, 1),
                aten.slice_scatter.default(a, b, 0, a.shape[0], a.shape[0] + 10, 1),
                aten.slice_scatter.default(a, b, 0, -20, 0, 1),
                aten.slice_scatter.default(a, b, 0, -20, -16, 1),
            )

        a = torch.arange(10, dtype=torch.float)
        b = torch.empty(0)
        self.common(fn, [a, b])

    @with_tf32_off
    def test_slice_scatter_reinplace(self):
        class M(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.linear1 = nn.Linear(64, 64, bias=False)
                self.cache_k = torch.zeros((56, 384, 8, 64), device=device)

            def forward(self, x, start_pos):
                bsz, seqlen, _, _ = x.shape
                xk = self.linear1(x)
                with torch.no_grad():
                    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
                keys = self.cache_k[:bsz, : start_pos + seqlen]
                scores = torch.matmul(
                    xk.transpose(1, 2), keys.transpose(1, 2).transpose(2, 3)
                )
                return scores

        kv_cache_module = M(self.device)
        inp = torch.randn(1, 32, 8, 64)

        # Test that the cache update is reinplaced such that the cache is updated inplace
        # rather than copy-scatter-copy-back.

        torch._inductor.metrics.generated_kernel_count = 0
        with torch.no_grad():
            self.common(kv_cache_module, (inp, 1), check_lowp=False)

        if (
            config.triton.native_matmul
            and config.cuda_backend == "triton"
            and self.device == "cuda"
        ):
            assertGeneratedKernelCountEqual(self, 2)
        else:
            assertGeneratedKernelCountEqual(self, 1)

    @skipIfMPS
    def test_slice_scatter_dtype_consistency(self):
        # Test dtype consistency of slice_scatter
        def fn(x, y):
            return torch.slice_scatter(y, x, 0)

        for dtype in [
            torch.int64,
            torch.float64,
        ]:
            self.common(
                fn,
                [
                    torch.tensor([0], dtype=dtype),
                    torch.tensor([0], dtype=torch.float32),
                ],
            )

    @skip_if_gpu_halide  # compile error on gpu
    def test_scatter1(self):
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b)

        self.common(
            fn,
            [
                torch.zeros(2, 3),
                -1,
                torch.tensor([[0]]),
                torch.ones(2, 3),
            ],
        )

    def test_scatter2(self):
        if self.device == "cuda":
            raise unittest.SkipTest("unstable on sm86")

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        def fn(a, dim, index, b):
            return aten.scatter.reduce(a, dim, index, b, reduce="add")

        self.common(
            fn,
            [
                torch.zeros(64, 512),
                0,
                torch.zeros((64, 512), dtype=torch.int64),
                torch.ones(64, 512),
            ],
            check_lowp=check_lowp,
        )

    def test_scatter3(self):
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b, reduce="add")

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        self.common(
            fn,
            [
                torch.randn(5, 29, 13),
                2,
                torch.tensor([[[3, 5, 7, 9]]]),
                0.8,  # src can be a scalar
            ],
            # Mismatched elements: 1 / 1885 (0.1%)
            # Greatest absolute difference: 0.00018310546875 at index (0, 0, 3) (up to 1e-05 allowed)
            # Greatest relative difference: 0.0022371364653243847 at index (0, 0, 3) (up to 0.001 allowed)
            atol=2e-4,
            rtol=1e-3,
            check_lowp=check_lowp,
        )

    def test_scatter4(self):
        def fn(x, ind, src):
            return torch.scatter(x, 0, ind, src)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                self.common(
                    fn,
                    [
                        torch.randn(196, 992),
                        torch.randint(196, (1, 992)),
                        torch.randn(1, 992),
                    ],
                    check_lowp=check_lowp,
                )

    def test_scatter5(self):
        def fn(a, dim, index, b, reduce):
            a = a.clone()
            a.scatter_(dim, index, b, reduce=reduce)
            a1 = a + 1.0
            a1.scatter_(dim, index, b, reduce=reduce)
            return (a, a1)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for reduce in ["add", "multiply"]:
            self.common(
                fn,
                [
                    torch.ones((4, 5)),
                    0,
                    torch.tensor([[1], [2], [3]], dtype=torch.int64),
                    torch.randn(4, 5),
                    reduce,
                ],
                check_lowp=check_lowp,
            )

    def test_scatter6(self):
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                self.common(
                    fn,
                    [
                        torch.randn(5, 8, 13),
                        2,
                        torch.tensor([[[3, 5, 7, 9]]]),
                        0.8,  # src can be a scalar
                    ],
                    check_lowp=check_lowp,
                )

    @unittest.skip("Flaky test, needs debugging")
    def test_scatter_add1(self):
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        self.common(
            fn,
            [
                torch.randn(2, 3),
                0,
                torch.tensor([[0]]),
                torch.randn(2, 3),
            ],
            check_lowp=check_lowp,
        )

    def test_scatter_add2(self):
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        self.common(
            fn,
            [
                torch.randn(2, 3),
                0,
                torch.tensor([[0, 0, 0], [1, 1, 1]]),
                torch.randn(2, 3),
            ],
            check_lowp=check_lowp,
        )

    def test_scatter_add3(self):
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for deterministic in [False, True]:
            if deterministic and self.device == "xpu":
                # There is no deterministic implementation for scatter_add on Intel GPU.
                continue
            with DeterministicGuard(deterministic):
                self.common(
                    fn,
                    [
                        torch.randn(5, 29, 13),
                        2,
                        torch.tensor([[[3, 5, 7, 9]]]),
                        torch.randn(1, 1, 10),
                    ],
                    check_lowp=check_lowp,
                )

    def test_scatter_reduce1(self):
        def fn(a, dim, index, b):
            return aten.scatter_reduce(a, dim, index, b, "sum")

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        self.common(
            fn,
            [
                torch.randn(5, 29, 13),
                2,
                torch.tensor([[[3, 5, 7, 9]]]),
                torch.randn(1, 1, 10),
            ],
            check_lowp=check_lowp,
        )

    def test_scatter_reduce2(self):
        def fn(a, dim, index, b, reduce):
            return aten.scatter_reduce(a, dim, index, b, reduce, include_self=False)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for reduce in ["sum", "amax"]:
            self.common(
                fn,
                [
                    torch.randn(2, 3),
                    0,
                    torch.zeros((2, 3), dtype=torch.int64),
                    torch.randn(2, 3),
                    reduce,
                ],
                check_lowp=check_lowp,
            )

    def test_scatter_reduce3(self):
        def fn(a, dim, index, b, reduce):
            a = a.clone()
            a.scatter_reduce_(dim, index, b, reduce=reduce)
            a1 = a + 1.0
            a1.scatter_reduce_(dim, index, b, reduce=reduce)
            return (a, a1)

        check_lowp = True
        if self.device == "xpu":
            check_lowp = False

        for reduce in ["sum", "prod"]:
            self.common(
                fn,
                [
                    torch.ones((4, 5)),
                    0,
                    torch.tensor([[1], [2], [3]], dtype=torch.int64),
                    torch.randn(4, 5),
                    reduce,
                ],
                check_lowp=check_lowp,
            )

    @skip_if_gpu_halide
    def test_dense_mask_index(self):
        r"""
        There will be a little difference for reduce order between aten and inductor
        https://github.com/pytorch/pytorch/pull/122289
        Absolute difference: 0.00067138671875 (up to 1e-05 allowed)
        Relative difference: 3.1747371732500974e-06 (up to 1.3e-06 allowed)
        """
        kwargs = {}
        if self.device == "cpu":
            kwargs["atol"] = 1e-4
            kwargs["rtol"] = 1.3e-5

        def fn(x, y):
            y = torch.ops.aten.select.int(y, 0, 2)
            z = x * y
            return z.sum()

        self.common(fn, [torch.randn(102400), torch.randn(3)], **kwargs)

    def test_empty1(self):
        def fn():
            return torch.empty((1, 128, 128))

        self.common(fn, [], assert_equal=False)

    def test_empty2(self):
        def fn():
            return aten.empty((1, 128, 128))

        self.common(fn, [], assert_equal=False)

    def test_new_empty(self):
        def fn(a):
            return aten.new_empty(a, [1, 128, 128])

        self.common(fn, [torch.randn(55)], assert_equal=False)

    def test_empty_strided(self):
        def fn():
            return aten.empty_strided([1, 128, 128], [16384, 128, 1])

        self.common(fn, [], assert_equal=False)

    def test_new_empty_strided(self):
        def fn(a):
            return aten.new_empty_strided(a, [1, 128, 128], [16384, 128, 1])

        self.common(fn, [torch.randn(55)], assert_equal=False)

    def test_dropout_trivial_0(self):
        def fn1(a):
            return torch.nn.functional.dropout(a, 0.0, True) + a

        self.common(fn1, [torch.randn(55)])

    def test_dropout_trivial_1(self):
        def fn2(a):
            return torch.nn.functional.dropout(a, 1.0, True) + a

        self.common(fn2, [torch.randn(55)])

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_dropout(self):
        random.seed(1234)
        torch.manual_seed(1234)

        @torch.compile(backend="inductor")
        def fn1(a):
            return torch.nn.functional.dropout(a)

        x = torch.ones(1000, device=self.device, dtype=torch.float32)
        result1 = fn1(x)
        self.assertTrue(400 < result1.nonzero().shape[0] < 600)
        self.assertTrue(0.9 < result1.mean().item() < 1.1)

        random.seed(1234)
        torch.manual_seed(1234)

        @torch.compile(backend="inductor")
        def fn2(a):
            return torch.nn.functional.dropout(a, 0.5, True)

        result2 = fn2(x)
        self.assertTrue(400 < result2.nonzero().shape[0] < 600)
        self.assertTrue(0.9 < result2.mean().item() < 1.1)

    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_dropout_deterministic(self):
        @torch.compile(backend="inductor")
        def fn(a):
            return torch.nn.functional.dropout(a, 0.55, True)

        for cg in [False, True]:
            with patch.object(config.triton, "cudagraphs", cg):
                torch._dynamo.reset()

                x = torch.ones(1024, device=self.device, dtype=torch.float32)

                torch.manual_seed(1234)
                a0 = fn(x).clone()
                a1 = fn(x).clone()
                a2 = fn(x).clone()

                torch.manual_seed(1234)
                b0 = fn(x).clone()
                b1 = fn(x).clone()
                b2 = fn(x).clone()

                # same seed, same values
                self.assertTrue(torch.allclose(a0, b0))
                self.assertTrue(torch.allclose(a1, b1))
                self.assertTrue(torch.allclose(a2, b2))

                # different calls, different values
                self.assertFalse(torch.allclose(a0, a1))
                self.assertFalse(torch.allclose(a1, a2))

    def test_rand_like_deterministic(self):
        @torch.compile(backend="inductor")
        def fn(a):
            return torch.rand_like(a), torch.rand_like(a)

        x = torch.ones(1024, device=self.device, dtype=torch.float32)

        torch.manual_seed(1234)
        a0 = fn(x)[0].clone()
        a1 = fn(x)[0].clone()
        a2 = fn(x)[0].clone()

        torch.manual_seed(1234)
        b0 = fn(x)[0].clone()
        b1 = fn(x)[0].clone()
        b2 = fn(x)[0].clone()

        # same seed, same values
        self.assertTrue(torch.allclose(a0, b0))
        self.assertTrue(torch.allclose(a1, b1))
        self.assertTrue(torch.allclose(a2, b2))

        # different calls, different values
        self.assertFalse(torch.allclose(a0, a1))
        self.assertFalse(torch.allclose(a1, a2))

        c, d = fn(x)
        self.assertFalse(torch.allclose(c, d))
        self.assertTrue((c >= 0).all())
        self.assertTrue((c < 1).all())
        self.assertTrue((d >= 0).all())
        self.assertTrue((d < 1).all())

    @config.patch(implicit_fallbacks=True)
    def test_needs_contiguous_strides(self):
        # Construct a custom op whose output strides are not contiguous
        @torch.library.custom_op("mylib::myop", mutates_args={})
        def myop(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(2, 2).t()

        @myop.register_fake
        def _(x):
            return torch.zeros(2, 2).t()

        # custom op that needs contiguous inputs
        @torch.library.custom_op(
            "mylib::second_op",
            mutates_args={},
            tags=[torch._C.Tag.needs_contiguous_strides],
        )
        def second_op(x: torch.Tensor) -> torch.Tensor:
            assert x.is_contiguous()
            return torch.ones(2, 2)

        @second_op.register_fake
        def _(x):
            return torch.ones(2, 2)

        def f(x):
            y = myop(x)
            return second_op(y)

        # Check that the x.is_contiguous() assertion never gets triggered
        x = torch.randn(2, 2)
        _ = torch.compile(f, backend="inductor", fullgraph=True)(x)

    @config.patch(implicit_fallbacks=True)
    def test_fallback_mutable_op_basic(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            def impl(a, b, c, d, e=2):
                a.add_(b[0] * c * e)
                if d is not None:
                    d.add_(b[1])

            m.define(
                "inplace_(Tensor(a!) a, Tensor[] b, SymInt c, *, Tensor(b!)? d, SymInt e=2) -> ()"
            )
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            # We do some clones and copy_ to test that Inductor doesn't reorder
            # the copy_ w.r.t. inplace_.
            def f(a, b1, b2, c, d):
                a_ = a.clone()
                d_ = d if d is None else d.clone()
                torch.ops.mylib.inplace_(a_, (b1, b2), c, d=d_)
                a.copy_(a_)
                if d is not None:
                    d.copy_(d_)
                return ()

            a = torch.tensor([0.0, 1.0, 2])
            b = [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([1.0, 4.0, 6.0])]
            c = 4
            d = torch.tensor([2.0, 1, 0])
            args = (a, b[0], b[1], c, d)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            mod = make_fx(f)(*cloned_args)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f = compile_fx_inner(mod, cloned_args)

            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f(list(cloned_args))
            f(*args)
            self.assertEqual(cloned_args, args)

    @skip_if_cpp_wrapper(
        "Without major redesign, cpp_wrapper will not support custom ops that are "
        "defined in Python."
    )
    @config.patch(implicit_fallbacks=True)
    def test_fallback_mutable_op_list_tensor(self):
        @torch.library.custom_op(
            "mylib::mysin",
            mutates_args=["out_list"],
            schema="(Tensor x, Tensor(a!)[]? out_list) -> Tensor",
        )
        def mysin(x, out_list) -> torch.Tensor:
            r = x.sin()
            if out_list is not None:
                out_list[0].copy_(r)
            return r

        @mysin.register_fake
        def _(x, out_list) -> torch.Tensor:
            return torch.empty_like(x)

        def fn(x):
            x = x * 3
            s = [torch.empty_like(x)]
            x = mysin(x, s)
            x = x / 3
            return x, s[0]

        x = torch.randn(3, requires_grad=False)
        expected = fn(x)
        result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(result, expected)

    @config.patch(implicit_fallbacks=True)
    def test_fallback_mutable_op_with_return(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            def impl(a, b, c, d, e=2):
                a.add_(b[0] * c * e)
                if d is not None:
                    d.add_(b[1])
                return b[0] + b[1]

            m.define(
                "inplace_(Tensor(a!) a, Tensor[] b, SymInt c, *, Tensor(b!)? d, SymInt e=2) -> Tensor"
            )
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            # We do some clones and copy_ to test that Inductor doesn't reorder
            # the copy_ w.r.t. inplace_.
            def f(a, b0, b1, c, d):
                a_ = a.clone()
                d_ = d if d is None else d.clone()
                res = torch.ops.mylib.inplace_(a_, (b0, b1), c, d=d_)
                a.copy_(a_)
                if d is not None:
                    d.copy_(d_)
                return (res,)

            a = torch.tensor([0.0, 1.0, 2])
            b = [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([1.0, 4.0, 6.0])]
            c = 4
            d = torch.tensor([2.0, 1, 0])
            args = (a, b[0], b[1], c, d)

            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            mod = make_fx(f)(*cloned_args)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f = compile_fx_inner(mod, cloned_args)

            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_out = compiled_f(list(cloned_args))
            out = f(*args)
            self.assertEqual(cloned_args, args)
            self.assertEqual(compiled_out, out)

    @config.patch(implicit_fallbacks=True)
    def test_fallback_mutable_op_no_mutated_tensors(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            def impl(a, b):
                if b is not None:
                    b.add_(1)

            m.define("inplace_(Tensor a, Tensor(b!)? b) -> ()")
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            def f(a):
                torch.ops.mylib.inplace_(a, None)
                return ()

            a = torch.tensor([0.0, 1.0, 2])
            args = (a,)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            mod = make_fx(f)(*cloned_args)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f = compile_fx_inner(mod, cloned_args)

            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            compiled_f(list(cloned_args))
            f(*args)
            self.assertEqual(cloned_args, args)

    @config.patch(implicit_fallbacks=True)
    @skip_if_cpp_wrapper(
        "Without major redesign, cpp_wrapper will not support custom ops that are "
        "defined in Python."
    )
    def test_fallback_mutable_op_list(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            def impl(a, b):
                for bi in b:
                    bi.add_(a)

            m.define("inplace_(Tensor a, Tensor(a!)[] b) -> ()")
            m.impl("inplace_", impl, "CompositeExplicitAutograd")

            def f(a, b):
                torch.ops.mylib.inplace_(a, b)
                return None

            a = torch.tensor([0.0, 1.0, 2])
            b = [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([1.0, 4.0, 6.0])]
            args = (a, b)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            mod = make_fx(f)(*cloned_args)
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)

            compiled_f = compile_fx_inner(mod, cloned_args)

        @torch.library.custom_op("mylib::sin_out", mutates_args={"outs"})
        def sin_out(x: torch.Tensor, outs: list[torch.Tensor]) -> None:
            x_np = x.numpy()
            assert len(outs) == 2
            out_np0 = out[0].numpy()
            out_np1 = out[1].numpy()
            np.sin(x_np, out=out_np0)
            np.sin(x_np, out=out_np1)

        @torch.compile
        def g(x):
            outs = [torch.empty_like(x) for _ in range(2)]
            sin_out(x, outs)
            return outs

        x = torch.randn(3)
        out = [torch.empty_like(x) for _ in range(2)]
        y = g(x)

    @xfail_if_mps_unimplemented  # rng_prims not supported for MPS
    def test_functionalize_rng_wrappers(self):
        # Ideally, we would like to use torch.compile for these operators. But
        # currently the plan is to introduce these operators at the partitioner
        # level, obviating the need to support them fully through the
        # torch.compile stack. To ensure that we have good enough debugging with
        # minifiers, we have ensure that they work with make_fx. This test uses
        # make_fx to do the testing. In future, we can move on torch.compile.
        def fn():
            rng_state1, a1 = torch._prims.rng_prims.run_and_save_rng_state(
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )
            rng_state2, a2 = torch._prims.rng_prims.run_and_save_rng_state(
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )

            b1 = torch._prims.rng_prims.run_with_rng_state(
                rng_state1,
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )
            b2 = torch._prims.rng_prims.run_with_rng_state(
                rng_state2,
                torch.ops.aten.rand.default,
                [4, 4],
                dtype=torch.float32,
                device=self.device,
            )

            return (a1, a2, b1, b2)

        mod = make_fx(fn)()
        compiled_f = compile_fx_inner(mod, ())
        a1, a2, b1, b2 = compiled_f(())
        self.assertEqual(a1, b1)
        self.assertEqual(a2, b2)

    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    @expectedFailureXPU
    @skip_if_gpu_halide  # rand
    @xfail_if_mps
    def test_philox_rand(self):
        if self.device == "cpu":
            raise unittest.SkipTest(
                f"functionalization of rng ops supported only on {GPU_TYPE}"
            )

        @torch.compile(backend="inductor")
        def fn(x):
            a = torch.rand_like(x) * x
            a = torch.rand_like(x) * a
            return a

        def check(x):
            torch.manual_seed(123)
            a = fn(x)

            torch.manual_seed(1234)
            b = fn(x)

            torch.manual_seed(123)
            c = fn(x)

            # same seed, same values
            self.assertTrue(torch.allclose(a, c))

            # different calls, different values
            self.assertFalse(torch.allclose(a, b))

        check(torch.ones(1024, device=self.device, dtype=torch.float32))
        # Need comment: should we add "_get_rng_state_offset" to common device interface?
        self.assertEqual(getattr(torch, self.device)._get_rng_state_offset(), 2048)
        # Check non-multiple of 4 numel
        check(torch.ones(3, device=self.device, dtype=torch.float32))
        self.assertEqual(getattr(torch, self.device)._get_rng_state_offset(), 8)

    # Already on by default, just want to make sure
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    def test_reuse_buffers_with_aliasing(self):
        def f(x):
            z = x + 1
            z = torch.view_as_complex(z)
            a = torch.view_as_real(z)
            out = a + 1
            return out, torch.view_as_real(z + 1)

        self.common(f, (torch.zeros((4, 2)),))

        code = run_and_get_triton_code(torch.compile(f), torch.zeros((4, 2)))
        # Make sure that we haven't added complex support and made this test
        # invalid. If we've added complex support please update the test to use
        # a different set of view ops we don't lower
        self.assertTrue("aten.view_as_real" in code)

        def f2(x):
            z = x + 1
            z = torch.view_as_complex(z)
            z = torch.view_as_real(z)
            z = torch.view_as_complex(z)
            a = torch.view_as_real(z)
            out = a + 1
            return out, torch.view_as_real(z + 1)

        self.common(f, (torch.zeros((4, 2)),))

    @xfail_if_triton_cpu  # libdevice.fma
    def test_softmax_backward_data(self):
        def fn(a, b):
            return aten._softmax_backward_data(a, b, dim=1, input_dtype=torch.float32)

        self.common(
            fn,
            (
                torch.randn(10, 10),
                torch.randn(10, 10),
            ),
        )

    def test_randn_like_empty(self):
        class Model(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, v1: torch.Tensor):
                vx = v1.min(dim=1).values
                v2 = torch.randn_like(vx)
                return v2

        model = Model()
        x = torch.rand(10, 3, 0)

        self.common(model, (x,), exact_stride=True)

    def test_randint(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return (
                torch.randint(10, [1024], device=x.device),
                torch.randint(-4, 7, [1024], dtype=torch.int32, device=x.device),
                torch.randint_like(x, 2**50),
            )

        torch.manual_seed(12345)
        a0, b0, c0 = fn(torch.zeros([40, 40], device=self.device))
        self.assertEqual(a0.shape, [1024])
        self.assertEqual(b0.shape, [1024])
        self.assertEqual(c0.shape, [40, 40])
        torch.manual_seed(12345)
        a1, b1, c1 = fn(torch.zeros([40, 40], device=self.device))
        self.assertEqual(a0, a1)
        self.assertEqual(b0, b1)
        self.assertEqual(c0, c1)

        self.assertEqual(a0.min(), 0)
        self.assertEqual(a0.max(), 9)

        self.assertEqual(b0.min(), -4)
        self.assertEqual(b0.max(), 6)

        self.assertGreaterEqual(c0.min(), 0)
        self.assertGreater(c0.max(), 2**40)
        self.assertLess(c0.max(), 2**50)

    def test_randint_distribution(self):
        @torch.compile(fullgraph=True)
        def fn(n_argsmax, size):
            return torch.randint(n_max, (size,), device=self.device)

        def bin(index, max_size):
            return index // (max_size // n_bins)

        size = 1_000_000
        n_max = int(0.75 * 2**32)
        n_bins = 8

        res = fn(n_max, size)
        bins = bin(res, n_max).float().cpu()
        hist, _ = bins.histogram(8, range=(0, n_bins))
        expected_bin = res.shape[0] / 8
        expected_error = math.sqrt(expected_bin) / expected_bin * 3
        error = (hist - expected_bin).abs().max() / expected_bin
        self.assertTrue(error < expected_error)

    @config.patch(fallback_random=True)
    @xfail_if_mps  # 100% are not close
    def test_like_rands(self):
        def fn(x):
            return torch.rand_like(x), torch.randn_like(x), torch.randint_like(x, 1, 11)

        self.common(fn, [torch.zeros([20, 20])], exact_stride=True)

    @config.patch(fallback_random=True)
    @xfail_if_mps  # 100% are not close
    def test_like_rands_sliced(self):
        def fn(x):
            return (
                torch.randn_like(x),
                torch.randn_like(x),
                torch.randint_like(x, 1, 11),
            )

        self.common(fn, (torch.zeros([3, 4])[:, ::2].permute(1, 0),), exact_stride=True)

    @config.patch(check_stack_no_cycles_TESTING_ONLY=True)
    def test_check_stack_no_cycles(self):
        if config.cpp_wrapper and self.device != "cpu":
            raise unittest.SkipTest(
                "codegen() gets called twice in cpp_wrapper GPU compilation, which "
                "causes this test to fail.  This can be removed if GPU compilation is "
                "done in a single pass."
            )

        @torch.compile()
        def fn(x):
            return x * 3

        r = fn(torch.randn(2, device=self.device, requires_grad=True))
        # Backward compilation isn't hooked into cprofile, it probably
        # should...
        # r.sum().backward()

    def test_like_rands2(self):
        # rand_like with kwargs `device` of str type
        d = self.device
        assert isinstance(d, str)

        @torch.compile
        def fn(x):
            return torch.rand_like(x, device=d)

        x = torch.ones(10, device=self.device, dtype=torch.float32)
        a0 = fn(x).clone()
        a1 = fn(x).clone()
        self.assertFalse(torch.allclose(a0, a1))
        self.assertEqual(a0.shape, a1.shape)
        self.assertEqual(a0.stride(), a1.stride())

    @requires_gpu()
    @skip_if_triton_cpu("Flaky on Triton CPU")
    def test_like_rands3(self):
        # rand_like with `device` which is different from `x.device`
        def test_like_rands_on_different_device(device1, device2):
            @torch.compile
            def fn(x, device):
                return torch.rand_like(x, device=device)

            x = torch.ones(10, device=device1, dtype=torch.float32)
            return fn(x, device2).clone()

        a0 = test_like_rands_on_different_device("cpu", GPU_TYPE)
        a1 = test_like_rands_on_different_device(GPU_TYPE, "cpu")
        self.assertTrue(a0.device.type == GPU_TYPE)
        self.assertTrue(a1.device.type == "cpu")
        self.assertEqual(a0.shape, a1.shape)
        self.assertEqual(a0.stride(), a1.stride())

    def test_max_pool2d_with_indices_backward(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [2, 2], [2, 2], [0, 0], [1, 1], False, c
            )

        x = torch.randn([2, 4, 18, 14])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [2, 2],
            [2, 2],
            [0, 0],
            [1, 1],
            False,
        )

        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    @xfail_if_mps  # Small tolerances bug
    @skip_if_gpu_halide  # slow
    def test_max_pool2d_with_indices_backward2(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [3, 3], [2, 2], [1, 1], [1, 1], True, c
            )

        x = torch.randn([2, 4, 40, 56])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [3, 3],
            [2, 2],
            [1, 1],
            [1, 1],
            True,
        )

        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    # From https://github.com/pytorch/torchdynamo/issues/1200
    def test_max_pool2d_with_indices_backward3(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [1, 1], [2, 2], [0, 0], [1, 1], False, c
            )

        x = torch.randn([32, 256, 37, 38])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [1, 1],
            [2, 2],
            0,
            1,
            False,
        )
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    # From https://github.com/pytorch/torchdynamo/issues/1352
    @xfail_if_mps  # Small tolerances bug
    @skip_if_halide  # hangs forever
    def test_max_pool2d_with_indices_backward4(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [5, 5], [1, 1], [2, 2], [1, 1], False, c
            )

        torch._inductor.metrics.generated_kernel_count = 0
        x = torch.randn([2, 64, 3, 4])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [5, 5],
            [1, 1],
            2,
            1,
            False,
        )
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )
        assertGeneratedKernelCountEqual(self, 1)

    @expectedFailureXPU
    def test_max_pool2d_with_indices_backward5(self):
        # Window size is too big. Should fallback
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [13, 13], [1, 1], [2, 2], [1, 1], False, c
            )

        torch._inductor.metrics.generated_kernel_count = 0
        x = torch.randn([2, 64, 20, 20])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [13, 13],
            [1, 1],
            2,
            1,
            False,
        )
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )
        assertGeneratedKernelCountEqual(self, 0)

    # From https://github.com/pytorch/pytorch/issues/93384
    def test_max_pool2d_with_indices_backward6(self):
        # dilation is not 1. Should fallback
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [3, 2], [2, 1], [1, 1], [1, 2], False, c
            )

        torch._inductor.metrics.generated_kernel_count = 0
        x = torch.randn([2, 2, 3, 6])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [3, 2],
            [2, 1],
            [1, 1],
            [1, 2],
            False,
        )
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )
        assertGeneratedKernelCountEqual(self, 0)

    def test_issue102546(self):
        def fn(x):
            return x.mean(0)

        self.common(fn, [torch.rand(())])

    def test_avg_pool2d_backward(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [2, 2],
                [2, 2],
                [0, 0],
                True,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([2, 4, 7, 7]),
                torch.randn([2, 4, 14, 14]),
            ],
        )

    @skip_if_gpu_halide  # slow
    def test_avg_pool2d_backward2(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [3, 3],
                [1, 1],
                [1, 1],
                True,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([1, 1, 20, 15]),
                torch.randn([1, 1, 20, 15]),
            ],
        )

    def test_avg_pool2d_backward3(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [1, 1],
                [2, 2],
                [0, 0],
                False,
                False,
                None,
            )

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            [
                torch.randn([1, 2016, 11, 11]),
                torch.randn([1, 2016, 21, 21]),
            ],
        )
        assertGeneratedKernelCountEqual(self, 1)

    def test_avg_pool2d_backward4(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [13, 13],
                [1, 1],
                [0, 0],
                True,
                False,
                None,
            )

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            [
                torch.randn([1, 16, 12, 12]),
                torch.randn([1, 16, 24, 24]),
            ],
            check_lowp=False,
        )
        assertGeneratedKernelCountEqual(self, 0)

    def test_avg_pool3d_backward(self):
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [2, 2, 2],
                [2, 2, 2],
                [0, 0, 0],
                True,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([2, 4, 7, 7, 7]),
                torch.randn([2, 4, 14, 14, 14]),
            ],
        )

    @skip_if_halide  # compiles for 5+ minutes
    def test_avg_pool3d_backward2(self):
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
                True,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([1, 1, 20, 20, 15]),
                torch.randn([1, 1, 20, 20, 15]),
            ],
        )

    def test_avg_pool3d_backward3(self):
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [1, 1, 1],
                [2, 2, 2],
                [0, 0, 0],
                False,
                False,
                None,
            )

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            [
                torch.randn([1, 2016, 11, 11, 11]),
                torch.randn([1, 2016, 21, 21, 21]),
            ],
        )
        assertGeneratedKernelCountEqual(self, 1)

    def test_avg_pool3d_backward4(self):
        def fn(a, b):
            return aten.avg_pool3d_backward(
                a,
                b,
                [13, 13, 13],
                [1, 1, 1],
                [0, 0, 0],
                True,
                False,
                None,
            )

        torch._inductor.metrics.generated_kernel_count = 0
        self.common(
            fn,
            [
                torch.randn([1, 16, 12, 12, 12]),
                torch.randn([1, 16, 24, 24, 24]),
            ],
            check_lowp=False,
        )
        assertGeneratedKernelCountEqual(self, 0)

    @config.patch(search_autotune_cache=False)
    def test_mm_views(self):
        def fn(a, b):
            return torch.mm(a.view(32, 32), b.view(32, 32))

        self.common(
            fn,
            (
                torch.randn([32, 32]).transpose(0, 1),
                torch.randn([1, 32, 32]).transpose(0, 1),
            ),
            check_lowp=False,
        )

        if (
            config.triton.native_matmul
            and config.cuda_backend == "triton"
            and self.device == "cuda"
        ):
            self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        else:
            # codegen mm kernel from template
            self.assertEqual(torch._inductor.metrics.generated_kernel_count, 0)



... [truncated: 6308 more lines]
```

*Note: Content truncated due to size*

## High-Level Overview


This Python file contains 26 class(es) and 1163 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCase`, `ToTuple`, `InputGen`, `SweepInputs2`, `skip_if_cpp_wrapper`, `CommonTemplate`, `Net`, `Model`, `Model`, `BatchNorm`, `M`, `ConvModel`, `ConvModel`, `Model`, `MyModel`, `MyModel`, `MyModel`, `Repro`, `ToComplex`, `Repro`

**Functions defined**: `_large_cumprod_input`, `define_custom_op_for_test`, `define_custom_op_2_for_test`, `define_custom_op_3_for_test`, `register_ops_with_aoti_compile`, `get_divisible_by_16`, `get_post_grad_graph`, `setUpClass`, `tearDownClass`, `setUp`, `tearDown`, `forward`, `dense`, `transposed`, `strided`, `broadcast1`, `broadcast2`, `broadcast3`, `double`, `int`

**Key imports**: contextlib, copy, dataclasses, functools, gc, importlib, itertools, math, operator, os


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `copy`
- `dataclasses`
- `functools`
- `gc`
- `importlib`
- `itertools`
- `math`
- `operator`
- `os`
- `random`
- `re`
- `subprocess`
- `sys`
- `threading`
- `time`
- `unittest`
- `unittest.mock`
- `weakref`
- `collections.abc`: Callable
- `pathlib`: Path
- `typing`: TypeVar
- `typing_extensions`: ParamSpec
- `numpy as np`
- `torch`
- `torch._dynamo.config as dynamo_config`
- `torch._inductor.aoti_eager`
- `torch.fx.traceback as fx_traceback`
- `torch.nn as nn`


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_torchinductor.py
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

- **File Documentation**: `test_torchinductor.py_docs.md`
- **Keyword Index**: `test_torchinductor.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
