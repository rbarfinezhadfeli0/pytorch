# Documentation: test_torchinductor.py

## File Metadata
- **Path**: `test/inductor/test_torchinductor.py`
- **Size**: 545214 bytes
- **Lines**: 16308
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
            return (torch.exp(a), 

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 60 class(es): TestCase, ToTuple, class, SweepInputs2, skip_if_cpp_wrapper, CommonTemplate, Net, Model, Model, BatchNorm, M, ConvModel, ConvModel, Model, MyModel, MyModel, MyModel, Repro, ToComplex, Repro

### Functions
This file defines 1722 function(s): _large_cumprod_input, define_custom_op_for_test, define_custom_op_2_for_test, define_custom_op_3_for_test, register_ops_with_aoti_compile, get_divisible_by_16, get_post_grad_graph, setUpClass, tearDownClass, setUp, tearDown, forward, dense, transposed, strided, broadcast1, broadcast2, broadcast3, double, int, compute_grads, gather_leaf_tensors, check_model, upcast_fn, compile_fx_wrapper, run, reference_to_expect, custom_assert_with_self, check_model_gpu, downcast_fn


## Key Components

The file contains 42722 words across 16308 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 545214 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
