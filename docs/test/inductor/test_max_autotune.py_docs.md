# Documentation: `test/inductor/test_max_autotune.py`

## File Metadata

- **Path**: `test/inductor/test_max_autotune.py`
- **Size**: 130,882 bytes (127.81 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import contextlib
import functools
import inspect
import json
import logging
import math
import os
import random
import re
import tempfile
import unittest
from collections.abc import Callable
from typing import Optional
from unittest import mock

import torch
from torch import multiprocessing as mp, nn
from torch._dynamo import reset
from torch._dynamo.exc import BackendCompilerFailed
from torch._dynamo.testing import rand_strided, reset_rng_state
from torch._dynamo.utils import counters, same
from torch._inductor import config
from torch._inductor.autotune_process import (
    _TestBenchmarkRequest,
    CUDA_VISIBLE_DEVICES,
    TuningProcess,
    TuningProcessPool,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import Buffer, ChoiceCaller, FixedLayout, FlexibleLayout
from torch._inductor.kernel.mm_plus_mm import aten_mm_plus_mm
from torch._inductor.select_algorithm import (
    add_feedback_saver,
    add_preprocessing_fn,
    AlgorithmSelectorCache,
    clear_feedback_savers,
    clear_preprocessing_fns,
    ExternKernelCaller,
    TritonTemplate,
    TritonTemplateCaller,
)
from torch._inductor.template_heuristics.registry import override_template_heuristics
from torch._inductor.template_heuristics.triton import (
    CUDAMMTemplateConfigHeuristic,
    GemmConfig,
)
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    NAVI_ARCH,
    parametrize,
    skipIfRocmArch,
    TEST_WITH_ROCM,
)
from torch.testing._internal.logging_utils import multiple_logs_to_string
from torch.utils._triton import (
    has_datacenter_blackwell_tma_device,
    has_triton_stable_tma_api,
    has_triton_tma_device,
)


aten = torch.ops.aten
from torch._inductor.mock_cache import global_stats, PatchCaches, Stats
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    fresh_cache,
    get_k_splits,
    run_and_get_code,
    use_decompose_k_choice,
)
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import MI300_ARCH, runOnRocmArch, skipIfXpu
from torch.testing._internal.inductor_utils import (
    get_func_call,
    get_kernel_launch,
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
)


torch.set_float32_matmul_precision("high")
if HAS_CUDA_AND_TRITON:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")


def benchmark_choice(choice, args, out, expected_out, timings):
    result = choice.benchmark(*args, out=out)
    if expected_out is not None:
        torch.testing.assert_close(out, expected_out)

    timings.copy_(torch.tensor(result))


class FailChoiceCaller(ChoiceCaller):
    def benchmark(self, *args, out):
        raise RuntimeError("This choice caller will always throw")


@unittest.mock.patch(
    "torch._inductor.select_algorithm.TritonTemplate.test_cache", new=True
)
@config.patch(enable_caching_generated_triton_templates=True)
@instantiate_parametrized_tests
class TestMaxAutotune(TestCase):
    @parametrize("dynamic", (False, True))
    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_max_autotune_mm_plus_mm_zero_size_input(self, dynamic, search_space):
        """
        Make sure autotuning mm_plus_mm with zero-size input works without crashes.
        """
        m, n, k = 0, 1536, 64

        def mm_plus_mm(a, b, c, d):
            return a @ b + c @ d

        a = torch.randn(m, k).to(GPU_TYPE)
        b = torch.randn(k, n).to(GPU_TYPE)
        c = torch.randn(m, k).to(GPU_TYPE)
        d = torch.randn(k, n).to(GPU_TYPE)

        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_search_space": search_space}
        ):
            torch.compile(mm_plus_mm, dynamic=dynamic)(a, b, c, d)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    def test_max_autotune_regular_mm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
    ):
        def mm(a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)

            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = (
            torch.randn(*((K, M) if a_transposed else (M, K)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        b = (
            torch.randn(*((N, K) if b_transposed else (K, N)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        if has_triton_stable_tma_api():
            make_desc_api = "triton.language.make_tensor_descriptor"
            read_api = "tl.load_tensor_descriptor"
            if tma_store:
                # Note: The tma_descriptor0 is generated by the kernel. If the
                # code generation process changes this could change.
                write_api = "tma_descriptor0.store"
            else:
                write_api = "tl.store"
        else:
            make_desc_api = (
                "triton.language.extra.cuda.experimental_device_tensormap_create2d"
            )
            read_api = "tl._experimental_descriptor_load"
            # TMA store is not supported with the experimental API
            write_api = "tl.store"

        # Verify that we are using a TMA implementation
        FileCheck().check("triton_tem_fused_mm").check(make_desc_api).check(
            read_api
        ).check(write_api).run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma_strided(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
    ):
        def mm(a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)
            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.mm(a, b)

        def next_multiple_16(a: int) -> int:
            return ((a + 15) // 16) * 16

        M, N, K = 21, 31, 11
        a_shape = (K, M) if a_transposed else (M, K)
        a_stride = (
            (next_multiple_16(M), 1) if a_transposed else (next_multiple_16(K), 1)
        )
        a = torch.empty_strided(a_shape, a_stride, dtype=torch.float16).to(GPU_TYPE)
        a[:] = torch.randn(a_shape, dtype=torch.float16)
        a = a.to(GPU_TYPE)
        b_shape = (N, K) if b_transposed else (K, N)
        b_stride = (
            (next_multiple_16(K), 1) if a_transposed else (next_multiple_16(N), 1)
        )
        b = torch.empty_strided(b_shape, b_stride, dtype=torch.float16)
        b[:] = torch.randn(b_shape, dtype=torch.float16)
        b = b.to(GPU_TYPE)
        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)
        # Verify that we are using a TMA implementation
        # depending on whether we're using the experimental API, we check for a different string
        check_str = "triton.language.extra.cuda.experimental_device_tensormap_create2d"
        if has_triton_stable_tma_api():
            check_str = "triton.language.make_tensor_descriptor"
        FileCheck().check("triton_tem_fused_mm").check(check_str).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    @parametrize("epilogue_subtile", (False, True))
    def test_blackwell_max_autotune_regular_mm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
        epilogue_subtile: bool,
    ):
        def mm(a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)
            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.mm(a, b)

        M, N, K = 32, 16, 48
        a = (
            torch.randn(*((K, M) if a_transposed else (M, K)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        b = (
            torch.randn(*((N, K) if b_transposed else (K, N)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "triton.enable_epilogue_subtiling": epilogue_subtile,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)
        write_count = 2 if epilogue_subtile else 1
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"
        FileCheck().check("triton_tem_fused_mm").check(
            "triton.language.make_tensor_descriptor"
        ).check("tl.load_tensor_descriptor").check_count(write_api, write_count).run(
            code[0]
        )

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @skipIfXpu(msg="TMA path on Intel GPU not require this check")
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma_illegal_alignment(self, dynamic):
        def mm(a, b):
            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(mm, dynamic=dynamic)(a, b)

        # Lowering to the persistent+TMA Triton template should be skipped
        # if any of the input inner dims are not 16-byte aligned. As a result,
        # given the config flags above, we should have no choices left.
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma_illegal_output_alignment(
        self, dynamic
    ):
        def mm(a, b, out):
            torch.mm(a, b, out=out)
            return out

        M, N, K = 21, 31, 32
        a = torch.empty_strided((M, K), (K, 1), dtype=torch.float16, device=GPU_TYPE)
        a[:] = torch.randn((M, K), dtype=torch.float16)
        b = torch.empty_strided((K, N), (1, K), dtype=torch.float16, device=GPU_TYPE)
        b[:] = torch.randn((K, N), dtype=torch.float16)
        # allocate an output with a stride not divisible by 16, so it can't satisfy TMA alignment checks.
        out = torch.empty_strided((M, N), (N, 1), dtype=torch.float16, device=GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "triton.enable_template_tma_store": True,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(mm, dynamic=dynamic)(a, b, out)

        # Lowering to the persistent+TMA Triton template should be skipped
        # since the output doesn't have a stride of 1 in any dim
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    def test_max_autotune_regular_mm_tma_dynamic_outer_dim(self):
        def mm(a, b):
            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte. All dims are
        # repeated due to the possible transpositions below.
        a = a.repeat(8, 8)
        b = b.repeat(8, 8)

        torch._dynamo.mark_dynamic(a, 0)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual = torch.compile(mm)(a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_zero_size_input(self, dynamic: bool):
        """
        Make sure autotuning mm with zero-size input works without crashes.
        """

        def mm(a, b):
            a = torch.sin(a)
            return a @ b

        a = torch.randn(0, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)

        with config.patch({"max_autotune": True}):
            torch.compile(mm, dynamic=dynamic)(a, b)

    # NOTE: the current Inductor template verifies that the scaling mode is either per-tensor or per-row
    # TODO: support additional scaling modes for Blackwell
    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    def test_blackwell_max_autotune_scaled_mm_per_tensor_persistent_tma(
        self,
        dynamic: bool,
        tma_store: bool,
    ):
        def scaled_mm(a, b, scale_a, scale_b):
            # NOTE: Inductor constrains a to be row_major and b to be col_major
            return torch._scaled_mm(
                a, b.t(), scale_a, scale_b, use_fast_accum=True, out_dtype=torch.float16
            )

        def get_scale_per_tensor(t):
            scale = torch.finfo(torch.float8_e4m3fn).max / t.abs().max()
            return scale.to(torch.float32)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte.
        M, N, K = 32, 16, 48
        a = (torch.randn((M, K)).to(torch.float16).to(GPU_TYPE)).repeat(8, 8)
        b = (torch.randn((N, K)).to(torch.float16).to(GPU_TYPE)).repeat(8, 8)

        scale_a = get_scale_per_tensor(a)
        scale_b = get_scale_per_tensor(b)

        a = a.to(torch.float8_e4m3fn)
        b = b.to(torch.float8_e4m3fn)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(scaled_mm, dynamic=dynamic), a, b, scale_a, scale_b
            )
            c_expected = scaled_mm(a, b, scale_a, scale_b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=0.5)
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"
        FileCheck().check("triton_tem_fused__scaled_mm").check(
            "triton.language.make_tensor_descriptor"
        ).check("tl.load_tensor_descriptor").check(write_api).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    def test_blackwell_max_autotune_scaled_mm_per_row_persistent_tma(
        self,
        dynamic: bool,
        tma_store: bool,
    ):
        def scaled_mm(a, b, scale_a, scale_b):
            # NOTE: Inductor constrains a to be row_major and b to be col_majo
            return torch._scaled_mm(
                a,
                b.t(),
                scale_a,
                scale_b.t(),
                use_fast_accum=True,
                out_dtype=torch.bfloat16,
            )

        def get_scale_per_row(t):
            scale = (
                torch.finfo(torch.float8_e4m3fn).max
                / t.abs().max(dim=1, keepdim=True).values
            )
            return scale.to(torch.float32)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte.
        M, N, K = 32, 16, 48
        a = (torch.randn((M, K)).to(torch.bfloat16).to(GPU_TYPE)).repeat(8, 8)
        b = (torch.randn((N, K)).to(torch.bfloat16).to(GPU_TYPE)).repeat(8, 8)

        scale_a = get_scale_per_row(a)
        scale_b = get_scale_per_row(b)

        a = a.to(torch.float8_e4m3fn)
        b = b.to(torch.float8_e4m3fn)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(scaled_mm, dynamic=dynamic), a, b, scale_a, scale_b
            )
            c_expected = scaled_mm(a, b, scale_a, scale_b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=0.5)
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"
        FileCheck().check("triton_tem_fused__scaled_mm").check(
            "triton.language.make_tensor_descriptor"
        ).check("tl.load_tensor_descriptor").check(write_api).run(code[0])

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    def test_max_autotune_addmm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
    ):
        def addmm(x, a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            x = x.repeat(8)
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)

            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = (
            torch.randn(*((K, M) if a_transposed else (M, K)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        b = (
            torch.randn(*((N, K) if b_transposed else (K, N)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(addmm, dynamic=dynamic), x, a, b
            )
            c_expected = addmm(x, a, b)

        if has_triton_stable_tma_api():
            make_desc_api = "triton.language.make_tensor_descriptor"
            read_api = "tl.load_tensor_descriptor"
            if tma_store:
                # Note: The tma_descriptor0 is generated by the kernel. If the
                # code generation process changes this could change.
                write_api = "tma_descriptor0.store"
            else:
                write_api = "tl.store"
        else:
            make_desc_api = (
                "triton.language.extra.cuda.experimental_device_tensormap_create2d"
            )
            read_api = "tl._experimental_descriptor_load"
            # TMA store is not supported with the experimental API
            write_api = "tl.store"

        # Verify that we are using a TMA implementation
        FileCheck().check("triton_tem_fused_addmm").check(make_desc_api).check(
            read_api
        ).check(write_api).run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    @parametrize("epilogue_subtile", (False, True))
    def test_blackwell_max_autotune_addmm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
        epilogue_subtile: bool,
    ):
        def addmm(x, a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            x = x.repeat(8)
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)

            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = (
            torch.randn(*((K, M) if a_transposed else (M, K)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        b = (
            torch.randn(*((N, K) if b_transposed else (K, N)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "triton.enable_epilogue_subtiling": epilogue_subtile,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(addmm, dynamic=dynamic), x, a, b
            )
            c_expected = addmm(x, a, b)

        make_desc_api = "triton.language.make_tensor_descriptor"
        read_api = "tl.load_tensor_descriptor"
        write_count = 2 if epilogue_subtile else 1
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"

        # Verify that we are using a TMA implementation
        FileCheck().check("triton_tem_fused_addmm").check(make_desc_api).check(
            read_api
        ).check_count(write_api, write_count).run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @skipIfXpu(msg="TMA path on Intel GPU not require this check")
    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_persistent_tma_illegal_alignment(self, dynamic):
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(addmm, dynamic=dynamic)(x, a, b)

        # Lowering to the persistent+TMA Triton template should be skipped
        # if any of the input inner dims are not 16-byte aligned. As a result,
        # given the config flags above, we should have no choices left.
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    def test_max_autotune_addmm_tma_dynamic_outer_dim(self):
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte. All dims are
        # repeated due to the possible transpositions below.
        x = x.repeat(8)
        a = a.repeat(8, 8)
        b = b.repeat(8, 8)

        torch._dynamo.mark_dynamic(a, 0)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual = torch.compile(addmm)(x, a, b)
            c_expected = addmm(x, a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @fresh_cache()
    @skipIfXpu(msg="XPU doesn't support sm carveout")
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support sm carveout")
    @unittest.skipIf(IS_WINDOWS, "Windows doesn't support persistent TMA")
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        has_datacenter_blackwell_tma_device(), "B200 doesn't support sm carveout"
    )
    @parametrize("carveout", (None, 0, 27))
    @parametrize("op", ("mm", "scaled_mm"))
    def test_honor_sm_carveout_with_triton_tma(self, carveout, op: str):
        def mm_func(a, b):
            return torch.mm(a, b)

        def scaled_mm(
            a,
            b,
            scale_a,
            scale_b,
        ):
            return torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)

        # Create large matrices to ensure we use all possible sms
        size = 2560
        a = torch.randn(size, size, device=GPU_TYPE, dtype=torch.bfloat16)
        b = (
            torch.randn(size, size, device=GPU_TYPE, dtype=torch.bfloat16)
            .transpose(0, 1)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.tensor(1, dtype=torch.float32, device=GPU_TYPE)
        scale_b = torch.tensor(1, dtype=torch.float32, device=GPU_TYPE)

        args = (
            (a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn), scale_a, scale_b)
            if op == "scaled_mm"
            else (a, b)
        )
        func = scaled_mm if op == "scaled_mm" else mm_func

        # Set the specified carveout value
        torch._C._set_sm_carveout_experimental(carveout)
        if carveout is None:
            self.assertIsNone(torch._C._get_sm_carveout_experimental())
        else:
            self.assertEqual(torch._C._get_sm_carveout_experimental(), carveout)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.native_matmul": False,
                "max_autotune_gemm_backends": "TRITON",
                "test_configs.autotune_choice_name_regex": "tma",
            }
        ):
            compiled_mm = torch.compile(func, mode="max-autotune-no-cudagraphs")
            compiled_mm(*args)  # Warm-up compilation

            with tempfile.NamedTemporaryFile() as f:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    # Run with the specified carveout
                    compiled_mm(*args)

                # Export trace and analyze results
                prof.export_chrome_trace(f.name)

                # Extract grid sizes from the trace events for TMA kernels
                kernel_name = "triton_tem_fused"
                with open(f.name) as file:
                    kernel_events = [
                        {
                            "grid": evt.get("args", {}).get("grid", []),
                            "grid_size": math.prod(evt.get("args", {}).get("grid", [])),
                        }
                        for evt in json.load(file)["traceEvents"]
                        if evt.get("cat", "") == "kernel"
                        and kernel_name in evt.get("name", "").lower()
                    ]

                # We should have exactly 1 kernel event for this run
                self.assertEqual(
                    len(kernel_events),
                    1,
                    f"Expected exactly 1 kernel event, but got {len(kernel_events)}",
                )

                # Check that grid size matches expected values based on carveout
                expected_grid_size = None
                max_grid_size = torch.cuda.get_device_properties(
                    "cuda"
                ).multi_processor_count
                careveout = 0 if carveout is None else carveout
                expected_grid_size = max_grid_size - careveout

                self.assertEqual(
                    kernel_events[0]["grid_size"],
                    expected_grid_size,
                    f"Grid size {kernel_events[0]['grid_size']} doesn't match {expected_grid_size} for carveout={carveout}",
                )

    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_zero_size_input(self, dynamic):
        """
        Make sure autotuning addmm with zero-size input works without crashes.
        """

        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        x = torch.randn(100).to(GPU_TYPE)
        a = torch.randn(0, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)
        with config.patch({"max_autotune": True}):
            torch.compile(addmm, dynamic=dynamic)(x, a, b)

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_autotune_conv1x1(self, search_space):
        # Assuming input has 3 channels and we want to produce 16 channels as output
        conv1x1 = (
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
            .to(memory_format=torch.channels_last)
            .to(GPU_TYPE)
        )

        # Example input tensor: batch size = 4, channels = 3, height = 32, width = 32
        # The memory format is set to `channels_last`
        input_tensor = (
            torch.randn(4, 3, 32, 32)
            .contiguous(memory_format=torch.channels_last)
            .to(GPU_TYPE)
        )

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune_gemm_search_space": search_space,
            }
        ):

            @torch.compile()
            def foo(mod, x):
                return mod(x)

            with torch.no_grad():
                out, code = run_and_get_code(foo, conv1x1, input_tensor)

            FileCheck().check_not("extern_kernels.convolution").run(code[0])
            self.assertEqual(conv1x1(input_tensor), out, atol=1e-2, rtol=0)

    @fresh_cache()
    @config.patch(max_autotune=True, max_fusion_size=2)
    def test_jit_fusion_matches_aot_fusion(self):
        # In this example, AOTInductor's JIT-compile will fuse(buf1, buf2) due
        # to proximity, we want to make sure AOT-compile pass does the same.
        # AOT could do fuse(buf2, buf4) instead if buf3 was pushed to the end
        # of the V.graph.buffers list because fuse(buf2, buf4) would have a
        # better proximity score than fuse(buf1, buf2). This scenario is possible
        # since finalizing MultiTemplateBuffers needs to replace buffers.
        def fn(x, number):
            buf0 = x + x
            buf1 = number.item()
            buf2 = x * x
            buf3 = x @ x  # MultiTemplateBuffer
            buf4 = x**2
            return buf0, buf1, buf2, buf3, buf4

        inputs = (
            torch.rand([256, 256], device=GPU_TYPE),
            torch.tensor(3, device=GPU_TYPE),
        )
        torch._export.aot_compile(fn, args=inputs)

    def test_cat_addmm(self):
        def fn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
            return torch.cat(
                [
                    torch.addmm(a, b, c),
                    torch.addmm(b, c, a),
                ],
                1,
            )

        args = [
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        ]
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @config.patch(
        benchmark_kernel=True,
        fallback_random=True,
        max_autotune_gemm=True,
    )
    @parametrize("device", ("cpu", GPU_TYPE))
    def test_matmul_dropout(self, device):
        def fwd(a, b):
            x = a @ b
            x = torch.nn.functional.dropout(x, 0.1)
            return x

        def fn(a, b):
            x = fwd(a, b).sum()
            x.backward()
            return a.grad

        N = 128
        a = torch.randn(N, N, device=device, requires_grad=True)
        b = torch.randn(N, N, device=device)

        opt_fn = torch.compile(fn)
        reset_rng_state()
        ref = fn(a, b)
        reset_rng_state()
        act = opt_fn(a, b)

        if N <= 8:
            print(f"ref\n{ref}\nact\n{act}")
        torch.testing.assert_close(ref, act, atol=1e-1, rtol=1e-1)

    @config.patch(
        max_autotune_gemm=True,
    )
    @unittest.skipIf(
        getattr(torch, GPU_TYPE).device_count() < 2,
        "Need at least 2 devices for this test",
    )
    def test_autotune_device_guard(self):
        x = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")
        y = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")

        def f(x, y):
            return x @ y

        with fresh_cache():
            act = torch.compile(f)(x, y)
        ref = f(x, y)
        self.assertTrue(torch.allclose(act, ref, atol=4 * 1e-3, rtol=4 * 1e-3))

    @config.patch(max_autotune=True)
    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    @parametrize("kernel_size", (1, 3))
    def test_empty_conv_input(self, search_space, kernel_size):
        x = torch.randn(0, 256, 14, 14, device=GPU_TYPE)
        weight = torch.randn(256, 256, kernel_size, kernel_size, device=GPU_TYPE)

        def f(x, weight):
            return torch.convolution(
                x,
                weight,
                bias=None,
                stride=[1, 1],
                padding=[0, 0],
                dilation=[1, 1],
                transposed=False,
                output_padding=[0, 0],
                groups=1,
            )

        with config.patch({"max_autotune_gemm_search_space": search_space}):
            opt_f = torch.compile(f)
            ref = f(x, weight)
            act = opt_f(x, weight)
            self.assertTrue(torch.allclose(ref, act, atol=4 * 1e-3, rtol=4 * 1e-3))

    @skipIfXpu(
        msg="Fails on Intel XPU; see https://github.com/pytorch/pytorch/issues/161484"
    )
    @config.patch(max_autotune_gemm_backends="TRITON")
    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_baddmm(self, search_space):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(64, 64, 192, dtype=torch.float16)
                )
                self.bias = torch.nn.Parameter(
                    torch.randn(64, 1, 192, dtype=torch.float16)
                )

            def forward(self, x):
                return torch.ops.aten.baddbmm.default(self.bias, x, self.weight)

        x = torch.randn(
            64, 2048, 64, dtype=torch.float16, requires_grad=False, device=GPU_TYPE
        )
        mod = M().to(GPU_TYPE)

        with config.patch({"max_autotune_gemm_search_space": search_space}):
            m_c = torch.compile(mode="max-autotune")(mod)
            out, code = run_and_get_code(m_c, x)
            self.assertEqual(out, mod(x), atol=2e-3, rtol=2e-3)

            if not config.triton.native_matmul:
                FileCheck().check("triton_tem_fused_baddbmm").run(code[0])

    @config.patch(max_autotune=True)
    def test_conv1x1_with_free_symbols(self):
        """
        Make sure there is no exception due to free symbols.
        """
        conv = nn.Conv2d(
            3, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        ).to(device=GPU_TYPE)

        @torch.compile
        def f(x, y, z):
            h = y.nonzero().size(0)
            w = z.nonzero().size(0)
            x = x[:, :, :h, :w]
            x = conv(x)
            return x

        x = torch.randn(4, 3, 224, 224).to(
            memory_format=torch.channels_last, device=GPU_TYPE
        )
        for _ in range(2):
            y = torch.randint(0, 10, (224,)).to(device=GPU_TYPE)
            z = torch.randint(0, 10, (224,)).to(device=GPU_TYPE)
            f(x, y, z)

    def _test_cat_max_autotune_impl(self, using_triton_mm):
        def f(x, y):
            y = torch.cos(y)
            x = torch.mm(x, x)
            return torch.cat([x, y])

        f_c = torch.compile(mode="max-autotune-no-cudagraphs")(f)
        inps = [
            torch.randn(32, 32, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
        ]
        _, code = run_and_get_code(f_c, inps[0], inps[1])
        self.assertEqual(f_c(*inps), f(*inps), atol=0.03, rtol=0.25)

        # mm kernel, and cos kernel
        count = 2 if (using_triton_mm or config.triton.native_matmul) else 1
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), count, exactly=True
        ).run(code[0])

        def f(x, y):
            y = torch.cos(y)
            x = torch.mm(x, x)
            out = torch.cat([x, y])
            return out, x + 1

        f_c = torch.compile(mode="max-autotune-no-cudagraphs")(f)
        _, code = run_and_get_code(f_c, inps[0], inps[1])
        self.assertEqual(f_c(*inps), f(*inps), atol=0.03, rtol=0.25)
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), 2, exactly=True
        ).run(code[0])

        def f(x, y):
            y = torch.cos(y)
            x = torch.mm(x, x)
            return torch.cat([x, y]), torch.cat([y, x])

        f_c = torch.compile(mode="max-autotune-no-cudagraphs")(f)
        self.assertEqual(f_c(*inps), f(*inps), atol=0.03, rtol=0.25)

    @config.patch("trace.enabled", True)
    @config.patch({"test_configs.force_extern_kernel_in_multi_template": True})
    @config.patch("triton.native_matmul", False)
    def test_mutation_rename(self):
        torch._logging.set_logs(ir_post_fusion=True)

        def f(x, y, z, other):
            mul = x * y
            diag = torch.diagonal(mul)
            diag.copy_(other)
            x = torch.mm(mul, z)
            y = torch.diagonal(x).add_(torch.tensor(1, device=GPU_TYPE))
            return y

        t = functools.partial(torch.randn, device=GPU_TYPE)
        inps = (t(3, 3), t(3, 3), t(3, 3), t(3))
        fn = torch.compile(f, mode="max-autotune-no-cudagraphs")

        (
            (
                pre_fusion_tream,
                post_fusion_stream,
            ),
            ctx,
        ) = multiple_logs_to_string(
            "torch._inductor.debug", "ir_pre_fusion", "ir_post_fusion"
        )

        with config.patch({"trace.debug_dir": tempfile.mkdtemp()}):
            with (
                self.assertLogs(
                    logging.getLogger("torch._inductor.debug"), level=logging.INFO
                ) as cm,
                ctx(),
            ):
                out = fn(*inps)

        self.assertEqual(f(*inps), out)

        pre_fusion_stream = cm.output[0]
        post_fusion_stream = cm.output[1]

        # before and after finalizing multi template buffer, deps should have the same normalization
        # wrt writes
        FileCheck().check("MultiTemplateBuffer").check("unmet").check_same("buf1").run(
            pre_fusion_stream
        )
        FileCheck().check("ExternKernelSchedulerNode").check("unmet").check_same(
            "buf1"
        ).run(post_fusion_stream)

        torch._logging.set_logs()

    @config.patch({"test_configs.force_extern_kernel_in_multi_template": True})
    def test_cat_max_autotune_extern(self):
        self._test_cat_max_autotune_impl(using_triton_mm=False)

    @skipIfXpu(
        msg="The fusion not happened because it do not speedup on XPU, see issue #146568"
    )
    @config.patch(
        {
            "max_autotune_gemm_backends": "TRITON",
            "benchmark_epilogue_fusion": False,
        }
    )
    def test_cat_max_autotune_triton(self):
        self._test_cat_max_autotune_impl(using_triton_mm=True)

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_conv_cat(self, search_space):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )

            def forward(self, x):
                x = self.conv(x)
                return torch.cat((x, x + 1))

        with config.patch({"max_autotune_gemm_search_space": search_space}):
            with torch.no_grad():
                m = ToyModel().to(device=GPU_TYPE)
                input_tensor = torch.randn(32, 3, 64, 64).to(device=GPU_TYPE)

                # convolution is not currently plannable
                m = torch.compile(m, mode="max-autotune-no-cudagraphs")
                out, code = run_and_get_code(m, input_tensor)
                self.assertEqual(out, m(input_tensor))

                if not TEST_WITH_ROCM:
                    FileCheck().check("def triton_poi_fused_add_cat_").run(code[0])

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_conv3d(self, search_space):
        fn = torch.nn.functional.conv3d
        image = torch.randn([1, 3, 8, 16, 32])
        filt = torch.randn([3, 3, 7, 7, 7])

        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_search_space": search_space}
        ):
            expected = fn(image, filt)
            actual = torch.compile(fn)(image, filt)
            torch.testing.assert_close(actual, expected, atol=6e-5, rtol=0.001)

    @config.patch(
        max_autotune=True, max_autotune_conv_backends="", layout_optimization=False
    )
    def test_conv_backend(self):
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),
        ).to(GPU_TYPE)
        inp = torch.randn([2, 3, 16, 16]).to(GPU_TYPE)

        with self.assertRaises(BackendCompilerFailed) as context:
            torch.compile(m)(inp)

        self.assertIn("NoValidChoicesError", str(context.exception))

    @skipIfRocmArch(NAVI_ARCH)
    def test_non_contiguous_input_mm(self):
        """
        Make sure the triton template can work with non-contiguous inputs without crash.
        Check https://github.com/pytorch/pytorch/issues/125437 for more details.
        """
        x = rand_strided(
            (50257, 2048), (1, 50304), dtype=torch.bfloat16, device=GPU_TYPE
        )
        y = rand_strided((2048, 768), (768, 1), dtype=torch.bfloat16, device=GPU_TYPE)

        @torch.compile(mode="max-autotune")
        def f(x, y):
            return x @ y

        ref = x @ y
        act = f(x, y)
        torch.testing.assert_close(act, ref, atol=2e-2, rtol=1e-2)

    @skipIfRocmArch(NAVI_ARCH)
    def test_non_contiguous_input_addmm(self):
        b = torch.randn((768), dtype=torch.bfloat16, device=GPU_TYPE)
        x = rand_strided(
            (50257, 2048), (1, 50304), dtype=torch.bfloat16, device=GPU_TYPE
        )
        y = rand_strided((2048, 768), (768, 1), dtype=torch.bfloat16, device=GPU_TYPE)

        @torch.compile(mode="max-autotune")
        def f(x, y):
            return torch.addmm(b, x, y)

        ref = torch.addmm(b, x, y)
        act = f(x, y)
        torch.testing.assert_close(act, ref, atol=2e-2, rtol=1e-2)

    @skipIfRocmArch(NAVI_ARCH)
    def test_non_contiguous_input_bmm(self):
        x = rand_strided(
            (1, 50257, 2048), (0, 1, 50304), dtype=torch.bfloat16, device=GPU_TYPE
        )
        y = rand_strided(
            (1, 2048, 768), (0, 768, 1), dtype=torch.bfloat16, device=GPU_TYPE
        )

        @torch.compile(mode="max-autotune")
        def f(x, y):
            return torch.bmm(x, y)

        ref = torch.bmm(x, y)
        act = f(x, y)
        torch.testing.assert_close(act, ref, atol=2e-2, rtol=1e-2)

    # TODO: fix accuracy failure of the triton template on XPU.
    # and enable this test case.
    @skipIfXpu
    @unittest.skipIf(
        config.triton.native_matmul,
        "native matmul and Triton template both have accuracy fail (2.2%)",
    )
    def test_non_contiguous_input_mm_plus_mm(self):
        x1 = rand_strided((50257, 2048), (1, 50304), device=GPU_TYPE)
        y1 = rand_strided((2048, 768), (768, 1), device=GPU_TYPE)

        x2 = rand_strided((50257, 2048), (1, 50304), device=GPU_TYPE)
        y2 = rand_strided((2048, 768), (768,
```



## High-Level Overview


This Python file contains 13 class(es) and 222 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FailChoiceCaller`, `TestMaxAutotune`, `M`, `ToyModel`, `TestMaxAutotunePrecompile`, `FakeChoiceCaller`, `TestMaxAutotuneSubproc`, `TestMaxAutotuneRemoteCache`, `Model`, `_TestTritonTemplateCaller`, `TestTuningProcess`, `TestTuningProcessPool`, `TestPrologueFusion`

**Functions defined**: `benchmark_choice`, `benchmark`, `test_max_autotune_mm_plus_mm_zero_size_input`, `mm_plus_mm`, `test_max_autotune_regular_mm_persistent_tma`, `mm`, `test_max_autotune_regular_mm_persistent_tma_strided`, `mm`, `next_multiple_16`, `test_blackwell_max_autotune_regular_mm_persistent_tma`, `mm`, `test_max_autotune_regular_mm_persistent_tma_illegal_alignment`, `mm`, `test_max_autotune_regular_mm_persistent_tma_illegal_output_alignment`, `mm`, `test_max_autotune_regular_mm_tma_dynamic_outer_dim`, `mm`, `test_max_autotune_regular_mm_zero_size_input`, `mm`, `test_blackwell_max_autotune_scaled_mm_per_tensor_persistent_tma`

**Key imports**: contextlib, functools, inspect, json, logging, math, os, random, re, tempfile


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `functools`
- `inspect`
- `json`
- `logging`
- `math`
- `os`
- `random`
- `re`
- `tempfile`
- `unittest`
- `collections.abc`: Callable
- `typing`: Optional
- `torch`
- `torch._dynamo`: reset
- `torch._dynamo.exc`: BackendCompilerFailed
- `torch._dynamo.testing`: rand_strided, reset_rng_state
- `torch._dynamo.utils`: counters, same
- `torch._inductor`: config
- `torch._inductor.graph`: GraphLowering
- `torch._inductor.ir`: Buffer, ChoiceCaller, FixedLayout, FlexibleLayout
- `torch._inductor.kernel.mm_plus_mm`: aten_mm_plus_mm
- `torch._inductor.template_heuristics.registry`: override_template_heuristics
- `torch.testing._internal.common_cuda`: PLATFORM_SUPPORTS_FP8
- `torch.testing._internal.logging_utils`: multiple_logs_to_string
- `torch._inductor.mock_cache`: global_stats, PatchCaches, Stats
- `torch._inductor.test_case`: run_tests, TestCase
- `torch._inductor.virtualized`: V


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_max_autotune.py
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

- **File Documentation**: `test_max_autotune.py_docs.md`
- **Keyword Index**: `test_max_autotune.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
