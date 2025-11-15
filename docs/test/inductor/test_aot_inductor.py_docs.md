# Documentation: `test/inductor/test_aot_inductor.py`

## File Metadata

- **Path**: `test/inductor/test_aot_inductor.py`
- **Size**: 288,299 bytes (281.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import itertools
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest import skip
from unittest.mock import patch

import torch
import torch._export
import torch._inductor
import torch._inductor.config
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torch.nn as nn
from torch._dynamo import config as dynamo_config
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided, same
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codecache import WritableTempFile
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.package import package_aoti
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import TestCase
from torch._inductor.utils import (
    is_big_gpu,
    maybe_aoti_standalone_config,
    run_and_get_cpp_code,
)
from torch._library import capture_triton
from torch._utils_internal import full_aoti_runtime_assert
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.export import Dim, export
from torch.export.pt2_archive._package import load_pt2
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import (
    _get_torch_cuda_version,
    CDNA2OrLater,
    IS_SM90,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    SM80OrLater,
    tf32_on_and_off,
)
from torch.testing._internal.common_device_type import (
    _has_sufficient_memory,
    e4m3_type,
    skipCUDAIf,
)
from torch.testing._internal.common_quantization import (
    _group_quantize_tensor,
    skip_if_no_torchvision,
    skipIfNoFBGEMM,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    IS_CI,
    IS_FBCODE,
    IS_MACOS,
    IS_WINDOWS,
    MACOS_VERSION,
    MI300_ARCH,
    parametrize,
    runOnRocm,
    skipIfMPS,
    skipIfRocm,
    skipIfRocmArch,
    skipIfWindows,
    skipIfWindowsXPU,
    skipIfXpu,
    TEST_MPS,
    TEST_WITH_ROCM,
)
from torch.testing._internal.custom_tensor import CustomTensorPlainOut
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    HAS_XPU_AND_TRITON,
    IS_BIG_GPU,
)
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.testing._internal.triton_utils import requires_gpu
from torch.utils import _pytree as pytree
from torch.utils._triton import (
    has_triton_experimental_host_tma,
    has_triton_tensor_descriptor_host_tma,
)


if HAS_GPU:
    import triton  # @manual
    from triton import language as tl

    from torch.testing._internal.triton_utils import (
        add_kernel,
        add_kernel_2d_autotuned,
        add_kernel_autotuned,
        add_kernel_autotuned_weird_param_order,
        add_kernel_on_device_tma_new_api,
        add_kernel_on_device_tma_old_api,
        add_kernel_with_boolean_param,
        add_kernel_with_none_param_and_equal_to_1_arg,
        add_kernel_with_optional_param,
        add_kernel_with_scaling,
        add_kernel_with_tma_1d_new_api,
        add_kernel_with_tma_1d_old_api,
        add_kernel_with_tma_2d_new_api,
        add_kernel_with_tma_2d_old_api,
        create_tensor_descriptor_shim,
        mul2_inplace_kernel,
        strange_config_matmul_kernel,
        sub_kernel_autotuned,
    )

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_aot_inductor_utils import (
            AOTIRunnerUtil,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from .test_control_flow import (
            CondModels,
            prepend_counters,
            prepend_predicates,
            WhileLoopModels,
        )
        from .test_torchinductor import copy_tests, requires_multigpu, TestFailure
    except ImportError:
        from test_aot_inductor_utils import (  # @manual=fbcode//caffe2/test/inductor:aot_inductor_utils-library
            AOTIRunnerUtil,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from test_control_flow import (  # @manual=fbcode//caffe2/test/inductor:control_flow-library
            CondModels,
            prepend_counters,
            prepend_predicates,
            WhileLoopModels,
        )
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            copy_tests,
            requires_multigpu,
            TestFailure,
        )
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)
    raise


def get_module_ext_type():
    if IS_WINDOWS:
        return "pyd"
    else:
        return "so"


class AOTInductorTestsTemplate:
    # Temporarily skipping test as pytorch/cpuinfo not able to retrieve cache size for
    # AMD EPYC 9575F 64-Core Processor CPU in gfx942 VM Runners
    @common_utils.parametrize("embed_kernel_binary", [False, True])
    @common_utils.parametrize("max_autotune", [False, True])
    @skipIfRocmArch(MI300_ARCH)
    def test_simple(self, embed_kernel_binary, max_autotune):
        if self.device == "cpu" and IS_MACOS and max_autotune:
            raise unittest.SkipTest("max_autotune not supported on macos")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model()
        with config.patch(
            {
                "aot_inductor.embed_kernel_binary": embed_kernel_binary,
                "max_autotune": max_autotune,
            }
        ):
            self.check_model(model, example_inputs)

            _, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )
            if self.device == "mps":
                FileCheck().check("aoti_torch_mps_get_kernel_function(").run(code)
            elif self.device == GPU_TYPE:
                FileCheck().check("launchKernel(").run(code)
                if config.aot_inductor.embed_kernel_binary:
                    # Not expect to see launchKernel("CUBIN_FILE_NAME"
                    FileCheck().check_not('launchKernel("').run(code)

        if self.use_minimal_arrayref_interface:
            self.code_check_count(
                model, example_inputs, "AOTInductorModelRunMinimalArrayrefInterface(", 1
            )

    def test_triton_kernel_bool_param(self):
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def forward(self, x):
                out = torch.zeros_like(x)
                add_kernel_with_boolean_param[1,](
                    in_ptr0=x,
                    in_ptr1=x,
                    out_ptr=out,
                    n_elements=x.numel(),
                    add_xy=True,
                    BLOCK_SIZE=1,
                )
                return out

        inputs = (torch.randn(4, device=self.device),)
        self.check_model(Model(), inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "toolchain doesn't support ptx to fatbin",
    )
    @skipIfMPS
    # Skip embed_kernel_binary == True for now as it shows random
    # failure on CI
    @common_utils.parametrize("embed_kernel_binary", [False])
    @unittest.skipIf(
        torch.version.hip is None and _get_torch_cuda_version() < (12, 6),
        "Test is only supported on CUDA 12.6+",
    )
    def test_simple_multi_arch(self, embed_kernel_binary):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU_TYPE")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 16)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 16, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model()
        with config.patch(
            {
                "aot_inductor.embed_kernel_binary": embed_kernel_binary,
                "aot_inductor.emit_multi_arch_kernel": True,
            }
        ):
            self.check_model(model, example_inputs)
            if not embed_kernel_binary:
                _, code = run_and_get_cpp_code(
                    AOTIRunnerUtil.compile, model, example_inputs
                )
                file_extension = (
                    ".spv"
                    if self.device == "xpu"
                    else (".hsaco" if torch.version.hip else ".fatbin")
                )
                FileCheck().check(file_extension).run(code)

    def test_small_constant(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"always_keep_tensor_constants": True}):
            self.check_model(Model().to(self.device), example_inputs)

    def test_output_path_1(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch("aot_inductor.output_path", "tmp_output_"):
            self.check_model(Model(), example_inputs)

    def test_output_path_2(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        model = Model().to(device=self.device)
        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        expected_path = normalize_path_separator(
            os.path.join(
                tempfile.mkdtemp(dir=cache_dir()), f"model.{get_module_ext_type()}"
            )
        )
        actual_path = AOTIRunnerUtil.legacy_compile(
            model, example_inputs, options={"aot_inductor.output_path": expected_path}
        )
        self.assertTrue(actual_path == expected_path)

    @unittest.skipIf(
        config.triton.native_matmul,
        "different # of input/output/constants in native matmul",
    )
    def test_empty_constant_folding(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                return torch.matmul(x, self.w) + self.b

        model = Model(self.device)
        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            so_path, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )
            # We should have 1 input, 1 output, 2 constants for the model.
            FileCheck().check_count("AOTInductorModelBase(1,", 1).check_next(
                "1,"
            ).check_next("2,").run(code)

    def test_constant_folding(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_constant_folding_with_update(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with (
            torch.no_grad(),
            config.patch(
                {
                    "always_keep_tensor_constants": True,
                    "aot_inductor.use_runtime_constant_folding": True,
                }
            ),
        ):
            model = Model(self.device)
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(4, 4, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        # Update with new weights on active buffer
        new_weights = {
            "L__self___b": torch.randn(4, device=self.device),
            "L__self___w_pre": torch.randn(4, 4, device=self.device),
        }
        model.w_pre = new_weights["L__self___w_pre"]
        model.b = new_weights["L__self___b"]
        expected = model(test_inputs)
        runner.update_constant_buffer(new_weights, False, False)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        # Update with new weights on inactive buffer
        new_weights = {
            "L__self___b": torch.randn(4, device=self.device),
            "L__self___w_pre": torch.randn(4, 4, device=self.device),
        }
        model.w_pre = new_weights["L__self___w_pre"]
        model.b = new_weights["L__self___b"]
        expected = model(test_inputs)
        runner.update_constant_buffer(new_weights, True, False)
        new_output = runner_call(test_inputs)
        # We have not yet swapped the buffer, new_output should be the same as the old one.
        self.assertEqual(output, new_output)
        # Swap the buffer, should get the correct result now.
        runner.swap_constant_buffer()
        new_output = runner_call(test_inputs)
        self.assertEqual(expected, new_output)

    @requires_gpu
    def test_duplicate_constant_folding(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w1 = torch.randn(4, 4, device=device)
                self.w2 = torch.randn(4, 4, device=device)
                self.w3 = torch.randn(4, 4, device=device)
                self.w4 = torch.randn(4, 4, device=device)

            def forward(self, x):
                w_concat = torch.cat((self.w1, self.w2, self.w3, self.w4))
                return torch.cat((x, w_concat))

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_autotune_with_constant_folding(self):
        class Model(torch.nn.Module):
            def __init__(self, device) -> None:
                super().__init__()
                self.x = torch.randn(2048, 2048, dtype=torch.float16, device=device)

            def _quantize(self, input):
                return torch.abs(input)

            def forward(self, y):
                abs_weight = self._quantize(self.x)
                abs_y = self._quantize(y)

                return abs_weight, abs_y

        input1 = (torch.rand(2048, 2048, dtype=torch.float16, device=self.device),)
        model = Model(self.device).to(self.device)

        _ = model(*input1)

        ep = torch.export.export(model, input1, dynamic_shapes=None, strict=False)
        torch._inductor.aoti_compile_and_package(
            ep, inductor_configs={"aot_inductor.use_runtime_constant_folding": True}
        )

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "Compilation error",
    )
    def test_aot_inductor_consts_cpp_build(self):
        class Model(torch.nn.Module):
            def __init__(self, device) -> None:
                super().__init__()
                self.x = torch.randn(2048, 2048, dtype=torch.float16, device=device)

            def _quantize(self, input):
                return torch.abs(input)

            def forward(self, y):
                abs_weight = self._quantize(self.x)
                abs_y = self._quantize(y)

                return abs_weight, abs_y

        input1 = (torch.rand(2048, 2048, dtype=torch.float16, device=self.device),)
        model = Model(self.device).to(self.device)

        _ = model(*input1)

        ep = torch.export.export(model, input1, dynamic_shapes=None, strict=False)
        torch._inductor.aoti_compile_and_package(
            ep,
            inductor_configs={
                "aot_inductor.use_runtime_constant_folding": True,
                "aot_inductor.use_consts_asm_build": False,
            },
        )

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_triton_kernel_on_device_tma(self, dynamic, tma_version):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_on_device_tma_new_api
            if tma_version == "new"
            else add_kernel_on_device_tma_old_api
        )

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                BLOCK_SIZE = 32
                out = torch.zeros_like(a)
                m, n = out.size()

                # Allocate workspace for on-device TMA descriptors
                # Need 128 bytes per descriptor, 3 descriptors total
                if tma_version == "old":
                    workspace = torch.zeros(3 * 128, dtype=torch.uint8, device=a.device)
                else:
                    workspace = None

                grid = (triton.cdiv(m, BLOCK_SIZE), triton.cdiv(n, BLOCK_SIZE))

                kernel[grid](
                    a,
                    b,
                    out,
                    m,
                    n,
                    workspace,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

                return out

        a = torch.randn((32 * 4, 32 * 8), device=self.device)
        b = torch.randn((32 * 4, 32 * 8), device=self.device)
        example_inputs = (a, b)

        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=GPU_TYPE
            )
        )

        dynamic_shapes = None
        if dynamic:
            dim0 = Dim("s0", min=2, max=1024)
            dim1 = Dim("s1", min=2, max=1024)
            dynamic_shapes = {
                "a": {0: dim0, 1: None},
                "b": {0: dim1, 1: None},
            }

        self.check_model(
            Model(),
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @requires_gpu
    def test_multi_device(self):
        if self.device == "cpu" and GPU_TYPE == "xpu":
            raise unittest.SkipTest(
                "In this scenario, the test case will run XPU code in "
                "AOTIModelContainerRunnerCpu, which is not reasonable,"
                "See issue #140805"
            )

        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                x = x.cpu()
                x = x + 2
                x = x.to(GPU_TYPE)
                return x

        example_inputs = (torch.randn(32, 64, device=self.device),)
        self.check_model(Model(), example_inputs)

    @unittest.skip(
        "install_free_tensors leads to OOM - https://github.com/pytorch/pytorch/issues/164062"
    )
    def test_large_weight(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2048, 262144)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 262144, device=self.device),
            torch.randn(1, 2048, device=self.device),
        )

        # We only test compilation since we often get OOM running in CI.
        model = Model()
        model = model.to(self.device)
        AOTIRunnerUtil.compile(model, example_inputs)

    def test_constant_type_propagation(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        model = Model(self.device)
        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            so_path, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )
            FileCheck().check_not("torch::aot_inductor::ConstantType::Unknown").run(
                code
            )

    def test_subclasses(self):
        device_to_init = self.device

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4, device=device_to_init))
                self.p2 = torch.nn.Parameter(
                    CustomTensorPlainOut(
                        torch.ones(3, 4, device=device_to_init),
                        torch.ones(3, 4, device=device_to_init),
                    )
                )

            def forward(self, x):
                a = (2 * self.p1 + self.p2).sum()
                return x + a

        m = Foo()
        ref_x = torch.randn(3, 4, device=device_to_init)

        with torch.no_grad():
            result = AOTIRunnerUtil.run(
                m,
                (ref_x,),
            )
        actual = m(ref_x)
        self.assertTrue(same(result, actual))

    def test_large_mmaped_weights(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(512, 250112)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 250112, device=self.device),
            torch.randn(1, 512, device=self.device),
        )
        with config.patch({"aot_inductor.force_mmap_weights": True}):
            self.check_model(Model(), example_inputs)

    def test_large_mmaped_weights_on_disk(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(512, 250112)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 250112, device=self.device),
            torch.randn(1, 512, device=self.device),
        )
        with config.patch(
            {"aot_inductor.package_constants_on_disk_format": "binary_blob"}
        ):
            self.check_model(Model(), example_inputs)

    def test_with_offset(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.orig_tensor = torch.randn(2, 15, 10, device=device)[0]
                self.tensor = self.orig_tensor[5:, :]

            def forward(self, x, y):
                return (
                    x
                    + torch.nn.functional.linear(y, self.orig_tensor[:10, :])
                    + self.tensor
                )

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    def test_freezing(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn(9, 10, device=device)
                self.padding = torch.randn(1, 10, device=device)

            def forward(self, x, y):
                padded_weight = torch.cat((self.weight, self.padding), dim=0)
                return x + torch.nn.functional.linear(y, padded_weight)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )

        with config.patch({"freezing": True}):
            self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    def test_conv_freezing(self):
        dtypes = [torch.bfloat16, torch.float] if SM80OrLater else [torch.float]
        for dtype, groups in itertools.product(dtypes, [1, 2]):
            iC = 2
            oC = 3

            class Model(torch.nn.Module):
                def __init__(self, device):
                    super().__init__()
                    self.weight = torch.randn(oC * groups, iC, 3, 3, device=device).to(
                        dtype
                    )

                def forward(self, y):
                    return torch.nn.functional.conv2d(y, self.weight, groups=groups)

            example_inputs = (
                torch.randn(2, iC * groups, 10, 10, device=self.device).to(dtype),
            )

            with config.patch({"freezing": True}):
                self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    @tf32_on_and_off(0.005)
    def test_deconv_freezing(self):
        dtypes = [torch.float]
        if torch._C._has_mkldnn and torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype, groups in itertools.product(dtypes, [2, 1]):
            iC = 4
            oC = 2

            class Model(torch.nn.Module):
                def __init__(self, device):
                    super().__init__()
                    self.weight = torch.randn(iC, oC * groups, 2, 2, device=device).to(
                        dtype
                    )

                def forward(self, y):
                    return torch.nn.functional.conv_transpose2d(
                        y, self.weight, groups=groups
                    )

            example_inputs = (torch.randn(1, iC, 3, 3, device=self.device).to(dtype),)
            with config.patch({"freezing": True}):
                self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    def test_linear_freezing(self):
        dtypes = [torch.bfloat16, torch.float] if SM80OrLater else [torch.float]
        for dtype in dtypes:

            class LinearModel(torch.nn.Module):
                def __init__(self, device):
                    super().__init__()
                    self.weight = torch.randn(10, 10, device=device).to(dtype)
                    self.bias = torch.randn(10, device=device).to(dtype)

                def forward(self, y):
                    return torch.nn.functional.linear(y, self.weight, self.bias)

            example_inputs = (torch.randn(10, 10, device=self.device).to(dtype),)

            with config.patch({"freezing": True}):
                model = LinearModel(device=self.device)
                self.check_model(model, example_inputs)

    def test_same_backing(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo2",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo2", "CompositeExplicitAutograd", lib=lib)
            def foo_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return a + b

            class M(torch.nn.Module):
                def forward(self, a, b):
                    x = a.shape[0]
                    y = b.shape[0]
                    a = torch.cat([a, a])
                    a = torch.ops.mylib.foo2(a, a)
                    a = a * x
                    b = torch.cat([b, b])
                    b = torch.ops.mylib.foo2(b, b)
                    b = b * y
                    return a, b

            inp = (torch.ones(3, device=self.device), torch.ones(3, device=self.device))
            self.check_model(M(), inp)

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "MPS BFloat16 is only supported on MacOS 14+",
    )
    def test_empty_cat_dtype_promotion(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                z = torch.cat([x, y], dim=1)
                z = z.to(dtype=torch.bfloat16)
                return z * 2

        model = Foo()
        inps = (torch.randn(4, 10, dtype=torch.bfloat16), torch.randn(4, 0))
        self.check_model(model, inps)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_linear_dynamic_maxautotune(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x)

        model = Model().to(device=self.device)
        compile_inputs = (torch.randn(2048, 1, device=self.device),)
        dim0_x = Dim("dim0_x", min=2, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}}
        ep = torch.export.export(
            model, compile_inputs, dynamic_shapes=dynamic_shapes, strict=True
        )
        optimized = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(
                ep,
                inductor_configs={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                },
            )
        )
        runtime_input = torch.randn(10, 1, device=self.device)
        self.assertTrue(same(optimized(runtime_input), model(runtime_input)))
        runtime_input = torch.randn(16, 1, device=self.device)
        self.assertTrue(same(optimized(runtime_input), model(runtime_input)))
        runtime_input = torch.randn(100, 1, device=self.device)
        self.assertTrue(same(optimized(runtime_input), model(runtime_input)))

    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
            "remove_split_with_size_one_pass": {},
            "merge_getitem_cat_pass": {},
            "merge_stack_tahn_unbind_pass": {},
            "merge_splits_pass": {},
            "mutate_cat_pass": {},
            "split_cat_pass": {},
            "unbind_stack_pass": {},
        },
        post_grad_fusion_options={},
    )
    def test_simple_split(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.cat(tensors=torch.split(x, 4, dim=1), dim=-2)

        example_inputs = (torch.randn(2, 8, device=self.device),)
        counters.clear()
        model = Model().to(device=self.device)
        actual = AOTIRunnerUtil.legacy_run(self.device, model, example_inputs)
        self.assertTrue(same(model(*example_inputs), actual))
        self.assertEqual(counters["inductor"]["scmerge_split_removed"], 1)
        self.assertEqual(counters["inductor"]["scmerge_cat_removed"], 1)
        self.assertEqual(counters["inductor"]["scmerge_split_sections_removed"], 1)

    def test_amp_fallback_random(self):
        def fn(x, w):
            return torch.functional.F.linear(x, w)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch({"fallback_random": True}):
            with torch.amp.autocast(device_type=self.device):
                self.check_model(fn, example_inputs)

    def test_missing_output(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.cos(b)
                return c

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_output_misaligned(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                x_getitem = cat[0]
                y_getitem = cat[1]
                x_sigmoid = torch.sigmoid(x_getitem)
                return x_sigmoid, y_getitem

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(), example_inputs)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @skip("Test was marked as expected failure, but does not fail always anymore.")
    def test_dynamic_smem_above_default_limit(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x @ y

        model = Model().to(self.device)
        # on A100, the generated Triton kernel for this MM
        # requires 55296 bytes of dynamic SMEM which is above
        # the A100's default dynamic SMEM limit of 49152 bytes.
        example_inputs = (
            torch.randn(10285, 96, device=self.device),
            torch.randn(96, 1, device=self.device),
        )
        self.check_model(
            model,
            example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    def test_seq(self):
        layernorm = torch.nn.LayerNorm(10)
        net = torch.nn.Sequential(
            layernorm,
            torch.nn.ReLU(),
            layernorm,
            torch.nn.ReLU(),
        )

        example_inputs = (torch.randn(10, device=self.device),)
        self.check_model(net.eval(), example_inputs)

    def test_addmm(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        # We should be able to call self.check_model here, but torch.export.export
        # constants (non-parameter, non-buffer) doesn't work today.
        example_inputs = (a,)
        self.check_model(model, example_inputs)

    def test_aliased_buffer_reuse(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                x = 2 * x
                y = 2 * y
                c = torch.cat([x, y], dim=-1)
                d = 1 + c
                m = torch.mm(d, d)
                return m[:, :2] + x

        example_inputs = (
            torch.randn(4, 2, device=self.device),
            torch.randn(4, 2, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_buffer_reuse(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.cos(y)
                c = torch.mm(a, b)
                d = torch.relu(c)
                e = torch.sigmoid(d)
                f = torch.mm(x, y)
                g = e + f
                return g

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_duplicated_params(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.rand(6))
                self.q = self.p

            def forward(self, x):
                return self.p * x + self.q

        example_inputs = (torch.rand(6, device=self.device),)
        self.check_model(Model(), example_inputs)

    @unittest.skip("Skip this test, only for local test. SIGABRT is produced.")
    def test_inf(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        x = torch.randn(10, 10, device=self.device)
        x[0][0] = float("Inf")
        example_inputs = (
            x,
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(
            Model().to(self.device),
            example_inputs,
            options={"debug_check_inf_and_nan": True},
        )

    @unittest.skip("Skip this test, only for local test. SIGABRT is produced.")
    def test_nan(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        x = torch.randn(10, 10, device=self.device)
        x[0][0] = float("nan")
        example_inputs = (
            x,
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(
            Model().to(self.device),
            example_inputs,
            options={"debug_check_inf_and_nan": True},
        )

    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_assert_async(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU_TYPE")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                u0 = x.item()
                torch._check(u0 > 3)
                return torch.ones(u0)[0]

        x = torch.tensor(23, device=self.device)
        example_inputs = (x,)
        self.check_model(Model(), example_inputs)

    def test_simple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        example_inputs = (x, y)
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    def test_large_dynamic_dim(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        # Use a dimension that exceeds the maximum value of a C long long (2^63 - 1)
        dim0_x = Dim("dim0_x", min=1, max=1171368248680556527362)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        example_inputs = (x, y)
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @skipIfXpu
    def test_fp8(self):
        # cuda only
        if self.device != "cuda":
            return

        class Model(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.out_dtype = dtype

            def forward(self, x, weight, bias, scale_a, scale_b):
                weight = weight.to(e4m3_type)
                output = torch._scaled_mm(
                    x,
                    weight,
                    bias=input_bias,
                    out_dtype=self.out_dtype,
                    scale_a=scale_a,
                    scale_b=scale_b,
                )
                return output

        dtype = torch.float16

        a_scale = torch.Tensor([1.0]).to(device=GPU_TYPE)
        b_scale = torch.Tensor([1.0]).to(device=GPU_TYPE)
        input_bias = torch.rand(32, device=GPU_TYPE, dtype=dtype)
        weight_shape = (32, 16)
        weight = torch.rand(*weight_shape, device=GPU_TYPE, dtype=dtype).T
        a_inverse_scale = 1 / a_scale
        b_inverse_scale = 1 / b_scale

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device=GPU_TYPE, dtype=dtype).to(e4m3_type)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = ({0: dim0_x}, None, None, None, None)
        self.check_model(
            Model(dtype),
            (x, weight, input_bias, a_inverse_scale, b_inverse_scale),
            dynamic_shapes=dynamic_shapes,
        )

    @unittest.skipIf(
        TEST_WITH_ROCM or not IS_SM90,
        "scaled_grouped_mm is only supported on SM90",
    )
    @skipIfXpu
    def test_scaled_grouped_mm(self):
        # Test torch._scaled_grouped_mm AOTI lowering
        # cuda only
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, weight, scale_a, scale_b, offsets):
                # x: [num_groups, batch, in_features] - FP8 inputs
                # weight: [total_out_features, in_features] - FP8 weights (transposed)
                # scale_a: [num_groups] - input scales
                # scale_b: [num_groups] - weight scales
                # offsets: [num_groups] - cumulative output sizes
                output = torch._scaled_grouped_mm(
                    x,
                    weight.t(),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    offs=offsets,
                    use_fast_accum=True,
                )
                return output.half()

        dtype = torch.float16
        num_groups = 3
        batch_size = 64
        in_features = 128
        out_features_list = [64, 128, 256]  # Different output sizes for each group

        device = GPU_TYPE

        # Calculate offsets (cumulative output sizes)
        offsets = torch.cumsum(torch.tensor(out_features_list), dim=0).to(
            device, dtype=torch.int32
        )
        total_out_features = sum(out_features_list)

        # Create FP8 input tensors - stacked for all groups
        x_fp16 = torch.randn(
            num_groups, batch_size, in_features, dtype=dtype, device=device
        )
        x_fp8 = x_fp16.to(torch.float8_e4m3fn)

        # Create FP8 weight tensor - concatenated and transposed
        weight_fp16 = torch.randn(
            total_out_features, in_features, dtype=dtype, device=device
        )
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)

        # Create scales
        scale_a = torch.ones(num_groups, batch_size, device=device, dtype=torch.float32)
        scale_b = torch.ones(total_out_features, device=device, dtype=torch.float32)

        self.check_model(
            Model(),
            (x_fp8, weight_fp8, scale_a, scale_b, offsets),
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @skipIfXpu
    def test_fp8_view_of_param(self):
        # cuda only
        if self.device != GPU_TYPE:
            return

        class Model(torch.nn.Module):
            def __init__(self, dtype, weight):
                super().__init__()
                self.out_dtype = dtype
                self.weight = weight

            def forward(self, x, bias, scale_a, scale_b):
                # test: do the view inside of the graph,
                # AOTI needs to materialize this view before passing
                # it into the scaled_mm extern kernel
                weight = self.weight.T
                output = torch._scaled_mm(
                    x,
                    weight,
                    bias=input_bias,
                    out_dtype=self.out_dtype,
                    scale_a=scale_a,
                    scale_b=scale_b,
                )
                return output

        dtype = torch.float16

        a_scale = torch.Tensor([1.0]).to(device=self.device)
        b_scale = torch.Tensor([1.0]).to(device=self.device)
        input_bias = torch.rand(32, device=self.device, dtype=dtype)
        weight_shape = (32, 16)
        weight = torch.rand(*weight_shape, device=self.device, dtype=dtype).to(
            e4m3_type
        )
        a_inverse_scale = 1 / a_scale
        b_inverse_scale = 1 / b_scale

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device=self.device, dtype=dtype).to(e4m3_type)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = ({0: dim0_x}, None, None, None)
        self.check_model(
            Model(dtype, weight),
            (x, input_bias, a_inverse_scale, b_inverse_scale),
            dynamic_shapes=dynamic_shapes,
        )

    def test_poi_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

       
```



## High-Level Overview


This Python file contains 229 class(es) and 675 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AOTInductorTestsTemplate`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Foo`, `Model`, `Model`

**Functions defined**: `get_module_ext_type`, `test_simple`, `__init__`, `forward`, `test_triton_kernel_bool_param`, `forward`, `test_simple_multi_arch`, `__init__`, `forward`, `test_small_constant`, `__init__`, `forward`, `test_output_path_1`, `__init__`, `forward`, `test_output_path_2`, `__init__`, `forward`, `test_empty_constant_folding`, `__init__`

**Key imports**: itertools, logging, os, pathlib, subprocess, sys, tempfile, unittest, zipfile, skip


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `logging`
- `os`
- `pathlib`
- `subprocess`
- `sys`
- `tempfile`
- `unittest`
- `zipfile`
- `unittest.mock`: patch
- `torch`
- `torch._export`
- `torch._inductor`
- `torch._inductor.config`
- `torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq`
- `torch.nn as nn`
- `torch._dynamo`: config as dynamo_config
- `torch._dynamo.device_interface`: get_interface_for_device
- `torch._dynamo.testing`: rand_strided, same
- `torch._dynamo.utils`: counters
- `torch._inductor.codecache`: WritableTempFile
- `torch._inductor.cpp_builder`: normalize_path_separator
- `torch._inductor.package`: package_aoti
- `torch._inductor.runtime.runtime_utils`: cache_dir
- `torch._inductor.test_case`: TestCase
- `torch._library`: capture_triton
- `torch._utils_internal`: full_aoti_runtime_assert
- `torch.ao.quantization.quantize_pt2e`: convert_pt2e, prepare_pt2e


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_aot_inductor.py
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

- **File Documentation**: `test_aot_inductor.py_docs.md`
- **Keyword Index**: `test_aot_inductor.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
