# Documentation: test_aot_inductor.py

## File Metadata
- **Path**: `test/inductor/test_aot_inductor.py`
- **Size**: 288299 bytes
- **Lines**: 7820
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        list_example_inputs = [(x, y)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device=self.device),
                torch.randn(64, 2048, device=self.device),
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device=self.device),
                torch.randn(211, 2048, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            Model(), list_example_inputs, dynamic_shapes=dynamic_shapes
        )

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_addmm_multiple_dynamic(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

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
        dim0_a = Dim("dim0_a", min=1, max=2048)
        dynamic_shapes = {"a": {0: dim0_a}}
        list_example_inputs = [(a,)]
        batch = 2048
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        batch = 128
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            dynamic_shapes=dynamic_shapes,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_bmm_multiple_dynamic(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)

        M = 8
        N = 6
        K = 16
        model = Model()
        batch = 1024
        a = torch.randn(batch, M, K, device=self.device)
        b = torch.randn(batch, K, N, device=self.device)
        dim0_a = Dim("dim0_a", min=1, max=2048)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_a}}
        list_example_inputs = [(a, b)]
        batch = 2048
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device=self.device),
                torch.randn(batch, K, N, device=self.device),
            ),
        )
        batch = 128
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device=self.device),
                torch.randn(batch, K, N, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
            dynamic_shapes=dynamic_shapes,
        )

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    def test_foreach_multiple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                return cat

        model = Model()
        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        list_example_inputs = [(x, y)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device=self.device),
                torch.randn(64, 2048, device=self.device),
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device=self.device),
                torch.randn(211, 2048, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    # scaled_dot_product_flash_attention
    @unittest.skipIf(
        not HAS_XPU_AND_TRITON and not SM80OrLater, "bfloat16 only supported in sm80+"
    )
    def test_sdpa(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)[0]

        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @unittest.skipIf(not SM80OrLater, "bfloat16 only supported in sm80+")
    @unittest.skipIf(
        # for archs where this isn't lowered to flash attention, the math
        # backend will be used and it doesn't work for bfloat16
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Some archs don't support SDPA with bfloat16",
    )
    def test_sdpa_2(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, q, k, v, x):
                t = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )[0]
                return x + t

        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @skipIfNoFBGEMM
    def test_quantized_linear(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn(10, 10, device=device)
                self.bias = torch.randn(10, device=device)

            def forward(self, x):
                return torch.ops.quantized.linear_dynamic_fp16_unpacked_weight(
                    x, self.weight, self.bias
                )

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    @skipIfNoFBGEMM
    def test_quantized_linear_bias_none(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn(10, 10, device=device)

            def forward(self, x):
                return torch.ops.quantized.linear_dynamic_fp16_unpacked_weight(
                    x, self.weight, None
                )

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    @skipIfNoFBGEMM
    def test_quanatized_int8_linear(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn(10, 10, device=device)
                self.bias = torch.randn(10, device=device)
                self.input_scale = torch.tensor(0.1)
                self.input_zero_point = torch.tensor(0)
                self.weight_scale = torch.tensor(0.1)
                self.weight_zero_point = torch.tensor(0)
                self.output_scale = torch.tensor(0.1)
                self.output_zero_point = torch.tensor(0)
                self.out_channel = 10

            def forward(self, x):
                return torch.ops._quantized.wrapped_quantized_linear(
                    x,
                    self.input_scale,
                    self.input_zero_point,
                    self.weight,
                    self.weight_scale,
                    self.weight_zero_point,
                    self.bias,
                    self.output_scale,
                    self.output_zero_point,
                    self.out_channel,
                )

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_zero_grid_with_unbacked_symbols(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                nz = torch.nonzero(x)
                b = torch.ones_like(nz, dtype=torch.float16)
                c = torch.zeros_like(nz, dtype=torch.float16)
                d = (b + c) @ y
                return d.sum()

        example_inputs = (
            torch.tensor([1, 1, 1], device=self.device),
            torch.randn((1, 32), dtype=torch.float16, device=self.device),
        )
        self.check_model(Repro(), example_inputs)

    @skipIfMPS
    @config.patch({"unbacked_symint_fallback": 12})
    @parametrize("shift_k", [0, 1, 2, 3])
    @parametrize("use_static_size", [True, False])
    def test_unbacked_expr_replacements(self, shift_k, use_static_size):
        """
        Test parameters
        - shift_k: Validates that torch._check assertion order doesn't affect
        results by shifting the order of torch._checks
        - use_static_size: Tests torch._check compatibility between unbacked
        symbolic expressions and static shapes
        """

        if self.device != GPU_TYPE:
            raise unittest.SkipTest("Need triton for user-defined triton kernel")

        def realize_out_tensor_with_size(size):
            STATIC_DIM = 256  # large enough to hit IMA w/o compute-sanitizer
            tensor = torch.ones((size, STATIC_DIM), device=self.device)
            # Realize the tensor as an intermediate buffer
            nrows, ncols = tensor.shape
            numel = tensor.numel()
            add_kernel[nrows,](
                in_ptr0=tensor,
                in_ptr1=tensor,
                out_ptr=tensor,
                n_elements=numel,
                BLOCK_SIZE=ncols,
            )
            return tensor

        class Repro(torch.nn.Module):
            def forward(self, x, y, lst):
                STATIC_SIZE = 300
                s0, s1 = x.shape
                s2, s3 = y.shape
                u0, u1, u2, u3, u100 = lst.tolist()

                expr1 = s0 + u0
                expr2 = s1 + u1
                expr3 = (s2 * s3) + (u2 // u3)  # make this one a lil complicated
                expr4 = STATIC_SIZE if use_static_size else u100

                t1 = realize_out_tensor_with_size(expr1)
                t2 = realize_out_tensor_with_size(expr2)
                t3 = realize_out_tensor_with_size(expr3)
                t4 = realize_out_tensor_with_size(expr4)

                # shift tensors to change up the torch._check order
                tensors = [t1, t2, t3, t4]
                shifted_tensors = tensors[shift_k:] + tensors[:shift_k]

                # torch.cat implicitly runs torch._check(lhs == rhs)
                cat = torch.cat(shifted_tensors, dim=1)

                return cat * cat

        # Disable cuda caching allocator to check for IMA
        torch.cuda.caching_allocator_enable(False)
        model = Repro()
        example_inputs = (
            # s0, s1
            torch.randn((100, 200), device=self.device),
            # s2, s3
            torch.randn((100, 3), device=self.device),
            # u0, u1, u2, u3, u100
            torch.tensor([200, 100, 0, 1, 300], device=self.device, dtype=torch.int),
        )
        spec = {
            "x": (Dim.DYNAMIC, Dim.DYNAMIC),
            "y": (Dim.DYNAMIC, Dim.DYNAMIC),
            "lst": (Dim.STATIC,),
        }
        self.check_model(model, example_inputs, dynamic_shapes=spec)
        torch.cuda.caching_allocator_enable(True)

    @skipIfMPS
    @config.patch({"unbacked_symint_fallback": 12})
    @config.patch({"triton.autotune_at_compile_time": None})
    def test_replace_unbacked_symbol_with_backed_expr(self):
        # This will test how autotune_at_compile_time generates sample inputs
        # when the user torch._checks(s0 + s1 == u0).
        # We may fail with IMA if the generated input sizes aren't correct.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires triton")

        def force_realize(tensor):
            # Realize the tensor as an intermediate buffer
            nrows, ncols = tensor.shape
            numel = tensor.numel()
            add_kernel[nrows,](
                in_ptr0=tensor,
                in_ptr1=tensor,
                out_ptr=tensor,
                n_elements=numel,
                BLOCK_SIZE=ncols,
            )

        INNER_DIM = 256

        class Repro(torch.nn.Module):
            def forward(self, x, y, lengths):
                # Realize an intermediate buffer with backed shape: s0 + s1
                relevant_embeddings = torch.cat([x, y], dim=0)
                force_realize(relevant_embeddings)

                # Realize an intermediate buffer with unbacked shape: u0
                num_relevant_embeddings = lengths.nonzero().size(0)
                ones = torch.ones((num_relevant_embeddings, INNER_DIM), device=x.device)
                force_realize(ones)

                # Add deferred runtime assertion: s0 + s1 == u0
                torch._check(relevant_embeddings.size(0) == ones.size(0))
                relevant_embeddings += ones
                return relevant_embeddings * relevant_embeddings

        torch.cuda.caching_allocator_enable(False)
        model = Repro()
        example_inputs = (
            torch.randn((1000, INNER_DIM), device=self.device),
            torch.randn((2000, INNER_DIM), device=self.device),
            torch.ones(3000),
        )
        spec = {
            "x": (Dim.DYNAMIC, Dim.STATIC),
            "y": (Dim.DYNAMIC, Dim.STATIC),
            "lengths": (Dim.DYNAMIC,),
        }
        self.check_model(model, example_inputs, dynamic_shapes=spec)
        torch.cuda.caching_allocator_enable(True)

    @config.patch({"triton.autotune_at_compile_time": None})
    def test_stride_with_unbacked_expr(self):
        class Repro(torch.nn.Module):
            def forward(self, x, y):
                u0 = x.item()
                torch._check(u0 >= 1)
                s0 = y.size(0)
                expr = u0 * s0
                sevens = torch.empty_strided(
                    size=(10, expr, 32), stride=(expr * 32, 32, 1), device=x.device
                ).fill_(7)
                return sevens * 3

        example_inputs = (
            torch.scalar_tensor(2, dtype=torch.int, device=self.device),
            torch.ones(8, device=self.device),
        )
        self.check_model(Repro(), example_inputs)

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "bfloat16 is only supported on MacOS 14+",
    )
    def test_size_with_unbacked_add_expr(self):
        # Tests AOTI autotuning to make sure the correct input tensor sizes
        # are generated for sizes that include an expr such as s0 + u0.

        class Repro(torch.nn.Module):
            def forward(self, values, repeats, mask, embeddings, x, z, scalar):
                repeat_interleave = torch.repeat_interleave(values, repeats)
                index = torch.clamp(repeat_interleave, min=0, max=400).int()
                index_select = torch.index_select(embeddings, 0, index)

                backed = z.size(0)
                unbacked = scalar.item()

                unbacked_add_expr = backed + unbacked
                repeated = x.repeat(unbacked_add_expr, 1)
                return torch.cat([repeated, index_select], dim=1)

        example_inputs = (
            torch.ones(64, dtype=torch.int64, device=self.device),
            torch.ones(64, dtype=torch.int64, device=self.device) * 12,
            torch.ones((768,), dtype=torch.int64, device=self.device).bool(),
            torch.randn((401, 8), dtype=torch.bfloat16, device=self.device),
            torch.randn((1, 256), dtype=torch.bfloat16, device=self.device),
            torch.ones(758, 127, dtype=torch.int64, device=self.device),
            torch.scalar_tensor(10, dtype=torch.int32, device=self.device),
        )
        spec = {
            "values": (Dim.DYNAMIC,),
            "repeats": (Dim.DYNAMIC,),
            "mask": (Dim.DYNAMIC,),
            "embeddings": (Dim.DYNAMIC, Dim.STATIC),
            "x": (Dim.STATIC, Dim.STATIC),
            "z": (Dim.DYNAMIC, Dim.STATIC),
            "scalar": (),
        }
        self.check_model(Repro(), example_inputs, dynamic_shapes=spec)

    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_size_with_unbacked_add_expr_transitive(self):
        # Edge case with torch._check(expr1, expr2) + torch._check(expr2, unbacked).
        # When generating example input sizes for autotuning, it should coalesce
        # expr1, expr2, unbacked into a single size.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            def forward(self, values, repeats, mask, embeddings, x, y, z, lst):
                index = torch.repeat_interleave(values, repeats)
                index_select = torch.index_select(embeddings, 0, index)

                u0, u1 = lst.tolist()
                backed0, backed1 = z.size(0), z.size(1)

                repeated0 = y.repeat(backed0 + u0, 1)
                repeated1 = x.repeat(backed1 + u1, 1)
                out1 = torch.empty_like(repeated1)
                add_kernel[(out1.numel(),)](
                    repeated1, repeated1, out1, out1.numel(), BLOCK_SIZE=2
                )

                # Implicitly add torch._check(expr2, unbacked)
                cat = torch.cat([out1, index_select], dim=1)
                add = repeated0 + repeated1

                # Explicitly add torch._check(expr1, expr2)
                torch._check(repeated0.size(0) == out1.size(0))
                return cat, add

        example_inputs = (
            torch.ones(64, dtype=torch.int64, device=self.device),
            torch.ones(64, dtype=torch.int64, device=self.device) * 24,
            torch.ones((768,), dtype=torch.int64, device=self.device).bool(),
            torch.randn((401, 8), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.ones(758, 758, dtype=torch.int64, device=self.device),
            torch.tensor([10, 10], dtype=torch.int32, device=self.device),
        )
        spec = {
            "values": (Dim.DYNAMIC,),
            "repeats": (Dim.DYNAMIC,),
            "mask": (Dim.DYNAMIC,),
            "embeddings": (Dim.DYNAMIC, Dim.STATIC),
            "x": (Dim.DYNAMIC, Dim.STATIC),
            "y": (Dim.DYNAMIC, Dim.STATIC),
            "z": (Dim.DYNAMIC, Dim.DYNAMIC),
            "lst": (Dim.STATIC,),
        }
        self.check_model(Repro(), example_inputs, dynamic_shapes=spec)

    @config.patch({"unbacked_symint_fallback": 128})
    def test_size_with_unbacked_add_and_mul_expr(self):
        # Edge case with torch._check(add_expr, mul_expr). When generating example
        # input sizes for autotuning, make sure they coalesce into a single size.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            def forward(self, values, repeats, mask, embeddings, x, y, z, lst):
                u0, u1, u2 = lst.tolist()
                backed = z.size(0)
                backed1 = z.size(1)

                unbacked_add_expr = backed + u0
                unbacked_mul_expr = backed1 + (u1 * u2)
                repeated0 = x.repeat(unbacked_add_expr, 1)
                repeated1 = y.repeat(unbacked_mul_expr, 1)
                out0 = torch.empty_like(repeated0)
                out1 = torch.empty_like(repeated1)
                add_kernel[(out0.numel(),)](
                    repeated0, repeated0, out0, out0.numel(), BLOCK_SIZE=2
                )
                add_kernel[(out1.numel(),)](
                    repeated1, repeated1, out1, out1.numel(), BLOCK_SIZE=2
                )

                return torch.cat([out1, out0], dim=1)

        example_inputs = (
            torch.ones(64, dtype=torch.int64, device=self.device),
            torch.ones(64, dtype=torch.int64, device=self.device) * 24,
            torch.ones((768,), dtype=torch.int64, device=self.device).bool(),
            torch.randn((401, 8), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.ones(758, 758, dtype=torch.int64, device=self.device),
            torch.tensor([10, 5, 2], dtype=torch.int32, device=self.device),
        )
        spec = {
            "values": (Dim.DYNAMIC,),
            "repeats": (Dim.DYNAMIC,),
            "mask": (Dim.DYNAMIC,),
            "embeddings": (Dim.DYNAMIC, Dim.STATIC),
            "x": (Dim.DYNAMIC, Dim.STATIC),
            "y": (Dim.DYNAMIC, Dim.STATIC),
            "z": (Dim.DYNAMIC, Dim.DYNAMIC),
            "lst": (Dim.STATIC,),
        }
        self.check_model(Repro(), example_inputs, dynamic_shapes=spec)

    @skipIfXpu(msg="_scaled_dot_product_flash_attention is not supported on XPU yet")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Some archs don't support flash SDPA"
    )
    def test_fallback_kernel_with_symexpr_output(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Module(torch.nn.Module):
            def forward(self, q, k, v):
                q = q.reshape(
                    q.shape[0],
                    2,
                    q.shape[2] * q.shape[3],
                    q.shape[1] // 2,
                )
                k = k.reshape(
                    k.shape[0],
                    2,
                    k.shape[2] * k.shape[3],
                    k.shape[1] // 2,
                )
                v = v.reshape(
                    v.shape[0],
                    2,
                    v.shape[2] * v.shape[3],
                    v.shape[1] // 2,
                )

                res = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q,
                    k,
                    v,
                )
                return res[0]

        m = Module().to(device=self.device)
        tensor_shape = (4, 32, 4, 4)
        inputs = (
            torch.randn(tensor_shape, dtype=torch.float16, device=self.device),
            torch.randn(tensor_shape, dtype=torch.float16, device=self.device),
            torch.randn(tensor_shape, dtype=torch.float16, device=self.device),
        )

        dynamic_shapes = {
            "q": {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
            "k": {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
            "v": {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
        }
        ep = torch.export.export(m, inputs, dynamic_shapes=dynamic_shapes, strict=False)
        path = torch._inductor.aot_compile(ep.module(), inputs)
        aot_model = torch._export.aot_load(path, device=self.device)
        torch.testing.assert_close(m(*inputs), aot_model(*inputs))

    def test_aoti_constant_tensor(self):
        class Foo(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.a = torch.ones(4, 4, device=device)
                self.b = torch.ones(4, 4, device=device)

            def forward(self, x):
                return torch.ops.aten.linear.default(x, self.a, self.b)

        example_inputs = (torch.ones(4, 4, device=self.device),)
        self.check_model(Foo(self.device), example_inputs)

    def test_aoti_constant_tensor_name_collision(self):
        class SubModule(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.register_buffer(
                    "_tensor_constant1",
                    torch.ones(1, device=device, dtype=torch.float32),
                    persistent=True,
                )

            def forward(self, x):
                return self.linear(x)

        class Foo(torch.nn.Module):
            def __init__(self, user_float_feature_idx, device):
                super().__init__()
                self.user_float_feature_idx = user_float_feature_idx
                self.register_buffer(
                    "_tensor_constant0",
                    torch.ones(5, device=device, dtype=torch.float32),
                    persistent=True,
                )
                self.register_buffer(
                    "_tensor_constant1",
                    torch.ones(1, device=device, dtype=torch.float32),
                    persistent=True,
                )
                self.sub_mod = SubModule(device)

            def forward(self, x):
                self._tensor_constant0[1:2] = 1
                return (
                    torch.index_select(
                        x, 1, torch.tensor(self.user_float_feature_idx, device=x.device)
                    ),
                    self._tensor_constant0,
                    self._tensor_constant1,
                    self.sub_mod._tensor_constant1,
                )

        example_inputs = (torch.ones(4, 4, device=self.device),)
        user_float_feature_idx = [1]
        # we have to have run_decomposition first to trigger the name collision
        ep = torch.export.export(
            Foo(user_float_feature_idx, self.device), example_inputs, strict=False
        ).run_decompositions()
        gm = ep.module()
        self.check_model(gm.to(self.device), example_inputs)

    def test_large_grid(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, primals_5):
                view = torch.ops.aten.reshape.default(primals_5, [-1, 2, 4])
                primals_5 = None
                permute = torch.ops.aten.permute.default(view, [0, 2, 1])
                clone = torch.ops.aten.clone.default(
                    permute, memory_format=torch.contiguous_format
                )
                return clone

        # let y_grid = 65537
        s0 = 16777472
        s1 = 8
        example_inputs = (torch.rand(s0, s1, device=self.device),)
        self.check_model(Model(), example_inputs)

    def test_cond_simple(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.Simple(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_nested(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_abc = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p0": {},
            "p1": {},
            "p2": {},
            "a": {0: dim0_abc, 1: None},
            "b": {0: dim0_abc, 1: None},
            "c": {0: dim0_abc, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.Nested(),
            prepend_predicates(inputs, num_predicates=3),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_parameters(self):
        inputs = (torch.randn((10, 20), device=self.device),)
        dim0_abc = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_abc, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.Parameters(self.device),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_reinterpret_view_inputs_outputs(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # TODO: the min value need to be 5 because in the body_fn, we're slicing over z1[2:],
        # since the output size is [dim0_ab-3], when we extract tensor metadata out of the output
        # we call guard_size_oblivious, which assumes the dim0_ab-3 != 0 or 1. So we have to set
        # the minimum to 5 for now. We need to relax this restriction either by writing a less
        # constrained shape checking in fake impl of cond.
        dim0_ab = Dim("s0", min=5, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.ReinterpretView(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_multiple_outputs(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((30, 40), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dim0_c = Dim("s1", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
            "c": {0: dim0_c, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.MultipleOutputs(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_outer_code_before_after(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.OuterCode(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_use_buffers_from_outer_scope(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_abc = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_abc, 1: None},
            "b": {0: dim0_abc, 1: None},
            "c": {0: dim0_abc, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.OuterBuffers(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_cond_non_tensor_predicates(self, dynamic):
        inputs1 = (
            torch.randn((10, 20), device=self.device),
            torch.randn((15, 20), device=self.device),
        )
        inputs2 = (
            torch.randn((10, 20), device=self.device),
            torch.randn((5, 20), device=self.device),
        )
        inputs = (inputs1,)
        dynamic_shapes = None
        if dynamic:
            inputs = (inputs1, inputs2)
            dim0_a = Dim("s0", min=2, max=1024)
            dim0_b = Dim("s1", min=2, max=1024)
            dynamic_shapes = {
                "a": {0: dim0_a, 1: None},
                "b": {0: dim0_b, 1: None},
            }
        self.check_model_with_multiple_inputs(
            CondModels.WithNonTensorPredicate(),
            inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_cond_unbacked_symint_closure(self, dynamic):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((15, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dynamic_shapes = None
        if dynamic:
            dim0_a = Dim("s0", min=2, max=1024)
            dim0_b = Dim("s1", min=2, max=1024)
            dynamic_shapes = {
                "p": {},
                "x": {0: dim0_a, 1: None},
                "y": {0: dim0_b, 1: None},
                "z": {0: dim0_a, 1: None},
            }
        self.check_model_with_multiple_inputs(
            CondModels.UnbackedSymIntClosure(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    @common_utils.parametrize("dynamic", [False, True])
    def test_cond_mismatched_branch_output(self, dynamic):
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dynamic_shapes = None
        if dynamic:
            # Note the minimum has to be 4 because the model
            # is slicing over the first dim with [2:], if first
            # dim is 2 or 3, the slicing will be 0/1 specialized,
            # causing a constraint violation error.
            dim0_a = Dim("s0", min=4, max=1024)
            dim0_b = Dim("s1", min=4, max=1024)
            dynamic_shapes = {
                "p": {},
                "x": {0: dim0_a, 1: None},
                "y": {0: dim0_b, 1: None},
                "z": {0: dim0_a, 1: None},
            }
        self.check_model_with_multiple_inputs(
            CondModels.MismatchedOutputSize(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_symint_input(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    return x + a

                def false_fn(x):
                    return x + b * z

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        input1 = (
            torch.ones(3, 3, device=self.device),
            torch.ones(5, device=self.device),
            torch.ones(3, 3, device=self.device),
        )
        input2 = (
            torch.ones(10, 3, device=self.device),
            torch.ones(6, device=self.device),
            torch.ones(10, 3, device=self.device),
        )
        inputs = (input1, input2)
        dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}
        self.check_model_with_multiple_inputs(
            M(),
            inputs,
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_symint_input_disable_one_pass(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    return x + a

                def false_fn(x):
                    return x + b * z

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        input1 = (
            torch.ones(3, 3, device=self.device),
            torch.ones(5, device=self.device),
            torch.ones(3, 3, device=self.device),
        )
        input2 = (
            torch.ones(10, 3, device=self.device),
            torch.ones(6, device=self.device),
            torch.ones(10, 3, device=self.device),
        )
        inputs = (input1, input2)
        dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}
        with torch._inductor.config.patch({"triton.autotune_at_compile_time": False}):
            self.check_model_with_multiple_inputs(
                M(),
                inputs,
                dynamic_shapes=dynamic_shapes,
            )

    def test_while_loop_simple(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "ci": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Simple(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_nested(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "ci": {},
            "cj": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Nested(),
            prepend_counters(inputs, num_counters=2),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_with_outer_code(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "c": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.OuterCode(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    # mps doesn't support float64
    @skipIfMPS
    @unittest.skipIf(
        config.triton.native_matmul,
        "FIXME: cannot do get_size on FakeTensor during lowering.",
    )
    def test_while_loop_with_parameters(self):
        inputs = (
            torch.randn(
                (
                    10,
                    20,
                ),
                dtype=torch.float64,
                device=self.device,
            ),
        )
        dim0_a = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "c": {},
            "a": {0: dim0_a, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Parameters(self.device),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_with_outer_buffers(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # dynamic shapes don't work now due to
        # https://github.com/pytorch/pytorch/issues/123596
        # dim0_ab = Dim("s0", min=2, max=1024)
        # dynamic_shapes = {
        #     "c": {},
        #     "a": {0: dim0_ab, 1: None},
        #     "b": {0: dim0_ab, 1: None},
        # }
        dynamic_shapes = None
        self.check_model_with_multiple_inputs(
            WhileLoopModels.OuterBuffers(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_with_pytree_inputs(self):
        inputs = (
            torch.tensor(0, device=self.device),
            (
                [torch.randn(10, 20, device=self.device)],
                {
                    "x": torch.randn(10, 20, device=self.device),
                    "y": torch.randn(10, 20, device=self.device),
                },
            ),
        )
        self.check_model_with_multiple_inputs(
            WhileLoopModels.PytreeCarry(),
            [inputs],
            dynamic_shapes=None,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_unbacked_symint_closure(self, dynamic):
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.UnbackedSymIntClosure(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_mixed_device(self, dynamic):
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.MixedDevice(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_sym_expr_cond(self, dynamic):
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.SymExprCond(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_conv(self, dynamic):
        inputs = (torch.randn(2, 4, 4, 4, device=self.device, dtype=torch.float64),)
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "x": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Conv(self.device),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @config.patch({"is_predispatch": True})
    def test_constant(self):
        class M(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device

            def forward(self, x):
                t = torch.tensor(x.size(-1), device=self.device, dtype=torch.float)
                t = torch.sqrt(t * 3)
                return x * t

        self.check_model(M(self.device), (torch.randn(5, 5, device=self.device),))

    @unittest.skipIf(IS_MACOS, "no CUDA on Mac")
    def test_zero_grid_with_backed_symbols(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, b):
                return x + b

        example_inputs = (
            torch.randn((3, 2), device=self.device),
            torch.randn((1, 2), device=self.device),
        )
        dynamic_shapes = {
            "x": {0: Dim("dx"), 1: Dim.STATIC},
            "b": None,
        }

        # Compile & run model where dynamic dim size > 0.
        package_path: str = AOTIRunnerUtil.compile(
            Repro(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        aot_inductor_module(*example_inputs)

        # Re-run where dynamic dim size is 0.
        example_inputs = (
            torch.randn((0, 2), device=self.device),
            torch.randn((1, 2), device=self.device),
        )
        actual = aot_inductor_module(*example_inputs)
        expected = Repro()(*example_inputs)
        torch.testing.assert_close(actual, expected)

    def test_repeat_interleave(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.repeat_interleave.Tensor(x, output_size=12)

        example_inputs = (torch.ones((1,), dtype=torch.int32, device=self.device) * 12,)
        self.check_model(Repro(), example_inputs)

    def test_dynamic_cat(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.cat([a, b], dim=0)

        a = torch.randn(2, 4, device=self.device)
        b = torch.randn(3, 4, device=self.device)
        dim0_a = Dim("dim0_a", min=1, max=10)
        dim0_b = Dim("dim0_b", min=1, max=20)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_b}}
        example_inputs = (a, b)
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_buffer_mutation_1(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.foo = torch.nn.Buffer(torch.randn(4, 4, device=device))

            def forward(self, x):
                self.foo.add_(1)
                return self.foo + x

        example_inputs = (torch.rand(4, 4, device=self.device),)
        self.check_model(Model(self.device), example_inputs)

    def test_non_tensor_input(self):
        class Model(torch.nn.Module):
            def forward(self, a, b, alpha=1.0):
                return torch.add(a, b, alpha=alpha)

        a = torch.randn(10, device=self.device)
        b = torch.randn(10, device=self.device)

        for simdlen in [0, None]:
            with torch._inductor.config.patch({"cpp.simdlen": simdlen}):
                so_path = torch._export.aot_compile(
                    torch.ops.aten.add,
                    args=(a, b),
                    kwargs={"alpha": 2.0},
                )
                kernel_runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)
                res = kernel_runner.run([a, b])
                self.assertTrue(isinstance(res, list))
                self.assertTrue(len(res) == 1)
                self.assertEqual(Model()(a, b, alpha=2.0), res[0])

    def test_buffer_mutation_2(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.foo = torch.nn.Buffer(torch.arange(10, device=device))
                self.bar = torch.nn.Buffer(torch.arange(10, device=device))

            def forward(self, x):
                self.bar.mul_(2)
                self.foo[5] = self.bar[0]
                return x + self.bar, x * self.foo

        example_inputs = (torch.randn(10, device=self.device),)
        self.check_model(Model(self.device), example_inputs)

    @skipIfWindows(
        msg="OpenMP crashed application on windows"
    )  # TODO: (xuhancn) need to root cause and fix.
    def test_buffer_mutation_3(self):
        class KVCache(torch.nn.Module):
            def __init__(
                self,
                max_batch_size,
                max_seq_length,
                n_heads,
                head_dim,
                dtype=torch.float,
            ):
                super().__init__()
                cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
                self.k_cache = torch.nn.Buffer(torch.zeros(cache_shape, dtype=dtype))
                self.v_cache = torch.nn.Buffer(torch.zeros(cache_shape, dtype=dtype))

            def update(self, input_pos, k_val, v_val):
                # input_pos: [S], k_val: [B, H, S, D]
                k_out = self.k_cache
                v_out = self.v_cache
                k_out[:, :, input_pos] = k_val
                v_out[:, :, input_pos] = v_val

                return k_out, v_out

        class Model(torch.nn.Module):
            def __init__(self, 

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 229 class(es): AOTInductorTestsTemplate, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Foo, Model, Model

### Functions
This file defines 674 function(s): get_module_ext_type, test_simple, __init__, forward, test_triton_kernel_bool_param, forward, test_simple_multi_arch, __init__, forward, test_small_constant, __init__, forward, test_output_path_1, __init__, forward, test_output_path_2, __init__, forward, test_empty_constant_folding, __init__, forward, test_constant_folding, __init__, forward, test_constant_folding_with_update, __init__, forward, runner_call, test_duplicate_constant_folding, __init__


## Key Components

The file contains 20571 words across 7820 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 288299 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
