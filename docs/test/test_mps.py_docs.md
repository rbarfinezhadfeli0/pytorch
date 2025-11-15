# Documentation: test_mps.py

## File Metadata
- **Path**: `test/test_mps.py`
- **Size**: 570742 bytes
- **Lines**: 13031
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: mps"]
# ruff: noqa: F841
import io
import sys
import math
import random
import unittest
import warnings
import shutil
import subprocess
import tempfile
import os
import copy
import gc
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from collections import defaultdict
from torch import inf
from torch.nn import Buffer, Parameter
from torch.testing._internal import opinfo
from torch.testing._internal.common_utils import \
    (gradcheck, gradgradcheck, parametrize, run_tests, TestCase, download_file, MACOS_VERSION, IS_CI,
     NoTest, skipIfSlowGradcheckEnv, suppress_warnings, serialTest, instantiate_parametrized_tests, xfailIf)
from torch.testing._internal.common_mps import mps_ops_modifier, mps_ops_grad_modifier, mps_ops_error_inputs_modifier
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import get_all_dtypes, integral_types
import torch.backends.mps
from torch.distributions import Uniform, Exponential
from torch.utils._python_dispatch import TorchDispatchMode
from functools import partial

from torch.testing._internal.common_methods_invocations import (
    op_db,
    UnaryUfuncInfo,
    ReductionOpInfo,
    SpectralFuncInfo,
    BinaryUfuncInfo,
)
from torch.testing._internal.common_device_type import ops, dtypes, instantiate_device_type_tests, OpDTypes
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_quantization import _group_quantize_tensor, _dynamically_quantize_per_channel
import numpy as np
import torch
import torch.utils._pytree as pytree
from itertools import product
import operator

test_consistency_op_db = copy.deepcopy(op_db)
test_error_inputs_op_db = copy.deepcopy(op_db)

# Add bicubic2d_aa to test_consistency_op_db
for op in op_db:
    if op.name != "_upsample_bilinear2d_aa":
        continue
    op = copy.deepcopy(op)
    op.name = "_upsample_bicubic2d_aa"
    op.op = torch.ops.aten._upsample_bicubic2d_aa
    test_consistency_op_db.append(op)
    break

# Copied from `test_ops.py` for the purposes of duplicating `test_numpy_ref`
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)

# Same logic as test_cuda.py
if not torch.backends.mps.is_available():
    print('MPS not available, skipping tests', file=sys.stderr)
    TestCase = NoTest  # noqa: F811
    NNTestCase = NoTest  # noqa: F811

total_memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))

MPS_UNSUPPORTED_TYPES = [torch.double, torch.cdouble]
MPS_DTYPES = [t for t in get_all_dtypes() if t not in MPS_UNSUPPORTED_TYPES]

# Determine whether to enable MPS memory leak check (uses same code as CUDA).
TEST_MPS_MEM_LEAK_CHECK = os.getenv('PYTORCH_TEST_MPS_MEM_LEAK_CHECK', '0') == '1'

def skipMPSMemoryLeakCheckIf(condition):
    def dec(fn):
        if getattr(fn, '_do_mps_memory_leak_check', True):
            fn._do_mps_memory_leak_check = not condition
        return fn
    return dec

class MpsMemoryLeakCheck:
    def __init__(self, testcase, name=None):
        self.name = testcase.id() if name is None else name
        self.testcase = testcase

    def __enter__(self):
        # Performs a gc if required (required if any memory is held)
        caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
        if caching_allocator_mem_allocated > 0:
            gc.collect()
            torch.mps.empty_cache()

        # Acquires caching allocator and driver statistics before the test is run
        self.caching_allocator_before = torch.mps.current_allocated_memory()
        self.driver_before = torch.mps.driver_allocated_memory()

    def __exit__(self, exc_type, exc_value, traceback):
        # Don't check for leaks if an exception was thrown
        if exc_type is not None:
            return
        # Compares caching allocator before/after statistics
        # An increase in allocated memory is a discrepancy indicating a possible memory leak
        discrepancy_detected = False
        caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
        if caching_allocator_mem_allocated > self.caching_allocator_before:
            discrepancy_detected = True

        # Short-circuits if no discrepancy detected
        if not discrepancy_detected:
            return
        # Validates the discrepancy persists after garbage collection and
        # is confirmed by the driver API
        gc.collect()
        torch.mps.empty_cache()

        discrepancy_detected = True
        # Query memory multiple items to ensure leak was not transient
        for _ in range(3):
            caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
            driver_mem_allocated = torch.mps.driver_allocated_memory()

            caching_allocator_discrepancy = False
            driver_discrepancy = False

            if caching_allocator_mem_allocated > self.caching_allocator_before:
                caching_allocator_discrepancy = True

            if driver_mem_allocated > self.driver_before:
                driver_discrepancy = True

            if not (caching_allocator_discrepancy or driver_discrepancy):
                # Leak was false positive, exit loop
                discrepancy_detected = False
                break

        if caching_allocator_discrepancy and not driver_discrepancy:
            # Just raises a warning if the leak is not validated by the driver API
            msg = ("MPS caching allocator reports a memory leak not "
                   f"verified by the driver API in {self.name}! "
                   f"Caching allocator allocated memory was {self.caching_allocator_before} "
                   f"and is now reported as {caching_allocator_mem_allocated}. "
                   f"MPS driver allocated memory was {self.driver_before} and is now {driver_mem_allocated}.")
            warnings.warn(msg)
        elif caching_allocator_discrepancy and driver_discrepancy:
            # A caching allocator discrepancy validated by the driver API is a failure
            msg = (f"MPS driver API confirmed a leak in {self.name}! "
                   f"Caching allocator allocated memory was {self.caching_allocator_before} "
                   f"and is now reported as {caching_allocator_mem_allocated}. "
                   f"MPS driver allocated memory was {self.driver_before} and is now {driver_mem_allocated}.")

            raise RuntimeError(msg)

class TestAutocastMPS(TestCase):

    def test_matmul_autocast(self):
        autocast_tensor_A = torch.rand((8, 8), device="mps")
        autocast_tensor_B = torch.rand((8, 8), device="mps")
        tensor_A = autocast_tensor_A.detach().clone()
        tensor_B = autocast_tensor_B.detach().clone()
        autocast_output_tensor = torch.empty(8, 8)
        output_tensor = autocast_output_tensor.detach().clone()

        with torch.autocast(device_type="mps"):
            autocast_output_tensor = torch.mm(autocast_tensor_A, autocast_tensor_B)
            autocast_output_tensor = torch.mm(autocast_tensor_A, autocast_output_tensor)

        output_tensor = torch.mm(tensor_A, tensor_B)
        output_tensor = torch.mm(tensor_A, output_tensor)

        self.assertEqual(autocast_output_tensor.dtype, torch.float16, "Autocast output tensor was not expected type float16")
        self.assertEqual(autocast_output_tensor,
                         output_tensor.to(torch.float16),
                         f"Autocast & non-autocast tensors did not match, \
                         got:\n{autocast_output_tensor} \n{output_tensor.to(torch.float16)}")

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_dot_product_attention_autocast(self, dtype):
        # Regression test for https://github.com/pytorch/pytorch/issues/141774

        query = torch.rand(4, 1, 16, 8, dtype=torch.float32, device="mps")
        key = torch.rand(4, 1, 16, 8, dtype=torch.float32, device="mps")
        value = torch.rand(4, 1, 16, 8, dtype=dtype, device="mps")

        with torch.amp.autocast(device_type="mps"):
            y_autocast = F.scaled_dot_product_attention(query, key, value)

        y = F.scaled_dot_product_attention(query, key, value.to(torch.float32))
        self.assertEqual(y.to(y_autocast.dtype), y_autocast)

    def test_conv_transpose3d_autocast_fp32(self):
        m = nn.ConvTranspose3d(16, 33, 3, stride=2).to("mps")
        x = torch.randn(20, 16, 10, 50, 100, device="mps")
        with torch.amp.autocast(device_type="mps"):
            y = m(x)
        self.assertEqual(y.dtype, torch.float32)

    def test_conv3d_autocast(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/160415
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv3d(3, 3, 1)
                self.c2 = nn.Conv3d(3, 3, 1)

            def forward(self, x):
                x = self.c1(x)
                x = self.c2(x)
                return x

        x = torch.randn(2, 3, 4, 4, 4, device="mps")
        model = Foo().to("mps")
        with torch.amp.autocast(device_type="mps"):
            y = model(x)
        self.assertEqual(y.dtype, torch.float16)

    def test_gradscaler_mps(self):
        # big model to force chunking/depth in the gradscaler dispatch
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 2048)
                self.fc2 = nn.Linear(2048, 2048)
                self.fc3 = nn.Linear(2048, 2048)
                self.fc4 = nn.Linear(2048, 2048)
                self.fc5 = nn.Linear(2048, 5)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.relu(self.fc3(x))
                x = self.relu(self.fc4(x))
                return self.fc5(x)
        torch.manual_seed(42)

        def helper(model_cpu, model_mps, dtype, iterations, batch_size, atol=3e-4, rtol=1e-5):
            optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
            optimizer_mps = torch.optim.SGD(model_mps.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            input_cpu = torch.randn(batch_size, 10)
            target_cpu = torch.randn(batch_size, 5)
            input_mps = input_cpu.to('mps')
            target_mps = target_cpu.to('mps')

            scaler_cpu = torch.amp.GradScaler(device="cpu")
            scaler_mps = torch.amp.GradScaler(device="mps")
            for _ in range(iterations):
                optimizer_cpu.zero_grad()
                optimizer_mps.zero_grad()

                with torch.amp.autocast(device_type="cpu", dtype=dtype):
                    output_cpu = model_cpu(input_cpu)
                    loss_cpu = loss_fn(output_cpu, target_cpu)
                scaler_cpu.scale(loss_cpu).backward()
                scaler_cpu.step(optimizer_cpu)
                scaler_cpu.update()

                with torch.autocast(device_type="mps", dtype=dtype):
                    output_mps = model_mps(input_mps)
                    loss_mps = loss_fn(output_mps, target_mps)
                scaler_mps.scale(loss_mps).backward()
                scaler_mps.step(optimizer_mps)
                scaler_mps.update()

            for p_cpu, p_mps in zip(model_cpu.parameters(), model_mps.parameters()):
                self.assertEqual(p_mps.cpu(), p_cpu, rtol=rtol, atol=atol)

        model_cpu = Model().to('cpu')
        model_mps = Model().to('mps')
        model_mps.load_state_dict(model_cpu.state_dict())

        helper(model_cpu, model_mps, torch.float16, iterations=5, batch_size=4)
        helper(model_cpu, model_mps, torch.bfloat16, iterations=5, batch_size=4)

    def test_non_fast_path_amp_unscale(self):
        torch.manual_seed(42)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)

            def forward(self, x):
                x = self.linear1(x)
                x = F.relu(x)
                x = self.linear2(x)
                x = x.mean(dim=1)
                return x

        cpu_model = Model().to("cpu")
        mps_model = copy.deepcopy(cpu_model).to("mps")

        cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.01)
        mps_optimizer = torch.optim.SGD(mps_model.parameters(), lr=0.01)
        cpu_scaler = torch.amp.GradScaler(device="cpu")
        mps_scaler = torch.amp.GradScaler(device="mps")

        def helper(model, optimizer, scaler, device, input, target, apply_grad_transform=False):
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(input)
                loss = nn.MSELoss()(output, target)
            scaler.scale(loss).backward()

            if apply_grad_transform:
                for p in model.parameters():
                    if p.grad is not None and p.grad.dim() >= 2:
                        p.grad = p.grad.as_strided(p.grad.size(), (1,) * p.grad.dim())

            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        # CPU forward/backward pass
        input_cpu = torch.randn(32, 10, device="cpu")
        target_cpu = torch.randn(32, device="cpu")
        helper(cpu_model, cpu_optimizer, cpu_scaler, "cpu", input_cpu, target_cpu)

        # MPS forward/backward pass
        input_mps = input_cpu.to("mps")
        target_mps = target_cpu.to("mps")
        helper(mps_model, mps_optimizer, mps_scaler, "mps", input_mps, target_mps, apply_grad_transform=True)

        updated_linear1_weight_cpu = cpu_model.linear1.weight.detach()
        updated_linear2_weight_cpu = cpu_model.linear2.weight.detach()
        updated_linear1_weight_mps = mps_model.linear1.weight.detach().cpu()
        updated_linear2_weight_mps = mps_model.linear2.weight.detach().cpu()

        self.assertEqual(updated_linear1_weight_cpu, updated_linear1_weight_mps, atol=6e-4, rtol=1e-6)
        self.assertEqual(updated_linear2_weight_cpu, updated_linear2_weight_mps, atol=6e-4, rtol=1e-6)

# Expand TestCase class with Memory Leak Detection on MPS device
class TestCaseMPS(TestCase):
    _do_mps_memory_leak_check = True

    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        test_method = getattr(self, method_name, None)
        if test_method is not None:
            # Wraps the tested method if we should do MPS memory check.
            if TEST_MPS_MEM_LEAK_CHECK:
                if self._do_mps_memory_leak_check:
                    self.wrap_with_mps_policy(method_name, self.assertLeaksNoMpsTensors)

    def assertLeaksNoMpsTensors(self, name=None):
        name = self.id() if name is None else name
        return MpsMemoryLeakCheck(self, name)

    def wrap_with_mps_policy(self, method_name, policy):
        test_method = getattr(self, method_name)
        setattr(self, method_name, super().wrap_method_with_policy(test_method, policy))

    # checks for leaks even if TEST_MPS_MEM_LEAK_CHECK is 0
    def wrap_with_mps_memory_check(self, method):
        return super().wrap_method_with_policy(method, self.assertLeaksNoMpsTensors)

class TestMemoryLeak(TestCaseMPS):
    def test_mps_memory_leak_detection(self):
        l = []

        @self.wrap_with_mps_memory_check
        def no_leak():
            pass

        # Trigger an intentional memory leak
        @self.wrap_with_mps_memory_check
        def leak_gpu0():
            # increasing to 8MB to force acquiring a new block and overcome blocksize differences across platforms
            l.append(torch.randn(1024 * 1024 * 8, device=torch.device("mps")))

        no_leak()

        # check if a runtime error for memory leak was emitted which would
        # confirm whether memory leak detection worked successfully or not.
        with self.assertRaisesRegex(RuntimeError, r"MPS driver API confirmed .+"):
            leak_gpu0()

    def test_copy_cast_no_leak(self):

        def step(x):
            x = x.to(device='cpu', dtype=torch.float32)
            x = x.to(device='mps', dtype=torch.float16)

        a = torch.randn(128, 128, device='mps', dtype=torch.float16)
        # Warm up / prebuild MPS shaders (otherwise check fails on 13.2)
        step(a)
        torch.mps.empty_cache()
        driver_before = torch.mps.driver_allocated_memory()
        step(a)
        torch.mps.empty_cache()
        driver_after = torch.mps.driver_allocated_memory()
        self.assertEqual(driver_before, driver_after, f"Detected {driver_after - driver_before} bytes leak of GPU memory")


class TestPixelShuffle(TestCaseMPS):
    def test_pixel_shuffle_unshuffle(self):
        def _test_pixel_shuffle_unshuffle_helper(num_input_dims, valid_channels_dim=True,
                                                 upscale_factor=None, is_contiguous=True):

            def generate_input():
                # If valid_channels_dim=False, add 1 to make channels dim indivisible by upscale_factor ** 2.
                channels = random.randint(1, 4) * upscale_factor ** 2 + (0 if valid_channels_dim else 1)
                height = random.randint(5, 10)
                width = random.randint(5, 10)

                if num_input_dims == 1:
                    input = torch.rand(channels, requires_grad=True, device='mps')
                    assert is_contiguous
                elif num_input_dims == 2:
                    input = torch.rand(width, height, requires_grad=True, device='mps').T
                    if is_contiguous:
                        input = input.contiguous()
                else:
                    batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                    input = torch.rand(*batch_sizes, channels, width, height, requires_grad=True, device='mps')
                    input = input.transpose(-1, -2)
                    if is_contiguous:
                        input = input.contiguous()

                if not is_contiguous and len(input.reshape(-1)) > 0:
                    assert not input.is_contiguous()

                input = input.detach().clone()
                input.requires_grad = True
                return input

            # Function to imperatively ensure pixels are shuffled to the correct locations.
            # Used to validate the batch operations in pixel_shuffle.
            def _verify_pixel_shuffle(input, output, upscale_factor):
                for c in range(output.size(-3)):
                    for h in range(output.size(-2)):
                        for w in range(output.size(-1)):
                            height_idx = h // upscale_factor
                            weight_idx = w // upscale_factor
                            channel_idx = (upscale_factor * (h % upscale_factor)) + (w % upscale_factor) + \
                                          (c * upscale_factor ** 2)
                            self.assertEqual(output[..., c, h, w], input[..., channel_idx, height_idx, weight_idx])

            upscale_factor = random.randint(2, 5) if upscale_factor is None else upscale_factor
            input = generate_input()

            ps = nn.PixelShuffle(upscale_factor)
            pus = nn.PixelUnshuffle(downscale_factor=upscale_factor)

            if num_input_dims >= 3 and valid_channels_dim and upscale_factor > 0:
                output = ps(input)
                _verify_pixel_shuffle(input, output, upscale_factor)
                output.backward(output.data)
                self.assertEqual(input.data, input.grad.data)

                # Ensure unshuffle properly inverts shuffle.
                unshuffle_output = pus(output)
                self.assertEqual(input, unshuffle_output)
            else:
                self.assertRaises(RuntimeError, lambda: ps(input))

        def _test_pixel_unshuffle_error_case_helper(num_input_dims, valid_height_dim=True, valid_width_dim=True,
                                                    downscale_factor=None):
            downscale_factor = random.randint(2, 5) if downscale_factor is None else downscale_factor
            channels = random.randint(1, 4)
            # If valid_height_dim=False, add 1 to make height dim indivisible by downscale_factor.
            height = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_height_dim else 1)
            # If valid_width_dim=False, add 1 to make width dim indivisible by downscale_factor.
            width = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_width_dim else 1)

            if num_input_dims == 1:
                input = torch.rand(channels, requires_grad=True, device='mps')
            elif num_input_dims == 2:
                input = torch.rand(height, width, requires_grad=True, device='mps')
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                input = torch.rand(*batch_sizes, channels, height, width, requires_grad=True, device='mps')

            pus = nn.PixelUnshuffle(downscale_factor)
            self.assertRaises(RuntimeError, lambda: pus(input))

        def _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims):
            # For 1D - 2D, this is an error case.
            # For 3D - 5D, this is a success case for pixel_shuffle + pixel_unshuffle.
            is_contiguous_check = [True, False] if num_input_dims > 1 else [True]
            for is_contiguous in is_contiguous_check:
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, is_contiguous=is_contiguous
                )
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, valid_channels_dim=False, is_contiguous=is_contiguous
                )
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, upscale_factor=0, is_contiguous=is_contiguous
                )
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, upscale_factor=-2, is_contiguous=is_contiguous
                )

                # Error cases for pixel_unshuffle.
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_height_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_width_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=0)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=-2)

        def test_pixel_shuffle_large_upscale_factor():
            with self.assertRaises(ValueError):
                ps = nn.PixelShuffle(545460846592)
                ps(torch.randn(2, 16, 9, 3))

        def test_pixel_shuffle_unshuffle_1D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=1)

        def test_pixel_shuffle_unshuffle_2D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=2)

        def test_pixel_shuffle_unshuffle_3D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=3)

        def test_pixel_shuffle_unshuffle_4D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=4)

        def test_pixel_shuffle_unshuffle_5D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=5)

        test_pixel_shuffle_large_upscale_factor()
        test_pixel_shuffle_unshuffle_1D()
        test_pixel_shuffle_unshuffle_2D()
        test_pixel_shuffle_unshuffle_3D()
        test_pixel_shuffle_unshuffle_4D()
        test_pixel_shuffle_unshuffle_5D()

class MPSReluTest(TestCaseMPS):
    def _npRelu(self, np_features):
        return np.maximum(np_features, np.zeros(np_features.shape)).astype(np_features.dtype)

    def testNpRelu(self):
        torch.testing.assert_close(
            np.array([[0., 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
            self._npRelu(
                np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                         0.9]])))

    def _testRelu(self, np_features, device):
        np_relu = self._npRelu(np_features)
        # Convert the numpy array to a PyTorch Tensor,
        # and move the Tensor to the CPU/GPU based on the "device" parameter
        py_tensor = torch.from_numpy(np_features).to(device)
        py_relu = torch.nn.ReLU(inplace=False)(py_tensor)
        py_relu_cpu = py_relu.to("cpu")

        self.assertEqual(np_relu, py_relu_cpu)

    def _testReluInPlace(self, np_features, device):
        np_relu = self._npRelu(np_features)
        # Convert the numpy array to a PyTorch Tensor,
        # and move the Tensor to the CPU/GPU based on the "device" parameter
        py_tensor = torch.from_numpy(np_features).to(device)
        py_relu = torch.nn.ReLU(inplace=True)(py_tensor)
        py_relu_cpu = py_relu.to("cpu")

        self.assertEqual(np_relu, py_relu_cpu)
        # Inplace Relu modifies the initial input and it should match the output of Relu
        self.assertEqual(np_relu, py_tensor.to("cpu"))

    def testNumbersCPU(self):
        for t in [np.int32]:
            # Force execution on CPU even if a GPU kernel is available for the type.
            self._testRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="cpu")
            self._testReluInPlace(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="cpu")

    def testNumbersGPU(self):
        for t in [np.float16, np.float32]:
            self._testRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="mps")
            self._testReluInPlace(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="mps")
            self._testRelu(np.array([]).astype(t), device="mps")
            self._testReluInPlace(np.array([]).astype(t), device="mps")

class MatmulTest(TestCaseMPS):
    def _helper(self, shape_tensor_1, shape_tensor_2, expand_tensor_1_shape=None, expand_tensor_2_shape=None):
        if expand_tensor_1_shape:
            tensor1_mps = torch.randn(shape_tensor_1, device="mps").expand(expand_tensor_1_shape)
        else:
            tensor1_mps = torch.randn(shape_tensor_1, device="mps")

        if expand_tensor_2_shape:
            tensor2_mps = torch.randn(shape_tensor_2, device="mps").expand(expand_tensor_2_shape)
        else:
            tensor2_mps = torch.randn(shape_tensor_2, device="mps")

        tensor1_cpu = tensor1_mps.to("cpu")
        tensor2_cpu = tensor2_mps.to("cpu")

        matmul_cpu = torch.matmul(tensor1_cpu, tensor2_cpu)
        matmul_mps = torch.matmul(tensor1_mps, tensor2_mps)

        self.assertEqual(matmul_cpu, matmul_mps.to("cpu"))

    def test_vector_x_vector(self):
        # uses `dot`
        self._helper(3, 3)

    def test_matrix_x_vector(self):
        # uses `addmv`
        self._helper((3, 4), 4)

    def test_batched_matrix_x_broadcasted_vector(self):
        self._helper((10, 3, 4), 4)

    def test_batched_matrix_x_batched_matrix(self):
        # uses `bmm.out`
        self._helper((10, 3, 4), (10, 4, 5))

    def test_batched_matrix_x_broadcasted_matrix(self):
        self._helper((10, 3, 4), (4, 5))

    @serialTest()
    def test_large_matmul(self):
        # Issue: #141909
        tensor1_mps = torch.randn(1, 1, 72250, dtype=torch.half, device="mps")
        tensor2_mps = torch.randn(1, 72250, 1, dtype=torch.half, device="mps")
        matmul_mps = torch.matmul(tensor1_mps, tensor2_mps)

        tensor1_cpu = tensor1_mps.to("cpu")
        tensor2_cpu = tensor2_mps.to("cpu")
        matmul_cpu = torch.matmul(tensor1_cpu, tensor2_cpu)

        self.assertEqual(matmul_cpu, matmul_mps.to("cpu"))

    def test_large_complex_matmul(self):
        # See https://github.com/pytorch/pytorch/issues/167727
        M, N, K = 64, 300, 3000
        a = torch.rand((M, K), device='mps', dtype=torch.cfloat)
        b = torch.rand((K, N), device='mps', dtype=torch.cfloat)
        out = torch.mm(a, b)
        out_cpu = torch.mm(a.cpu(), b.cpu())
        # Operation order in large matmul can affect the results
        # Float ulp is 1e-6, multiplied by inner dim results in 5e-3
        self.assertEqual(out.cpu(), out_cpu, atol=5e-3, rtol=1e-5)

    def test_large_complex_addmm(self):
        # See https://github.com/pytorch/pytorch/issues/167727
        M, N, K = 64, 300, 3000
        a = torch.rand((M, N), device="mps", dtype=torch.cfloat)
        b = torch.rand((M, K), device='mps', dtype=torch.cfloat)
        c = torch.rand((K, N), device='mps', dtype=torch.cfloat)
        out = torch.addmm(a, b, c, alpha=1.0, beta=0.5j)
        out_cpu = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5j)
        # Operation order in large matmul can affect the results
        # Float ulp is 1e-6, multiplied by inner dim results in 5e-3
        self.assertEqual(out.cpu(), out_cpu, atol=5e-3, rtol=2e-5)

    def test_empty_matmul_vec(self):
        tensor_1 = torch.rand((0, 100), device="mps")
        tensor_2 = torch.rand((100, ), device="mps")
        self.assertEqual((tensor_1 @ tensor_2).cpu(), tensor_1.cpu() @ tensor_2.cpu())

class MPSLeakyReluTest(TestCaseMPS):
    def _npLeakyRelu(self, np_features, negative_slope=0.1):
        return np.maximum(np_features, negative_slope * np_features).astype(np_features.dtype)

    def testNpLeakyRelu(self):
        torch.testing.assert_close(
            np.array([[-0.09, 0.7, -0.05, 0.3, -0.01],
                      [0.1, -0.03, 0.5, -0.07, 0.9]]),
            self._npLeakyRelu(
                np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                         0.9]]),
                negative_slope=0.1))

    def _testLeakyRelu(self, shape, dtype, negative_slope, contiguous):
        cpu_x = torch.randn(shape, device='cpu', dtype=dtype)
        mps_x = cpu_x.detach().clone().to('mps')

        if not contiguous and not (0 in shape or len(shape) < 2):
            # Transposing will make the tensor non-contiguous
            cpu_x = cpu_x.transpose(0, 1)
            mps_x = mps_x.transpose(0, 1)
            assert not mps_x.is_contiguous()

        cpu_x.requires_grad_()
        mps_x.requires_grad_()

        relu_op = torch.nn.LeakyReLU(negative_slope)

        cpu_leaky_relu = relu_op(cpu_x)
        mps_leaky_relu = relu_op(mps_x)
        torch.testing.assert_close(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # test backward pass

        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')

        mps_leaky_relu.backward(gradient=mps_grad)
        cpu_leaky_relu.backward(gradient=cpu_grad)

        assert cpu_x.grad is not None  # Check that the grad is well-populated
        self.assertEqual(cpu_x.grad, mps_x.grad)

    def testNumbersCPU(self):
        for t in [torch.float, torch.half]:
            for shape in [[], (0,), (0, 3), (4,), (4, 3), (5, 4, 3)]:
                for contiguous in [True, False]:
                    self._testLeakyRelu(shape,
                                        dtype=t,
                                        negative_slope=0.2,
                                        contiguous=contiguous)

class TestAvgPool(TestCaseMPS):
    def _sum_pool2d(self, x, kernel_size):
        windows = torch.nn.functional.unfold(x, kernel_size=kernel_size, stride=kernel_size)
        return torch.sum(windows, dim=1)

    def _sum_pool3d(self, x, kernel_size):
        # Because unfold does not support 3D sliding window we will split tensor to multiple tensors and calculate sum
        h = kernel_size[0]
        splited_x = [t.sum(0) for t in x.split(h) if t.size(0) == h]
        # sum_pool2d assumes tensor in (1, 1, n, m) view, so unsqueeze two times
        splited_x = [self._sum_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size[1:]) for t in splited_x]
        joined_x = torch.cat(splited_x)
        return joined_x.view(1, joined_x.numel())

    def _avg_pool2d(self, x, kernel_size):
        size = reduce(operator.mul, kernel_size)  # noqa: F821
        return self._sum_pool2d(x, kernel_size) / size

    def _avg_pool3d(self, x, kernel_size):
        size = reduce(operator.mul, kernel_size)  # noqa: F821
        return self._sum_pool3d(x, kernel_size) / size

    def test_avg_pool2d_with_zero_divisor(self):
        self.assertRaisesRegex(RuntimeError, "divisor must be not zero",
                               lambda: F.avg_pool2d(torch.zeros(3, 3, 3), (2, 2), divisor_override=0))

    def test_doubletensor_avg_pool2d_with_divisor(self):
        n, m = 3, 3
        input = torch.rand(1, 1, n, m)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for divisor in [1, 7, i * j]:
                    actual = F.avg_pool2d(input[0], (i, j), divisor_override=divisor)
                    actual = actual.view(1, actual.numel())
                    expected = self._sum_pool2d(input, (i, j)) / divisor
                    self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_avg_pool2d_ceil_mode(self):
        # Regression test for gh-36977
        x = 10 * torch.randn((1, 16, 4, 4))
        y = torch.nn.functional.avg_pool2d(
            x, ceil_mode=True, count_include_pad=True, kernel_size=(1, 2),
            padding=(0, 1), stride=2)
        self.assertFalse(torch.isnan(y).any())
        y = torch.nn.functional.avg_pool2d(
            x.to('mps'), ceil_mode=True, count_include_pad=True, kernel_size=(1, 2),
            padding=(0, 1), stride=2)
        self.assertFalse(torch.isnan(y).any())

    # Test some cases for avg_pool2d which used to mismatch CPU results.
    # Addresses this issue: https://github.com/pytorch/pytorch/issues/160743
    def test_avg_pool2d_ceil_mode_mismatch(self):
        sizes = [
            (4, 2, 3),
            (5, 2, 3),
            (50, 2, 3),
            (4, 1, 2, 3),
            (4, 4, 2, 3),
            (2, 2, 4, 6),
            (5, 40, 60),
            (2, 2, 40, 60),
        ]

        kwargs = dict(kernel_size=[1, 3],
                      stride=[2, 3],
                      ceil_mode=True,
                      divisor_override=7)

        for input_size in sizes:
            model = torch.nn.AvgPool2d(**kwargs)
            x = torch.arange(math.prod(input_size), dtype=torch.float).reshape(input_size)
            out_cpu = model(x)
            out_mps = model(x.to("mps"))
            msg = f'{input_size=}, {kwargs=}'
            self.assertEqual(out_mps, out_cpu, msg=msg)


class TestMPS(TestCaseMPS):
    def test_exp(self, device="mps", dtype=torch.float):
        for v in (2, -2) + ((1j, 1 + 1j) if dtype.is_complex else ()):
            b = torch.arange(18, dtype=dtype, device=device) / 3 * math.pi
            a = torch.tensor(v, dtype=dtype, device="mps") * b
            self.compare_with_numpy(torch.exp, np.exp, a)

    @xfailIf(MACOS_VERSION > 15.0)
    def test_conv_raises_error(self, device='mps', dtype=torch.float):
        conv = nn.Conv1d(1, 65537, 3, padding=1).to('mps')

        x = torch.ones([1, 1, 3])
        with self.assertRaises(NotImplementedError):
            y = conv(x.to("mps"))

    @xfailIf(MACOS_VERSION < 15.1)
    def test_conv_high_channel_size(self):
        out_channels = 65537
        weight = torch.randn(out_channels, 1, 1)
        x = torch.ones([1, 1, 1])
        y_cpu = F.conv1d(x.to("cpu"), weight.to("cpu"))
        y_mps = F.conv1d(x.to("mps"), weight.to("mps"))
        self.assertEqual(y_cpu, y_mps)

    def test_triu_inf(self, device="mps", dtype=torch.float):
        for diag in [-1, 0, 1]:
            mask = torch.full((3, 6, 6), float("-inf"))
            mask_mps = mask.detach().clone().to('mps')
            cpu_ref = torch.triu(mask, diagonal=diag)
            mps_out = torch.triu(mask_mps, diagonal=diag)
            self.assertEqual(cpu_ref, mps_out)

    def test_exp1(self, device="mps", dtype=torch.float):
        input = torch.tensor([-0.1, 1.0, -0.9, 0.1], device=device, dtype=dtype)
        output = torch.exp(input)
        output_cpu = torch.exp(input.cpu())
        # If exponentWithTensor: MPS call is used on M1 running 14.5 test will fail with
        # Mismatched elements: 3 / 4 (75.0%)
        # Greatest absolute difference: 1.1920928955078125e-07 at index (3,) (up to 1e-08 allowed)
        # Greatest relative difference: 1.0786502002702036e-07 at index (3,) (up to 1e-08 allowed)
        self.assertEqual(output, output_cpu, atol=1e-8, rtol=1e-8)

    def test_exp_strided_output(self):
        x = torch.rand((256, 10), device='mps')
        x_cpu = x.to("cpu")

        x = x.permute(1, 0)
        x_cpu = x_cpu.permute(1, 0)

        res = x.exp()
        res_cpu = x_cpu.exp()
        self.assertEqual(res, res_cpu)

    def _testLeakyRelu(self, np_features, negative_slope, device):
        cpu_x = torch.from_numpy(np_features).requires_grad_()
        mps_x = torch.from_numpy(np_features).to('mps').requires_grad_()
        relu_op = torch.nn.LeakyReLU(negative_slope)

        cpu_leaky_relu = relu_op(cpu_x)
        mps_leaky_relu = relu_op(mps_x)
        torch.testing.assert_close(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # test backward pass
        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')
        cpu_leaky_relu.backward(gradient=cpu_grad)
        mps_leaky_relu.backward(gradient=mps_grad)
        torch.testing.assert_close(cpu_x.grad, mps_x.grad.to('cpu'))

    def testNumbersGPU(self):
        for t in [np.float32]:
            self._testLeakyRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                negative_slope=0.1,
                device="mps")

    def test_fill(self):

        def helper(val, shape, dtype):
            tensor = torch.zeros(shape, device='mps', dtype=dtype)
            tensor_mps = tensor.fill_(val)

            tensor_0 = torch.zeros(shape, device='cpu', dtype=dtype)
            tensor_cpu = tensor_0.fill_(val)

            self.assertEqual(tensor_mps, tensor_cpu)

        helper(0, [1024], torch.float32)
        helper(0.2, [2, 3], torch.float32)
        helper(0.2 + 0.5j, [2, 3], torch.complex64)

    def test_fill_storage_offset(self):
        shape = [2, 10]
        val = 0.2
        tensor = torch.ones(shape, device="mps")
        tensor_mps = tensor[:][1].fill_(val)
        tensor_0 = torch.ones(shape, device="cpu")
        tensor_cpu = tensor_0[:][1].fill_(val)

        self.assertEqual(tensor_mps, tensor_cpu)
        self.assertEqual(tensor, tensor_0)

        shape = [1, 10]
        val = 0.0
        tensor = torch.ones(shape, device="mps")
        val_tensor_mps = torch.tensor(val, device="mps")
        tensor_mps = tensor[:, 9].fill_(val_tensor_mps)
        # Regression test for https://github.com/pytorch/pytorch/issues/114692
        tensor[:, 5].fill_(val_tensor_mps)
        tensor_0 = torch.ones(shape, device="cpu")
        val_tensor_cpu = torch.tensor(val, device="cpu")
        tensor_cpu = tensor_0[:, 9].fill_(val_tensor_cpu)
        tensor_0[:, 5].fill_(val_tensor_cpu)

        self.assertEqual(tensor_mps.to(device="cpu"), tensor_cpu)
        self.assertEqual(tensor.to(device="cpu"), tensor_0)

    def test_cdist_large(self, device="mps"):
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(100, 10, device=device)
            y = torch.randn(100, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertEqual(expected, actual)

    def test_cdist_large_batch(self, device="mps"):
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(4, 3, 100, 10, device=device)
            y = torch.randn(4, 3, 100, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertEqual(expected, actual)

    def test_cdist_non_contiguous(self, device="mps"):
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(5, 7, device=device).mT
            y = torch.randn(5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(7, 5, device=device)
            y = torch.randn(5, 3, device=device).t()
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(5, 7, device=device).t()
            y = torch.randn(3, 5, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            self.assertEqual(expected, actual)

    def test_cdist_non_contiguous_batch(self, device="mps"):
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(4, 3, 2, 5, 7, device=device).mT
            y = torch.randn(4, 3, 2, 5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(7, 2, 7, 5, device=device)
            y = torch.randn(7, 2, 5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(4, 5, 7, device=device).mT
            y = torch.randn(4, 3, 5, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            self.assertEqual(expected, actual)

    def test_cdist_euclidean_large(self, device="mps"):
        def _test_euclidean_large_cdist(sizex, sizey=None):
            if sizey is None:
                sizey = sizex
            x = torch.randn(sizex, device=device, dtype=torch.float)
            y = torch.randn(sizey, device=device, dtype=torch.float)
            eps = 1e-6
            # to avoid extremum
            x = x - (((x - y) < eps).float() * 2 * eps)
            x.requires_grad = True
            y.requires_grad = True
            dist = torch.cdist(x, y, p=2)
            # Do a backward pass to check that it is valid for large
            # matrices
            loss = dist.sum()
            loss.backward()

        _test_euclidean_large_cdist((2000, 5))

    def test_cdist_same_inputs(self, device="mps"):
        # Test to detect issues in cdist gradient calculation
        # When the distances are 0
        sizex = (1, 27, 32)
        for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            y = x.clone()
            eps = 1e-6
            x.requires_grad = True
            d = torch.cdist(x, y)
            d.backward(dist_grad)
            # Check that the backward pass does not contain invalid
            # values such as nan or inf
            assert torch.isfinite(x.grad).all()


    def _brute_cdist(self, x, y, p=2):
        r1 = x.shape[-2]
        r2 = y.shape[-2]
        if r1 == 0 or r2 == 0:
            return torch.empty(r1, r2, device=x.device)
        return torch.norm(x[..., None, :] - y[..., None, :, :], p=p, dim=-1)

    def test_cdist_norm(self, device="mps"):
        for r1 in [3, 4]:
            for m in [2, 3]:
                for r2 in [4, 6]:
                    for p in [0, 1, 1.5, 2.5, float('inf')]:
                        x = torch.randn(r1, m, device=device)
                        y = torch.randn(r2, m, device=device)
                        if p == 2:
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    def test_cdist_norm_batch(self, device="mps"):
        for r1 in [3, 4]:
            for m in [2, 3]:
                for r2 in [4, 6]:
                    for p in [0, 3, 1.5, 2.5, float('inf')]:
                        x = torch.randn(2, 3, 6, r1, m, device=device)
                        y = torch.randn(2, 3, 6, r2, m, device=device)
                        if p == 2:
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    def test_mm(self):
        B = torch.ones(5, 6).to("mps")
        C = torch.ones(6, 5).to("mps")
        D = torch.mm(B, C).cpu()
        torch.testing.assert_close(D, torch.full((5, 5), 6.0))

    def test_linalg_cross(self):
        def helper(dtype):
            device = "mps"
            if dtype is torch.int32 or dtype is torch.int64:
                x = torch.randint(0, 99999, (100, 3, 100), dtype=dtype, device=device)
                y = torch.randint(0, 99999, (100, 3, 100), dtype=dtype, device=device)
            else:
                x = torch.rand(100, 3, 100, dtype=dtype, device=device)
                y = torch.rand(100, 3, 100, dtype=dtype, device=device)
            x_cpu = x.to("cpu")
            y_cpu = y.to("cpu")
            res1 = torch.linalg.cross(x, y, dim=1)
            res2 = torch.tensor((), dtype=dtype, device=device)
            res1_cpu = torch.linalg.cross(x_cpu, y_cpu, dim=1)
            res2_cpu = torch.tensor((), dtype=dtype, device="cpu")
            torch.linalg.cross(x, y, dim=1, out=res2)
            torch.linalg.cross(x_cpu, y_cpu, dim=1, out=res2_cpu)
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res1_cpu)
            self.assertEqual(res2, res2_cpu)

            # test for broadcastable inputs
            if dtype is torch.int32 or dtype is torch.int64:
                x = torch.randint(0, 99999, (1, 3, 2), dtype=dtype, device=device)
                y = torch.randint(0, 99999, (4, 3, 1), dtype=dtype, device=device)
            else:
                x = torch.rand(1, 3, 2, dtype=dtype, device=device)
                y = torch.rand(4, 3, 1, dtype=dtype, device=device)
            x_cpu = x.to("cpu")
            y_cpu = y.to("cpu")
            res1 = torch.linalg.cross(x, y, dim=1)
            res2 = torch.tensor((), dtype=dtype, device=device)
            res1_cpu = torch.linalg.cross(x_cpu, y_cpu, dim=1)
            res2_cpu = torch.tensor((), dtype=dtype, device="cpu")
            torch.linalg.cross(x, y, dim=1, out=res2)
            torch.linalg.cross(x_cpu, y_cpu, dim=1, out=res2_cpu)
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res1_cpu)
            self.assertEqual(res2, res2_cpu)
        [helper(dtype) for dtype in [torch.int32, torch.int64, torch.float32]]

    def test_cross(self):
        a = torch.randn(4, 3, device="mps")
        b = torch.randn(4, 3, device="mps")
        a_cpu = a.to("cpu")
        b_cpu = b.to("cpu")
        res = torch.cross(a, b, dim=1)
        res_cpu = torch.cross(a_cpu, b_cpu, dim=1)
        self.assertEqual(res, res_cpu)

    def test_addmm(self):
        A = torch.ones(5, 5).to("mps")
        B = torch.ones(5, 6).to("mps")
        C = torch.ones(6, 5).to("mps")
        D = torch.addmm(A, B, C).to("cpu")
        torch.testing.assert_close(D, torch.full((5, 5), 7.0))

    def test_bmm(self):
        batch1_cpu = torch.randn(10, 3, 4)
        batch2_cpu = torch.randn(10, 4, 5)

        batch1_mps = batch1_cpu.detach().clone().to("mps")
        batch2_mps = batch2_cpu.detach().clone().to("mps")

        output_cpu = torch.bmm(batch1_cpu, batch2_cpu)
        output_mps = torch.bmm(batch1_mps, batch2_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    @xfailIf(MACOS_VERSION < 15.0)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_bmm(self, dtype):
        B, M, N = 11, 20064, 128
        batch1 = torch.randn(B, M, N, dtype=dtype, device='mps')
        batch2 = torch.randn(B, N, M, dtype=dtype, device='mps')
        output_mps = torch.bmm(batch1, batch2)

        # For performance reasons, check only one(non-first) batch for correctness
        # TODO: Check two when https://github.com/pytorch/pytorch/issues/153560 is fixed
        batch_idx = torch.randint(1, B, size=()).item()
        output_cpu = torch.mm(batch1[batch_idx].cpu(), batch2[batch_idx].cpu())
        # Using the low precision comparison for FP16
        tol = 1e-2 if dtype == torch.float16 else None
        self.assertEqual(output_cpu, output_mps[batch_idx], atol=tol, rtol=tol)

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_take_along_dim(self, dtype):
        x = torch.tensor([[-5.], [0.], [5.]], dtype=dtype)
        inds = torch.tensor([[0], [1], [2]])
        ref = torch.take_along_dim(x, inds, 0)
        x_mps = x.detach().clone().to('mps')
        inds_mps = inds.detach().clone().to('mps')
        res = torch.take_along_dim(x_mps, inds_mps, 0)
        self.assertEqual(res, ref)

    def test_addr(self):
        A = torch.ones(5, 10).to("mps")
        B = torch.ones(5).to("mps")
        C = torch.ones(10).to("mps")
        D = torch.addr(A, B, C).to("cpu")
        torch.testing.assert_close(D, torch.full((5, 10), 2.0))

    def test_trace(self):
        M_cpu = torch.randn(3, 3)
        M_mps = M_cpu.detach().clone().to("mps")

        output_cpu = torch.trace(M_cpu)
        output_mps = torch.trace(M_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    def test_addbmm(self):
        M_cpu = torch.randn(3, 5)
        batch1_cpu = torch.randn(10, 3, 4)
        batch2_cpu = torch.randn(10, 4, 5)

        M_mps = M_cpu.detach().clone().to("mps")
        batch1_mps = batch1_cpu.detach().clone().to("mps")
        batch2_mps = batch2_cpu.detach().clone().to("mps")

        output_cpu = torch.addbmm(M_cpu, batch1_cpu, batch2_cpu)
        output_mps = torch.addbmm(M_mps, batch1_mps, batch2_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    def test_baddbmm(self):
        def helper(input_shape, batch1_shape, batch2_shape):
            M_cpu = torch.randn(input_shape)
            batch1_cpu = torch.randn(batch1_shape)
            batch2_cpu = torch.randn(batch2_shape)
            alpha = 1.2
            beta = 0.8

            M_mps = M_cpu.detach().clone().to("mps")
            batch1_mps = batch1_cpu.detach().clone().to("mps")
            batch2_mps = batch2_cpu.detach().clone().to("mps")

            output_cpu = torch.baddbmm(M_cpu, batch1_cpu, batch2_cpu, beta=beta, alpha=alpha)
            output_mps = torch.baddbmm(M_mps, batch1_mps, batch2_mps, beta=beta, alpha=alpha)

            self.assertEqual(output_cpu, output_mps)
            self.assertEqual(output_cpu.size(), output_mps.size())

        helper(input_shape=(3, 5), batch1_shape=(10, 3, 4), batch2_shape=(10, 4, 5))
        helper(input_shape=(10, 3, 5), batch1_shape=(10, 3, 4), batch2_shape=(10, 4, 5))
        helper(input_shape=(1, 77, 77), batch1_shape=(8, 77, 64), batch2_shape=(8, 64, 77))

    def test_local_scalar_dense_mps(self):
        x_cpu = torch.randn(1)
        y_mps = x_cpu.to("mps")
        torch.testing.assert_close(x_cpu.item(), y_mps.item())

    def test_linear_1d_weight(self):
        device = 'cpu'
        projected = torch.rand([8]).to(device)
        x = torch.rand([1, 2, 2, 8]).to(device)
        x_mps = x.to('mps')
        projected_mps = projected.to('mps')
        linear = F.linear(x, projected)
        linear_mps = F.linear(x_mps, projected_mps)

        self.assertEqual(linear, linear_mps)

        projected = torch.rand([1, 8]).to(device)
        x = torch.rand([1, 2, 2, 8]).to(device)
        x_mps = x.to('mps')
        projected_mps = projected.to('mps')
        linear = F.linear(x, projected)
        linear_mps = F.linear(x_mps, projected_mps)

        self.assertEqual(linear, linear_mps)

    def test_linear_bias(self):
        def helper(bias_shape):
            device = "cpu"
            x = torch.randn(2, 2, 2, 64, device=device)
            linear = torch.nn.Linear(64, 4, device=device)
            linear.bias = torch.nn.Parameter(torch.randn(bias_shape, dtype=torch.float32, device=device))
            y = linear(x)
            device = "mps"
            x_mps = x.to(device)
            linear.to(device)
            y_mps = linear(x_mps)
            self.assertEqual(y, y_mps)

        helper(())
        helper((2, 4))

    def test_linear_errors(self):
        # Mixed CPU<->MPS tensors
        size = (3, 3)

        # Unsupported dtypes
        with self.assertRaisesRegex(RuntimeError, "does not support linear for non-float weights"):
            torch.nn.functional.linear(torch.rand(size, device='mps'),
                                       torch.randint(-10, 10, size, dtype=torch.int8, device='mps'))

        # Weights on wrong device
        with self.assertRaisesRegex(RuntimeError, "argument weight is on cpu but expected on mps"):
            torch.nn.functional.linear(torch.rand(size, device='mps'),
                                       torch.rand(size, device='cpu'))

        # Input on wrong device
        with self.assertRaisesRegex(RuntimeError, "argument input is on cpu but expected on mps"):
            torch.nn.functional.linear(torch.rand(size, device='cpu'),
                                       torch.rand(size, device='mps'))

    def test_linear_non_contiguous(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/161640
        # Slice tensors to force non-contiguity
        large_weight = torch.randn(12, 8, device='mps')
        weight_sliced = large_weight[::2, ::1]
        weight_contiguous_equiv = weight_sliced.contiguous()
        input_s = torch.randn(2, 8, device='mps')
        result_sliced = torch.nn.functional.linear(input_s, weight_sliced)
        result_contig = torch.nn.functional.linear(input_s, weight_contiguous_equiv)
        self.assertEqual(result_contig, result_sliced)

    def _linear_helper(self, in_features, out_features, shape, bias=True, backward_pass=False):
        cpu_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, device="cpu", bias=bias)
        mps_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, device="mps", bias=bias)

        # Use the same weights and bias as the ones from the cpu
        mps_linear.weight.data = cpu_linear.weight.data.detach().clone().to("mps")

        if bias:
            mps_linear.bias.data = cpu_linear.bias.data.detach().clone().to("mps")

        linear_mps_input = torch.randn(shape).to('mps')
        linear_cpu_input = linear_mps_input.detach().clone().to('cpu')

        if backward_pass:
            linear_mps_input = linear_mps_input.requires_grad_()
            linear_cpu_input = linear_cpu_input.requires_grad_()

        linear_cpu_output = cpu_linear(linear_cpu_input)
        linear_mps_output = mps_linear(linear_mps_input)

        self.assertEqual(linear_cpu_output, linear_mps_output.to('cpu'))
        self.assertEqual(linear_cpu_output.size(), linear_mps_output.size())

        if backward_pass:
            cpu_grad = torch.rand_like(linear_cpu_output, requires_grad=True)
            grad = cpu_grad.detach().to('mps').requires_grad_()

            linear_cpu_output.backward(gradient=cpu_grad, create_graph=True)
            linear_mps_output.backward(gradient=grad, create_graph=True)

            self.assertEqual(linear_cpu_input.grad.size(), linear_mps_input.grad.size())
            self.assertEqual(linear_cpu_input.grad, linear_mps_input.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)

            self.assertEqual(cpu_linear.weight.grad.size(), mps_linear.weight.grad.size())
            self.assertEqual(cpu_linear.weight.grad, mps_linear.weight.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)
            if bias:
                self.assertEqual(cpu_linear.bias.grad.size(), mps_linear.bias.grad.size())
                self.assertEqual(cpu_linear.bias.grad, mps_linear.bias.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)

            # test gradgrad
            x_grad_out = torch.rand_like(linear_cpu_input)
            x_grad_out_mps = x_grad_out.to("mps")
            w_grad_out = torch.rand_like(cpu_linear.weight)
            w_grad_out_mps = w_grad_out.to("mps")

            linear_cpu_input.grad.detach().zero_()
            linear_mps_input.grad.detach().zero_()
            cpu_linear.weight.grad.detach().zero_()
            mps_linear.weight.grad.detach().zero_()
            if bias:
                b_grad_out = torch.rand_like(cpu_linear.bias)
                b_grad_out_mps = b_grad_out.to("mps")
                cpu_linear.bias.grad.detach().zero_()
                mps_linear.bias.grad.detach().zero_()

            linear_cpu_input.grad.backward(x_grad_out, retain_graph=True)
            linear_mps_input.grad.backward(x_grad_out_mps, retain_graph=True)
            cpu_linear.weight.grad.backward(w_grad_out, retain_graph=True)
            mps_linear.weight.grad.backward(w_grad_out_mps, retain_graph=True)
            if bias:
                cpu_linear.bias.grad.backward(b_grad_out, retain_graph=True)
                mps_linear.bias.grad.backward(b_grad_out_mps, retain_graph=True)

            self.assertEqual(cpu_grad.grad, grad.grad)
            self.assertEqual(linear_cpu_input.grad, linear_mps_input.grad)
            self.assertEqual(cpu_linear.weight.grad, mps_linear.weight.grad)
            if bias:
                self.assertEqual(cpu_linear.bias.grad, mps_linear.bias.grad)

    def test_linear1D(self):
        self._linear_helper(in_features=2, out_features=3, shape=([2]), bias=True, backward_pass=False)

    def test_linear1D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=([2]), bias=True, backward_pass=True)

    def test_linear2D(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=True, backward_pass=False)

    def test_linear2D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=True, backward_pass=True)

    def test_linear2D_no_bias(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=False, backward_pass=False)

    def test_linear2D_no_bias_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=False, backward_pass=True)

    def test_linear3D(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=False)

    def test_linear3D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=True)

    def test_linear3D_no_bias(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=False)

    def test_linear3D_no_bias_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=True)

    def test_linear_large(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/122045
        x_cpu = torch.randn(9, 1024, 1, device='cpu')
        w_cpu = torch.randn(50304, 1, device='cpu')
        x_mps = x_cpu.detach().clone().to('mps')
        w_mps = w_cpu.detach().clone().to('mps')

        out_cpu = F.linear(x_cpu, w_cpu, None)
        out_mps = F.linear(x_mps, w_mps, None)

        self.assertEqual(out_cpu, out_mps)

    def test_uniform(self):
        low = torch.zeros(5, 5, requires_grad=True)
        high = (torch.ones(5, 5) * 3).requires_grad_()
        low_1d = torch.zeros(1, requires_grad=True)
        high_1d = (torch.ones(1) * 3).requires_grad_()
        self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
        self.assertEqual(Uniform(low, high).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
        self.assertEqual(Uniform(low_1d, high_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Uniform(0.0, 1.0).sample((1,)).size(), (1,))

        # Check log_prob computation when value outside range
        uniform = Uniform(low_1d, high_1d, validate_args=False)
        above_high = torch.tensor([4.0])
        below_low = torch.tensor([-1.0])
        self.assertEqual(uniform.log_prob(above_high).item(), -inf)
        self.assertEqual(uniform.log_prob(below_low).item(), -inf)

        # check cdf computation when value outside range
        self.assertEqual(uniform.cdf(below_low).item(), 0)
        self.assertEqual(uniform.cdf(above_high).item(), 1)

        state = torch.get_rng_state()
        rand = low.new(low.size()).uniform_()
        torch.set_rng_state(state)
        u = Uniform(low, high).rsample()
        u.backward(torch.ones_like(u))
        self.assertEqual(low.grad, 1 - rand)
        self.assertEqual(high.grad, rand)
        low.grad.zero_()
        high.grad.zero_()

    def test_randperm(self, device="mps"):
        rng_device = None
        for n in (5, 100, 50000, 100000):
            for dtype in (torch.long, torch.half, torch.float):
                if n > 2049 and dtype == torch.half:  # Large n for torch.half will raise an exception, do not test here.
                    continue
                if n > 256 and dtype == torch.bfloat16:
                    continue
                with torch.random.fork_rng(devices=rng_device):
                    res1 = torch.randperm(n, dtype=dtype, device=device)
                res2 = torch.empty(0, dtype=dtype, device=device)
                torch.randperm(n, out=res2, dtype=dtype, device=device)
                self.assertEqual(res1.cpu().sort().values.long(), torch.arange(n, device=device))

        # Default type is long
        for n in (100, 10000):
            self.assertEqual(torch.randperm(n, device=device).dtype, torch.long)

        # randperm of 0 elements is an empty tensor
        res1 = torch.randperm(0)
        res2 = torch.tensor(5, dtype=dtype, device=device)
        torch.randperm(0, out=res2)
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # Test non-contiguous tensors
        for n in (4, 5, 6, 10, 20):
            non_contiguous_tensor = torch.zeros((2, 3), dtype=torch.long, device=device).t()
            self.assertFalse(non_contiguous_tensor.is_contiguous())
            with torch.random.fork_rng(devices=rng_device):
                res = torch.randperm(n, dtype=torch.long, device=device)
            torch.randperm(n, out=non_contiguous_tensor)
            self.assertEqual(res.cpu().sort().values.long(), torch.arange(n, device=device))

    # Test forward maxpool2d
    def test_max_pool2d(self):
        def helper(shape, ks, padding=0, dilation=1, ceil_mode=False, return_indices=False, test_ties=False):

            cpu_x = None
            if (test_ties):
                cpu_x = torch.ones(shape, device='cpu', dtype=torch.float, requires_grad=True)
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            pool = torch.nn.MaxPool2d(kernel_size=ks, padding=padding, dilation=dilation,
                                      ceil_mode=ceil_mode, return_indices=return_indices)

            if (return_indices is False):
                y = pool(x)
                ref_y = pool(cpu_x)

                cpu_grad = torch.ones_like(ref_y)
                grad = cpu_grad.to('mps')

                y.backward(gradient=grad)
                ref_y.backward(gradient=cpu_grad)

                self.assertEqual(y, ref_y)
                self.assertEqual(x.grad, cpu_x.grad)
            else:
                y, idx = pool(x)
                ref_y, ref_idx = pool(cpu_x)

                cpu_grad = torch.ones_like(ref_y)
                grad = cpu_grad.to('mps')

                y.backward(gradient=grad)
                ref_y.backward(gradient=cpu_grad)

                self.assertEqual(y, ref_y)
                self.assertEqual(idx, ref_idx)
                self.assertEqual(x.grad, cpu_x.grad)

        # Test with no batch dimension
        helper((8, 4, 4), ks=2)
        helper((2, 8, 4, 4), ks=2)
        helper((1, 1000, 32, 32), ks=4)
        helper((1, 1000, 1, 4), ks=(1, 4))  # test for max_pool1d
        # Test padding
        helper((1, 1000, 32, 32), ks=4, padding=1)
        helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 1))  # test for max_pool1d
        # Test dilation
        helper((1, 1000, 32, 32), ks=4, dilation=2)
        helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 2))  # test for max_pool1d
        # Test ceil mode
        helper((1, 1000, 32, 32), ks=4, ceil_mode=True)
        helper((1, 1000, 1, 4), ks=(1, 4), ceil_mode=True)  # test for max_pool1d

        # Test return indices
        for test_ties in [False, True]:
            # Test with no batch dimension
            helper((8, 4, 4), ks=2, return_indices=True, test_ties=test_ties)
            helper((2, 8, 4, 4), ks=2, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 32, 32), ks=4, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test padding
            helper((1, 1000, 32, 32), ks=4, padding=1, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 1),
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test dilation
            helper((1, 1000, 32, 32), ks=4, dilation=2, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 2),
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test ceil mode
            helper((1, 1000, 32, 32), ks=4, ceil_mode=True, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), ceil_mode=True,
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d

    def test_adaptive_avg_pool2d_output_size_one(self):
        def helper(size, memory_format):
            x = torch.randint(1, 10, size, dtype=torch.float, device='mps', requires_grad=True)
            if memory_format == 'non_contiguous':
                x = x[::2, ::2, ::2, ::2]
            else:
                x = x.to(memory_format=memory_format)

            net = torch.nn.AdaptiveAvgPool2d((1, 1))
            out = net(x)
            ref_out = x.contiguous().mean((-1, -2)).view((x.size(0), x.size(1), 1, 1))

            out.sum().backward()    # make sure it doesn't crash

            self.assertEqual(out, ref_out)
            if memory_format == torch.channels_last:
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, c, c])
            else:
                self.assertTrue(out.is_contiguous())
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, 1, 1])

        helper((2, 3, 6, 6), torch.contiguous_format)

    def test_masked_scatter(self):
        def helper(shape):
            x_mps = torch.randn(shape, device="mps")
            x_cpu = x_mps.detach().clone().cpu()

            mask_mps = torch.rand(shape, device="mps") < 0.6
            mask_cpu = mask_mps.detach().clone().cpu()

            y_mps = torch.randn(shape, device="mps")
            y_cpu = y_mps.detach().clone().cpu()

            y_mps.masked_scatter_(mask_mps, x_mps)
            y_cpu.masked_scatter_(mask_cpu, x_cpu)

            self.assertEqual(y_mps, y_cpu)
        helper([2, 5])
        helper([10, 10])
        helper([5, 10, 3])
        helper([10, 5, 10, 3])
        helper([10, 5, 10, 3, 20])

    def test_masked_fill(self):
        device = "mps"
        dtype = torch.float32
        mask_dtype = torch.bool
        num_dest = 10

        dst = torch.zeros(num_dest, dtype=dtype, device=device)
        mask = torch.randint(2, (num_dest,), dtype=mask_dtype, device=device)
        val = random.random()
        dst2 = torch.zeros(num_dest, dtype=dtype)
        mask_cpu = mask.to("cpu")

        dst.masked_fill_(mask, val)
        for i in range(num_dest):
            if mask_cpu[i]:
                dst2[i] = val
        self.assertEqual(dst.to("cpu"), dst2, atol=0, rtol=0)

        # Regression test for https://github.com/pytorch/pytorch/issues/143477
        # Allocating 48x25x1024x1024 tensor crashes on MacOS-13
        mask_bool = torch.triu(torch.ones(1024, 1024, device=device), diagonal=1).bool()
        attn_scores = torch.rand(48, 25, 1024, 1024, device=device)
        attn_scores.masked_fill_(mask_bool, 0)

    def test_masked_fill__non_contiguous(self):
        shape = (3, 5)

        x_mps = torch.randn(shape, device="mps")
        x_cpu = x_mps.detach().clone().cpu()
        mask_mps = torch.zeros(shape, device="mps", dtype=torch.bool)
        mask_cpu = mask_mps.detach().clone().cpu()

        x_mps_strided = x_mps.T
        x_cpu_strided = x_cpu.T

        x_mps_strided.masked_fill_(mask_mps.T, float("-inf"))
        x_cpu_strided.masked_fill_(mask_cpu.T, float("-inf"))

        self.assertEqual(x_mps_strided, x_cpu_strided)
        self.assertFalse((x_mps_strided == float("-inf")).any())

    def test_nhwc_operation(self):
        def helper(shape, channels_last=False):
            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # This passes
            self.assertEqual(x, cpu_x)

        helper((2, 2, 2, 2), True)

    # Test forward batch norm
    def test_batch_norm(self):
        def helper(shape, eps=1, momentum=0.1, wts=False, training=False, channels_last=False,
                   track_running_stats=True, test_module=False):

            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            mean_shape = [shape[1]]
            cpu_running_mean = None
            cpu_running_var = None
            running_mean = None
            running_var = None
            if (track_running_stats):
                mean_arr = (240 - 140) * np.random.random_sample(size=mean_shape) + 140
                cpu_running_mean = torch.tensor(mean_arr, device='cpu', dtype=torch.float)
                var_arr = 32 * np.random.random_sample(size=mean_shape)
                cpu_running_var = torch.tensor(var_arr, device='cpu', dtype=torch.float)
                running_mean = cpu_running_mean.detach().clone().to('mps')
                running_var = cpu_running_var.detach().clone().to('mps')

            weight = None
            cpu_weight = None
            bias = None
            cpu_bias = None
            if (wts):
                cpu_weight = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                weight = cpu_weight.detach().clone().to('mps').requires_grad_()
                cpu_bias = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = None
            ref_y = None

            if (not test_module):
                y = torch.nn.functional.batch_norm(x, running_mean, running_var,
                                                   weight=weight,
                                                   bias=bias,
                                                   training=training,
                                                   momentum=momentum, eps=eps)
                ref_y = torch.nn.functional.batch_norm(cpu_x, cpu_running_mean, cpu_running_var,
                                                       weight=cpu_weight,
                                                       bias=cpu_bias,
                                                       training=training,
                                                       momentum=momentum, eps=eps)

            else:

                batchnorm_op = None
                mps_batchnorm_op = None

                if (len(shape) == 3):
                    batchnorm_op = torch.nn.BatchNorm1d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm1d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')
                elif (len(shape) == 4):
                    batchnorm_op = torch.nn.BatchNorm2d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm2d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')
                elif (len(shape) == 5):
                    batchnorm_op = torch.nn.BatchNorm3d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm3d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')

                if (track_running_stats):
                    batchnorm_op.running_mean = cpu_running_mean
                    batchnorm_op.running_var = cpu_running_var
                    mps_batchnorm_op.running_mean = running_mean
                    mps_batchnorm_op.running_var = running_var
                if (wts):
                    batchnorm_op.weight = torch.nn.Parameter(cpu_weight)
                    batchnorm_op.bias = torch.nn.Parameter(cpu_bias)
                    mps_batchnorm_op.weight = torch.nn.Parameter(weight)
                    mps_batchnorm_op.bias = torch.nn.Parameter(bias)

                ref_y = batchnorm_op(cpu_x)
                y = mps_batchnorm_op(x)

            self.assertEqual(y, ref_y)
            if (not test_module):
                self.assertEqual(running_mean, cpu_running_mean)
                self.assertEqual(running_var, cpu_running_var)
            else:
                self.assertEqual(mps_batchnorm_op.running_mean, batchnorm_op.running_mean)
                self.assertEqual(mps_batchnorm_op.running_var, batchnorm_op.running_var)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')
            ref_y.backward(gradient=cpu_grad)
            y.backward(gradient=grad)

            self.assertEqual(x.grad, cpu_x.grad)
            if (wts):
                if (not test_module):
                    self.assertEqual(weight.grad, cpu_weight.grad)
                    self.assertEqual(bias.grad, cpu_bias.grad)
                else:
                    self.assertEqual(mps_batchnorm_op.weight.grad, batchnorm_op.weight.grad)
                    self.assertEqual(mps_batchnorm_op.bias.grad, batchnorm_op.bias.grad)

        for shape in [(2, 3, 2, 2), (2, 3, 2, 2, 2), (2, 3, 2)]:
            for test_module in [False, True]:
                for track_running_stats in [True, False]:
                    for channels_last in [False]:
                        if (channels_last and len(shape) != 4):
                            continue
                        # Running stats must be tracked in eval mode
                        if (track_running_stats):
                            helper(shape, eps=1e-5, momentum=1, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1e-05, momentum=0.1, wts=False, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1e-5, momentum=1.0, wts=False, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1, momentum=1, wts=True, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=3, momentum=0.67, wts=True, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1e-05, momentum=0.1, wts=False, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1e-5, momentum=1.0, wts=False, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1, momentum=1, wts=True, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=3, momentum=0.67, wts=True, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)

    def test_batch_norm_backward(self):
        inputs = torch.rand(1, 8, 4, 4, device="mps", requires_grad=True)
        x = torch.nn.BatchNorm2d(8).to("mps")
        y = torch.nn.BatchNorm2d(8).to("mps")
        y.weight.requires_grad = False
        y.bias.requires_grad = False
        outputs = y(x(inputs))
        # This used to crash, see https://github.com/pytorch/pytorch/issues/98602
        outputs.sum().backward()

    def test_batch_norm_slices(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/133520
        bn_cpu = nn.BatchNorm2d(100, affine=False, device='cpu')
        bn_mps = nn.BatchNorm2d(100, affine=False, device='mps')

        x_cpu = torch.randn(100, 100, 35, 45).to('cpu')
        x_mps = x_cpu.to('mps')

        res_cpu = bn_cpu(x_cpu[5:])
        res_mps = bn_mps(x_mps[5:])

        self.assertEqual(res_cpu, res_mps)

    def test_batch_norm_backward_weight_bias_gradients(self):
        # See issue: https://github.com/pytorch/pytorch/issues/156555
        N, C, L = 4, 3, 5
        x = torch.randn(N, C, L)
        y = torch.randn(N, C, L)
        bn_cpu = nn.BatchNorm1d(C, affine=True).cpu().train()
        bn_mps = nn.BatchNorm1d(C, affine=True).to('mps').train()
        bn_mps.load_state_dict(bn_cpu.state_dict())

        out_cpu = bn_cpu(x)
        out_mps = bn_mps(x.to('mps'))

        loss_cpu = ((out_cpu - y) ** 2).mean()
        loss_mps = ((out_mps - y.to('mps')) ** 2).mean()
        loss_cpu.backward()
        loss_mps.backward()

        self.assertEqual(bn_cpu.weight.grad, bn_mps.weight.grad, atol=1e-5, rtol=1e-5)
        self.assertEqual(bn_cpu.bias.grad, bn_mps.bias.grad, atol=1e-5, rtol=1e-5)

    def test_layer_norm_backward(self):
        inputs = torch.rand(4, 4, device="mps", requires_grad=True)
        x = torch.nn.LayerNorm(4).to("mps")
        y = torch.nn.LayerNorm(4).to("mps")
        y.weight.requires_grad = False
        y.bias.requires_grad = False
        outputs = y(x(inputs))
        # This used to crash, see https://github.com/pytorch/pytorch/issues/98602
        outputs.sum().backward()

    def test_norm(self):
        a = torch.arange(9, dtype=torch.float, device="mps") - 4
        b = a.reshape((3, 3))

        a_cpu = torch.arange(9, dtype=torch.float, device="cpu") - 4
        b_cpu = a_cpu.reshape((3, 3))

        res = torch.norm(a)
        res_cpu = torch.norm(a_cpu)
        self.assertEqual(res, res_cpu)

        res = torch.norm(b)
        res_cpu = torch.norm(b_cpu)
        self.assertEqual(res, res_cpu)

        res = torch.norm(a, float('inf'))
        res_cpu = torch.norm(a_cpu, float('inf'))
        self.assertEqual(res, res_cpu)

        res = torch.norm(b, float('inf'))
        res_cpu = torch.norm(b_cpu, float('inf'))
        self.assertEqual(res, res_cpu)

        c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float, device="mps")
        c_cpu = torch.tensor([[1, 2, 3], [-1, 1, 4]] , dtype=torch.float, device="cpu")

        res = torch.norm(c, dim=0)
        res_cpu = torch.norm(c_cpu, dim=0)
        self.assertEqual(res, res_cpu)

        res = torch.norm(c, dim=1)
        res_cpu = torch.norm(c_cpu, dim=1)
        self.assertEqual(res, res_cpu)

        res = torch.norm(c, p=1, dim=1)
        res_cpu = torch.norm(c_cpu, p=1, dim=1)
        self.assertEqual(res, res_cpu)

        d = torch.arange(8, dtype=torch.float, device="mps").reshape(2, 2, 2)
        d_cpu = torch.arange(8, dtype=torch.float, device="cpu").reshape(2, 2, 2)

        res = torch.norm(d, dim=(1, 2))
        res_cpu = torch.norm(d_cpu, dim=(1, 2))
        self.assertEqual(res, res_cpu)

        res = torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
        res_cpu = torch.norm(d_cpu[0, :, :]), torch.norm(d_cpu[1, :, :])
        self.assertEqual(res, res_cpu)

    def test_linalg_vector_norm(self):
        x_mps = torch.tensor([0, 0, 0, 2, 3], dtype=torch.float, device="mps")
        x_cpu = x_mps.detach().clone().cpu()

        res_mps = torch.linalg.vector_norm(x_mps, ord=0)
        res_cpu = torch.linalg.vector_norm(x_cpu, ord=0)
        self.assertEqual(res_mps, res_cpu)

        a_mps = torch.arange(27, dtype=torch.float, device="mps") - 4
        a_cpu = torch.arange(27, dtype=torch.float, device="cpu") - 4

        B_mps = a_mps.reshape(3, 3, 3)
        B_cpu = a_cpu.reshape(3, 3, 3)

        res_mps = torch.linalg.vector_norm(a_mps, ord=3.5)
        res_cpu = torch.linalg.vector_norm(a_cpu, ord=3.5)
        self.assertEqual(res_mps, res_cpu)

        res_mps = torch.linalg.vector_norm(B_mps, ord=3.5)
        res_cpu = torch.linalg.vector_norm(B_cpu, ord=3.5)
        self.assertEqual(res_mps, res_cpu)

        for dim in range(B_mps.dim()):
            res_mps = torch.linalg.vector_norm(B_mps, ord=3.5, dim=dim)
            res_cpu = torch.linalg.vector_norm(B_cpu, ord=3.5, dim=dim)
            self.assertEqual(res_mps, res_cpu)

    def test_linalg_lu_factor_ex(self):
        from torch.testing._internal.common_utils import make_fullrank_matrices_with_distinct_singular_values

        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device="cpu", dtype=torch.float32)

        def run_lu_factor_ex_test(size, *batch_dims, check_errors, atol=1e-5, rtol=1e-6):
            input_cpu = make_arg(*batch_dims, size, size)
            input_mps = input_cpu.to('mps')
            out_cpu = torch.linalg.lu_factor_ex(input_cpu, check_errors=check_errors)
            out_mps = torch.linalg.lu_factor_ex(input_mps, check_errors=check_errors)
            self.assertEqual(out_cpu, out_mps, atol=atol, rtol=rtol)

            out_cpu = torch.linalg.lu_factor_ex(input_cpu.mT, check_errors=check_errors)
            out_mps = torch.linalg.lu_factor_ex(input_mps.mT, check_errors=check_errors)
            self.assertEqual(out_cpu, out_mps, atol=atol, rtol=rtol)

        # test with different even/odd matrix sizes
        matrix_sizes = [1, 2, 3, 4]
        # even/odd batch sizes
        batch_sizes = [1, 2, 4]

        for check_errors in [True, False]:
            for size in matrix_sizes:
                for batch_size in batch_sizes:
                    run_lu_factor_ex_test(size, batch_size, check_errors=check_errors)
        # test >3D matrices
        run_lu_factor_ex_test(32, 10, 10, check_errors=False)
        run_lu_factor_ex_test(32, 2, 2, 10, 10, check_errors=True)
        # big matrix check with batch size > 1
        run_lu_factor_ex_test(256, 2, check_errors=False, atol=3e-5, rtol=5e-6)

    def test_linalg_lu_factor_singular(self):
        # Explicit singular matrix
        A = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device="mps")

        with self.assertRaisesRegex(RuntimeError, "result in a division by zero"):
            torch.linalg.lu_factor(A)

    def test_linalg_solve(self):
        from torch.testing._internal.common_utils import make_fullrank_matrices_with_distinct_singular_values

        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device="cpu", dtype=torch.float32)

        def run_linalg_solve_test(size, *batch_dims):
            A_cpu = make_arg(*batch_dims, size, size)
            A_mps = A_cpu.to('mps')

            for left in [True, False]:
                if left:
                    b_cpu = torch.randn(*batch_dims, size, 3, device='cpu', dtype=torch.float32)
                else:
                    b_cpu = torch.randn(*batch_dims, 3, size, device='cpu', dtype=torch.float32)

                b_mps = b_cpu.to('mps')

                # Solve the system
                X_cpu = torch.linalg.solve(A_cpu, b_cpu, left=left)
                X_mps = torch.linalg.solve(A_mps, b_mps, left=left)
                self.assertEqual(X_cpu, X_mps)

                # Test with transposed matrices
                X_cpu_t = torch.linalg.solve(A_cpu.mT, b_cpu, left=left)
                X_mps_t = torch.linalg.solve(A_mps.mT, b_mps, left=left)
                self.assertEqual(X_cpu_t, X_mps_t)

        # test with different even/odd matrix sizes
        matrix_sizes = [1, 2, 3, 4]
        # even/odd batch sizes
        batch_sizes = [1, 2, 4]

        for size in matrix_sizes:
            for batch_size in batch_sizes:
                run_linalg_solve_test(size, batch_size)

        # test >3D matrices
        run_linalg_solve_test(32, 10, 10)
        run_linalg_solve_test(32, 2, 2, 2, 2, 10, 10)

    def test_linalg_solve_singular(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/163962

        # Explicit singular matrix
        A = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device="mps")
        b = torch.rand_like(A)

        with self.assertRaisesRegex(RuntimeError, "input matrix is singular"):
            torch.linalg.solve(A, b)

    def test_linalg_solve_with_broadcasting(self):
        from functools import partial
        import torch
        from torch.testing._internal.common_utils import (
            make_fullrank_matrices_with_distinct_singular_values
        )

        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device="cpu", dtype=torch.float32)

        batch_size = 4
        size = 3

        A_cpu = make_arg(batch_size, size, size)
        A_mps = A_cpu.to('mps')

        for left in [True, False]:
            b_cpu = torch.randn(batch_size, size, device='cpu', dtype=torch.float32)
            b_mps = b_cpu.to('mps')

            if left:
                b_cpu = b_cpu.unsqueeze(-1)
                b_mps = b_mps.unsqueeze(-1)
            else:
                b_cpu = b_cpu.view(batch_size, 1, size)
                b_mps = b_mps.view(batch_size, 1, size)

            X_cpu = torch.linalg.solve(A_cpu, b_cpu, left=left)
            X_mps = torch.linalg.solve(A_mps, b_mps, left=left)
            self.assertEqual(X_cpu, X_mps)

            X_cpu_t = torch.linalg.solve(A_cpu.mT, b_cpu, left=left)
            X_mps_t = torch.linalg.solve(A_mps.mT, b_mps, left=left)
            self.assertEqual(X_cpu_t, X_mps_t)

    def test_linalg_det(self):
        from torch.testing._internal.common_utils import make_fullrank_matrices_with_distinct_singular_values

        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device="cpu", dtype=torch.float32)

        def run_det_test(size, *batch_dims):
            input_cpu = make_arg(*batch_dims, size, size)
            input_mps = input_cpu.to('mps')
            out_cpu = torch.linalg.det(input_cpu)
            out_mps = torch.linalg.det(input_mps)
            self.assertEqual(out_cpu, out_mps)

            # non-contiguous matrices
            input_cpu_T = input_cpu.mT
            input_mps_T = input_mps.mT
            out_cpu_T = torch.linalg.det(input_cpu_T)
            out_mps_T = torch.linalg.det(input_mps_T)
            self.assertEqual(out_cpu_T, out_mps_T)

        # test with different even/odd matrix sizes
        matrix_sizes = [2, 3, 4]
        # even/odd batch sizes
        batch_sizes = [1, 2, 4]

        for size in matrix_sizes:
            for batch_size in batch_sizes:
                run_det_test(size, batch_size)

        # test >3D matrices
        run_det_test(32, 10, 10)
        run_det_test(32, 2, 2, 10, 10)

    def test_layer_norm(self):
        def helper(input_shape, normalized_shape, eps=1e-05, elementwise_affine=True, dtype=torch.float32, non_contiguous=False):
            cpu_x = torch.randn(input_shape, device='cpu', dtype=dtype, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()
            if non_contiguous:
                x = x.mT
                cpu_x = cpu_x.mT
                normalized_shape[-1], normalized_shape[-2] = normalized_shape[-2], normalized_shape[-1]

            cpu_op = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device='cpu', dtype=dtype)
            mps_op = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device='mps', dtype=dtype)
            cpu_wt = torch.randn(normalized_shape, device='cpu', dtype=dtype, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()
            cpu_bias = torch.randn(normalized_shape, device='cpu', dtype=dtype, requires_grad=True)
            bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            if (elementwise_affine):
                cpu_op.weight = torch.nn.Parameter(cpu_wt)
                mps_op.weight = torch.nn.Parameter(wt)
                cpu_op.bias = torch.nn.Parameter(cpu_bias)
                mps_op.bias = torch.nn.Parameter(bias)

            cpu_result = cpu_op(cpu_x)
            result = mps_op(x)

            cpu_grad = torch.randn(cpu_result.shape)
            grad = cpu_grad.to('mps')

            cpu_result.backward(cpu_grad)
            result.backward(grad)

            self.assertEqual(result, cpu_result)
            self.assertEqual(x.grad, cpu_x.grad)
            if (elementwise_affine):
                self.assertEqual(mps_op.weight.grad, cpu_op.weight.grad)
                self.assertEqual(mps_op.bias.grad, cpu_op.bias.grad)

        for (elementwise_affine, non_contiguous) in itertools.product([True, False], [True, False]):
            helper((2, 2, 2, 2), [2, 2], elementwise_affine=elementwise_affine, non_contiguous=non_contiguous)
            helper((2, 3, 4, 5), [4, 5], elementwise_affine=elementwise_affine, non_contiguous=non_contiguous)
            helper((2, 3, 4, 5, 6), [4, 5, 6], elementwise_affine=elementwise_affine, non_contiguous=non_contiguous)

        # Regression test for https://github.com/pytorch/pytorch/issues/96113
        torch.nn.LayerNorm((16,), elementwise_affine=True).to("mps")(torch.randn(1, 2, 16).to("mps", dtype=torch.float16))

    def test_ifft(self):
        # See: https://github.com/pytorch/pytorch/issues/124096
        device = torch.device("mps")

        N = 64
        signal = torch.rand(N, device=device)
        fft_result = torch.fft.rfft(signal)
        ifft_result = torch.fft.irfft(fft_result, n=signal.shape[0])

        # Expecting the inverted to yield the original signal
        self.assertEqual(ifft_result, signal)

    def test_fftfreq(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/135223
        freq_cpu = torch.fft.fftfreq(10**4, device='cpu')
        freq_mps = torch.fft.fftfreq(10**4, device='mps')
        self.assertEqual(freq_cpu, freq_mps)

    def test_instance_norm(self):
        def helper(shape, eps=1, momentum=0.1, wts=False, channels_last=False, track_running_stats=True, test_module=False):

            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            mean_shape = [shape[1]]
            cpu_running_mean = None
            cpu_running_var = None
            running_mean = None
            running_var = None
            if (track_running_stats):
                mean_arr = (240 - 140) * np.random.random_sample(size=mean_shape) + 140
                cpu_running_mean = torch.tensor(mean_arr, device='cpu', dtype=torch.float)
                var_arr = 32 * np.random.random_sample(size=mean_shape)
                cpu_running_var = torch.tensor(var_arr, device='cpu', dtype=torch.float)
                running_mean = cpu_running_mean.detach().clone().to('mps')
                running_var = cpu_running_var.detach().clone().to('mps')

            weight = None
            cpu_weight = None
            bias = None
            cpu_bias = None
            if (wts):
                cpu_weight = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                weight = cpu_weight.detach().clone().to('mps').requires_grad_()
                cpu_bias = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = None
            ref_y = None

            if (not test_module):
                ref_y = torch.nn.functional.instance_norm(cpu_x, cpu_running_mean, cpu_running_var,
                                                          weight=cpu_weight,
                                                          bias=cpu_bias,
                                                          momentum=momentum, eps=eps)
                y = torch.nn.functional.instance_norm(x, running_mean, running_var,
                                                      weight=weight,
                                                      bias=bias,
                                                      momentum=momentum, eps=eps)

            else:

                instancenorm_op = None
                mps_instancenorm_op = None

                if (len(shape) == 3):
                    instancenorm_op = torch.nn.InstanceNorm1d(shape[1],
                                                              eps=eps,
                                                              momentum=momentum,
                                                              affine=wts,
                                                              track_running_stats=track_running_stats,
                                                              device='cpu')
                    mps_instancenorm_op = torch.nn.InstanceNo

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 37 class(es): MpsMemoryLeakCheck, TestAutocastMPS, Foo, Model, Model, TestCaseMPS, TestMemoryLeak, TestPixelShuffle, MPSReluTest, MatmulTest, MPSLeakyReluTest, TestAvgPool, TestMPS, TestLargeTensors, TestLogical, TestSmoothL1Loss, TestNLLLoss, TestTopK, TestNNMPS, Layer

### Functions
This file defines 895 function(s): skipMPSMemoryLeakCheckIf, dec, __init__, __enter__, __exit__, test_matmul_autocast, test_scaled_dot_product_attention_autocast, test_conv_transpose3d_autocast_fp32, test_conv3d_autocast, __init__, forward, test_gradscaler_mps, __init__, forward, helper, test_non_fast_path_amp_unscale, __init__, forward, helper, __init__, assertLeaksNoMpsTensors, wrap_with_mps_policy, wrap_with_mps_memory_check, test_mps_memory_leak_detection, no_leak, leak_gpu0, test_copy_cast_no_leak, step, test_pixel_shuffle_unshuffle, _test_pixel_shuffle_unshuffle_helper


## Key Components

The file contains 44954 words across 13031 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 570742 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
