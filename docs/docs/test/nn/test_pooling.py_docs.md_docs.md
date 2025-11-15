# Documentation: `docs/test/nn/test_pooling.py_docs.md`

## File Metadata

- **Path**: `docs/test/nn/test_pooling.py_docs.md`
- **Size**: 53,962 bytes (52.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/nn/test_pooling.py`

## File Metadata

- **Path**: `test/nn/test_pooling.py`
- **Size**: 86,221 bytes (84.20 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: nn"]
import itertools
import math
import operator
import os
import random
import subprocess
import sys
import unittest
from functools import partial, reduce
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inf, nan
from torch.autograd import gradcheck, gradgradcheck
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfMPS,
    expectedFailureMeta,
    expectedFailureMPS,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipCUDAIfRocm,
    TEST_WITH_ROCM,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_nn import (
    _test_bfloat16_ops,
    _test_module_empty_input,
    NNTestCase,
)
from torch.testing._internal.common_utils import (
    gcIfJetson,
    instantiate_parametrized_tests,
    parametrize as parametrize_test,
    run_tests,
    set_default_dtype,
    skipIfTorchDynamo,
    slowTest,
    subtest,
    TEST_WITH_UBSAN,
    TestCase,
)


class TestAvgPool(TestCase):
    def _sum_pool2d(self, x, kernel_size):
        windows = torch.nn.functional.unfold(
            x, kernel_size=kernel_size, stride=kernel_size
        )
        return torch.sum(windows, dim=1)

    def _sum_pool3d(self, x, kernel_size):
        # Because unfold does not support 3D sliding window we will split tensor to multiple tensors and calculate sum
        h = kernel_size[0]
        splited_x = [t.sum(0) for t in x.split(h) if t.size(0) == h]
        # sum_pool2d assumes tensor in (1, 1, n, m) view, so unsqueeze two times
        splited_x = [
            self._sum_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size[1:])
            for t in splited_x
        ]
        joined_x = torch.cat(splited_x)
        return joined_x.view(1, joined_x.numel())

    def _avg_pool2d(self, x, kernel_size):
        size = reduce(operator.mul, kernel_size)
        return self._sum_pool2d(x, kernel_size) / size

    def _avg_pool3d(self, x, kernel_size):
        size = reduce(operator.mul, kernel_size)
        return self._sum_pool3d(x, kernel_size) / size

    def test_doubletensor_avg_pool2d(self):
        n, m = 5, 8
        input = torch.rand(1, 1, n, m, dtype=torch.double)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                actual = torch.nn.functional.avg_pool2d(input[0], (i, j))
                actual = actual.view(1, actual.numel())
                expected = self._avg_pool2d(input, (i, j))
                self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_doubletensor_avg_pool2d_with_divisor(self):
        n, m = 3, 3
        input = torch.rand(1, 1, n, m, dtype=torch.double)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for divisor in [1, 7, i * j]:
                    actual = F.avg_pool2d(input[0], (i, j), divisor_override=divisor)
                    actual = actual.view(1, actual.numel())
                    expected = self._sum_pool2d(input, (i, j)) / divisor
                    self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_doubletensor_avg_pool3d(self):
        h, w, d = 5, 6, 7
        input = torch.rand(h, w, d, dtype=torch.double)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                for k in range(1, d + 1):
                    actual = torch.nn.functional.avg_pool3d(
                        input.unsqueeze(0), (i, j, k)
                    )
                    actual = actual.view(1, actual.numel())
                    expected = self._avg_pool3d(input, (i, j, k))
                    self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_doubletensor_avg_pool3d_with_divisor(self):
        h, w, d = 6, 5, 7
        input = torch.rand(h, w, d, dtype=torch.double)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                for k in range(1, d + 1):
                    for divisor in [1, 7, i * j]:
                        actual = torch.nn.functional.avg_pool3d(
                            input.unsqueeze(0), (i, j, k), divisor_override=divisor
                        )
                        actual = actual.view(1, actual.numel())
                        expected = self._sum_pool3d(input, (i, j, k)) / divisor
                        self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_avg_pool1d_ceil_mode(self):
        # Regression test for gh-36977
        x = 10 * torch.randn((1, 16, 4))
        y = torch.nn.functional.avg_pool1d(
            x, ceil_mode=True, count_include_pad=True, kernel_size=1, stride=2
        )
        self.assertTrue(not torch.isnan(y).any())

        if TEST_CUDA:
            y = torch.nn.functional.avg_pool1d(
                x.to("cuda"),
                ceil_mode=True,
                count_include_pad=True,
                kernel_size=1,
                stride=2,
            )
            self.assertTrue(not torch.isnan(y).any())

    def test_avg_pool2d_ceil_mode(self):
        # Regression test for gh-36977
        x = 10 * torch.randn((1, 16, 4, 4))
        y = torch.nn.functional.avg_pool2d(
            x,
            ceil_mode=True,
            count_include_pad=True,
            kernel_size=(1, 2),
            padding=(0, 1),
            stride=2,
        )
        self.assertTrue(not torch.isnan(y).any())

        if TEST_CUDA:
            y = torch.nn.functional.avg_pool2d(
                x.to("cuda"),
                ceil_mode=True,
                count_include_pad=True,
                kernel_size=(1, 2),
                padding=(0, 1),
                stride=2,
            )
            self.assertTrue(not torch.isnan(y).any())

    def test_avg_pool3d_ceil_mode(self):
        # Regression test for gh-36977
        x = 10 * torch.randn((1, 16, 4, 4, 4))
        y = torch.nn.functional.avg_pool3d(
            x, ceil_mode=True, count_include_pad=True, kernel_size=(1, 2, 3), stride=2
        )
        self.assertTrue(not torch.isnan(y).any())

        if TEST_CUDA:
            y = torch.nn.functional.avg_pool3d(
                x.to("cuda"),
                ceil_mode=True,
                count_include_pad=True,
                kernel_size=(1, 2, 3),
                stride=2,
            )
            self.assertTrue(not torch.isnan(y).any())


class TestPoolingNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def test_adaptive_pooling_size_none(self):
        for numel in (2, 3):
            for pool_type in ("Max", "Avg"):
                cls_name = f"Adaptive{pool_type}Pool{numel}d"
                module_cls = getattr(nn, cls_name)
                output_size = (2,) * (numel - 1) + (None,)
                module = module_cls(output_size)

                input = torch.randn((4,) * (numel + 1))
                output = module(input)
                self.assertEqual(output.size(), (4,) + (2,) * (numel - 1) + (4,))

    @unittest.skipIf(TEST_WITH_UBSAN, "signed integer overflow error with UBSAN")
    def test_adaptive_pooling_size_overflow(self):
        # 0x0x3fffffffffffffff * 2 * 2 = 0xfffffffffffffffc = -4 as int64_t
        # Tensor::numel() return int64_t, so following check that negative allocs are correctly handled
        self.assertRaises(
            RuntimeError,
            lambda: torch.nn.AdaptiveMaxPool1d(0x3FFFFFFFFFFFFFFF)(
                torch.empty([2, 2, 2])
            ),
        )

    def test_adaptive_pooling_avg_nhwc(self):
        device_list = ["cpu"]
        if TEST_CUDA:
            device_list.append("cuda")

        for device in device_list:
            input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32).to(device)
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            grad = torch.randint(1, 10, (4, 8, 7, 7), dtype=torch.float32).to(device)
            pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)

            out = pool(input)
            out.backward(grad)
            ref_out = ref_pool(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(input.grad, ref_input.grad)

    def test_adaptive_pooling_avg_nhwc_non_contiguous(self):
        device_list = ["cpu"]
        if TEST_CUDA:
            device_list.append("cuda")

        for device in device_list:
            input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32).to(device)
            input = input.contiguous(memory_format=torch.channels_last)
            input = input[:, ::2, :, :].requires_grad_()
            grad = torch.randint(1, 10, (4, 8, 7, 7), dtype=torch.float32).to(device)
            grad = grad[:, ::2, :, :]
            pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)

            out = pool(input)
            out.backward(grad)
            ref_out = ref_pool(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(input.grad, ref_input.grad)

    def test_adaptive_pooling_lower_precision(self):
        def _test_adaptive_pooling_lower_precision(
            self, device, dtype, mod, memory_format
        ):
            input = torch.randint(1, 10, (3, 19, 8, 8), dtype=torch.float32)
            input = input.to(device).to(memory_format=memory_format).requires_grad_()
            pool = mod((7, 7)).to(device)

            input2 = input.detach().clone().to(dtype=dtype).requires_grad_(True)

            out = pool(input)
            out.sum().backward()
            out2 = pool(input2)
            out2.sum().backward()

            self.assertTrue(out2.is_contiguous(memory_format=memory_format))
            self.assertEqual(out2.dtype, dtype)
            self.assertEqual(input2.grad.dtype, dtype)
            self.assertEqual(out, out2.float(), atol=0.1, rtol=0)
            self.assertEqual(input.grad, input2.grad.float(), atol=0.1, rtol=0)

        device_list = ["cpu"]
        for device in device_list:
            for dtype in [torch.bfloat16, torch.float16]:
                _test_adaptive_pooling_lower_precision(
                    self,
                    device,
                    dtype,
                    torch.nn.AdaptiveAvgPool2d,
                    torch.contiguous_format,
                )
                _test_adaptive_pooling_lower_precision(
                    self, device, dtype, torch.nn.AdaptiveAvgPool2d, torch.channels_last
                )
                _test_adaptive_pooling_lower_precision(
                    self,
                    device,
                    dtype,
                    torch.nn.AdaptiveMaxPool2d,
                    torch.contiguous_format,
                )
                _test_adaptive_pooling_lower_precision(
                    self, device, dtype, torch.nn.AdaptiveMaxPool2d, torch.channels_last
                )

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @largeTensorTest("12GB", device="cuda")
    def test_adaptive_pooling_avg_nhwc_launch_config_backward(self):
        input = torch.randint(
            1, 10, (1, 32, 2**17 + 1, 32), dtype=torch.float32, device="cuda"
        )
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        grad = torch.randint(1, 10, (1, 32, 10, 32), dtype=torch.float32, device="cuda")

        pool = torch.nn.AdaptiveAvgPool2d((10, 32)).cuda()

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_pool = torch.nn.AdaptiveAvgPool2d((10, 32)).cuda()

        out = pool(input)
        out.backward(grad)
        ref_out = ref_pool(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(input.grad, ref_input.grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @largeTensorTest("12GB", device="cuda")
    def test_adaptive_pooling_avg_nhwc_launch_config_forward(self):
        input = torch.randint(
            1, 10, (1, 32, 16, 16), dtype=torch.float32, device="cuda"
        )
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        pool = torch.nn.AdaptiveAvgPool2d((2**17 + 1, 32)).cuda()

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_pool = torch.nn.AdaptiveAvgPool2d((2**17 + 1, 32)).cuda()

        out = pool(input)
        ref_out = ref_pool(ref_input)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_adaptive_avg_pooling_overflow(self):
        input = torch.randint(
            -256, 256, (20, 32, 256, 256), dtype=torch.half, device="cuda"
        )
        avg_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        out = avg_pool(input)
        self.assertFalse(torch.isinf(out).any())
        self.assertFalse(torch.isnan(out).any())

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_adaptive_avg_pooling_nhwc_overflow(self):
        input = torch.randint(
            -256, 256, (20, 32, 256, 256), dtype=torch.half, device="cuda"
        )
        input = input.contiguous(memory_format=torch.channels_last)
        avg_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        out = avg_pool(input)
        self.assertFalse(torch.isinf(out).any())
        self.assertFalse(torch.isnan(out).any())

    def test_MaxUnpool2d_output_size(self):
        m = nn.MaxPool2d(3, stride=2, return_indices=True)
        mu = nn.MaxUnpool2d(3, stride=2)
        big_t = torch.rand(1, 1, 6, 6)
        big_t[0][0][4][4] = 100
        output_big, indices_big = m(big_t)
        self.assertRaises(RuntimeError, lambda: mu(output_big, indices_big))

        small_t = torch.rand(1, 1, 5, 5)
        for i in range(0, 4, 2):
            for j in range(0, 4, 2):
                small_t[:, :, i, j] = 100
        output_small, indices_small = m(small_t)
        for h in range(3, 10):
            for w in range(3, 10):
                if 4 <= h <= 6 and 4 <= w <= 6:
                    size = (h, w)
                    if h == 6:
                        size = (1, 1) + size

                    mu(output_small, indices_small, output_size=size)
                else:
                    self.assertRaises(
                        ValueError, lambda: mu(output_small, indices_small, (h, w))
                    )

    def test_max_unpool2d_nhwc_cpu(self):
        input = torch.randn(2, 10, 9, 9).float().cpu()
        input = input.contiguous(memory_format=torch.channels_last)
        ref_input = input.clone().contiguous()

        pool = nn.MaxPool2d(3, stride=2, return_indices=True).cpu()
        ref_pool = nn.MaxPool2d(3, stride=2, return_indices=True).cpu()

        out, ind = pool(input)
        ref_out, ref_ind = ref_pool(ref_input)
        out.requires_grad_()
        ref_out.requires_grad_()

        unpool = nn.MaxUnpool2d(3, stride=2).cpu()
        ref_unpool = nn.MaxUnpool2d(3, stride=2).cpu()

        upout = unpool(out, ind)
        ref_upout = ref_unpool(ref_out, ref_ind)

        grad = torch.randn(upout.size()).float().cpu()
        grad = grad.contiguous(memory_format=torch.channels_last)
        ref_grad = grad.clone().contiguous()

        upout.backward(grad)
        ref_upout.backward(ref_grad)

        self.assertTrue(upout.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_upout.is_contiguous())
        self.assertTrue(torch.allclose(upout, ref_upout))
        self.assertTrue(torch.allclose(out.grad, ref_out.grad))

    def test_max_unpool(self):
        with set_default_dtype(torch.double):
            # Test 1D
            output, indices = F.max_pool1d(
                torch.randn([1, 1, 4]), 2, stride=2, return_indices=True
            )
            self.assertEqual(
                F.max_unpool1d(output, indices, 2),
                F.max_unpool1d(output, indices, 2, stride=2),
            )

            # Test list / tuple passed as argument to max_unpool1d
            input = torch.randn([1, 1, 5], requires_grad=True)
            output, indices = F.max_pool1d(input, 2, stride=2, return_indices=True)
            self.assertEqual(
                F.max_unpool1d(output, indices, 2, stride=2, output_size=input.shape),
                F.max_unpool1d(output, indices, 2, stride=2, output_size=input.size()),
            )
            gradcheck(F.max_unpool1d, (output, indices, 2), check_forward_ad=True)

            # Test 2D
            output, indices = F.max_pool2d(
                torch.randn([1, 1, 4, 4], requires_grad=True),
                2,
                stride=2,
                return_indices=True,
            )
            self.assertEqual(
                F.max_unpool2d(output, indices, 2),
                F.max_unpool2d(output, indices, 2, stride=2),
            )
            gradcheck(F.max_unpool2d, (output, indices, 2), check_forward_ad=True)

            # Test 3D
            output, indices = F.max_pool3d(
                torch.randn([4, 4, 4, 4, 4], requires_grad=True),
                2,
                stride=2,
                return_indices=True,
            )
            self.assertEqual(
                F.max_unpool3d(output, indices, 2),
                F.max_unpool3d(output, indices, 2, stride=2),
            )
            gradcheck(F.max_unpool3d, (output, indices, 2), check_forward_ad=True)

    def test_max_unpool3d_input_check(self):
        x = torch.ones(1, 3, 1, 1, 1)
        with self.assertRaises(RuntimeError):
            F.max_unpool3d(x, torch.zeros(x.shape, dtype=int), [1, 1])

    def test_quantized_max_pool1d_empty_kernel(self):
        # This used to segfault when called with an empty kernel
        # see https://github.com/pytorch/pytorch/issues/116323
        base = torch.randn(1)
        temp_tensor = torch.quantize_per_tensor(base, 0.1, 10, torch.quint2x4)
        with self.assertRaises(RuntimeError):
            torch.quantized_max_pool1d(temp_tensor, [])

    def test_quantized_max_pool3d(self):
        # This used to segfault when called with a negative dilation
        # see https://github.com/pytorch/pytorch/issues/136716
        input = torch.randn([1, 1, 1, 1, 1])
        input = torch.quantize_per_tensor(input, -0.1, -10, torch.qint32)
        with self.assertRaisesRegex(RuntimeError, "Expected dilation >= 1"):
            torch.quantized_max_pool3d(
                input, (1, 1, 1), (1, 1, 1), (0, 0, 0), (-3, 1, 1)
            )


class TestPoolingNNDeviceType(NNTestCase):
    @expectedFailureMPS  # No double, float shape prop does not work
    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double)
    def test_adaptive_pooling_zero_batch(self, dtype, device):
        inp = torch.ones(0, 10, dtype=dtype, device=device)
        mod = torch.nn.AdaptiveAvgPool1d(5).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        inp = torch.ones(0, 10, 10, dtype=dtype, device=device)
        mod = torch.nn.AdaptiveAvgPool2d((5, 5)).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        inp = torch.ones(0, 10, 10, 10, dtype=dtype, device=device)
        mod = torch.nn.AdaptiveAvgPool3d((5, 5, 5)).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

    # The tests are used to verify the functions raises errors for backward propagation
    # when output_size = 0, in adaptive_{avg, max}_pool and its variants.
    # These tests are explicitly written because ErrorInputs does not support backward calls
    # Issue: https://github.com/pytorch/pytorch/issues/78868
    @expectedFailureMPS  # No double, float shape prop does not work
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.float32, torch.float64, torch.bfloat16, torch.float16)
    def test_adaptive_pooling_empty_output_size(self, dtype, device):
        error_msg = (
            "Expected grad_output to have non-zero size for non-batch dimensions"
        )

        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=True)
        input = make_arg((1, 64, 10, 9))
        output_size = 0

        fns = (
            nn.functional.adaptive_avg_pool2d,
            nn.functional.adaptive_avg_pool3d,
            nn.functional.adaptive_max_pool2d,
            nn.functional.adaptive_max_pool3d,
        )

        for fn in fns:
            with self.assertRaisesRegex(RuntimeError, error_msg):
                fn(input, output_size).sum().backward()

        fns2 = (
            nn.functional.adaptive_avg_pool1d,
            nn.functional.adaptive_max_pool1d,
        )
        input2 = make_arg((1, 64))

        for fn in fns2:
            with self.assertRaisesRegex(RuntimeError, error_msg):
                fn(input2, output_size).sum().backward()

    @expectedFailureMPS  # Error message does not match
    @onlyNativeDeviceTypes
    def test_adaptive_avg_pooling_backward_fails(self, device):
        grad_output = torch.randn(1, 2, 7, device=device)
        input = torch.randn(1, 2, 3, 3, device=device)
        with self.assertRaisesRegex(RuntimeError, "Expected dimensions"):
            torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, input)

        grad_output = torch.randn(1, 2, 7, 7, device=device)
        input = torch.randn(1, 2, 3, 3, 3, device=device)
        with self.assertRaisesRegex(RuntimeError, "Expected dimensions"):
            torch.ops.aten._adaptive_avg_pool3d_backward(grad_output, input)

    @onlyNativeDeviceTypes
    def test_adaptive_max_pooling_backward_fails(self, device):
        grad_output = torch.randn(1, 2, 7, 7, device=device)
        input = torch.randn(1, 2, 7, 7, device=device)
        indices = torch.ones(1, 2, 3, 3, dtype=torch.long, device=device)
        with self.assertRaisesRegex(RuntimeError, "expected sizes"):
            torch.ops.aten.adaptive_max_pool2d_backward(grad_output, input, indices)

        grad_output = torch.randn(1, 2, 7, 7, 7, device=device)
        input = torch.randn(1, 2, 3, 3, 3, device=device)
        indices = torch.ones(1, 2, 3, 3, dtype=torch.long, device=device)
        with self.assertRaisesRegex(RuntimeError, "expected dimensions"):
            torch.ops.aten.adaptive_max_pool3d_backward(grad_output, input, indices)

    @expectedFailureMPS  # Op not implemented
    @onlyNativeDeviceTypes
    def test_FractionalMaxPool2d_zero_batch(self, device):
        mod = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        inp = torch.ones(0, 16, 50, 32, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Expected input"):
            inp = torch.randn(1, 0, 50, 32, device=device)
            mod(inp)

    @expectedFailureMPS  # Op not implemented
    @onlyNativeDeviceTypes
    def test_FractionalMaxPool3d_zero_batch(self, device):
        mod = nn.FractionalMaxPool3d(3, output_ratio=(0.5, 0.5, 0.5)).to(device)
        inp = torch.ones(0, 16, 50, 32, 32, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Expected input"):
            inp = torch.randn(1, 0, 50, 32, 32, device=device)
            mod(inp)

    @expectedFailureMPS  # Op not implemented
    @onlyNativeDeviceTypes
    def test_FractionalMaxPool2d_zero_out_size(self, device):
        mod = nn.FractionalMaxPool2d([2, 2], output_size=[0, 1])
        inp = torch.rand([16, 50, 32, 32], device=device)
        out = mod(inp)
        self.assertEqual(out, torch.empty((16, 50, 0, 1), device=device))

    @expectedFailureMPS  # Op not implemented
    @onlyNativeDeviceTypes
    def test_FractionalMaxPool3d_zero_out_size(self, device):
        mod = nn.FractionalMaxPool3d([3, 2, 2], output_size=[0, 1, 1])
        inp = torch.rand([16, 50, 32, 32], device=device)
        out = mod(inp)
        self.assertEqual(out, torch.empty((16, 0, 1, 1), device=device))

    @expectedFailureMPS  # Op not implemented
    @onlyNativeDeviceTypes
    def test_FractionalMaxPool2d_zero_samples(self, device):
        samples = torch.rand([0, 16, 2], device=device)
        mod = nn.FractionalMaxPool2d(
            [2, 2], output_size=[1, 1], _random_samples=samples
        )
        inp = torch.randn([0, 16, 32, 32], device=device)
        out = mod(inp)
        self.assertEqual(out, torch.empty((0, 16, 1, 1), device=device))

        inp1 = torch.randn([1, 16, 32, 32], device=device)
        with self.assertRaisesRegex(RuntimeError, "Expect _random_samples"):
            mod(inp1)

    @expectedFailureMPS  # Op not implemented
    @onlyNativeDeviceTypes
    def test_FractionalMaxPool3d_zero_samples(self, device):
        samples = torch.rand([0, 16, 3], device=device)
        mod = nn.FractionalMaxPool3d(
            [3, 2, 2], output_size=[1, 1, 1], _random_samples=samples
        )
        inp = torch.randn([0, 16, 50, 32, 32], device=device)
        out = mod(inp)
        self.assertEqual(out, torch.empty((0, 16, 1, 1, 1), device=device))

        inp1 = torch.randn([1, 16, 50, 32, 32], device=device)
        with self.assertRaisesRegex(RuntimeError, "Expect _random_samples"):
            mod(inp1)

    @onlyNativeDeviceTypes
    def test_FractionalMaxPool3d_errors(self, device):
        samples = torch.rand([0, 16, 3], device=device)
        with self.assertRaisesRegex(ValueError, "kernel_size must greater than 0"):
            nn.FractionalMaxPool3d(0, output_size=[1, 1, 1], _random_samples=samples)
        with self.assertRaisesRegex(ValueError, "kernel_size must greater than 0"):
            nn.FractionalMaxPool3d(
                [0, 0, 0], output_size=[1, 1, 1], _random_samples=samples
            )
        samples = torch.randn(1, 3, 10, 10, 10)
        with self.assertRaisesRegex(RuntimeError, "too large relative to"):
            nn.FractionalMaxPool3d(
                kernel_size=9223372036854775803,
                output_size=[1, 1, 1],
            )(samples)
        with self.assertRaisesRegex(ValueError, "kernel_size must greater than 0"):
            nn.FractionalMaxPool3d(
                kernel_size=-1,
                output_size=[1, 1, 1],
            )(samples)

    @onlyNativeDeviceTypes
    def test_MaxPool3d_errors(self, device):
        samples = torch.randn(1, 3, 10, 10, 10)
        with self.assertRaisesRegex(RuntimeError, "integer out of range"):
            nn.MaxPool3d(
                kernel_size=9223372036854775803,
            )(samples)
        with self.assertRaisesRegex(
            RuntimeError, "kernel size should be greater than zero"
        ):
            nn.MaxPool3d(
                kernel_size=-1,
            )(samples)

    @onlyNativeDeviceTypes
    def test_MaxPool_zero_batch_dim(self, device):
        inp = torch.randn(0, 16, 50, device=device)
        mod = torch.nn.MaxPool1d(3, stride=2).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 1D is supposed to be okay with 0 numel() inputs so dont test
        # error raising for that case.

        inp = torch.randn(0, 16, 50, 32, device=device)
        mod = torch.nn.MaxPool2d(3, stride=2).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Expected"):
            inp = torch.randn(1, 0, 50, 32, device=device)
            mod(inp)

        inp = torch.ones(0, 16, 50, 44, 31, device=device)
        mod = torch.nn.MaxPool3d(3, stride=2).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Expected"):
            inp = torch.ones(1, 0, 50, 44, 31, device=device)
            mod(inp)

    @onlyNativeDeviceTypes
    def test_MaxUnpool_zero_batch_dim(self, device):
        pool = torch.nn.MaxPool1d(2, stride=2, return_indices=True).to(device)
        unpool = torch.nn.MaxUnpool1d(2, stride=2).to(device)
        inp = torch.randn(0, 10, 10, requires_grad=True, device=device)
        output, indices = pool(inp)
        output.requires_grad_(True)
        unpool_out = unpool(output, indices)
        unpool_out.sum().backward()

        self.assertEqual(inp.grad, torch.zeros_like(inp))
        self.assertEqual(unpool_out, torch.zeros_like(unpool_out))

        pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True).to(device)
        unpool = torch.nn.MaxUnpool2d(2, stride=2).to(device)
        inp = torch.randn(0, 10, 10, 10, requires_grad=True, device=device)
        output, indices = pool(inp)
        unpool_out = unpool(output, indices)
        unpool_out.sum().backward()

        self.assertEqual(inp.grad, torch.zeros_like(inp))
        self.assertEqual(unpool_out, torch.zeros_like(unpool_out))

        pool = torch.nn.MaxPool3d(2, stride=2, return_indices=True).to(device)
        unpool = torch.nn.MaxUnpool3d(2, stride=2).to(device)
        inp = torch.randn(0, 10, 10, 10, 10, requires_grad=True, device=device)
        output, indices = pool(inp)
        output.requires_grad_(True)
        unpool_out = unpool(output, indices)
        unpool_out.sum().backward()

        self.assertEqual(inp.grad, torch.zeros_like(inp))
        self.assertEqual(unpool_out, torch.zeros_like(unpool_out))

    @slowTest
    @onlyNativeDeviceTypes
    @skipCUDAIfRocm
    @parametrize_test(
        "module_name,module_size,output_size,test_index,should_error",
        [
            # Some tests are failing in trunk https://github.com/pytorch/pytorch/issues/103854
            subtest(
                ("MaxUnpool2d", (2, 2), (1, 3, 4, 5), -1, True),
                name="case1",
            ),
            subtest(
                ("MaxUnpool2d", (2, 2), (1, 3, 4, 5), 2 * 2 * 4 * 5, True),
                name="case2",
            ),
            subtest(
                ("MaxUnpool2d", (2, 2), (1, 3, 4, 5), (2 * 2 * 4 * 5) - 1, False),
                name="case3",
            ),
            subtest(
                ("MaxUnpool2d", (2, 3), (2, 1, 4, 2), 2 * 3 * 4 * 2, True),
                name="case4",
            ),
            subtest(
                ("MaxUnpool2d", (2, 3), (2, 1, 4, 2), (2 * 3 * 4 * 2) - 1, False),
                name="case5",
            ),
            subtest(
                ("MaxUnpool3d", (2, 2, 2), (1, 3, 4, 5), -1, True),
                name="case6",
            ),
            subtest(
                ("MaxUnpool3d", (2, 2, 2), (1, 3, 4, 5), 2 * 2 * 2 * 3 * 4 * 5, True),
                name="case7",
            ),
            subtest(
                (
                    "MaxUnpool3d",
                    (2, 2, 2),
                    (1, 3, 4, 5),
                    (2 * 2 * 2 * 3 * 4 * 5) - 1,
                    False,
                ),
                name="case8",
            ),
            subtest(
                ("MaxUnpool3d", (2, 2, 2), (2, 3, 4, 1), 2 * 2 * 2 * 3 * 4 * 1, True),
                name="case9",
            ),
            subtest(
                (
                    "MaxUnpool3d",
                    (2, 2, 2),
                    (2, 3, 4, 1),
                    (2 * 2 * 2 * 3 * 4 * 1) - 1,
                    False,
                ),
                name="case10",
            ),
        ],
    )
    def test_MaxUnpool_index_errors(
        self, device, module_name, module_size, output_size, test_index, should_error
    ):
        # NOTE: CUDA tests need to be run in a subprocess because they cause device asserts
        if torch.device(device).type == "cuda":
            error_msgs = {
                "MaxUnpool2d": r"Assertion `maxind >= 0 && maxind < outputImageSize` failed",
                "MaxUnpool3d": r"Assertion `index >= 0 && index < outputImageSize` failed",
            }

            script = f"""
import torch
unpool = torch.nn.{module_name}({module_size}).to('{device}')
output = torch.rand({output_size}, dtype=torch.float32, device='{device}')
indices = torch.zeros({output_size}, dtype=torch.int64, device='{device}')
indices.flatten()[0] = {test_index}
unpool(output, indices)
torch.cuda.synchronize()
"""
            p = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.path.dirname(os.path.realpath(__file__)),
                capture_output=True,
                text=True,
            )

            output = p.stdout + "\n" + p.stderr

            error_msg = error_msgs[module_name]

            if should_error:
                self.assertIn(error_msg, output, "The expected error was not found")
            else:
                self.assertNotIn("Error", output, "Should not have produced an error")
        else:
            module_class = getattr(torch.nn, module_name)
            unpool = module_class(module_size).to(device)
            output = torch.rand(output_size, dtype=torch.float32, device=device)
            indices = torch.zeros(output_size, dtype=torch.int64, device=device)
            indices.flatten()[0] = test_index

            if should_error:
                with self.assertRaisesRegex(
                    RuntimeError, r"Found an invalid max index:"
                ):
                    unpool(output, indices)
            else:
                unpool(output, indices)

    # https://github.com/pytorch/pytorch/issues/163409
    @onlyNativeDeviceTypes
    def test_MaxUnpool_invalid_output_size(self, device):
        input2d = torch.randn(1, 1, 1)
        input3d = torch.randn(1, 1, 1, 1, 1)
        unpool2d = torch.nn.MaxUnpool2d(())
        unpool3d = torch.nn.MaxUnpool3d(())

        with self.assertRaisesRegex(RuntimeError, "There should be exactly"):
            unpool2d(input2d, torch.zeros_like(input2d, dtype=torch.int64))

        with self.assertRaisesRegex(RuntimeError, "There should be exactly"):
            unpool3d(input3d, torch.zeros_like(input3d, dtype=torch.int64))

    @expectedFailureMPS
    @onlyNativeDeviceTypes
    def test_AdaptiveMaxPool_zero_batch_dim(self, device):
        inp = torch.randn(0, 16, 50, device=device)
        mod = torch.nn.AdaptiveMaxPool1d(3).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Expected"):
            inp = torch.randn(1, 0, 50, device=device)
            mod(inp)

        inp = torch.randn(0, 16, 50, 32, device=device)
        mod = torch.nn.AdaptiveMaxPool2d(3).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Expected"):
            inp = torch.randn(1, 0, 50, 32, device=device)
            mod(inp)

        inp = torch.ones(0, 16, 50, 44, 31, device=device)
        mod = torch.nn.AdaptiveMaxPool3d(3).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Expected"):
            inp = torch.ones(1, 0, 50, 44, 31, device=device)
            mod(inp)

    @onlyCPU
    def test_LPPool1d_kernel_size_overflow_large(self, device):
        avgpool = torch.nn.LPPool1d(
            -1.38119e150, 7879455037536781369, ceil_mode=True
        ).to(device)
        inp = torch.randn(3, 15, device=device)

        with self.assertRaisesRegex(RuntimeError, "integer out of range"):
            avgpool(inp)

    @onlyNativeDeviceTypes
    def test_AvgPool2d_empty(self, device):
        avgpool = torch.nn.AvgPool2d(3, stride=2).to(device)
        inp = torch.randn(0, 16, 20, 32, device=device)
        _test_module_empty_input(self, avgpool, inp, check_size=False)

        clast_inp = torch.randn(0, 16, 20, 32, device=device).contiguous(
            memory_format=torch.channels_last
        )
        _test_module_empty_input(self, avgpool, clast_inp, check_size=False)

        # test with empty non-batch input
        with self.assertRaisesRegex(RuntimeError, "3D or 4D"):
            inp = torch.randn(16, 0, 20, 32, device=device)
            avgpool(inp)

    @parametrize_test("kernel", ["max", "avg"])
    @parametrize_test("pooling_dims", [1, 2, 3])
    def test_pooling_shape(self, device, kernel, pooling_dims):
        """Test the output shape calculation for pooling functions"""

        if kernel == "max" and pooling_dims == 1:
            # This case causes the process to abort, so need to skip it for now
            self.skipTest("Skipping to avoid abort")

        # Checks output shape against expected for 1D, 2D and 3D
        def check(expected_out_shape, sizes, *args, **kwargs):
            if hasattr(torch.nn.functional, f"{kernel}_pool{pooling_dims}d"):
                op = getattr(torch.nn.functional, f"{kernel}_pool{pooling_dims}d")
                t = torch.randn(sizes[: pooling_dims + 2], device=device)
                self.assertEqual(
                    op(t, *args, **kwargs).shape, expected_out_shape[: pooling_dims + 2]
                )

        check(
            (1, 1, 3, 3, 4),
            (1, 1, 5, 6, 7),
            kernel_size=1,
            stride=2,
            padding=0,
            ceil_mode=True,
        )
        check(
            (1, 1, 2, 3, 3),
            (1, 1, 3, 4, 5),
            kernel_size=2,
            stride=2,
            padding=1,
            ceil_mode=False,
        )
        check(
            (1, 1, 2, 3, 3),
            (1, 1, 3, 4, 5),
            kernel_size=2,
            stride=2,
            padding=1,
            ceil_mode=True,
        )

        # Test case from issue https://github.com/pytorch/pytorch/issues/45357
        x = torch.randn(1, 1, 6, 7, device=device)
        y = torch.nn.functional.max_pool2d(
            x, 1, stride=(2, 2), padding=0, ceil_mode=True
        )
        self.assertEqual(y.size(), (1, 1, 3, 4))

    @onlyNativeDeviceTypes  # TODO: fix on XLA
    def test_adaptive_avg_pool2d_output_size_one(self, device):
        def helper(size, memory_format):
            x = torch.randint(
                1, 10, size, dtype=torch.float, device=device, requires_grad=True
            )
            if memory_format == "non_contiguous":
                x = x[::2, ::2, ::2, ::2]
            else:
                x = x.to(memory_format=memory_format)

            net = torch.nn.AdaptiveAvgPool2d((1, 1))
            out = net(x)
            ref_out = x.contiguous().mean((-1, -2)).view((x.size(0), x.size(1), 1, 1))

            out.sum().backward()  # make sure it doesn't crash

            self.assertEqual(out, ref_out)
            if memory_format == torch.channels_last:
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, c, c])
            else:
                self.assertTrue(out.is_contiguous())
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, 1, 1])

        for mf in (torch.contiguous_format, torch.channels_last, "non_contiguous"):
            helper((2, 3, 6, 6), mf)

    @onlyNativeDeviceTypes
    def test_adaptive_avg_pool3d_output_size_one(self, device):
        x = torch.randn(
            (2, 3, 6, 6, 6), dtype=torch.float, device=device, requires_grad=True
        )

        net = torch.nn.AdaptiveAvgPool3d(1)
        out = net(x)
        ref_out = x.contiguous().mean((-1, -2, -3)).view(out.shape)

        out.sum().backward()  # make sure it doesn't crash

        self.assertEqual(out, ref_out)
        self.assertTrue(out.is_contiguous())
        c = out.size(1)
        self.assertEqual(out.stride(), [c, 1, 1, 1, 1])

    @expectedFailureMPS  # Runtime Error not raised for mps
    @expectedFailureMeta  # Runtime Error not raised for meta
    @onlyNativeDeviceTypes
    @dtypes(torch.uint8, torch.int8, torch.short, torch.int, torch.long)
    def test_adaptive_pooling_no_suppot_input(self, device, dtype):
        for numel in (2, 3):
            for pool_type in ("Max", "Avg"):
                cls_name = f"Adaptive{pool_type}Pool{numel}d"
                module_cls = getattr(nn, cls_name)
                output_size = (2,) * numel
                module = module_cls(output_size)
                input = torch.randn((4,) * (numel + 1), device=device).to(dtype)
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    module(input)

    @expectedFailureMPS  # TODO: fixme
    @onlyNativeDeviceTypes
    @gcIfJetson
    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    def test_avg_pool2d_nhwc(self, device, dtype):
        def helper(
            n,
            c,
            h,
            w,
            kernel_size,
            stride=None,
            count_include_pad=True,
            divisor_override=None,
            padding=0,
        ):
            if stride is None:
                stride = kernel_size
            input = torch.randn(n, c, h, w, dtype=dtype, device=device)
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            grad = torch.randn(
                n,
                c,
                (h - kernel_size) // stride + 1,
                (w - kernel_size) // stride + 1,
                dtype=dtype,
                device=device,
            )
            pool = torch.nn.AvgPool2d(
                kernel_size,
                stride=stride,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            ).to(device)

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.AvgPool2d(
                kernel_size,
                stride=stride,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            ).to(device)

            out = pool(input)
            out.backward(grad)
            ref_out = ref_pool(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(input.grad, ref_input.grad)

        helper(4, 8, 8, 8, 3)
        helper(4, 8, 8, 8, 3, count_include_pad=False, padding=1)
        helper(4, 8, 8, 8, 3, count_include_pad=False, padding=2, stride=2)
        helper(4, 8, 8, 8, 3, divisor_override=42)
        helper(4, 8, 8, 8, 7)
        # ROCm 16GB MI25 hits OOM error. Clear caching allocator prior to running large subtest.
        if TEST_WITH_ROCM and "cuda" in device:
            torch.cuda.empty_cache()
        helper(200, 512, 28, 28, 2)
        helper(4, 8, 7, 7, 3, stride=1)
        helper(4, 8, 7, 7, 3, padding=2, stride=1)
        helper(10, 512, 31, 31, 3, stride=2)
        helper(1, 129, 8, 8, 3, stride=2)

    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_max_pool1d_corner_cases(self, device, dtype):
        def check(x, args, expected):
            model = torch.nn.MaxPool1d(*args)
            if isinstance(x, list):
                x = torch.tensor(x, device=device, dtype=dtype)
                expected = torch.tensor(expected, device=device, dtype=dtype)
            self.assertEqual(model(x), expected)

        # Pooling args: (kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        check([[1]], (1, None, 0, 1, False, False), [[1]])
        check([[1]], (2, None, 1, 2, False, False), [[float("-inf")]])
        check(
            [[1], [1]],
            (2, None, 1, 2, False, False),
            [[float("-inf")], [float("-inf")]],
        )
        check([[1, 2]], (2, 1, 1, 2, False, False), [[2, 1]])
        check([[1, 2]], (2, 2, 1, 2, False, True), [[2, 2]])

    @onlyCPU
    @dtypes(torch.float, torch.double)
    @skipIfTorchDynamo("OOMs https://github.com/pytorch/pytorch/issues/111320")
    def test_max_pool1d(self, device, dtype):
        # FIXME For now compare against max_pool1d with indices
        def check(x, *args, **kwargs):
            model = torch.nn.MaxPool1d(*args, **kwargs)
            ref_model = torch.nn.MaxPool1d(*args, **kwargs, return_indices=True)
            self.assertEqual(model(x), ref_model(x)[0])

        sizes = [random.sample(range(8, 128), 3) for _ in range(3)]
        kernel_sizes = random.sample(range(1, 5), 3)
        strides = random.sample(range(1, 5), 3)
        dilations = random.sample(range(1, 5), 3)
        ceil_modes = [True, False]

        for size, kernel_size, stride, dilation, ceil_mode in itertools.product(
            sizes, kernel_sizes, strides, dilations, ceil_modes
        ):
            padding = random.sample(range(math.floor(kernel_size / 2) + 1), 1)
            check(
                torch.randn(size, device=device, dtype=dtype),
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode=ceil_mode,
            )

        # Non-contiguous test
        tensor = torch.randn(5, 151, 33, device=device, dtype=dtype)[::2, ::3, ::2]
        check(tensor, 3, 2, 1, 2, ceil_mode=True)
        check(tensor.transpose(1, 2), 3, 2, 1, 2, ceil_mode=True)

    @onlyCUDA
    @gcIfJetson
    def test_max_pool2d(self, device):
        def helper(n, c, h, w, ks):
            x = torch.randn(
                n, c, h, w, device="cuda", dtype=torch.float, requires_grad=True
            )
            ref_x = x.detach().clone().cpu().requires_grad_()

            pool = torch.nn.MaxPool2d(kernel_size=ks)

            y = pool(x)
            ref_y = pool(ref_x)

            y.sum().backward()
            ref_y.sum().backward()

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, ref_x.grad)

        helper(2, 8, 4, 4, ks=2)
        helper(1, 100000, 32, 32, ks=4)
        helper(1, 100000, 1, 4, ks=(1, 4))  # test for max_pool1d

    @expectedFailureMPS  # TODO: Fixme
    @onlyNativeDeviceTypes
    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @gcIfJetson
    def test_max_pool2d_nhwc(self, device, dtype):
        def helper(n, c, h, w, kernel_size, stride=None):
            if stride is None:
                stride = kernel_size
            input = torch.randn(n, c, h, w, dtype=dtype, device=device)
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            grad = torch.randn(
                n,
                c,
                (h - kernel_size) // stride + 1,
                (w - kernel_size) // stride + 1,
                dtype=dtype,
                device=device,
            )
            pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(
                device
            )

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(
                device
            )

            out, ind = pool(input)
            out.backward(grad)
            ref_out, ref_ind = ref_pool(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ind.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_ind.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(ind, ref_ind)
            self.assertEqual(input.grad, ref_input.grad)

        helper(4, 8, 8, 8, 7)
        helper(200, 512, 28, 28, 2)
        helper(4, 8, 7, 7, 3, stride=1)
        helper(10, 512, 31, 31, 3, stride=2)
        helper(1, 129, 8, 8, 3, stride=2)

    @onlyCPU
    @dtypes(torch.int32, torch.int64)
    def test_max_pool2d_corner_cases(self, device, dtype):
        def check(x, args, expected, memory_format):
            model = torch.nn.MaxPool2d(*args)
            if isinstance(x, list):
                x = torch.tensor(x, device=device, dtype=dtype).to(
                    memory_format=memory_format
                )
                expected = torch.tensor(expected, device=device, dtype=dtype).to(
                    memory_format=memory_format
                )
            self.as
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/nn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/nn`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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
python docs/test/nn/test_pooling.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/nn`):

- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_load_state_dict.py_kw.md_docs.md`](./test_load_state_dict.py_kw.md_docs.md)
- [`test_embedding.py_kw.md_docs.md`](./test_embedding.py_kw.md_docs.md)
- [`test_module_hooks.py_kw.md_docs.md`](./test_module_hooks.py_kw.md_docs.md)
- [`test_dropout.py_docs.md_docs.md`](./test_dropout.py_docs.md_docs.md)
- [`test_dropout.py_kw.md_docs.md`](./test_dropout.py_kw.md_docs.md)
- [`test_packed_sequence.py_docs.md_docs.md`](./test_packed_sequence.py_docs.md_docs.md)
- [`test_multihead_attention.py_docs.md_docs.md`](./test_multihead_attention.py_docs.md_docs.md)
- [`test_pruning.py_kw.md_docs.md`](./test_pruning.py_kw.md_docs.md)
- [`test_init.py_docs.md_docs.md`](./test_init.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_pooling.py_docs.md_docs.md`
- **Keyword Index**: `test_pooling.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
