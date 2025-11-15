# Documentation: `test/test_mkldnn.py`

## File Metadata

- **Path**: `test/test_mkldnn.py`
- **Size**: 77,488 bytes (75.67 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: mkldnn"]

import copy
import itertools
import functools
import unittest
import warnings
from contextlib import nullcontext

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

import torch
import torch.nn.functional as F
import torch.jit
import torch.backends.mkldnn
from torch.utils import mkldnn as mkldnn_utils
from torch.testing._internal.common_utils import TestCase, \
    run_tests, TemporaryFileName, gradcheck, gradgradcheck, IS_WINDOWS, \
    skipIfTorchDynamo, xfailIfTorchDynamo, recover_orig_fp32_precision
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
)
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off

# batched grad doesn't support mkldnn
gradcheck = functools.partial(gradcheck, check_batched_grad=False)
gradgradcheck = functools.partial(gradgradcheck, check_batched_grad=False)


types = [torch.float, torch.bfloat16, torch.half]

# Comment the line below to find out the CI machines having MKL-DNN build disabled
@unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled")
class TestMkldnn(TestCase):
    def test_conversion(self):
        for cpu_tensor in [torch.randn((1, 2, 3, 4),
                                       dtype=torch.float, device=torch.device('cpu')),
                           torch.randn((1, 2, 3, 4, 5),
                                       dtype=torch.float, device=torch.device('cpu'))[:, :, :, :, 1]]:
            cpu_tensor.requires_grad_()
            convert_dtypes = {torch.half: [torch.half, torch.float],
                              torch.bfloat16: [torch.bfloat16, torch.float],
                              torch.float: [torch.bfloat16, torch.half]}
            # float/bfloat16/half cpu tensor to mkldnn tensortensor.
            for dtype1 in types:
                mkldnn_tensor = cpu_tensor.to_mkldnn(dtype1)
                self.assertEqual(mkldnn_tensor.dtype, dtype1)
                cpu_tensor_1 = mkldnn_tensor.to_dense()
                # not given dtype for to_dense, mkldnn tensor has same dtype with cpu tensor
                self.assertEqual(mkldnn_tensor.dtype, cpu_tensor_1.dtype)
                # mkldnn float/bfloat tensor to cpu float or bfloat tensor
                for dtype2 in convert_dtypes[dtype1]:
                    cpu_tensor_2 = mkldnn_tensor.to_dense(dtype2)
                    self.assertEqual(cpu_tensor_2.dtype, dtype2)
                    atol = 1e-5 if dtype1 == torch.float and dtype2 == torch.float else 1e-2
                    self.assertEqual(cpu_tensor, cpu_tensor_2.float(), atol=atol, rtol=0)

                self.assertEqual(mkldnn_tensor.device, torch.device('cpu'))
                self.assertEqual(mkldnn_tensor.size(), torch.Size([1, 2, 3, 4]))
                self.assertEqual(mkldnn_tensor.numel(), cpu_tensor.numel())
                if dtype1 == torch.float:
                    self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor.element_size())
                else:
                    self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor.element_size() / 2)
                self.assertRaisesRegex(RuntimeError,
                                       "Cannot access data pointer of Tensor that doesn't have storage",
                                       lambda: mkldnn_tensor.data_ptr() != 0)

            # bfloat cpu tensor to mkldnn float tensor or bfloat tensor.
            for orig_dtype in [torch.half, torch.bfloat16]:
                cpu_tensor_lower = cpu_tensor.to(dtype=orig_dtype)
                for dtype1 in convert_dtypes[orig_dtype]:
                    mkldnn_tensor = cpu_tensor_lower.to_mkldnn(dtype1)
                    self.assertEqual(mkldnn_tensor.dtype, dtype1)
                    cpu_tensor_1 = mkldnn_tensor.to_dense()
                    # not given dtype for to_dense, mkldnn tensor has same dtype with cpu tensor
                    self.assertEqual(mkldnn_tensor.dtype, cpu_tensor_1.dtype)
                    # mkldnn float/bfloat/half tensor to cpu float/bfloat/half tensor
                    for dtype2 in convert_dtypes[cpu_tensor_lower.dtype]:
                        cpu_tensor_2 = mkldnn_tensor.to_dense(dtype2)
                        self.assertEqual(cpu_tensor_2.dtype, dtype2)
                        self.assertEqual(cpu_tensor_lower,
                                         cpu_tensor_2.to(dtype=cpu_tensor_lower.dtype), atol=1e-5, rtol=0)

                    self.assertEqual(mkldnn_tensor.device, torch.device('cpu'))
                    self.assertEqual(mkldnn_tensor.size(), torch.Size([1, 2, 3, 4]))
                    self.assertEqual(mkldnn_tensor.numel(), cpu_tensor.numel())
                    if dtype1 in [torch.bfloat16, torch.half]:
                        self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor_lower.element_size())
                    else:
                        self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor_lower.element_size() * 2)
                    self.assertRaisesRegex(RuntimeError,
                                           "Cannot access data pointer of Tensor that doesn't have storage",
                                           lambda: mkldnn_tensor.data_ptr() != 0)

    def test_conversion_byte_char(self):
        int8_types = [torch.int8, torch.uint8]
        for int8_type in int8_types:
            low = -100 if int8_type is torch.int8 else 0
            high = 100
            for cpu_tensor in [torch.randint(
                               low=low,
                               high=high,
                               size=(1, 2, 3, 4),
                               dtype=torch.int64,
                               device=torch.device('cpu')),
                               torch.randint(
                               low=low,
                               high=high,
                               size=(1, 2, 3, 4, 5),
                               dtype=torch.int64,
                               device=torch.device('cpu'))[:, :, :, :, :]]:

                cpu_tensor = cpu_tensor.to(dtype=int8_type)
                mkldnn_tensor = cpu_tensor.to_mkldnn(int8_type)
                self.assertEqual(mkldnn_tensor.dtype, int8_type)
                cpu_tensor_1 = mkldnn_tensor.to_dense()
                self.assertEqual(mkldnn_tensor.dtype, cpu_tensor_1.dtype)
                self.assertEqual(cpu_tensor, cpu_tensor_1)
                self.assertEqual(mkldnn_tensor.device, torch.device('cpu'))
                self.assertEqual(mkldnn_tensor.size(), cpu_tensor.size())
                self.assertEqual(mkldnn_tensor.numel(), cpu_tensor.numel())
                self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor.element_size())
                self.assertRaisesRegex(RuntimeError,
                                       "Cannot access data pointer of Tensor that doesn't have storage",
                                       lambda: mkldnn_tensor.data_ptr() != 0)

    def test_copy(self):
        x = torch.randn(4, 5, dtype=torch.float32)
        mkldnn_x = x.to_mkldnn()
        mkldnn_y = torch.randn(4, 5, dtype=torch.float32).to_mkldnn()
        mkldnn_z = torch.randn(4, 10, dtype=torch.float32).to_mkldnn()
        mkldnn_y.copy_(mkldnn_x)
        self.assertEqual(x, mkldnn_y.to_dense())
        self.assertRaisesRegex(RuntimeError,
                               "copy_mkldnn_: only support same size tensor.",
                               lambda: mkldnn_z.copy_(mkldnn_x))
        self.assertRaisesRegex(RuntimeError,
                               "copy_mkldnn_: between mkldnn layout and dense Tensors is not implemented! "
                               "Found self type = torch.FloatTensor and src type = Mkldnntorch.FloatTensor",
                               lambda: x.copy_(mkldnn_x))
        self.assertRaisesRegex(RuntimeError,
                               "copy_mkldnn_: between mkldnn layout and dense Tensors is not implemented! "
                               "Found self type = Mkldnntorch.FloatTensor and src type = torch.FloatTensor",
                               lambda: mkldnn_x.copy_(x))

    def test_unsupported(self):
        # unsupported types and unsupported types with gpu
        for dtype in [torch.double, torch.uint8, torch.int8,
                      torch.short, torch.int, torch.long]:
            with self.assertRaises(RuntimeError):
                torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cpu')).to_mkldnn()
            if torch.cuda.is_available():
                with self.assertRaises(RuntimeError):
                    torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cuda')).to_mkldnn()
        # supported type with gpu
        if torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                torch.randn(1, 2, 3, 4, dtype=torch.float, device=torch.device('cuda')).to_mkldnn()
        # some factory functions
        for creator in [torch.ones, torch.randn, torch.rand]:
            with self.assertRaises(RuntimeError):
                creator(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu'), layout=torch._mkldnn)

    def test_mkldnn_conv_shapecheck(self):
        input = torch.full((1, 1, 1, 24,), 1, dtype=torch.float32)
        w1 = torch.full((1, 1, 1, 24,), 1, dtype=torch.float32)
        b1 = torch.full((1,), 1, dtype=torch.float32)
        w2 = torch.full((1, 1, 2, 24,), 1, dtype=torch.float32)
        b2 = torch.full((2,), 1, dtype=torch.float32)
        options = zip([-1, 0, 0, 0, 0, 0, 0],  # padding
                      [1, 0, 1, 1, 1, 1, 1],  # stride
                      [1, 1, 0, 1, 1, 1, 1],  # dilation
                      [1, 1, 1, 0, 2, 1, 1],  # groups
                      [w1, w1, w1, w1, w1, w1, w2],  # weight
                      [b1, b1, b1, b1, b1, b2, b1])  # bias
        for pad, st, dil, gr, w, b in options:
            with self.assertRaises(RuntimeError) as _:
                torch.mkldnn_convolution(input, w, b, [pad] * 2, [st] * 2, [dil] * 2, gr)

    def test_autograd_to_mkldnn(self):
        # MKLDNN only supports float32
        root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

        def func(root):
            return root.to_mkldnn().to_dense()

        # because MKLDNN only supports float32, we need to lessen the precision.
        # these numbers are just empirical results that seem to work.
        self.assertWarnsRegex(UserWarning,
                              'double precision floating point',
                              lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2))
        self.assertWarnsRegex(UserWarning,
                              'double precision floating point',
                              lambda: gradgradcheck(func, [root], atol=4e-2, rtol=1e-2))

    def test_autograd_from_mkldnn(self):
        # MKLDNN only supports float32
        root = torch.randn(4, 5, dtype=torch.float32).to_mkldnn().requires_grad_()

        def func(root):
            return root.to_dense()

        # because MKLDNN only supports float32, we need to lessen the precision.
        # these numbers are just empirical results that seem to work.
        self.assertWarnsRegex(UserWarning,
                              'double precision floating point',
                              lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2))

    def test_detach(self):
        root = torch.randn(4, 5, dtype=torch.float32).to_mkldnn().requires_grad_()

        detach = root.detach()
        self.assertEqual((4, 5), detach.size())
        self.assertFalse(detach.requires_grad)
        self.assertTrue(root.requires_grad)

        detach_ = root.detach_()
        self.assertEqual((4, 5), detach_.size())
        self.assertFalse(detach_.requires_grad)
        self.assertFalse(root.requires_grad)

    def test_repr(self):
        self.assertTrue("layout=torch._mkldnn" in str(torch.randn((1, 2, 3, 4),
                                                                  dtype=torch.float, device=torch.device('cpu')).to_mkldnn()))

    def _test_conv_base(self, dim):
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        options = itertools.product([True, False], [True, False], [1, 2], [1, 4])
        for train, bias, dilation, groups in options:
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32)
            conv = conv_module[dim](in_channels=C,
                                    out_channels=M,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=dilation,
                                    bias=bias,
                                    groups=groups).float()
            x1 = x.clone()
            x2 = x.clone().to_mkldnn()
            if not train:
                mkldnn_conv = mkldnn_utils.to_mkldnn(copy.deepcopy(conv))
            elif train and dim != 1:
                # TODO: enable conv1d training.
                x1.requires_grad_()
                x2.requires_grad_()
                mkldnn_conv = copy.deepcopy(conv)
            with torch.backends.mkldnn.flags(enabled=False):
                y_aten = conv(x1)
                if train and dim != 1:
                    loss1 = y_aten.sum()
                    loss1.backward()
            if not train or (train and dim != 1):
                y_mkldnn = mkldnn_conv(x2).to_dense()
                if self.precision != 0:
                    self.assertEqual(y_aten, y_mkldnn, atol=self.precision, rtol=self.precision)
                else:
                    self.assertEqual(y_aten, y_mkldnn)
            if not train:
                self._test_serialization(mkldnn_conv, (x.to_mkldnn(),))
                self._test_tracing(mkldnn_conv, (x.to_mkldnn(),))
            elif dim != 1:
                loss2 = y_mkldnn.sum()
                loss2.backward()
                self.assertTrue(x2.grad.is_mkldnn)
                self.assertEqual(x1.grad, x2.grad.to_dense())
                self.assertEqual(conv.weight.grad,
                                 mkldnn_conv.weight.grad,
                                 atol=1e-3,
                                 rtol=1e-3)
                if bias:
                    self.assertEqual(conv.bias.grad, mkldnn_conv.bias.grad)

    @reduced_f32_on_and_off()
    def test_conv1d(self):
        self._test_conv_base(dim=1)

    @reduced_f32_on_and_off()
    def test_conv2d(self):
        self._test_conv_base(dim=2)

    @reduced_f32_on_and_off()
    def test_conv3d(self):
        self._test_conv_base(dim=3)

    def _test_conv_deconv_lower_precision_base(self, dim, conv_module, dtype):
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        options = itertools.product([True, False], [1, 2], [1, 4])
        for bias, dilation, groups in options:
            N = torch.randint(1, 3, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32)
            # TODO: remove this when group depthwise is supported:
            if conv_module in [torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d] and groups > 1 and C == groups:
                continue
            conv = conv_module(in_channels=C,
                               out_channels=M,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               dilation=dilation,
                               bias=bias,
                               groups=groups).float()
            x_lower = x.to(dtype=dtype)
            if (dtype == torch.bfloat16 and torch.ops.mkldnn._is_mkldnn_bf16_supported()) or \
               (dtype == torch.half and torch.ops.mkldnn._is_mkldnn_fp16_supported()):
                mkldnn_conv = mkldnn_utils.to_mkldnn(copy.deepcopy(conv))
                mkldnn_conv_lower = mkldnn_utils.to_mkldnn(copy.deepcopy(conv), dtype)
                y = mkldnn_conv(x.to_mkldnn()).to_dense()
                y_lower = mkldnn_conv_lower(x_lower.to_mkldnn()).to_dense(torch.float32)
                self.assertEqual(y, y_lower, atol=1e-1, rtol=1e-3)
            else:
                msg = {
                    torch.bfloat16: r"bf16 path needs the cpu support avx_ne_convert or avx512bw, avx512vl and avx512dq",
                    torch.half: r"fp16 path needs the cpu support avx_ne_convert or avx512_fp16",
                }
                with self.assertRaisesRegex(RuntimeError, msg[dtype]):
                    mkldnn_conv_lower = mkldnn_utils.to_mkldnn(copy.deepcopy(conv), dtype)
                    y_lower = mkldnn_conv_lower(x_lower.to_mkldnn()).to_dense(torch.float32)
            # test thnn impl
            conv_lower = copy.deepcopy(conv).to(dtype=dtype)
            conv_ref = copy.deepcopy(conv_lower).float()
            with torch.backends.mkldnn.flags(enabled=False):
                x_ref = x_lower.clone().float().detach().requires_grad_()
                x_lower.requires_grad_()
                y = conv_ref(x_ref)
                y_lower = conv_lower(x_lower).float()
                self.assertEqual(y, y_lower, atol=5e-2, rtol=5e-3)

    @dtypes(torch.float16, torch.bfloat16)
    def test_conv_deconv_1d_lower_precision(self, dtype):
        self._test_conv_deconv_lower_precision_base(1, torch.nn.Conv1d, dtype=dtype)
        self._test_conv_deconv_lower_precision_base(1, torch.nn.ConvTranspose1d, dtype=dtype)

    @dtypes(torch.float16, torch.bfloat16)
    def test_conv_deconv_2d_lower_precision(self, dtype):
        self._test_conv_deconv_lower_precision_base(2, torch.nn.Conv2d, dtype=dtype)
        self._test_conv_deconv_lower_precision_base(2, torch.nn.ConvTranspose2d, dtype=dtype)

    @dtypes(torch.float16, torch.bfloat16)
    def test_conv_deconv_3d_lower_precision(self, dtype):
        self._test_conv_deconv_lower_precision_base(3, torch.nn.Conv3d, dtype=dtype)
        self._test_conv_deconv_lower_precision_base(3, torch.nn.ConvTranspose3d, dtype=dtype)

    def _test_conv_deconv_nhwc_base(self, conv_module, weight_memory_format, dtype, prec=None):
        input_shapes = {2: (55, 55), 3: (14, 14, 14)}
        options = itertools.product([True, False], [True, False], [1, 2], [1, 4])
        if conv_module in [torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
            cl_format = torch.channels_last
            input_shape = input_shapes[2]
        elif conv_module in [torch.nn.Conv3d, torch.nn.ConvTranspose3d]:
            cl_format = torch.channels_last_3d
            input_shape = input_shapes[3]

        for train, bias, dilation, groups in options:
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shape
            x = torch.randn(x_shape, dtype=dtype)

            # conv1: mkldnn conv/deconv in contiguous memory format (nchw)
            # conv2: mkldnn conv/deconv in channels last memory format (nhwc)
            conv1 = conv_module(in_channels=C,
                                out_channels=M,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                dilation=dilation,
                                bias=bias,
                                groups=groups).to(dtype=dtype)
            conv2 = copy.deepcopy(conv1).to(memory_format=weight_memory_format)
            x1 = x.clone()
            x2 = x.clone().to(memory_format=cl_format)
            if train:
                x1.requires_grad_()
                x2.requires_grad_()
            y1 = conv1(x1)
            y2 = conv2(x2)
            self.assertEqual(y1, y2, atol=prec, rtol=prec)

            if train:
                y1.sum().backward()
                y2.sum().backward()
                self.assertTrue(x2.grad.is_contiguous(memory_format=cl_format))
                self.assertEqual(conv1.weight.grad,
                                 conv2.weight.grad,
                                 atol=1e-3,
                                 rtol=1e-3)
                if bias:
                    self.assertEqual(conv1.bias.grad, conv2.bias.grad, atol=prec, rtol=prec)
                self.assertEqual(x1.grad, x2.grad, atol=prec, rtol=prec)

    @reduced_f32_on_and_off()
    def test_conv_nhwc_fp32(self):
        self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.contiguous_format, dtype=torch.float32)
        self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.channels_last, dtype=torch.float32)
        self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.contiguous_format, dtype=torch.float32)
        self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.channels_last_3d, dtype=torch.float32)

    @dtypes(torch.float16, torch.bfloat16)
    def test_conv_nhwc_lower_precision(self, dtype):
        # when torch.ops.mkldnn._is_mkldnn_bf16_supported() or torch.ops.mkldnn._is_mkldnn_fp16_supported()
        # returns false, bf16/fp16 CPU conv will fall back to thnn impl
        support_checks = {
            torch.bfloat16: torch.ops.mkldnn._is_mkldnn_bf16_supported,
            torch.float16: torch.ops.mkldnn._is_mkldnn_fp16_supported
        }
        if support_checks[dtype]():
            self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.contiguous_format, dtype=dtype)
            self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.channels_last, dtype=dtype)
            self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.contiguous_format, dtype=dtype)
            self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.channels_last_3d, dtype=dtype)

        # BF16/FP16 fallback implementations are divided into two parts im2col+gemm,
        # and the number of data type conversions in the middle is more than that of onednn's direct conv,
        # resulting in additional accuracy loss.
        precisions = {
            torch.bfloat16: 1e-2,
            torch.float16: 2e-3,
        }
        prec = precisions[dtype]
        with torch.backends.mkldnn.flags(enabled=False):
            self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.contiguous_format, dtype=dtype, prec=prec)
            self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.channels_last, dtype=dtype, prec=prec)
            self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.contiguous_format, dtype=dtype, prec=prec)
            self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.channels_last_3d, dtype=dtype, prec=prec)


    @reduced_f32_on_and_off()
    def test_conv_transpose_nhwc_fp32(self):
        self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.contiguous_format, dtype=torch.float32)
        self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.channels_last, dtype=torch.float32)
        self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.contiguous_format, dtype=torch.float32)
        self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.channels_last_3d, dtype=torch.float32)

    @dtypes(torch.float16, torch.bfloat16)
    def test_conv_transpose_nhwc_lower_precision(self, dtype):
        # when torch.ops.mkldnn._is_mkldnn_bf16_supported() or torch.ops.mkldnn._is_mkldnn_fp16_supported()
        # returns false, bf16/fp16 CPU conv will fall back to thnn impl
        support_checks = {
            torch.bfloat16: torch.ops.mkldnn._is_mkldnn_bf16_supported,
            torch.float16: torch.ops.mkldnn._is_mkldnn_fp16_supported
        }
        if support_checks[dtype]():
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.contiguous_format, dtype=dtype)
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.channels_last, dtype=dtype)
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.contiguous_format, dtype=dtype)
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.channels_last_3d, dtype=dtype)

        # BF16/FP16 fallback implementations are divided into two parts col2im+gemm,
        # and the number of data type conversions in the middle is more than that of onednn's direct conv,
        # resulting in additional accuracy loss.
        precisions = {
            torch.bfloat16: 2e-2,
            torch.float16: 3e-3,
        }
        prec = precisions[dtype]
        with torch.backends.mkldnn.flags(enabled=False):
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.contiguous_format, dtype=dtype, prec=prec)
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.channels_last, dtype=dtype, prec=prec)
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.contiguous_format, dtype=dtype, prec=prec)
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.channels_last_3d, dtype=dtype, prec=prec)

    def _test_conv_transpose_base(self, dim):
        conv_module = {
            1: torch.nn.ConvTranspose1d,
            2: torch.nn.ConvTranspose2d,
            3: torch.nn.ConvTranspose3d
        }
        input_shapes = {1: (55,), 2: (28, 28), 3: (14, 14, 14)}
        options = itertools.product([True, False], [True, False], [1, 2], [1, 4])
        for train, bias, dilation, groups in options:
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shapes[dim]
            data = torch.randn(x_shape, dtype=torch.float32)
            # conv: mkldnn transpose conv fp32
            # conv_ref: thnn transpose conv fp32
            conv = conv_module[dim](in_channels=C,
                                    out_channels=M,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    dilation=dilation,
                                    bias=bias,
                                    groups=groups).to(dtype=torch.float32)
            x = data.clone()
            x_ref = x.clone()
            if train:
                x.requires_grad_()
                x_ref.requires_grad_()

            conv_ref = copy.deepcopy(conv)
            with torch.backends.mkldnn.flags(enabled=False):
                y_ref = conv_ref(x_ref)
                if train:
                    y_ref.sum().backward()

            y = conv(x)
            if train:
                y.sum().backward()

            if self.precision != 0:
                self.assertEqual(y, y_ref, atol=self.precision, rtol=self.precision)
            else:
                self.assertEqual(y, y_ref)

            if train:
                self.assertEqual(x.grad, x_ref.grad)
                self.assertEqual(conv.weight.grad,
                                 conv_ref.weight.grad,
                                 atol=1e-3,
                                 rtol=1e-3)
                if bias:
                    self.assertEqual(conv.bias.grad, conv_ref.bias.grad)

    @reduced_f32_on_and_off()
    def test_conv_transpose1d(self):
        self._test_conv_transpose_base(dim=1)

    @reduced_f32_on_and_off()
    def test_conv_transpose2d(self):
        self._test_conv_transpose_base(dim=2)

    @reduced_f32_on_and_off()
    def test_conv_transpose3d(self):
        self._test_conv_transpose_base(dim=3)

    def test_conv2d_legacy_jit_model(self):
        """
        MKLDNN integration used to serialize models with 5d weight for grouped
        convolutions, we'd like to preserve this behavior
        """
        g = 4
        conv2d = torch.nn.Conv2d(16, 16, 3, groups=g)
        conv2d_mkldnn = torch.utils.mkldnn.to_mkldnn(conv2d)

        # contrive legacy conv2d module with a 5-d weight
        o, i, h, w = conv2d.weight.shape
        weight_5d = conv2d.weight.reshape((g, o // g, i, h, w))
        conv2d_mkldnn.weight = weight_5d.to_mkldnn()

        x = torch.randn(1, 16, 8, 8)

        with TemporaryFileName() as fname:
            torch.jit.save(conv2d_mkldnn, fname)
            conv2d_loaded = torch.jit.load(fname)

            self.assertEqual(conv2d_mkldnn.weight.ndimension(), 5)
            self.assertEqual(conv2d_loaded.weight.ndimension(), 4)
            self.assertEqual(
                conv2d(x),
                conv2d_loaded(x.to_mkldnn()).to_dense())

    # This test is to check whether 1D conv is supported for mkldnn tensor,
    # which is exposed by Issue https://github.com/pytorch/pytorch/issues/68034.
    def test_conv1d_functional(self):
        input = torch.randn(2, 3, 10).to_mkldnn()
        weight = torch.randn(3, 3, 3).to_mkldnn()
        bias = torch.randn(3).to_mkldnn()
        output = torch.nn.functional.conv1d(input, weight, bias)
        self.assertEqual(output.size(), torch.Size([2, 3, 8]))

    def test_relu(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        y1 = torch.relu(x1)
        y2 = torch.relu(x2).to_dense()
        loss1 = y1.sum()
        loss2 = y2.sum()
        loss1.backward()
        loss2.backward()
        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_relu_(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        y1 = torch.relu_(x1.clone())
        y2 = torch.relu_(x2.clone()).to_dense()
        loss1 = y1.sum()
        loss2 = y2.sum()
        loss1.backward()
        loss2.backward()
        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad.to_dense())

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_relu_bf16_base(self, name):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x_bf16 = x.bfloat16()
        fn = getattr(torch, name)
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            y = fn(x.to_mkldnn()).to_dense()
            y_bf16 = fn(x_bf16.to_mkldnn()).to_dense(torch.float32)
            self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
        else:
            msg = r"bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: fn(x_bf16.to_mkldnn()))

    def test_relu_bf16(self):
        self._test_relu_bf16_base("relu")

    def test_relu_inplace_bf16(self):
        self._test_relu_bf16_base("relu_")

    def test_gelu(self):
        m = torch.nn.GELU()
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        y1 = m(x1)
        y2 = m(x2).to_dense()
        loss1 = y1.sum()
        loss2 = y2.sum()
        loss1.backward()
        loss2.backward()
        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad.to_dense())

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def test_gelu_bf16(self):
        m = torch.nn.GELU()
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x1 = x.clone().to_mkldnn().requires_grad_()
        x2 = x.clone().to_mkldnn(torch.bfloat16).requires_grad_()
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            y1 = m(x1).to_dense()
            y2 = m(x2).to_dense()
            loss1 = y1.sum()
            loss2 = y2.sum()
            loss1.backward()
            loss2.backward()
            self.assertEqual(y1, y2.to(torch.float32), atol=1e-1, rtol=0)
            self.assertEqual(x1.grad.to_dense(), x2.grad.to_dense(torch.float32), atol=1e-2, rtol=0)
        else:
            msg = r"bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: m(x2))

    def _test_prelu_base(self, size, num_channels):
        x = torch.randn(size, dtype=torch.float32)
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        x3 = x.clone().to_mkldnn().requires_grad_()
        m1 = torch.nn.PReLU(num_channels)
        m2 = mkldnn_utils.to_mkldnn(copy.deepcopy(m1))
        m3 = copy.deepcopy(m1)
        y1 = m1(x1)
        y2 = m2(x2).to_dense()
        y3 = m3(x3).to_dense()  # Only convert data to mkldnn, weight is Aten tensor
        loss1 = y1.sum()
        loss1.backward()
        loss2 = y2.sum()
        loss2.backward()
        loss3 = y3.sum()
        loss3.backward()
        self.assertEqual(y1, y2)
        self.assertEqual(y1, y3)
        self.assertEqual(x1.grad, x2.grad.to_dense())
        self.assertEqual(x1.grad, x3.grad.to_dense())

    def test_prelu(self):
        self._test_prelu_base(torch.Size([16]), 1)
        self._test_prelu_base(torch.Size([16, 64]), 1)
        self._test_prelu_base(torch.Size([16, 64]), 64)
        self._test_prelu_base(torch.Size([16, 64, 112]), 1)
        self._test_prelu_base(torch.Size([16, 64, 112]), 64)
        self._test_prelu_base(torch.Size([16, 64, 112, 112]), 1)
        self._test_prelu_base(torch.Size([16, 64, 112, 112]), 64)
        self._test_prelu_base(torch.Size([16, 64, 112, 112, 1]), 1)
        self._test_prelu_base(torch.Size([16, 64, 112, 112, 1]), 64)

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_prelu_bf16_base(self, size, num_channels):
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            x = torch.randn(size, dtype=torch.float32)
            x_fp32 = x.clone().to_mkldnn().requires_grad_()
            x_bf16 = x.clone().to_mkldnn(torch.bfloat16).requires_grad_()
            m = mkldnn_utils.to_mkldnn(torch.nn.PReLU())
            m_bf16 = mkldnn_utils.to_mkldnn(torch.nn.PReLU(), torch.bfloat16)

            y = m(x_fp32).to_dense()
            y_bf16 = m_bf16(x_bf16).to_dense()
            self.assertEqual(y, y_bf16.to(torch.float32), atol=1e-1, rtol=1e-3)

            loss = y.sum()
            loss.backward()
            loss_bf16 = y_bf16.sum()
            loss_bf16.backward()
            self.assertEqual(x_fp32.grad.to_dense(), x_bf16.grad.to_dense(torch.float32))
        else:
            x_bf16 = torch.randn(size, dtype=torch.bfloat16).requires_grad_()
            m_bf16 = mkldnn_utils.to_mkldnn(torch.nn.PReLU(), torch.bfloat16)
            msg = r"bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: m_bf16(x_bf16))

    def test_prelu_bf16(self):
        self._test_prelu_bf16_base(torch.Size([16]), 1)
        self._test_prelu_bf16_base(torch.Size([16, 64]), 1)
        self._test_prelu_bf16_base(torch.Size([16, 64]), 64)
        self._test_prelu_bf16_base(torch.Size([16, 64, 112]), 1)
        self._test_prelu_bf16_base(torch.Size([16, 64, 112]), 64)
        self._test_prelu_bf16_base(torch.Size([16, 64, 112, 112, 1]), 1)
        self._test_prelu_bf16_base(torch.Size([16, 64, 112, 112, 1]), 64)

    def _test_max_pool_base(self, dim, input):
        pool_module = {2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        for stride in [1, 2, 3]:
            for ceil_mode in [False, True]:
                max_pool = pool_module[dim](
                    kernel_size=3 if not ceil_mode else 7,
                    stride=stride,
                    padding=1,
                    ceil_mode=ceil_mode)

                x1 = input.clone().requires_grad_()
                x2 = input.clone().to_mkldnn().requires_grad_()
                y1 = max_pool(x1)
                y2 = max_pool(x2).to_dense()
                loss1 = y1.sum()
                loss2 = y2.sum()
                loss1.backward()
                loss2.backward()
                self.assertEqual(y1, y2)
                self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_max_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            self._test_max_pool_base(dim=2, input=x)

    def test_max_pool3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
            x = torch.randn(N, C, D, H, W, dtype=torch.float32) * 10
            self._test_max_pool_base(dim=3, input=x)


    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_max_pool_bf16_base(self, dim, input):
        pool_module = {2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        x_bf16 = input.bfloat16()
        for stride in [1, 2, 3]:
            for ceil_mode in [False, True]:
                max_pool = pool_module[dim](
                    kernel_size=3 if not ceil_mode else 7,
                    stride=stride,
                    padding=1,
                    ceil_mode=ceil_mode)

                if torch.ops.mkldnn._is_mkldnn_bf16_supported():
                    y = max_pool(input.to_mkldnn()).to_dense()
                    y_bf16 = max_pool(x_bf16.to_mkldnn()).to_dense(torch.float32)
                    self.assertEqual(y, y_bf16, atol=0.1, rtol=1e-3)
                else:
                    msg = f"mkldnn_max_pool{dim:d}d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
                    self.assertRaisesRegex(RuntimeError,
                                           msg,
                                           lambda: max_pool(x_bf16.to_mkldnn()))

    def test_max_pool2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            self._test_max_pool_bf16_base(dim=2, input=x)

    def test_max_pool3d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
            x = torch.randn(N, C, D, H, W, dtype=torch.float32) * 10
            self._test_max_pool_bf16_base(dim=3, input=x)

    def test_max_pool2d_stride_none(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            for ceil_mode in [False, True]:
                y1 = F.max_pool2d(
                    x,
                    kernel_size=3 if not ceil_mode else 7,
                    stride=None,
                    padding=1,
                    ceil_mode=ceil_mode)

                y2 = F.max_pool2d(
                    x.to_mkldnn(),
                    kernel_size=3 if not ceil_mode else 7,
                    stride=None,
                    padding=1,
                    ceil_mode=ceil_mode)

                self.assertEqual(y1, y2.to_dense())

    # https://github.com/pytorch/pytorch/issues/127111
    @xfailIfTorchDynamo
    def test_max_pool_unsupported(self):
        # OneDNN not support dilation max_pooling, will be avilabled in v2.0.
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        # 2d dilation case
        x = torch.randn(N, C, 7, 7, dtype=torch.float32).to_mkldnn()
        max_pool2d = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=3,
            padding=1,
            dilation=2)
        self.assertRaisesRegex(RuntimeError,
                               'mkldnn_max_pool2d does not support dilation case',
                               lambda: max_pool2d(x))

        # 3d dilation case
        x = torch.randn(N, C, 7, 7, 7, dtype=torch.float32).to_mkldnn()
        max_pool3d = torch.nn.MaxPool3d(
            kernel_size=3,
            stride=3,
            padding=1,
            dilation=2)
        self.assertRaisesRegex(RuntimeError,
                               'mkldnn_max_pool3d does not support dilation case',
                               lambda: max_pool3d(x))

    def _test_avg_pool_base(self, dim, input):
        avg_module = {2: torch.nn.AvgPool2d, 3: torch.nn.AvgPool3d}
        for count_include_pad in [True, False]:
            avg_pool = avg_module[dim](
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            x1 = input.clone().requires_grad_()
            x2 = input.clone().to_mkldnn().requires_grad_()
            y1 = avg_pool(x1)
            y2 = avg_pool(x2).to_dense()
            loss1 = y1.sum()
            loss2 = y2.sum()
            loss1.backward()
            loss2.backward()
            self.assertEqual(y1, y2)
            self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_avg_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_base(dim=2, input=x)

    def test_avg_pool3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_base(dim=3, input=x)

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_avg_pool_bf16_base(self, dim, input):
        avg_module = {2: torch.nn.AvgPool2d, 3: torch.nn.AvgPool3d}
        x_bf16 = input.bfloat16()
        for count_include_pad in [True, False]:
            avg_pool = avg_module[dim](
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)
            if torch.ops.mkldnn._is_mkldnn_bf16_supported():
                y = avg_pool(input.to_mkldnn()).to_dense()
                y_bf16 = avg_pool(x_bf16.to_mkldnn()).to_dense(torch.float)
                self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
            else:
                msg = f"mkldnn_avg_pool{dim:d}d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
                self.assertRaisesRegex(RuntimeError,
                                       msg,
                                       lambda: avg_pool(x_bf16.to_mkldnn()))

    def test_avg_pool2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_bf16_base(dim=2, input=x)

    def test_avg_pool3d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_bf16_base(dim=3, input=x)

    def test_avg_pool2d_stride_none(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10

        for count_include_pad in [True, False]:
            y1 = F.avg_pool2d(
                x,
                kernel_size=3,
                stride=None,
                padding=1,
                count_include_pad=count_include_pad)
            y2 = F.avg_pool2d(
                x.to_mkldnn(),
                kernel_size=3,
                stride=None,
                padding=1,
                count_include_pad=count_include_pad)

            self.assertEqual(y1, y2.to_dense())

    def test_adaptive_avg_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        y1 = adaptive_avg_pool2d(x1)
        y2 = adaptive_avg_pool2d(x2).to_dense()

        loss1 = y1.sum()
        loss2 = y2.sum()
        loss1.backward()
        loss2.backward()

        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad.to_dense())

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def test_adaptive_avg_pool2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        x_bf16 = x.bfloat16()
        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            y = adaptive_avg_pool2d(x.to_mkldnn()).to_dense()
            y_bf16 = adaptive_avg_pool2d(x.to_mkldnn()).to_dense(torch.float32)
            self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
        else:
            msg = "mkldnn_adaptive_avg_pool2d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: adaptive_avg_pool2d(x_bf16.to_mkldnn()))

    def _test_batch_norm_base(self, dim, channels, input):
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}
        bn = bn_module[dim](channels).float().train(False)
        mkldnn_bn = mkldnn_utils.to_mkldnn(copy.deepcopy(bn))
        self.assertEqual(
            bn(input),
            mkldnn_bn(input.to_mkldnn()).to_dense())

        self._test_serialization(mkldnn_bn, (input.to_mkldnn(),))
        self._test_tracing(mkldnn_bn, (input.to_mkldnn(),))

    def _test_batch_norm_train_base(self, dim, channels, input):
        # TODO: support 3d batchnorm training.
        bn_module = {2 : torch.nn.BatchNorm2d}
        # TODO: support none affine.
        options = itertools.product([True], [True, False])
        for affine, track_running_stats in options:
            bn = bn_module[dim](
                num_features=channels,
                affine=affine,
                track_running_stats=track_running_stats).float().train(True)
            mkldnn_bn = copy.deepcopy(bn)
            x1 = input.clone().requires_grad_()
            x2 = input.clone().to_mkldnn().requires_grad_()
            y1 = bn(x1)
            y2 = mkldnn_bn(x2).to_dense()
            loss1 = y1.sum()
            loss2 = y2.sum()
            loss1.backward()
            loss2.backward()
            self.assertEqual(y1, y2)
            self.assertEqual(x1.grad, x2.grad.to_dense())
            self.assertEqual(bn.weight.grad, mkldnn_bn.weight.grad, rtol=1e-3, atol=1e-3)
            if track_running_stats:
                self.assertEqual(bn.running_mean, mkldnn_bn.running_mean)
                self.assertEqual(bn.running_var, mkldnn_bn.running_var, rtol=1e-5, atol=1e-5)

    def test_batch_norm_2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        self._test_batch_norm_base(dim=2, channels=C, input=x)
        self._test_batch_norm_train_base(dim=2, channels=C, input=x)

    def test_batch_norm_3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 30, 30, 30, dtype=torch.float32) * 10
        self._test_batch_norm_base(dim=3, channels=C, input=x)

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_batch_norm_bf16_base(self, dim, channels, input):
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}
        x_bf16 = input.bfloat16()
        # TODO: support training
        for train in [False]:
            bn = bn_module[dim](channels).float().train(train)
            mkldnn_bn = mkldnn_utils.to_mkldnn(copy.deepcopy(bn))  # noqa: F841
            if torch.ops.mkldnn._is_mkldnn_bf16_supported():
                y = bn(input.to_mkldnn().to_dense())
                y_bf16 = bn(input.to_mkldnn().to_dense(torch.float))
                self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
            else:
                msg = "mkldnn_batch_norm: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
                self.assertRaisesRegex(RuntimeError,
                                       msg,
                                       lambda: bn(x_bf16.to_mkldnn()))

    def test_batch_norm_2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        self._test_batch_norm_bf16_base(dim=2, channels=C, input=x)

    def test_batch_norm_3d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 30, 30, 30, dtype=torch.float32) * 10
        self._test_batch_norm_bf16_base(dim=3, channels=C, input=x)

    def test_add(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        alpha = torc
```



## High-Level Overview


This Python file contains 2 class(es) and 105 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestMkldnn`, `EnsureMkldnn`

**Functions defined**: `test_conversion`, `test_conversion_byte_char`, `test_copy`, `test_unsupported`, `test_mkldnn_conv_shapecheck`, `test_autograd_to_mkldnn`, `func`, `test_autograd_from_mkldnn`, `func`, `test_detach`, `test_repr`, `_test_conv_base`, `test_conv1d`, `test_conv2d`, `test_conv3d`, `_test_conv_deconv_lower_precision_base`, `test_conv_deconv_1d_lower_precision`, `test_conv_deconv_2d_lower_precision`, `test_conv_deconv_3d_lower_precision`, `_test_conv_deconv_nhwc_base`

**Key imports**: copy, itertools, functools, unittest, warnings, nullcontext, torchvision, torch, torch.nn.functional as F, torch.jit


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `itertools`
- `functools`
- `unittest`
- `warnings`
- `contextlib`: nullcontext
- `torchvision`
- `torch`
- `torch.nn.functional as F`
- `torch.jit`
- `torch.backends.mkldnn`
- `torch.utils`: mkldnn as mkldnn_utils
- `torch.testing._internal.common_utils`: TestCase, 
- `torch.testing._internal.common_mkldnn`: reduced_f32_on_and_off


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_mkldnn.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_mkldnn.py_docs.md`
- **Keyword Index**: `test_mkldnn.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
