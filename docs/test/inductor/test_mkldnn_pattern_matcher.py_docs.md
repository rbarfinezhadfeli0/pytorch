# Documentation: test_mkldnn_pattern_matcher.py

## File Metadata
- **Path**: `test/inductor/test_mkldnn_pattern_matcher.py`
- **Size**: 168545 bytes
- **Lines**: 4778
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["oncall: cpu inductor"]
import contextlib
import copy
import itertools
import unittest

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import counters
from torch._inductor import config, metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    is_mkldnn_bf16_supported,
    is_mkldnn_fp16_supported,
    run_and_get_code,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.nn import functional as F
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off
from torch.testing._internal.common_quantization import (
    _generate_qdq_quantized_model,
    skipIfNoDynamoSupport,
    skipIfNoONEDNN,
    skipIfNoONEDNNBF16,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_LINUX,
    IS_X86,
    MI300_ARCH,
    MI350_ARCH,
    parametrize,
    skipIfNoXPU,
    skipIfRocm,
    skipIfRocmArch,
    skipIfXpu,
    TEST_ACL,
    TEST_MKL,
    xfailIfACL,
)
from torch.testing._internal.inductor_utils import (
    _check_has_dynamic_shape,
    clone_preserve_strides_offset,
    HAS_CPU,
)


# The dict value is match_nodes(computation_op+unary_op)

unary_list = {
    torch.nn.ReLU(): 2,
    torch.nn.Sigmoid(): 2,
    torch.nn.Tanh(): 2,
    torch.nn.Hardswish(): 6,
    torch.nn.LeakyReLU(0.1, inplace=False): 4,
    # Use floats for min/max, otherwise they can get converted to symints
    torch.nn.Hardtanh(min_val=-0.5, max_val=4.0, inplace=False): 3,
    torch.nn.Hardtanh(min_val=-0.5, max_val=float("inf"), inplace=False): 3,
    torch.nn.GELU(approximate="none"): 6,
    torch.nn.GELU(approximate="tanh"): 10,
    torch.nn.ReLU6(): 3,
    torch.nn.SiLU(): 3,
    torch.nn.Hardsigmoid(): 5,
}

non_decomposed_unary_list = [
    torch.nn.ReLU,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
]

# The dict value is (match_count, match_nodes, inplace)
binary_list = {
    lambda x, y: torch.add(x, y): (1, 2, False),  # call_function
    lambda x, y: torch.add(y, x): (1, 2, False),  # call_function
    lambda x, y: x.add(y): (1, 2, False),  # call_method
    lambda x, y: x.add_(y): (1, 2, True),  # call_method
    lambda x, y: torch.sub(x, y): (1, 2, False),  # call_function
    lambda x, y: x.sub(y): (1, 2, False),  # call_method
    lambda x, y: x.sub_(y): (1, 2, True),  # call_method
}

quantization_add_fn_list = [
    lambda x, y: torch.add(x, y),
    lambda x, y: x.add(y),
]

quantization_inplace_add_fn_list = [
    lambda x, y: x.add_(y),
]


def get_default_quantizer(is_qat, is_dynamic):
    quantizer = X86InductorQuantizer()
    quantizer.set_global(
        xiq.get_default_x86_inductor_quantization_config(
            is_qat=is_qat, is_dynamic=is_dynamic
        )
    )
    return quantizer


def cal_conv_generated_kernel_number(mod, input, dtype, dim=4, device="cpu"):
    # this function is to decide how many kernels are generated
    # while testing conv2d/3d/deconv2d
    # the assumption is:
    #   (1) There will be a to_dtype kernel for input for lp
    #   (2) inductor always use channel_last format, there will
    #       be a to_channel_last format for input
    #   (3) to_dtype and to_channel_last for input can be fused
    #   (4) inductor always get channel last format from mkldnn_conv_pointwise(binary),
    #       and force the output to have same stride with eager.
    #       So there will be a to_contiguous for output if eager output is contiguouse
    mod = copy.deepcopy(mod)
    mod = mod.to(device=device)
    input = input.clone()
    input = input.to(device)

    if dtype == torch.float32:
        maybe_autocast = contextlib.nullcontext()
    else:
        maybe_autocast = torch.amp.autocast(device_type=device, dtype=dtype)
    with torch.no_grad(), maybe_autocast:
        output = mod(input)
    input_kernel, output_kernel = 0, 0
    if (
        input.is_contiguous(memory_format=torch.contiguous_format)
        or dtype != torch.float32
        or (TEST_ACL and dim == 4)
    ):
        input_kernel = 1
    if output.is_contiguous(memory_format=torch.contiguous_format) or (
        TEST_ACL and dtype == torch.bfloat16
    ):
        output_kernel = 1

    return input_kernel + output_kernel


class TestPatternMatcherBase(TestCase):
    def setUp(self):
        super().setUp()
        self.ctx_stack = contextlib.ExitStack()
        self.ctx_stack.enter_context(config.patch({"freezing": True}))

    def tearDown(self):
        TestCase.tearDown(self)
        self.ctx_stack.close()

    def _check_unary_is_decomposed(self, unary_fn):
        return not any(
            isinstance(unary_fn, fn)
            for fn in [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh]
        )

    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _test_common(
        self,
        mod,
        inputs,
        matcher_check_fn,
        atol=1e-5,
        rtol=1.3e-6,
        check_autocast=torch.float32,
        check_quantization=False,
        is_qat=False,
        dtype=None,
        is_dynamic=False,
        quantizer=None,
        compile_options={},  # noqa: B006
        quantization_with_autocast=False,
    ):
        if not hasattr(self, "device"):
            has_xpu = any(
                isinstance(input, torch.Tensor) and input.device.type == "xpu"
                for input in inputs
            )
            device = "xpu" if has_xpu else "cpu"
        else:
            device = self.device

        mod = mod.to(device=device)
        if device != "cpu":
            inputs = tuple(
                clone_preserve_strides_offset(x, device=device) for x in inputs
            )
        counters.clear()
        torch._dynamo.reset()
        if check_autocast == torch.bfloat16 and is_mkldnn_bf16_supported(device):
            maybe_autocast = torch.amp.autocast(
                device_type=device, dtype=torch.bfloat16
            )
            atol, rtol = 1e-2, 1e-2
        elif check_autocast == torch.float16 and (is_mkldnn_fp16_supported(device)):
            maybe_autocast = torch.amp.autocast(device_type=device, dtype=torch.float16)
            atol, rtol = 1e-2, 1e-2
        else:
            assert check_autocast == torch.float32
            maybe_autocast = contextlib.nullcontext()
        if check_quantization:
            if quantization_with_autocast:
                with maybe_autocast:
                    convert_model = _generate_qdq_quantized_model(
                        mod, inputs, is_qat, is_dynamic, quantizer
                    )
            else:
                convert_model = _generate_qdq_quantized_model(
                    mod, inputs, is_qat, is_dynamic, quantizer
                )
            with torch.no_grad(), maybe_autocast:
                _ = torch.compile(convert_model)(*inputs)
                matcher_check_fn()
        else:
            with torch.no_grad(), maybe_autocast:
                clone_inputs = self._clone_inputs(inputs)
                expected = mod(*inputs)
                actual = torch.compile(mod, **compile_options)(*clone_inputs)
                if self.precision != 0:
                    torch.testing.assert_close(
                        actual, expected, atol=self.precision, rtol=self.precision
                    )
                else:
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                matcher_check_fn()

    def _test_code_common(
        self,
        mod,
        inputs,
        include_ops,
        exclude_ops,
        atol=1e-5,
        rtol=1.3e-6,
        check_quantization=False,
        check_dynamic=None,
        num_include_ops=None,
        quantizer=None,
    ):
        with torch.no_grad():
            clone_inputs = self._clone_inputs(inputs)
            if check_quantization:
                mod = _generate_qdq_quantized_model(mod, inputs, quantizer=quantizer)
            expected = mod(*inputs)
            actual, (source_code,) = run_and_get_code(
                torch.compile(mod, fullgraph=True, dynamic=check_dynamic),
                *clone_inputs,
            )
            assert_keywords = ["assert_size_stride", "assert_alignment"]
            filtered_lines = [
                line
                for line in source_code.splitlines()
                if not any(assert_key in line for assert_key in assert_keywords)
            ]
            source_code = "\n".join(filtered_lines)

            for op in include_ops:
                self.assertIn(op, source_code)
            if num_include_ops is not None:
                assert len(include_ops) == len(num_include_ops)
                for i in range(len(include_ops)):
                    self.assertEqual(
                        source_code.count(include_ops[i]), num_include_ops[i]
                    )
            for op in exclude_ops:
                self.assertNotIn(op, source_code)
            if check_dynamic is not None:
                _check_has_dynamic_shape(self, source_code)
            if not check_quantization:
                # Skip due to reduce range setting for Quantization on preCI system.
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class TestPatternMatcherGeneric(TestPatternMatcherBase):
    def _test_conv_unary_base(self, dim=4):
        assert dim == 4 or dim == 5

        class M(torch.nn.Module):
            def __init__(
                self,
                unary_fn,
                **kwargs,
            ):
                super().__init__()
                if dim == 4:
                    self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    self.conv = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.unary_fn = unary_fn

            def forward(self, x):
                x = self.conv(x)
                return self.unary_fn(x)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        options = itertools.product(
            unary_list.keys(),
            [torch.contiguous_format, cl_format],
            dtypes,
        )

        for (
            unary_fn,
            memory_format,
            dtype,
        ) in options:
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue
            metrics.reset()
            if dim == 4:
                x_shape = (1, 3, 56, 56)
            else:
                x_shape = (1, 3, 20, 56, 56)
            mod = M(unary_fn).to(memory_format=memory_format).eval()

            v = (
                torch.randn(x_shape, dtype=torch.float32)
                .add(1)
                .to(memory_format=memory_format)
            )

            def matcher_check_fn():
                match_nodes = unary_list[unary_fn]
                if dtype in (
                    torch.float16,
                    torch.bfloat16,
                ) and self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            generated_kernel_count = cal_conv_generated_kernel_number(
                mod, v, dtype, dim, self.device
            )
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    @reduced_f32_on_and_off()
    def test_conv2d_unary(self, device):
        self.device = device
        self._test_conv_unary_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    @reduced_f32_on_and_off()
    def test_conv3d_unary(self, device):
        self.device = device
        self._test_conv_unary_base(dim=5)

    def _test_conv_transpose_unary_base(self, dim=4):
        assert dim == 4 or dim == 5

        class M(torch.nn.Module):
            def __init__(
                self,
                unary_fn,
                **kwargs,
            ):
                super().__init__()
                if dim == 4:
                    self.conv_transpose = torch.nn.ConvTranspose2d(
                        3, 16, 3, stride=2, padding=1
                    )
                else:
                    self.conv_transpose = torch.nn.ConvTranspose3d(
                        3, 16, 3, stride=2, padding=1
                    )
                self.unary_fn = unary_fn

            def forward(self, x):
                x = self.conv_transpose(x)
                return self.unary_fn(x)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)

        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        options = itertools.product(
            unary_list,
            [torch.contiguous_format, cl_format],
            dtypes,
        )

        for unary_fn, memory_format, dtype in options:
            metrics.reset()
            if dim == 4:
                x_shape = (1, 3, 28, 28)
            else:
                x_shape = (1, 3, 17, 28, 28)
            mod = M(unary_fn).eval()

            v = torch.randn(x_shape, dtype=torch.float32).to(
                memory_format=memory_format
            )

            def matcher_check_fn():
                match_nodes = unary_list[unary_fn]
                if dtype in (
                    torch.float16,
                    torch.bfloat16,
                ) and self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            generated_kernel_count = cal_conv_generated_kernel_number(
                mod, v, dtype, dim, self.device
            )
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    @skipIfXpu(
        msg="The operator 'mkldnn::_convolution_transpose_pointwise' is not currently implemented for the XPU device."
    )
    @reduced_f32_on_and_off()
    def test_conv_transpose2d_unary(self, device):
        self.device = device
        self._test_conv_transpose_unary_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    @skipIfXpu(
        msg="The operator 'mkldnn::_convolution_transpose_pointwise' is not currently implemented for the XPU device."
    )
    @reduced_f32_on_and_off()
    def test_conv_transpose3d_unary(self, device):
        self.device = device
        self._test_conv_transpose_unary_base(dim=5)

    def _test_conv_binary_base(self, dim=4):
        assert dim == 4 or dim == 5

        class M(torch.nn.Module):
            def __init__(
                self,
                binary_fn,
                has_relu,
                **kwargs,
            ):
                super().__init__()
                if dim == 4:
                    self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                    self.conv2 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    self.conv1 = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                    self.conv2 = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.binary_fn = binary_fn
                self.has_relu = has_relu

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                if has_relu:
                    return self.binary_fn(x1, x2).relu()
                else:
                    return self.binary_fn(x1, x2)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        test_memory_format = [torch.contiguous_format, cl_format]
        options = itertools.product(
            binary_list,
            [True, False],
            test_memory_format,
            dtypes,
        )

        for (
            binary_fn,
            has_relu,
            memory_format,
            dtype,
        ) in options:
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue
            metrics.reset()
            if dim == 4:
                x_shape = (1, 3, 56, 56)
            else:
                x_shape = (1, 3, 20, 56, 56)
            mod = M(binary_fn, has_relu).eval()
            v = (
                torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
            )

            def matcher_check_fn():
                match_nodes = binary_list[binary_fn][1]
                if has_relu:
                    match_nodes += 1
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 2
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            generated_kernel_count = cal_conv_generated_kernel_number(
                mod, v, dtype, dim, self.device
            )
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    @reduced_f32_on_and_off(0.02)
    def test_conv2d_binary(self, device):
        self.device = device
        self._test_conv_binary_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    @reduced_f32_on_and_off(0.02)
    def test_conv3d_binary(self, device):
        self.device = device
        self._test_conv_binary_base(dim=5)

    def _test_conv_binary_broadcast_shapes_base(self, dim=4):
        assert dim == 4 or dim == 5

        class M(torch.nn.Module):
            def __init__(
                self,
                binary_fn,
                has_relu,
                **kwargs,
            ):
                super().__init__()
                if dim == 4:
                    self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    self.conv = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.binary_fn = binary_fn
                self.has_relu = has_relu

            def forward(self, x, x2):
                x1 = self.conv(x)
                if has_relu:
                    return self.binary_fn(x1, x2).relu()
                else:
                    return self.binary_fn(x1, x2)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        test_memory_format = [torch.contiguous_format, cl_format]
        if dim == 4:
            input_shapes = [
                [2, 3, 56, 56],
            ]
            other_shapes = [[2, 16, 1, 1], [1, 16, 1, 1], [1, 1, 1, 1]]
        else:
            input_shapes = [
                [2, 3, 20, 56, 56],
            ]
            other_shapes = [[2, 16, 1, 1, 1], [1, 16, 1, 1, 1], [1, 1, 1, 1, 1]]
        options = itertools.product(
            binary_list,
            input_shapes,
            other_shapes,
            [True, False],
            test_memory_format,
            dtypes,
        )

        for (
            binary_fn,
            x_shape,
            other_shape,
            has_relu,
            memory_format,
            dtype,
        ) in options:
            metrics.reset()
            mod = M(binary_fn, has_relu).eval()
            x = (
                torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
            )
            other = (
                torch.randn(other_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
                .to(dtype)
            )

            def matcher_check_fn():
                match_nodes = binary_list[binary_fn][1]
                if has_relu:
                    match_nodes += 1
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_nodes"], 1
                )

            self._test_common(mod, (x, other), matcher_check_fn, check_autocast=dtype)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    @reduced_f32_on_and_off()
    def test_conv2d_binary_broadcast_shapes(self, device):
        self.device = device
        self._test_conv_binary_broadcast_shapes_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    @reduced_f32_on_and_off()
    def test_conv3d_binary_broadcast_shapes(self, device):
        self.device = device
        self._test_conv_binary_broadcast_shapes_base(dim=5)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    @unittest.skipIf(IS_FBCODE, "Failing in fbcode")
    @reduced_f32_on_and_off()
    def test_conv2d_linear_add_broadcast_shapes(self, device):
        self.device = device

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.linear = torch.nn.Linear(3, 16)

            def forward(self, x1, x2):
                return self.conv(x1) + self.linear(x2)[:, :, None, None]

        metrics.reset()
        mod = M().eval()
        x1 = torch.randn(2, 3, 56, 56)
        x2 = torch.randn(2, 3)

        def matcher_check_fn():
            match_nodes = 0 if TEST_ACL else 2
            self.assertEqual(
                counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_nodes"],
                match_nodes,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_conv_weight_pack_matcher_nodes"], 1
            )

        self._test_common(mod, (x1, x2), matcher_check_fn)


class TestPatternMatcher(TestPatternMatcherBase):
    @reduced_f32_on_and_off()
    def test_linear_unary(self, device="cpu"):
        self.device = device

        class M(torch.nn.Module):
            def __init__(
                self,
                unary_fn,
                in_features,
                out_features,
                bias,
                **kwargs,
            ):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features,
                    out_features,
                    bias,
                    **kwargs,
                )
                self.unary_fn = unary_fn

            def forward(self, x):
                x = self.linear(x)
                return self.unary_fn(x)

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        if torch.backends.mkldnn.matmul.fp32_precision in ["bf16", "tf32"]:
            dtypes.append(torch.float32)
        options = itertools.product(unary_list, [True, False], dtypes)
        for unary_fn, bias, dtype in options:
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue
            metrics.reset()
            mod = M(unary_fn, 10, 30, bias=bias).eval()
            # only fuse for linear when the dtype is bf16
            v = torch.randn(2, 10)

            def matcher_check_fn():
                match_nodes = unary_list[unary_fn]
                if dtype != torch.float32 and self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            # only generated 1 kernel for "to_dtype"
            expected_kernel_count = 2 if TEST_ACL else 1
            if dtype == torch.float32:
                # In BF32, input is float32, will not generate kernel for "to_dtype"
                expected_kernel_count -= 1
            self.assertEqual(metrics.generated_kernel_count, expected_kernel_count)

    @reduced_f32_on_and_off()
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    def test_linear_fp32(self, device="cpu"):
        self.device = device

        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(10, 30, bias)

            def forward(self, x):
                return self.linear(x)

        for bias in [True, False]:
            mod = M(bias=bias).eval()
            v = torch.randn(2, 10)

            # packing pass.
            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn)

    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    def test_linear_input_non_contiguous_3D_wo_bias(self, device="cpu"):
        self.device = device

        # Activation is 3D, non-contiguous and without Bias
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4096, 1024, bias=False)

            def forward(self, x):
                x = torch.ops.aten.permute.default(x, [0, 2, 1, 3])
                x = torch.ops.aten.reshape.default(x, [4, 1, 4096])
                return self.linear(x)

        mod = M().eval()
        v = torch.randn(4, 32, 1, 128)

        dtypes = [torch.float]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)

        for dtype in dtypes:
            torch._dynamo.reset()
            autocast_enabled = dtype in [torch.bfloat16, torch.float16]
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type="cpu",
                    enabled=autocast_enabled,
                    dtype=dtype,
                ),
            ):
                expected = mod(v)
                actual, (source_code,) = run_and_get_code(
                    torch.compile(mod, fullgraph=True),
                    v,
                )
                self.assertIn(
                    "torch.ops.mkldnn._linear_pointwise.default"
                    if autocast_enabled
                    else "torch.ops.mkl._mkl_linear.default",
                    source_code,
                )
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @skipIfXpu(
        msg="Different with CPU, two linears will be concat on XPU for better performance"
    )
    def test_linear_add_bias(self, device="cpu"):
        self.device = device

        class M(torch.nn.Module):
            def __init__(self, device, dtype, unary_fn, cast_bias):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 64, bias=False)
                self.bias1 = torch.randn(64, device=device)
                self.linear2 = torch.nn.Linear(10, 64, bias=False)
                self.bias2 = torch.randn(64, device=device)
                if cast_bias:
                    self.bias1 = self.bias1.to(dtype=dtype, device=device)
                    self.bias2 = self.bias2.to(dtype=dtype, device=device)
                self.unary_fn = unary_fn

            def forward(self, x):
                a = self.linear1(x) + self.bias1
                b = self.linear2(x) + self.bias2
                return self.unary_fn(a), self.unary_fn(b)

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        options = itertools.product(unary_list, dtypes)
        for unary_fn, dtype in options:
            metrics.reset()
            fold_mod = M(self.device, dtype, unary_fn, cast_bias=True).eval()
            v = torch.randn(2, 10)

            def folder_matcher_check_fn():
                match_nodes = unary_list[unary_fn]
                if self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                # we have 2 linears, so we double the matcher_count/nodes
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_count"],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes * 2,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 2
                )

            self._test_common(
                fold_mod,
                (v,),
                folder_matcher_check_fn,
                check_autocast=dtype,
            )
            self.assertEqual(metrics.generated_kernel_count, 3 if TEST_ACL else 1)
            # we won't fold the bias if bias is not same dtype with weight
            # https://github.com/pytorch/pytorch/pull/129138
            metrics.reset()
            mod = M(self.device, dtype, unary_fn, cast_bias=False).eval()

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 2
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            # 1 kernel for "to_lowp", 2 kernels for unary ops
            self.assertEqual(metrics.generated_kernel_count, 3)

    @reduced_f32_on_and_off()
    def test_linear_binary(self, device="cpu"):
        self.device = device

        class M(torch.nn.Module):
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.binary_fn = binary_fn

            def forward(self, x, y):
                x = self.linear(x)
                x = self.binary_fn(x, y.clone())
                return x

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        if torch.backends.mkldnn.matmul.fp32_precision in ["bf16", "tf32"]:
            dtypes.append(torch.float32)
        options = itertools.product(
            binary_list, [[2, 3, 10], [2, 10]], [True, False], dtypes
        )
        out_feature = 30

        for binary_fn, input_shape, bias, dtype in options:
            metrics.reset()
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else 2,
                )
                reshape_linear_reshape_match_nodes = 3 if len(input_shape) == 3 else 0
                self.assertEqual(
                    counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"],
                    reshape_linear_reshape_match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 1
                )

            mod = M(binary_fn, input_shape[-1], out_feature, bias).eval()
            v = torch.randn(input_shape)
            other = torch.randn(input_shape[:-1] + [out_feature]).to(dtype)
            self._test_common(
                mod,
                (
                    v,
                    other,
                ),
                matcher_check_fn,
                check_autocast=dtype,
            )
            # only generated 1 kernel for "to_dtype"
            expected_kernel_count = 2 if TEST_ACL else 1
            if dtype == torch.float32:
                # In BF32, input is float32, will not generate kernel for "to_dtype"
                expected_kernel_count -= 1
            self.assertEqual(metrics.generated_kernel_count, expected_kernel_count)

    def test_linear_binary_broadcast_shapes(self, device="cpu"):
        self.device = device

        class M(torch.nn.Module):
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.binary_fn = binary_fn

            def forward(self, x, y):
                x = self.linear(x)
                x = self.binary_fn(x, y.clone())
                return x

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        options = itertools.product(
            binary_list,
            (
                ([2, 3, 10], [1, 1, 30]),
                ([2, 10], [1, 30]),
            ),
            (True, False),
            dtypes,
        )
        out_feature = 30

        for binary_fn, (input_shape, other_shape), bias, dtype in options:
            metrics.reset()
            mod = M(binary_fn, input_shape[-1], out_feature, bias).eval()
            v = torch.randn(input_shape)
            other = torch.randn(other_shape).to(dtype)

            def matcher_check_fn():
                reshape_linear_reshape_match_nodes = 3 if len(input_shape) == 3 else 0
                self.assertEqual(
                    counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"],
                    reshape_linear_reshape_match_nodes,
                )
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_nodes"], 1
                )

            self._test_common(
                mod,
                (
                    v,
                    other,
                ),
                matcher_check_fn,
                check_autocast=dtype,
            )
            self.assertEqual(metrics.generated_kernel_count, 2 if TEST_ACL else 1)

    @skipIfXpu(
        msg="Different with CPU, two linears will be concat on XPU for better performance"
    )
    def test_multi_linear_share_same_input(self, device="cpu"):
        self.device = device

        # llama pattern.
        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.w1 = torch.nn.Linear(16, 16, bias=False)
                self.w2 = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                return F.silu(self.w1(x)) * F.relu(self.w2(x))

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                0 if TEST_ACL else 7,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_unary_fusion_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"], 6
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 2
            )

        for dtype in dtypes:
            mod = M().to(dtype).eval()
            v = torch.randn(2, 4, 16).to(dtype)
            self._test_common(mod, (v,), matcher_check_fn, rtol=1e-2, atol=1e-2)

    def _qconv2d_test_helper(
        self,
        device="cpu",
        int8_mixed_bf16=False,
        quantization_with_autocast=False,
    ):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1)
                self.conv3 = torch.nn.Conv2d(
                    128, 128, kernel_size=3, stride=1, groups=4
                )

            def forward(self, x):
                return self.conv3(self.conv2(self.conv(x)))

        mod = M().eval().to(device=device)
        v = (
            torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False)
            .add(1)
            .to(device=device)
        )

        def matcher_check_fn():
            # 1. Dequant-Conv2D pattern matched in QConv2D weight prepack * 1
            #    int8_mixed_fp32: [dequant_node, dequantize_per_channel, clone, convolution]
            #    int8_mixed_bf16: [dequant_node, optional(convert_element_type_4),
            #     dequantize_per_channel, optional(convert_element_type_3), clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 3
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_nodes"],
                (16 if quantization_with_autocast else 18) if int8_mixed_bf16 else 12,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 3
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            quantization_with_autocast=quantization_with_autocast,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_cpu(self):
        r"""
        This testcase will quantize a single Conv2d module.
        """
        self._qconv2d_test_helper("cpu")

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_xpu(self):
        r"""
        This testcase will quantize a single Conv2d module.
        """
        self._qconv2d_test_helper("xpu")

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocmArch(MI300_ARCH + MI350_ARCH)
    def test_qconv2d_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a single Conv2d module with int8_mixed_bf16 quantization.
        """
        self._qconv2d_test_helper(int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocmArch(MI300_ARCH + MI350_ARCH)
    def test_qconv2d_int8_mixed_bf16_use_autocast(self):
        r"""
        This testcase will quantize a single Conv2d module with int8_mixed_bf16 quantization.
        """
        self._qconv2d_test_helper(int8_mixed_bf16=True, quantization_with_autocast=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_int8_mixed_bf16_xpu(self):
        r"""
        This testcase will quantize a single Conv2d module with int8_mixed_bf16 quantization.
        """
        self._qconv2d_test_helper(device="xpu", int8_mixed_bf16=True)

    def _qconv2d_unary_test_helper(
        self,
        device="cpu",
        int8_mixed_bf16=False,
        unary_op=torch.nn.ReLU(),
        qconv_unary_matcher_nodes=None,
    ):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.unary_fn = copy.deepcopy(unary_op)
                self.conv2 = torch.nn.Conv2d(
                    128, 128, kernel_size=3, stride=1, bias=False
                )
                self.unary_fn2 = copy.deepcopy(unary_op)

            def forward(self, x):
                tmp = self.unary_fn(self.conv(x))
                return self.unary_fn2(self.conv2(tmp))

        mod = M().eval().to(device=device)
        v = (
            torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False)
            .add(1)
            .to(device=device)
        )

        def matcher_check_fn():
            # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 2
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 2
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 2
            )
            if qconv_unary_matcher_nodes:
                self.assertEqual(
                    counters["inductor"]["qconv_unary_matcher_nodes"],
                    0 if TEST_ACL else qconv_unary_matcher_nodes,
                )

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_relu_cpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern.
        """
        self._qconv2d_unary_test_helper(device="cpu")

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_relu_xpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern.
        """
        self._qconv2d_unary_test_helper(device="xpu")

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_relu_int8_mixed_bf16_xpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qconv2d_unary_test_helper(int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_relu6_cpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU6 pattern.
        """
        self._qconv2d_unary_test_helper(device="cpu", unary_op=torch.nn.ReLU6())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_relu6_xpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU6 pattern.
        """
        self._qconv2d_unary_test_helper(device="xpu", unary_op=torch.nn.ReLU6())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_hardtanh_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern.
        """
        self._qconv2d_unary_test_helper(device="cpu", unary_op=torch.nn.Hardtanh())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_hardtanh_xpu(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern.
        """
        self._qconv2d_unary_test_helper(device="xpu", unary_op=torch.nn.Hardtanh())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_hardtanh_int8_mixed_bf16_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, clamp_min, clamp_max, convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, clamp_min, clamp_max, convert_element_type]
        """
        self._qconv2d_unary_test_helper(
            unary_op=torch.nn.Hardtanh(),
            int8_mixed_bf16=True,
            qconv_unary_matcher_nodes=11,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_hardtanh_int8_mixed_bf16_xpu(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, clamp_min, clamp_max, convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, clamp_min, clamp_max, convert_element_type]
        """
        self._qconv2d_unary_test_helper(
            device="xpu",
            unary_op=torch.nn.Hardtanh(),
            int8_mixed_bf16=True,
            qconv_unary_matcher_nodes=11,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_hardswish_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern.
        """
        self._qconv2d_unary_test_helper(device="cpu", unary_op=torch.nn.Hardswish())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_hardswish_xpu(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern.
        """
        self._qconv2d_unary_test_helper(device="xpu", unary_op=torch.nn.Hardswish())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_hardswish_int8_mixed_bf16_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, add, clamp_min,
             clamp_max, mul, div, convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, add, clamp_min, clamp_max, mul, div, convert_element_type]
        """
        self._qconv2d_unary_test_helper(
            unary_op=torch.nn.Hardswish(),
            int8_mixed_bf16=True,
            qconv_unary_matcher_nodes=17,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_hardswish_int8_mixed_bf16_xpu(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, add, clamp_min,
             clamp_max, mul, div, convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, add, clamp_min, clamp_max, mul, div, convert_element_type]
        """
        self._qconv2d_unary_test_helper(
            device="xpu",
            unary_op=torch.nn.Hardswish(),
            int8_mixed_bf16=True,
            qconv_unary_matcher_nodes=17,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_silu_cpu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern.
        """
        self._qconv2d_unary_test_helper(device="cpu", unary_op=torch.nn.SiLU())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_silu_xpu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern.
        """
        self._qconv2d_unary_test_helper(device="xpu", unary_op=torch.nn.SiLU())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_silu_int8_mixed_bf16_cpu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, sigmoid, mul,
             convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, sigmoid, mul, convert_element_type]
        """
        self._qconv2d_unary_test_helper(
            unary_op=torch.nn.SiLU(),
            int8_mixed_bf16=True,
            qconv_unary_matcher_nodes=11,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_silu_int8_mixed_bf16_xpu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, sigmoid, mul,
             convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, sigmoid, mul, convert_element_type]
        """
        self._qconv2d_unary_test_helper(
            device="xpu",
            unary_op=torch.nn.SiLU(),
            int8_mixed_bf16=True,
            qconv_unary_matcher_nodes=11,
        )

    def _qconv2d_add_test_helper(
        self, device="cpu", use_relu=False, int8_mixed_bf16=False
    ):
        r"""
        This testcase will quantize a Conv2d->Add pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
           Optional(relu)
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                use_relu,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU()
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, bias=False)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, bias=False)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)
                tmp1 = self.conv3(tmp)
                tmp2 = self.conv4(tmp)
                res = self.add_fn2(tmp1, tmp2)
                if self.use_relu:
                    res = self.relu2(res)
                return res

        for add_fn in quantization_add_fn_list + quantization_inplace_add_fn_list:
            mod = M(add_fn, use_relu).eval().to(device=device)
            v = (
                torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False)
                .add(1)
                .to(device=device)
            )

            def matcher_check_fn():
                # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 4
                self.assertEqual(
                    counters["inductor"]["qconv_weight_prepack_matcher_count"], 4
                )
                # 2. Qconv2d Binary Unary fusion in post-grad fusion pass * 2
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_lower_count"],
                    0 if TEST_ACL else 2,
                )

            self._test_common(
                mod,
                (v,),
                matcher_check_fn,
                check_quantization=True,
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            )

    def _qconv2d_add_test_helper2(
        self, device="cpu", use_relu=False, int8_mixed_bf16=False
    ):
        r"""
        This testcase will quantize two Conv2d->Add patterns as:

        Conv(X)   extra input
               \   /
                Add
                 |
           Optional(relu)
                 |
                 Y

        , and

        extra input   Conv(X)
               \   /
                Add
                 |
           Optional(relu)
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                use_relu,
                swap_inputs,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, bias=False)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU()
                self.use_relu = use_relu
                self.swap_inputs = swap_inputs

            def forward(self, x, x2, x3):
                x1 = self.conv1(x)
                if self.swap_inputs:
                    tmp = self.add_fn(x2, x1)
                else:
                    tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)
                tmp1 = self.conv2(tmp)
                if self.swap_inputs:
                    res = self.add_fn2(x3, tmp1)
                else:
                    res = self.add_fn2(tmp1, x3)
                if self.use_relu:
                    res = self.relu2(res)
                return res

        for add_fn, swap_inputs in itertools.product(
            quantization_add_fn_list + quantization_inplace_add_fn_list, [False, True]
        ):
            mod = M(add_fn, use_relu, swap_inputs).eval().to(device=device)
            x = torch.randn(
                (1, 3, 8, 8), dtype=torch.float32, requires_grad=False, device=device
            )
            x2 = torch.randn(
                (1, 6, 6, 6), dtype=torch.float32, requires_grad=False, device=device
            )
            x3 = torch.randn(
                (1, 6, 4, 4), dtype=torch.float32, requires_grad=False, device=device
            )

            def matcher_check_fn():
                # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 2
                self.assertEqual(
                    counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
                )
                # 2. Qconv2d Binary Unary fusion in post-grad fusion pass * 2
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_lower_count"],
                    0 if TEST_ACL else 2,
                )

            self._test_common(
                mod,
                (x, x2, x3),
                matcher_check_fn,
                check_quantization=True,
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_cpu(self):
        self._qconv2d_add_test_helper()
        self._qconv2d_add_test_helper2()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_add_xpu(self):
        self._qconv2d_add_test_helper(device="xpu")
        self._qconv2d_add_test_helper2(device="xpu")

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_add_int8_mixed_bf16(self):
        self._qconv2d_add_test_helper(int8_mixed_bf16=True)
        self._qconv2d_add_test_helper2(int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_add_int8_mixed_bf16_xpu(self):
        self._qconv2d_add_test_helper(device="xpu", int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_relu_cpu(self):
        self._qconv2d_add_test_helper(use_relu=True)
        self._qconv2d_add_test_helper2(use_relu=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_add_relu_xpu(self):
        self._qconv2d_add_test_helper(device="xpu", use_relu=True)
        self._qconv2d_add_test_helper2(device="xpu", use_relu=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_add_relu_int8_mixed_bf16(self):
        self._qconv2d_add_test_helper(use_relu=True, int8_mixed_bf16=True)
        self._qconv2d_add_test_helper2(use_relu=True, int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qconv2d_add_relu_int8_mixed_bf16_xpu(self):
        self._qconv2d_add_test_helper(device="xpu", use_relu=True, int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_broadcast_shapes_cpu(self):
        r"""
        This testcase will quantize Conv2d->add pattern using broadcast shape inputs.
        Conv2d->Add fusion will fail for the broadcast shape inputs case.
        """

        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.conv = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1)

            def forward(self, x1, x2):
                return torch.add(self.conv(x1), x2)

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            x1 = torch.randn((2, 32, 9, 9))
            x2 = torch.randn((2, 32, 1, 1))

            def matcher_check_fn():
                # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 1
                self.assertEqual(
                    counters["inductor"]["qconv_weight_prepack_matcher_count"], 1
                )
                # 2. Qconv2d Binary Unary fusion in post-grad fusion pass * 0
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"], 0
                )

            self._test_common(
                mod,
                (x1, x2),
                matcher_check_fn,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_with_concat_cpu(self):
        channel_1 = 32
        channel_2 = 16
        channel_3 = 8
        channel_4 = int(channel_2 * 2 + channel_3)

        class Model(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    channel_1, channel_2, 1, stride=1, dilation=1, padding=0
                )
                self.conv2 = torch.nn.Conv2d(
                    channel_1, channel_2, 1, stride=1, dilation=1, padding=0
                )
                self.conv3 = torch.nn.Conv2d(
                    channel_2, channel_3, 3, stride=1, dilation=1, padding=1
                )

                self.conv = torch.nn.Conv2d(
                    channel_4, channel_2, 1, stride=1, dilation=1, padding=0
                )

            def forward(self, x: torch.Tensor):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                x3 = self.conv3(x2)
                res = torch.cat([x1, x2, x3], dim=1)
                res = self.conv(res)
                return res

        mod = Model().eval()
        v = torch.randn(
            (8, channel_1, 40, 40), dtype=torch.float32, requires_grad=False
        )

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 4
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 3,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 4
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_2(self):
        r"""
        This testcase prevents this pattern be matched as a conv_binary fusion by mistake.
                Conv(X)  3
                    \   /
                     Add
        We see this pattern in Mobilenet v3 large which add is decomposed from torch.nn.Hardswish or torch.nn.Hardsigmoid.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                post_op,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.post_op = post_op

            def forward(self, x):
                return self.post_op(self.conv(x))

        for post_op in [
            torch.nn.Hardswish(inplace=True),
            torch.nn.Hardsigmoid(inplace=True),
        ]:
            mod = M(post_op).eval()
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )

            def matcher_check_fn():
                # Shouldn't hit conv binary fusion
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"], 0
                )

            self._test_common(
                mod,
                (v,),
                matcher_check_fn,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_3(self):
        r"""
        This testcase will test below model:
             x
           /   \
        conv1  maxpool
          \    /   \
           add    conv2
            \     /
              cat
        Based on default recipe of x86InductorQuantizer, we will see this pattern after convert:
        qconv1    maxpool
         \           |
          \         q1
           \       /   \
            \     dq1  qconv2
             \   /
              add
               |
               q2
        Since q1 has 2 users and qconv2 is not ancestor node of qconv1, we shouldn't fuse:
                int8
                 /
        qconv1 dq1
           \   /
            add
             |
             q2
             |
            int8
        Instead we can match and fuse this pattern into qconv_binary:
        qconv1  fp32
            \   /
             add
              |
             fp32
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
                self.maxpool = torch.nn.MaxPool2d(
                    kernel_size=3, stride=1, padding=0, dilation=1
                )

            def forward(self, x):
                tmp1 = self.conv1(x)
                tmp2 = self.maxpool(x)
                add = torch.add(tmp1, tmp2)
                tmp3 = self.conv2(tmp2)
                return torch.cat((add, tmp3), dim=1)

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_count"],
                0 if TEST_ACL else 1,
            )
            # The matched qconv binary pattern should have 2 nodes [qconv, add]
            # instead of 11 which has dequant in binary input and output quant
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_nodes"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_lower_count"],
                0 if TEST_ACL else 1,
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d(self):
        r"""
        This testcase will quantize a single Conv2d module with qat flow.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.bn = torch.nn.BatchNorm2d(128)

            def forward(self, x):
                return self.bn(self.conv(x))

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 1
            #    [dequantize_per_tensor, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 1
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_nodes"], 4
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default, quantize_per_tensor]
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 1,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_nodes"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 1
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            is_qat=True,
        )

    def _qat_qconv2d_unary_cpu_test_helper(
        self,
        unary_op=torch.nn.ReLU(),
    ):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                self.unary_fn = copy.deepcopy(unary_op)
                self.bn = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                self.unary_fn2 = copy.deepcopy(unary_op)
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                tmp = self.unary_fn(self.bn(self.conv(x)))
                return self.unary_fn2(self.bn2(self.conv2(tmp)))

        mod = M()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 1
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default, relu, div_1, round_2, add_1, clamp_min_1, clamp_max_1, convert_element_type_2]
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 2
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            is_qat=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_qconv2d_relu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern with qat flow.
        """

        self._qat_qconv2d_unary_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_qconv2d_relu6(self):
        r"""
        This testcase will quantize Conv2d->ReLU6 pattern with qat flow.
        """
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.ReLU6())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_qconv2d_hardtanh(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern with qat flow.
        """
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.Hardtanh())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_qconv2d_silu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern with qat flow.
        """
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.SiLU())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_qconv2d_hardswish(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern with qat flow.
        """
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.Hardswish())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_add(self):
        r"""
        This testcase will quantize a Conv2d->Add pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn1 = torch.nn.BatchNorm2d(6)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn2 = torch.nn.BatchNorm2d(6)

            def forward(self, x):
                x1 = self.bn1(self.conv1(x))
                x2 = self.bn2(self.conv2(x))
                return x1 + x2

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 2
            #    [dequantize_per_tensor, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_nodes"], 8
            )
            # 2. Qconv2d Binary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default_1, dequantize_per_tensor, add_3, quantize_per_tensor]
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_count"],
                0 if TEST_ACL else 1,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_nodes"],
                0 if TEST_ACL else 4,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_lower_count"],
                0 if TEST_ACL else 1,
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            is_qat=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_add_relu(self):
        r"""
        This testcase will quantize a Conv2d->Add->ReLU pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
                ReLU
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn1 = torch.nn.BatchNorm2d(6)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn2 = torch.nn.BatchNorm2d(6)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.bn1(self.conv1(x))
                x2 = self.bn2(self.conv2(x))
                return self.relu(x1 + x2)

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 2
            #    [dequantize_per_tensor, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_nodes"], 8
            )
            # 2. Qconv2d Binary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default_1, dequantize_per_tensor, add_3, relu, quantize_per_tensor]
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_count"],
                0 if TEST_ACL else 1,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_nodes"],
                0 if TEST_ACL else 5,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_lower_count"],
                0 if TEST_ACL else 1,
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            is_qat=True,
        )

    def _test_qconv2d_dequant_promotion_helper(self, device="cpu"):
        r"""
        This testcase tests if dequant node before conv2d is promoted correctly:
                 X
                 |
              Conv1(X)
               /   \
        Conv2(X)   Conv3(X)
               \   /
                Add
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)

            def forward(self, x):
                temp = self.conv1(x)
                temp = self.conv2(temp) + self.conv3(temp)
                return temp

        mod = M().eval().to(device=device)
        v = (
            torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False)
            .add(1)
            .to(device=device)
        )

        def matcher_check_fn():
            # 1. Dequant pattern matcher for dequant promotion * 1
            #    [dequantize_per_tensor]
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_nodes"], 1)
            # 2. Dequant-conv pattern matched in quantization weight prepack * 3
            #    [dequantize_per_tensor, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 3
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_nodes"], 12
            )
            # 3. Qconv2d Binary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default_1, add_3]
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_count"],
                0 if TEST_ACL else 1,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_nodes"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_lower_count"],
                0 if TEST_ACL else 1,
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_dequant_promotion_cpu(self):
        self._test_qconv2d_dequant_promotion_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    @skipIfNoXPU
    def test_qconv2d_dequant_promotion_xpu(self):
        self._test_qconv2d_dequant_promotion_helper(device="xpu")

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv1d_relu_cpu(self):
        r"""
        This testcase will quantize Conv1d->ReLU pattern.
        """
        device = "cpu"
        unary_op = torch.nn.ReLU()

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 128, kernel_size=3, stride=1)
                self.unary_fn = copy.deepcopy(unary_op)
                self.conv2 = torch.nn.Conv1d(
                    128, 128, kernel_size=3, stride=1, bias=False
                )
                self.unary_fn2 = copy.deepcopy(unary_op)

            def forward(self, x):
                tmp = self.unary_fn(self.conv(x))
                return self.unary_fn2(self.conv2(tmp))

        mod = M().eval().to(device=device)
        v = (
            torch.randn((1, 3, 8), dtype=torch.float32, requires_grad=False)
            .add(1)
            .to(device=device)
        )

        def matcher_check_fn():
            # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 2
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 2
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 2
            )

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            matcher_check_fn=matcher_check_fn,
        )

    def _qlinear_test_helper(
        self,
        inputs,
        device="cpu",
        int8_mixed_bf16=False,
        do_permute=False,
        matcher_check_fn=None,
        bias=True,
        is_dynamic=False,
        is_qat=False,
        quantization_with_autocast=False,
    ):
        class M(torch.nn.Module):
            def __init__(self, use_bias, do_permute=False):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3, use_bias)
                self.linear2 = torch.nn.Linear(3, 4, use_bias)
                self.do_permute = do_permute

            def forward(self, x):
                if self.do_permute:
                    x = torch.reshape(torch.permute(x, (0, 2, 3, 1)), (2, 12, 4))
                return self.linear2(self.linear(x))

        mod = M(bias, do_permute=do_permute).eval().to(device=device)
        assert isinstance(inputs, tuple)

        def __convert_tensor_to_device(input, device):
            return input.to(device=device) if isinstance(input, torch.Tensor) else input

        inputs = tuple(__convert_tensor_to_device(input, device) for input in inputs)

        def _default_matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
            )

        self._test_common(
            mod,
            inputs,
            matcher_check_fn=(
                matcher_check_fn
                if matcher_check_fn is not None
                else _default_matcher_check_fn
            ),
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            check_quantization=True,
            is_qat=is_qat,
            is_dynamic=is_dynamic,
            quantization_with_autocast=quantization_with_autocast,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_cpu(self):
        r"""
        This testcase will quantize a single Linear Module.
        """
        for bias in [True, False]:
            self._qlinear_test_helper((torch.randn((2, 4)),), bias=bias)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qlinear_xpu(self):
        r"""
        This testcase will quantize a single Linear Module.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 4)).to(device="xpu"),), device="xpu", bias=bias
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_dynamic_qlinear_cpu(self):
        r"""
        This testcase will quantize a single Linear Module.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 4)),), bias=bias, is_dynamic=True
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_dynamic_qlinear_qat_cpu(self):
        r"""
        This testcase will quantize a single Linear Module.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 4)),), bias=bias, is_dynamic=True, is_qat=True
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_dynamic_qlinear_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a single Linear Module.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 3, 4)),), bias=bias, is_dynamic=True
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a single Linear Module with int8_mixed_bf16 quantization.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 4)),), int8_mixed_bf16=True, bias=bias
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16_use_autocast(self):
        r"""
        This testcase will quantize a single Linear Module with int8_mixed_bf16 quantization.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 4)),),
                int8_mixed_bf16=True,
                bias=bias,
                quantization_with_autocast=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoXPU
    def test_qlinear_int8_mixed_bf16_xpu(self):
        r"""
        This testcase will quantize a single Linear Module with int8_mixed_bf16 quantization.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 4)).to(device="xpu"),),
                device="xpu",
                int8_mixed_bf16=True,
                bias=bias,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a single Linear Module.
        """
        for bias in [True, False]:
            self._qlinear_test_helper((torch.randn((2, 3, 4)),), bias=bias)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qlinear_input_dim_exceeds_2_xpu(self):
        r"""
        This testcase will quantize a single Linear Module.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 3, 4)).to(device="xpu"),), device="xpu", bias=bias
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a single Linear Module with int8_mixed_bf16 quantization.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 3, 4)),), int8_mixed_bf16=True, bias=bias
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2_use_autocast(self):
        r"""
        This testcase will quantize a single Linear Module with int8_mixed_bf16 quantization.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 3, 4)),),
                int8_mixed_bf16=True,
                bias=bias,
                quantization_with_autocast=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2_xpu(self):
        r"""
        This testcase will quantize a single Linear Module with int8_mixed_bf16 quantization.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 3, 4)).to(device="xpu"),),
                device="xpu",
                int8_mixed_bf16=True,
                bias=bias,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_input_dim_exceeds_2_and_not_contiguous(self):
        r"""
        This testcase will quantize a single Linear Module.
        * Input dim exceeds 2
        * Input not contiguous
        """
        for bias in [True, False]:

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    13 if bias else 12,
                )

            self._qlinear_test_helper(
                (torch.randn((2, 4, 3, 4)),),
                do_permute=True,
                matcher_check_fn=matcher_check_fn,
                bias=bias,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2_and_not_contiguous(self):
        r"""
        This testcase will quantize a single Linear Module for int8_bf16.
        * Input dim exceeds 2
        * Input not contiguous
        """
        for bias in [True, False]:

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    17 if bias else 16,
                )

            self._qlinear_test_helper(
                (torch.randn((2, 4, 3, 4)),),
                int8_mixed_bf16=True,
                do_permute=True,
                matcher_check_fn=matcher_check_fn,
                bias=bias,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2_and_not_contiguous_use_autocast(
        self,
    ):
        r"""
        This testcase will quantize a single Linear Module for int8_bf16.
        * Input dim exceeds 2
        * Input not contiguous
        """
        for bias in [True, False]:

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    16 if bias else 15,
                )

            self._qlinear_test_helper(
                (torch.randn((2, 4, 3, 4)),),
                int8_mixed_bf16=True,
                do_permute=True,
                matcher_check_fn=matcher_check_fn,
                bias=bias,
                quantization_with_autocast=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2_and_not_contiguous_xpu(self):
        r"""
        This testcase will quantize a single Linear Module for int8_bf16.
        * Input dim exceeds 2
        * Input not contiguous
        """
        for bias in [True, False]:

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    17 if bias else 16,
                )

            self._qlinear_test_helper(
                (torch.randn((2, 4, 3, 4)).to(device="xpu"),),
                device="xpu",
                int8_mixed_bf16=True,
                do_permute=True,
                matcher_check_fn=matcher_check_fn,
                bias=bias,
            )

    def _qlinear_unary_test_helper(
        self, inputs, unary_op=torch.nn.ReLU(), device="cpu", int8_mixed_bf16=False
    ):
        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, use_bias)
                self.unary_fn = copy.deepcopy(unary_op)
                self.linear2 = torch.nn.Linear(4, 4, use_bias)
                self.unary_fn2 = copy.deepcopy(unary_op)

            def forward(self, x):
                tmp = self.unary_fn(self.linear(x))
                return self.unary_fn2(self.linear2(tmp))

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval().to(device=device)

            def matcher_check_fn():
                # 1. dequant-linear pattern matched in quantization weight prepack
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                # 2. QLinear Unary fusion in post-grad fusion pass
                self.assertEqual(
                    counters["inductor"]["qlinear_unary_matcher_count"],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_unary_lower_count"],
                    0 if TEST_ACL else 2,
                )

            self._test_common(
                mod,
                inputs,
                matcher_check_fn,
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_relu_cpu(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern.
        """
        self._qlinear_unary_test_helper((torch.randn((2, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qlinear_relu_xpu(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern.
        """
        self._qlinear_unary_test_helper(
            (torch.randn((2, 4)).to(device="xpu"),), device="xpu"
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_relu_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qlinear_unary_test_helper((torch.randn((2, 4)),), int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qlinear_relu_int8_mixed_bf16_xpu(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qlinear_unary_test_helper(
            (torch.randn((2, 4)).to(device="xpu"),), device="xpu", int8_mixed_bf16=True
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_relu_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern.
        """
        self._qlinear_unary_test_helper((torch.randn((2, 3, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qlinear_relu_input_dim_exceeds_2_xpu(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern.
        """
        self._qlinear_unary_test_helper(
            (torch.randn((2, 3, 4)).to(device="xpu"),), device="xpu"
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_relu_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qlinear_unary_test_helper((torch.randn((2, 3, 4)),), int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qlinear_relu_int8_mixed_bf16_input_dim_exceeds_2_xpu(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qlinear_unary_test_helper(
            (torch.randn((2, 3, 4)).to(device="xpu"),),
            device="xpu",
            int8_mixed_bf16=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_gelu_cpu(self):
        r"""
        This testcase will quantize a Linear->GELU pattern.
        """
        for gelu in [torch.nn.GELU("none"), torch.nn.GELU("tanh")]:
            self._qlinear_unary_test_helper((torch.randn((2, 4)),), gelu)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qlinear_gelu_xpu(self):
        r"""
        This testcase will quantize a Linear->GELU pattern.
        """
        for gelu in [torch.nn.GELU("none"), torch.nn.GELU("tanh")]:
            self._qlinear_unary_test_helper(
                (torch.randn((2, 4)).to(device="xpu"),), gelu, device="xpu"
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_gelu_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a Linear->GELU pattern with int8_mixed_bf16 quantization.
        """
        for gelu in [torch.nn.GELU("none"), torch.nn.GELU("tanh")]:
            self._qlinear_unary_test_helper(
                (torch.randn((2, 4)),), gelu, int8_mixed_bf16=True
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfNoXPU
    def test_qlinear_gelu_int8_mixed_bf16_xpu(self):
        r"""
        This testcase will quantize a Linear->GELU pattern with int8_mixed_bf16 quantization.
        """
        for gelu in [torch.nn.GELU("none"), torch.nn.GELU("tanh")]:
            self._qlinear_unary_test_helper(
                (torch.randn((2, 4)).to(device="xpu"),),
                gelu,
                device="xpu",
                int8_mixed_bf16=True,
            )

    def _qlinear_add_test_helper(
        self,
        device="cpu",
        use_relu=False,
        int8_mixed_bf16=False,
        is_qat=True,
        is_dynamic=True,
    ):
        r"""
        This testcase will quantize two consecutive Linear->Add(->relu) patterns as:
                 X
               /   \
        linear(X)   linear(X)
               \   /
                Add
                 |
           Optional(relu)
               /   \
        linear(X)   linear(X)
               \   /
                Add
                 |
           Optional(relu)
                 |
                 Y
        """

        def fake_quant(x):
            # to produce a float32 result as extra input
            qlib = torch.ops.quantized_decomposed
            if device == "cpu":
                qmin, qmax, dtype = 0, 255, torch.uint8
            else:
                qmin, qmax, dtype = -128, 127, torch.int8
            x = qlib.quantize_per_tensor.default(x, 0.0166785, 42, qmin, qmax, dtype)
            x = qlib.dequantize_per_tensor.default(x, 0.0166785, 42, qmin, qmax, dtype)
            return x

        class M(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                use_relu,
                fake_quant_before_extra_input,
            ):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU()
                self.linear3 = torch.nn.Linear(4, 4)
                self.linear4 = torch.nn.Linear(4, 4)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU()
                self.use_relu = use_relu
                self.fake_quant_before_extra_input = fake_quant_before_extra_input

            def forward(self, x):
                x1 = self.linear1(x)
                x2 = self.linear2(x)
                if self.fake_quant_before_extra_input:
                    x2 = fake_quant(x2)
                tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)
    

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 70 class(es): TestPatternMatcherBase, TestPatternMatcherGeneric, M, M, M, M, M, TestPatternMatcher, M, M, M, M, M, M, M, M, M, M, M, M

### Functions
This file defines 345 function(s): get_default_quantizer, cal_conv_generated_kernel_number, setUp, tearDown, _check_unary_is_decomposed, _clone_inputs, clone, _test_common, _test_code_common, _test_conv_unary_base, __init__, forward, matcher_check_fn, test_conv2d_unary, test_conv3d_unary, _test_conv_transpose_unary_base, __init__, forward, matcher_check_fn, test_conv_transpose2d_unary, test_conv_transpose3d_unary, _test_conv_binary_base, __init__, forward, matcher_check_fn, test_conv2d_binary, test_conv3d_binary, _test_conv_binary_broadcast_shapes_base, __init__, forward


## Key Components

The file contains 11227 words across 4778 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 168545 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
