# Documentation: `test/inductor/test_cpu_repro.py`

## File Metadata

- **Path**: `test/inductor/test_cpu_repro.py`
- **Size**: 204,130 bytes (199.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: cpu inductor"]
import contextlib
import copy
import functools
import itertools
import math
import os
import platform
import sys
import unittest
from collections.abc import Callable
from unittest.mock import patch

import torch
from torch import nn
from torch._C import FileCheck
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config, cpu_vec_isa, metrics, test_operators
from torch._inductor.codegen.cpp import CppOverrides, CppVecOverrides
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
from torch._inductor.exc import InductorError
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import timed
from torch._prims_common import is_float_dtype
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    parametrize,
    skipIfRocm,
    slowTest,
    TEST_MKL,
    xfailIfS390X,
)
from torch.utils._python_dispatch import TorchDispatchMode


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


vec_dtypes = test_torchinductor.vec_dtypes
_lowp_fp_dtypes = (
    torch.bfloat16,
    torch.float16,
)
run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code
TestCase = test_torchinductor.TestCase
aten = torch.ops.aten
check_model = test_torchinductor.check_model

requires_vectorization = unittest.skipUnless(
    cpu_vec_isa.valid_vec_isa_list() and os.getenv("ATEN_CPU_CAPABILITY") != "default",
    "Does not support vectorization",
)


def _can_check_vec_metrics():
    return (
        cpu_vec_isa.valid_vec_isa_list()
        and os.getenv("ATEN_CPU_CAPABILITY") != "default"
        and config.cpp.simdlen != 1
    )


def check_metrics_vec_kernel_count(num_expected_vec_kernels):
    if _can_check_vec_metrics():
        assert metrics.generated_cpp_vec_kernel_count == num_expected_vec_kernels


def simd_lengths_to_test():
    """Returns a minimal list of simd lengths to cover common cases"""
    simdlens = [None, 1]
    valid_isa_list = cpu_vec_isa.valid_vec_isa_list()
    if valid_isa_list:
        simdlens.append(valid_isa_list[0].bit_width())
    return simdlens


@contextlib.contextmanager
def set_num_threads(num_threads):
    orig_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    yield
    torch.set_num_threads(orig_num_threads)


class LstmModule(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        bidirectional=False,
        batch_first=False,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

    def forward(self, x, h=None):
        x, h = self.lstm(x, h)
        return x, h


@instantiate_parametrized_tests
class CPUReproTests(TestCase):
    common = check_model

    def test_torch_linalg_qr_tuple_slice(self):
        def fn(x):
            return torch.linalg.qr(x)[:1]

        x = torch.randn(4, 4)
        compiled = torch.compile(fn, backend="inductor")

        expected = fn(x)
        actual = compiled(x)

        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 1)
        torch.testing.assert_close(actual[0], expected[0])

    @skipIfRocm
    def test_conv_stride_constraints(self):
        for fmt in [torch.contiguous_format, torch.channels_last]:
            # TorchDispatch doesn't work in our cuda invocation for some reason
            m = torch.nn.Conv2d(5, 6, [3, 3])

            def fn(inp, weight):
                return (
                    F.conv2d(
                        inp, weight, None, m.stride, m.padding, m.dilation, m.groups
                    ),
                )

            inp = torch.randn([2, 5, 16, 16])
            inps = [inp, m.weight.to(memory_format=fmt)]
            fn_fx = make_fx(fn)(*inps)
            fn_compiled = compile_fx_inner(fn_fx, inps)
            test_self = self
            conv_seen = False

            class RecordFunctions(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    kwargs = kwargs if kwargs else {}
                    if func == torch.ops.aten.convolution.default:
                        # For CPU and mkldnn enable, we always using channels last
                        nonlocal fmt
                        if (
                            torch.backends.mkldnn.enabled
                            and torch.backends.mkldnn.is_available()
                        ):
                            fmt = torch.channels_last
                        test_self.assertTrue(args[0].is_contiguous(memory_format=fmt))
                        test_self.assertTrue(args[1].is_contiguous(memory_format=fmt))
                        nonlocal conv_seen
                        conv_seen = True

                    return func(*args, **kwargs)

            with RecordFunctions():
                fn_compiled(inps)

            self.assertTrue(conv_seen)

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_bn_mixed_dtype(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3,
                    16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    dtype=torch.bfloat16,
                )
                self.bn = torch.nn.BatchNorm2d(
                    16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
                )

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        v = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)
        mod = Model().eval()
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    def test_complex_cholesky_mh_view_fallback(self):
        torch.manual_seed(0)

        n = 8

        def fn(inp: torch.Tensor):
            I0 = torch.eye(n, dtype=inp.dtype, device=inp.device)
            I = I0.unsqueeze(0).expand(inp.shape[0], n, n).contiguous()
            hermitian = I + 0.5 * (inp @ inp.mH)
            chol = torch.linalg.cholesky(hermitian, upper=True)
            return chol.abs().sum()

        base = torch.randn(4, n, n, dtype=torch.complex64)

        def run(compiled_fn):
            inp = base.clone().detach().requires_grad_(True)
            loss = compiled_fn(inp)
            loss.backward()
            return loss.detach(), inp.grad.detach()

        expected_loss, expected_grad = run(fn)

        compiled = torch.compile(fn, backend="inductor")
        actual_loss, actual_grad = run(compiled)

        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_nn_fold(self):
        # Fix https://github.com/pytorch/pytorch/issues/147848

        class Model(torch.nn.Module):
            def __init__(self, output_size, kernel_size, stride) -> None:
                super().__init__()
                self.fold = torch.nn.Fold(
                    output_size=output_size, kernel_size=kernel_size, stride=stride
                )

            def forward(self, x):
                x = self.fold(x)
                return x

        output_sizes = [(64, 64), (64, 64)]
        kernel_sizes = [(32, 32), (32, 32)]
        strides = [(1, 1), (2, 2)]
        input_sizes = [(1, 32 * 32, 1089), (1, 64 * 64, 289)]

        for idx in range(len(output_sizes)):
            output_size = output_sizes[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            input_size = input_sizes[idx]

            for num_threads in [1, None]:
                torch._dynamo.reset()
                metrics.reset()
                v = torch.randn(*input_size)
                mod = Model(output_size, kernel_size, stride).eval()
                with (
                    contextlib.nullcontext()
                    if (num_threads != 1)
                    else set_num_threads(1)
                ):
                    with torch.no_grad():
                        self.common(
                            mod,
                            (v,),
                        )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_packed(self):
        options = itertools.product([[3, 56, 56]], [True, False], [0, (0,)])
        for x_shape, mode_train, padding in options:
            mod = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, 3, padding=padding)
            ).train(mode=mode_train)
            v = torch.randn(x_shape, dtype=torch.float32)

            with torch.no_grad():
                self.common(
                    mod,
                    (v,),
                )

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_autocast(self):
        v = torch.randn(1, 3, 28, 18, dtype=torch.float32)
        mod = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 3)).eval()
        with torch.no_grad(), torch.cpu.amp.autocast():
            self.common(
                mod,
                (v,),
            )

    def test_conv1d_strided_weight_torch_compile(self):
        def fn(x, w):
            wt = w.transpose(2, 1)
            y = F.conv1d(x, wt)
            return y.clone()

        x_eager = torch.randn(2, 3, 5, requires_grad=True)
        w_eager = torch.randn(4, 2, 3, requires_grad=True)

        out_eager = fn(x_eager, w_eager)
        grad = torch.randn_like(out_eager)
        out_eager_val = out_eager.detach()
        out_eager.backward(grad)
        grad_x_eager = x_eager.grad.detach().clone()
        grad_w_eager = w_eager.grad.detach().clone()

        x_comp = x_eager.detach().requires_grad_(True)
        w_comp = w_eager.detach().requires_grad_(True)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        out_comp = compiled(x_comp, w_comp)
        out_comp_val = out_comp.detach()
        out_comp.backward(grad)

        torch.testing.assert_close(out_comp_val, out_eager_val)
        torch.testing.assert_close(x_comp.grad, grad_x_eager)
        torch.testing.assert_close(w_comp.grad, grad_w_eager)

    @config.patch(freezing=True)
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @patch("torch.cuda.is_available", lambda: False)
    def test_mkl_linear(self):
        dtypes = [torch.float32]
        options = itertools.product([[2, 3, 10]], [2], [True, False], dtypes)
        for input_shape, out_dim, bias, dtype in options:
            mod = torch.nn.Sequential(
                torch.nn.Linear(input_shape[-1], out_dim, bias=bias)
            ).eval()

            v = torch.randn(input_shape)
            with torch.no_grad():
                self.common(
                    mod.to(dtype),
                    (v.to(dtype),),
                )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_unsupported_conv_transpose(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    3, 6, 3, stride=1, padding=1, output_padding=1
                )

            def forward(self, input_tensor):
                x = self.conv_transpose(input_tensor)
                output = torch.tanh(x)
                return output

        input = torch.randn(1, 3, 28, 28)
        m = Model().eval()

        with torch.no_grad():
            compiled_m = torch.compile(m)
            # The cpp_wrapper C-shim can't utilize the Python error API, so error
            # messages are printed to stderr directly, and the intercepted RuntimeError
            # is significantly less verbose.
            msg = (
                r"aoti_torch_cpu_convolution\(.*\) API call failed"
                if config.cpp_wrapper
                else "output padding must be smaller than either stride or dilation"
            )
            with self.assertRaisesRegex(RuntimeError, msg):
                compiled_m(input)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_used_from_multiple_places(self):
        class M(torch.nn.Module):
            def __init__(self, conv_in_channel, conv_out_channel) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(conv_in_channel, conv_out_channel, (3, 3))

            def forward(self, x):
                res = self.conv(x)
                res = F.relu(res)
                res = self.conv(res)
                return res

        with torch.no_grad():
            mod = M(3, 3).eval()
            x = torch.randn(1, 3, 224, 224)
            self.common(
                mod,
                (x,),
            )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_linear_used_from_multiple_places(self):
        class M(torch.nn.Module):
            def __init__(self, in_channel, out_channel) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(in_channel, out_channel)

            def forward(self, x):
                res = self.linear(x)
                res = F.relu(res)
                res = self.linear(res)
                return res

        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            with torch.no_grad():
                m = M(224, 224).to(dtype).eval()
                m_opt = torch.compile(m)
                x = torch.randn(224, 224, dtype=dtype)
                m_opt(x)
                self.assertEqual(m(x), m_opt(x))

    @config.patch(implicit_fallbacks=True)
    def test_multihead_attention_cpu(self):
        def fn(
            q,
            k,
            v,
            embed_dim,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            mask,
            need_weights,
        ):
            return torch._native_multi_head_attention(
                q,
                k,
                v,
                embed_dim,
                num_heads,
                qkv_weight,
                qkv_bias,
                proj_weight,
                proj_bias,
                mask,
                need_weights,
            )

        B = 1
        T = 3
        embed_dim = 6
        num_heads = 2
        q = torch.randn([B, T, embed_dim])
        k = torch.randn([B, T, embed_dim])
        v = torch.randn([B, T, embed_dim])
        qkv_weight = torch.randn([3 * embed_dim, embed_dim])
        qkv_bias = torch.randn([3 * embed_dim])
        proj_weight = torch.randn([3 * embed_dim, embed_dim])
        proj_bias = torch.randn([3 * embed_dim])
        mask = None
        need_weights = False

        inps = [
            q,
            k,
            v,
            embed_dim,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            mask,
            need_weights,
        ]
        self.common(fn, inps)

    @config.patch(freezing=True)
    def test_module_buffer_mutation(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Buffer(torch.rand((3, 10)))

            def forward(self, x):
                lx = [x, x.clone(), x.clone()]
                y = []
                for i in range(3):
                    y.append(lx[i] + self.foo[i])
                return torch.cat(y, 1)

        with torch.no_grad():
            example_inputs = (torch.rand(1, 10),)
            self.common(Model(), example_inputs)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_linear_packed(self):
        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        options = itertools.product(
            [[2, 3, 10], [2, 10], [10], [2, 0]], [3, 0], [True, False], dtypes
        )
        for input_shape, out_dim, bias, dtype in options:
            mod = torch.nn.Sequential(
                torch.nn.Linear(input_shape[-1], out_dim, bias=bias)
            ).eval()

            v = torch.randn(input_shape)
            with torch.no_grad():
                self.common(
                    mod.to(dtype),
                    (v.to(dtype),),
                )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_transpose2d_packed_cpu(self):
        options = itertools.product([[1, 3, 28, 28], [3, 28, 28]], [0, (0,)])
        for x_shape, padding in options:
            mod = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(3, 64, 3, 3, padding=padding)
            ).eval()
            v = torch.randn(x_shape, dtype=torch.float32)
            with torch.no_grad():
                self.common(
                    mod,
                    (v,),
                )

    @torch._dynamo.config.patch(
        {"dynamic_shapes": True, "assume_static_by_default": False}
    )
    def test_full_boolean_dynamic_shape(self):
        def fn(n):
            x = torch.full((1024,), n >= 1024)
            return x, x + 1

        self.common(fn, (1024,))
        self.common(fn, (1023,))

    @config.patch(freezing=True)
    @unittest.skipIf(not torch._C._has_mkldnn, "MKLDNN is not enabled")
    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_conv_in_channel_1_dynamic_shapes(self):
        class M(torch.nn.Module):
            def __init__(self, in_channel, out_channel) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channel, out_channel, 3)

            def forward(self, x):
                res = self.conv(x)
                res = F.relu(res)
                return res

        # test the case where the channels dim of the input is 1
        # Reproducer from the maml_omniglot model in Torchbench
        in_channel = 1
        out_channel = 3
        amp_enabled_configs = [False]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            # When amp is enabled here, the input to Conv is a FlexibleLayout.
            # While it's disabled, the input is a FixedLayout.
            amp_enabled_configs.append(True)
        for amp_enabled in amp_enabled_configs:
            mod = M(in_channel, out_channel).eval()
            v = torch.randn(5, in_channel, 15, 15)
            with torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
                self.common(
                    mod,
                    (v,),
                )

    @unittest.skipIf(not torch._C._has_mkldnn, "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    @torch._dynamo.config.patch(allow_rnn=True)
    @config.patch(freezing=True)
    def _test_lstm_packed(
        self,
        unbatched,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        bias,
        empty_state,
        batch_first,
        batch_size,
        seq_len,
        change_input_sizes=False,
    ):
        from torch._dynamo.utils import counters

        dtypes = [torch.float]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            counters.clear()
            num_directions = 2 if bidirectional else 1

            seq_len_var = seq_len + 3
            if unbatched:
                v = torch.randn(seq_len, input_size)
                v_var = torch.randn(seq_len_var, input_size)
                h = torch.randn(num_layers * num_directions, hidden_size)
                c = torch.randn(num_layers * num_directions, hidden_size)
            else:
                if batch_first:
                    v = torch.randn(batch_size, seq_len, input_size)
                    v_var = torch.randn(batch_size, seq_len_var, input_size)
                else:
                    v = torch.randn(seq_len, batch_size, input_size)
                    v_var = torch.randn(seq_len_var, batch_size, input_size)
                h = torch.randn(num_layers * num_directions, batch_size, hidden_size)
                c = torch.randn(num_layers * num_directions, batch_size, hidden_size)

            mod = LstmModule(
                input_size,
                hidden_size,
                num_layers,
                bias,
                bidirectional,
                batch_first,
            ).eval()
            maybe_autocast = (
                torch.cpu.amp.autocast()
                if dtype == torch.bfloat16
                else contextlib.nullcontext()
            )

            with torch.no_grad(), maybe_autocast:
                inps = [v]
                if not empty_state:
                    inps.append((h, c))

                fn_opt = torch.compile(mod, backend="inductor")
                _, code = run_and_get_cpp_code(fn_opt, *inps)

                # Check that _flat_weights are not functional_tensor, otherwise
                # deepcopy will fail during recompilation.
                fn_opt_copy = copy.deepcopy(fn_opt)
                _flat_weights = fn_opt_copy.lstm._flat_weights
                for _flat_weight in _flat_weights:
                    self.assertFalse(torch._is_functional_tensor(_flat_weight))

                self.assertTrue("aten.mkldnn_rnn_layer" in code)
                self.assertEqual(fn_opt(*inps), mod(*inps))
                self.assertEqual(
                    counters["inductor"]["pattern_matcher_count"],
                    num_layers * num_directions
                    + 2,  # num of mkldnn_rnn_layer call + 2 view call on the concatenated hy, cy.
                )

                # Change input sizes
                if change_input_sizes:
                    inps_var = [v_var]
                    self.assertEqual(fn_opt(*inps_var), mod(*inps_var))

    @parametrize(
        "unbatched, input_size, hidden_size, num_layers, bidirectional, bias, empty_state, batch_first, batch_size, seq_len",
        itertools.product(
            *[
                [True, False],
                [1, 7],
                [7],
                [1, 7],
                [False, True],
                [False, True],
                [False, True],
                [True, False],
                [1, 7],
                [1, 7],
            ]
        ),
    )
    def test_lstm_packed(
        self,
        unbatched,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        bias,
        empty_state,
        batch_first,
        batch_size,
        seq_len,
    ):
        self._test_lstm_packed(
            unbatched,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            bias,
            empty_state,
            batch_first,
            batch_size,
            seq_len,
        )

    _test_lstm_packed_change_input_sizes_cpu_params = list(
        itertools.product(
            *[
                [False],
                [2],
                [5],
                [3],
                [True],
                [True],
                [False],
                [False],
                [2],
                [3],
            ]
        )
    )

    @parametrize(
        "unbatched, input_size, hidden_size, num_layers, bidirectional, bias, empty_state, batch_first, batch_size, seq_len",
        _test_lstm_packed_change_input_sizes_cpu_params,
    )
    def test_lstm_packed_change_input_sizes_cpu(
        self,
        unbatched,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        bias,
        empty_state,
        batch_first,
        batch_size,
        seq_len,
    ):
        self._test_lstm_packed(
            unbatched,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            bias,
            empty_state,
            batch_first,
            batch_size,
            seq_len,
            change_input_sizes=True,
        )

    def test_set_source_Tensor(self):
        class MaskedConv2d(torch.nn.Conv2d):
            def __init__(
                self,
                *,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                padding: int = 0,
            ) -> None:
                super().__init__(
                    in_channels, out_channels, kernel_size, padding=padding
                )
                mask = torch.zeros_like(self.weight)

                mask[:, :, : kernel_size // 2, :] = 1
                mask[:, :, kernel_size // 2, : kernel_size // 2] = 1
                self.register_buffer("mask", mask)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    self.weight.data *= self.mask
                return super().forward(x)

        class M(torch.nn.Module):
            def __init__(
                self, num_channels: int, num_colors: int, H: int, W: int
            ) -> None:
                super().__init__()
                self.num_channels = num_channels
                self.num_colors = num_colors
                self.H = H
                self.W = W
                kernel_size = 7
                padding = (kernel_size - 1) // 2
                # 1 7x7 Mask
                layers = [
                    MaskedConv2d(
                        in_channels=self.num_channels,
                        out_channels=64,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                ]
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.permute(0, 3, 1, 2)
                return self.model(x)

        model = M(H=32, W=32, num_channels=4, num_colors=2)
        fn_opt = torch.compile(model, backend="inductor")
        v = (torch.rand(10, 32, 32, 4) > 0.5).to(torch.float32)
        inp = v.clone()
        result, code = run_and_get_cpp_code(fn_opt, inp)
        self.assertIn(
            "aoti_torch_cpu_set__source_Tensor"
            if config.cpp_wrapper
            else "aten.set_.source_Tensor",
            code,
        )
        expected = model(inp)
        self.assertEqual(expected, result)

        # test cpp_wrapper_build_separate
        with config.patch(cpp_wrapper=True, cpp_wrapper_build_separate=True):
            result, code = run_and_get_cpp_code(fn_opt, inp)
            self.assertIn("kernel_src", code)
            self.assertEqual(expected, result)

        with config.patch(cpp_wrapper=True, cpp_wrapper_build_separate=False):
            result, code = run_and_get_cpp_code(fn_opt, inp)
            self.assertNotIn("kernel_src", code)
            self.assertEqual(expected, result)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    @torch._dynamo.config.patch(allow_rnn=True)
    def test_pack_padded_sequence_lstm(self):
        embedding_dim = 12
        hidden_dim = 10
        batch_size = 24
        num_layers = 1
        bidirectional = True
        num_direc = 2
        max_lens = 96

        sent = torch.randn(batch_size, max_lens, embedding_dim)
        hid_0 = torch.rand(num_layers * num_direc, batch_size, hidden_dim)
        hid_1 = torch.randn(num_layers * num_direc, batch_size, hidden_dim)

        sent_lens = torch.Tensor(
            [1, 2, 3, 4, 5, 1, 3, 2, 96, 5, 3, 1, 1, 2, 1, 2, 3, 6, 1, 2, 4, 6, 2, 1]
        )

        assert sent_lens.shape[0] == batch_size
        assert sent_lens.max().item() == max_lens

        hidden_0 = hid_0.clone().requires_grad_(False)
        hidden_1 = hid_1.clone().requires_grad_(False)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(
            sent, sent_lens, batch_first=True, enforce_sorted=False
        )

        mod = LstmModule(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bias=True,
            bidirectional=bidirectional,
            batch_first=True,
        ).eval()

        with torch.no_grad():
            inps = [embeds, (hidden_0, hidden_1)]
            fn_opt = torch.compile(mod, backend="inductor")
            _, code = run_and_get_cpp_code(fn_opt, *inps)
            # This case is unsupported
            self.assertFalse("torch.ops.mkldnn._lstm" in code)
            self.assertEqual(fn_opt(*inps), mod(*inps))

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_transpose2d_has_output_size_input(self):
        # https://github.com/pytorch/pytorch/issues/100344.
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x):
                return self.conv_transpose(x, output_size=(10, 10))

        mod = M().eval()
        v = torch.randn(1, 3, 10, 10, dtype=torch.float32)
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    def test_pad_with_nan_value(self):
        # https://github.com/pytorch/pytorch/issues/100988.
        class Model(torch.nn.Module):
            def forward(self, x):
                x = F.pad(x, (1, 1, 1, 1), value=float("nan"))
                return x

        mod = Model().eval()
        v = torch.randn(1, 3, 10, 10, dtype=torch.float32)
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    def test_masked_fill_with_inf_or_nan_value(self):
        def fn(value, mask):
            y1 = torch.masked_fill(value, mask, float("inf"))
            y2 = torch.masked_fill(value, mask, float("-inf"))
            y3 = torch.masked_fill(value, mask, float("nan"))
            return y1, y2, y3

        value = torch.randn((2, 17))
        mask = torch.randint(0, 1, size=(2, 17), dtype=torch.uint8).to(torch.bool)
        with torch.no_grad():
            self.common(
                fn,
                (value, mask),
            )

    def test_relu_with_inf_value(self):
        # https://github.com/pytorch/pytorch/issues/117544.

        def fn(out):
            out = torch.sinh(input=out)
            out = torch.relu(input=out)
            return out

        x = torch.Tensor([-572373.5000, 755109.1250, 330995.5625])
        with torch.no_grad():
            self.common(
                fn,
                (x,),
            )

    def test_acosh_with_negative_large_input(self):
        # https://github.com/pytorch/pytorch/issues/118267.

        def fn(input):
            out = torch.acosh(input)
            return out

        x = torch.Tensor(
            [
                [
                    -8493.9854,
                    431654.1250,
                    71741.5859,
                    608234.5000,
                    -103814.7500,
                    -699397.0000,
                    -910685.8125,
                    -832737.1875,
                    875343.5000,
                ]
            ]
        ).repeat(3, 9)

        for dtype in [torch.float32, torch.bfloat16, torch.double]:
            with torch.no_grad():
                torch._dynamo.reset()
                metrics.reset()
                _x = x.to(dtype)
                self.common(
                    fn,
                    (_x,),
                )

    @requires_vectorization
    def test_asinh_with_corner_inputs(self):
        # https://github.com/pytorch/pytorch/issues/142345

        def fn(input):
            out = torch.asinh(input)
            return out

        x = torch.tensor([0, 0, 0, -10000.1]).repeat(3, 4)

        bit_widths = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()]
        for dtype in [torch.float32, torch.bfloat16, torch.float16, torch.double]:
            for simdlen in bit_widths:
                with torch.no_grad(), config.patch({"cpp.simdlen": simdlen}):
                    torch._dynamo.reset()
                    metrics.reset()
                    _x = x.to(dtype)
                    self.common(fn, (_x,))
                    check_metrics_vec_kernel_count(1)

    @config.patch(fallback_random=True)
    def test_require_stride_order_non_owning(self):
        def test_concat_with_conv():
            x1 = torch.randn(2, 3, 4, 4).to(memory_format=torch.channels_last)
            x2 = torch.randn(2, 5, 4, 4).to(memory_format=torch.channels_last)

            # First do the concatenation
            cat_result = torch.cat([x1, x2], dim=1)

            # Then use x1 (which was an input to the cat) in a conv
            conv_weight = torch.randn(4, 3, 3, 3).to(memory_format=torch.channels_last)
            x1_conv = torch.nn.functional.conv2d(x1, conv_weight, padding=1)

            return cat_result, x1_conv

        torch.manual_seed(1)
        f_c = torch.compile(test_concat_with_conv)
        out_result, code = run_and_get_cpp_code(f_c)

        torch.manual_seed(1)
        self.assertEqual(out_result, test_concat_with_conv())

        # both inputs to conv should be channels last
        if config.cpp_wrapper:
            FileCheck().check("{2L, 3L, 4L, 4L}").check("{128L, 1L, 32L, 8L}").check(
                "{4L, 3L, 3L, 3L}"
            ).check("{27L, 1L, 9L, 3L}").check("aoti_torch_empty_strided").run(code)
        else:
            FileCheck().check("(2, 3, 4, 4), (128, 1, 32, 8)").check(
                "empty_strided_cpu((4, 3, 3, 3), (27, 1, 9, 3)"
            ).run(code)

    @config.patch(implicit_fallbacks=True)
    def test_repeat_interleave(self):
        def fn(y):
            return torch.repeat_interleave(y, 2, output_size=8)

        a = torch.tensor([[1, 2], [3, 4]])
        self.common(
            fn,
            (a,),
        )

    def test_inplace_squeeze_needed(self):
        mod = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.LayerNorm(10),
            torch.nn.ReLU(),
        ).eval()

        def fn(x):
            return mod(x)

        v = torch.randn(10)
        # TODO: OMP parallel reduction order is not deterministic.
        # Hence, the accuracy might vary up and down. For short term,
        # we increase the tolerance and will fix it later by using
        # aten parallel.
        self.common(fn, (v,), atol=5e-1, rtol=5e-1)

    def test_parallel_reduction_vectorization(self):
        # Fix issue: https://github.com/pytorch/pytorch/issues/151523
        class Model(torch.nn.Module):
            def __init__(self, enable_masked_tail_vec):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=(1, 7),
                    stride=(2, 1),
                    padding=0,
                )
                self.enable_masked_tail_vec = enable_masked_tail_vec

            def forward(self, x, weight):
                x = self.conv(x)
                if not self.enable_masked_tail_vec:
                    x = F.hardshrink(x, lambd=0)
                x = x.view(x.size(0), -1)
                x = torch.mv(weight, x[0])
                return x

        for enable_masked_tail_vec in [True, False]:
            mod = Model(enable_masked_tail_vec).eval()
            x = torch.randn(2, 3, 127, 255)
            weight = torch.randn(10, 254976)
            # Use same criterion as test_inplace_squeeze_needed
            # for parallel reduction.
            self.common(mod, (x, weight), atol=5e-1, rtol=5e-1)

    def test_cat_mul(self):
        # https://github.com/pytorch/pytorch/issues/93365
        def fn(p0, p1):
            y1 = torch.cat([p0, p1], dim=0)
            y2 = torch.mul(y1, y1)
            return y1, y2

        p0 = torch.randn(3, 4)
        p1 = torch.randn(3, 4)
        self.common(fn, (p0, p1))

    def test_pow_cos(self):
        # https://github.com/pytorch/pytorch/issues/98149
        def fn(x):
            t = x.pow(5)
            return torch.cos(t)

        x = torch.tensor([4], dtype=torch.uint8)
        self.common(fn, (x,))

    def test_reduce_with_masked(self):
        # https://github.com/pytorch/pytorch/issues/96484
        def fn(a, b):
            a = torch.nn.functional.pad(a, (0, -1))
            c = a + b
            return c.min(0).values

        a = torch.randn([2])
        b = torch.randn([2])
        self.common(fn, (a, b))

    def test_scalar_sign_with_min(self):
        # https://github.com/pytorch/pytorch/issues/101340
        def fn(a):
            t1 = torch.tanh(a)
            t2 = torch.sign(t1)
            return torch.min(t1, t2)

        a = torch.randn(1, 3)
        self.common(fn, (a,))

    def test_tanh_atan2(self):
        # https://github.com/pytorch/pytorch/issues/148241
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shrink = nn.Tanhshrink()

            def forward(self, x):
                x = self.shrink(x)
                x = torch.atan2(x, x)
                return x

        x = torch.randn(1, 3, 64, 64)
        self.common(Model(), (x,))

    @unittest.skipIf(
        os.getenv("ATEN_CPU_CAPABILITY") == "default",
        "Failing in periodic nogpu_NO_AVX2 after added in #152542",
    )
    @config.patch("cpp.use_decompose_tanh", "1")
    def test_tanh_atan2_use_decompose_tanh(self):
        # https://github.com/pytorch/pytorch/issues/148241
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shrink = nn.Tanhshrink()

            def forward(self, x):
                x = self.shrink(x)
                x = torch.atan2(x, x)
                return x

        x = torch.randn(1, 3, 64, 64)
        with self.assertRaises(AssertionError):
            self.common(Model(), (x,))

    def test_index_propagation_issue_102065(self):
        def fn(x):
            x = torch.arange(x.numel())
            return (x.unsqueeze(0) - x.unsqueeze(1)) ** 2

        self.common(
            fn,
            (torch.randn(8),),
        )

    def test_low_fp_index_expr_issue_147279(self):
        # https://github.com/pytorch/pytorch/issues/147279
        def fn(start, end, dtype, dim):
            return torch.sum(
                torch.arange(start=start, end=end, dtype=dtype),
                dim=dim,
            )

        self.common(
            fn,
            (300, 400, torch.float16, (0,)),
        )

    def test_index_put(self):
        # https://github.com/pytorch/pytorch/issues/138908
        def fn(x, y):
            x = x + 10
            y[x] += y[x]

        x = torch.randint(-10, -9, (1, 2), dtype=torch.int64)
        y = torch.randn((2, 32), dtype=torch.float32)
        x_clone = x.clone()
        y_clone = y.clone()
        with torch.no_grad():
            fn(x, y)
            torch.compile(fn)(x_clone, y_clone)
            self.assertEqual(y, y_clone, atol=1e-3, rtol=1e-3)

    def test_index_put2(self):
        # https://github.com/pytorch/pytorch/issues/138908
        def fn(y, index0, index1):
            y[index1] += y[index0]

        y = torch.randn((2, 32), dtype=torch.float32)
        index0 = torch.tensor([[0, 1]])
        index1 = torch.tensor([[1, 0]])
        y_clone = y.clone()
        index0_clone = index0.clone()
        index1_clone = index1.clone()
        with torch.no_grad():
            fn(y, index0, index1)
            torch.compile(fn)(y_clone, index0_clone, index1_clone)
            self.assertEqual(y, y_clone, atol=1e-3, rtol=1e-3)

    def test_index_add(self):
        # https://github.com/pytorch/pytorch/issues/138908
        def fn(x, y, scale_y, index):
            values = x[index] + y * scale_y
            out = x.index_add_(dim=0, source=values, index=index)
            return out

        inp = (
            torch.randn(10, 10),
            torch.randn(5, 10),
            torch.randn(10),
            torch.randperm(10, device="cpu")[:5].to(torch.int32),
        )
        inp_clones = []
        for i in range(3):
            inp_clones.append(
                [
                    inp[0].clone(),
                    inp[1].clone(),
                    inp[2].clone(),
                    inp[3].clone()
                    if i == 0
                    else torch.zeros(10, device="cpu")[:5].to(torch.int32),
                ]
            )
        inp_clone, inp_clone2, inp_clone3 = inp_clones
        with torch.no_grad():
            cfn = torch.compile(fn)
            ref = fn(*inp)
            res = cfn(*inp_clone)
            self.assertEqual(ref, res, atol=1e-3, rtol=1e-3)
            ref = fn(*inp_clone2)
            res = cfn(*inp_clone3)
            self.assertEqual(ref, res, atol=1e-3, rtol=1e-3)

    def test_ModularIndexing_range_issue_103133(self):
        def fn(q, k):
            einsum = torch.einsum("bcxd,bcyd->bcxy", (q, k))
            constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                einsum, [0, 0, 0, 1], 0.0
            )
            view = torch.ops.aten.view.default(constant_pad_nd, [12, 1, 512, 513])
            y = view.new_zeros((12, 2, 256, 513))
            y[:, :-1, :, 256:] = view[:, :, :256, :257]
            return y

        self.common(
            fn,
            (
                torch.empty_strided((12, 1, 512, 64), (64, 196608, 768, 1)),
                torch.empty_strided((12, 1, 512, 64), (64, 196608, 768, 1)),
            ),
        )

    @patch("torch.cuda.is_available", lambda: False)
    def test_max_reduction_lowp_fp(self):
        def fn(x):
            return torch.ops.aten.max(x, 1, keepdim=True)[0].float()

        for dtype in _lowp_fp_dtypes:
            self.common(
                fn,
                (torch.randn(1, 32, 4, 4).to(dtype),),
            )

    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_transpose_lowp_fp(self):
        for dtype in _lowp_fp_dtypes:

            def fn(x):
                return x.to(memory_format=torch.channels_last).to(dtype)

            self.common(
                fn,
                (torch.randn(2, 3, 4, 4),),
            )

    def test_load_inf_bf16(self):
        def fn1(x):
            return torch.where(x > 0, x, math.inf)

        def fn2(x):
            return torch.where(x > 0, x, -math.inf)

        for fn in [fn1, fn2]:
            self.common(
                fn,
                (torch.randn(1, 3, 16, 16),),
            )

    @patch("torch.cuda.is_available", lambda: False)
    def test_fp32_load_with_to_lowp_fp(self):
        # From llama model.
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cache_k = torch.zeros(8, 4, 2, 2)

            def forward(self, x, xk):
                bsz, seqlen, _ = x.shape
                self.cache_k = self.cache_k.to(x)
                self.cache_k[:bsz, 1 : 1 + seqlen] = xk
                return self.cache_k

        for dtype in _lowp_fp_dtypes:
            ref_model = Model().eval()
            opt_model = torch.compile()(Model().eval())
            x = torch.randn(4, 2, 2).to(dtype)
            xk = torch.randn(4, 2, 2, 2).to(dtype)
            self.assertEqual(opt_model(x, xk), ref_model(x, xk))

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_sigmoid_with_reduction(self):
        def fn(x):
            x = torch.ops.aten.sigmoid.default(x)
            return torch.ops.aten.mean.dim(x, [-1, -2], True)

        x = torch.randn((1, 8, 8, 8))
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))

    def test_slice_scatter_default_end_value(self):
        # From HF AllenaiLongformerBase.
        def fn(query, key, window_overlap):
            batch_size, seq_len, num_heads, head_dim = query.size()
            assert seq_len % (window_overlap * 2) == 0, (
                f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
            )

            chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
            diagonal_chunked_attention_scores = key
            diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
                (
                    batch_size * num_heads,
                    chunks_count + 1,
                    window_overlap,
                    window_overlap * 2 + 1,
                )
            )
            diagonal_attention_scores[:, :3, :, window_overlap:] = (
                diagonal_chunked_attention_scores[
                    :, :, :window_overlap, : window_overlap + 1
                ]
            )
            return diagonal_attention_scores

        self.common(
            fn,
            (
                torch.randn(1, 1024, 12, 64),
                torch.randn(12, 3, 512, 513),
                256,
            ),
        )

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_to_uint8_rounding_method(self):
        def fn(x):
            return x.to(torch.uint8)

        numerical_testsuit = [4.4, 4.5, 4.6, 5.5]
        for numerical_number in numerical_testsuit:
            x = torch.ones(17) * numerical_number
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def _test_decomposed_dequant_relu_quant_helper(self, dtype):
        def fn(
            x, scale, zero_point, use_dequant, use_quant, quant_min, quant_max, dtype
        ):
            # For quantized_decomposed.dequantize_per_tensor
            # Refer to torch/ao/quantization/fx/_decomposed.py
            if use_dequant:
                x = (x.to(torch.float32) - zero_point) * scale

            x = torch.relu(x)

            # For quantized_decomposed.quantize_per_tensor
            # Refer to torch/ao/quantization/fx/_decomposed.py
            if use_quant:
                inv_scale = 1.0 / scale
                x = torch.clamp(
                    torch.round(x * inv_scale) + zero_point, quant_min, quant_max
                ).to(dtype)
            return x

        assert dtype in [torch.uint8, torch.int8]
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        use_dequant_list = [False, True]
        use_quant_list = [False, True]
        for use_dequant, use_quant in itertools.product(
            use_dequant_list, use_quant_list
        ):
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            )
            if use_dequant:
                x = x.to(dtype)
            zero_point = 100
            scale = 0.01
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(
                    fn,
                    (
                        x,
                        scale,
                        zero_point,
                        use_dequant,
                        use_quant,
                        quant_min,
                        quant_max,
                        dtype,
                    ),
                )
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_decomposed_dequant_relu_quant_uint8(self):
        self._test_decomposed_dequant_relu_quant_helper(torch.uint8)

    @requires_vectorization
    def test_decomposed_dequant_relu_quant_int8(self):
        self._test_decomposed_dequant_relu_quant_helper(torch.int8)

    def _test_dequant_quant_lowering_helper(self, dtype, dequant_out_dtype=None):
        def fn(
            x,
            scale,
            zero_point,
            use_dequant,
            use_quant
```



## High-Level Overview


This Python file contains 42 class(es) and 506 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LstmModule`, `CPUReproTests`, `RecordFunctions`, `Model`, `Model`, `Model`, `M`, `M`, `Model`, `M`, `MaskedConv2d`, `M`, `M`, `Model`, `Model`, `Model`, `Model`, `Model`, `Mod`, `Model`

**Functions defined**: `_can_check_vec_metrics`, `check_metrics_vec_kernel_count`, `simd_lengths_to_test`, `set_num_threads`, `__init__`, `forward`, `test_torch_linalg_qr_tuple_slice`, `fn`, `test_conv_stride_constraints`, `fn`, `__torch_dispatch__`, `test_conv2d_bn_mixed_dtype`, `__init__`, `forward`, `test_complex_cholesky_mh_view_fallback`, `fn`, `run`, `test_nn_fold`, `__init__`, `forward`

**Key imports**: contextlib, copy, functools, itertools, math, os, platform, sys, unittest, Callable


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `copy`
- `functools`
- `itertools`
- `math`
- `os`
- `platform`
- `sys`
- `unittest`
- `collections.abc`: Callable
- `unittest.mock`: patch
- `torch`
- `torch._C`: FileCheck
- `torch._dynamo.testing`: rand_strided
- `torch._dynamo.utils`: same
- `torch._inductor`: config, cpu_vec_isa, metrics, test_operators
- `torch._inductor.codegen.cpp`: CppOverrides, CppVecOverrides
- `torch._inductor.exc`: InductorError
- `torch._inductor.graph`: GraphLowering
- `torch._inductor.utils`: timed
- `torch._prims_common`: is_float_dtype
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.nn`: functional as F
- `torch.utils._python_dispatch`: TorchDispatchMode
- `.`: test_torchinductor
- `test_torchinductor  `
- `torch.profiler`: profile
- `torch.nn.attention`: sdpa_kernel, SDPBackend


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

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_cpu_repro.py
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

- **File Documentation**: `test_cpu_repro.py_docs.md`
- **Keyword Index**: `test_cpu_repro.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
