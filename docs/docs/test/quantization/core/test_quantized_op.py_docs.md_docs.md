# Documentation: `docs/test/quantization/core/test_quantized_op.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/core/test_quantized_op.py_docs.md`
- **Size**: 54,306 bytes (53.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/core/test_quantized_op.py`

## File Metadata

- **Path**: `test/quantization/core/test_quantized_op.py`
- **Size**: 394,619 bytes (385.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841


import copy
import itertools
import operator
import random
import unittest
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

import torch
import torch.jit
import torch.nn.functional as F
import torch.testing._internal.hypothesis_utils as hu

from hypothesis import assume, given, HealthCheck, note, settings, strategies as st
from packaging.version import Version
from torch import _VF
if TYPE_CHECKING:
    from torch._ops import OpOverloadPacket
from torch.nn.modules.utils import _pair, _single

hu.assert_deadline_disabled()

from typing import Optional

import torch.backends.xnnpack
from torch.ao.quantization import PerChannelMinMaxObserver
from torch.testing._internal.common_cuda import (
    SM80OrLater,
    TEST_CUDA,
    TEST_CUDNN,
    TEST_CUDNN_VERSION,
)
from torch.testing._internal.common_quantization import (
    skipIfNoFBGEMM,
    skipIfNoONEDNN,
    skipIfNoQNNPACK,
)
from torch.testing._internal.common_quantized import (
    _calculate_dynamic_qparams,
    _dequantize,
    _quantize,
    _snr,
    override_qengines,
    override_quantized_engine,
    qengine_is_onednn,
    qengine_is_qnnpack,
    supported_qengines,
)
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_FBCODE,
    IS_MACOS,
    IS_PPC,
    IS_SANDCASTLE,
    raise_on_run_directly,
    TestCase,
)
from torch.testing._internal.optests import opcheck

from torch.utils.cpp_extension import ROCM_HOME

np_dtype = {torch.quint8: np.uint8, torch.qint8: np.int8, torch.qint32: np.int32}

TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None

class PointwisePostOp(NamedTuple):
    binary_attr : str = "none"
    alpha : float = 1.0
    unary_attr : str = "none"
    scalars : list = []
    algorithm : str = ""

# Make sure we won't have overflows from vpmaddubsw instruction used in FBGEMM.
# On the current Intel x86 architecture, we need to utilize vpmaddubsw instruction
# for the 8-bit int multiplication. This instruction vertically multiplies each
# unsigned 8-bit integer from a with the corresponding signed 8-bit integer from
# b, producing intermediate signed 16-bit integers. This function modifies the
# weights to eliminate the overflow on the signed 16-bit integers.
def avoid_vpmaddubsw_overflow_linear(
    batch_size, input_channels, output_channels, X, X_min, X_max, W, W_min, W_max
):
    if Version(np.__version__) >= Version("2.1"):
        raise unittest.SkipTest("numpy 2.1 overflow error")
    for i, j in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            if x0 * w0 + x1 * w1 < -(1 << 15):
                w1_adjusted = (-(1 << 15) - float(x0) * w0) / x1
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min
            elif x0 * w0 + x1 * w1 > (1 << 15) - 1:
                w1_adjusted = ((1 << 15) - 1 - float(x0) * w0) / x1
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min

    # Go through the same loop again to double check we don't have any overflow
    for i, j in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            assert -(1 << 15) <= x0 * w0 + x1 * w1 < (1 << 15)


# Reference quantized Linear operator
def qlinear_ref(X_q, X_scale, X_zp, W_q, W_scale, W_zp, b_q, Y_scale, Y_zp, dtype=np.uint8):
    X_q = np.reshape(X_q, (-1, X_q.shape[X_q.ndim - 1]))
    row_offsets_ref = X_q.sum(axis=1).astype(np.int32).reshape((-1, 1))
    col_offsets_ref = W_q.sum(axis=1).astype(np.int32).reshape((1, -1))
    assert X_q.ndim == 2
    batch_size, input_channels = X_q.shape
    Prod_XqWq_ref = (
        np.matmul(X_q.astype(np.int32), W_q.astype(np.int32).T)
        - W_zp * row_offsets_ref
        - X_zp * col_offsets_ref
        + input_channels * X_zp * W_zp
    )
    if b_q is not None:
        Prod_XqWq_ref += b_q
    Y_q_ref = _quantize(Prod_XqWq_ref, Y_scale / (X_scale * W_scale), Y_zp, dtype=dtype)
    return Y_q_ref

"""Computes the output shape given pooling parameters."""
def pool_output_shape(input_size, kernel_size, padding, stride,
                      dilation, ceiling_mode=False):
    if stride is None:
        stride = kernel_size
    output_size = (
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1
         + (stride - 1 if ceiling_mode else 0)) // stride + 1)
    if (ceiling_mode and
            ((output_size - 1) * stride >= input_size + padding)):
        output_size -= 1
    return output_size

"""
Util for creating a random tensor and quantization params when Hypothesis
is undesirable.
"""
def _get_random_tensor_and_q_params(shapes, rand_scale, torch_type):
    X = (torch.rand(*shapes, dtype=torch.float) - 0.5) * rand_scale
    # Calculate reasonable quantization params
    min_val = torch.min(X)
    max_val = torch.max(X)
    if torch_type == torch.qint32:
        X_zero_point = int(torch.randint(-1 * (2 ** 31), 2 ** 31 - 1, (1,)))
        num_bins = 2 ** 32
        X_scale = float(max_val - min_val) / num_bins
    elif torch_type == torch.qint8:
        X_zero_point = int(torch.randint(-128, 127, (1,)))
        num_bins = 2 ** 8
        X_scale = float(max_val - min_val) / num_bins
    else:  # torch.quint8
        X_zero_point = 127
        num_bins = 2 ** 8
        X_scale = float(max_val - min_val) / num_bins
    if X_scale == 0:
        X_scale = 1e-10
    return X, X_scale, X_zero_point

def _quantize_fp8e4m3(t: torch.Tensor, channelwise: bool, scale: Optional[torch.Tensor] = None):
    quant_max = torch.finfo(torch.float8_e4m3fn).max
    eps = torch.Tensor([torch.finfo(torch.float32).eps])
    if channelwise:
        scale = scale or t.reshape(t.shape[0], -1).abs().max(-1)[0] / quant_max
        scale = torch.max(scale, eps)
        scale_reshape = scale.reshape((-1,) + (1,) * (t.dim() - 1))
        qt = t / scale_reshape
    else:
        scale = scale or t.abs().max().reshape([1]) / quant_max
        scale = torch.max(scale, eps) if isinstance(scale, torch.Tensor) else max(scale, eps.item())
        qt = t / scale
    # Clamp to avoid NaN. Convert in two steps to align with fp32 -> fp16 -> fp8
    qt = qt.clamp(-448, 448).half().to(torch.float8_e4m3fn)
    return qt, scale

def _dequantize_fp8e4m3(qt: torch.Tensor, scale: torch.Tensor):
    dqt = qt.float()
    if scale.numel() == 1:
        # per tensor
        dqt = dqt * scale
    else:
        # per channel
        scale_reshape = scale.reshape((-1,) + (1,) * (qt.dim() - 1))
        dqt = dqt * scale_reshape
    return dqt

class TestQuantizedOps(TestCase):

    """Helper function to test quantized activation functions."""
    def _test_activation_function(self, X, fn_name, test_configs):
        r"""
            When writing a unit test for the activation function,
            instead of specifying the test routines only applicable to the activation function itself,
            you utilize the _test_activation_function that provides general testing.
            To utilize the helper function, a test config must be provided.
            A test config is a list that contains metadata about the quantized activation
            functions that will be tested and how the tests need to be set up; it allows simpler and
            more concise unit tests to be written by specifying the configurations needed
            and calling the provided helper function _test_activation_function.
            Inside the list, each config (as a dictionary) represents a suite of tests that assert the
            correctness of various quantization functions.
            You can check out the test_qrelu, test_qrelu6, test_qsigmoid, and test_qhardsigmoid for
            how their test configs are specified.
            Here's a list of the fields that can be included in a test config:
            quantized_fn: a list of the quantized functions to be tested
            reference_fn: the original reference function to be called on the
            the dequantized X
            extra_kwargs: the additional keyword arguments
            for each test entry in ops_under_test, it must have at least the fields
            for quantized_fn and reference_fn.
            output_range: the output range the operator will map to. By default, if it is
            no specified, the range will not be controlled and depend on Xmin and Xmax.
            change_zero_point: a boolean flag indicating if the zero point parameter should
            be determined based on torch_type during quantization (see sigmoid/hardsigmoid for
            examples). By default, if it is not specified, change_zero_point is assumed to be
            False and zero point will just take on the default value from X.
            `output_is_observed`: if specified and is True, we'll append extra
             output_scale/output_zero_point keyword argument when calling quantized op
        """
        # Retrieves the default parameters from X.
        X, (scale, zero_point, torch_type) = X
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
        if (X.device.type == 'cuda') and (torch.backends.quantized.engine == 'qnnpack'):
            return
        # Quantizes the reference to account for max error.
        # q_min and q_max only depend on the initial torch_type.
        q_min, q_max = torch.iinfo(torch_type).min, torch.iinfo(torch_type).max

        for op_group in test_configs:
            ref_op = op_group['reference_fn']
            for q_op in op_group['quantized_fn']:

                for memory_format in (torch.channels_last, torch.contiguous_format):
                    if memory_format == torch.channels_last and len(X.shape) != 4:
                        continue
                    X = X.to(memory_format=memory_format)

                    # Retrieves the inplace keyword arguments
                    # some functions require inplace=True to test in-place.
                    # copy.copy is needed because these are modified in place
                    extra_kwargs = \
                        copy.copy(op_group.get('extra_kwargs', {}))
                    output_is_observed = \
                        copy.copy(op_group.get('output_is_observed', False))

                    # Quantizes and dequantizes to account for max error.
                    qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                                   dtype=torch_type)
                    dqX = qX.dequantize()
                    dqY_hat = ref_op(dqX.clone(), **extra_kwargs)

                    # Adjusts output_scale if needed.
                    # The output_scale determines the quantization scale for functions that
                    # have a constrained output range. e.x. sigmoid ranges from 0 to 1.
                    output_scale = scale
                    if 'output_range' in op_group:
                        (f_min, f_max) = op_group['output_range']
                        output_scale = (f_max - f_min) / (q_max - q_min + 1.0)

                    # Adjusts output_zero_point if needed (see explanation for the
                    # change_zero_point parameter above).
                    # output_zero_point determines the additional offset that will be
                    # added to a scaled value during quantization.
                    if op_group.get('change_zero_point', False):
                        output_zero_point = 0 if torch_type == torch.qint32 else q_min
                    else:
                        output_zero_point = zero_point

                    # Quantizes the dequantized version of Y_hat.
                    qY_hat = torch.quantize_per_tensor(dqY_hat, scale=output_scale,
                                                       zero_point=output_zero_point,
                                                       dtype=torch_type)

                    if output_is_observed:
                        extra_kwargs.update({'output_scale': output_scale, 'output_zero_point': output_zero_point})

                    # Finds qY using in-place or non-in-place quantized operators.
                    qY = q_op(qX, **extra_kwargs)

                    self.assertEqual(qY, qY_hat, msg=f'{fn_name} - {q_op} failed: ({qY} vs. {qY_hat})')

    """Tests the correctness of the quantized::relu op."""
    @override_qengines
    def test_qrelu(self):
        relu_test_configs = [
            {
                'quantized_fn': [
                    torch.relu,
                    torch.relu_,
                    torch.nn.functional.relu,
                    torch.nn.functional.relu,
                ],
                'reference_fn': torch.nn.functional.relu
            },
            {
                'quantized_fn': [
                    torch.nn.functional.relu,
                    torch.nn.functional.relu,
                ],
                'reference_fn': torch.nn.functional.relu,
                'extra_kwargs': {
                    'inplace': True
                }
            }
        ]
        devices = ["cpu", "cuda"] if TEST_CUDA else ["cpu"]
        for device in devices:
            shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
            dtypes = (torch.quint8, torch.qint8)
            scales = (0.05, 0.1)
            zero_points = (0, 5)
            test_cases = itertools.product(shapes, dtypes, scales, zero_points)
            for shape, dtype, scale, zero_point in test_cases:
                X = torch.randn(*shape, device=device)
                X = (X, (scale, zero_point, dtype))
                self._test_activation_function(X, 'relu', relu_test_configs)

    """Tests the correctness of the quantized::relu6 op."""
    def test_qrelu6(self):
        relu6_test_configs = [
            {
                'quantized_fn': [
                    torch.ops.quantized.relu6,
                    torch.ao.nn.quantized.ReLU6(inplace=False),
                    torch.ao.nn.quantized.ReLU6(inplace=True)
                ],
                'reference_fn': torch.nn.functional.relu6
            }
        ]
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        scales = (0.05, 0.1)
        zero_points = (0, 5)
        test_cases = itertools.product(shapes, dtypes, scales, zero_points)
        for shape, dtype, scale, zero_point in test_cases:
            X = torch.randn(*shape) * 10
            X = (X, (scale, zero_point, dtype))
            self._test_activation_function(X, 'relu6', relu6_test_configs)

    """Tests the correctness of the quantized::sigmoid op."""
    @override_qengines
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_sigmoid_non_observed(self, X):
        sigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.sigmoid
                ],
                'reference_fn': torch.sigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True
            }
        ]
        self._test_activation_function(X, 'sigmoid', sigmoid_test_configs)

    """Tests the correctness of the quantized::sigmoid op."""
    # TODO: enable after observed output is supported in qnnpack
    # @override_qengines
    @skipIfNoFBGEMM
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_sigmoid(self, X):
        sigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.ops.quantized.sigmoid
                ],
                'reference_fn': torch.sigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
                'output_is_observed': True,
            }
        ]
        self._test_activation_function(X, 'sigmoid', sigmoid_test_configs)

    @skipIfNoFBGEMM
    def test_sigmoid_dequantize_rounding_error(self):
        # issue #107030
        sigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.ops.quantized.sigmoid
                ],
                'reference_fn': torch.sigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
                'output_is_observed': True,
            }
        ]
        X = (np.full(64, 514., dtype=np.float32), (1028.02, 255, torch.quint8))
        self._test_activation_function(X, 'sigmoid', sigmoid_test_configs)

    """Tests the correctness of the quantized::hardsigmoid op."""
    @override_qengines
    def test_qhardsigmoid(self):
        hardsigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.ao.nn.quantized.functional.hardsigmoid,
                ],
                'reference_fn': torch.nn.functional.hardsigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
            },
            {
                'quantized_fn': [
                    torch.ao.nn.quantized.functional.hardsigmoid,
                ],
                'reference_fn': torch.nn.functional.hardsigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
                'extra_kwargs': {
                    'inplace': True,
                },
            },
        ]
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        test_cases = itertools.product(shapes, dtypes)
        for shape, dtype in test_cases:
            X = (np.random.rand(*shape).astype(np.float32), (1.0, 0, dtype))
            self._test_activation_function(X, 'hardsigmoid', hardsigmoid_test_configs)

    @override_qengines
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_leaky_relu_observed_output(self, X):
        leaky_relu_test_configs = [
            {
                'quantized_fn': [
                    torch.ops.quantized.leaky_relu
                ],
                'reference_fn': torch.nn.functional.leaky_relu,
                'extra_kwargs': {
                    'negative_slope': 0.1,
                    'inplace': False,
                },
                'output_is_observed': True,
            }
        ]
        self._test_activation_function(X, 'leaky_relu', leaky_relu_test_configs)

    """Tests the correctness of the quantized::relu op."""
    def test_leaky_relu(self):
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        memory_formats = (torch.channels_last, torch.contiguous_format)
        test_cases = itertools.product(shapes, dtypes, memory_formats)
        for shape, dtype, memory_format in test_cases:
            if memory_format == torch.channels_last and len(shape) != 4:
                continue
            X, scale, zero_point, torch_type, alpha = \
                torch.randn(*shape), 0.1, 0, dtype, 0.01
            X = X.to(memory_format=memory_format)

            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            dqX = qX.dequantize()

            # torch.nn.functional
            op = torch.nn.functional.leaky_relu
            dqY = op(dqX, negative_slope=alpha)
            qY = torch.quantize_per_tensor(dqY, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            qY_hat = op(qX, negative_slope=alpha)
            self.assertEqual(qY.dequantize(), qY_hat.dequantize(),
                             msg=f"F.leaky_relu failed ({qY} vs {qY_hat})")

    """Tests the correctness of the quantized::elu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams()),
           alpha=st.floats(0.01, 10.0, allow_nan=False, allow_infinity=False))
    def test_qelu(self, X, alpha):
        X, (scale, zero_point, torch_type) = X
        output_scale = 0.5
        output_zero_point = 1

        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # calculate ELU(dqX) and quantize
        dqX = qX.dequantize()
        dqY_hat = dqX.clone()
        dqY_hat = torch.nn.functional.elu(dqX, alpha)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=output_scale, zero_point=output_zero_point,
                                           dtype=torch_type)

        qY = torch.ao.nn.quantized.functional.elu(qX, output_scale, output_zero_point, alpha=alpha)
        self.assertEqual(qY, qY_hat,
                         msg=f"F.elu failed ({qY} vs {qY_hat})")


    """Tests the correctness of the quantized::celu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       elements=hu.floats(-1e2, 1e2, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(scale_max=9.999999747378752e-06)),
           alpha=st.floats(0.01, 100.0, allow_nan=False, allow_infinity=False))
    def test_qcelu(self, X, alpha):
        X, (scale, zero_point, torch_type) = X
        output_scale = 0.5
        output_zero_point = 1

        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # calculate CELU(dqX) and quantize
        dqX = qX.dequantize()
        dqY_hat = torch.nn.functional.celu(dqX, alpha)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=output_scale, zero_point=output_zero_point,
                                           dtype=torch_type)

        # test regular
        qY = torch.ops.quantized.celu(qX, output_scale, output_zero_point, alpha=alpha)
        self.assertEqual(qY, qY_hat,
                         msg=f"F.celu failed ({qY} vs {qY_hat})")

    """Tests the correctness of the quantized::gelu op."""
    def test_qgelu(self):
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        memory_formats = (torch.channels_last, torch.contiguous_format)
        approximation = ['none', 'tanh']
        test_cases = itertools.product(shapes, dtypes, memory_formats, approximation)
        devices = ["cpu", "cuda"] if TEST_CUDA else ["cpu"]
        for shape, dtype, memory_format, approximate in test_cases:
            if memory_format == torch.channels_last and len(shape) != 4:
                continue

            X, scale, zero_point, torch_type = \
                torch.randn(*shape), 0.1, 0, dtype
            X = X.to(memory_format=memory_format)
            for device in devices:
                X = X.to(device=device)
                qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                               dtype=torch_type)
                dqX = qX.dequantize()

                op = torch.nn.functional.gelu
                dqY = op(dqX, approximate=approximate)
                qY = torch.quantize_per_tensor(dqY, scale=scale, zero_point=zero_point,
                                               dtype=torch_type)
                qY_hat = op(qX)
                self.assertEqual(qY.dequantize(), qY_hat.dequantize(),
                                 msg=f"F.gelu failed ({qY} vs {qY_hat})")

    """Tests the correctness of the quantized::prelu op."""
    def test_qprelu(self):
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        num_params = (0, 1)  # 0: num_parameter = num_channels
        dtypes = (torch.quint8, torch.qint8)
        memory_formats = (torch.channels_last, torch.contiguous_format)
        test_cases = itertools.product(shapes, num_params, dtypes, memory_formats)
        for shape, num_param, dtype, memory_format in test_cases:
            if memory_format == torch.channels_last and len(shape) != 4:
                continue
            X, scale, zero_point, torch_type = \
                torch.randn(*shape), 0.1, 0, dtype
            X = X.to(memory_format=memory_format)
            num_parameter = 1 if num_param == 1 or len(shape) == 1 else shape[1]
            W = torch.randn(num_parameter)
            W, w_scale, w_zero_point = \
                torch.randn(num_parameter), 0.2, 0

            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            dqX = qX.dequantize()
            qW = torch.quantize_per_tensor(W, scale=w_scale, zero_point=w_zero_point,
                                           dtype=torch_type)
            dqW = qW.dequantize()

            op = torch.nn.functional.prelu
            qop = torch.ops.quantized.prelu
            dqY = op(dqX, dqW)
            qY = torch.quantize_per_tensor(dqY, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            qY_hat = qop(qX, qW, scale, zero_point)
            self.assertEqual(qY.dequantize(), qY_hat.dequantize(),
                             msg=f"F.prelu failed ({qY} vs {qY_hat})")

    """Tests the correctness of the quantized::qlayer_norm op."""
    @skipIfNoFBGEMM
    def test_qlayer_norm(self):
        # hypothesis is flaky for this test, create test cases manually
        side_lens = (1, 8, 11)
        torch_types = (torch.qint8, torch.quint8)
        y_scales = (0.1, 4.23)
        y_zero_points = (0, 1)
        channels_last_list = (True, False)
        affine_list = (True, False)
        combined = [side_lens, torch_types, y_scales, y_zero_points,
                    channels_last_list, affine_list]
        test_cases = itertools.product(*combined)

        with override_quantized_engine("fbgemm"):
            for test_case in test_cases:

                side_len, torch_type, Y_scale, Y_zero_point, channels_last, \
                    affine = test_case
                shapes = [side_len] * 4

                # In the FP kernel, mean and variance are calculated in floating point.
                # In the quantized kernel, they are calculated in integer arithmetic.
                # Because of this, the numerics do not always match exactly which is
                # expected and acceptable. We do two things to allow this failure
                # in this test:
                # 1. do not use Hypothesis to generate the input tensor.  Hypothesis
                #    favors homogeneous inputs in its search strategies which isn't
                #    representative of the inputs we care about, and tends to maximize
                #    this particular numerics difference.
                # 2. allow a small % of off by Y_scale errors.  Even when the
                #    variance of the input is high, there can be off by one errors
                #    in the result if the input value happens to fall exactly on
                #    the bin boundary of the output scale.
                #
                # If we want the numerics to match we could switch to calculating
                # mean+var in floating point in the future, at the cost of speed.
                X, X_scale, X_zero_point = \
                    _get_random_tensor_and_q_params(shapes, 1.0, torch_type)

                qX = torch.quantize_per_tensor(X, scale=X_scale,
                                               zero_point=X_zero_point,
                                               dtype=torch_type)
                if channels_last:
                    qX = qX.contiguous(memory_format=torch.channels_last)
                dqX = qX.dequantize()

                # Enforce non-homogeneous inputs
                enough_unique_vals_in_each_layer = sum(
                    1 if (
                        dqX[i].shape[0] < 5 or
                        float(torch.unique(dqX[i]).shape[0]) / dqX[i].shape[0] > 0.01
                    ) else 0
                    for i in range(dqX.shape[0])
                ) == dqX.shape[0]
                assume(enough_unique_vals_in_each_layer)

                # Initialize the weights non-randomly for reproducibility, to avoid
                # flaky tests
                if affine:
                    weight = torch.ones(*qX.size()[1:], dtype=torch.float) * 0.5
                    bias = torch.ones(*qX.size()[1:], dtype=torch.float) * 1
                else:
                    weight = None
                    bias = None
                epsilon = 1e-5

                qY = torch.ops.quantized.layer_norm(
                    qX, qX.size()[1:], weight=weight, bias=bias, eps=epsilon,
                    output_scale=Y_scale, output_zero_point=Y_zero_point)

                Y_hat = F.layer_norm(
                    dqX, dqX.size()[1:], weight=weight, bias=bias, eps=epsilon)
                qY_hat = torch.quantize_per_tensor(
                    Y_hat, scale=Y_scale, zero_point=Y_zero_point, dtype=torch_type)

                # Due to the numerics difference mentioned above between calculating
                # the variance in float vs int, the results can still be slightly
                # different.
                dqY = qY.dequantize()
                dqY_hat = qY_hat.dequantize()
                diff = dqY - dqY_hat

                # off-by-one errors are magnitude of Y_scale
                num_diff = torch.sum(diff > Y_scale * 1.0001)
                pct_diff = float(num_diff) / (diff.numel() + 1e-5)
                num_diff_off_by_one = torch.sum((diff > 0) * (diff <= Y_scale))
                pct_diff_off_by_one = float(num_diff_off_by_one) / (diff.numel() + 1e-5)

                self.assertTrue(pct_diff < 1e-6)
                self.assertTrue(pct_diff_off_by_one < 0.01)


    """Tests the correctness of the quantized::qnnpack_tanh op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    def test_qtanh(self, X):
        # Note: QNNPACK is tested separately in TestQNNPackOps
        X, (scale, zero_point, torch_type) = X

        X = torch.from_numpy(X)
        Y = torch.tanh(X)

        qX = torch.quantize_per_tensor(X, scale=scale,
                                       zero_point=zero_point,
                                       dtype=torch_type)

        # Quantize the reference to account for max error.
        # Note that the output scale has +1, because we use scale of 2.0/2^BITS
        # in the implementations.
        f_min, f_max = -1.0, 1.0
        q_min, q_max = torch.iinfo(torch_type).min, torch.iinfo(torch_type).max
        output_scale = (f_max - f_min) / (q_max - q_min + 1.0)
        output_zero_point = int(round((q_max + q_min) / 2.0))
        qY = torch.quantize_per_tensor(Y, scale=output_scale,
                                       zero_point=output_zero_point,
                                       dtype=torch_type)
        qY_hat = torch.tanh(qX)
        self.assertEqual(qY, qY_hat,
                         msg=f"TanH failed: {qY} vs. {qY_hat}")

    """Tests the correctness of the quantized::threshold op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams()),
           threshold=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
           value=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False))
    def test_qthreshold(self, X, threshold, value):
        X, (scale, zero_point, torch_type) = X
        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # calculate threshold(dqX) and quantize
        dqX = qX.dequantize()
        dqY_hat = dqX.clone()
        dqY_hat = torch.nn.functional.threshold(dqY_hat, threshold, value)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

        ops_under_test = {
            'native': torch.threshold,
            'nn.functional': torch.nn.functional.threshold,
            'ao.nn.quantized.functional': torch.ao.nn.quantized.functional.threshold,
        }

        for name, op in ops_under_test.items():
            qY = op(qX, threshold, value)
            self.assertEqual(qY, qY_hat, msg=f"{name} qthreshold failed")

    """Tests the correctness of the quantized::clamp op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 8, 1, 8, max_numel=10**5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           min_val=hu.floats(-1e6, 1e6, allow_nan=False),
           max_val=hu.floats(-1e6, 1e6, allow_nan=False))
    def test_qclamp(self, X, min_val, max_val):
        X, (scale, zero_point, torch_type) = X

        assume(min_val <= max_val)
        Y_clamp = torch.clamp(torch.from_numpy(X), min=min_val, max=max_val)
        qY_clamp = torch.quantize_per_tensor(Y_clamp, scale=scale,
                                             zero_point=zero_point, dtype=torch_type)

        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)
        ops_under_test = {
            'ops.quantized': torch.ops.quantized.clamp,
        }

        for name, op in ops_under_test.items():
            qY_clamp_hat = op(qX, min=min_val, max=max_val)
            self.assertEqual(qY_clamp, qY_clamp_hat, msg=f"{name} qclamp failed")

        if torch.backends.quantized.engine == 'fbgemm':
            with override_quantized_engine('fbgemm'):
                Y_min_clamp = torch.clamp(X, min=min_val)
                Y_max_clamp = torch.clamp(X, max=max_val)

                qY_min_clamp = torch.quantize_per_tensor(Y_min_clamp, scale=scale,
                                                         zero_point=zero_point, dtype=torch_type)
                qY_max_clamp = torch.quantize_per_tensor(Y_max_clamp, scale=scale,
                                                         zero_point=zero_point, dtype=torch_type)


                for name, op in ops_under_test.items():
                    qY_min_clamp_hat = op(qX, min=min_val)
                    self.assertEqual(qY_min_clamp, qY_min_clamp_hat, msg=f"{name} qclamp failed")
                    qY_max_clamp_hat = op(qX, max=max_val)
                    self.assertEqual(qY_max_clamp, qY_max_clamp_hat, msg=f"{name} qclamp failed")

    """Tests the correctness of the quantized::hardtanh op."""
    @skipIfNoFBGEMM
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 8, 1, 8, max_numel=10**5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams()),
           min_val=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
           max_val=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    def test_hardtanh(self, X, min_val, max_val):
        with override_quantized_engine('fbgemm'):
            X, (scale, zero_point, torch_type) = X

            assume(min_val <= max_val)
            Y = X.copy()
            Y[Y < min_val] = min_val
            Y[Y > max_val] = max_val
            qY = torch.quantize_per_tensor(torch.from_numpy(Y), scale=scale,
                                           zero_point=zero_point, dtype=torch_type)
            X = torch.from_numpy(X)
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

            ops_under_test = {
                'ao.nn.quantized.functional.hardtanh':
                    torch.ao.nn.quantized.functional.hardtanh,
            }

            for name, op in ops_under_test.items():
                qY_hat = op(qX, min_val, max_val)
                self.assertEqual(qY, qY_hat, msg=f"{name} hardtanh failed")

            ops_under_test_inplace = {
                'inplace ao.nn.quantized.functional.hardtanh':
                    torch.ao.nn.quantized.functional.hardtanh,
            }

            for name, op_ in ops_under_test_inplace.items():
                qY_hat = qX.clone()
                op_(qY_hat, min_val, max_val, inplace=True)
                self.assertEqual(qY, qY_hat, msg=f"{name} hardtanh failed")

    """Tests the correctness of the quantized::hardswish op."""
    @override_qengines
    def test_hardswish(self):
        max_sides = (3, 4)
        side_lens = (1, 7)
        torch_types = (torch.quint8, torch.qint8)
        y_scales = (0.1, )
        y_zero_points = (1,)
        combined = [max_sides, side_lens, torch_types, y_scales, y_zero_points]
        test_cases = itertools.product(*combined)
        for test_case in test_cases:
            max_side, side_len, torch_type, Y_scale, Y_zero_point = test_case

            if torch.backends.quantized.engine == 'qnnpack' and torch_type != torch.quint8:
                continue

            shapes = [side_len] * max_side
            X, X_scale, X_zero_point = \
                _get_random_tensor_and_q_params(shapes, 2.0, torch_type)
            for memory_format in torch.channels_last, torch.contiguous_format:
                if memory_format == torch.channels_last and len(shapes) == 4:
                    X = X.to(memory_format=memory_format)
                qX = torch.quantize_per_tensor(X, scale=X_scale, zero_point=X_zero_point,
                                               dtype=torch_type)
                dqX = qX.dequantize()

                dqY_hat = F.hardswish(dqX)
                qY_hat = torch.quantize_per_tensor(dqY_hat, scale=Y_scale,
                                                   zero_point=Y_zero_point,
                                                   dtype=torch_type)

                qY = torch.ao.nn.quantized.functional.hardswish(
                    qX, scale=Y_scale, zero_point=Y_zero_point)
                self.assertEqual(
                    qY, qY_hat,
                    msg=f"Hardswish failed: {qY} vs {qY_hat}, {torch.backends.quantized.engine}")

    """Tests the correctness of the binary op + scalar."""
    def _test_binary_op_scalar_relu(self, A, b, binary_op_name, binary_op, quantized_op, quantized_op_relu):
        import copy
        op_scalar = quantized_op
        op_scalar_relu = quantized_op_relu

        A, (scale, zero_point, dtype) = A
        A = A.astype(np.float32)
        qA = torch.quantize_per_tensor(torch.from_numpy(A), scale, zero_point, dtype)

        if binary_op_name == 'add':
            C = binary_op(qA.dequantize(), round(b / scale) * scale)
        else:
            C = binary_op(qA.dequantize(), b)
        C_relu = copy.deepcopy(C)
        C_relu[C_relu < 0] = 0

        C_hat = op_scalar(qA, b)
        C_ref = torch.quantize_per_tensor(C, C_hat.q_scale(), C_hat.q_zero_point(), dtype)
        C_relu_hat = op_scalar_relu(qA, b)
        C_relu_ref = torch.quantize_per_tensor(
            C_relu, C_relu_hat.q_scale(), C_relu_hat.q_zero_point(), dtype)

        self.assertEqual(C_ref.dequantize(), C_hat.dequantize(),
                         msg=f"{binary_op_name}_scalar results don't match: "
                         f"{C_ref.dequantize()} vs {C_hat.dequantize()}")
        self.assertEqual(C_relu_ref.dequantize(), C_relu_hat.dequantize(),
                         msg=f"{binary_op_name}_scalar_relu results don't match: "
                         f"{C_relu_ref.dequantize()} vs {C_relu_hat.dequantize()}")

    @unittest.skipIf(IS_MACOS, "skipping macos test")
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 4, 1, 5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           b=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    def test_add_scalar_relu(self, A, b):
        self._test_binary_op_scalar_relu(A, b, "add", operator.add, torch.ops.quantized.add, torch.ops.quantized.add_relu)

    @unittest.skipIf(IS_MACOS, "skipping macos test")
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 4, 1, 5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           b=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    def test_mul_scalar_relu(self, A, b):
        self._test_binary_op_scalar_relu(A, b, "mul", operator.mul, torch.ops.quantized.mul, torch.ops.quantized.mul_relu)

    """Tests the correctness of the add and add_relu op."""
    def test_qadd_relu_same_qparams(self):
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            add_relu = torch.ops.quantized.add_relu
            add = torch.ops.quantized.add
            add_out = torch.ops.quantized.add
            add_relu_out = torch.ops.quantized.add_relu

            # NB: This is a strange size so that we exercise both the vectorized
            # implementation (64-element chunks at at time) as well as the scalar
            # implementation
            A = torch.arange(-128, 130, dtype=torch.float)
            B = torch.arange(-128, 130, dtype=torch.float)
            scale = 2.0
            zero_point = 127
            qA = torch.quantize_per_tensor(A, scale=scale, zero_point=zero_point,
                                           dtype=dtype)
            qB = torch.quantize_per_tensor(B, scale=scale, zero_point=zero_point,
                                           dtype=dtype)

            # Add ReLU ground truth
            C = (qA.dequantize() + qB.dequantize()).numpy()
            qC = _quantize(C, scale, zero_point, dtype=np_dtype[dtype])
            qC_hat = add(qA, qB, scale=scale, zero_point=zero_point)
            np.testing.assert_equal(qC, qC_hat.int_repr(),
                                    "Quantized addition failed.")
            qC_out_hat = torch._empty_affine_quantized(qC.shape,
                                                       scale=scale,
                                                       zero_point=zero_point,
                                                       dtype=dtype)
            add_out(qA, qB, out=qC_out_hat)
            self.assertEqual(qC_hat, qC_out_hat, msg="Add.out failed")

            # Add + ReLU ground truth
            Crelu = C.copy()
            Crelu[C < 0] = 0
            qCrelu = _quantize(Crelu, scale, zero_point, dtype=np_dtype[dtype])
            qCrelu_hat = add_relu(qA, qB, scale=scale, zero_point=zero_point)
            np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                    "Quantized addition with ReLU failed.")
            qCrelu_out_hat = torch._empty_affine_quantized(qCrelu.shape,
                                                           scale=scale,
                                                           zero_point=zero_point,
                                                           dtype=dtype)
            add_relu_out(qA, qB, out=qCrelu_out_hat)
            self.assertEqual(qCrelu_hat, qCrelu_out_hat,
                             msg="AddReLU.out failed")

    """Tests the correctness of the cudnn add and add_relu op
    (Similar to test_qadd_relu_different_qparams, will probably merge in the future)"""
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skipIf(not SM80OrLater, "requires sm80 or later.")
    @unittest.skipIf(TEST_ROCM, "not supported on rocm.")
    @unittest.skip("not currently working and feature isn't used")
    def test_qadd_relu_cudnn(self):
        dtype = torch.qint8
        add_relu = torch.ops.quantized.add_relu
        add = torch.ops.quantized.add

        A = torch.arange(-128, 130, dtype=torch.float).to(torch.device("cuda"))
        B = torch.arange(-128, 130, dtype=torch.float).to(torch.device("cuda"))
        scale_A = 2.5
        scale_B = 6.3
        scale_C = 12.9
        zero_point = 0
        qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                       dtype=dtype)
        qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                       dtype=dtype)
        # Add ground truth
        C = (qA.dequantize() + qB.dequantize()).to(device="cpu").numpy()
        qC = _quantize(C, scale_C, zero_point, dtype=np_dtype[dtype])
        qC_hat = add(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized addition failed.")

        # Add + ReLU ground truth
        Crelu = C.copy()
        Crelu[C < 0] = 0
        qCrelu = _quantize(Crelu, scale_C, zero_point, dtype=np_dtype[dtype])
        qCrelu_hat = add_relu(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                "Quantized addition with ReLU failed.")

    """Tests the correctness of the cudnn add and add_relu op for nhwc format"""
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skipIf(not SM80OrLater, "requires sm80 or later.")
    @unittest.skipIf(TEST_ROCM, "not supported on rocm.")
    @unittest.skip("not currently working and feature isn't used")
    def test_qadd_relu_cudnn_nhwc(self):
        dtype = torch.qint8
        add_relu = torch.ops.quantized.add_relu
        add = torch.ops.quantized.add

        A = torch.rand(16, 8, 4, 12).to(device="cuda")
        B = torch.rand(16, 8, 4, 12).to(device="cuda")
        scale_A = 2.5
        scale_B = 6.3
        scale_C = 12.9
        zero_point = 0
        qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                       dtype=dtype)
        qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                       dtype=dtype)
        # Add ground truth
        C = (qA.dequantize() + qB.dequantize()).to(device="cpu").numpy()
        qC = _quantize(C, scale_C, zero_point, dtype=np_dtype[dtype])
        qC_hat = add(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized addition failed.")

        # Add + ReLU ground truth
        Crelu = C.copy()
        Crelu[C < 0] = 0
        qCrelu = _quantize(Crelu, scale_C, zero_point, dtype=np_dtype[dtype])
        qCrelu_hat = add_relu(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                "Quantized addition with ReLU failed.")

    """Tests the correctness of the add and add_relu op."""
    def test_qadd_relu_different_qparams(self):
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            add_relu = torch.ops.quantized.add_relu
            add = torch.ops.quantized.add
            add_out = torch.ops.quantized.add
            add_relu_out = torch.ops.quantized.add_relu

            # NB: This is a strange size so that we exercise both the vectorized
            # implementation (64-element chunks at at time) as well as the scalar
            # implementation
            A = torch.arange(-128, 130, dtype=torch.float)
            B = torch.arange(-128, 130, dtype=torch.float)
            scale_A = 3.0
            zero_point_A = 7
            scale_B = 5.0
            zero_point_B = 127

            scale_C = 0.5
            zero_point_C = 5

            qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point_A,
                                           dtype=dtype)
            qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point_B,
                                           dtype=dtype)

            # Add ground truth
            C = (qA.dequantize() + qB.dequantize()).numpy()
            qC = _quantize(C, scale_C, zero_point_C, dtype=np_dtype[dtype])
            qC_hat = add(qA, qB, scale=scale_C, zero_point=zero_point_C)
            np.testing.assert_equal(qC, qC_hat.int_repr(),
                                    "Quantized addition failed.")
            qC_out_hat = torch._empty_affine_quantized(qC.shape,
                                                       scale=scale_C,
                                                       zero_point=zero_point_C,
                                                       dtype=dtype)
            add_out(qA, qB, out=qC_out_hat)
            self.assertEqual(qC_hat, qC_out_hat, msg="Add.out failed")

            # Add + ReLU ground truth
            Crelu = C.copy()
            Crelu[C < 0] = 0
            qCrelu = _quantize(Crelu, scale_C, zero_point_C, dtype=np_dtype[dtype])
            qCrelu_hat = add_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)
            np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                    "Quantized addition 
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/quantization/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/quantization/core`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/quantization/core/test_quantized_op.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/quantization/core`):

- [`test_quantized_op.py_kw.md_docs.md`](./test_quantized_op.py_kw.md_docs.md)
- [`test_workflow_module.py_kw.md_docs.md`](./test_workflow_module.py_kw.md_docs.md)
- [`test_quantized_tensor.py_kw.md_docs.md`](./test_quantized_tensor.py_kw.md_docs.md)
- [`test_backend_config.py_docs.md_docs.md`](./test_backend_config.py_docs.md_docs.md)
- [`test_workflow_module.py_docs.md_docs.md`](./test_workflow_module.py_docs.md_docs.md)
- [`test_top_level_apis.py_docs.md_docs.md`](./test_top_level_apis.py_docs.md_docs.md)
- [`test_quantized_module.py_docs.md_docs.md`](./test_quantized_module.py_docs.md_docs.md)
- [`test_quantized_functional.py_kw.md_docs.md`](./test_quantized_functional.py_kw.md_docs.md)
- [`test_utils.py_kw.md_docs.md`](./test_utils.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_quantized_op.py_docs.md_docs.md`
- **Keyword Index**: `test_quantized_op.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
