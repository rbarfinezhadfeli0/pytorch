# Documentation: `test/test_transformers.py`

## File Metadata

- **Path**: `test/test_transformers.py`
- **Size**: 229,655 bytes (224.27 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: sdpa"]

import contextlib
from functools import partial
from collections import namedtuple
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.bias import CausalVariant, causal_lower_right, causal_upper_left
from torch.nn.parameter import Parameter
import unittest
from unittest.mock import patch, MagicMock, ANY
import math
import itertools
import torch.optim as optim
from torch.testing._internal.common_device_type import expectedFailureMPS, instantiate_device_type_tests, onlyCUDA, largeTensorTest
from typing import Optional
import torch.utils.cpp_extension
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    TEST_WITH_ROCM,
    skipIfRocm,
    skipIfRocmArch,
    MI300_ARCH,
    skipIfTorchDynamo,
    TEST_FAIRSEQ,
    run_tests,
    parametrize,
    freeze_rng_state,
    TEST_WITH_CROSSREF,
    slowTest,
    set_default_dtype,
    gradcheck,
    make_tensor,
    NOTEST_CPU,
    IS_WINDOWS,
    TEST_WITH_TORCHDYNAMO,
    TEST_XPU,
)
from torch._dynamo.testing import CompileCounterWithBackend


from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from torch.testing._internal.common_cuda import (
    IS_JETSON,
    SM80OrLater,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    tf32_on_and_off,
    tf32_enabled,
)

if TEST_FAIRSEQ:
    import fairseq.models.transformer as fairseq_transformer

SdpaShape = namedtuple('Sdpa_Shape', ['batch', 'num_heads', 'seq_len', 'head_dim'])
Tolerances = namedtuple('Tolerances', ['atol', 'rtol'])


@contextlib.contextmanager
def use_deterministic_algorithims(mode: bool, warn_only: bool):
    r"""
    This context manager can be used to temporarily enable or disable deterministic algorithms.
    Upon exiting the context manager, the previous state of the flag will be restored.
    """
    previous_mode: bool = torch.are_deterministic_algorithms_enabled()
    previous_warn_only: bool = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(mode, warn_only=warn_only)
        yield {}
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


# Found in torch/testing/_comparison.py
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}

isSM8XDevice = torch.cuda.is_available() and torch.cuda.get_device_capability() in [(8, 6), (8, 7), (8, 9)]
isSM90Device = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
isSM120Device = torch.cuda.is_available() and torch.cuda.get_device_capability() in [(12, 0), (12, 1)]
isSM5xDevice = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 5
isLessThanSM80Device = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8

TEST_WITH_CK = TEST_WITH_ROCM and torch.backends.cuda.preferred_rocm_fa_library() == torch.backends.cuda._ROCmFABackends['ck']

def _check_equal(
    golden: torch.Tensor,
    reference: torch.Tensor,
    test: torch.Tensor,
    fudge_factor: float,
    tensor_name: Optional[str] = None
) -> None:
    """
    Compare test tensor against golden and reference tensors.
    Golden is the highest precision possible serving as the "ground truth"
    Reference is the same precision as test and should also serve as less precisie ground truth.
    We calcculate the "reference error" by comparing the golden to reference and use this as the
    measruing stick for the test tensor.

    Raises ValueError if compiled error exceeds threshold.

    Args:
        golden (torch.Tensor): The golden tensor to compare against.
        reference (torch.Tensor): The reference tensor for error calculation.
        test (torch.Tensor): The test tensor to be evaluated.
        fudge_factor (float): A multiplier for the reference error to determine the threshold.
        tensor_name (Optional[str], optional): Name of the tensor for error reporting. Defaults to None.

    Raises:
        ValueError: If the test tensor contains NaN values while the reference does not,
                    or if the test error exceeds the calculated threshold.

    Notes:
        - For nested tensors, the function recursively calls itself on each nested element.
        - The error threshold is calculated as the maximum of a default tolerance for float32
          and the product of the reference error and the fudge_factor.
        - If the test error exceeds the threshold, a ValueError is raised with a detailed message.
    """
    if golden.is_nested and reference.is_nested and test.is_nested:
        for gold, ref, tst in zip(golden.unbind(), reference.unbind(), test.unbind()):
            _check_equal(gold, ref, tst, fudge_factor, tensor_name)
        return

    # Compute error between golden
    test_error = (golden - test).abs().max()
    ref_error = (golden - reference).abs().max()

    if torch.isnan(test_error).any() and not torch.isnan(ref_error).any():
        raise ValueError("Output/Grad with NaN")

    # Calculate the error threshold as the maximum of:
    # 1. A predefined default tolerance for float32
    # 2. The reference error multiplied by the fudge factor
    threshold = max(default_atol[torch.float32], ref_error * fudge_factor)
    if test_error > threshold:
        name = tensor_name or ""
        msg = f"{name} Test error {test_error} is greater than threshold {threshold}!"
        raise ValueError(msg)


def check_out_and_grad(
    out_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_query_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_key_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_value_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_attn_mask_tuple: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    fudge_factors: Optional[dict[str, float]] = None
) -> None:
    """
    Check output and gradients of attention mechanism tensors.
    Compares compiled results against reference and low-precision reference tensors.

    Args:
        out_tuple: Tuple of (ref, lp_ref, compiled) for output tensor
        grad_query_tuple: Tuple of (ref, lp_ref, compiled) for query gradient
        grad_key_tuple: Tuple of (ref, lp_ref, compiled) for key gradient
        grad_value_tuple: Tuple of (ref, lp_ref, compiled) for value gradient
        grad_attn_mask_tuple: Optional tuple of (ref, lp_ref, compiled) for attention mask gradient
        fudge_factors: Dictionary of fudge factors for each tensor type (default uses 5.0 for all)
    """
    default_fudge_factor = 5.0
    if fudge_factors is None:
        fudge_factors = {}

    out_ref, out_lp_ref, out = out_tuple

    with torch.no_grad():
        _check_equal(out_ref, out_lp_ref, out, fudge_factors.get('out', default_fudge_factor), "out")

        grad_checks = [
            (grad_query_tuple, "grad_query"),
            (grad_key_tuple, "grad_key"),
            (grad_value_tuple, "grad_value")
        ]

        for grad_tuple, name in grad_checks:
            ref_grad, lp_ref_grad, comp_grad = grad_tuple
            _check_equal(ref_grad, lp_ref_grad, comp_grad, fudge_factors.get(name, default_fudge_factor), name)

        if grad_attn_mask_tuple:
            attn_mask_ref_grad, attn_mask_ref_lp_grad, attn_mask_grad = grad_attn_mask_tuple
            _check_equal(
                attn_mask_ref_grad,
                attn_mask_ref_lp_grad,
                attn_mask_grad,
                fudge_factors.get("grad_attn_mask", default_fudge_factor),
                "grad_attn_mask",
            )


def query_key_value_clones(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dtype: torch.dtype = None):
    """ Clones the query, key, and value tensors and moves them to the specified dtype. """
    if dtype is None:
        dtype = query.dtype
    query_ref = query.detach().clone().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.detach().clone().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.detach().clone().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref

def get_platform_specific_sdpa():
    ret = []
    if PLATFORM_SUPPORTS_FLASH_ATTENTION:
        ret.append(SDPBackend.FLASH_ATTENTION)
    if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
        ret.append(SDPBackend.CUDNN_ATTENTION)
    if not ret:
        # Add a placeholder, an empty list causes "An empty arg_values was passed to @parametrize"
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    return ret

PLATFORM_SPECIFIC_SDPA = get_platform_specific_sdpa()
# Indicate the Efficient attention backend can support:
# 1. sequence longher than 512
# 2. head dimsion larger than 64
MEM_EFF_CAPABILITY_MATCHES_SM80 = SM80OrLater or TEST_WITH_ROCM

def rand_sdpa_tensor(shape: SdpaShape, device: str, dtype: torch.dtype, type: str,
                     requires_grad: bool = False, packed: bool = False) -> torch.Tensor:
    """Creates rand dense or nested tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        type (str): Nested or Dense
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    batch, num_heads, seq_len, head_dim = shape.batch, shape.num_heads, shape.seq_len, shape.head_dim
    if type == "nested":
        if isinstance(seq_len, list):
            def _size(i):
                return (seq_len[i], num_heads, head_dim) if not packed else (seq_len[i], 3 * num_heads * head_dim)

            return torch.nested.nested_tensor([
                torch.randn(_size(i), device=device, dtype=dtype, requires_grad=requires_grad)
                for i in range(batch)])
        else:
            size = (seq_len, num_heads, head_dim) if not packed else (seq_len, 3 * num_heads * head_dim)
            return torch.nested.nested_tensor([
                torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)
                for _ in range(batch)])
    else:
        assert (isinstance(seq_len, int))
        size = (batch, seq_len, num_heads, head_dim) if not packed else (batch, seq_len, 3 * num_heads * head_dim)
        return torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)


class TestTransformers(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @onlyCUDA
    @unittest.skip("4D mask not supported yet - activate when 4D mask supported")
    def test_self_attn_TxT_attn_mask(self, device):
        embed_dim = 16
        num_heads = 4
        batch_size = 10
        tgt_len = 16

        query = torch.rand(batch_size, tgt_len, embed_dim, device=device)  # [N, T, D]
        attn_mask = torch.randint(0, 2, (tgt_len, tgt_len)).cuda().float()  # [T, T]
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, 0.0)

        attn_mask_4d = attn_mask.expand(batch_size, num_heads, tgt_len, tgt_len)

        mta_model = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
        mta_model.eval()

        # Generate 3D results
        with torch.inference_mode():
            output_mask_4d = mta_model(query, query, query, attn_mask=attn_mask_4d)[0]
            output_mask_4d = output_mask_4d.transpose(0, 1)  # [N, T, D]

            output_mask_TxT = mta_model(query, query, query, attn_mask=attn_mask)[0]
            output_mask_TxT = output_mask_TxT.transpose(0, 1)  # [N, T, D]

            self.assertEqual(output_mask_4d, output_mask_TxT)

    @slowTest
    def test_train_with_pad_and_catch_error(self, device):
        iters = 100
        pad_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool).to(device)
        layer = nn.TransformerEncoderLayer(
            d_model=2,
            dim_feedforward=4,
            nhead=2,
            batch_first=True,
            activation="gelu",
            dropout=0,
        )
        criterion = nn.MSELoss()
        encoder = nn.TransformerEncoder(layer, 2).to(device)
        optimizer = optim.SGD(encoder.parameters(), lr=0.1, momentum=0.9)
        encoder.train()
        for _ in range(iters):
            encoder.train()
            optimizer.zero_grad()
            inputs = torch.cat([torch.randn(1, 2, 2), torch.zeros(1, 2, 2)], dim=1).to(device)

            outputs = encoder(inputs, src_key_padding_mask=pad_mask)

            loss = criterion(outputs[:, 0:2, :], inputs[:, 0:2, :])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                test = torch.cat([torch.randn(1, 2, 2), torch.zeros(1, 2, 2)], dim=1).to(device)

                # Expect uint8 type not supported
                e = None
                try:
                    encoder(test, src_key_padding_mask=pad_mask.to(torch.uint8))
                except AssertionError:
                    continue
                self.assertFalse(e, "Failed to catch unsupported uint8 type exception")

                test_train_bool = encoder(test, src_key_padding_mask=pad_mask)
                encoder.eval()

                # Expect long type not supported
                e = None
                try:
                    encoder(test, src_key_padding_mask=pad_mask.to(torch.int64))
                except AssertionError as e:
                    continue
                self.assertFalse(e, "Failed to catch unsupported Long type exception")

                test_eval_bool = encoder(test, src_key_padding_mask=pad_mask)
                l1_bool = nn.L1Loss()(test_train_bool[:, 0:2, :], test_eval_bool[:, 0:2, :]).item()
                self.assertTrue(l1_bool < 1e-4, "Eval/Train difference in pad_mask BOOL")

    @tf32_on_and_off(0.001)
    @parametrize("attn_mask_dim", [2, 3, None])
    @parametrize("key_padding_mask_dim", [2, None])
    @parametrize("mask_dtype", [torch.bool, torch.float32])
    def test_multiheadattention_fastpath_attn_mask(self, device, attn_mask_dim, key_padding_mask_dim, mask_dtype):
        # MHA converts all
        with torch.no_grad():
            B = 2
            L = 4
            D = 8
            H = 4

            if attn_mask_dim == 2:
                attn_mask = make_tensor((L, L), dtype=mask_dtype, device=device)
            elif attn_mask_dim == 3:
                attn_mask = make_tensor((B, 1, L, L), dtype=mask_dtype, device=device).expand(B, H, L, L).reshape(B * H, L, L)
            elif attn_mask_dim is None:
                attn_mask = None

            if key_padding_mask_dim == 2:
                key_padding_mask = make_tensor((B, L), dtype=mask_dtype, device=device)
            elif key_padding_mask_dim is None:
                key_padding_mask = None

            mha = nn.MultiheadAttention(D, H, batch_first=True, device=device)
            X = torch.randn(B, L, D, device=device)

            mha.train()  # disable fast path
            out, _ = mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
            mha.eval()  # enable fast path
            out_fp, _ = mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
            # The FP kernel will return NaNs while the sdpa kernel which is ran when the fast path is turned off returns 0 instead
            # of NaNs for fully masked rows
            self.assertEqual(out, out_fp.nan_to_num())

    @parametrize("nhead", [1, 4, 8])
    def test_transformerencoderlayer_src_mask(self, device, nhead):
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        model = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True).to(device)
        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        model(src, src_mask=src_mask)
        model.eval()
        with torch.no_grad():
            model(src, src_mask=src_mask)

    @parametrize("nhead", [3, 4])
    def test_transformerencoderlayer_no_fastpath_with_hooks(self, device, nhead):
        batch_size = 2
        seqlen = 4
        d_model = 12

        model = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model,
            batch_first=True).to(device).eval()
        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model

        cache = []

        # forward hook to save output
        def hook(module, inputs, output):
            cache.append(output[0].detach())

        # register hook to get the output of the self-attention layer
        handle = model.self_attn.register_forward_hook(hook)

        # forward pass
        with torch.inference_mode():
            model(src)

        # output of the self-attention layer
        assert len(cache) == 1, f"Expected 1 output, got {len(cache)}"

        # remove hook
        handle.remove()

    @skipIfRocmArch(MI300_ARCH)
    @tf32_on_and_off(0.001)
    @parametrize("use_torchscript", [False])
    @parametrize("enable_nested_tensor", [True, False])
    @parametrize("use_autocast", [True, False])
    @parametrize("d_model", [12, 256])
    def test_transformerencoder_fastpath(self, device, use_torchscript, enable_nested_tensor, use_autocast, d_model):
        """
        Test TransformerEncoder fastpath output matches slowpath output
        """
        torch.manual_seed(1234)
        nhead = 4
        dim_feedforward = d_model
        batch_first = True

        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=batch_first),
            num_layers=2,
            enable_nested_tensor=enable_nested_tensor
        ).to(device).eval()

        if use_torchscript:
            model = torch.jit.script(model)

        # each input is (input, mask)
        input_mask_pairs = [
            (
                torch.rand(3, 2, d_model),
                [
                    [0, 1],
                    [0, 1],
                    [1, 1]
                ]
            ),
            (
                torch.rand(2, 100, d_model),
                [
                    [0] * 98 + [1] * 2,
                    [0] * 90 + [1] * 10
                ]
            ),
            # softmax.cu switches from fast->slowpath at masked seqlen 1024. test 1024.
            (
                torch.rand(2, 1024, d_model),
                [
                    [0] * 1020 + [1] * 4,
                    [0] * 1024,
                ]
            ),
            (
                torch.rand(1, 1026, d_model),
                [[0] * 1024 + [1] * 2]
            ),
            # softmax.cu switches from fast->slowpath at masked seqlen 1024. test range of masks above 1024.
            (
                torch.rand(4, 1040, d_model),
                [
                    [0] * 1024 + [1] * 16,
                    [0] * 1025 + [1] * 15,
                    [0] * 1031 + [1] * 9,
                    [0] * 1040,
                ]
            )
        ]
        input_mask_pairs = [
            (
                torch.tensor(pair[0], device=device, dtype=torch.get_default_dtype()),  # float input
                torch.tensor(pair[1], device=device, dtype=torch.bool)  # bool mask
            ) for pair in input_mask_pairs
        ]

        maybe_autocast = torch.autocast("cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()
        with maybe_autocast:
            for input, src_key_padding_mask in input_mask_pairs:
                with torch.no_grad():
                    fastpath_output = model(input, src_key_padding_mask=src_key_padding_mask)
                slowpath_output = model(input, src_key_padding_mask=src_key_padding_mask)  # reference
                # Make sure fastpath_output is same shape as slowpath_output and mask.
                # When enable_nested_tensor=true, fastpath_output may be smaller than input tensor.
                # Eg if input bs=1, seqlen=6, and we mask out 2 tokens, fastpath_output will have bs=1, seqlen=4.
                # Expand back to old size to match.
                bs, true_seqlen, embed_dim = fastpath_output.shape
                expanded_seqlen = src_key_padding_mask.shape[1]
                fastpath_output_expanded = torch.zeros(bs, expanded_seqlen, embed_dim, device=device)
                fastpath_output_expanded[:, :true_seqlen, :] = fastpath_output
                # no garauntees on output corresponding to masked tokens, so they may vary between slow/fast path. set all to 0.
                fastpath_output_expanded = fastpath_output_expanded.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
                slowpath_output = slowpath_output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
                self.assertEqual(fastpath_output_expanded, slowpath_output)

    @tf32_on_and_off(0.001)
    @parametrize("with_no_grad", [True, False])
    @parametrize("training", [True, False])
    @parametrize("enable_nested_tensor", [False])
    def test_transformerencoder_square_input(self, with_no_grad, training, enable_nested_tensor, device):
        """
        Test for edge cases when input of shape (batch size, sequence length, embedding dimension) has
        batch size == sequence length
        """
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True),
            num_layers=2,
            enable_nested_tensor=enable_nested_tensor
        ).to(device)

        with torch.no_grad():
            # set constant weights of the model
            for p in model.parameters():
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

        if training:
            model = model.train()
        else:
            model = model.eval()
        x = torch.arange(0, 16).reshape(2, 2, 4).to(torch.get_default_dtype()).to(device)
        src_mask = torch.Tensor([[0, 1], [0, 0]]).to(torch.bool).to(device)

        if with_no_grad:
            cm = torch.no_grad()
        else:
            cm = contextlib.nullcontext()
        with cm:
            result = model(x, mask=src_mask)

        ref_output = torch.Tensor([[[2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351],
                                    [2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351]],
                                   [[2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689],
                                    [2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689]]]
                                  ).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        self.assertEqual(result, ref_output)

    @parametrize("batch_first", [True, False])
    @parametrize("training", [True, False])
    @parametrize("enable_nested_tensor", [True, False])
    def test_transformerencoder(self, batch_first, training, enable_nested_tensor, device):
        def get_a_test_layer(activation, batch_first=False):
            d_model = 4
            nhead = 2
            dim_feedforward = 16
            dropout = 0.0

            layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
            ).to(device)

            with torch.no_grad():
                # set constant weights of the model
                for p in layer.parameters():
                    x = p.data
                    sz = x.view(-1).size(0)
                    shape = x.shape
                    x = torch.cos(torch.arange(0, sz).float().view(shape))
                    p.data.copy_(x)

            return layer

        # this is a deterministic test for TransformerEncoder
        activation = F.relu

        def _test(batch_first, training, enable_nested_tensor):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            encoder_layer = get_a_test_layer(activation=activation,
                                             batch_first=batch_first)

            model = nn.TransformerEncoder(
                encoder_layer, 1, enable_nested_tensor=enable_nested_tensor
            ).to(device)

            if not training:
                model = model.eval()

            # deterministic input
            encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                   [0.5387, 0.1655, 0.3565, 0.0471]],
                                                  [[0.8335, 0.2799, 0.5031, 0.2947],
                                                   [0.1402, 0.0318, 0.7636, 0.1346]],
                                                  [[0.6333, 0.9344, 0.1376, 0.9938],
                                                   [0.8924, 0.2872, 0.6692, 0.2944]],
                                                  [[0.9897, 0.6915, 0.3154, 0.1733],
                                                   [0.8645, 0.3513, 0.3064, 0.0767]],
                                                  [[0.8117, 0.2366, 0.4838, 0.7881],
                                                   [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                 )).to(device)
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                                [2.427987, 0.021213, -0.602496, -0.084103]],
                                               [[2.424689, 0.019155, -0.604793, -0.085672],
                                                [2.413863, 0.022211, -0.612486, -0.072490]],
                                               [[2.433774, 0.021598, -0.598343, -0.087548],
                                                [2.425104, 0.019748, -0.604515, -0.084839]],
                                               [[2.436185, 0.022682, -0.596625, -0.087261],
                                                [2.433556, 0.021891, -0.598509, -0.086832]],
                                               [[2.416246, 0.017512, -0.610712, -0.082961],
                                                [2.422901, 0.024187, -0.606178, -0.074929]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # all 0 src_mask
            src_mask = torch.zeros([5, 5]).to(device) == 1
            result = model(encoder_input, mask=src_mask)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # all 0
            mask = torch.zeros([2, 5]).to(device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            mask[0, 1] = 1
            mask[1, 3] = 1
            mask[1, 4] = 1
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                                [2.428811, 0.021445, -0.601912, -0.084252]],
                                               [[2.425009, 0.019155, -0.604566, -0.085899],
                                                [2.415408, 0.02249, -0.611415, -0.073]],
                                               [[2.434199, 0.021682, -0.598039, -0.087699],
                                                [2.42598, 0.019941, -0.603896, -0.085091]],
                                               [[2.436457, 0.022736, -0.59643, -0.08736],
                                                [2.434021, 0.022093, -0.598179, -0.08679]],
                                               [[2.416531, 0.017498, -0.610513, -0.083181],
                                                [2.4242, 0.024653, -0.605266, -0.074959]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # test case 2, multiple layers no norm
            model = nn.TransformerEncoder(encoder_layer, 2, enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.419051, 0.017446, -0.608738, -0.085003],
                                                [2.419102, 0.017452, -0.608703, -0.085026]],
                                               [[2.419043, 0.017445, -0.608744, -0.084999],
                                                [2.419052, 0.017446, -0.608738, -0.085004]],
                                               [[2.419067, 0.017448, -0.608727, -0.085010],
                                                [2.419098, 0.017452, -0.608706, -0.085024]],
                                               [[2.419072, 0.017449, -0.608724, -0.085012],
                                                [2.419119, 0.017455, -0.608691, -0.085034]],
                                               [[2.419019, 0.017442, -0.608761, -0.084989],
                                                [2.419075, 0.017449, -0.608722, -0.085014]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            model = nn.TransformerEncoder(encoder_layer, 6, enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # test case 3, multiple layers with norm
            # d_model = 4
            norm = nn.LayerNorm(4)
            model = nn.TransformerEncoder(encoder_layer, 2, norm=norm,
                                          enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[1.695949, -0.357635, -0.893077, -0.445238],
                                                [1.695955, -0.357639, -0.893050, -0.445266]],
                                               [[1.695948, -0.357634, -0.893082, -0.445233],
                                                [1.695950, -0.357635, -0.893077, -0.445238]],
                                               [[1.695951, -0.357636, -0.893069, -0.445246],
                                                [1.695955, -0.357639, -0.893052, -0.445264]],
                                               [[1.695952, -0.357636, -0.893066, -0.445249],
                                                [1.695957, -0.357641, -0.893041, -0.445276]],
                                               [[1.695946, -0.357632, -0.893095, -0.445220],
                                                [1.695952, -0.357637, -0.893065, -0.445251]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            model = nn.TransformerEncoder(encoder_layer, 6, norm=norm,
                                          enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # TODO: remove set default dtype to double by making ref_output more precise.
        # Added because this test was copied from test_nn.py, which has default
        # dtype double. If default dtype is float, tests will say tensors not close because
        # ref output precision too low
        with set_default_dtype(torch.double):
            if training:
                cm = contextlib.nullcontext()
            else:
                cm = torch.no_grad()  # transformer fast path requires no grad
            with cm:
                _test(batch_first, training, enable_nested_tensor)

    @unittest.skipIf(sys.version_info < (3, 11), "not supported on pre-3.11 Python")
    def test_encoder_padding_and_src_mask_bool(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            dim_feedforward=32,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(16)
        encoder = nn.TransformerEncoder(
            encoder_layer, 2, encoder_norm
        )

        inputs = torch.randn(2, 3, 16)

        src_mask = torch.ones(3, 3, dtype=torch.bool).triu_(diagonal=1)
        input_seq_len = torch.tensor([3, 2])
        padding_mask = (
            torch.arange(3)[None, :].cpu() >= input_seq_len[:, None]
        )

        with (self.assertNoLogs(None) if not TEST_WITH_TORCHDYNAMO else contextlib.nullcontext()):
            encoder(
                inputs,
                mask=src_mask,
                src_key_padding_mask=padding_mask,
            )

    @unittest.skipIf(sys.version_info < (3, 11), "not supported on pre-3.11 Python")
    def test_decoder_padding_and_src_mask_bool(self):

        def transformer_decoder(inputs, input_seq_len, memory):
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=16,
                nhead=2,
                dim_feedforward=32,
                dropout=0.1,
                activation='relu',
                batch_first=True,
            )
            decoder_norm = nn.LayerNorm(16)
            decoder = nn.TransformerDecoder(
                decoder_layer, 2, decoder_norm
            )

            src_mask = torch.ones(
                inputs.shape[1], inputs.shape[1], dtype=torch.bool
            ).triu_(diagonal=1)
            padding_mask = (
                torch.arange(inputs.shape[1])[None, :].cpu()
                >= input_seq_len[:, None]
            )

            return decoder(
                inputs,
                memory,
                tgt_mask=src_mask,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )

        inputs = torch.randn(2, 3, 16)
        memory = torch.randn(2, 3, 16)
        input_seq_len = torch.tensor([3, 2])

        with self.assertNoLogs(None):
            transformer_decoder(inputs, input_seq_len, memory)

    def test_encoder_is_causal(self):

        d_model = 3
        layer = torch.nn.TransformerEncoderLayer(d_model, 1, 6, batch_first=True)
        layer.eval()
        x = torch.randn(1, 5, d_model)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))
        is_causal_output = layer(x, src_mask=mask, is_causal=True)
        masked_output = layer(x, src_mask=mask)

        self.assertEqual(masked_output, is_causal_output)

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform does not supposrt pre-SM80 hardware"
    )
    def test_math_backend_high_precision(self):
        xq = torch.rand([1, 128, 2, 80], device="cuda", dtype=torch.bfloat16) * 5
        xk = torch.rand([1, 128, 2, 80], device="cuda", dtype=torch.bfloat16) * 5
        xv = torch.randn([1, 128, 2, 80], device="cuda", dtype=torch.bfloat16)
        mask = None

        def scaled_dot_product_attention(
            xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, mask: Optional[torch.Tensor], backend: SDPBackend
        ) -> torch.Tensor:
            n_rep = 1
            xq, xk, xv = (tensor.transpose(1, 2) for tensor in (xq, xk, xv))
            xk = xk.repeat_interleave(n_rep, dim=1)
            xv = xv.repeat_interleave(n_rep, dim=1)

            with sdpa_kernel(backends=[backend]):
                attn_output = F.scaled_dot_product_attention(
                    xq, xk, xv, attn_mask=mask, dropout_p=0.0
                )
            return attn_output.transpose(1, 2)

        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        sdp_math_low_prec_out = scaled_dot_product_attention(xq, xk, xv, mask, SDPBackend.MATH)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        sdp_math_high_prec_out = scaled_dot_product_attention(xq, xk, xv, mask, SDPBackend.MATH)

        sdp_math_fp64_out_ref = scaled_dot_product_attention(
            xq.double(), xk.double(), xv.double(), mask, SDPBackend.MATH
        ).bfloat16()

        torch.testing.assert_close(sdp_math_high_prec_out, sdp_math_fp64_out_ref, atol=1e-2, rtol=1e-2)

        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close"):
            torch.testing.assert_close(sdp_math_low_prec_out, sdp_math_fp64_out_ref, atol=1e-2, rtol=1e-2)

    @onlyCUDA
    @parametrize("nb_heads", [1, 8])
    @parametrize("bias", [True, False])
    def test_mha_native_args(self, nb_heads, bias):

        B, L, F = 8, 100, 128
        batch_first = True
        fast_path = True
        use_pad_mask = (bias % 2) == 1

        mha = nn.MultiheadAttention(
            embed_dim=F,
            num_heads=nb_heads,
            batch_first=batch_first,
            bias=bias
        ).cuda()
        mha.eval()

        ctx = torch.no_grad if fast_path else contextlib.nullcontext
        with ctx():
            x = torch.randn(B, L, F).cuda()
            if not batch_first:
                x = x.transpose(0, 1)

            pad_mask = None
            if use_pad_mask:
                pad_mask = torch.zeros((B, L), dtype=torch.bool).cuda()

            mha(query=x, key=x, value=x, key_padding_mask=pad_mask)

    def test_kpm_mask_trailing_column_with_nested_tensor(self, device):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            activation='gelu',
            norm_first=False,
            batch_first=False,
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=True).to(device)

        x = torch.randn(10, 6, 256).to(device)
        mask = torch.ones(6, 10)
        mask[0, :] = 0  # here I masked 5 columns instead of just one
        mask = mask.bool().to(device)
        out = transformer_encoder(src=x, src_key_padding_mask=mask)
        self.assertEqual(out.shape[1], 6)

    # CPU unit test has_torch_functions in test environment,
    #   preventing successful completion
    @onlyCUDA
    def test_with_nested_tensor_input(self, device):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            activation='gelu',
            norm_first=False,
            batch_first=True,
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=True).to(device)

        transformer_encoder.eval()
        with torch.no_grad():
            x = torch.randn(6, 10, 256).to(device)
            mask = torch.ones(6, 10)
            mask[0, 0:] = 0  # here I masked 5 columns instead of just one
            mask[2, 2:] = 0  # here I masked 5 columns instead of just one
            mask[4, 4:] = 0  # here I masked 5 columns instead of just one
            mask[5, 8:] = 0  # here I masked 5 columns instead of just one
            mask = mask.bool().to(device)
            x = torch._nested_tensor_from_mask(x, mask.logical_not(), mask_check=False)
            out = transformer_encoder(src=x, src_key_padding_mask=None)

        self.assertEqual(out.is_nested, True)



    def test_script_encoder_subclass(self, device):
        class MyCustomLayer(nn.TransformerEncoderLayer):
            pass

        encoder = nn.TransformerEncoder(
            MyCustomLayer(d_model=256, nhead=8), num_layers=6
        ).to(device=device)
        torch.jit.script(encoder)

    # brazenly adapted from test_transformerencoderlayer_src_mask to test execution of
    # torchscripted transformerencoderlayer subclass
    def test_transformerencoderlayer_subclass(self, device):
        class MyCustomLayer(nn.TransformerEncoderLayer):
            pass

        nhead = 4
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        model = MyCustomLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True).to(device)
        script_model = torch.jit.script(model)

        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        torch.manual_seed(42)
        result = model(src, src_mask=src_mask)
        torch.manual_seed(42)
        scripted_result = script_model(src, src_mask=src_mask)
        self.assertEqual(result, scripted_result)

        model.eval()
        script_model = torch.jit.script(model)

        with torch.no_grad():
            result = model(src, src_mask=src_mask)
            scripted_result = script_model(src, src_mask=src_mask)
            self.assertEqual(result, scripted_result)


    def test_transformerencoderlayer_subclass_model(self, device):
        class MyCustomLayer(nn.TransformerEncoderLayer):
            pass

        nhead = 4
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        layer = MyCustomLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        model = nn.TransformerEncoder(
            layer, num_layers=6
        ).to(device=device)
        script_model = torch.jit.script(model)

        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        torch.manual_seed(42)
        result = model(src, mask=src_mask)
        torch.manual_seed(42)
        scripted_result = script_model(src, mask=src_mask)
        self.assertEqual(result, scripted_result)

        model.eval()
        script_model = torch.jit.script(model)

        with torch.no_grad():
            result = model(src, mask=src_mask)
            scripted_result = script_model(src, mask=src_mask)
            self.assertEqual(result, scripted_result)


    @onlyCUDA
    @unittest.skipIf(not TEST_FAIRSEQ, "Fairseq not found")
    def test_decoder_only_layer(self):
        class FairseqDecoder(torch.nn.Module):
            def __init__(
                self,
                embed_dim,
                attention_heads,
                ffn_embed_dim,
                num_layers,
                embedding_layer,  # torch.nn.Embedding. Must have a padding_idx field
                dropout=0,
                normalize_before=False,
                torch_encoder=None,  # torch encoder that you can map weights from
                activation="relu",
            ):
                super().__init__()

                cfg = fairseq_transformer.TransformerConfig()
                cfg.decoder.embed_dim = embed_dim
                cfg.decoder.output_dim = embed_dim
                cfg.decoder.attention_heads = attention_heads
                cfg.decoder.ffn_embed_dim = ffn_embed_dim
                cfg.dropout = dropout
                cfg.decoder.normalize_before = normalize_before
                cfg.decoder.layers = num_layers
                # make embedding behavior same as other encoders
                cfg.no_token_positional_embeddings = True
                cfg.no_scale_embedding = True
                cfg.activation_fn = activation

                dictionary = {}  # TODO: verify what this is

                self.decoder = fairseq_transformer.TransformerDecoder(
                    cfg,
                    dictionary,
                    embedding_layer,
                    no_encoder_attn=True,
                    output_projection=None,
                )

                if torch_encoder is not None:
                    self.decoder = torch_to_fairseq(torch_encoder, self.decoder)  # noqa: F821
                self.decoder = self.decoder.eval().cuda().half()

            def forward(
                self,
                tokens,
                src_lengths=None,
                with_triangle_mask=False,
                incremental_state=None,
            ):
                return self.decoder(
                    prev_output_tokens=tokens,
                    encoder_out=None,
                    incremental_state=incremental_state,
                    features_only=True,
                    full_context_alignment=not with_triangle_mask,
                    alignment_layer=None,
                    alignment_heads=None,
                    src_lengths=src_lengths,
                    return_all_hiddens=False,
                )[0]

    @tf32_on_and_off(0.003)
    @parametrize("batch_size", [0, 5])
    @parametrize("input_dim,attn_mask_dim,is_causal",
                 [(3, None, False), (3, 2, False), (3, 2, True), (3, 3, False), (3, 3, True),
                  (4, None, False), (4, 2, False), (4, 2, True), (4, 4, False), (4, 4, True)],
                 name_fn=lambda input_dim, attn_dim, is_causal: (
                     f"{input_dim}D_input_dim_" + (
                         f"{attn_dim}D_{'causal_' if is_causal else ''}attn_mask"
                         if attn_dim is not None else "no_attn_mask")))
    @parametrize("dropout_p", [0.0, 0.2, 0.5])
    @sdpa_kernel(backends=[SDPBackend.MATH])
    def test_scaled_dot_product_attention(self, device, batch_size, input_dim, attn_mask_dim, is_causal, dropout_p):
        def sdp_ref(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0):
            E = q.size(-1)
            q = q / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            if attn_mask is not None:
                attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
            else:
                at
```



## High-Level Overview


This Python file contains 12 class(es) and 166 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestTransformers`, `MyCustomLayer`, `MyCustomLayer`, `MyCustomLayer`, `FairseqDecoder`, `TestSDPAFailureModes`, `TestSDPA`, `TestSDPACpuOnly`, `TestSDPACudaOnly`, `TestSDPAXpuOnly`, `TestAttnBias`

**Functions defined**: `use_deterministic_algorithims`, `_check_equal`, `check_out_and_grad`, `query_key_value_clones`, `get_platform_specific_sdpa`, `rand_sdpa_tensor`, `_size`, `test_self_attn_TxT_attn_mask`, `test_train_with_pad_and_catch_error`, `test_multiheadattention_fastpath_attn_mask`, `test_transformerencoderlayer_src_mask`, `test_transformerencoderlayer_no_fastpath_with_hooks`, `hook`, `test_transformerencoder_fastpath`, `test_transformerencoder_square_input`, `test_transformerencoder`, `get_a_test_layer`, `_test`, `perm_fn`, `test_encoder_padding_and_src_mask_bool`

**Key imports**: contextlib, partial, namedtuple, os, sys, torch, torch.nn as nn, torch.nn.functional as F, scaled_dot_product_attention, sdpa_kernel, SDPBackend


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `functools`: partial
- `collections`: namedtuple
- `os`
- `sys`
- `torch`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.nn.functional`: scaled_dot_product_attention
- `torch.nn.attention`: sdpa_kernel, SDPBackend
- `torch.nn.attention.bias`: CausalVariant, causal_lower_right, causal_upper_left
- `torch.nn.parameter`: Parameter
- `unittest`
- `unittest.mock`: patch, MagicMock, ANY
- `math`
- `itertools`
- `torch.optim as optim`
- `torch.testing._internal.common_device_type`: expectedFailureMPS, instantiate_device_type_tests, onlyCUDA, largeTensorTest
- `typing`: Optional
- `torch.utils.cpp_extension`
- `torch.testing._internal.common_nn`: NNTestCase
- `torch._dynamo.testing`: CompileCounterWithBackend
- `torch.testing._internal.common_methods_invocations`: wrapper_set_seed
- `fairseq.models.transformer as fairseq_transformer`
- `time`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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
python test/test_transformers.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_transformers.py_docs.md`
- **Keyword Index**: `test_transformers.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
