# Documentation: test_transformers.py

## File Metadata
- **Path**: `test/test_transformers.py`
- **Size**: 229655 bytes
- **Lines**: 4678
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
                attn = torch.bmm(q, k.transpose(-2, -1))

            attn = torch.nn.functional.softmax(attn, dim=-1)
            if dropout_p > 0.0:
                attn = torch.nn.functional.dropout(attn, p=dropout_p)
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output = torch.bmm(attn, v)
            return output
        # TODO: Support cross-device / dtype testing properly when instantiate_device_type_tests() is used.
        dtypes = [torch.double, torch.float]
        for dtype in dtypes:
            N = batch_size

            def rand_tensor(*shape):
                return torch.randn(shape, device=device, dtype=dtype)

            # This test compares python and C++ implementations of SDP.
            N_prime, L, S, E = 2, 4, 3, 6
            if input_dim == 3:
                query = rand_tensor(N, L, E)
                key = rand_tensor(N, S, E)
                value = rand_tensor(N, S, E)
            elif input_dim == 4:
                query = rand_tensor(N, N_prime, L, E)
                key = rand_tensor(N, N_prime, S, E)
                value = rand_tensor(N, N_prime, S, E)
            else:
                self.fail(f'Invalid input_dim {input_dim} encountered in SDP test')

            attn_mask = None
            if attn_mask_dim is not None:
                assert attn_mask_dim in [2, input_dim]
                mask_size = (L, S) if attn_mask_dim == 2 else ((N, L, S) if input_dim == 3 else (N, N_prime, L, S))
                attn_mask = (torch.ones(mask_size, device=device, dtype=torch.bool).tril() if is_causal
                             else torch.randint(0, 2, size=mask_size, device=device, dtype=torch.bool))

            with freeze_rng_state():
                # Python impl only supports float mask and 3D inputs.
                attn_mask_float = attn_mask
                if attn_mask_float is not None:
                    attn_mask_float = torch.zeros_like(attn_mask, dtype=query.dtype)
                    attn_mask_float.masked_fill_(attn_mask.logical_not(), float("-inf"))
                q, k, v = query.view(-1, L, E), key.view(-1, S, E), value.view(-1, S, E)
                a = attn_mask_float
                if a is not None and attn_mask_dim > 3:
                    a = a.view(-1, L, S)
                expected = sdp_ref(q, k, v, attn_mask=a, dropout_p=dropout_p)
                if input_dim > 3:
                    expected = expected.view(-1, N_prime, L, E)

            with freeze_rng_state():
                if is_causal:
                    # NB: Don't pass attn_mask here
                    actual = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, None, dropout_p, is_causal)

                    # Error case: both explicit attn_mask and is_causal are set
                    with self.assertRaisesRegex(RuntimeError,
                                                "Explicit attn_mask should not be set when is_causal=True"):
                        torch.nn.functional.scaled_dot_product_attention(
                            query, key, value, attn_mask, dropout_p, is_causal)
                else:
                    actual = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask, dropout_p, is_causal)
                    # This test the fully masked out rows case
                if torch.isnan(expected).any():
                    row_sums = attn_mask.sum(dim=-1)
                    masked_out_rows = (row_sums == 0)

                    for _ in range((input_dim - attn_mask_dim) - 1):
                        masked_out_rows = masked_out_rows.unsqueeze(0)

                    masked_out_rows = masked_out_rows.expand(expected.shape[:-1])
                    # Slice out the fully masked rows from expected and actual
                    expected_masked_out = expected[masked_out_rows]
                    actual_masked_out = actual[masked_out_rows]

                    expected_all_nan = torch.isnan(expected_masked_out).all()
                    actual_all_zero = (actual_masked_out.abs().sum() == 0)

                    self.assertTrue(expected_all_nan)
                    self.assertTrue(actual_all_zero)
                    return

                self.assertEqual(actual, expected)

        if attn_mask_dim is None:
            q = q.double().clone()
            k = k.double().clone()
            v = v.double().clone()
            q.requires_grad_()
            k.requires_grad_()
            v.requires_grad_()

            assert gradcheck(lambda *args, **kwargs: wrapper_set_seed(sdp_ref, *args, **kwargs),
                             (q, k, v, attn_mask, dropout_p))
            assert gradcheck(lambda *args, **kwargs:
                             wrapper_set_seed(torch.nn.functional.scaled_dot_product_attention, *args, **kwargs),
                             (q, k, v, attn_mask, dropout_p))

        def test_incompatible_mask(self, device):
            def ones_tensor(*shape):
                return torch.ones(shape, dtype=torch.float32)
            S, L, E, H = 1, 2, 4, 1
            qkv = ones_tensor(S, L, E)

            mha = nn.MultiheadAttention(E, H)
            mha.in_proj_weight = Parameter(torch.ones((E * 3, E)))
            mha.out_proj.weight = Parameter(torch.ones((E, E)))
            qkv = qkv.to(float)
            kpm = ones_tensor(S, L) * float("-inf")
            am = ones_tensor(L, L).to(bool)

            def func():
                return mha(qkv, qkv, qkv, need_weights=False, key_padding_mask=kpm, attn_mask=am)

            self.assertRaises(RuntimeError, func)

    @unittest.skipIf(TEST_WITH_CROSSREF, 'Fastpath not available with crossref')
    @torch.no_grad()
    def test_mask_check_fastpath(self):
        """
        Test that fastpath is executed independently of the masks that are passed.
        If the passed key padding mask is left aligned or mask_check=False, test that nested tensors are used
        (sparsity fastpath), otherwise use fastpath with traditional tensors.
        Also test that fast path is executed with both key padding mask and attention mask passed at the same time.
        """

        x = torch.Tensor([[[1, 2], [3, 4], [5, 6]]]).to(torch.float)

        def _test_fastpath(model, key_padding_mask, mock_return_value, attn_mask=None, nested_tensors=True):
            with patch('torch._transformer_encoder_layer_fwd') as fastpath_mock:
                fastpath_mock.return_value = mock_return_value
                model(x, src_key_padding_mask=key_padding_mask, mask=attn_mask)

                # If mock was called, fastpath was taken
                self.assertTrue(fastpath_mock.called)

                # If mock was called with nested tensors, sparsity fastpath was taken
                for call_args, _ in fastpath_mock.call_args_list:
                    self.assertEqual(call_args[0].is_nested, nested_tensors)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=2, nhead=2, dim_feedforward=8, batch_first=True)

        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=True, mask_check=True)
        model.eval()

        aligned_key_padding_mask = torch.Tensor([[0, 0, 1]]).to(torch.bool)
        not_aligned_key_padding_mask = torch.Tensor([[1, 0, 1]]).to(torch.bool)
        attn_mask = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).to(torch.bool)
        nested_tensor_return_value = torch.nested.nested_tensor([torch.ones((2, 2), dtype=torch.float)])
        tensor_return_value = torch.ones((1, 3, 2), dtype=torch.float)

        # Left aligned mask results in sparsity fastpath
        _test_fastpath(model, aligned_key_padding_mask, nested_tensor_return_value, nested_tensors=True)

        # Not aligned mask results in fastpath
        _test_fastpath(model, not_aligned_key_padding_mask, tensor_return_value, nested_tensors=False)

        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False, mask_check=True)
        model.eval()

        # If nested tensor disabled, fastpath is always taken
        _test_fastpath(model, aligned_key_padding_mask, tensor_return_value, nested_tensors=False)
        _test_fastpath(model, not_aligned_key_padding_mask, tensor_return_value, nested_tensors=False)
        # Fast path is taken if both attention mask and key padding mask are present
        _test_fastpath(model, aligned_key_padding_mask, tensor_return_value, attn_mask=attn_mask, nested_tensors=False)

        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=True, mask_check=False)
        model.eval()

        # Mask check disabled results in sparisty fastpath, independently of the mask
        _test_fastpath(model, aligned_key_padding_mask, nested_tensor_return_value, nested_tensors=True)
        _test_fastpath(model, not_aligned_key_padding_mask, nested_tensor_return_value, nested_tensors=True)

    # Test failing MHA when bias was NoneType
    def test_bias_is_none(self):
        x = torch.rand((1, 5, 10))
        model = torch.nn.modules.activation.MultiheadAttention(10, 1, bias=False, batch_first=True)
        model.eval()
        model(x, x, x)
        # completes without error

    def test_transformer_bias_is_none(self, device):
        batch_size = 2
        seqlen = 3
        d_model = 8
        nhead = 4

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, bias=False, batch_first=True, device=device)
        encoder_layer.eval()
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        # runs without error
        encoder_layer(x)

        with self.assertWarnsRegex(UserWarning, "encoder_layer.self_attn was passed bias=False"):
            encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1).eval()
            encoder(x)

        with self.assertWarnsRegex(UserWarning, "self_attn was passed bias=False"):
            transformer = torch.nn.Transformer(
                d_model=d_model, nhead=nhead, bias=False, batch_first=True, device=device
            ).eval()
            transformer(x, x)

    def test_train_with_is_causal(self, device):
        # training with is_causal
        S, L, E, H = 1, 2, 2, 1
        layer = nn.TransformerEncoderLayer(
            d_model=2,
            dim_feedforward=4,
            nhead=H,
            batch_first=True,
            activation="gelu",
            dropout=0,
        )
        criterion = nn.MSELoss()
        encoder = nn.TransformerEncoder(layer, 2).to(device)
        optimizer = optim.SGD(encoder.parameters(), lr=0.1, momentum=0.9)
        encoder.train()

        encoder.train()
        optimizer.zero_grad()
        inputs = torch.randn(S, L, E).to(device)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            inputs.size(1), device=device
        )

        outputs = encoder(inputs, mask=mask, is_causal=True)

        loss = criterion(outputs[:, 0:2, :], inputs[:, 0:2, :])
        loss.backward()
        optimizer.step()

        # inference with is_causal
        t_qvk = torch.randn((S, L, E), device=device, dtype=torch.float32)
        mha = nn.MultiheadAttention(E, H).to(device)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            S, device=device
        )

        attn_out, _ = mha(t_qvk, t_qvk, t_qvk, attn_mask=mask, is_causal=True)

        # Can't give only is_causal
        with self.assertRaises(RuntimeError):
            mha(t_qvk, t_qvk, t_qvk, is_causal=True)

        # # Passing a causal mask sets is_causal to 1
        causal_mask = torch.triu(
            torch.ones(L, L, device=inputs.device) * float('-inf'), diagonal=1
        ).to(torch.bool)

        mock_layer = MagicMock(torch.nn.MultiheadAttention(E, H), return_value=inputs)
        encoder.layers[1] = mock_layer
        outputs = encoder(inputs, mask=causal_mask)
        mock_layer.assert_called_with(ANY, src_mask=ANY, is_causal=True, src_key_padding_mask=ANY)

        # check expected numerical values with all kernels
        self.is_causal_kernels([SDPBackend.MATH], device)

    def is_causal_kernels(self, kernels, device):
        def ones_tensor(*shape):
            return torch.ones(shape, device=device, dtype=torch.float32).to(device)
        S, L, E, H = 1, 2, 4, 1
        qkv = ones_tensor(S, L, E)

        mha = nn.MultiheadAttention(E, H).to(device)
        mha.in_proj_weight = Parameter(torch.ones((E * 3, E), device=device))
        mha.out_proj.weight = Parameter(torch.ones((E, E), device=device))
        expected = torch.ones(size=(S, L, E)).to(device) * 16
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            qkv.size(1), device=device
        )

        for kernel in kernels:
            with sdpa_kernel(backends=[kernel]):
                actual, _ = mha(qkv, qkv, qkv, attn_mask=mask, need_weights=False, is_causal=True)
                self.assertTrue(torch.equal(actual, expected))

                if kernel != SDPBackend.MATH:
                    # fails with embedding size not multiple of 4
                    with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                        qkv_f, mha_f = ones_tensor(S, L, 2), nn.MultiheadAttention(2, H).to(device)
                        mask = torch.nn.Transformer.generate_square_subsequent_mask(
                            qkv_f.size(1), device=device
                        )
                        _ = mha_f(qkv_f, qkv_f, qkv_f, attn_mask=mask, need_weights=False, is_causal=True)
                        torch.cuda.synchronize()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform does not supposrt fused SDPA or pre-SM80 hardware"
    )
    def test_is_causal_gpu(self):
        device = 'cuda'
        self.is_causal_kernels([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION], device)

    def test_script_mha_in_proj_weight_none(self):
        mha = torch.nn.MultiheadAttention(
            embed_dim=128, num_heads=8, kdim=256, vdim=256
        ).eval()

        torch.jit.script(mha)

    @unittest.skipIf(TEST_WITH_CROSSREF, 'Fastpath not available with crossref')
    @torch.no_grad()
    def test_disable_fastpath(self, device):
        def _test_te_fastpath_called(model, args, kwargs=None, return_value=None, is_called=True):
            if kwargs is None:
                kwargs = {}
            with patch('torch._transformer_encoder_layer_fwd') as fastpath_mock:
                fastpath_mock.return_value = return_value
                model(*args, **kwargs)
                self.assertTrue(fastpath_mock.called == is_called)

        def _test_mha_fastpath_called(model, args, kwargs=None, return_value=None, is_called=True):
            if kwargs is None:
                kwargs = {}
            with patch('torch._native_multi_head_attention') as fastpath_mock:
                fastpath_mock.return_value = return_value
                model(*args, **kwargs)
                self.assertTrue(fastpath_mock.called == is_called)

        inp = torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float32, device=device)
        src_key_padding_mask = torch.tensor([[1, 0, 1]], dtype=torch.bool, device=device)
        te_return_value = torch.ones((1, 3, 2), dtype=torch.float32)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=2, nhead=2, dim_feedforward=8, batch_first=True)
        te = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=True, mask_check=True)
        te = te.to(device).eval()

        t = torch.nn.Transformer(d_model=2, nhead=2, batch_first=True, device=device).eval()
        src = torch.tensor([[[0, 1], [2, 3], [4, 5]]], dtype=torch.float32, device=device)
        tgt = torch.tensor([[[0, 1], [2, 3], [4, 5], [6, 7]]], dtype=torch.float32, device=device)
        t_return_value = torch.ones((1, 3, 2), dtype=torch.float32, device=device)

        mha = nn.MultiheadAttention(2, 2, batch_first=True, device=device).eval()
        q = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.float32, device=device)
        mha_return_value = torch.ones((1, 3, 2), dtype=torch.float32, device=device)

        _test_te_fastpath_called(
            te, (inp,), kwargs={'src_key_padding_mask': src_key_padding_mask},
            return_value=te_return_value, is_called=True
        )
        _test_te_fastpath_called(t, (src, tgt), return_value=t_return_value, is_called=True)
        _test_mha_fastpath_called(mha, (q, q, q,), return_value=mha_return_value, is_called=True)

        torch.backends.mha.set_fastpath_enabled(False)
        _test_te_fastpath_called(
            te, (inp,), kwargs={'src_key_padding_mask': src_key_padding_mask},
            return_value=te_return_value, is_called=False
        )
        _test_te_fastpath_called(t, (src, tgt), return_value=t_return_value, is_called=False)
        _test_mha_fastpath_called(mha, (q, q, q,), return_value=mha_return_value, is_called=False)

        torch.backends.mha.set_fastpath_enabled(True)
        _test_te_fastpath_called(
            te, (inp,), kwargs={'src_key_padding_mask': src_key_padding_mask},
            return_value=te_return_value, is_called=True
        )
        _test_te_fastpath_called(t, (src, tgt), return_value=t_return_value, is_called=True)
        _test_mha_fastpath_called(mha, (q, q, q,), return_value=mha_return_value, is_called=True)


class TestSDPAFailureModes(NNTestCase):
    """ Used to test the failure modes of scaled_dot_product_attention
    """
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION or not isSM8XDevice or not isSM120Device,
        "Does not support fused SDPA or not SM86+ hardware",
    )
    @parametrize("head_dim", [193, 256])
    @parametrize("dropout_p", [0.0, 0.2])
    def test_flash_backward_failure_sm86plus(self, device, head_dim: int, dropout_p: float):
        dtype = torch.float16
        make_tensor = partial(torch.rand, device=device, dtype=dtype)
        # See check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89 in
        # pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.h
        size = (2, 2, 4, head_dim)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            # Should not fail because inputs don't require grad
            flash_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

            self.assertEqual(math_ref, flash_ref, atol=1e-3, rtol=1e-3)

            # Should fail because inputs require grad
            q = make_tensor(size, requires_grad=True)
            k = make_tensor(size, requires_grad=True)
            v = make_tensor(size, requires_grad=True)
            if 192 < head_dim <= 224 or (head_dim > 224 and dropout_p != 0.0):
                self.assertRaises(
                    RuntimeError,
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, None, dropout_p, False
                    ),
                )
            else:
                flash_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, dropout_p, False)

    @onlyCUDA
    def test_dispatch_fails_no_backend(self, device):
        dtype = torch.float16
        with sdpa_kernel(backends=[SDPBackend.ERROR]):
            size = (2, 3, 4)
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch._fused_sdp_choice(q, k, v))
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )
    def test_invalid_fused_inputs_dim_3(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Dim is not 4
            size = (2, 3, 8)
            dtype = torch.float16
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            with self.assertWarnsRegex(UserWarning, "All fused kernels requires query, key and value to be 4 dimensional"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )
    def test_invalid_fused_inputs_broadcast(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            #  Fused Kernels don't support broadcasting for dense inputs
            dtype = torch.float16
            size = (2, 4, 3, 8)
            size_broadcast = (1, 4, 3, 8)
            q = torch.randn(size_broadcast, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize("kernel", PLATFORM_SPECIFIC_SDPA)
    def test_invalid_sequence_lengths(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Passing in a q,k,v with 0 length sequences will error
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 0, 8)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            with self.assertWarnsRegex(UserWarning, "All fused kernels do not support zero seq_len_q or seq_len_kv."):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize("kernel", PLATFORM_SPECIFIC_SDPA)
    def test_invalid_last_dim_stride(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Passing in a q,k,v with last dim stride not equal to 1 will error
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 8, 8)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            q.as_strided_(size, [2, 2, 2, 2])
            with self.assertWarnsRegex(UserWarning, "All fused kernels require the last dimension of the input to have stride 1."):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
        or not PLATFORM_SUPPORTS_CUDNN_ATTENTION,
        "Efficient or cuDNN Attention was not built for this system",
    )
    @parametrize("kernel", [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION])
    def test_mask_invalid_last_dim_stride(self, device, kernel):
        with sdpa_kernel(backends=[kernel]):
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 8, 8)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            attn_mask = make_tensor((2, 2, 8, 8))
            # Passing in a attn_mask with last dim stride not equal to 1 will error
            attn_mask.as_strided_(size, [2, 2, 2, 2])

            with self.assertWarnsRegex(
                UserWarning,
                "GPU backends require attn_mask's last dimension to have stride 1 while the CPU does not",
            ):
                self.assertRaises(
                    RuntimeError,
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask, 0.0, False
                    ),
                )

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    @parametrize("fused_kernel", [SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_sdpa_kernel_grouped_query_attention_cuda(self, device, fused_kernel):
        rand_query = torch.rand(8, 8, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
        rand_key = torch.rand(8, 4, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
        rand_value = torch.rand(8, 4, 64, 64, device=device, dtype=torch.float16, requires_grad=True)

        with sdpa_kernel(fused_kernel):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                with self.assertWarnsRegex(UserWarning, "For dense inputs, both fused kernels require query, "
                                           "key and value to have"):
                    F.scaled_dot_product_attention(rand_query, rand_key, rand_value, dropout_p=0.0,
                                                   is_causal=False, enable_gqa=True)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not flash_attention fused scaled dot product attention")
    @parametrize("kernel", PLATFORM_SPECIFIC_SDPA)
    def test_invalid_fused_inputs_head_dim(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # The embed dim per head is not divisible by 8 for flash attention
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 3, 9) if kernel == SDPBackend.EFFICIENT_ATTENTION else SdpaShape(2, 2, 3, 257)
            if TEST_WITH_ROCM:  # On ROCM, FA and EA share the backend GPU kernels
                size = SdpaShape(2, 2, 3, 513)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )
    def test_invalid_fused_inputs_invalid_dtype(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Invalid dtype for both Flash Attention and Mem Efficient Attention
            size = SdpaShape(2, 2, 3, 16)
            make_tensor = partial(torch.rand, device=device, dtype=torch.float64)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION])
    def test_invalid_fused_inputs_attn_mask_present(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Failures for unsupported SDP args
            size = SdpaShape(2, 2, 3, 16)
            make_tensor = partial(torch.rand, size, device=device, dtype=torch.float16)
            q, k, v = make_tensor(), make_tensor(), make_tensor()
            # Non-None attention mask
            mask = torch.ones((2, 2, 3, 3), device=device, dtype=q.dtype)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, mask, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support fused SDPA or pre-SM80 hardware")
    def test_unaligned_tensors(self, device):
        # The alignment is dependent on arch so we specify SM80OrLater
        dtype = torch.float16
        size = SdpaShape(2, 2, 8, 5)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            ctxmgr = self.assertRaises(RuntimeError)
            with ctxmgr:
                torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support fused SDPA or pre-SM80 hardware")
    def test_flash_fail_fp32(self, device):
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "Expected query, key and value to all be of dtype: {Half, BFloat16}"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    def test_flash_autocast_fp32_float16(self, device):
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    def test_flash_autocast_fp32_bfloat16(self, device):
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    # Note: do not truncate the list according to platforms. These tests should always raise errors.
    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_different_datatypes(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Different datatypes
            shape = (1, 4, 8, 16)
            query = torch.randn(shape, dtype=torch.float32, device=device)
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @onlyCUDA
    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_different_devices(self, device, kernel: SDPBackend):
        # Different devices
        shape = (1, 4, 8, 16)
        query = torch.randn(shape, dtype=torch.float32, device=device)
        key = torch.randn(shape, dtype=torch.float16, device='cpu')
        value = torch.randn(shape, dtype=torch.float16, device='cpu')
        self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_1_dimensional_inputs(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # 1 dimensional input
            shape = (1, 4)
            query = torch.randn(4, dtype=torch.float16, device=device)
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_fused_kernels_nested_broadcasting_error_cases(self, device):
        # one of k,v needs to be broadcasted and other has non consistent seq_len dim
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float32)
        batch, num_heads, head_dim = 32, 8, 64
        seq_lens_q = torch.randint(low=1, high=32, size=(batch,)).tolist()
        seq_lens_v = torch.randint(low=1, high=32, size=(batch,)).tolist()

        q_shape = SdpaShape(batch, num_heads, seq_lens_q, head_dim)
        k_shape = SdpaShape(1, num_heads, 1, head_dim)
        v_shape = SdpaShape(batch, num_heads, seq_lens_v, head_dim)

        query = rand_nested_tensor(q_shape).transpose(1, 2)
        key = rand_nested_tensor(k_shape).transpose(1, 2)
        value = rand_nested_tensor(v_shape).transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    def test_nested_fails_on_padding_head_dim(self, device):
        dtype = torch.bfloat16
        seq_len_list = [2, 4, 5, 6, 7]
        shape = SdpaShape(5, 8, seq_len_list, 57)
        make_tensor = partial(rand_sdpa_tensor, shape=shape, type="nested", device=device, dtype=dtype)
        q, k, v = make_tensor().transpose(1, 2), make_tensor().transpose(1, 2), make_tensor().transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "For NestedTensor inputs, Flash attention requires"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION or not isLessThanSM80Device,
                     "Current platform does not support fused SDPA or is an SM80+ device.")
    def test_mem_efficient_fail_bfloat16_less_than_sm80(self, device):
        dtype = torch.bfloat16
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "Expected query, key and value to all be of dtype: {Half, Float}"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    def test_flash_atteention_large_bf16_nan_values(self, device):
        query = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")
        key = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")
        value = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        self.assertFalse(torch.isnan(out).any(), "Output should not contain NaNs!")

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
    def test_fused_kernels_seq_len_0_inputs(self, device, fused_kernel):
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16)
        batch, num_heads, head_dim = 32, 16, 64
        seq_lens = torch.randint(low=1, high=32, size=(batch,))
        # make sure some seq_lens are 0
        num_zeros = 10
        indices = torch.randint(low=0, high=batch, size=(num_zeros,))
        seq_lens.scatter_(0, indices, 0)

        shape = SdpaShape(batch, num_heads, seq_lens.tolist(), head_dim)
        query = rand_nested_tensor(shape)
        key = rand_nested_tensor(shape)
        value = rand_nested_tensor(shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdpa_kernel(backends=[fused_kernel]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    def test_fused_kernels_nested_broadcasting_requires_grad_failure(self, device):
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16, requires_grad=True)
        batch, num_heads, head_dim, head_dim_v = 32, 16, 64, 64
        seq_lens = torch.randint(low=1, high=32, size=(batch,)).tolist()
        q_shape = SdpaShape(1, num_heads, 1, head_dim)
        k_shape = SdpaShape(batch, num_heads, seq_lens, head_dim)
        v_shape = SdpaShape(batch, 1, seq_lens, head_dim_v)

        # create a dense query
        query = torch.randn(q_shape, device=device, dtype=torch.float16, requires_grad=True)
        key = rand_nested_tensor(k_shape)
        value = rand_nested_tensor(v_shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "Both fused kernels do not support training with broadcasted NT inputs"):
                with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                    torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    def test_flash_attention_fail_with_non_square_causal_attention(self, device):
        dtype = torch.bfloat16
        q_shape = SdpaShape(1, 1, 8, 16)
        kv_shape = SdpaShape(1, 1, 12, 16)
        make_q = partial(torch.rand, q_shape, device=device, dtype=dtype)
        make_kv = partial(torch.rand, kv_shape, device=device, dtype=dtype)
        q, k, v = make_q(), make_kv(), make_kv()
        warning_str = "Flash attention does not support the is_causal flag when seqlen_q != seqlen_k."
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, warning_str):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, is_causal=True))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support Efficient Attention")
    def test_mem_eff_attention_fail_with_batch_size_geq_65536(self):
        batch_size = 2**16
        query = torch.rand([batch_size, 2, 2, 8], device='cuda', dtype=torch.float16, requires_grad=True)
        key = torch.rand([batch_size, 2, 2, 8], device='cuda', dtype=torch.float16, requires_grad=True)
        value = torch.rand([batch_size, 2, 2, 8], device='cuda', dtype=torch.float16, requires_grad=True)
        q_cpu, k_cpu, v_cpu = (query.detach().cpu().requires_grad_(True),
                               key.detach().cpu().requires_grad_(True),
                               value.detach().cpu().requires_grad_(True))
        with sdpa_kernel(backends=SDPBackend.EFFICIENT_ATTENTION):
            out = F.scaled_dot_product_attention(query, key, value)
        out_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
        grad_out = torch.rand_like(out)
        out.backward(grad_out)
        out_cpu.backward(grad_out.cpu())

        self.assertEqual(out, out_cpu, atol=2e-3, rtol=1e-4)
        self.assertEqual(query.grad, q_cpu.grad, atol=2e-3, rtol=1e-4)
        self.assertEqual(key.grad, k_cpu.grad, atol=2e-3, rtol=1e-4)
        self.assertEqual(value.grad, v_cpu.grad, atol=2e-3, rtol=1e-4)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support Efficient Attention")
    def test_mem_eff_attention_fail_with_batch_size_geq_65536_error(self):
        query = torch.rand([2**16, 2, 2, 8], device='cuda', dtype=torch.float16)
        key = torch.rand([2**16, 2, 2, 8], device='cuda', dtype=torch.float16)
        value = torch.rand([2**16, 2, 2, 8], device='cuda', dtype=torch.float16)
        error_str = (r"Efficient attention cannot produce valid seed and offset outputs when "
                     r"the batch size exceeds \(65535\)\.")
        with self.assertRaisesRegex(RuntimeError, error_str):
            torch._scaled_dot_product_efficient_attention(query, key, value,
                                                          attn_bias=None, compute_log_sumexp=True,
                                                          dropout_p=0.01)

    @largeTensorTest("15GB", "cuda")
    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support Efficient Attention")
    def test_mem_eff_attention_large_seq_len_uniform_attention(self):
        device = torch.device("cuda")
        dtype = torch.bfloat16

        num_queries = 49999
        num_heads = 2
        feature_dim = 16

        # Q and K are all zeros -> uniform attention
        query = torch.zeros(1, num_heads, num_queries, feature_dim, device=device, dtype=dtype, requires_grad=True)
        key = torch.zeros(1, num_heads, num_queries, feature_dim, device=device, dtype=dtype, requires_grad=True)
        value = torch.ones(1, num_heads, num_queries, feature_dim, device=device, dtype=dtype, requires_grad=True)
        mask = torch.ones((num_queries, num_queries), dtype=torch.bool, device=device)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
            )
            expected = torch.ones_like(output)
            grad_output = torch.ones_like(output)
            output.backward(grad_output)

            self.assertTrue(torch.allclose(output, expected))
            self.assertTrue(torch.allclose(query.grad, torch.zeros_like(query)))
            self.assertTrue(torch.allclose(key.grad, torch.zeros_like(key)))
            # For value, since each input position contributed 1/num_queries to each output, the grad should sum accordingly
            # for all ones grad_output, each value position receives grad of 1 (because sum of all softmax weights per row is 1)
            self.assertTrue(torch.allclose(value.grad, torch.ones_like(value)))


def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        return 128 if not is_dropout else 64
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        if is_sm8x:
            return 64 if (not is_dropout and is_causal) else 32
        else:
            return 64 if not is_dropout else 32
    elif head_dim <= 160:
        if is_sm8x:
            return 64
        else:
            return 32
    elif head_dim <= 192:
        return 64
    elif head_dim <= 224:
        return 64
    elif head_dim <= 256:
        return 64


def pad_last_dim(input_tensor, alignment_size, slice: bool = False):
    last_dim_size = input_tensor.size(-1)
    if (last_dim_size % alignment_size == 0):
        return input_tensor, last_dim_size
    pad_count = alignment_size - (last_dim_size % alignment_size)
    padded_tensor = F.pad(input_tensor, (0, pad_count))
    if slice:
        return padded_tensor[..., :last_dim_size], last_dim_size
    return padded_tensor, last_dim_size


class TestSDPA(NNTestCase):
    """ Used to test generic functionality of scaled_dot_product_attention
    Summary:
        If you are adding a new test to this class, make sure that it runs
        for both cpu and cuda. If you're test is only applicable to cuda,
        add it to TestSDPACudaOnly.
    """
    @expectedFailureMPS  # No double support
    @parametrize("contiguous_inputs", [True, False])
    def test_sdp_math_gradcheck(self, device, contiguous_inputs: bool):

        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = make_tensor(shape)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if contiguous_inputs:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            assert gradcheck(lambda *args, **kwargs:
                             wrapper_set_seed(torch.nn.functional.scaled_dot_product_attention, *args, **kwargs),
                             (query, key, value, None, 0.0, False)
                             )

    @parametrize("kernel", [SDPBackend.MATH])
    def test_scaled_dot_product_attention_math_with_negative_scale(self, device, kernel: SDPBackend):
        # https://github.com/pytorch/pytorch/issues/105190.
        def ref(x):
            v1 = torch.matmul(x, x.transpose(-1, -2))
            v2 = v1 / -0.0001
            v3 = v2.softmax(dim=-1)
            v4 = torch.matmul(v3, x)
            return v4

        x = torch.randn(1, 3, 64, 64, device=device)
        ref_result = ref(x)
        with sdpa_kernel(backends=[kernel]):
            sdp_math = torch.nn.functional.scaled_dot_product_attention(x, x, x, scale=-1.0 / 0.0001)
        self.assertEqual(ref_result, sdp_math)

    def test_scaled_dot_product_attention_fp16_overflow(self, device):
        # Regression test for https://github.com/pytorch/pytorch/issues/160841
        x = torch.full((1, 32, 23, 80), 256.0, dtype=torch.half, device=device)
        y = torch.nn.functional.scaled_dot_product_attention(x, x, x)
        self.assertFalse(y.isnan().any().item())

class TestSDPACpuOnly(NNTestCase):
    """ Used to test CPU only functionality of scaled_dot_product_attention """

    @parametrize("type", ["dense", "nested"])
    @parametrize("dropout", [0.0, 0.7])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.half])
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_cpu(self, device, type: str, dropout: float, dtype: torch.dtype):
        # Test that cpu and nestedtensor cpu return MATH backend
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype)
        size = SdpaShape(2, 8, 128, 64)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        if type == "nested" \
                or dropout > 0.0 \
                or dtype not in [torch.float32, torch.float64, torch.bfloat16, torch.float16]:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.MATH.value
        else:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.FLASH_ATTENTION.value

    def _generate_fixed_qkv_helper(
        self,
        device,
        dtype,
        batch_size,
        q_n_head,
        kv_n_head,
        q_seq_len,
        kv_seq_len,
        head_dim
    ):
        torch.manual_seed(777)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, requires_grad=False)
        q_shape = SdpaShape(batch_size, q_n_head, q_seq_len, head_dim)
        kv_shape = SdpaShape(batch_size, kv_n_head, kv_seq_len, head_dim)
        q = make_tensor(q_shape

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 12 class(es): TestTransformers, MyCustomLayer, def, MyCustomLayer, MyCustomLayer, FairseqDecoder, TestSDPAFailureModes, TestSDPA, TestSDPACpuOnly, TestSDPACudaOnly, TestSDPAXpuOnly, TestAttnBias

### Functions
This file defines 166 function(s): use_deterministic_algorithims, _check_equal, check_out_and_grad, query_key_value_clones, get_platform_specific_sdpa, rand_sdpa_tensor, _size, test_self_attn_TxT_attn_mask, test_train_with_pad_and_catch_error, test_multiheadattention_fastpath_attn_mask, test_transformerencoderlayer_src_mask, test_transformerencoderlayer_no_fastpath_with_hooks, hook, test_transformerencoder_fastpath, test_transformerencoder_square_input, test_transformerencoder, get_a_test_layer, _test, perm_fn, test_encoder_padding_and_src_mask_bool, test_decoder_padding_and_src_mask_bool, transformer_decoder, test_encoder_is_causal, test_math_backend_high_precision, scaled_dot_product_attention, test_mha_native_args, test_kpm_mask_trailing_column_with_nested_tensor, test_with_nested_tensor_input, test_script_encoder_subclass, test_transformerencoderlayer_subclass


## Key Components

The file contains 17401 words across 4678 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 229655 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
