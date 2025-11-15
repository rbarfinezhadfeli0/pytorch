# Documentation: `test/inductor/test_flex_attention.py`

## File Metadata

- **Path**: `test/inductor/test_flex_attention.py`
- **Size**: 264,924 bytes (258.71 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
# flake8: noqa: B950

import functools
import json
import os
import random
import string
import tempfile
import unittest
import warnings
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from typing import Optional, TypeVar, Union
from unittest import expectedFailure, skip, skipUnless
from unittest.mock import patch

import torch
import torch.nn as nn
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._inductor import config, metrics
from torch._inductor.runtime.triton_compat import HAS_WARP_SPEC
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention import SDPBackend
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import (
    _create_empty_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
    _WARNINGS_SHOWN,
    and_masks,
    AuxOutput,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
    flex_attention_hop,
    noop_mask,
    or_masks,
)
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_BF16, TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfXPU,
    flex_attention_supported_platform as supported_platform,
    instantiate_device_type_tests,
    largeTensorTest,
    skipCPUIf,
    skipCUDAIf,
    skipXPUIf,
)
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils._triton import has_triton, has_triton_tma_device


# Use this decorator only when hitting Triton bugs on H100
running_on_a100_only = skipUnless(
    (
        (torch.cuda.is_available() and has_triton())
        and (torch.cuda.get_device_capability() == (8, 0) or torch.version.hip)
    )
    or (torch.xpu.is_available() and has_triton()),
    "Requires Triton + A100 or Triton + ROCm or Triton + Intel GPU",
)

Tolerances = namedtuple("Tolerances", ["atol", "rtol"])
torch.set_float32_matmul_precision("high")

index = torch.ops.aten.index
Tensor = torch.Tensor


T = TypeVar("T")
M = TypeVar("M", bound=Callable)


def large_tensor_test_class(
    size: str, device: Optional[Union[torch.device, str]] = None
) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        for name, method in list(cls.__dict__.items()):
            if callable(method) and name.startswith("test_"):
                setattr(cls, name, largeTensorTest(size, device)(method))
        return cls

    return decorator


@contextmanager
def temp_float32_matmul_precision(precision: str):
    """
    Temporarily set the float32 matmul precision and restore it after the context is exited.

    Args:
    precision (str): The precision to set ('highest', 'high', or 'medium').
    """

    def set_float32_matmul_precision_xpu(precision: str):
        if precision == "highest":
            torch._C._set_onednn_allow_tf32(False)
        if precision == "high":
            torch._C._set_onednn_allow_tf32(True)

    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        if TEST_ON_XPU:
            set_float32_matmul_precision_xpu(precision)
        yield
    finally:
        torch.set_float32_matmul_precision(original_precision)
        if TEST_ON_XPU:
            set_float32_matmul_precision_xpu(original_precision)


def skip_on_cpu(test_func):
    """Decorator to skip tests that are not supported on CPU."""
    decorated_func = skipCPUIf(True, "Not supported on CPU")(test_func)
    return decorated_func


def skip_on_cuda(test_func):
    """Decorator to skip tests that are not supported on CUDA."""
    decorated_func = skipCUDAIf(True, "Not supported on CUDA")(test_func)
    return decorated_func


def skip_on_rocm(test_func):
    """Decorator to skip tests that are not supported on CUDA."""
    IS_ROCM = torch.cuda.is_available() and torch.version.hip
    decorated_func = skipCUDAIf(IS_ROCM, "Not supported on ROCM")(test_func)
    return decorated_func


def skip_on_xpu(test_func):
    """Decorator to skip tests that are not supported on Intel GPU."""
    decorated_func = skipXPUIf(True, "Not supported on Intel GPU")(test_func)
    return decorated_func


def rmse(ref, res):
    """
    Calculate root mean squared error
    """
    ref = ref.to(torch.float64)
    res = res.to(torch.float64)
    return torch.sqrt(torch.mean(torch.square(ref - res)))


def create_attention(score_mod, block_mask, enable_gqa=False, kernel_options=None):
    return functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        kernel_options=kernel_options,
    )


def create_block_mask_test(score_mod, query, key):
    block_mask = create_block_mask(
        score_mod,
        1,
        1,
        query.shape[-2],
        key.shape[-2],
        query.device,
    )
    return block_mask


@dataclass
class DeviceConfig:
    dtypes: list[torch.dtype]
    dtypes_fast: list[torch.dtype]


TEST_ON_CUDA = (
    torch.cuda.is_available()
    and torch.utils._triton.has_triton()
    and torch.cuda.get_device_capability() >= (8, 0)
)
TEST_ON_XPU = torch.xpu.is_available() and torch.utils._triton.has_triton()

device_configs = {}
if HAS_GPU:
    if TEST_ON_CUDA:
        test_device = (
            "cuda",
            "cpu",
        )
    elif TEST_ON_XPU:
        torch._C._set_onednn_allow_tf32(True)
        test_device = ("xpu",)
else:
    test_device = ("cpu",)


class SubstringSet:
    def __init__(self, items):
        self.items = set(items)

    def __contains__(self, item):
        if "cuda" in item:
            item = "cuda"
        if "xpu" in item:
            item = "xpu"
        return item in self.items


DEVICE_SUPPORTS_BACKWARDS = SubstringSet(
    [
        "cuda",
    ]
)

device_configs["cuda"] = DeviceConfig(
    dtypes=(
        [torch.float32, torch.bfloat16, torch.float16]
        if PLATFORM_SUPPORTS_BF16
        else [torch.float16, torch.float32]
    ),
    dtypes_fast=[torch.float16],
)
device_configs["xpu"] = DeviceConfig(
    dtypes=([torch.float32, torch.bfloat16, torch.float16]),
    dtypes_fast=[torch.float16],
)
device_configs["cpu"] = DeviceConfig(
    dtypes=(
        [torch.float32, torch.bfloat16, torch.float16]
        if torch.backends.mkldnn.is_available()
        and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        else [torch.float32]
    ),
    dtypes_fast=[torch.float32],
)

torch_config_string = torch.__config__.show()
LONG_COMPILATION_ON_CPU = False

if "CLANG" in torch_config_string.upper():
    # if the compiler is clang, skip UT for CPU due to long compilation time found in CI
    # TODO: check reason of long compile time
    LONG_COMPILATION_ON_CPU = True


# --------- Useful score mod functions for testing ---------
def _causal(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return torch.where(token_q >= token_kv, score, float("-inf"))


def _rel_bias(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return score + (token_q - token_kv)


def _rel_causal(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return torch.where(token_q >= token_kv, score + (token_q - token_kv), float("-inf"))


def _generate_alibi_bias(num_heads: int):
    def _alibi_bias(
        score: Tensor,
        batch: Tensor,
        head: Tensor,
        token_q: Tensor,
        token_kv: Tensor,
    ) -> Tensor:
        scale = torch.exp2(-((head + 1) * 8.0 / num_heads))
        return score + (token_kv - token_q) * scale

    return _alibi_bias


def _inverse_causal(score, b, h, m, n):
    return torch.where(m <= n, score, float("-inf"))


def _times_two(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return score * 2


def _squared(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return score * score


def _head_offset(dtype: torch.dtype, device: str):
    """Captured Buffer"""
    head_offset = torch.rand(H, device=device, dtype=dtype)

    def score_mod(score, b, h, m, n):
        return score * head_offset[h]

    return score_mod


def _trig(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return torch.sin(torch.cos(score)) + torch.tan(b)


def _trig2(score, b, h, m, n):
    """Branching joint graph"""
    cos_score = torch.cos(score)
    sin_score = torch.sin(score)
    z = cos_score * sin_score + torch.tan(b)
    return z


# --------- Useful mask mod functions for testing ---------
def _causal_mask(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return token_q >= token_kv


def _inverse_causal_mask(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return token_q <= token_kv


test_score_mods = [
    _identity,
    _times_two,
    _squared,
    _causal,
    _inverse_causal,
    _rel_bias,
    _rel_causal,
    _generate_alibi_bias(8),
]

test_score_mask_mod_map = {
    _identity: noop_mask,
    _times_two: noop_mask,
    _squared: noop_mask,
    _causal: _causal_mask,
    _inverse_causal: _inverse_causal_mask,
    _rel_bias: noop_mask,
    _rel_causal: _causal_mask,
    _generate_alibi_bias(8): noop_mask,
}

captured_buffers_map = {
    "_head_offset": _head_offset,
}

B = 2
H = 4
S = 256
D = 64

test_Hq_Hkv = [
    (4, 2),
    (4, 1),
]

test_Bq_Bkv = [
    (3, 1),
    (4, 1),
    (5, 1),
]

test_block_size = [
    128,
    256,
    (128, 256),
    (256, 128),
]

test_strides = [
    ((H * S * D, S * D, D, 1), 997),  # offset
    ((H * D, D, B * H * D, 1), 499),  # transposed dimensions
    ((H * S * D, D, H * D, 1), 0),  # heads/sequence transposed
    (
        (S * (D + 1), B * S * (D + 1), (D + 1), 1),
        293,
    ),  # additional buffer on one dim
    (
        (1, D, (B + 1) * (H + 1) * D, 1),
        97,
    ),  # additional buffer on multiple dim + shared dimension
]


def query_key_value_clones(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
):
    """Clones the query, key, and value tensors and moves them to the specified dtype."""
    if dtype is None:
        dtype = query.dtype
    query_ref = query.detach().clone().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.detach().clone().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.detach().clone().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref


def batch_reserve(paged_attention: PagedAttention, target_seq_len: Tensor):
    (B,) = target_seq_len.shape
    for b in range(B):
        paged_attention.reserve(
            torch.tensor(b),
            target_seq_len[b],
        )


@large_tensor_test_class("2GB", device=test_device[0])
class TestFlexAttention(InductorTestCase):
    def setUp(self):
        super().setUp()
        skipCPUIf(
            LONG_COMPILATION_ON_CPU,
            "skip UT for CPU due to long compilation time found in CI",
        )

    def _check_equal(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        fudge_factor: float,
        tensor_name: Optional[str] = None,
        fudge_atol: float = 0,
    ):
        compiled_error = (golden_out - compiled_out).abs().mean()
        ref_error = (golden_out - ref_out).abs().mean()
        if torch.isnan(compiled_error).any() or torch.isnan(ref_error).any():
            self.fail("Output/Grad with NaN")
        name = tensor_name if tensor_name is not None else ""
        msg = f"{name} Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
        torch.testing.assert_close(
            compiled_error, ref_error, rtol=fudge_factor, atol=1e-7, msg=msg
        )

    def _check_out(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        is_paged_attention: bool = False,
    ):
        dtype = ref_out.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
                if is_paged_attention:
                    # paged attention is less accurate since it may reorder
                    # the blocks from block mask
                    fudge_factor = 20.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_out, ref_out, compiled_out, fudge_factor, "Out")

    def _check_out_and_grad(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        q_gold: torch.Tensor,
        q_ref: torch.Tensor,
        q: torch.Tensor,
        k_gold: torch.Tensor,
        k_ref: torch.Tensor,
        k: torch.Tensor,
        v_gold: torch.Tensor,
        v_ref: torch.Tensor,
        v: torch.Tensor,
    ):
        dtype = ref_out.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_out, ref_out, compiled_out, fudge_factor, "Out")

            # Check gradients
            q_fudge_factor = 1.0 * fudge_factor
            self._check_equal(
                q_gold.grad, q_ref.grad, q.grad, q_fudge_factor, "Grad_Query"
            )
            k_fudge_factor = 1.0 * fudge_factor
            self._check_equal(
                k_gold.grad, k_ref.grad, k.grad, k_fudge_factor, "Grad_Key"
            )
            v_fudge_factor = 1.0 * fudge_factor
            self._check_equal(
                v_gold.grad, v_ref.grad, v.grad, v_fudge_factor, "Grad_Value"
            )

    def run_test(
        self,
        score_mod: _score_mod_signature,
        dtype: torch.dtype,
        device: str,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: Optional[int] = None,
        KV_H: Optional[int] = None,
        KV_S: Optional[int] = None,
        V_D: Optional[int] = None,
        block_mask: Optional[BlockMask] = None,
    ):
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if KV_B is None:
            KV_B = Q_B
        if KV_H is None:
            KV_H = Q_H
        if KV_S is None:
            KV_S = Q_S
        if V_D is None:
            V_D = Q_D

        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        if block_mask is None:
            block_mask = create_block_mask(
                noop_mask, Q_B, Q_H, Q_S, KV_S, device=device
            )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=(Q_H != KV_H))

        compiled_sdpa = torch.compile(sdpa_partial)
        golden_out = sdpa_partial(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)
        compiled_out = compiled_sdpa(q, k, v)

        assert isinstance(golden_out, torch.Tensor)
        assert isinstance(ref_out, torch.Tensor)
        assert isinstance(compiled_out, torch.Tensor)

        if not requires_grad:
            self._check_out(
                golden_out,
                ref_out,
                compiled_out,
                is_paged_attention=False,
            )
        else:
            backward_grad = torch.randn(
                (Q_B, Q_H, Q_S, V_D), dtype=dtype, device=device
            )

            golden_out.backward(backward_grad.to(torch.float64))
            ref_out.backward(backward_grad)
            compiled_out.backward(backward_grad)

            self._check_out_and_grad(
                golden_out,
                ref_out,
                compiled_out,
                q_gold,
                q_ref,
                q,
                k_gold,
                k_ref,
                k,
                v_gold,
                v_ref,
                v,
            )

    def preprocess_paged_attention(
        self,
        score_mod: Optional[Callable],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block_mask,
        dtype: torch.dtype,
        device: str,
        page_size: int = 128,
    ) -> tuple[Tensor, Tensor, BlockMask, _score_mod_signature]:
        assert block_mask is not None, "Must provide block_mask"
        Q_B, Q_H, Q_S, _ = q.shape
        KV_B, KV_H, KV_S, QK_D = k.shape
        _, _, _, V_D = v.shape

        # test with different batch size
        max_batch_size = max(Q_B, KV_B) + 3

        n_pages = (KV_S + page_size - 1) // page_size * max_batch_size

        # allocate cache
        MAX_CACHED_SEQ_LEN = n_pages * page_size
        k_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            QK_D,
            device=device,
            dtype=dtype,
        )
        v_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            V_D,
            device=device,
            dtype=dtype,
        )

        # For testing purposes, we randomly initialize the page table, which maps
        # (batch_idx, logical_block_idx) to physical_block_idx. Specifically, PagedAttention
        # maintains a stack empty_pages of unused physical_block_idx. The `batch_reserve`
        # function grabs physical_block_idx from the top of empty_pages until there are enough
        # pages for each batch index (i.e., num pages for batch_idx >= target_seq_len[batch_idx]).
        # For example, at the first batch_reserve call, physical block indices (1,...,KV_S//4)
        # are allocated to batch index 0, and physical block indices
        # (KV_S//4+1, ..., KV_S//4 + KV_S//2) are allocated to batch index 1, etc.
        # Thus, kv tensors of batch index 1 will be scattered in the kv cache, simulating
        # a real use case of paged attention.
        paged_attention = PagedAttention(
            n_pages, page_size, max_batch_size, device=device
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 4, KV_S // 2, KV_S // 4, KV_S // 3], device=device),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 4, KV_S // 2, KV_S // 2, KV_S // 2], device=device),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 2, KV_S, KV_S // 2, KV_S], device=device),
        )
        batch_reserve(
            paged_attention, torch.tensor([KV_S, KV_S, KV_S, KV_S], device=device)
        )

        # update cache with k and v
        input_pos = (
            torch.arange(KV_S, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(KV_B, KV_S)
        )
        batch_idx = torch.arange(KV_B, device=device, dtype=torch.int32)
        paged_attention.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        # convert block mask and score mod
        kv_len_tensor = torch.full((KV_B,), KV_S, device=device, dtype=torch.int64)
        converted_block_mask = paged_attention.convert_logical_block_mask(
            block_mask, kv_len=kv_len_tensor
        )
        converted_score_mod = paged_attention.get_score_mod(
            score_mod, kv_len=kv_len_tensor
        )
        return k_cache, v_cache, converted_block_mask, converted_score_mod

    def run_paged_attention(
        self,
        score_mod: Optional[Callable],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dtype: torch.dtype,
        device: str,
        block_mask: Optional[BlockMask] = None,
        kernel_options: Optional[dict] = None,
    ) -> tuple[Tensor, Tensor]:
        B, Q_H, Q_S, KV_H, KV_S = (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[1],
            k.shape[2],
        )

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, B, 1, Q_S, KV_S, device=device)

        (
            k_cache,
            v_cache,
            converted_block_mask,
            converted_score_mod,
        ) = self.preprocess_paged_attention(
            score_mod, q, k, v, block_mask, dtype, device, block_mask.BLOCK_SIZE[1]
        )

        compiled_sdpa = torch.compile(flex_attention)

        # compute
        return_lse = True
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if requires_grad:
            compiled_out, compiled_lse = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=return_lse,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(Q_H != KV_H),
                kernel_options=kernel_options,
            )
        else:
            return_lse = False
            compiled_lse = None
            compiled_out = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=return_lse,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(Q_H != KV_H),
                kernel_options=kernel_options,
            )
        return compiled_out, compiled_lse

    def run_test_with_paged_attention(
        self,
        score_mod: Optional[Callable],
        dtype: torch.dtype,
        device,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        QK_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        V_D: int = D,
        block_mask: Optional[BlockMask] = None,
    ):
        assert Q_H % KV_H == 0
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        q = torch.randn(
            (Q_B, Q_H, Q_S, QK_D), dtype=dtype, device=device, requires_grad=False
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, QK_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, Q_B, 1, Q_S, KV_S, device=device)

        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=(Q_H != KV_H))
        golden_out, golden_lse = sdpa_partial(q_gold, k_gold, v_gold, return_lse=True)
        ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)

        compiled_out, compiled_lse = self.run_paged_attention(
            score_mod, q, k, v, dtype, device, block_mask
        )
        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
            is_paged_attention=True,
        )
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if requires_grad:
            self._check_out(
                golden_lse,
                ref_lse,
                compiled_lse,
                is_paged_attention=True,
            )

    def run_test_with_call(
        self,
        sdpa_call: Callable,
        dtype: torch.dtype,
        device: str,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        V_D: int = D,
    ):
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        compiled_sdpa = torch.compile(sdpa_call)
        golden_out = sdpa_call(q_gold, k_gold, v_gold)
        ref_out = sdpa_call(q_ref, k_ref, v_ref)
        compiled_out = compiled_sdpa(q, k, v)
        if not requires_grad:
            self._check_out(
                golden_out,
                ref_out,
                compiled_out,
                is_paged_attention=False,
            )
        else:
            backward_grad = torch.randn(
                (Q_B, Q_H, Q_S, V_D), dtype=dtype, device=device
            )

            golden_out.backward(backward_grad.to(torch.float64))
            ref_out.backward(backward_grad)
            compiled_out.backward(backward_grad)

            self._check_out_and_grad(
                golden_out,
                ref_out,
                compiled_out,
                q_gold,
                q_ref,
                q,
                k_gold,
                k_ref,
                k,
                v_gold,
                v_ref,
                v,
            )

    def run_dynamic_test(
        self,
        score_mask_mod: tuple[Callable, Callable],
        dtype: torch.dtype,
        device,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        score_mod, mask_mod = score_mask_mod

        # First batch with original dimensions (B, H, S, D)
        block_mask1 = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        sdpa_partial1 = create_attention(score_mod, block_mask=block_mask1)

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        q1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q1_ref, k1_ref, v1_ref = query_key_value_clones(q1, k1, v1)
        q1_gold, k1_gold, v1_gold = query_key_value_clones(q1, k1, v1, torch.float64)
        ref_out1 = sdpa_partial1(q1_ref, k1_ref, v1_ref)
        golden_out1 = sdpa_partial1(q1_gold, k1_gold, v1_gold)

        if requires_grad:
            backward_grad1 = torch.randn((B, H, S, D), dtype=dtype, device=device)
            golden_out1.backward(backward_grad1.to(torch.float64))
            ref_out1.backward(backward_grad1)

        # Second batch with modified dimensions (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)

        q2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q2_ref, k2_ref, v2_ref = query_key_value_clones(q2, k2, v2)
        q2_gold, k2_gold, v2_gold = query_key_value_clones(q2, k2, v2, torch.float64)
        ref_out2 = sdpa_partial2(q2_ref, k2_ref, v2_ref)
        golden_out2 = sdpa_partial2(q2_gold, k2_gold, v2_gold)

        if requires_grad:
            backward_grad2 = torch.randn((B, H, S, D), dtype=dtype, device=device)
            golden_out2.backward(backward_grad2.to(torch.float64))
            ref_out2.backward(backward_grad2)

        # Third batch with modified dimensions (B * 2, H, S / 4, D)
        S = int(S / 2)
        block_mask3 = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)

        q3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q3_ref, k3_ref, v3_ref = query_key_value_clones(q3, k3, v3)
        q3_gold, k3_gold, v3_gold = query_key_value_clones(q3, k3, v3, torch.float64)
        ref_out3 = sdpa_partial3(q3_ref, k3_ref, v3_ref)
        golden_out3 = sdpa_partial3(q3_gold, k3_gold, v3_gold)

        if requires_grad:
            backward_grad3 = torch.randn((B, H, S, D), dtype=dtype, device=device)
            golden_out3.backward(backward_grad3.to(torch.float64))
            ref_out3.backward(backward_grad3)

        # Clear dynamo counters
        torch._dynamo.reset()

        # First compilation with original dimensions
        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_sdpa1 = torch.compile(sdpa_partial1, backend=backend, dynamic=True)
        compiled_out1 = compiled_sdpa1(q1, k1, v1)

        if requires_grad:
            compiled_out1.backward(backward_grad1)

            self._check_out_and_grad(
                golden_out1,
                ref_out1,
                compiled_out1,
                q1_gold,
                q1_ref,
                q1,
                k1_gold,
                k1_ref,
                k1,
                v1_gold,
                v1_ref,
                v1,
            )
        else:
            self._check_out(golden_out1, ref_out1, compiled_out1)
        self.assertEqual(backend.frame_count, 1)

        # Second compilation with new dimensions
        compiled_sdpa2 = torch.compile(sdpa_partial2, backend=backend, dynamic=True)
        compiled_out2 = compiled_sdpa2(q2, k2, v2)

        if requires_grad:
            compiled_out2.backward(backward_grad2)

            self._check_out_and_grad(
                golden_out2,
                ref_out2,
                compiled_out2,
                q2_gold,
                q2_ref,
                q2,
                k2_gold,
                k2_ref,
                k2,
                v2_gold,
                v2_ref,
                v2,
            )
        else:
            self._check_out(golden_out2, ref_out2, compiled_out2)
        self.assertEqual(backend.frame_count, 1)

        # Third compilation with new dimensions
        compiled_sdpa3 = torch.compile(sdpa_partial3, backend=backend, dynamic=True)
        compiled_out3 = compiled_sdpa3(q3, k3, v3)

        if requires_grad:
            compiled_out3.backward(backward_grad3)

            self._check_out_and_grad(
                golden_out3,
                ref_out3,
                compiled_out3,
                q3_gold,
                q3_ref,
                q3,
                k3_gold,
                k3_ref,
                k3,
                v3_gold,
                v3_ref,
                v3,
            )
        else:
            self._check_out(golden_out3, ref_out3, compiled_out3)
        self.assertEqual(backend.frame_count, 1)

    def run_automatic_dynamic_test(
        self,
        score_mod: Callable,
        dtype: torch.dtype,
        device: str,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        block_mask1 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial1 = create_attention(score_mod, block_mask=block_mask1)
        # The first eager batch, shape (B, H, S, D)
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        q1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        golden_out1 = sdpa_partial1(
            q1.to(torch.float64), k1.to(torch.float64), v1.to(torch.float64)
        )
        ref_out1 = sdpa_partial1(q1, k1, v1)

        # The second eager batch, shape (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)
        q2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        golden_out2 = sdpa_partial2(
            q2.to(torch.float64), k2.to(torch.float64), v2.to(torch.float64)
        )
        ref_out2 = sdpa_partial2(q2, k2, v2)

        # The third eager batch, shape (B * 4, H, S / 4, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask3 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)
        q3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        golden_out3 = sdpa_partial3(
            q3.to(torch.float64), k3.to(torch.float64), v3.to(torch.float64)
        )
        ref_out3 = sdpa_partial3(q3, k3, v3)

        # Need to clear dynamo counters, since flex attention eager mode also uses dynamo tracing.
        # We check dynamo counters["frames"]["ok"] to ensure:
        # 1, the first batch is compiled with static shape
        # 2, the second batch is compiled with dynamic shape
        # 3, no re-compilation in the third batch
        torch._dynamo.reset()

        # Note, it seems like we really are less accurate than the float32
        # computation, likely due to the online softmax
        if dtype == torch.float32:
            fudge_factor = 10.0
        else:
            fudge_factor = 1.1

        # The first batch.
        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_out1 = torch.compile(sdpa_partial1, backend=backend, fullgraph=True)(
            q1, k1, v1
        )
        self._check_equal(golden_out1, ref_out1, compiled_out1, fudge_factor)
        self.assertEqual(backend.frame_count, 1)

        # The second batch (automatic dynamic).
        compiled_out2 = torch.compile(sdpa_partial2, backend=backend, fullgraph=True)(
            q2, k2, v2
        )
        self._check_equal(golden_out2, ref_out2, compiled_out2, fudge_factor)
        self.assertEqual(backend.frame_count, 2)

        # The third batch (no re-compilation).
        compiled_out3 = torch.compile(sdpa_partial3, backend=backend, fullgraph=True)(
            q3, k3, v3
        )
        self._check_equal(golden_out3, ref_out3, compiled_out3, fudge_factor)
        self.assertEqual(backend.frame_count, 2)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods(self, device, dtype, score_mod: Callable):
        self.run_test(score_mod, dtype, device=device)
        self.run_test_with_paged_attention(score_mod, dtype, device=device)

    @running_on_a100_only
    @common_utils.parametrize("score_mod", test_score_mods)
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_builtin_score_mods_seqlen_lt_default_sparse_block_size(
        self, device, dtype, score_mod: Callable
    ):
        # _DEFAULT_SPARSE_BLOCK_SIZE is 128
        attention = functools.partial(
            flex_attention,
            score_mod=score_mod,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
        )
        self.run_test_with_call(attention, dtype, device, B, H, 64, D, B, H, 64, D)

    @running_on_a100_only
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_seqlen_lt_custom_sparse_block_size(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        def causal_mask(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(
            causal_mask, 1, 1, 64, 64, BLOCK_SIZE=256, device=device
        )
        attention = functools.partial(
            flex_attention,
            score_mod=score_mod,
            block_mask=block_mask,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
        )
        self.run_test_with_call(
            attention,
            dtype,
            device,
            B,
            H,
            64,
            D,
            B,
            H,
            64,
            D,
        )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mask_mod", test_score_mask_mod_map.items())
    def test_builtin_score_mods_dynamic(
        self, device, dtype: torch.dtype, score_mask_mod: tuple[Callable, Callable]
    ):
        self.run_dynamic_test(score_mask_mod, dtype, S=1024, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_automatic_dynamic(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        self.run_automatic_dynamic_test(score_mod, dtype, S=1024, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_different_seqlen(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        inputs = (
            score_mod,
            dtype,
            device,
            B,
            H,
            S // 2,  # Seqlen of Q is different from seqlen of K/V
            D,
            B,
            H,
            S,
            D,
        )
        self.run_test(*inputs)
        self.run_test_with_paged_attention(*inputs)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("BLOCK_SIZE", test_block_size)
    def test_builtin_score_mods_different_block_size(
        self,
        device,
        dtype: torch.dtype,
        score_mod: Callable,
        BLOCK_SIZE: Union[int, tuple[int, int]],
    ):
        block_mask = create_block_mask(
            noop_mask, B, H, S, S, BLOCK_SIZE=BLOCK_SIZE, device=device
        )
        self.run_test(score_mod, dtype, block_mask=block_mask, device=device)
        self.run_test_with_paged_attention(
            score_mod, dtype, block_mask=block_mask, device=device
        )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("batch_dims", test_Bq_Bkv)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_kv_batch_broadcast(
        self,
        device,
        dtype: torch.dtype,
        batch_dims: tuple[int, int],
        head_dims: tuple[int, int],
        score_mod: Callable,
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0

        Bq, Bkv = batch_dims
        assert Bq > 1 and Bkv == 1

        block_mask = create_block_mask(noop_mask, Bq, 1, S, S, device=device)

        self.run_test(
            score_mod, dtype, device, Bq, Hq, S, D, Bkv, Hkv, S, D, block_mask
        )

    @supported_platform
    @skip_on_cpu
    def test_small_block_mask(self, device):
        compiled_create_block_mask = torch.compile(create_block_mask)

        def create_block_mask_from_seqlens(
            q_batch: torch.Tensor,
            kv_batch: torch.Tensor,
        ) -> BlockMask:
            B, H = None, None
            Q_LEN = q_batch.size(0)
            KV_LEN = kv_batch.size(0)

            def batch_mask_mod(
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ):
                q_idx_batch = q_batch[q_idx]
                kv_idx_batch = kv_batch[kv_idx]
                batch_mask = (
                    (q_idx_batch == kv_idx_batch)
                    & (q_idx_batch != -1)
                    & (kv_idx_batch != -1)
                )

                return batch_mask

            return compiled_create_block_mask(
                batch_mask_mod,
                B=B,
                H=H,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                device=device,
            )

        a = torch.tensor([2, 42, 18, 21, 4, 2, 7, 1, 1], device=device)
        b = torch.tensor([57, 21, 16, 8], device=device)

        for seqlen in [a, b]:
            create_block_mask_from_seqlens(seqlen, seqlen)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("batch_dims", test_Bq_Bkv)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_kv_batch_broadcast_causal_mask(
        self,
        device,
        dtype: torch.dtype,
        batch_dims: tuple[int, int],
        head_dims: tuple[int, int],
        score_mod: Callable,
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0

        Bq, Bkv = batch_dims
        assert Bq > 1 and Bkv == 1

        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, Bq, 1, S, S, device=device)
        attention = functools.partial(
            flex_attention, block_mask=block_mask, enable_gqa=(Hq != Hkv)
        )

        self.run_test_with_call(attention, dtype, device, Bq, Hq, S, D, Bkv, Hkv, S, D)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    @skip_on_rocm  # TODO: NaNs on ROCM
    @skip_on_xpu  # TODO: NaNs on XPU like ROCM, need another PR to fix.
    def test_GQA(self, device, dtype: torch.dtype, score_mod: Callable):
        inputs = (
            score_mod,
            dtype,
            device,
            B,
            H * 4,  # Hq = 4*Hkv.
            S // 8,
            D,
            B,
            H,
            S,
            D,
        )
        self.run_test(*inputs)
        self.run_test_with_paged_attention(*inputs)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize(
        "q_s", test_strides[:2]
    )  # TODO: fix layout for query braodcasting
    @common_utils.parametrize(
        "k_s,v_s",
        [
            (test_strides[0], test_strides[0]),
            (test_strides[0], test_strides[1]),
            (test_strides[2], test_strides[3]),
            (test_strides[3], test_strides[1]),
            # (test_strides[2], test_strides[4]), # TODO: Doesn't work for
            # broadcasting reasons i think
        ],
    )
    @common_utils.parametrize("do_s", test_strides[:3])
    def test_strided_inputs(self, device, dtype: torch.dtype, q_s, k_s, v_s, do_s):
        q1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)
        k1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)
        v1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)
        do1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)

        q_shape = (B, H, S // 2, D)
        k_shape = (B, H, S, D)
        v_shape = (B, H, S, D)
        do_shape = (B, H, S // 2, D)

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        def coerce_to_strides(val, shape, strides):
            strides, offset = strides
            val_max = [x * (y - 1) for x, y in zip(strides, shape)]
            assert sum(val_max) + offset < B * H * S * D * 2
            assert strides[-1] == 1
            return torch.as_strided(val, shape, strides, offset).requires_grad_(
                requires_grad
            )

        q = coerce_to_strides(q1, q_shape, q_s)
        k = coerce_to_strides(k1, k_shape, k_s)
        v = coerce_to_strides(v1, v_shape, v_s)
        do = coerce_to_strides(do1, do_shape, do_s)

        kernel_options = {"USE_TMA": True}

        block_mask = _create_empty_block_mask(q, k)
        score_mod = _generate_alibi_bias(8)
        sdpa_partial = create_attention(
            score_mod=score_mod, block_mask=block_mask, kernel_options=kernel_options
        )
        compiled_sdpa = torch.compile(sdpa_partial, fullgraph=True)
        ref_out = sdpa_partial(q, k, v)
        compiled_out = compiled_sdpa(q, k, v)

        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            ref_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )
        if requires_grad:
            ref_out.backward(do)
            ref_grads = [q.grad, k.grad, v.grad]
            q.grad = None
            k.grad = None
            v.grad = None

            compiled_out.backward(do)
            compiled_grads = [q.grad, k.grad, v.grad]
            q.grad = None
            k.grad = None
            v.grad = None
            torch.testing.assert_close(
                compiled_grads[0],
                ref_grads[0],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )
            torch.testing.assert_close(
                compiled_grads[1],
                ref_grads[1],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )
            torch.testing.assert_close(
                compiled_grads[2],
                ref_grads[2],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )

        # test paged attention which does not support backward
        q.requires_grad, k.requires_grad, v.requires_grad = False, False, False
        paged_compiled_out, _ = self.run_paged_attention(
            score_mod, q, k, v, dtype, device=device, kernel_options=kernel_options
        )
        torch.testing.assert_close(
```



## High-Level Overview


This Python file contains 32 class(es) and 433 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DeviceConfig`, `SubstringSet`, `TestFlexAttention`, `SimpleAttention`, `ApplyMask`, `Repro`, `Attention`, `Model`, `SimpleAttention`, `SimpleAttention`, `GraphModule`, `score_mod_0`, `mask_fn_0`, `GraphModule`, `fw_graph0`, `joint_graph0`, `mask_graph0`, `AsStridedErrorTensor`, `TestModule`, `FlexAttentionCPB`

**Functions defined**: `large_tensor_test_class`, `decorator`, `temp_float32_matmul_precision`, `set_float32_matmul_precision_xpu`, `skip_on_cpu`, `skip_on_cuda`, `skip_on_rocm`, `skip_on_xpu`, `rmse`, `create_attention`, `create_block_mask_test`, `__init__`, `__contains__`, `_causal`, `_rel_bias`, `_rel_causal`, `_generate_alibi_bias`, `_alibi_bias`, `_inverse_causal`, `_times_two`

**Key imports**: functools, json, os, random, string, tempfile, unittest, warnings, namedtuple, Callable


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `json`
- `os`
- `random`
- `string`
- `tempfile`
- `unittest`
- `warnings`
- `collections`: namedtuple
- `collections.abc`: Callable
- `contextlib`: contextmanager
- `dataclasses`: dataclass
- `itertools`: product
- `typing`: Optional, TypeVar, Union
- `unittest.mock`: patch
- `torch`
- `torch.nn as nn`
- `torch._dynamo.testing`: CompileCounterWithBackend, normalize_gm
- `torch._inductor`: config, metrics
- `torch._inductor.runtime.triton_compat`: HAS_WARP_SPEC
- `torch._inductor.test_case`: TestCase as InductorTestCase
- `torch._inductor.utils`: run_and_get_code
- `torch.nn.attention`: SDPBackend
- `torch.nn.attention.experimental._paged_attention`: PagedAttention
- `torch.testing`: FileCheck
- `torch.testing._internal`: common_utils
- `torch.testing._internal.common_cuda`: PLATFORM_SUPPORTS_BF16, TEST_MULTIGPU
- `torch.testing._internal.inductor_utils`: HAS_GPU
- `torch.utils._triton`: has_triton, has_triton_tma_device


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
python test/inductor/test_flex_attention.py
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

- **File Documentation**: `test_flex_attention.py_docs.md`
- **Keyword Index**: `test_flex_attention.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
