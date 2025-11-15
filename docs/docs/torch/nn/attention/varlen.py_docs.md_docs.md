# Documentation: `docs/torch/nn/attention/varlen.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/attention/varlen.py_docs.md`
- **Size**: 12,712 bytes (12.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/attention/varlen.py`

## File Metadata

- **Path**: `torch/nn/attention/varlen.py`
- **Size**: 10,101 bytes (9.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Variable-length attention implementation using Flash Attention.

This module provides a high-level Python interface for variable-length attention
that calls into the optimized Flash Attention kernels.
"""

import logging
from functools import lru_cache
from typing import Any, NamedTuple

import torch


log = logging.getLogger(__name__)

__all__ = ["varlen_attn", "AuxRequest"]


@lru_cache(maxsize=8)
def _should_use_cudnn(device_index: int) -> bool:
    """Cache device capability check to avoid repeated CUDA calls."""
    return False


class AuxRequest(NamedTuple):
    """
    Request which auxiliary outputs to compute from varlen_attn.

    Each field is a boolean indicating whether that auxiliary output should be computed.
    """

    lse: bool = False


@torch.library.custom_op("torch_attn::_varlen_attn", mutates_args={})
def _varlen_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Private custom op for variable-length attention.

    This is the internal implementation. Users should use the public varlen_attn function instead.
    """

    use_cudnn = query.is_cuda and _should_use_cudnn(query.device.index)

    if use_cudnn:
        log.info("Using cuDNN backend for varlen_attn")
        result = torch.ops.aten._cudnn_attention_forward(
            query,
            key,
            value,
            None,  # attn_bias
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            True,  # compute_log_sumexp
            0.0,  # dropout_p hardcoded to 0.0
            is_causal,
            False,  # return_debug_mask
        )
        # cuDNN returns: (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask)
        output, softmax_lse, rng_state = result[0], result[1], result[6]
    else:
        log.info("Using Flash Attention backend for varlen_attn")
        output, softmax_lse, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
            query,
            key,
            value,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            0.0,  # dropout_p hardcoded to 0.0
            is_causal,
            return_debug_mask=False,
        )

    rng_state_ = torch.zeros(
        (2,), dtype=torch.uint64, device=query.device
    )  # hardcoded since dropout is hardcoded to 0
    return output, softmax_lse, rng_state_


@_varlen_attn.register_fake
def _varlen_attn_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake implementation for meta tensor computation and tracing.

    Based on the 3D varlen path from meta__flash_attention_forward:
    - query shape: (total, num_heads, head_dim)
    - logsumexp shape: (num_heads, total_q)
    """
    # Output has same shape as query
    output = torch.empty_like(query)

    # For varlen path: logsumexp shape is (num_heads, total_q)
    total_q = query.size(0)
    num_heads = query.size(1)
    logsumexp = torch.empty(
        (num_heads, total_q), dtype=torch.float, device=query.device
    )

    rng_state = torch.empty((2,), dtype=torch.uint64, device=query.device)

    return output, logsumexp, rng_state


def varlen_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    return_aux: AuxRequest | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Compute variable-length attention using Flash Attention.
    This function is similar to scaled_dot_product_attention but optimized for
    variable-length sequences using cumulative sequence position tensors.
    Args:
    - query (Tensor): Query tensor; shape :math:`(T_q, H, D)`
    - key (Tensor): Key tensor; shape :math:`(T_k, H, D)`
    - value (Tensor): Value tensor; shape :math:`(T_k, H, D)`
    - cu_seq_q (Tensor): Cumulative sequence positions for queries; shape :math:`(N+1,)`
    - cu_seq_k (Tensor): Cumulative sequence positions for keys/values; shape :math:`(N+1,)`
    - max_q (int): Maximum query sequence length in the batch.
    - max_k (int): Maximum key/value sequence length in the batch.
    - is_causal (bool, optional): If set to True, applies causal masking (default: False).
    - return_aux (Optional[AuxRequest]): If not None and ``return_aux.lse`` is True, also returns the logsumexp tensor.

    Shape legend:
    - :math:`N`: Batch size
    - :math:`T_q`: Total number of query tokens in the batch (sum of all query sequence lengths)
    - :math:`T_k`: Total number of key/value tokens in the batch (sum of all key/value sequence lengths)
    - :math:`H`: Number of attention heads
    - :math:`D`: Head dimension

    Returns:
    - Tensor: Output tensor from attention computation
    - If ``return_aux`` is not None and ``return_aux.lse`` is True, returns a tuple of Tensors:
    (output, lse), where lse is the logsumexp

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> batch_size, max_seq_len, embed_dim, num_heads = 2, 512, 1024, 16
        >>> head_dim = embed_dim // num_heads
        >>> seq_lengths = []
        >>> for _ in range(batch_size):
        ...     length = torch.randint(1, max_seq_len // 64 + 1, (1,)).item() * 64
        ...     seq_lengths.append(min(length, max_seq_len))
        >>> seq_lengths = torch.tensor(seq_lengths, device="cuda")
        >>> total_tokens = seq_lengths.sum().item()
        >>>
        >>> # Create packed query, key, value tensors
        >>> query = torch.randn(
        ...     total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda"
        ... )
        >>> key = torch.randn(
        ...     total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda"
        ... )
        >>> value = torch.randn(
        ...     total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda"
        ... )
        >>>
        >>> # Build cumulative sequence tensor
        >>> cu_seq = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        >>> cu_seq[1:] = seq_lengths.cumsum(0)
        >>> max_len = seq_lengths.max().item()
        >>>
        >>> # Call varlen_attn
        >>> output = varlen_attn(
        ...     query, key, value, cu_seq, cu_seq, max_len, max_len, is_causal=False
        ... )
    """
    out, lse, _ = torch.ops.torch_attn._varlen_attn(
        query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal
    )
    if return_aux is not None and return_aux.lse:
        return out, lse
    return out


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal = inputs
    out, lse, rng_state = output

    ctx.save_for_backward(query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state)

    ctx.max_q = max_q
    ctx.max_k = max_k
    ctx.is_causal = is_causal


@torch.library.custom_op("torch_attn::_varlen_attn_backward", mutates_args={})
def _varlen_attn_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    rng_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unused = torch.empty(0, device=query.device)

    use_cudnn = query.is_cuda and _should_use_cudnn(query.device.index)
    if use_cudnn:
        log.info("Using cuDNN backend for varlen_attn")
        dq, dk, dv = torch.ops.aten._cudnn_attention_backward(
            grad_out,
            query,
            key,
            value,
            out,
            lse,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            0.0,
            is_causal,
            rng_state,
            unused,
        )
    else:
        log.info("Using Flash Attention backend for varlen_attn")
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad_out,
            query,
            key,
            value,
            out,
            lse,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            0.0,
            is_causal,
            rng_state,
            unused,
        )
    return dq, dk, dv


@_varlen_attn_backward.register_fake
def _varlen_attn_backward_fake(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    rng_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake implementation for meta tensor computation and tracing.
    """

    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)

    return grad_query, grad_key, grad_value


def _backward(
    ctx: Any, grad_out: torch.Tensor, grad_lse: torch.Tensor, grad_rng: torch.Tensor
) -> tuple[torch.Tensor | None, ...]:
    query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state = ctx.saved_tensors

    max_q = ctx.max_q
    max_k = ctx.max_k
    is_causal = ctx.is_causal

    dq, dk, dv = torch.ops.torch_attn._varlen_attn_backward(
        grad_out,
        query,
        key,
        value,
        out,
        lse,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        rng_state,
    )
    return dq, dk, dv, None, None, None, None, None, None


_varlen_attn.register_autograd(_backward, setup_context=_setup_context)

```



## High-Level Overview

"""Variable-length attention implementation using Flash Attention.This module provides a high-level Python interface for variable-length attentionthat calls into the optimized Flash Attention kernels.

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AuxRequest`

**Functions defined**: `_should_use_cudnn`, `_varlen_attn`, `_varlen_attn_fake`, `varlen_attn`, `_setup_context`, `_varlen_attn_backward`, `_varlen_attn_backward_fake`, `_backward`

**Key imports**: logging, lru_cache, Any, NamedTuple, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/attention`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `functools`: lru_cache
- `typing`: Any, NamedTuple
- `torch`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/nn/attention`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_registry.py_docs.md`](./_registry.py_docs.md)
- [`_fa4.py_docs.md`](./_fa4.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`bias.py_docs.md`](./bias.py_docs.md)
- [`flex_attention.py_docs.md`](./flex_attention.py_docs.md)


## Cross-References

- **File Documentation**: `varlen.py_docs.md`
- **Keyword Index**: `varlen.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn/attention`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/attention`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/nn/attention`):

- [`_registry.py_docs.md_docs.md`](./_registry.py_docs.md_docs.md)
- [`bias.py_docs.md_docs.md`](./bias.py_docs.md_docs.md)
- [`_fa4.py_kw.md_docs.md`](./_fa4.py_kw.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`flex_attention.py_kw.md_docs.md`](./flex_attention.py_kw.md_docs.md)
- [`_fa4.py_docs.md_docs.md`](./_fa4.py_docs.md_docs.md)
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`_registry.py_kw.md_docs.md`](./_registry.py_kw.md_docs.md)
- [`varlen.py_kw.md_docs.md`](./varlen.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `varlen.py_docs.md_docs.md`
- **Keyword Index**: `varlen.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
