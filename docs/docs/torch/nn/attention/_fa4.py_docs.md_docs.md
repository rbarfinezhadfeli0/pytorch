# Documentation: `docs/torch/nn/attention/_fa4.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/attention/_fa4.py_docs.md`
- **Size**: 15,346 bytes (14.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/attention/_fa4.py`

## File Metadata

- **Path**: `torch/nn/attention/_fa4.py`
- **Size**: 11,619 bytes (11.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""UBER PROTOTYPE!!!"""
# mypy: allow-untyped-defs

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache
from typing import Any, TYPE_CHECKING
from typing_extensions import TypeVarTuple, Unpack

from . import _registry


if TYPE_CHECKING:
    from types import ModuleType

import torch
from torch.library import Library


__all__ = [
    "register_flash_attention_fa4",
]


_FA4_MODULE_PATH: str | None = None


@dataclass
class _FA4Handle:
    library: Library | None

    def remove(self) -> None:
        self.library = None


@cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


def register_flash_attention_fa4(
    module_path: str = "flash_attn.cute.interface",
) -> _FA4Handle:
    """
    Register FA4 flash attention kernels with the PyTorch dispatcher.

    Args:
        module_path: Python module path to the FA4 implementation.
    """
    global _FA4_MODULE_PATH
    _ = _fa4_import_module(module_path)
    _FA4_MODULE_PATH = module_path
    return _FA4Handle(_fa4_register_kernels())


@cache
def _fa4_import_module(module_path: str) -> ModuleType:
    module = importlib.import_module(module_path)
    if not hasattr(module, "_flash_attn_fwd") or not hasattr(module, "_flash_attn_bwd"):
        raise RuntimeError(f"Module '{module_path}' does not expose FA4 kernels")
    return module


def _fa4_register_kernels() -> Library:
    lib = Library("aten", "IMPL", "CUDA")  # noqa: TOR901
    lib.impl("_flash_attention_forward", _fa4_flash_attention_forward_impl, "CUDA")
    lib.impl("_flash_attention_backward", _fa4_flash_attention_backward_impl, "CUDA")
    lib.impl(
        "_scaled_dot_product_flash_attention",
        _fa4_scaled_dot_product_flash_attention_forward_impl,
        "CUDA",
    )
    lib.impl(
        "_scaled_dot_product_flash_attention_backward",
        _fa4_scaled_dot_product_flash_attention_backward_impl,
        "CUDA",
    )
    return lib


def _fa4_common_support_error(
    query: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    cum_seq_q: torch.Tensor | None,
    require_fp32: tuple[tuple[str, torch.Tensor], ...] = (),
) -> str | None:
    if not all(t.is_cuda for t in tensors):
        return "inputs must be CUDA tensors"
    if len({t.device for t in tensors}) != 1:
        return "inputs must share device"
    if query.dtype not in (torch.float16, torch.bfloat16):
        return "query dtype must be float16 or bfloat16"
    for name, tensor in require_fp32:
        if tensor.dtype != torch.float32:
            return f"{name} dtype must be float32"
    if cum_seq_q is None and query.dim() != 4:
        return "dense query must be 4D"
    if cum_seq_q is not None and query.dim() != 3:
        return "ragged query must be 3D"
    if not torch.cuda.is_available():
        return "CUDA not available"
    if _get_device_major(query.device) not in (9, 10):
        return "FA4 requires compute capability 9.0 or 10.0"
    return None


def _fa4_forward_support_error(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    return_debug_mask: bool,
    alibi_slopes: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    cum_seq_q: torch.Tensor | None,
) -> str | None:
    if dropout_p != 0.0:
        return "dropout_p must be 0"
    if return_debug_mask:
        return "return_debug_mask must be False"
    if alibi_slopes is not None:
        return "alibi_slopes not supported"
    if seqused_k is not None:
        if seqused_k.dtype != torch.int32:
            return "seqused_k must be int32"
        if not seqused_k.is_cuda:
            return "seqused_k must be CUDA"
    error = _fa4_common_support_error(
        query,
        (query, key, value),
        cum_seq_q,
    )
    if error is not None:
        if error == "inputs must share device":
            return "query, key, value must be on same device"
        return error
    return None


def _fa4_backward_support_error(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    dropout_p: float,
    cum_seq_q: torch.Tensor | None,
    window_size_left: int | None,
    window_size_right: int | None,
) -> str | None:
    if dropout_p != 0.0:
        return "dropout_p must be 0"
    if window_size_left is not None or window_size_right is not None:
        return "windowed attention not supported"
    error = _fa4_common_support_error(
        query,
        (grad_out, query, key, value, out, logsumexp),
        cum_seq_q,
        require_fp32=(("logsumexp", logsumexp),),
    )
    if error is not None:
        return error
    return None


Ts = TypeVarTuple("Ts")


def _transpose_dense(*tensors: Unpack[Ts]) -> tuple[Unpack[Ts]]:
    return tuple(t.transpose(1, 2) for t in tensors)  # type: ignore[attr-defined]


def _fa4_run_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    seqused_k: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _FA4_MODULE_PATH is None:
        raise RuntimeError("FA4 not registered")
    module = _fa4_import_module(_FA4_MODULE_PATH)
    kwargs: dict[str, Any] = {
        "softmax_scale": scale,
        "causal": is_causal,
        "window_size_left": window_size_left,
        "window_size_right": window_size_right,
        "return_lse": True,
        "cu_seqlens_q": cu_seq_q,
        "cu_seqlens_k": cu_seq_k,
        "seqused_k": seqused_k.contiguous() if seqused_k is not None else None,
    }
    out, lse = module._flash_attn_fwd(query, key, value, **kwargs)
    return out, lse.contiguous()


def _fa4_run_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    scale: float | None,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _FA4_MODULE_PATH is None:
        raise RuntimeError("FA4 not registered")
    module = _fa4_import_module(_FA4_MODULE_PATH)
    dq, dk, dv = module._flash_attn_bwd(
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp.contiguous(),
        softmax_scale=scale,
        causal=is_causal,
        cu_seqlens_q=cu_seq_q,
        cu_seqlens_k=cu_seq_k,
    )
    return dq, dk, dv


def _fa4_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    *,
    scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
):
    error = _fa4_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        alibi_slopes,
        seqused_k,
        cum_seq_q,
    )
    if error is not None:
        raise RuntimeError(f"FA4 flash_attention forward unsupported: {error}")
    out, lse = _fa4_run_forward(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        scale,
        is_causal,
        window_size_left,
        window_size_right,
        seqused_k,
    )
    rng_state = torch.zeros((2,), dtype=torch.uint64, device=query.device)
    philox_offset = torch.zeros((), dtype=torch.uint64, device=query.device)
    debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)
    return out, lse, rng_state, philox_offset, debug_mask


def _fa4_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    rng_state: torch.Tensor,
    unused: torch.Tensor,
    *,
    scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
):
    error = _fa4_backward_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        cum_seq_q,
        window_size_left,
        window_size_right,
    )
    if error is not None:
        raise RuntimeError(f"FA4 flash_attention backward unsupported: {error}")
    dq, dk, dv = _fa4_run_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        scale,
        is_causal,
    )
    return dq, dk, dv


def _fa4_scaled_dot_product_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
):
    error = _fa4_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        None,
        None,
        None,
    )
    if error is not None:
        raise RuntimeError(f"FA4 SDPA forward unsupported: {error}")
    q, k, v = _transpose_dense(query, key, value)

    max_q_flash = q.size(1)
    max_k_flash = k.size(1)
    out, lse, rng_state, philox_offset, debug_mask = _fa4_flash_attention_forward_impl(
        q,
        k,
        v,
        None,
        None,
        max_q_flash,
        max_k_flash,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
    )
    (out,) = _transpose_dense(out)
    max_q = query.size(2)
    max_k = key.size(2)
    return (
        out,
        lse,
        None,
        None,
        max_q,
        max_k,
        rng_state,
        philox_offset,
        debug_mask,
    )


def _fa4_scaled_dot_product_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: float | None = None,
):
    error = _fa4_backward_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        None,
        None,
        None,
    )
    if error is not None:
        raise RuntimeError(f"FA4 SDPA backward unsupported: {error}")
    q, k, v, o, go = _transpose_dense(query, key, value, out, grad_out)
    max_q = query.size(2)
    max_k = key.size(2)
    dq, dk, dv = _fa4_flash_attention_backward_impl(
        go,
        q,
        k,
        v,
        o,
        logsumexp,
        None,
        None,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )
    dq, dk, dv = _transpose_dense(dq, dk, dv)
    return dq, dk, dv


_registry.register_flash_attention_impl("FA4", register_fn=register_flash_attention_fa4)

```



## High-Level Overview

"""UBER PROTOTYPE!!!"""# mypy: allow-untyped-defsfrom __future__ import annotationsimport importlibfrom dataclasses import dataclassfrom functools import cachefrom typing import Any, TYPE_CHECKINGfrom typing_extensions import TypeVarTuple, Unpackfrom . import _registryif TYPE_CHECKING:    from types import ModuleTypeimport torchfrom torch.library import Library__all__ = [    "register_flash_attention_fa4",]_FA4_MODULE_PATH: str | None = None@dataclassclass _FA4Handle:    library: Library | None    def remove(self) -> None:        self.library = None@cachedef _get_device_major(device: torch.device) -> int:    major, _ = torch.cuda.get_device_capability(device)    return majordef register_flash_attention_fa4(    module_path: str = "flash_attn.cute.interface",) -> _FA4Handle:

This Python file contains 2 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_FA4Handle`

**Functions defined**: `remove`, `_get_device_major`, `register_flash_attention_fa4`, `_fa4_import_module`, `_fa4_register_kernels`, `_fa4_common_support_error`, `_fa4_forward_support_error`, `_fa4_backward_support_error`, `_transpose_dense`, `_fa4_run_forward`, `_fa4_run_backward`, `_fa4_flash_attention_forward_impl`, `_fa4_flash_attention_backward_impl`, `_fa4_scaled_dot_product_flash_attention_forward_impl`, `_fa4_scaled_dot_product_flash_attention_backward_impl`

**Key imports**: annotations, importlib, dataclass, cache, Any, TYPE_CHECKING, TypeVarTuple, Unpack, _registry, ModuleType, torch, Library


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/attention`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `importlib`
- `dataclasses`: dataclass
- `functools`: cache
- `typing`: Any, TYPE_CHECKING
- `typing_extensions`: TypeVarTuple, Unpack
- `.`: _registry
- `types`: ModuleType
- `torch`
- `torch.library`: Library


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
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`bias.py_docs.md`](./bias.py_docs.md)
- [`flex_attention.py_docs.md`](./flex_attention.py_docs.md)
- [`varlen.py_docs.md`](./varlen.py_docs.md)


## Cross-References

- **File Documentation**: `_fa4.py_docs.md`
- **Keyword Index**: `_fa4.py_kw.md`
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
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`_registry.py_kw.md_docs.md`](./_registry.py_kw.md_docs.md)
- [`varlen.py_kw.md_docs.md`](./varlen.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_fa4.py_docs.md_docs.md`
- **Keyword Index**: `_fa4.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
