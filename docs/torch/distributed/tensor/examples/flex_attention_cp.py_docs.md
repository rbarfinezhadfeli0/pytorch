# Documentation: `torch/distributed/tensor/examples/flex_attention_cp.py`

## File Metadata

- **Path**: `torch/distributed/tensor/examples/flex_attention_cp.py`
- **Size**: 5,625 bytes (5.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
"""
To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=4 flex_attention_cp.py
"""

import os
from functools import lru_cache
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Partial, Shard
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)


def get_device_type() -> str:
    return "cuda"


@lru_cache
def create_block_mask_cached(
    score_mod: _mask_mod_signature,
    B: Optional[int],
    H: Optional[int],
    M: int,
    N: int,
    device: str = "cuda",
) -> BlockMask:
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def flex_attn_example(world_size: int, rank: int) -> None:
    device_type = get_device_type()
    device_handle = getattr(torch, device_type, None)
    assert device_handle is not None, f"Unsupported device type: {device_type}"
    num_devices_per_host = device_handle.device_count()
    device_handle.set_device(rank % num_devices_per_host)
    torch._dynamo.config.cache_size_limit = 1000

    # init device mesh
    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(world_size,),
        mesh_dim_names=("cp",),
    )

    def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        return q_idx >= kv_idx

    # Compile the flex_attention function
    compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

    # init input
    torch.manual_seed(10)
    dtype = torch.float32
    B = 8
    H = 8
    S = 32 * world_size
    D = 32

    qkv = [
        torch.rand(
            (B, H, S, D),
            device=device_type,
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(3)
    ]

    # input distribution
    seq_dim = 2
    qkv_dist = [
        distribute_tensor(
            t.detach().clone().requires_grad_(), device_mesh, [Shard(seq_dim)]
        )
        for t in qkv
    ]

    # local forward pass
    block_mask = create_block_mask_cached(
        causal_mask,
        B=1,
        H=1,
        M=S,
        N=S,
        device=device_type,
    )

    q, k, v = qkv
    out = compiled_flex_attention(q, k, v, score_mod=None, block_mask=block_mask)
    assert isinstance(out, torch.Tensor)
    expect_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, expect_out, atol=1e-1, rtol=1e-2)

    # context parallel forward pass
    def rewrite_mask_mod_for_cp(
        mask_mod: _mask_mod_signature,
        rank: int,
        shard_size: int,
    ) -> _mask_mod_signature:
        # since we're sharding on `seq_dim`, global q_idx is mapped to q_idx % shard_size
        # on each rank which means q_idx = q_idx_on_rank + shard_size * rank
        return lambda b, h, q_idx, kv_idx: mask_mod(
            b, h, q_idx + rank * shard_size, kv_idx
        )

    # manually do context parallel on attention
    # the input hook of Context Parallel
    q_local = qkv_dist[0].to_local()

    # kv all-gather
    # NOTE: we don't consider load-balance for now
    # NOTE: wait() is immediately called in all_gather_tensor when gather_dim != 0
    k_full, v_full = (t.full_tensor(grad_placements=[Partial()]) for t in qkv_dist[1:])

    # rewrite `block_mask`
    mask_mod: _mask_mod_signature = block_mask.mask_mod
    shard_size = S // world_size
    cp_mask_mod = rewrite_mask_mod_for_cp(mask_mod, rank, shard_size)
    cp_block_mask = create_block_mask_cached(
        cp_mask_mod, B=1, H=1, M=shard_size, N=S, device=device_type
    )

    # TODO: this doesn't address the return_lse=True case
    cp_out = compiled_flex_attention(
        q_local,
        k_full,
        v_full,
        score_mod=None,
        block_mask=cp_block_mask,
    )
    assert isinstance(cp_out, torch.Tensor)

    # wrap the local output into a DTensor
    cp_out_dist = DTensor.from_local(cp_out, device_mesh, [Shard(seq_dim)])
    # compare with the flex_attention output
    torch.testing.assert_close(cp_out_dist.full_tensor(), out, atol=1e-1, rtol=1e-2)

    # local backward pass
    grad_out = torch.randn(
        (B, H, S, D),
        device=device_type,
        dtype=dtype,
    )
    grad_out_dist = distribute_tensor(
        grad_out.detach().clone().requires_grad_(), device_mesh, [Shard(seq_dim)]
    )

    out.backward(grad_out)
    grad1 = [t.grad for t in qkv]
    for t in qkv:
        t.grad = None

    expect_out.backward(grad_out)
    grad2 = [t.grad for t in qkv]
    for t in qkv:
        t.grad = None

    for flex_grad, expect_grad in zip(grad1, grad2):
        torch.testing.assert_close(flex_grad, expect_grad, atol=1e-1, rtol=1e-2)

    # context parallel backward pass
    cp_out.backward(grad_out_dist.to_local())

    for cp_flex_grad_dist, expect_grad in zip([t.grad for t in qkv_dist], grad2):
        assert isinstance(cp_flex_grad_dist, DTensor)
        torch.testing.assert_close(
            cp_flex_grad_dist.full_tensor(), expect_grad, atol=1e-1, rtol=1e-2
        )


if __name__ == "__main__":
    # this script is launched via torchrun which automatically manages ProcessGroup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # assert world_size == 4  # our example uses 4 worker ranks

    try:
        flex_attn_example(world_size, rank)
    finally:
        dist.barrier()
        dist.destroy_process_group()

```



## High-Level Overview

"""To run the example, use the following command:torchrun --standalone --nnodes=1 --nproc-per-node=4 flex_attention_cp.py

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_device_type`, `create_block_mask_cached`, `flex_attn_example`, `causal_mask`, `rewrite_mask_mod_for_cp`

**Key imports**: os, lru_cache, Optional, torch, torch.distributed as dist, torch.nn.functional as F, init_device_mesh, distribute_tensor, DTensor, Partial, Shard


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor/examples`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `functools`: lru_cache
- `typing`: Optional
- `torch`
- `torch.distributed as dist`
- `torch.nn.functional as F`
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.tensor`: distribute_tensor, DTensor, Partial, Shard


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/distributed/tensor/examples`):

- [`visualize_sharding_example.py_docs.md`](./visualize_sharding_example.py_docs.md)
- [`comm_mode_features_example.py_docs.md`](./comm_mode_features_example.py_docs.md)
- [`torchrec_sharding_example.py_docs.md`](./torchrec_sharding_example.py_docs.md)
- [`convnext_example.py_docs.md`](./convnext_example.py_docs.md)


## Cross-References

- **File Documentation**: `flex_attention_cp.py_docs.md`
- **Keyword Index**: `flex_attention_cp.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
