# Documentation: `docs/torch/nn/attention/experimental/_paged_attention.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/attention/experimental/_paged_attention.py_docs.md`
- **Size**: 15,384 bytes (15.02 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/attention/experimental/_paged_attention.py`

## File Metadata

- **Path**: `torch/nn/attention/experimental/_paged_attention.py`
- **Size**: 13,026 bytes (12.72 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
"""
This module implements Paged Attention on top of flex_attention.
This module is experimental and subject to change.
"""

import torch
from torch.nn.attention.flex_attention import (
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
    BlockMask,
    noop_mask,
)


__all__ = ["PagedAttention"]


def _cdiv(x: int | float | torch.Tensor, multiple: int | float | torch.Tensor):
    return (x + multiple - 1) // multiple


class PagedAttention:
    """
    PagedAttention supports flex attention inference with a large batch size.
    With PagedAttention, a batch of key/value tensors with varying kv length
    is split into tensor blocks of fixed length and cached in a compact way.
    Thus we can avoid redundant memory consumption due to varying kv length and
    support a larger batch size.
    """

    def __init__(
        self,
        n_pages: int,
        page_size: int,
        max_batch_size: int,
        device: str = "cuda",
    ) -> None:
        # number of pages
        self.n_pages = n_pages

        # number of tokens per page
        self.page_size = page_size

        # page table: [batch, logical_block_idx] -> physical_page_idx
        self.page_table = -torch.ones(
            (max_batch_size, self.n_pages), dtype=torch.int64, device=device
        )

        # capacity: batch_idx -> allocated sequence length
        self.capacity = torch.zeros(max_batch_size, dtype=torch.int64, device=device)

        # index of empty pages that is available for allocation
        self.empty_pages = list(range(n_pages - 1, -1, -1))

        # mapping from physical page index to logical page index
        self.physical_to_logical = -torch.ones(
            (max_batch_size, n_pages), dtype=torch.int64, device=device
        )

    def reserve(self, batch_idx: torch.Tensor, seq_len: torch.Tensor) -> None:
        """
        Requests the capacity of a given batch to be at least enough to
        hold `seq_len` elements.

        Args:
            batch_idx (Tensor): batch index to be reserved; shape :math:`(1)`.
            seq_len (Tensor): minimum capacity for the given batch; shape :math:`(1)`.
        """

        if seq_len <= self.capacity[batch_idx]:
            return

        num_pages_to_allocate = _cdiv(
            seq_len - self.capacity[batch_idx], self.page_size
        )

        assert len(self.empty_pages) >= num_pages_to_allocate, (
            f"requested {num_pages_to_allocate.item()} pages "
            f"but there are only {len(self.empty_pages)} empty pages"
        )

        start_page_idx = self.capacity[batch_idx] // self.page_size
        end_page_idx = start_page_idx + num_pages_to_allocate

        # find empty physical pages
        allocated_pages = torch.tensor(
            self.empty_pages[-num_pages_to_allocate:],
            device=num_pages_to_allocate.device,
        )
        self.empty_pages = self.empty_pages[:-num_pages_to_allocate]

        # update page table
        self.page_table[
            batch_idx,
            start_page_idx:end_page_idx,
        ] = allocated_pages

        # update metadata
        self.physical_to_logical[batch_idx, allocated_pages] = torch.arange(
            start_page_idx.item(),
            end_page_idx.item(),
            device=num_pages_to_allocate.device,
        )
        self.capacity[batch_idx] += num_pages_to_allocate * self.page_size

    def erase(self, batch_idx: torch.Tensor) -> None:
        """
        Removes a single batch from paged attention.

        Args:
            batch_idx (Tensor): batch index to be removed; shape :math:`(1)`.
        """

        # find allocated pages
        allocated_page_idx = self.page_table[batch_idx] != -1
        allocated_pages = self.page_table[batch_idx][allocated_page_idx]

        # clean metadata
        self.capacity[batch_idx] = 0
        self.empty_pages += allocated_pages.tolist()
        self.physical_to_logical[batch_idx][:, allocated_pages] = -1
        self.page_table[batch_idx] = -1

    def assign(
        self,
        batch_idx: torch.Tensor,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        """
        Assigns new contents `val` to the storage `cache` at the location
        `batch_idx` and `input_pos`.

        Args:
            batch_idx (Tensor): batch index; shape :math:`(B)`.
            input_pos (Tensor): input positions to be assigned for the given batch; shape :math:`(B, S)`.
            val (Tensor): value to be assigned; shape :math:`(B, H, S, D)`
            cache (Tensor): the cache to store the values; shape:`(1, H, MAX_S, D)`
        """
        if k_val.requires_grad:
            raise RuntimeError("val must not require gradient")

        B, H, S, K_D = k_val.shape
        V_D = v_val.shape[3]
        if B != batch_idx.shape[0]:
            raise RuntimeError(
                f"Expect val and batch_idx have the same batch size "
                f"but got B={B} and B={batch_idx.shape[0]}."
            )
        if H != k_cache.shape[1]:
            raise RuntimeError(
                f"Expect val and cache has the same number of heads "
                f"but got H={H} and H={k_cache.shape[1]}."
            )
        if S != input_pos.shape[1]:
            raise RuntimeError(
                f"Expect val and input_pos has the same length "
                f"but got S={S} and S={input_pos.shape[0]}."
            )
        if K_D != k_cache.shape[3]:
            raise RuntimeError(
                f"Expect k_val and k_cache has the same hidden dim "
                f"but got D={K_D} and D={k_cache.shape[3]}."
            )
        if V_D != v_cache.shape[3]:
            raise RuntimeError(
                f"Expect v_val and v_cache has the same hidden dim "
                f"but got D={V_D} and D={v_cache.shape[3]}."
            )

        # find address
        logical_block_idx = input_pos // self.page_size  # [B, S]
        logical_block_offset = input_pos % self.page_size  # [B, S]
        physical_block_idx = torch.gather(
            self.page_table[batch_idx], 1, logical_block_idx.to(torch.int64)
        ).to(torch.int32)  # [B, S]

        addr = (physical_block_idx * self.page_size + logical_block_offset).view(
            -1
        )  # [B*S]

        k_val = k_val.permute(1, 0, 2, 3).contiguous().view(1, H, B * S, K_D)
        v_val = v_val.permute(1, 0, 2, 3).contiguous().view(1, H, B * S, V_D)

        k_cache[:, :, addr, :] = k_val
        v_cache[:, :, addr, :] = v_val

    def convert_logical_block_mask(
        self,
        block_mask: BlockMask,
        batch_idx: torch.Tensor | None = None,
        kv_len: torch.Tensor | None = None,
    ) -> BlockMask:
        """
        Converts a logical block mask by mapping its logical kv indices to the corresponding
        physical kv indices.

        Args:
            block_mask (BlockMask): logical block mask;
                kv_indices shape :math:`(B, H, ROWS, MAX_BLOCKS_IN_COL)`.
            batch_idx (Tensor): batch index corresponding to the block_mask
                batch dimension. This provides flexibility to convert a
                block mask with smaller batch size than the page table;
                shape :math:`(B)`.
            kv_len (Optional[Tensor]): actual KV sequence length for upper bound check;
                shape :math:`(B,)` to handle multiple batches.
        """
        B, H, ROWS, MAX_BLOCKS_IN_COL = block_mask.kv_indices.shape

        if block_mask.BLOCK_SIZE[1] != self.page_size:
            raise RuntimeError(
                f"Expect block_mask has the same column block size as page_size"
                f"but got size={block_mask.BLOCK_SIZE[1]} and size={self.page_size}"
            )

        # Increase the num columns of converted block mask from logical block mask's
        # num columns to n_pages, since a) the converted block mask
        # may have larger indices values; and b) `_ordered_to_dense` realizes
        # a dense tensor with these converted indices. There would be an IndexError
        # if using the logical block mask's num columns.

        device = block_mask.kv_num_blocks.device

        if batch_idx is None:
            batch_idx = torch.arange(B, device=device)
        page_table = self.page_table[batch_idx]

        new_kv_num_blocks = block_mask.kv_num_blocks.clone()

        new_kv_indices = torch.zeros(
            (B, H, ROWS, self.n_pages), dtype=torch.int32, device=device
        )
        new_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
            torch.gather(
                page_table, 1, block_mask.kv_indices.view(B, -1).to(torch.int64)
            )
            .view(block_mask.kv_indices.shape)
            .to(torch.int32)
        )

        new_full_kv_indices, new_full_kv_num_blocks = None, None
        if block_mask.full_kv_num_blocks is not None:
            assert block_mask.full_kv_indices is not None
            new_full_kv_num_blocks = block_mask.full_kv_num_blocks.clone()
            new_full_kv_indices = torch.zeros(
                (B, H, ROWS, self.n_pages), dtype=torch.int32, device=device
            )
            new_full_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
                torch.gather(
                    page_table,
                    1,
                    block_mask.full_kv_indices.view(B, -1).to(torch.int64),
                )
                .view(block_mask.full_kv_indices.shape)
                .to(torch.int32)
            )

        new_mask_mod = self.get_mask_mod(block_mask.mask_mod, kv_len)

        seq_lengths = (block_mask.seq_lengths[0], self.n_pages * self.page_size)
        return BlockMask.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            block_mask.BLOCK_SIZE,
            new_mask_mod,
            seq_lengths=seq_lengths,
        )

    def get_mask_mod(
        self,
        mask_mod: _mask_mod_signature | None,
        kv_len: torch.Tensor | None = None,
    ) -> _mask_mod_signature:
        """
        Converts a mask_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            mask_mod (_mask_mod_signature): mask_mod based on the logical block index.
            kv_len (Optional[torch.Tensor]): actual KV sequence length for upper bound check.
        """
        if mask_mod is None:
            mask_mod = noop_mask

        def new_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            physical_kv_block = physical_kv_idx // self.page_size
            physical_kv_offset = physical_kv_idx % self.page_size
            logical_block_idx = self.physical_to_logical[b, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
            live_block = logical_block_idx >= 0
            within_upper_bound = (
                logical_kv_idx < kv_len[b] if kv_len is not None else True
            )
            within_lower_bound = logical_kv_idx >= 0
            is_valid = live_block & within_upper_bound & within_lower_bound

            return torch.where(is_valid, mask_mod(b, h, q_idx, logical_kv_idx), False)

        return new_mask_mod

    def get_score_mod(
        self,
        score_mod: _score_mod_signature | None,
        kv_len: torch.Tensor | None = None,
    ) -> _score_mod_signature:
        """
        Converts a score_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            score_mod (_score_mod_signature): score_mod based on the logical block index.
            `kv_len (Optional[torch.Tensor]): actual KV sequence length for upper bound check.

        """
        if score_mod is None:
            score_mod = _identity

        def new_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            physical_kv_block = physical_kv_idx // self.page_size
            physical_kv_offset = physical_kv_idx % self.page_size
            logical_block_idx = self.physical_to_logical[b, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
            live_block = logical_block_idx >= 0
            within_upper_bound = (
                logical_kv_idx < kv_len[b] if kv_len is not None else True
            )
            within_lower_bound = logical_kv_idx >= 0
            is_valid = live_block & within_upper_bound & within_lower_bound

            return torch.where(
                is_valid,
                score_mod(score, b, h, q_idx, logical_kv_idx),
                float("-inf"),
            )

        return new_score_mod

```



## High-Level Overview

"""This module implements Paged Attention on top of flex_attention.This module is experimental and subject to change.

This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PagedAttention`

**Functions defined**: `_cdiv`, `__init__`, `reserve`, `erase`, `assign`, `convert_logical_block_mask`, `get_mask_mod`, `new_mask_mod`, `get_score_mod`, `new_score_mod`

**Key imports**: torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/attention/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`torch/nn/attention/experimental`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `_paged_attention.py_docs.md`
- **Keyword Index**: `_paged_attention.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn/attention/experimental`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/attention/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/nn/attention/experimental`):

- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`_paged_attention.py_kw.md_docs.md`](./_paged_attention.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_paged_attention.py_docs.md_docs.md`
- **Keyword Index**: `_paged_attention.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
