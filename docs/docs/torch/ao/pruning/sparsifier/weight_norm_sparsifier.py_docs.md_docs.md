# Documentation: `docs/torch/ao/pruning/sparsifier/weight_norm_sparsifier.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/pruning/sparsifier/weight_norm_sparsifier.py_docs.md`
- **Size**: 13,353 bytes (13.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/pruning/sparsifier/weight_norm_sparsifier.py`

## File Metadata

- **Path**: `torch/ao/pruning/sparsifier/weight_norm_sparsifier.py`
- **Size**: 9,405 bytes (9.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import operator
from collections.abc import Callable
from functools import reduce

import torch
import torch.nn.functional as F

from .base_sparsifier import BaseSparsifier


__all__ = ["WeightNormSparsifier"]


def _flat_idx_to_2d(idx, shape):
    rows = idx // shape[1]
    cols = idx % shape[1]
    return rows, cols


class WeightNormSparsifier(BaseSparsifier):
    r"""Weight-Norm Sparsifier

    This sparsifier computes the norm of every sparse block and "zeroes-out" the
    ones with the lowest norm. The level of sparsity defines how many of the
    blocks is removed.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out
    2. `sparse_block_shape` defines the shape of the sparse blocks. Note that
        the sparse blocks originate at the zero-index of the tensor.
    3. `zeros_per_block` is the number of zeros that we are expecting in each
        sparse block. By default we assume that all elements within a block are
        zeroed-out. However, setting this variable sets the target number of
        zeros per block. The zeros within each block are chosen as the *smallest
        absolute values*.

    Args:

        sparsity_level: The target level of sparsity
        sparse_block_shape: The shape of a sparse block (see note below)
        zeros_per_block: Number of zeros in a sparse block
        norm: Norm to use. Could be either `int` or a callable.
            If `int`, only L1 and L2 are implemented.

    Note::
        The `sparse_block_shape` is tuple representing (block_ROWS, block_COLS),
        irrespective of what the rows / cols mean in the data tensor. That means,
        if you were to sparsify a weight tensor in the nn.Linear, which has a
        weight shape `(Cout, Cin)`, the `block_ROWS` would refer to the output
        channels, while the `block_COLS` would refer to the input channels.

    Note::
        All arguments to the WeightNormSparsifier constructor are "default"
        arguments and could be overridden by the configuration provided in the
        `prepare` step.
    """

    def __init__(
        self,
        sparsity_level: float = 0.5,
        sparse_block_shape: tuple[int, int] = (1, 4),
        zeros_per_block: int | None = None,
        norm: Callable | int | None = None,
    ):
        if zeros_per_block is None:
            zeros_per_block = reduce(operator.mul, sparse_block_shape)
        defaults = {
            "sparsity_level": sparsity_level,
            "sparse_block_shape": sparse_block_shape,
            "zeros_per_block": zeros_per_block,
        }
        if norm is None:
            norm = 2
        if callable(norm):
            self.norm_fn = norm
        elif norm == 1:
            self.norm_fn = lambda T: T.abs()
        elif norm == 2:
            self.norm_fn = lambda T: T * T
        else:
            raise NotImplementedError(f"L-{norm} is not yet implemented.")
        super().__init__(defaults=defaults)

    def _scatter_fold_block_mask(
        self,
        output_shape,
        dim,
        indices,
        block_shape,
        mask=None,
        input_shape=None,
        device=None,
    ):
        r"""Creates patches of size `block_shape` after scattering the indices."""
        if mask is None:
            if input_shape is None:
                raise AssertionError("input_shape must be provided when mask is None")
            mask = torch.ones(input_shape, device=device)
        mask.scatter_(dim=dim, index=indices, value=0)
        mask.data = F.fold(
            mask, output_size=output_shape, kernel_size=block_shape, stride=block_shape
        )
        return mask

    def _make_tensor_mask(
        self, data, input_shape, sparsity_level, sparse_block_shape, mask=None
    ):
        r"""Creates a tensor-level mask.

        Tensor-level mask is described as a mask, where the granularity of sparsification of the
        smallest patch is the sparse_block_shape. That means, that for a given mask and a
        sparse_block_shape, the smallest "patch" of zeros/ones could be the sparse_block_shape.

        In this context, `sparsity_level` describes the fraction of sparse patches.
        """
        h, w = data.shape[-2:]
        block_h, block_w = sparse_block_shape
        dh = (block_h - h % block_h) % block_h
        dw = (block_w - w % block_w) % block_w

        if mask is None:
            mask = torch.ones(h + dh, w + dw, device=data.device)

        if sparsity_level >= 1.0:
            mask.data = torch.zeros_like(mask)
            return mask
        elif sparsity_level <= 0.0:
            mask.data = torch.ones_like(mask)
            return mask

        values_per_block = reduce(operator.mul, sparse_block_shape)
        if values_per_block > 1:
            # Reduce the data
            data = F.avg_pool2d(
                data[None, None, :],
                kernel_size=sparse_block_shape,
                stride=sparse_block_shape,
                ceil_mode=True,
            )
        data = data.flatten()
        num_blocks = len(data)

        data = data.repeat(1, values_per_block, 1)

        threshold_idx = round(sparsity_level * num_blocks)
        threshold_idx = max(0, min(num_blocks - 1, threshold_idx))  # Sanity check
        _, sorted_idx = torch.topk(data, k=threshold_idx, dim=2, largest=False)

        # Temp reshape for mask
        mask_reshape = mask.reshape(data.shape)  # data might be reshaped
        self._scatter_fold_block_mask(
            dim=2,
            output_shape=(h + dh, w + dw),
            indices=sorted_idx,
            block_shape=sparse_block_shape,
            mask=mask_reshape,
        )
        mask.data = mask_reshape.squeeze().reshape(mask.shape)[:h, :w].contiguous()
        return mask

    def _make_block_mask(self, data, sparse_block_shape, zeros_per_block, mask=None):
        r"""Creates a block-level mask.

        Block-level mask is described as a mask, where the granularity of sparsification of the
        largest patch is the sparse_block_shape. That means that for a given mask and a
        sparse_block_shape, the sparsity is computed only within a patch of a size sparse_block_shape.

        In this context the `zeros_per_block` describes the number of zeroed-out elements within a patch.
        """
        h, w = data.shape[-2:]
        block_h, block_w = sparse_block_shape
        dh = (block_h - h % block_h) % block_h
        dw = (block_w - w % block_w) % block_w
        values_per_block = reduce(operator.mul, sparse_block_shape)

        if mask is None:
            mask = torch.ones((h + dh, w + dw), device=data.device)

        if values_per_block == zeros_per_block:
            # Everything should be sparsified
            mask.data = torch.zeros_like(mask)
            return mask

        # create a new padded tensor like data (to match the block_shape)
        padded_data = torch.ones(h + dh, w + dw, dtype=data.dtype, device=data.device)
        padded_data.fill_(torch.nan)
        padded_data[:h, :w] = data
        unfolded_data = F.unfold(
            padded_data[None, None, :],
            kernel_size=sparse_block_shape,
            stride=sparse_block_shape,
        )

        # Temp reshape for mask
        mask_reshape = mask.reshape(unfolded_data.shape)
        _, sorted_idx = torch.topk(
            unfolded_data, k=zeros_per_block, dim=1, largest=False
        )

        self._scatter_fold_block_mask(
            dim=1,
            indices=sorted_idx,
            output_shape=padded_data.shape,
            block_shape=sparse_block_shape,
            mask=mask_reshape,
        )

        mask.data = mask_reshape.squeeze().reshape(mask.shape).contiguous()
        return mask

    def update_mask(  # type: ignore[call-override, override]
        self,
        module,
        tensor_name,
        sparsity_level,
        sparse_block_shape,
        zeros_per_block,
        **kwargs,
    ):
        values_per_block = reduce(operator.mul, sparse_block_shape)
        if zeros_per_block > values_per_block:
            raise ValueError(
                "Number of zeros per block cannot be more than the total number of elements in that block."
            )
        if zeros_per_block < 0:
            raise ValueError("Number of zeros per block should be positive.")

        mask = getattr(module.parametrizations, tensor_name)[0].mask
        if sparsity_level <= 0 or zeros_per_block == 0:
            mask.data = torch.ones_like(mask)
        elif sparsity_level >= 1.0 and (zeros_per_block == values_per_block):
            mask.data = torch.zeros_like(mask)
        else:
            ww = self.norm_fn(getattr(module, tensor_name))
            tensor_mask = self._make_tensor_mask(
                data=ww,
                # pyrefly: ignore [missing-attribute]
                input_shape=ww.shape,
                sparsity_level=sparsity_level,
                sparse_block_shape=sparse_block_shape,
            )
            if values_per_block != zeros_per_block:
                block_mask = self._make_block_mask(
                    data=ww,
                    sparse_block_shape=sparse_block_shape,
                    zeros_per_block=zeros_per_block,
                )
                tensor_mask = torch.logical_or(tensor_mask, block_mask)
            mask.data = tensor_mask

```



## High-Level Overview

r"""Weight-Norm Sparsifier    This sparsifier computes the norm of every sparse block and "zeroes-out" the    ones with the lowest norm. The level of sparsity defines how many of the    blocks is removed.    This sparsifier is controlled by three variables:    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out    2. `sparse_block_shape` defines the shape of the sparse blocks. Note that        the sparse blocks originate at the zero-index of the tensor.    3. `zeros_per_block` is the number of zeros that we are expecting in each        sparse block. By default we assume that all elements within a block are        zeroed-out. However, setting this variable sets the target number of        zeros per block. The zeros within each block are chosen as the *smallest        absolute values*.    Args:        sparsity_level: The target level of sparsity        sparse_block_shape: The shape of a sparse block (see note below)        zeros_per_block: Number of zeros in a sparse block        norm: Norm to use. Could be either `int` or a callable.            If `int`, only L1 and L2 are implemented.    Note::        The `sparse_block_shape` is tuple representing (block_ROWS, block_COLS),        irrespective of what the rows / cols mean in the data tensor. That means,        if you were to sparsify a weight tensor in the nn.Linear, which has a        weight shape `(Cout, Cin)`, the `block_ROWS` would refer to the output

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WeightNormSparsifier`

**Functions defined**: `_flat_idx_to_2d`, `__init__`, `_scatter_fold_block_mask`, `_make_tensor_mask`, `_make_block_mask`, `update_mask`

**Key imports**: operator, Callable, reduce, torch, torch.nn.functional as F, BaseSparsifier


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/pruning/sparsifier`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `collections.abc`: Callable
- `functools`: reduce
- `torch`
- `torch.nn.functional as F`
- `.base_sparsifier`: BaseSparsifier


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/ao/pruning/sparsifier`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`nearly_diagonal_sparsifier.py_docs.md`](./nearly_diagonal_sparsifier.py_docs.md)
- [`base_sparsifier.py_docs.md`](./base_sparsifier.py_docs.md)


## Cross-References

- **File Documentation**: `weight_norm_sparsifier.py_docs.md`
- **Keyword Index**: `weight_norm_sparsifier.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/pruning/sparsifier`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/pruning/sparsifier`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/ao/pruning/sparsifier`):

- [`nearly_diagonal_sparsifier.py_kw.md_docs.md`](./nearly_diagonal_sparsifier.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`nearly_diagonal_sparsifier.py_docs.md_docs.md`](./nearly_diagonal_sparsifier.py_docs.md_docs.md)
- [`weight_norm_sparsifier.py_kw.md_docs.md`](./weight_norm_sparsifier.py_kw.md_docs.md)
- [`base_sparsifier.py_docs.md_docs.md`](./base_sparsifier.py_docs.md_docs.md)
- [`base_sparsifier.py_kw.md_docs.md`](./base_sparsifier.py_kw.md_docs.md)
- [`utils.py_kw.md_docs.md`](./utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `weight_norm_sparsifier.py_docs.md_docs.md`
- **Keyword Index**: `weight_norm_sparsifier.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
