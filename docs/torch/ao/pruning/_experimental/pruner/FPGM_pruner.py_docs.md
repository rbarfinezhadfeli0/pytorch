# Documentation: `torch/ao/pruning/_experimental/pruner/FPGM_pruner.py`

## File Metadata

- **Path**: `torch/ao/pruning/_experimental/pruner/FPGM_pruner.py`
- **Size**: 3,471 bytes (3.39 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections.abc import Callable

import torch

from .base_structured_sparsifier import BaseStructuredSparsifier


__all__ = ["FPGMPruner"]


class FPGMPruner(BaseStructuredSparsifier):
    r"""Filter Pruning via Geometric Median (FPGM) Structured Pruner
    This sparsifier prune filter (row) in a tensor according to distances among filters according to
    `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`_.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of filters (rows) that are zeroed-out.
    2. `dist` defines the distance measurement type. Default: 3 (L2 distance).
    Available options are: [1, 2, (custom callable distance function)].

    Note::
        Inputs should be a 4D convolutional tensor of shape (N, C, H, W).
            - N: output channels size
            - C: input channels size
            - H: height of kernel
            - W: width of kernel
    """

    def __init__(self, sparsity_level: float = 0.5, dist: Callable | int | None = None):
        defaults = {
            "sparsity_level": sparsity_level,
        }

        if dist is None:
            dist = 2

        if callable(dist):
            self.dist_fn = dist
        elif dist == 1:
            self.dist_fn = lambda x: torch.cdist(x, x, p=1)
        elif dist == 2:
            self.dist_fn = lambda x: torch.cdist(x, x, p=2)
        else:
            raise NotImplementedError("Distance function is not yet implemented.")
        super().__init__(defaults=defaults)

    def _compute_distance(self, t):
        r"""Compute distance across all entries in tensor `t` along all dimension
        except for the one identified by dim.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
        Returns:
            distance (torch.Tensor): distance computed across filtters
        """
        dim = 0  # prune filter (row)

        size = t.size(dim)
        slc = [slice(None)] * t.dim()

        # flatten the tensor along the dimension
        t_flatten = [
            t[tuple(slc[:dim] + [slice(i, i + 1)] + slc[dim + 1 :])].reshape(-1)
            for i in range(size)
        ]
        t_flatten = torch.stack(t_flatten)

        # distance measurement
        dist_matrix = self.dist_fn(t_flatten)

        # more similar with other filter indicates large in the sum of row
        # pyrefly: ignore [bad-argument-type]
        distance = torch.sum(torch.abs(dist_matrix), 1)

        return distance

    def update_mask(  # type: ignore[override]
        self, module, tensor_name, sparsity_level, **kwargs
    ):
        tensor_weight = getattr(module, tensor_name)
        mask = getattr(module.parametrizations, tensor_name)[0].mask

        if sparsity_level <= 0:
            mask.data = torch.ones_like(mask).bool()
        elif sparsity_level >= 1.0:
            mask.data = torch.zeros_like(mask).bool()
        else:
            distance = self._compute_distance(tensor_weight)

            tensor_size = tensor_weight.shape[0]  # prune filter (row)
            nparams_toprune = round(sparsity_level * tensor_size)
            nparams_toprune = min(
                max(nparams_toprune, 0), tensor_size
            )  # clamp to [0, tensor_size]
            topk = torch.topk(distance, k=nparams_toprune, largest=False)
            mask[topk.indices] = False

```



## High-Level Overview

r"""Filter Pruning via Geometric Median (FPGM) Structured Pruner    This sparsifier prune filter (row) in a tensor according to distances among filters according to    `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`_.    This sparsifier is controlled by three variables:    1. `sparsity_level` defines the number of filters (rows) that are zeroed-out.    2. `dist` defines the distance measurement type. Default: 3 (L2 distance).    Available options are: [1, 2, (custom callable distance function)].    Note::        Inputs should be a 4D convolutional tensor of shape (N, C, H, W).            - N: output channels size            - C: input channels size            - H: height of kernel            - W: width of kernel

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FPGMPruner`

**Functions defined**: `__init__`, `_compute_distance`, `update_mask`

**Key imports**: Callable, torch, BaseStructuredSparsifier


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/pruning/_experimental/pruner`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `torch`
- `.base_structured_sparsifier`: BaseStructuredSparsifier


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/ao/pruning/_experimental/pruner`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`prune_functions.py_docs.md`](./prune_functions.py_docs.md)
- [`parametrization.py_docs.md`](./parametrization.py_docs.md)
- [`base_structured_sparsifier.py_docs.md`](./base_structured_sparsifier.py_docs.md)
- [`match_utils.py_docs.md`](./match_utils.py_docs.md)
- [`lstm_saliency_pruner.py_docs.md`](./lstm_saliency_pruner.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`saliency_pruner.py_docs.md`](./saliency_pruner.py_docs.md)


## Cross-References

- **File Documentation**: `FPGM_pruner.py_docs.md`
- **Keyword Index**: `FPGM_pruner.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
