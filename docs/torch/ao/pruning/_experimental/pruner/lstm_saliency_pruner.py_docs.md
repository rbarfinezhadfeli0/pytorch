# Documentation: `torch/ao/pruning/_experimental/pruner/lstm_saliency_pruner.py`

## File Metadata

- **Path**: `torch/ao/pruning/_experimental/pruner/lstm_saliency_pruner.py`
- **Size**: 2,197 bytes (2.15 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any, cast

import torch
from torch import nn

from .base_structured_sparsifier import BaseStructuredSparsifier
from .parametrization import FakeStructuredSparsity


class LSTMSaliencyPruner(BaseStructuredSparsifier):
    """
    Prune packed LSTM weights based on saliency.
    For each layer {k} inside a LSTM, we have two packed weight matrices
    - weight_ih_l{k}
    - weight_hh_l{k}

    These tensors pack the weights for the 4 linear layers together for efficiency.

    [W_ii | W_if | W_ig | W_io]

    Pruning this tensor directly will lead to weights being misassigned when unpacked.
    To ensure that each packed linear layer is pruned the same amount:
        1. We split the packed weight into the 4 constituent linear parts
        2. Update the mask for each individual piece using saliency individually

    This applies to both weight_ih_l{k} and weight_hh_l{k}.
    """

    def update_mask(self, module: nn.Module, tensor_name: str, **kwargs: Any) -> None:
        weights = getattr(module, tensor_name)

        for p in getattr(module.parametrizations, tensor_name):
            if isinstance(p, FakeStructuredSparsity):
                mask = cast(torch.Tensor, p.mask)

                # select weights based on magnitude
                if weights.dim() <= 1:
                    raise Exception(  # noqa: TRY002
                        "Structured pruning can only be applied to a 2+dim weight tensor!"
                    )
                # take norm over all but first dim
                dims = tuple(range(1, weights.dim()))
                saliency = weights.norm(dim=dims, p=1)

                # handle weights in 4 groups
                split_size = len(mask) // 4
                masks = torch.split(mask, split_size)
                saliencies = torch.split(saliency, split_size)

                for keep_mask, sal in zip(masks, saliencies):
                    # mask smallest k values to be removed
                    k = int(len(keep_mask) * kwargs["sparsity_level"])
                    prune = sal.topk(k, largest=False, sorted=False).indices
                    keep_mask.data[prune] = False  # modifies underlying p.mask directly

```



## High-Level Overview

"""    Prune packed LSTM weights based on saliency.    For each layer {k} inside a LSTM, we have two packed weight matrices    - weight_ih_l{k}    - weight_hh_l{k}    These tensors pack the weights for the 4 linear layers together for efficiency.    [W_ii | W_if | W_ig | W_io]    Pruning this tensor directly will lead to weights being misassigned when unpacked.    To ensure that each packed linear layer is pruned the same amount:        1. We split the packed weight into the 4 constituent linear parts        2. Update the mask for each individual piece using saliency individually    This applies to both weight_ih_l{k} and weight_hh_l{k}.

This Python file contains 1 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LSTMSaliencyPruner`

**Functions defined**: `update_mask`

**Key imports**: Any, cast, torch, nn, BaseStructuredSparsifier, FakeStructuredSparsity


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/pruning/_experimental/pruner`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, cast
- `torch`
- `.base_structured_sparsifier`: BaseStructuredSparsifier
- `.parametrization`: FakeStructuredSparsity


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
- [`FPGM_pruner.py_docs.md`](./FPGM_pruner.py_docs.md)
- [`base_structured_sparsifier.py_docs.md`](./base_structured_sparsifier.py_docs.md)
- [`match_utils.py_docs.md`](./match_utils.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`saliency_pruner.py_docs.md`](./saliency_pruner.py_docs.md)


## Cross-References

- **File Documentation**: `lstm_saliency_pruner.py_docs.md`
- **Keyword Index**: `lstm_saliency_pruner.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
