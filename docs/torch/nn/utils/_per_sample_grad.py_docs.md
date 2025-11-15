# Documentation: `torch/nn/utils/_per_sample_grad.py`

## File Metadata

- **Path**: `torch/nn/utils/_per_sample_grad.py`
- **Size**: 5,775 bytes (5.64 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import functools

import torch
from torch.nn.utils._expanded_weights.expanded_weights_impl import ExpandedWeight
from torch.utils import _pytree as pytree


# dependency on `functional_call` means that this can't be exposed in utils
# without creating circular dependency
def call_for_per_sample_grads(
    module,
    *,
    batch_size=None,
    loss_reduction="sum",
    batch_first=True,
):
    r"""
    Return a forward function for a module, populating grad_sample with per sample gradients on backward invocation.

    Args:
        module: The ``nn.Module`` to get per sample gradients with respect to. All trainable
          parameters will compute per sample gradients, located in a ``grad_sample``
          field when ``backward`` is invoked
        batch_size: The batch size of the input. If None is passed, all tensor arguments in args and kwargs must have
          the same batch size, which is the size of the first dimension. Otherwise, it must be passed manually.
          Default: None
        loss_reduction: Indicates if the loss reduction (for aggregating the gradients) is a sum or a mean operation. If
          "mean", per sample gradients will be scaled by the batch size to offset the crossbatch interaction from
          running mean across a batch. Must be "mean" or "sum". Default: "sum"
        batch_first: Indicates if the batch dimension is the first dimension. If True, the batch dimension is the first
          dimension. If False, it's the second dimension. Default: True.

    Examples::
        >>> # xdoctest: +SKIP
        >>> model = nn.Linear(4, 3)
        >>> batched_input = torch.randn(5, 4)  # batch size of 5
        >>> res = call_for_per_sample_grads(model)(batched_input).sum()
        >>> res.backward()
        >>> assert model.weight.shape == (3, 4)
        >>> assert model.weight.grad_sample.shape == (5, 3, 4)
        >>> assert model.weight.grad is None
        >>> assert model.bias.shape == (3,)
        >>> assert model.bias.grad_sample.shape == (5, 3)
        >>> assert model.bias.grad is None

    An example using "mean" loss reduction. The grad_sample fields will be scaled by batch_size from what they would be
    if we ran the same code with loss_reduction="sum". This is because the mean at the end will scale all
    grad_outputs by 1 / batch_size from cross batch interaction.
        >>> model = nn.Linear(4, 3)
        >>> batched_input = torch.randn(5, 4)  # batch size of 5
        >>> res = call_for_per_sample_grads(model, 5, loss_reduction="mean")(
        ...     batched_input
        ... ).mean()
        >>> res.backward()

    Note::
        Does not work with any `nn.RNN`, including `nn.GRU` or `nn.LSTM`. Please use custom
        rewrites that wrap an `nn.Linear` module. See Opacus for an example
    """

    def maybe_build_expanded_weight(og_tensor, batch_size):
        if og_tensor.requires_grad:
            return ExpandedWeight(og_tensor, batch_size, loss_reduction)
        else:
            return og_tensor

    def compute_batch_size(*args, **kwargs):
        args_and_kwargs = pytree.arg_tree_leaves(*args, **kwargs)
        batch_size = None
        for arg in args_and_kwargs:
            if not isinstance(arg, torch.Tensor):
                continue

            arg_batch_size = arg.shape[0] if batch_first else arg.shape[1]
            if batch_size is not None and batch_size != arg_batch_size:
                raise RuntimeError(
                    "When computing batch size, found at least one input with batch size "
                    f"{batch_size} and one with batch size {arg_batch_size}. Please specify it "
                    "explicitly using the batch size kwarg in call_for_per_sample_grads"
                )
            batch_size = arg_batch_size
        if batch_size is None:
            raise RuntimeError(
                "Unable to find a tensor in the passed args and kwargs. They may not be pytree-able "
                "and so ExpandedWeights cannot compute the batch size from the inputs. Please specify "
                "it explicitly"
            )
        return batch_size

    if loss_reduction not in ["sum", "mean"]:
        raise RuntimeError(
            f"Expected loss_reduction argument to be sum or mean, got {loss_reduction}"
        )

    if not isinstance(module, torch.nn.Module):
        raise RuntimeError(
            f"Module passed must be nn.Module, got {type(module).__name__}"
        )
    if not (batch_size is None or isinstance(batch_size, int)):
        raise RuntimeError(
            f"Batch size passed must be None or an integer, got {type(batch_size).__name__}"
        )
    if batch_size is not None and batch_size < 1:
        raise RuntimeError(f"Batch size must be positive, got {batch_size}")
    for weight in module.parameters():
        if hasattr(weight, "grad_sample") and weight.grad_sample is not None:  # type: ignore[attr-defined]
            raise RuntimeError(
                "Current Expanded Weights accumulates the gradients, which will be incorrect for multiple "
                f"calls without clearing gradients. Please clear out the grad_sample parameter of {weight} or "
                "post an issue to pytorch/pytorch to prioritize correct behavior"
            )

    @functools.wraps(module.forward)
    def wrapper(*args, **kwargs):
        wrapper_batch_size = batch_size
        if wrapper_batch_size is None:
            wrapper_batch_size = compute_batch_size(*args, **kwargs)

        params = {
            name: maybe_build_expanded_weight(value, wrapper_batch_size)
            for (name, value) in module.named_parameters()
        }
        return torch.func.functional_call(module, params, args, kwargs)

    return wrapper

```



## High-Level Overview

r"""    Return a forward function for a module, populating grad_sample with per sample gradients on backward invocation.    Args:        module: The ``nn.Module`` to get per sample gradients with respect to. All trainable          parameters will compute per sample gradients, located in a ``grad_sample``          field when ``backward`` is invoked        batch_size: The batch size of the input. If None is passed, all tensor arguments in args and kwargs must have          the same batch size, which is the size of the first dimension. Otherwise, it must be passed manually.          Default: None        loss_reduction: Indicates if the loss reduction (for aggregating the gradients) is a sum or a mean operation. If          "mean", per sample gradients will be scaled by the batch size to offset the crossbatch interaction from          running mean across a batch. Must be "mean" or "sum". Default: "sum"        batch_first: Indicates if the batch dimension is the first dimension. If True, the batch dimension is the first          dimension. If False, it's the second dimension. Default: True.    Examples::        >>> # xdoctest: +SKIP        >>> model = nn.Linear(4, 3)        >>> batched_input = torch.randn(5, 4)  # batch size of 5        >>> res = call_for_per_sample_grads(model)(batched_input).sum()        >>> res.backward()        >>> assert model.weight.shape == (3, 4)        >>> assert model.weight.grad_sample.shape == (5, 3, 4)        >>> assert model.weight.grad is None        >>> assert model.bias.shape == (3,)        >>> assert model.bias.grad_sample.shape == (5, 3)        >>> assert model.bias.grad is None    An example using "mean" loss reduction. The grad_sample fields will be scaled by batch_size from what they would be    if we ran the same code with loss_reduction="sum". This is because the mean at the end will scale all    grad_outputs by 1 / batch_size from cross batch interaction.        >>> model = nn.Linear(4, 3)

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `call_for_per_sample_grads`, `maybe_build_expanded_weight`, `compute_batch_size`, `wrapper`

**Key imports**: functools, torch, ExpandedWeight, _pytree as pytree


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `torch`
- `torch.nn.utils._expanded_weights.expanded_weights_impl`: ExpandedWeight
- `torch.utils`: _pytree as pytree


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`torch/nn/utils`):

- [`_deprecation_utils.py_docs.md`](./_deprecation_utils.py_docs.md)
- [`parametrizations.py_docs.md`](./parametrizations.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`stateless.py_docs.md`](./stateless.py_docs.md)
- [`parametrize.py_docs.md`](./parametrize.py_docs.md)
- [`spectral_norm.py_docs.md`](./spectral_norm.py_docs.md)
- [`prune.py_docs.md`](./prune.py_docs.md)
- [`fusion.py_docs.md`](./fusion.py_docs.md)
- [`weight_norm.py_docs.md`](./weight_norm.py_docs.md)


## Cross-References

- **File Documentation**: `_per_sample_grad.py_docs.md`
- **Keyword Index**: `_per_sample_grad.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
