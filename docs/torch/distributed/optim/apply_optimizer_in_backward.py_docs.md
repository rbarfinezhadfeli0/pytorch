# Documentation: `torch/distributed/optim/apply_optimizer_in_backward.py`

## File Metadata

- **Path**: `torch/distributed/optim/apply_optimizer_in_backward.py`
- **Size**: 5,214 bytes (5.09 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Iterable
from typing import Any, no_type_check

import torch


__all__: list[str] = []

# WeakTensorKeyDictionary to store relevant meta-data for the Tensor/Parameter
# without changing it's life-time.
# NOTE: Alternative is to add the meta-data as an attribute to the tensor,
#       but that will serialize the meta-data if Tensor is serialized.
param_to_optim_hook_handle_map = torch.utils.weak.WeakTensorKeyDictionary()
param_to_acc_grad_map = torch.utils.weak.WeakTensorKeyDictionary()


@no_type_check
def _apply_optimizer_in_backward(
    optimizer_class: type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: dict[str, Any],
    register_hook: bool = True,
) -> None:
    """
    Upon ``backward()``, the optimizer specified for each parameter will fire after
    the gradient has been accumulated into the parameter.

    Note - gradients for these parameters will be set to None after ``backward()``.
    This means that any other optimizer not specified via `_apply_optimizer_in_backward`
    over this parameter will be a no-op.

    Args:
        optimizer_class: (Type[torch.optim.Optimizer]): Optimizer to apply to parameter
        params: (Iterator[nn.Parameter]): parameters to apply optimizer state to
        optimizer_kwargs: (Dict[str, Any]): kwargs to pass to optimizer constructor
        register_hook: (bool): whether to register a hook that runs the optimizer
            after gradient for this parameter is accumulated. This is the default
            way that optimizer in backward is implemented, but specific use cases
            (such as DDP) may wish to override this to implement custom behavior.
            (Default = True)

    Example::
        params_generator = model.parameters()
        param_1 = next(params_generator)
        remainder_params = list(params_generator)

        apply_optimizer_in_backward(torch.optim.SGD, [param_1], {"lr": 0.02})
        apply_optimizer_in_backward(torch.optim.Adam, remainder_params, {"lr": 0.04})

        model(...).sum().backward()  # after backward, parameters will already
        # have their registered optimizer(s) applied.

    """
    torch._C._log_api_usage_once("torch.distributed.optim.apply_optimizer_in_backward")

    @no_type_check
    def _apply_optimizer_in_backward_to_param(param: torch.nn.Parameter) -> None:
        # view_as creates a node in autograd graph that allows us access to the
        # parameter's AccumulateGrad autograd function object. We register a
        # hook on this object to fire the optimizer when the gradient for
        # this parameter is ready (has been accumulated into .grad field)

        # Don't create a new acc_grad if we already have one
        # i.e. for shared parameters or attaching multiple optimizers to a param.
        if param not in param_to_acc_grad_map:
            param_to_acc_grad_map[param] = param.view_as(param).grad_fn.next_functions[
                0
            ][0]

        optimizer = optimizer_class([param], **optimizer_kwargs)

        if not hasattr(param, "_in_backward_optimizers"):
            param._in_backward_optimizers = []  # type: ignore[attr-defined]
            # TODO: Remove these attributes once we have a better way of accessing
            # optimizer classes and kwargs for a parameter.
            param._optimizer_classes = []  # type: ignore[attr-defined]
            param._optimizer_kwargs = []  # type: ignore[attr-defined]

        param._in_backward_optimizers.append(optimizer)  # type: ignore[attr-defined]
        param._optimizer_classes.append(optimizer_class)  # type: ignore[attr-defined]
        param._optimizer_kwargs.append(optimizer_kwargs)  # type: ignore[attr-defined]

        if not register_hook:
            return

        def optimizer_hook(*_unused) -> None:
            for opt in param._in_backward_optimizers:  # type: ignore[attr-defined]
                opt.step()

            param.grad = None

        handle = param_to_acc_grad_map[param].register_hook(optimizer_hook)  # type: ignore[attr-defined]
        if param not in param_to_optim_hook_handle_map:
            param_to_optim_hook_handle_map[param] = []
        param_to_optim_hook_handle_map[param].append(handle)

    for param in params:
        _apply_optimizer_in_backward_to_param(param)


def _get_in_backward_optimizers(module: torch.nn.Module) -> list[torch.optim.Optimizer]:
    """
    Return a list of in-backward optimizers applied to ``module``'s parameters. Note that these
    optimizers are not intended to directly have their ``step`` or ``zero_grad`` methods called
    by the user and are intended to be used for things like checkpointing.

    Args:
        module: (torch.nn.Module): model to retrieve in-backward optimizers for

    Returns:
        List[torch.optim.Optimizer]: the in-backward optimizers.

    Example::
        _apply_optimizer_in_backward(torch.optim.SGD, model.parameters(), {"lr": 0.01})
        optims = _get_optimizers_in_backward(model)
    """
    optims: list[torch.optim.Optimizer] = []
    for param in module.parameters():
        optims.extend(getattr(param, "_in_backward_optimizers", []))

    return optims

```



## High-Level Overview

"""    Upon ``backward()``, the optimizer specified for each parameter will fire after    the gradient has been accumulated into the parameter.    Note - gradients for these parameters will be set to None after ``backward()``.    This means that any other optimizer not specified via `_apply_optimizer_in_backward`    over this parameter will be a no-op.    Args:        optimizer_class: (Type[torch.optim.Optimizer]): Optimizer to apply to parameter        params: (Iterator[nn.Parameter]): parameters to apply optimizer state to        optimizer_kwargs: (Dict[str, Any]): kwargs to pass to optimizer constructor        register_hook: (bool): whether to register a hook that runs the optimizer            after gradient for this parameter is accumulated. This is the default            way that optimizer in backward is implemented, but specific use cases            (such as DDP) may wish to override this to implement custom behavior.            (Default = True)    Example::        params_generator = model.parameters()        param_1 = next(params_generator)        remainder_params = list(params_generator)        apply_optimizer_in_backward(torch.optim.SGD, [param_1], {"lr": 0.02})        apply_optimizer_in_backward(torch.optim.Adam, remainder_params, {"lr": 0.04})        model(...).sum().backward()  # after backward, parameters will already

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_apply_optimizer_in_backward`, `_apply_optimizer_in_backward_to_param`, `optimizer_hook`, `_get_in_backward_optimizers`

**Key imports**: Iterable, Any, no_type_check, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Iterable
- `typing`: Any, no_type_check
- `torch`


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

Files in the same folder (`torch/distributed/optim`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`zero_redundancy_optimizer.pyi_docs.md`](./zero_redundancy_optimizer.pyi_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`functional_adadelta.py_docs.md`](./functional_adadelta.py_docs.md)
- [`post_localSGD_optimizer.py_docs.md`](./post_localSGD_optimizer.py_docs.md)
- [`functional_adamax.py_docs.md`](./functional_adamax.py_docs.md)
- [`named_optimizer.py_docs.md`](./named_optimizer.py_docs.md)
- [`functional_adagrad.py_docs.md`](./functional_adagrad.py_docs.md)
- [`functional_rprop.py_docs.md`](./functional_rprop.py_docs.md)
- [`functional_adam.py_docs.md`](./functional_adam.py_docs.md)


## Cross-References

- **File Documentation**: `apply_optimizer_in_backward.py_docs.md`
- **Keyword Index**: `apply_optimizer_in_backward.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
