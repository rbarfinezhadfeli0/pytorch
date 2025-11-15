# Documentation: `torch/distributed/optim/utils.py`

## File Metadata

- **Path**: `torch/distributed/optim/utils.py`
- **Size**: 2,238 bytes (2.19 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

from torch import optim

from .functional_adadelta import _FunctionalAdadelta
from .functional_adagrad import _FunctionalAdagrad
from .functional_adam import _FunctionalAdam
from .functional_adamax import _FunctionalAdamax
from .functional_adamw import _FunctionalAdamW
from .functional_rmsprop import _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop
from .functional_sgd import _FunctionalSGD


# dict to map a user passed in optimizer_class to a functional
# optimizer class if we have already defined inside the
# distributed.optim package, this is so that we hide the
# functional optimizer to user and still provide the same API.
functional_optim_map = {
    optim.Adagrad: _FunctionalAdagrad,
    optim.Adam: _FunctionalAdam,
    optim.AdamW: _FunctionalAdamW,
    optim.SGD: _FunctionalSGD,
    optim.Adadelta: _FunctionalAdadelta,
    optim.RMSprop: _FunctionalRMSprop,
    optim.Rprop: _FunctionalRprop,
    optim.Adamax: _FunctionalAdamax,
}


def register_functional_optim(key, optim):
    """
    Interface to insert a new functional optimizer to functional_optim_map
    ``fn_optim_key`` and ``fn_optimizer`` are user defined. The optimizer and key
    need not be of :class:`torch.optim.Optimizer` (e.g. for custom optimizers)
    Example::
        >>> # import the new functional optimizer
        >>> # xdoctest: +SKIP
        >>> from xyz import fn_optimizer
        >>> from torch.distributed.optim.utils import register_functional_optim
        >>> fn_optim_key = "XYZ_optim"
        >>> register_functional_optim(fn_optim_key, fn_optimizer)
    """
    if key not in functional_optim_map:
        functional_optim_map[key] = optim


def as_functional_optim(optim_cls: type, *args, **kwargs):
    try:
        functional_cls = functional_optim_map[optim_cls]
    except KeyError as e:
        raise ValueError(
            f"Optimizer {optim_cls} does not have a functional counterpart!"
        ) from e

    return _create_functional_optim(functional_cls, *args, **kwargs)


def _create_functional_optim(functional_optim_cls: type, *args, **kwargs):
    return functional_optim_cls(
        [],
        *args,
        **kwargs,
        _allow_empty_param_list=True,
    )

```



## High-Level Overview

"""    Interface to insert a new functional optimizer to functional_optim_map    ``fn_optim_key`` and ``fn_optimizer`` are user defined. The optimizer and key    need not be of :class:`torch.optim.Optimizer` (e.g. for custom optimizers)    Example::        >>> # import the new functional optimizer        >>> # xdoctest: +SKIP        >>> from xyz import fn_optimizer        >>> from torch.distributed.optim.utils import register_functional_optim        >>> fn_optim_key = "XYZ_optim"        >>> register_functional_optim(fn_optim_key, fn_optimizer)

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `register_functional_optim`, `as_functional_optim`, `_create_functional_optim`

**Key imports**: optim, _FunctionalAdadelta, _FunctionalAdagrad, _FunctionalAdam, _FunctionalAdamax, _FunctionalAdamW, _FunctionalRMSprop, _FunctionalRprop, _FunctionalSGD, the new functional optimizer


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`: optim
- `.functional_adadelta`: _FunctionalAdadelta
- `.functional_adagrad`: _FunctionalAdagrad
- `.functional_adam`: _FunctionalAdam
- `.functional_adamax`: _FunctionalAdamax
- `.functional_adamw`: _FunctionalAdamW
- `.functional_rmsprop`: _FunctionalRMSprop
- `.functional_rprop`: _FunctionalRprop
- `.functional_sgd`: _FunctionalSGD
- `the new functional optimizer`
- `xyz`: fn_optimizer
- `torch.distributed.optim.utils`: register_functional_optim


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
- [`functional_adadelta.py_docs.md`](./functional_adadelta.py_docs.md)
- [`post_localSGD_optimizer.py_docs.md`](./post_localSGD_optimizer.py_docs.md)
- [`functional_adamax.py_docs.md`](./functional_adamax.py_docs.md)
- [`named_optimizer.py_docs.md`](./named_optimizer.py_docs.md)
- [`functional_adagrad.py_docs.md`](./functional_adagrad.py_docs.md)
- [`functional_rprop.py_docs.md`](./functional_rprop.py_docs.md)
- [`functional_adam.py_docs.md`](./functional_adam.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
