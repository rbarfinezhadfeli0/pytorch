# Documentation: `torch/ao/pruning/scheduler/lambda_scheduler.py`

## File Metadata

- **Path**: `torch/ao/pruning/scheduler/lambda_scheduler.py`
- **Size**: 2,416 bytes (2.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import warnings
from collections.abc import Callable

from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from .base_scheduler import BaseScheduler


__all__ = ["LambdaSL"]


class LambdaSL(BaseScheduler):
    """Sets the sparsity level of each parameter group to the final sl
    times a given function. When last_epoch=-1, sets initial sl as zero.
    Args:
        sparsifier (BaseSparsifier): Wrapped sparsifier.
        sl_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in sparsifier.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming sparsifier has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95**epoch
        >>> # xdoctest: +SKIP
        >>> scheduler = LambdaSL(sparsifier, sl_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        sparsifier: BaseSparsifier,
        sl_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.sparsifier = sparsifier

        if not isinstance(sl_lambda, list) and not isinstance(sl_lambda, tuple):
            self.sl_lambdas = [sl_lambda] * len(sparsifier.groups)
        else:
            if len(sl_lambda) != len(sparsifier.groups):
                raise ValueError(
                    f"Expected {len(sparsifier.groups)} lr_lambdas, but got {len(sl_lambda)}"
                )
            self.sl_lambdas = list(sl_lambda)
        super().__init__(sparsifier, last_epoch, verbose)  # type: ignore[no-untyped-call]

    def get_sl(self) -> list[float]:
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.",
                stacklevel=2,
            )
        return [
            base_sl * lmbda(self.last_epoch)
            for lmbda, base_sl in zip(self.sl_lambdas, self.base_sl)
        ]

```



## High-Level Overview

"""Sets the sparsity level of each parameter group to the final sl    times a given function. When last_epoch=-1, sets initial sl as zero.    Args:        sparsifier (BaseSparsifier): Wrapped sparsifier.        sl_lambda (function or list): A function which computes a multiplicative            factor given an integer parameter epoch, or a list of such            functions, one for each group in sparsifier.param_groups.        last_epoch (int): The index of last epoch. Default: -1.        verbose (bool): If ``True``, prints a message to stdout for            each update. Default: ``False``.    Example:        >>> # Assuming sparsifier has two groups.        >>> lambda1 = lambda epoch: epoch // 30        >>> lambda2 = lambda epoch: 0.95**epoch        >>> # xdoctest: +SKIP        >>> scheduler = LambdaSL(sparsifier, sl_lambda=[lambda1, lambda2])        >>> for epoch in range(100):        >>>     train(...)        >>>     validate(...)        >>>     scheduler.step()

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LambdaSL`

**Functions defined**: `__init__`, `get_sl`

**Key imports**: warnings, Callable, BaseSparsifier, BaseScheduler


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/pruning/scheduler`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `warnings`
- `collections.abc`: Callable
- `torch.ao.pruning.sparsifier.base_sparsifier`: BaseSparsifier
- `.base_scheduler`: BaseScheduler


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

Files in the same folder (`torch/ao/pruning/scheduler`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`base_scheduler.py_docs.md`](./base_scheduler.py_docs.md)
- [`cubic_scheduler.py_docs.md`](./cubic_scheduler.py_docs.md)


## Cross-References

- **File Documentation**: `lambda_scheduler.py_docs.md`
- **Keyword Index**: `lambda_scheduler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
