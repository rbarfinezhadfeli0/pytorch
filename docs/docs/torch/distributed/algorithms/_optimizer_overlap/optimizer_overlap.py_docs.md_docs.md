# Documentation: `docs/torch/distributed/algorithms/_optimizer_overlap/optimizer_overlap.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/algorithms/_optimizer_overlap/optimizer_overlap.py_docs.md`
- **Size**: 6,768 bytes (6.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/algorithms/_optimizer_overlap/optimizer_overlap.py`

## File Metadata

- **Path**: `torch/distributed/algorithms/_optimizer_overlap/optimizer_overlap.py`
- **Size**: 3,753 bytes (3.67 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import inspect
from abc import ABC, abstractmethod

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
    _hook_then_optimizer,
    _OptimizerHookState,
)
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.optim import as_functional_optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer


# Contains the mappings between the regular and overlapped optimizer types.
_registered_overlapped_optims: dict[type, type] = {}


def register_overlapped(optim_cls):
    def decorator(target_overlapped_optim_cls):
        if target_overlapped_optim_cls in _registered_overlapped_optims:
            raise ValueError(
                f"{target_overlapped_optim_cls} already registered with optim_cls "
                f"{_registered_overlapped_optims[optim_cls]} {optim_cls}, trying to"
                f"re-register it for {optim_cls} is not supported."
            )
        _registered_overlapped_optims[optim_cls] = target_overlapped_optim_cls
        return target_overlapped_optim_cls

    return decorator


class OverlappedOptimizer(ABC):
    def __init__(self, optim_cls: type) -> None:
        """
        Initialize the OverlappedOptimizer.

        Overlappedoptimizer is a base class that child classes can implement to
        specify how different optimizers will register themselves with DDP.
        """
        self.optim_cls = optim_cls

    @abstractmethod
    def register_ddp(self, ddp: DistributedDataParallel) -> None:
        """Registers the overlapped optimizer with DDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped DDP."
        )

    @abstractmethod
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Registers the overlapped optimizer with FSDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped FSDP."
        )


@register_overlapped(Optimizer)
class _OverlappedStandardOptimizer(OverlappedOptimizer):
    """Overlaps a regular ``Optimizer``."""

    def __init__(self, optim_cls: type, params, *optim_args, **optim_kwargs) -> None:
        super().__init__(optim_cls)
        f_optim = as_functional_optim(self.optim_cls, *optim_args, **optim_kwargs)
        self._opt_hook_state = _OptimizerHookState(f_optim, params)

    def register_ddp(self, ddp_inst: DistributedDataParallel):
        # NOTE: using a custom communication hook and fused optimizer is not
        # yet supported.
        ddp_inst.register_comm_hook(  # type: ignore[operator]
            None,  # wrapped hook state
            _hook_then_optimizer(allreduce_hook, self._opt_hook_state),
        )

    # TODO: register_fsdp once FSDP supports communication hook.
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Register the overlapped optimizer with FSDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped FSDP."
        )


def _as_overlapped_optim(optim_cls: type, params, *args, **kwargs):
    """Return a new ``OverlappedOptimizer`` instance that supports ``optim_cls``."""
    for clz in inspect.getmro(optim_cls):
        try:
            return _registered_overlapped_optims[clz](
                optim_cls, params, *args, **kwargs
            )
        except KeyError:
            pass

    # Fallback to standard overlapped optimizer, which will raise errors if user
    # is attempting to use an unsupported optimizer.
    return _OverlappedStandardOptimizer(optim_cls, params, *args, **kwargs)

```



## High-Level Overview

"""        Initialize the OverlappedOptimizer.        Overlappedoptimizer is a base class that child classes can implement to        specify how different optimizers will register themselves with DDP.

This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OverlappedOptimizer`, `_OverlappedStandardOptimizer`

**Functions defined**: `register_overlapped`, `decorator`, `__init__`, `register_ddp`, `register_fsdp`, `__init__`, `register_ddp`, `register_fsdp`, `_as_overlapped_optim`

**Key imports**: inspect, ABC, abstractmethod, allreduce_hook, FullyShardedDataParallel, as_functional_optim, DistributedDataParallel, Optimizer


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/algorithms/_optimizer_overlap`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `inspect`
- `abc`: ABC, abstractmethod
- `torch.distributed.algorithms.ddp_comm_hooks.default_hooks`: allreduce_hook
- `torch.distributed.fsdp`: FullyShardedDataParallel
- `torch.distributed.optim`: as_functional_optim
- `torch.nn.parallel`: DistributedDataParallel
- `torch.optim`: Optimizer


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/distributed/algorithms/_optimizer_overlap`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `optimizer_overlap.py_docs.md`
- **Keyword Index**: `optimizer_overlap.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/algorithms/_optimizer_overlap`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/algorithms/_optimizer_overlap`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/distributed/algorithms/_optimizer_overlap`):

- [`optimizer_overlap.py_kw.md_docs.md`](./optimizer_overlap.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `optimizer_overlap.py_docs.md_docs.md`
- **Keyword Index**: `optimizer_overlap.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
