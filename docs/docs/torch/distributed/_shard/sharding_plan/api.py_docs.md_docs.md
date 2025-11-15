# Documentation: `docs/torch/distributed/_shard/sharding_plan/api.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/_shard/sharding_plan/api.py_docs.md`
- **Size**: 8,232 bytes (8.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/_shard/sharding_plan/api.py`

## File Metadata

- **Path**: `torch/distributed/_shard/sharding_plan/api.py`
- **Size**: 3,649 bytes (3.56 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import abc
from dataclasses import dataclass
from typing import Optional, Union

import torch.nn as nn
from torch.distributed._shard.sharder import Sharder
from torch.distributed._shard.sharding_spec import ShardingSpec


@dataclass
class ShardingPlan:
    """
    Representation of a sharding plan, describes how to shard a module
    across hosts. `plan` is used to shard module parameters according to the spec provided,
    `output_plan` and `return_local_tensor` are optional, they are used to specify the output
    layout of a module with a spec, and when to convert back to data parallel fashion.

    Args:
        plan (Dict[str, Union[:class:`torch.distributed._shard.sharding_spec.ShardingSpec`,
              :class:`torch.distributed._shard.sharder.Sharder`]):
            a dict describes how to shard a module, there're currently two ways to shard a module:
                1. directly shard a module parameter by a `ShardingSpec`, keyed by the name of
                   a parameter to a `ShardingSpec`.
                2. shard a submodule by applying a `Sharder` on it, keyed by the name of a module
                   to a `Sharder` object.
        output_plan (Dict[str, :class:`torch.distributed._shard.sharding_spec.ShardingSpec`), optional):
            a dict specifies the layout of a module's output which produces a ShardedTensor,
            keyed by the name of module to ShardingSpec("" in key means the root module).
            Default: `None`
        return_local_tensor (List[str], optional): a list of string, each element enables
            a module's sharded output to be returned as a Tensor from its local shards to
            ensure further processing in a data parallel fashion. ("" in list means the
            root module).
            Default: None
    Example:
      Suppose we want to shard a module with two linear layers and then run it with DDP, we also
      want to convert the output of the second linear layer back to DDP, we can do it as follows:

        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> class MyModule(nn.Module):
        >>>     def __init__(self) -> None:
        >>>        super().__init__()
        >>>        self.fc1 = nn.Linear()
        >>>        self.gelu = nn.GELU()
        >>>        self.fc2 = nn.Linear()
        >>>        self.relu = nn.Linear()
        >>>
        >>>     def forward(self, input):
        >>>         return self.relu(self.fc2(self.gelu(self.fc1(input))))


        >>> # xdoctest: +SKIP("Undefined spec1, spec2)
        >>> sharding_plan = ShardingPlan(
        >>>    plan={
        >>>        "fc1.weight": spec1,
        >>>        "fc2.weight": spec2
        >>>    },
        >>>    output_plan={
        >>>        "fc2": output_spec
        >>>    },
        >>>    return_local_tensor=["fc2"]
        >>> )
    """

    plan: dict[str, Union[ShardingSpec, Sharder]]
    output_plan: Optional[dict[str, ShardingSpec]] = None
    return_local_tensor: Optional[list[str]] = None


class ShardingPlanner(abc.ABC):
    """
    Default ShardingPlanner interface, can be extended and
    implement advanced sharding strategies.
    """

    @abc.abstractmethod
    def build_plan(self, module: nn.Module) -> ShardingPlan:
        """
        Given a nn.Module, define how to shard the module across
        ranks, return a ShardingPlan
        Args:
            module (:class:`torch.nn.Module`):
                The module to apply sharding to.
        Returns:
            A :class:`torch.distributed._shard.sharding_plan.ShardingPlan` object that
            represents how to shard the module.
        """

```



## High-Level Overview

"""    Representation of a sharding plan, describes how to shard a module    across hosts. `plan` is used to shard module parameters according to the spec provided,    `output_plan` and `return_local_tensor` are optional, they are used to specify the output    layout of a module with a spec, and when to convert back to data parallel fashion.    Args:        plan (Dict[str, Union[:class:`torch.distributed._shard.sharding_spec.ShardingSpec`,              :class:`torch.distributed._shard.sharder.Sharder`]):            a dict describes how to shard a module, there're currently two ways to shard a module:                1. directly shard a module parameter by a `ShardingSpec`, keyed by the name of                   a parameter to a `ShardingSpec`.                2. shard a submodule by applying a `Sharder` on it, keyed by the name of a module                   to a `Sharder` object.        output_plan (Dict[str, :class:`torch.distributed._shard.sharding_spec.ShardingSpec`), optional):            a dict specifies the layout of a module's output which produces a ShardedTensor,            keyed by the name of module to ShardingSpec("" in key means the root module).            Default: `None`        return_local_tensor (List[str], optional): a list of string, each element enables            a module's sharded output to be returned as a Tensor from its local shards to            ensure further processing in a data parallel fashion. ("" in list means the            root module).            Default: None    Example:      Suppose we want to shard a module with two linear layers and then run it with DDP, we also      want to convert the output of the second linear layer back to DDP, we can do it as follows:        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)        >>> class MyModule(nn.Module):        >>>     def __init__(self) -> None:        >>>        super().__init__()        >>>        self.fc1 = nn.Linear()        >>>        self.gelu = nn.GELU()        >>>        self.fc2 = nn.Linear()        >>>        self.relu = nn.Linear()        >>>        >>>     def forward(self, input):        >>>         return self.relu(self.fc2(self.gelu(self.fc1(input))))

This Python file contains 4 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ShardingPlan`, `MyModule`, `ShardingPlanner`

**Functions defined**: `__init__`, `forward`, `build_plan`

**Key imports**: abc, dataclass, Optional, Union, torch.nn as nn, Sharder, ShardingSpec


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/_shard/sharding_plan`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`
- `dataclasses`: dataclass
- `typing`: Optional, Union
- `torch.nn as nn`
- `torch.distributed._shard.sharder`: Sharder
- `torch.distributed._shard.sharding_spec`: ShardingSpec


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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

Files in the same folder (`torch/distributed/_shard/sharding_plan`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `api.py_docs.md`
- **Keyword Index**: `api.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/_shard/sharding_plan`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/_shard/sharding_plan`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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

Files in the same folder (`docs/torch/distributed/_shard/sharding_plan`):

- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `api.py_docs.md_docs.md`
- **Keyword Index**: `api.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
