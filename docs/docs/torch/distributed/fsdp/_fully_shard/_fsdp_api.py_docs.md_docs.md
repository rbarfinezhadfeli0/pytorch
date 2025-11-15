# Documentation: `docs/torch/distributed/fsdp/_fully_shard/_fsdp_api.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/fsdp/_fully_shard/_fsdp_api.py_docs.md`
- **Size**: 10,097 bytes (9.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/fsdp/_fully_shard/_fsdp_api.py`

## File Metadata

- **Path**: `torch/distributed/fsdp/_fully_shard/_fsdp_api.py`
- **Size**: 5,442 bytes (5.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist


_ReduceOp = Union[dist.ReduceOp, dist.ReduceOp.RedOpType]


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """
    This configures FSDP's mixed precision. Unlike autocast, this applies mixed
    precision at the module level, not op level, which means low-precision
    activations are saved for backward and high-to-low-precision casts are
    incurred only at module boundaries.

    FSDP works well with module-level mixed precision since it keeps the
    high-precision sharded parameters in memory anyway. In other words, FSDP
    does not require any extra memory to keep a high-precision copy of the
    parameters for the optimizer step.

    Attributes:
        param_dtype (Optional[torch.dtype]): This specifies the dtype for
            the unsharded parameter and hence the dtype for forward/backward
            computation and the parameter all-gather. If this is ``None``, then
            the unsharded parameter uses the original dtype. The optimizer step
            uses the sharded parameter in the original dtype. (Default:
            ``None``)
        reduce_dtype (Optional[torch.dtype]): This specifies the dtype for
            gradient reduction (i.e. reduce-scatter or all-reduce). If this is
            ``None`` but ``param_dtype`` is not ``None``, then the reduction
            uses the compute dtype. This can be used to run gradient reduction
            in full precision while using low precision for compute. If also
            gradient reduction is disabled via :meth:`set_requires_gradient_sync`,
            then FSDP will accumulate gradients using ``reduce_dtype``.
            (Default: ``None``)
        output_dtype (Optional[torch.dtype]): This specifies the dtype for
            casting floating-point forward outputs. This can be used to
            help implement cases where different modules have different mixed
            precision policies. (Default: ``None``)
        cast_forward_inputs (bool): This specifies whether FSDP should cast the
            forward's floating-point input tensors to ``param_dtype`` or not.
    """

    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True


class Comm(ABC):
    """
    Interface for communication primitives.
    A primitive primarily needs to handle 3 tasks, namely:

    1. How to allocate memory for communication
       Depending on the goal, an implementation can choose to:
       a. associate each call to a temporary buffer
          (best for flexibility and simplicity)
       b. reuse an persistent buffer for efficiency reasons

    2. Where to allocate memory
       (e.g. NCCL mem pool or regular cuda caching allocator)

    3. What to do/call upon the comm is called
       (see `AllGather` interface as an example)
    """

    @abstractmethod
    def allocate(
        self,
        size: Sequence[Union[int, torch.SymInt]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        This handles the "how to allocate memory" part.

        A default implementation could be simply:

        .. code-block:: python
            with self.mem_pool:
                torch.empty(...)

        Args:
            size (Sequence[Union[int, torch.SymInt]]): size of the tensor buffer
            dtype (torch.dtype): dtype of the tensor buffer
            device (torch.device): which device to allocate the tensor onto
        """
        ...


class AllGather(Comm):
    """
    Interface for all_gather comm primitive
    """

    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Optional[dist.Work]: ...


class ReduceScatter(Comm):
    """
    Interface for reduce_scatter comm primitive
    """

    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> Optional[dist.Work]: ...


@dataclass
class OffloadPolicy:
    """
    This base class represents the policy of no offloading and is only used as
    the default value for the ``offload_policy`` arg.
    """


@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    """
    This offload policy offloads parameters, gradients, and optimizer states to
    CPU. Sharded parameters are copied host-to-device before all-gather. The
    all-gathered parameters are freed according to ``reshard_after_forward``.
    Sharded gradients are copied device-to-host in backward, and the optimizer
    step runs on CPU with CPU optimizer states.

    Attributes:
        pin_memory (bool): Whether to pin sharded parameter and gradient
            memory. Pinning memory allows both more efficient H2D/D2H copies
            and for the copies to overlap with compute. However, the pinned
            memory cannot be used by other processes. Set this to ``False`` if
            you have insufficient CPU memory. (Default: ``True``)
    """

    pin_memory: bool = True

```



## High-Level Overview

"""    This configures FSDP's mixed precision. Unlike autocast, this applies mixed    precision at the module level, not op level, which means low-precision    activations are saved for backward and high-to-low-precision casts are    incurred only at module boundaries.    FSDP works well with module-level mixed precision since it keeps the    high-precision sharded parameters in memory anyway. In other words, FSDP    does not require any extra memory to keep a high-precision copy of the    parameters for the optimizer step.    Attributes:        param_dtype (Optional[torch.dtype]): This specifies the dtype for            the unsharded parameter and hence the dtype for forward/backward            computation and the parameter all-gather. If this is ``None``, then            the unsharded parameter uses the original dtype. The optimizer step            uses the sharded parameter in the original dtype. (Default:            ``None``)        reduce_dtype (Optional[torch.dtype]): This specifies the dtype for            gradient reduction (i.e. reduce-scatter or all-reduce). If this is            ``None`` but ``param_dtype`` is not ``None``, then the reduction            uses the compute dtype. This can be used to run gradient reduction            in full precision while using low precision for compute. If also            gradient reduction is disabled via :meth:`set_requires_gradient_sync`,            then FSDP will accumulate gradients using ``reduce_dtype``.            (Default: ``None``)        output_dtype (Optional[torch.dtype]): This specifies the dtype for            casting floating-point forward outputs. This can be used to            help implement cases where different modules have different mixed            precision policies. (Default: ``None``)        cast_forward_inputs (bool): This specifies whether FSDP should cast the            forward's floating-point input tensors to ``param_dtype`` or not.

This Python file contains 8 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MixedPrecisionPolicy`, `Comm`, `AllGather`, `ReduceScatter`, `OffloadPolicy`, `CPUOffloadPolicy`

**Functions defined**: `allocate`, `__call__`, `__call__`

**Key imports**: ABC, abstractmethod, Sequence, dataclass, Optional, Union, torch, torch.distributed as dist


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/fsdp/_fully_shard`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`: ABC, abstractmethod
- `collections.abc`: Sequence
- `dataclasses`: dataclass
- `typing`: Optional, Union
- `torch`
- `torch.distributed as dist`


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


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

Files in the same folder (`torch/distributed/fsdp/_fully_shard`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_fsdp_init.py_docs.md`](./_fsdp_init.py_docs.md)
- [`_fsdp_state.py_docs.md`](./_fsdp_state.py_docs.md)
- [`_fsdp_common.py_docs.md`](./_fsdp_common.py_docs.md)
- [`_fsdp_param_group.py_docs.md`](./_fsdp_param_group.py_docs.md)
- [`_fsdp_collectives.py_docs.md`](./_fsdp_collectives.py_docs.md)
- [`_fsdp_param.py_docs.md`](./_fsdp_param.py_docs.md)
- [`_fully_shard.py_docs.md`](./_fully_shard.py_docs.md)


## Cross-References

- **File Documentation**: `_fsdp_api.py_docs.md`
- **Keyword Index**: `_fsdp_api.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/fsdp/_fully_shard`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/fsdp/_fully_shard`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


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

Files in the same folder (`docs/torch/distributed/fsdp/_fully_shard`):

- [`_fsdp_common.py_docs.md_docs.md`](./_fsdp_common.py_docs.md_docs.md)
- [`_fsdp_collectives.py_kw.md_docs.md`](./_fsdp_collectives.py_kw.md_docs.md)
- [`_fsdp_init.py_docs.md_docs.md`](./_fsdp_init.py_docs.md_docs.md)
- [`_fsdp_param.py_kw.md_docs.md`](./_fsdp_param.py_kw.md_docs.md)
- [`_fsdp_state.py_kw.md_docs.md`](./_fsdp_state.py_kw.md_docs.md)
- [`_fsdp_collectives.py_docs.md_docs.md`](./_fsdp_collectives.py_docs.md_docs.md)
- [`_fully_shard.py_docs.md_docs.md`](./_fully_shard.py_docs.md_docs.md)
- [`_fsdp_init.py_kw.md_docs.md`](./_fsdp_init.py_kw.md_docs.md)
- [`_fsdp_state.py_docs.md_docs.md`](./_fsdp_state.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_fsdp_api.py_docs.md_docs.md`
- **Keyword Index**: `_fsdp_api.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
