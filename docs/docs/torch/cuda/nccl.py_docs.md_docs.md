# Documentation: `docs/torch/cuda/nccl.py_docs.md`

## File Metadata

- **Path**: `docs/torch/cuda/nccl.py_docs.md`
- **Size**: 7,438 bytes (7.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/cuda/nccl.py`

## File Metadata

- **Path**: `torch/cuda/nccl.py`
- **Size**: 4,591 bytes (4.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import collections
import warnings
from collections.abc import Sequence
from typing import Optional, Union

import torch.cuda


__all__ = ["all_reduce", "reduce", "broadcast", "all_gather", "reduce_scatter"]

SUM = 0  # ncclRedOp_t


def is_available(tensors):
    if not hasattr(torch._C, "_nccl_all_reduce"):
        warnings.warn("PyTorch is not compiled with NCCL support", stacklevel=2)
        return False

    devices = set()
    for tensor in tensors:
        if tensor.is_sparse:
            return False
        if not tensor.is_contiguous():
            return False
        if not tensor.is_cuda:
            return False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    return True


def version():
    """
    Returns the version of the NCCL.


    This function returns a tuple containing the major, minor, and patch version numbers of the NCCL.
    The suffix is also included in the tuple if a version suffix exists.
    Returns:
        tuple: The version information of the NCCL.
    """
    ver = torch._C._nccl_version()
    major = ver >> 32
    minor = (ver >> 16) & 65535
    patch = ver & 65535
    suffix = torch._C._nccl_version_suffix().decode("utf-8")
    if suffix == "":
        return (major, minor, patch)
    else:
        return (major, minor, patch, suffix)


def unique_id():
    return torch._C._nccl_unique_id()


def init_rank(num_ranks, uid, rank):
    return torch._C._nccl_init_rank(num_ranks, uid, rank)


def _check_sequence_type(inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> None:
    if not isinstance(inputs, collections.abc.Container) or isinstance(
        inputs, torch.Tensor
    ):
        raise TypeError("Inputs should be a collection of tensors")


def all_reduce(inputs, outputs=None, op=SUM, streams=None, comms=None):
    _check_sequence_type(inputs)
    if outputs is None:
        outputs = inputs
    _check_sequence_type(outputs)
    torch._C._nccl_all_reduce(inputs, outputs, op, streams, comms)


# `output` used to be `outputs`, taking in a list of tensors. So we have two
# arguments for BC reasons.
def reduce(
    inputs: Sequence[torch.Tensor],
    output: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    root: int = 0,
    op: int = SUM,
    streams: Optional[Sequence[torch.cuda.Stream]] = None,
    comms=None,
    *,
    outputs: Optional[Sequence[torch.Tensor]] = None,
) -> None:
    _check_sequence_type(inputs)
    _output: torch.Tensor
    if outputs is not None:
        if output is not None:
            raise ValueError(
                "'output' and 'outputs' can not be both specified. 'outputs' is deprecated in "
                "favor of 'output', taking in a single output tensor. The signature of reduce is: "
                "reduce(inputs, output=None, root=0, op=SUM, streams=None, comms=None)."
            )
        else:
            warnings.warn(
                "`nccl.reduce` with an output tensor list is deprecated. "
                "Please specify a single output tensor with argument 'output' instead instead.",
                FutureWarning,
                stacklevel=2,
            )
            _output = outputs[root]
    elif not isinstance(output, torch.Tensor) and isinstance(
        output, collections.abc.Sequence
    ):
        # User called old API with positional arguments of list of output tensors.
        warnings.warn(
            "nccl.reduce with an output tensor list is deprecated. "
            "Please specify a single output tensor.",
            FutureWarning,
            stacklevel=2,
        )
        _output = output[root]
    else:
        _output = inputs[root] if output is None else output
    torch._C._nccl_reduce(inputs, _output, root, op, streams, comms)


def broadcast(
    inputs: Sequence[torch.Tensor], root: int = 0, streams=None, comms=None
) -> None:
    _check_sequence_type(inputs)
    torch._C._nccl_broadcast(inputs, root, streams, comms)


def all_gather(
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    streams=None,
    comms=None,
) -> None:
    _check_sequence_type(inputs)
    _check_sequence_type(outputs)
    torch._C._nccl_all_gather(inputs, outputs, streams, comms)


def reduce_scatter(
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    op: int = SUM,
    streams=None,
    comms=None,
) -> None:
    _check_sequence_type(inputs)
    _check_sequence_type(outputs)
    torch._C._nccl_reduce_scatter(inputs, outputs, op, streams, comms)

```



## High-Level Overview

"""    Returns the version of the NCCL.    This function returns a tuple containing the major, minor, and patch version numbers of the NCCL.    The suffix is also included in the tuple if a version suffix exists.    Returns:        tuple: The version information of the NCCL.

This Python file contains 0 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `is_available`, `version`, `unique_id`, `init_rank`, `_check_sequence_type`, `all_reduce`, `reduce`, `broadcast`, `all_gather`, `reduce_scatter`

**Key imports**: collections, warnings, Sequence, Optional, Union, torch.cuda


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `warnings`
- `collections.abc`: Sequence
- `typing`: Optional, Union
- `torch.cuda`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/cuda`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`streams.py_docs.md`](./streams.py_docs.md)
- [`jiterator.py_docs.md`](./jiterator.py_docs.md)
- [`_sanitizer.py_docs.md`](./_sanitizer.py_docs.md)
- [`graphs.py_docs.md`](./graphs.py_docs.md)
- [`gds.py_docs.md`](./gds.py_docs.md)
- [`_pin_memory_utils.py_docs.md`](./_pin_memory_utils.py_docs.md)
- [`_device_limits.py_docs.md`](./_device_limits.py_docs.md)
- [`green_contexts.py_docs.md`](./green_contexts.py_docs.md)


## Cross-References

- **File Documentation**: `nccl.py_docs.md`
- **Keyword Index**: `nccl.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/cuda`):

- [`profiler.py_docs.md_docs.md`](./profiler.py_docs.md_docs.md)
- [`sparse.py_docs.md_docs.md`](./sparse.py_docs.md_docs.md)
- [`tunable.py_kw.md_docs.md`](./tunable.py_kw.md_docs.md)
- [`_pin_memory_utils.py_kw.md_docs.md`](./_pin_memory_utils.py_kw.md_docs.md)
- [`nccl.py_kw.md_docs.md`](./nccl.py_kw.md_docs.md)
- [`gds.py_kw.md_docs.md`](./gds.py_kw.md_docs.md)
- [`jiterator.py_docs.md_docs.md`](./jiterator.py_docs.md_docs.md)
- [`memory.py_kw.md_docs.md`](./memory.py_kw.md_docs.md)
- [`random.py_docs.md_docs.md`](./random.py_docs.md_docs.md)
- [`nvtx.py_kw.md_docs.md`](./nvtx.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `nccl.py_docs.md_docs.md`
- **Keyword Index**: `nccl.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
