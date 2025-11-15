# Documentation: `docs/torch/nn/parallel/parallel_apply.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/parallel/parallel_apply.py_docs.md`
- **Size**: 7,756 bytes (7.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/parallel/parallel_apply.py`

## File Metadata

- **Path**: `torch/nn/parallel/parallel_apply.py`
- **Size**: 4,683 bytes (4.57 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import threading
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch._utils import ExceptionWrapper
from torch.cuda._utils import _get_device_index
from torch.nn.modules import Module


__all__ = ["get_a_var", "parallel_apply"]


def get_a_var(
    obj: torch.Tensor | list[Any] | tuple[Any, ...] | dict[Any, Any],
) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Sequence[dict[str, Any]] | None = None,
    devices: Sequence[int | torch.device | None] | None = None,
) -> list[Any]:
    r"""Apply each `module` in :attr:`modules` in parallel on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs), (
        f"The number of modules {len(modules)} is not equal to the number of inputs {len(inputs)}"
    )
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = (cast(dict[str, Any], {}),) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    streams = [torch.accelerator.current_stream(x) for x in devices]
    assert torch.accelerator.is_available(), "No available accelerator found."
    device_type = torch.accelerator.current_accelerator().type  # type: ignore[union-attr]
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = (
        torch.is_grad_enabled(),
        torch.is_autocast_enabled(),
    )

    def _worker(
        i: int,
        module: Module,
        input: Any,
        kwargs: dict[str, Any],
        device: int | torch.device | None = None,
        stream: torch.Stream | None = None,
    ) -> None:
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            t = get_a_var(input)
            if t is None:
                with lock:
                    results[i] = ExceptionWrapper(
                        where=f"in replica {i}, no device was provided and no tensor input was found; "
                        "device cannot be resolved"
                    )
                return
            device = t.get_device()
        if isinstance(device, torch.device):
            device = device.index
        if stream is None:
            stream = torch.accelerator.current_stream(device)
        try:
            with (
                torch.accelerator.device_index(device),
                stream,
                torch.amp.autocast(device_type, enabled=autocast_enabled),
            ):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where=f"in replica {i} on device {device}"
                )

    if len(modules) > 1:
        threads = [
            threading.Thread(
                target=_worker, args=(i, module, input, kwargs, device, stream)
            )
            for i, (module, input, kwargs, device, stream) in enumerate(
                zip(modules, inputs, kwargs_tup, devices, streams, strict=True)
            )
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs

```



## High-Level Overview

r"""Apply each `module` in :attr:`modules` in parallel on each of :attr:`devices`.    Args:        modules (Module): modules to be parallelized        inputs (tensor): inputs to the modules        devices (list of int or torch.device): CUDA devices    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and    :attr:`devices` (if given) should all have same length. Moreover, each    element of :attr:`inputs` can either be a single object as the only argument    to a module, or a collection of positional arguments.

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_a_var`, `parallel_apply`, `_worker`

**Key imports**: threading, Sequence, Any, cast, torch, ExceptionWrapper, _get_device_index, Module


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/parallel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `threading`
- `collections.abc`: Sequence
- `typing`: Any, cast
- `torch`
- `torch._utils`: ExceptionWrapper
- `torch.cuda._utils`: _get_device_index
- `torch.nn.modules`: Module


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`torch/nn/parallel`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`data_parallel.py_docs.md`](./data_parallel.py_docs.md)
- [`replicate.py_docs.md`](./replicate.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`_functions.py_docs.md`](./_functions.py_docs.md)
- [`scatter_gather.py_docs.md`](./scatter_gather.py_docs.md)
- [`comm.py_docs.md`](./comm.py_docs.md)


## Cross-References

- **File Documentation**: `parallel_apply.py_docs.md`
- **Keyword Index**: `parallel_apply.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn/parallel`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/parallel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`docs/torch/nn/parallel`):

- [`replicate.py_docs.md_docs.md`](./replicate.py_docs.md_docs.md)
- [`replicate.py_kw.md_docs.md`](./replicate.py_kw.md_docs.md)
- [`scatter_gather.py_kw.md_docs.md`](./scatter_gather.py_kw.md_docs.md)
- [`parallel_apply.py_kw.md_docs.md`](./parallel_apply.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`comm.py_kw.md_docs.md`](./comm.py_kw.md_docs.md)
- [`distributed.py_docs.md_docs.md`](./distributed.py_docs.md_docs.md)
- [`comm.py_docs.md_docs.md`](./comm.py_docs.md_docs.md)
- [`_functions.py_docs.md_docs.md`](./_functions.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `parallel_apply.py_docs.md_docs.md`
- **Keyword Index**: `parallel_apply.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
