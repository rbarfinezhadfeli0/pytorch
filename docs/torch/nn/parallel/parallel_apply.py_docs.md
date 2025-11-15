# Documentation: parallel_apply.py

## File Metadata
- **Path**: `torch/nn/parallel/parallel_apply.py`
- **Size**: 4683 bytes
- **Lines**: 135
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Functions
This file defines 3 function(s): get_a_var, parallel_apply, _worker


## Key Components

The file contains 469 words across 135 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4683 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
