# Documentation: `docs/torch/nn/parallel/_functions.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/parallel/_functions.py_docs.md`
- **Size**: 7,575 bytes (7.40 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/parallel/_functions.py`

## File Metadata

- **Path**: `torch/nn/parallel/_functions.py`
- **Size**: 4,946 bytes (4.83 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import warnings
from itertools import chain

import torch
from torch._utils import _get_device_index
from torch.autograd import Function
from torch.nn.parallel import comm


class Broadcast(Function):
    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        assert all(i.device.type != "cpu" for i in inputs), (
            "Broadcast function not implemented for CPU tensors"
        )
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.target_gpus = target_gpus
        if len(inputs) == 0:
            return ()
        ctx.num_inputs = len(inputs)
        ctx.input_device = inputs[0].get_device()
        outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
        non_differentiables = []
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                non_differentiables.extend(output[idx] for output in outputs)
        ctx.mark_non_differentiable(*non_differentiables)
        return tuple(chain.from_iterable(outputs))

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + ReduceAddCoalesced.apply(
            ctx.input_device, ctx.num_inputs, *grad_outputs
        )


class ReduceAddCoalesced(Function):
    @staticmethod
    def forward(ctx, destination, num_inputs, *grads):
        ctx.target_gpus = [
            grads[i].get_device() for i in range(0, len(grads), num_inputs)
        ]

        grads_ = [grads[i : i + num_inputs] for i in range(0, len(grads), num_inputs)]
        return comm.reduce_add_coalesced(grads_, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (
            None,
            None,
        ) + Broadcast.apply(ctx.target_gpus, *grad_outputs)


class Gather(Function):
    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        assert all(i.device.type != "cpu" for i in inputs), (
            "Gather function not implemented for CPU tensors"
        )
        if target_device == "cpu":
            ctx.target_device = "cpu"
        else:
            target_device = _get_device_index(target_device, True)
            ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(i.get_device() for i in inputs)
        if all(t.dim() == 0 for t in inputs) and dim == 0:
            inputs = tuple(t.view(1) for t in inputs)
            warnings.warn(
                "Was asked to gather along dimension 0, but all "
                "input tensors were scalars; will instead unsqueeze "
                "and return a vector.",
                stacklevel=2,
            )
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False
        ctx.input_sizes = tuple(i.size(ctx.dim) for i in inputs)
        return comm.gather(inputs, ctx.dim, ctx.target_device)

    @staticmethod
    def backward(ctx, grad_output):
        scattered_grads = Scatter.apply(
            ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output
        )
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        return (None, None) + scattered_grads


class Scatter(Function):
    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None
        if torch.accelerator.is_available() and ctx.input_device == -1:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(torch.device(device)) for device in target_gpus]
        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)
        # Synchronize with the copy stream
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.accelerator.device_index(target_gpus[i]):
                    main_stream = torch.accelerator.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)


# background streams used for copying
_streams: list[torch.Stream | None] | None = None


def _get_stream(device: torch.device):
    """Get a background stream for copying between CPU and target device."""
    global _streams
    if device.type == "cpu" or not torch.accelerator.is_available():
        return None
    assert torch.accelerator.current_accelerator().type == device.type
    if _streams is None:
        _streams = [None] * torch.accelerator.device_count()
    if _streams[device.index] is None:
        _streams[device.index] = torch.Stream(device.index)
    return _streams[device.index]

```



## High-Level Overview


This Python file contains 4 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Broadcast`, `ReduceAddCoalesced`, `Gather`, `Scatter`

**Functions defined**: `forward`, `backward`, `forward`, `backward`, `forward`, `backward`, `forward`, `backward`, `_get_stream`

**Key imports**: warnings, chain, torch, _get_device_index, Function, comm


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/parallel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `warnings`
- `itertools`: chain
- `torch`
- `torch._utils`: _get_device_index
- `torch.autograd`: Function
- `torch.nn.parallel`: comm


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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
- [`parallel_apply.py_docs.md`](./parallel_apply.py_docs.md)
- [`replicate.py_docs.md`](./replicate.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`scatter_gather.py_docs.md`](./scatter_gather.py_docs.md)
- [`comm.py_docs.md`](./comm.py_docs.md)


## Cross-References

- **File Documentation**: `_functions.py_docs.md`
- **Keyword Index**: `_functions.py_kw.md`
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

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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
- [`parallel_apply.py_docs.md_docs.md`](./parallel_apply.py_docs.md_docs.md)
- [`parallel_apply.py_kw.md_docs.md`](./parallel_apply.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`comm.py_kw.md_docs.md`](./comm.py_kw.md_docs.md)
- [`distributed.py_docs.md_docs.md`](./distributed.py_docs.md_docs.md)
- [`comm.py_docs.md_docs.md`](./comm.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_functions.py_docs.md_docs.md`
- **Keyword Index**: `_functions.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
