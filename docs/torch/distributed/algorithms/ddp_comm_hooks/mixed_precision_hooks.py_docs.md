# Documentation: `torch/distributed/algorithms/ddp_comm_hooks/mixed_precision_hooks.py`

## File Metadata

- **Path**: `torch/distributed/algorithms/ddp_comm_hooks/mixed_precision_hooks.py`
- **Size**: 3,254 bytes (3.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from dataclasses import dataclass
from typing import Any, no_type_check

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.distributed.utils import _free_storage


@dataclass
class _AllreduceUpcastHookState:
    """
    State to manage DDP mixed precision in backward / gradient communication.

    This contains a weakref to the DDP module for access to reducer and process
    group, and a stream to run parameter and gradient upcasts.
    """

    ddp_weakref: Any
    upcast_stream: torch.Stream
    wait_for_stream_enqueued: bool = False


@no_type_check
def _reducer_allreduce_and_upcast_hook(
    hook_state: _AllreduceUpcastHookState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Perform allreduce in precision ``reduce_dtype``, upcast to prepare for optimizer.

    Performs allreduce in the reduced precision given by DDP's mixed precision
    reduce_dtype, and upcasts parameters and gradients to fp32 in preparation
    to run the optimizer.
    """
    ddp_weakref = hook_state.ddp_weakref
    reducer, process_group = ddp_weakref().reducer, ddp_weakref().process_group
    # Cast bucket if different than param_dtype.
    if (
        ddp_weakref().mixed_precision.param_dtype
        != ddp_weakref().mixed_precision.reduce_dtype
    ):
        # Cast bucket tensor to reduce_dtype
        bucket.set_buffer(
            bucket.buffer().to(ddp_weakref().mixed_precision.reduce_dtype)
        )
    fut = reducer._run_allreduce_hook(bucket)
    ret_fut = torch.futures.Future()
    stream = hook_state.upcast_stream
    with stream:
        fut.wait()
        bucket.buffer().div_(process_group.size())
        ret_fut.set_result(bucket.buffer())

        # Upcast parameters and gradients so optimizer step can run in fp32.
        for p in bucket.parameters():
            p.data = p._fp_param
            # free storage for mp param as it will be allocated again in next
            # forward pass.
            _free_storage(p._mp_param)
            p.grad.data = p.grad.to(p.data.dtype)

    # enqueue a callback to wait for this stream at end of backward
    def wait_for_stream_cb():
        torch.accelerator.current_stream().wait_stream(stream)
        # Remove post-backward hooks since they are re-installed in next
        # iteration, similar to FSDP.
        # Parameters that don't require grad still needed to be casted since
        # they may participate in computation. However, they would not be recast
        # by hook above as they don't have a grad hook installed, so cast them
        # back here.
        for _, p in ddp_weakref().module.named_parameters():
            if hasattr(p, "_ddp_mp_hook_state"):
                p._ddp_mp_hook_state[1].remove()
                delattr(p, "_ddp_mp_hook_state")
            if not p.requires_grad and not hasattr(p, "_ddp_ignored"):
                p.data = p._fp_param

        # reset for next backward pass
        hook_state.wait_for_stream_enqueued = False

    if not hook_state.wait_for_stream_enqueued:
        Variable._execution_engine.queue_callback(wait_for_stream_cb)
        # mark that the callback is enqueued
        hook_state.wait_for_stream_enqueued = True

    return ret_fut

```



## High-Level Overview

"""    State to manage DDP mixed precision in backward / gradient communication.    This contains a weakref to the DDP module for access to reducer and process    group, and a stream to run parameter and gradient upcasts.

This Python file contains 2 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_AllreduceUpcastHookState`

**Functions defined**: `_reducer_allreduce_and_upcast_hook`, `wait_for_stream_cb`

**Key imports**: dataclass, Any, no_type_check, torch, torch.distributed as dist, Variable, _free_storage


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/algorithms/ddp_comm_hooks`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`: dataclass
- `typing`: Any, no_type_check
- `torch`
- `torch.distributed as dist`
- `torch.autograd`: Variable
- `torch.distributed.utils`: _free_storage


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


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

Files in the same folder (`torch/distributed/algorithms/ddp_comm_hooks`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`debugging_hooks.py_docs.md`](./debugging_hooks.py_docs.md)
- [`post_localSGD_hook.py_docs.md`](./post_localSGD_hook.py_docs.md)
- [`powerSGD_hook.py_docs.md`](./powerSGD_hook.py_docs.md)
- [`optimizer_overlap_hooks.py_docs.md`](./optimizer_overlap_hooks.py_docs.md)
- [`default_hooks.py_docs.md`](./default_hooks.py_docs.md)
- [`ddp_zero_hook.py_docs.md`](./ddp_zero_hook.py_docs.md)
- [`quantization_hooks.py_docs.md`](./quantization_hooks.py_docs.md)


## Cross-References

- **File Documentation**: `mixed_precision_hooks.py_docs.md`
- **Keyword Index**: `mixed_precision_hooks.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
