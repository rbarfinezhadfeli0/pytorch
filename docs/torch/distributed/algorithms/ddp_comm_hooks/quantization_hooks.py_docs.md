# Documentation: `torch/distributed/algorithms/ddp_comm_hooks/quantization_hooks.py`

## File Metadata

- **Path**: `torch/distributed/algorithms/ddp_comm_hooks/quantization_hooks.py`
- **Size**: 8,318 bytes (8.12 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch
import torch.distributed as dist
from torch import nn


def _quantize_per_tensor_backend(x, scale, zero_point):
    y = torch.round(x / scale) + zero_point
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y


def _dequantize_per_tensor_backend(y, scale, zero_point):
    x = scale * (y.to(torch.float32) - zero_point)
    return x


def _quantize_per_channel_backend(x, scale, zero_point):
    y = torch.zeros(x.size(), device=x.device)
    for i in range(x.size()[0]):
        y[i, :] = torch.round(x[i, :] / scale[i]) + zero_point[i]
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y


def _dequantize_per_channel_backend(y, scale, zero_point):
    y = y.to(torch.float32).to(y.device)
    x = torch.zeros_like(y, device=y.device)
    for i in range(x.size()[0]):
        x[i, :] = scale[i] * (y[i, :] - zero_point[i])
    return x


def _get_allgather_out_list(all_gather_in_list, world_size):
    out_list = [
        torch.zeros_like(
            all_gather_in_list,
            device=all_gather_in_list.device,
            dtype=all_gather_in_list.dtype,
        )
        for _ in range(world_size)
    ]
    return out_list


def quantization_pertensor_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Apply ``torch.quantize_per_tensor`` logic to DDP using ``allgather`` protocol.

    Workers first allgather the scale and zero point of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, quantization_pertensor_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    # pyrefly: ignore [missing-attribute]
    world_size = group_to_use.size()

    tensor = bucket.buffer()

    myObserver = torch.ao.quantization.MinMaxObserver().to(tensor.device)
    myObserver(tensor)

    s, z = myObserver.calculate_qparams()
    s_and_z = torch.FloatTensor([s, z]).to(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)

    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # Store scale and zeros across all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their own ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_tensor_backend(
            tensor, all_ranks_s_and_z[rank][0], all_ranks_s_and_z[rank][1]
        )
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = torch.zeros_like(
            all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_tensor_backend(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return aggregated_dequantized_tensor / world_size

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)


def quantization_perchannel_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket, bucket_size=512
) -> torch.futures.Future[torch.Tensor]:
    """
    Apply``torch.quantize_per_channel`` logic to DDP using ``allgather`` protocol.

    Compared to per-tensor, the main motivation of per-channel is
    for considerably large tensors such as a tensor that contains 6 million
    elements quantizing per a bucket size of 512 (or 128) elements may significantly
    increase the resolution.

    It first splits ``GradBucket`` tensor into multiple chunks (channels) of ``bucket_size``
    elements. Then, workers allgather the scales and zero points of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes, flattens, and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, quantization_perchannel_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    # pyrefly: ignore [missing-attribute]
    world_size = group_to_use.size()

    tensor = bucket.buffer()

    tensor_in_channels = (
        nn.functional.pad(
            input=tensor,
            pad=(0, bucket_size - len(tensor) % bucket_size),
            mode="constant",
            value=0,
        )
        .view(-1, bucket_size)
        .to(tensor.device)
    )

    myPerChannelObserver = torch.ao.quantization.PerChannelMinMaxObserver().to(
        tensor.device
    )
    myPerChannelObserver(tensor_in_channels)

    s_ch, z_ch = myPerChannelObserver.calculate_qparams()
    s_and_z = torch.stack((s_ch, z_ch)).to(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)
    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # Store scale and zeros across all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their corresponding ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_channel_backend(
            tensor_in_channels,
            all_ranks_s_and_z[rank, 0, :],
            all_ranks_s_and_z[rank, 1, :],
        )
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = torch.zeros_like(
            all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_channel_backend(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return (
            torch.flatten(aggregated_dequantized_tensor).to(tensor.device)[
                : tensor.size()[0]
            ]
            / world_size
        )

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)

```



## High-Level Overview

"""    Apply ``torch.quantize_per_tensor`` logic to DDP using ``allgather`` protocol.

This Python file contains 0 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_quantize_per_tensor_backend`, `_dequantize_per_tensor_backend`, `_quantize_per_channel_backend`, `_dequantize_per_channel_backend`, `_get_allgather_out_list`, `quantization_pertensor_hook`, `quantize_and_allgather`, `dequantize_and_aggregate`, `quantization_perchannel_hook`, `quantize_and_allgather`, `dequantize_and_aggregate`

**Key imports**: torch, torch.distributed as dist, nn


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/algorithms/ddp_comm_hooks`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed as dist`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
- [`mixed_precision_hooks.py_docs.md`](./mixed_precision_hooks.py_docs.md)
- [`default_hooks.py_docs.md`](./default_hooks.py_docs.md)
- [`ddp_zero_hook.py_docs.md`](./ddp_zero_hook.py_docs.md)


## Cross-References

- **File Documentation**: `quantization_hooks.py_docs.md`
- **Keyword Index**: `quantization_hooks.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
