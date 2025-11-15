# Documentation: `torch/distributed/_shard/sharded_tensor/_ops/binary_cmp.py`

## File Metadata

- **Path**: `torch/distributed/_shard/sharded_tensor/_ops/binary_cmp.py`
- **Size**: 2,736 bytes (2.67 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d
from torch.distributed._shard.sharded_tensor import _sharded_op_impl, ShardedTensor


def _communicate_result(result, pg):
    # Gather results from all ranks.
    if result:
        result_tensor = torch.ones(1, device=torch.device(torch.cuda.current_device()))
    else:
        result_tensor = torch.zeros(1, device=torch.device(torch.cuda.current_device()))

    dist.all_reduce(result_tensor, group=pg)

    expected_result = torch.ones(
        1, device=torch.device(torch.cuda.current_device())
    ) * dist.get_world_size(pg)

    return torch.equal(result_tensor, expected_result)


def binary_cmp(cmp_fun, types, args, kwargs=None, process_group=None):
    if len(args) != 2:
        raise ValueError(f"Expected two arguments for torch.{cmp_fun.__name__}")

    st1 = args[0]
    st2 = args[1]
    if not (isinstance(st1, ShardedTensor) and isinstance(st2, ShardedTensor)):
        raise TypeError(
            f"Both arguments to torch.{cmp_fun.__name__} need to be of type ShardedTensor"
        )

    # Verify same PG
    if st1._process_group != st2._process_group:
        return False

    if distributed_c10d._rank_not_in_group(
        st1._process_group
    ) or distributed_c10d._rank_not_in_group(st2._process_group):
        return distributed_c10d._rank_not_in_group(
            st1._process_group
        ) == distributed_c10d._rank_not_in_group(st2._process_group)

    # Verify metadata
    if st1.metadata() != st2.metadata():
        return _communicate_result(False, st1._process_group)

    # Verify number of local shards
    st1_local_shards = st1.local_shards()
    st2_local_shards = st2.local_shards()
    if len(st1_local_shards) != len(st2_local_shards):
        return _communicate_result(False, st1._process_group)

    # kwargs must be dict-like
    if kwargs is None:
        kwargs = {}
    # Verify each local shard
    for idx in range(len(st1_local_shards)):
        if st1_local_shards[idx].metadata != st2_local_shards[idx].metadata:
            return _communicate_result(False, st1._process_group)
        if not cmp_fun(
            st1_local_shards[idx].tensor, st2_local_shards[idx].tensor, **kwargs
        ):
            return _communicate_result(False, st1._process_group)

    return _communicate_result(True, st1._process_group)


@_sharded_op_impl(torch.equal)
def equal(types, args, kwargs, process_group):
    return binary_cmp(torch.equal, types, args, kwargs, process_group)


@_sharded_op_impl(torch.allclose)
def allclose(types, args, kwargs, process_group):
    return binary_cmp(torch.allclose, types, args, kwargs, process_group)

```



## High-Level Overview


This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_communicate_result`, `binary_cmp`, `equal`, `allclose`

**Key imports**: torch, torch.distributed as dist, torch.distributed.distributed_c10d as distributed_c10d, _sharded_op_impl, ShardedTensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/_shard/sharded_tensor/_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed as dist`
- `torch.distributed.distributed_c10d as distributed_c10d`
- `torch.distributed._shard.sharded_tensor`: _sharded_op_impl, ShardedTensor


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/distributed/_shard/sharded_tensor/_ops`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_common.py_docs.md`](./_common.py_docs.md)
- [`misc_ops.py_docs.md`](./misc_ops.py_docs.md)
- [`init.py_docs.md`](./init.py_docs.md)
- [`tensor_ops.py_docs.md`](./tensor_ops.py_docs.md)


## Cross-References

- **File Documentation**: `binary_cmp.py_docs.md`
- **Keyword Index**: `binary_cmp.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
