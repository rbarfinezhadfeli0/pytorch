# Documentation: `torch/testing/_internal/distributed/_shard/sharded_tensor/_test_ops_common.py`

## File Metadata

- **Path**: `torch/testing/_internal/distributed/_shard/sharded_tensor/_test_ops_common.py`
- **Size**: 4,011 bytes (3.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
# mypy: allow-untyped-defs

import builtins

import torch
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec._internals import (
    get_chunked_dim_size,
    get_split_size,
)


def generate_chunk_sharding_specs_for_test(sharding_dim):
    return [
        ChunkShardingSpec(
            dim=sharding_dim,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        ),
        # Test different ordering. (Case 1)
        ChunkShardingSpec(
            dim=sharding_dim,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        ),
        # Test different ordering. (Case 2)
        ChunkShardingSpec(
            dim=sharding_dim,
            placements=[
                "rank:3/cuda:3",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
            ],
        ),
    ]


def generate_enumerable_sharding_specs_for_test():
    return [
        EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )
    ]


def generate_local_weight_sharding_params_for_test(
    local_weight, sharded_dim, gpu_num, spec, rank
):
    """
    Shard the local weight based the given spec, so we can compare against
    the one from sharded tensor.

    Args:
        local_weight: weight matrix to be sharded.
        sharded_dim: The dimension which we shard on.
        gpu_num: number of ranks.
        spec: sharding spec.
        rank: # of cuda process.

    Returns:
        start_pos: start position of sharded weight on the given rank.
        chunk_size: chunk size of sharded weight on the given rank.
    """
    sharding_dim_size = local_weight.size(sharded_dim)
    split_size = get_split_size(sharding_dim_size, gpu_num)
    current_offsets = 0
    start_pos = current_offsets
    for idx, placement in enumerate(spec.placements):
        chunk_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        if rank == placement.rank():
            start_pos = current_offsets
            break
        current_offsets += chunk_size
    return start_pos, chunk_size


def clone_module_parameter(module, param_name):
    """
    Clone a parameter from a given existing module.

    Args:
        module (:class:`torch.nn.Module`): Module whose parameter needs to be cloned.
        param_name (str): Name of the parameter of ``module`` that needs to be cloned.

    Returns: cloned tensor as :class:`torch.nn.Parameter`.
    """
    tensor = getattr(module, param_name)
    return torch.nn.Parameter(tensor.detach().clone())


def gen_binary_op_func(python_op, inplace=False):
    src_lines = ["def f(lhs, rhs):"]
    if "torch" in python_op:
        src_lines.append(f"  return {python_op}(lhs, rhs)\n")
    elif inplace:
        src_lines.append(f"  lhs {python_op}= rhs\n  return lhs\n")
    else:
        src_lines.append(f"  return lhs {python_op} rhs\n")

    code_str = "\n".join(src_lines)
    g = {"torch": torch}
    builtins.exec(code_str, g)
    return g["f"]

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `generate_chunk_sharding_specs_for_test`, `generate_enumerable_sharding_specs_for_test`, `generate_local_weight_sharding_params_for_test`, `clone_module_parameter`, `gen_binary_op_func`, `f`

**Key imports**: builtins, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal/distributed/_shard/sharded_tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `builtins`
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python torch/testing/_internal/distributed/_shard/sharded_tensor/_test_ops_common.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal/distributed/_shard/sharded_tensor`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_test_st_common.py_docs.md`](./_test_st_common.py_docs.md)


## Cross-References

- **File Documentation**: `_test_ops_common.py_docs.md`
- **Keyword Index**: `_test_ops_common.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
