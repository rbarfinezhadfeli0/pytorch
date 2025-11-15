# Documentation: `docs/torch/distributed/tensor/_ops/_embedding_ops.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/_ops/_embedding_ops.py_docs.md`
- **Size**: 6,806 bytes (6.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/tensor/_ops/_embedding_ops.py`

## File Metadata

- **Path**: `torch/distributed/tensor/_ops/_embedding_ops.py`
- **Size**: 4,286 bytes (4.19 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import cast

import torch
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    StrategyType,
)
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import (
    MaskPartial,
    Partial,
    Replicate,
    Shard,
)


aten = torch.ops.aten


@register_op_strategy(aten.embedding.default)
def embedding_strategy(op_schema: OpSchema) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    """
    weight_strategy = cast(OpStrategy, op_schema.args_schema[0])
    indices_strategy = cast(OpStrategy, op_schema.args_schema[1])
    mesh = op_schema.get_mesh_from_args()

    weight_shape = weight_strategy.shape
    indices_shape = indices_strategy.shape
    output_emd_dim = len(indices_shape)

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, weight, input_indices]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 3
    single_mesh_dim_strategies.append(all_replicate)

    # colwise sharding, output shard on last dim, weight shard on dim 1, input replicate
    colwise_sharding: PlacementList = [Shard(output_emd_dim), Shard(1), Replicate()]
    single_mesh_dim_strategies.append(colwise_sharding)

    # rowwise sharding, output is embedding partial, weight shard on dim 0, input accepts embedding partial
    embedding_partial_placement = MaskPartial(offset_shape=weight_shape, offset_dim=0)

    # NOTE we want to reuse the same mask partial placement so that we can reuse the same mask that generates
    # from the input indices and use it for output reduction
    rowwise_sharding: PlacementList = [
        embedding_partial_placement,
        Shard(0),
        embedding_partial_placement,
    ]
    single_mesh_dim_strategies.append(rowwise_sharding)

    # batch dim sharding, weight replicated, input can shard on any dim, output follows input
    for input_dim in range(len(indices_shape)):
        batch_sharding: PlacementList = [
            Shard(input_dim),
            Replicate(),
            Shard(input_dim),
        ]
        single_mesh_dim_strategies.append(batch_sharding)

    return expand_to_full_mesh_op_strategy(mesh, op_schema, single_mesh_dim_strategies)


@register_op_strategy(aten.embedding_dense_backward.default)
def embedding_dense_backward_strategy(op_schema: OpSchema) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    """
    grad_out_strategy = cast(OpStrategy, op_schema.args_schema[0])
    indices_strategy = cast(OpStrategy, op_schema.args_schema[1])
    mesh = op_schema.get_mesh_from_args()

    grad_out_shape = grad_out_strategy.shape
    indices_shape = indices_strategy.shape
    grad_out_ndim = len(grad_out_shape)

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, weight, input_indices]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 3
    single_mesh_dim_strategies.append(all_replicate)

    # colwise sharding backward, grad_out shard on last dim, input replicate,
    # weight grad shard colwise
    colwise_sharding: PlacementList = [Shard(1), Shard(grad_out_ndim - 1), Replicate()]
    single_mesh_dim_strategies.append(colwise_sharding)

    # batch dim sharding, weight replicated, grad_out/input have same sharding
    # that can shard on any dim, weight grad partial
    for input_dim in range(len(indices_shape)):
        batch_sharding: PlacementList = [Partial(), Shard(input_dim), Shard(input_dim)]
        single_mesh_dim_strategies.append(batch_sharding)

    # grad_out partial, input replicate, weight grad keep partial
    partial_sharding: PlacementList = [Partial(), Partial(), Replicate()]
    single_mesh_dim_strategies.append(partial_sharding)

    return expand_to_full_mesh_op_strategy(mesh, op_schema, single_mesh_dim_strategies)

```



## High-Level Overview

"""    This strategy handles embedding op. We have two possible embedding shardings:    rowwise and colwise

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `embedding_strategy`, `embedding_dense_backward_strategy`

**Key imports**: cast, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor/_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: cast
- `torch`


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

Files in the same folder (`torch/distributed/tensor/_ops`):

- [`_view_ops.py_docs.md`](./_view_ops.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_tensor_ops.py_docs.md`](./_tensor_ops.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`_einsum_strategy.py_docs.md`](./_einsum_strategy.py_docs.md)
- [`_matrix_ops.py_docs.md`](./_matrix_ops.py_docs.md)
- [`_pointwise_ops.py_docs.md`](./_pointwise_ops.py_docs.md)
- [`_math_ops.py_docs.md`](./_math_ops.py_docs.md)
- [`_mask_buffer.py_docs.md`](./_mask_buffer.py_docs.md)
- [`_common_rules.py_docs.md`](./_common_rules.py_docs.md)


## Cross-References

- **File Documentation**: `_embedding_ops.py_docs.md`
- **Keyword Index**: `_embedding_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/tensor/_ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/tensor/_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/distributed/tensor/_ops`):

- [`_tensor_ops.py_docs.md_docs.md`](./_tensor_ops.py_docs.md_docs.md)
- [`_matrix_ops.py_docs.md_docs.md`](./_matrix_ops.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`_matrix_ops.py_kw.md_docs.md`](./_matrix_ops.py_kw.md_docs.md)
- [`_random_ops.py_kw.md_docs.md`](./_random_ops.py_kw.md_docs.md)
- [`_pointwise_ops.py_docs.md_docs.md`](./_pointwise_ops.py_docs.md_docs.md)
- [`_tensor_ops.py_kw.md_docs.md`](./_tensor_ops.py_kw.md_docs.md)
- [`_math_ops.py_kw.md_docs.md`](./_math_ops.py_kw.md_docs.md)
- [`_einsum_strategy.py_kw.md_docs.md`](./_einsum_strategy.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_embedding_ops.py_docs.md_docs.md`
- **Keyword Index**: `_embedding_ops.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
