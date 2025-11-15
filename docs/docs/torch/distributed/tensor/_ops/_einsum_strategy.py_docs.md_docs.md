# Documentation: `docs/torch/distributed/tensor/_ops/_einsum_strategy.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/_ops/_einsum_strategy.py_docs.md`
- **Size**: 9,873 bytes (9.64 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/tensor/_ops/_einsum_strategy.py`

## File Metadata

- **Path**: `torch/distributed/tensor/_ops/_einsum_strategy.py`
- **Size**: 7,175 bytes (7.01 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import itertools
from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSpec, OpStrategy
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


@dataclass
class EinsumDims:
    contracting_dims: list[str]
    batch_dims: list[str]
    lhs_out_only_dims: list[str]
    rhs_out_only_dims: list[str]

    @classmethod
    def parse_equation(cls, equation: str) -> tuple[list[str], str]:
        # parse einop equation and extract arg specs
        """
        Parse the einsum equation str to input dim chars and output dim char
        """
        inputs, outputs = equation.split("->")
        input_dims, output_dims = inputs.split(","), outputs.split(",")

        # NOTE: only support at most two inputs, and single output
        # extend to support more inputs if needed in future
        assert len(input_dims) <= 2, "Only support at most two inputs"
        assert len(output_dims) == 1, "Only support single output"
        output_dim = output_dims[0]
        return input_dims, output_dim

    @classmethod
    def parse_dims(cls, input_dims: list[str], output_dim: str) -> "EinsumDims":
        """
        Parse the dims and extract the contracting, batch, and free dimensions
        for the left and right hand sides.
        """
        dim_char_set: set[str] = set()
        for input_dim in input_dims:
            dim_char_set.update(input_dim)

        # get a deterministic order of all dim chars
        all_dim_chars = sorted(dim_char_set)

        # parse input and output dimensions
        lhs_out_only_dims, rhs_out_only_dims = [], []
        batch_dims, contracting_dims = [], []

        for dim_char in all_dim_chars:
            if dim_char not in output_dim:
                contracting_dims.append(dim_char)
            else:
                is_batch_dim = True
                for input_dim in input_dims:
                    is_batch_dim = is_batch_dim and dim_char in input_dim

                if is_batch_dim:
                    batch_dims.append(dim_char)
                else:
                    assert len(input_dims) == 2, (
                        "free dimension only supported for two inputs!"
                    )
                    lhs, rhs = input_dims
                    if dim_char in lhs:
                        lhs_out_only_dims.append(dim_char)
                    elif dim_char in rhs:
                        rhs_out_only_dims.append(dim_char)
                    else:
                        raise RuntimeError("Invalid dimension character")

        return cls(
            contracting_dims=contracting_dims,
            batch_dims=batch_dims,
            lhs_out_only_dims=lhs_out_only_dims,
            rhs_out_only_dims=rhs_out_only_dims,
        )


def gen_einsum_strategies(
    equation: str,
    mesh: DeviceMesh,
    *,
    linearity: bool = False,
) -> OpStrategy:
    """
    Generate a strategy list for the ops that follow einsum style notation.

    In principle, each mesh dim is independent of other device mesh dim when we
    generate strategies. So we generate strategy over each device mesh dim and
    do product combination on all mesh dims. We basically follow the below rule
    for each device mesh dim:

    1. Shard on contracting dim: When both inputs shard on contracting dim over
       the same device dim. The result will be Partial over that device dim.

    2. Shard on noncontracting dim:
        2.1: Shard on batch dim: output, both inputs all should shard on batch
        dim.
        2.2: Shard on lhs only dim or rhs only dim: both output and lhs or rhs
        input should shard on this free dim.

    3. Linearity (Partial): If enabled, set Partial on output and inputs over
       the same device mesh dim.
    """
    # parse einop equation and extract dims
    input_dims, output_dim = EinsumDims.parse_equation(equation)
    edims = EinsumDims.parse_dims(input_dims, output_dim)
    all_mesh_dim_strategies = []

    # generate strategies for each mesh dim and do cartesian product for final strategy. E.g., for a 2D mesh, we can have [P(),R,R]
    strategies_over_one_mesh_dim = []

    # placement list stores placements of [output, input1, input2, ...]
    # first we always have replicate all for inputs and output
    placement_list: list[Placement] = [Replicate()] * (len(input_dims) + 1)
    strategies_over_one_mesh_dim.append(placement_list)

    # split batch dim
    for batch_dim in edims.batch_dims:
        output_batch_dim = output_dim.index(batch_dim)
        placement_list = [Shard(output_batch_dim)]
        for input_dim in input_dims:
            input_batch_dim = input_dim.index(batch_dim)
            placement_list.append(Shard(input_batch_dim))

        strategies_over_one_mesh_dim.append(placement_list)

    # split contracting dim
    for contracting_dim in edims.contracting_dims:
        # Contracting dim can shard on same device axis for both inputs. This
        # results in the output being Partial on that device axis. For example:
        # bmk_{x},k_{x}n -> bmn{Ux} (becomes partial over device axis x)
        placement_list = [Partial()]
        for input_dim in input_dims:
            input_contracting_dim = input_dim.index(contracting_dim)
            placement_list.append(Shard(input_contracting_dim))

        strategies_over_one_mesh_dim.append(placement_list)

    # split lhs free dim
    for lhs_dim in edims.lhs_out_only_dims:
        lhs_free_dim_output = output_dim.index(lhs_dim)
        lhs_free_dim_input = input_dims[0].index(lhs_dim)
        # this means split the lhs input and output
        # i.e. S(0), R -> S(0)
        lhs_placement_list: list[Placement] = [
            Shard(lhs_free_dim_output),
            Shard(lhs_free_dim_input),
            Replicate(),
        ]
        strategies_over_one_mesh_dim.append(lhs_placement_list)

    # split rhs free dim
    for rhs_dim in edims.rhs_out_only_dims:
        rhs_free_dim_output = output_dim.index(rhs_dim)
        rhs_free_dim_input = input_dims[1].index(rhs_dim)
        rhs_placement_list: list[Placement] = [
            Shard(rhs_free_dim_output),
            Replicate(),
            Shard(rhs_free_dim_input),
        ]
        strategies_over_one_mesh_dim.append(rhs_placement_list)

    # linearity strategy
    if linearity:
        linearity_placement_list: list[Placement] = [Partial()]
        for _ in input_dims:
            linearity_placement_list.append(Partial())
        strategies_over_one_mesh_dim.append(linearity_placement_list)

    # generate strategies for entire mesh
    all_mesh_dim_strategies = [strategies_over_one_mesh_dim] * mesh.ndim
    strategy_combs = itertools.product(*all_mesh_dim_strategies)
    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = [DTensorSpec(mesh, tuple(specs)) for specs in zip(*strategy_comb)]
        strat = OpSpec(output_specs=spec_list[0], input_specs=spec_list[1:])
        all_strategies.append(strat)

    return OpStrategy(all_strategies)

```



## High-Level Overview

"""        Parse the einsum equation str to input dim chars and output dim char

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `EinsumDims`

**Functions defined**: `parse_equation`, `parse_dims`, `gen_einsum_strategies`

**Key imports**: itertools, dataclass, DeviceMesh, DTensorSpec, OpSpec, OpStrategy


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor/_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `dataclasses`: dataclass
- `torch.distributed.device_mesh`: DeviceMesh
- `torch.distributed.tensor._dtensor_spec`: DTensorSpec
- `torch.distributed.tensor._op_schema`: OpSpec, OpStrategy


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
- [`_matrix_ops.py_docs.md`](./_matrix_ops.py_docs.md)
- [`_pointwise_ops.py_docs.md`](./_pointwise_ops.py_docs.md)
- [`_math_ops.py_docs.md`](./_math_ops.py_docs.md)
- [`_mask_buffer.py_docs.md`](./_mask_buffer.py_docs.md)
- [`_common_rules.py_docs.md`](./_common_rules.py_docs.md)


## Cross-References

- **File Documentation**: `_einsum_strategy.py_docs.md`
- **Keyword Index**: `_einsum_strategy.py_kw.md`
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
- [`_embedding_ops.py_docs.md_docs.md`](./_embedding_ops.py_docs.md_docs.md)
- [`_einsum_strategy.py_kw.md_docs.md`](./_einsum_strategy.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_einsum_strategy.py_docs.md_docs.md`
- **Keyword Index**: `_einsum_strategy.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
