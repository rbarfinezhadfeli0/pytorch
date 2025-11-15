# Documentation: `docs/torch/distributed/tensor/_ops/_conv_ops.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/_ops/_conv_ops.py_docs.md`
- **Size**: 6,114 bytes (5.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/tensor/_ops/_conv_ops.py`

## File Metadata

- **Path**: `torch/distributed/tensor/_ops/_conv_ops.py`
- **Size**: 3,488 bytes (3.41 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.utils import register_prop_rule


aten = torch.ops.aten


@register_prop_rule(aten.convolution.default)
def convolution_rules(op_schema: OpSchema) -> OutputSharding:
    (
        input_spec,
        weight_spec,
        bias_spec,
        stride,
        padding,
        dilation,
        _transposed,
        _output_padding,
        _groups,
    ) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    assert isinstance(bias_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None
    assert weight_spec.tensor_meta is not None
    in_shape = input_spec.tensor_meta.shape
    weight_shape = weight_spec.tensor_meta.shape
    assert isinstance(stride, list)
    assert isinstance(padding, list)
    assert isinstance(dilation, list)
    assert isinstance(weight_shape, torch.Size)
    out_conv_shape = [
        (d + 2 * padding[i] - dilation[i] * (weight_shape[i + 1] - 1) - 1) // stride[i]
        + 1
        for (i, d) in enumerate(in_shape[2:])
    ]
    output_shape = [in_shape[0], weight_shape[0]] + out_conv_shape
    output_stride = [1]
    for i in range(1, len(output_shape)):
        output_stride.insert(0, output_stride[0] * output_shape[-i])
    output_dim_map = input_spec.dim_map
    pending_sums = input_spec.sums

    tensor_meta = TensorMeta(
        torch.Size(output_shape),
        tuple(output_stride),
        input_spec.tensor_meta.dtype,
    )
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            output_dim_map,
            pending_sums,
            tensor_meta=tensor_meta,
        )
    )


@register_prop_rule(aten.convolution_backward.default)
def convolution_backward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]
    (
        grad_output_spec,
        input_spec,
        weight_spec,
        bias_shape_opt,
        _stride,
        _padding,
        _dilation,
        _transposed,
        _output_padding,
        _groups,
        _output_mask,
    ) = op_schema.args_schema

    assert isinstance(grad_output_spec, DTensorSpec)
    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    assert isinstance(bias_shape_opt, list)
    assert input_spec.tensor_meta is not None
    weight_tensor_meta = weight_spec.tensor_meta
    bias_tensor_meta = TensorMeta(
        torch.Size(bias_shape_opt),
        (1,),
        input_spec.tensor_meta.dtype,
    )

    grad_input_spec = input_spec
    grad_weight_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1, -1, -1, -1],
        [0],
        tensor_meta=weight_tensor_meta,
    )
    grad_bias_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1],
        [0],
        tensor_meta=bias_tensor_meta,
    )
    # TODO: actually the output_mask is not respected here, we should
    # set the corresponding spec to `None` if the output_mask is not `False`
    # for a certain output Tensor. This also applies to the conv handler
    # in torch/distributed/tensor/_tp_conv.py
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])

```



## High-Level Overview


This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `convolution_rules`, `convolution_backward_rules`

**Key imports**: torch, DTensorSpec, TensorMeta, OpSchema, OutputSharding, register_prop_rule


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor/_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed.tensor._dtensor_spec`: DTensorSpec, TensorMeta
- `torch.distributed.tensor._op_schema`: OpSchema, OutputSharding
- `torch.distributed.tensor._ops.utils`: register_prop_rule


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

- **File Documentation**: `_conv_ops.py_docs.md`
- **Keyword Index**: `_conv_ops.py_kw.md`
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

- **File Documentation**: `_conv_ops.py_docs.md_docs.md`
- **Keyword Index**: `_conv_ops.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
