# Documentation: `docs/torch/ao/quantization/backend_config/_qnnpack_pt2e.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/backend_config/_qnnpack_pt2e.py_docs.md`
- **Size**: 8,951 bytes (8.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/backend_config/_qnnpack_pt2e.py`

## File Metadata

- **Path**: `torch/ao/quantization/backend_config/_qnnpack_pt2e.py`
- **Size**: 6,431 bytes (6.28 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import operator

import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)


weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)


def get_linear_configs():
    linear_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]

    # TODO: need to fix the way we insert observers for this pattern
    # should be solved in the new fusion API
    # reason that this doesn't work: the pattern is a bit complicated and we don't
    # have a way to specify which input of the pattern we would like to observe
    # pattern:
    # bias input weight
    # \     |    /
    #  \    |   t
    #   \   |  /
    #    addmm
    # we want to observe "weight" as weight, but there is not way to convey this
    # information with current pattern language
    #
    # right now:
    # original:
    #         weight - t \
    #         input  - addmm
    # observed (no hack):
    #      weight - t - observer \
    #       input - observer - addmm
    # target:
    #      weight - observer - t \
    #        input - observer - addmm

    # def root_node_getter(node_pattern):
    #     addmm, bias, act, weight = node_pattern
    #     return addmm

    # linear_configs.append(
    #     BackendPatternConfig((torch.ops.aten.addmm.default, MatchAllNode, MatchAllNode, torch.ops.aten.t.default))
    #     .set_observation_type(observation_type)  # noqa: E131
    #     .set_dtype_configs(dtype_configs)
    #     ._set_root_node_getter(root_node_getter))

    linear_configs.append(
        BackendPatternConfig(torch.ops.aten.addmm.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 0})
    )
    # linear is decomposed to `t - mm` if bias is not present
    linear_configs.append(
        BackendPatternConfig(torch.ops.aten.mm.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1})
    )
    return linear_configs


def get_conv_configs():
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    conv_configs.append(
        BackendPatternConfig(torch.ops.aten.convolution.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    conv_configs.append(
        BackendPatternConfig(
            (torch.ops.aten.convolution.default, torch.ops.aten.relu.default)
        )
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    # TODO: remove when functionalization is supported in PT2 mode
    conv_configs.append(
        BackendPatternConfig(
            (torch.ops.aten.convolution.default, torch.ops.aten.relu_.default)
        )
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    return conv_configs


def get_pooling_configs():
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]

    def root_node_getter(node_pattern):
        _getitem, maxpool, _index = node_pattern
        return maxpool

    backend_pattern_configs.append(
        BackendPatternConfig()
        ._set_pattern_complex_format(
            (operator.getitem, torch.ops.aten.max_pool2d_with_indices.default, 0)
        )
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_root_node_getter(root_node_getter)
    )

    return backend_pattern_configs


def get_relu_configs():
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    backend_pattern_configs.append(
        BackendPatternConfig(torch.ops.aten.relu.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
    )
    return backend_pattern_configs


def get_binary_op_configs():
    binary_op_configs: list[BackendPatternConfig] = []
    dtype_configs = [weighted_op_quint8_dtype_config]
    num_tensor_args_to_observation_type_mapping = {
        # TODO: this is not used right now since we have extra check in prepare
        # will need to change this to NO_OBSERVER later after we implemented
        # Tensor dtype inference properly
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    for op_with_quantized_bop_scalar_variant in [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
    ]:
        bop_patterns = [
            (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu.default),
            op_with_quantized_bop_scalar_variant,
            # TODO: remove when functionalization is supported in pt2_mode
            (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu_.default),
        ]
        binary_op_configs.extend(
            BackendPatternConfig(bop_pattern)
            .set_dtype_configs(dtype_configs)  # noqa: E131
            ._set_num_tensor_args_to_observation_type(
                num_tensor_args_to_observation_type_mapping
            )
            for bop_pattern in bop_patterns
        )

    return binary_op_configs


def get_qnnpack_pt2e_backend_config():
    return (
        BackendConfig("qnnpack_pytorch_2.0_export")
        .set_backend_pattern_configs(get_linear_configs())
        .set_backend_pattern_configs(get_binary_op_configs())
        .set_backend_pattern_configs(get_conv_configs())
        .set_backend_pattern_configs(get_pooling_configs())
        .set_backend_pattern_configs(get_relu_configs())
    )

```



## High-Level Overview


This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_linear_configs`, `root_node_getter`, `get_conv_configs`, `get_pooling_configs`, `root_node_getter`, `get_relu_configs`, `get_binary_op_configs`, `get_qnnpack_pt2e_backend_config`

**Key imports**: operator, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/backend_config`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
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

Files in the same folder (`torch/ao/quantization/backend_config`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`executorch.py_docs.md`](./executorch.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`_common_operator_config_utils.py_docs.md`](./_common_operator_config_utils.py_docs.md)
- [`fbgemm.py_docs.md`](./fbgemm.py_docs.md)
- [`x86.py_docs.md`](./x86.py_docs.md)
- [`tensorrt.py_docs.md`](./tensorrt.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`native.py_docs.md`](./native.py_docs.md)
- [`onednn.py_docs.md`](./onednn.py_docs.md)


## Cross-References

- **File Documentation**: `_qnnpack_pt2e.py_docs.md`
- **Keyword Index**: `_qnnpack_pt2e.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/quantization/backend_config`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization/backend_config`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/ao/quantization/backend_config`):

- [`onednn.py_docs.md_docs.md`](./onednn.py_docs.md_docs.md)
- [`backend_config.py_docs.md_docs.md`](./backend_config.py_docs.md_docs.md)
- [`onednn.py_kw.md_docs.md`](./onednn.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`executorch.py_docs.md_docs.md`](./executorch.py_docs.md_docs.md)
- [`x86.py_docs.md_docs.md`](./x86.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`qnnpack.py_docs.md_docs.md`](./qnnpack.py_docs.md_docs.md)
- [`executorch.py_kw.md_docs.md`](./executorch.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_qnnpack_pt2e.py_docs.md_docs.md`
- **Keyword Index**: `_qnnpack_pt2e.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
