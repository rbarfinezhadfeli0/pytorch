# Documentation: _qnnpack_pt2e.py

## File Metadata
- **Path**: `torch/ao/quantization/backend_config/_qnnpack_pt2e.py`
- **Size**: 6431 bytes
- **Lines**: 181
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Functions
This file defines 8 function(s): get_linear_configs, root_node_getter, get_conv_configs, get_pooling_configs, root_node_getter, get_relu_configs, get_binary_op_configs, get_qnnpack_pt2e_backend_config


## Key Components

The file contains 468 words across 181 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 6431 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
