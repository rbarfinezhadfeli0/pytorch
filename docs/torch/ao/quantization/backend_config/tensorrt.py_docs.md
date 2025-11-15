# Documentation: `torch/ao/quantization/backend_config/tensorrt.py`

## File Metadata

- **Path**: `torch/ao/quantization/backend_config/tensorrt.py`
- **Size**: 3,021 bytes (2.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch

from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_conv_configs,
    _get_linear_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)
from .backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)


__all__ = [
    "get_tensorrt_backend_config",
    "get_tensorrt_backend_config_dict",
]


def get_tensorrt_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for the TensorRT backend.
    NOTE: Current api will change in the future, it's just to unblock experimentation for
    new backends, please don't use it right now.
    TODO: add a README when it's more stable
    """
    # dtype configs
    weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )
    non_weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
    )

    addmm_config = (
        BackendPatternConfig(torch.addmm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .add_dtype_config(weighted_op_qint8_dtype_config)
        ._set_input_type_to_index(
            {
                "bias": 0,
                "input": 1,
                "weight": 2,
            }
        )
    )
    cat_config = (
        BackendPatternConfig(torch.cat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .add_dtype_config(non_weighted_op_qint8_dtype_config)
    )
    conv_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    linear_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    binary_op_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    tensor_info_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    # there might be things not supported in fx2trt, but it will error out
    # during fx2trt conversion and can support them after that
    return (
        BackendConfig("tensorrt")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_config(addmm_config)
        .set_backend_pattern_config(cat_config)
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_tensor_info_op_configs(tensor_info_op_dtype_configs)
        )
    )


def get_tensorrt_backend_config_dict():
    """
    Return the `BackendConfig` for the TensorRT backend in dictionary form.
    """
    return get_tensorrt_backend_config().to_dict()

```



## High-Level Overview

"""    Return the `BackendConfig` for the TensorRT backend.    NOTE: Current api will change in the future, it's just to unblock experimentation for    new backends, please don't use it right now.    TODO: add a README when it's more stable

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_tensorrt_backend_config`, `get_tensorrt_backend_config_dict`

**Key imports**: torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/backend_config`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

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
- [`README.md_docs.md`](./README.md_docs.md)
- [`native.py_docs.md`](./native.py_docs.md)
- [`onednn.py_docs.md`](./onednn.py_docs.md)


## Cross-References

- **File Documentation**: `tensorrt.py_docs.md`
- **Keyword Index**: `tensorrt.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
