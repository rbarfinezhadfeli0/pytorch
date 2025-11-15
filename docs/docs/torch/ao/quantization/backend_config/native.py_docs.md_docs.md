# Documentation: `docs/torch/ao/quantization/backend_config/native.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/backend_config/native.py_docs.md`
- **Size**: 10,717 bytes (10.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/backend_config/native.py`

## File Metadata

- **Path**: `torch/ao/quantization/backend_config/native.py`
- **Size**: 8,242 bytes (8.05 KB)
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
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_ln_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)
from .backend_config import BackendConfig, DTypeConfig


__all__ = [
    "get_test_only_legacy_native_backend_config",
    "default_op_quint8_dtype_config",
    "default_op_fp16_dtype_config",
    "default_dynamic_int8_dtype_config",
    "default_dynamic_float16_dtype_config",
    "input_output_only_quint8_dtype_config",
    "weight_only_quint8_dtype_config",
    "weight_only_quint4x2_dtype_config",
    "get_native_backend_config",
    "get_native_backend_config_dict",
    "get_test_only_legacy_native_backend_config_dict",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

# weighted op int8 dtype config
# this is config for ops that has quantized weights, like linear, conv
weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    # currently the dtype check is not yet enabled, so we provided the dtype_configs but
    # it is not really used yet,
    # we will enable it a bit later after we moved everything to backend_config_dict
    is_dynamic=True,
)

default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    # currently the dtype check is not yet enabled, so we provided the dtype_configs but
    # it is not really used yet,
    # we will enable it a bit later after we moved everything to backend_config_dict
    is_dynamic=True,
)

# Needed for LayerNorm and f.layer_norm, since currently the kernel only supports float weights
input_output_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.float,
    bias_dtype=torch.float,
)

weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_test_only_legacy_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional fp16 ops.
    """
    conv_dtype_configs = [weighted_op_quint8_dtype_config]
    linear_dtype_configs = [
        weighted_op_quint8_dtype_config,
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
        default_op_fp16_dtype_config,
    ]
    binary_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    default_op_dtype_configs = [default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    tensor_info_op_dtype_configs = [
        default_op_quint8_dtype_config,
    ]
    rnn_op_dtype_configs = [
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        weight_only_quint8_dtype_config,
        weight_only_quint4x2_dtype_config,
    ]
    layer_norm_op_dtype_configs = [input_output_only_quint8_dtype_config]
    return (
        BackendConfig("_native_and_fp16")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_tensor_info_op_configs(tensor_info_op_dtype_configs)
        )
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs))
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_embedding_op_configs(embedding_op_dtype_configs)
        )
    )


def get_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack).
    """
    # TODO: express this BackendConfig as a union of the FBGEMM and QNNPACK BackendConfigs
    conv_dtype_configs = [weighted_op_quint8_dtype_config]
    linear_dtype_configs = [
        weighted_op_quint8_dtype_config,
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [default_op_quint8_dtype_config]
    default_op_dtype_configs = [default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [default_op_quint8_dtype_config]
    share_qparams_op_dtype_configs = [default_op_quint8_dtype_config]
    tensor_info_op_dtype_configs = [default_op_quint8_dtype_config]
    rnn_op_dtype_configs = [
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        weight_only_quint8_dtype_config,
        weight_only_quint4x2_dtype_config,
    ]
    layer_norm_op_dtype_configs = [input_output_only_quint8_dtype_config]
    return (
        BackendConfig("native")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_tensor_info_op_configs(tensor_info_op_dtype_configs)
        )
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs))
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_embedding_op_configs(embedding_op_dtype_configs)
        )
    )


def get_native_backend_config_dict():
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) in dictionary form.
    """
    return get_native_backend_config().to_dict()


def get_test_only_legacy_native_backend_config_dict():
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional
    fp16 ops in dictionary form.
    """
    return get_test_only_legacy_native_backend_config().to_dict()

```



## High-Level Overview


This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_test_only_legacy_native_backend_config`, `get_native_backend_config`, `get_native_backend_config_dict`, `get_test_only_legacy_native_backend_config_dict`

**Key imports**: torch, BackendConfig, DTypeConfig


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/backend_config`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `.backend_config`: BackendConfig, DTypeConfig


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
- [`onednn.py_docs.md`](./onednn.py_docs.md)


## Cross-References

- **File Documentation**: `native.py_docs.md`
- **Keyword Index**: `native.py_kw.md`
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
- [`_qnnpack_pt2e.py_docs.md_docs.md`](./_qnnpack_pt2e.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`qnnpack.py_docs.md_docs.md`](./qnnpack.py_docs.md_docs.md)
- [`executorch.py_kw.md_docs.md`](./executorch.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `native.py_docs.md_docs.md`
- **Keyword Index**: `native.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
