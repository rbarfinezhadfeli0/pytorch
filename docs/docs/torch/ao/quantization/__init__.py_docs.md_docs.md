# Documentation: `docs/torch/ao/quantization/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/__init__.py_docs.md`
- **Size**: 10,259 bytes (10.02 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/__init__.py`

## File Metadata

- **Path**: `torch/ao/quantization/__init__.py`
- **Size**: 7,613 bytes (7.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs

import sys
from collections.abc import Callable
from typing import Optional, Union

import torch
from torch import Tensor

from .fake_quantize import *  # noqa: F403
from .fuse_modules import fuse_modules, fuse_modules_qat  # noqa: F403
from .fuser_method_mappings import *  # noqa: F403
from .observer import *  # noqa: F403
from .pt2e._numeric_debugger import (  # noqa: F401
    compare_results,
    CUSTOM_KEY,
    extract_results_from_loggers,
    generate_numeric_debug_handle,
    NUMERIC_DEBUG_HANDLE_KEY,
    prepare_for_propagation_comparison,
)
from .pt2e.export_utils import (
    _allow_exported_model_train_eval as allow_exported_model_train_eval,
    _move_exported_model_to_eval as move_exported_model_to_eval,
    _move_exported_model_to_train as move_exported_model_to_train,
)

# pyrefly: ignore [deprecated]
from .qconfig import *  # noqa: F403
from .qconfig_mapping import *  # noqa: F403
from .quant_type import *  # noqa: F403
from .quantization_mappings import *  # noqa: F403 # type: ignore[no-redef]
from .quantize import *  # noqa: F403
from .quantize_jit import *  # noqa: F403
from .stubs import *  # noqa: F403


# ensure __module__ is set correctly for public APIs
if sys.version_info < (3, 12):
    ObserverOrFakeQuantize = Union[ObserverBase, FakeQuantizeBase]
    ObserverOrFakeQuantize.__module__ = "torch.ao.quantization"
else:
    from typing import TypeAliasType

    ObserverOrFakeQuantize = TypeAliasType(
        "ObserverOrFakeQuantize", ObserverBase | FakeQuantizeBase
    )

for _f in [
    compare_results,
    extract_results_from_loggers,
    generate_numeric_debug_handle,
    prepare_for_propagation_comparison,
]:
    _f.__module__ = "torch.ao.quantization"

__all__ = [
    "DeQuantStub",
    "FakeQuantize",
    "FakeQuantizeBase",
    "FixedQParamsFakeQuantize",
    "FixedQParamsObserver",
    "FusedMovingAvgObsFakeQuantize",
    "HistogramObserver",
    "MatchAllNode",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "NoopObserver",
    "ObserverBase",
    "ObserverOrFakeQuantize",
    "Pattern",
    "PerChannelMinMaxObserver",
    "PlaceholderObserver",
    "QConfig",
    "QConfigAny",
    "QConfigDynamic",
    "QConfigMapping",
    "QuantStub",
    "QuantType",
    "QuantWrapper",
    "RecordingObserver",
    "ReuseInputObserver",
    "UniformQuantizationObserverBase",
    "add_quant_dequant",
    "convert",
    "convert_dynamic_jit",
    "convert_jit",
    "default_affine_fixed_qparams_fake_quant",
    "default_affine_fixed_qparams_observer",
    "default_debug_observer",
    "default_dynamic_fake_quant",
    "default_dynamic_quant_observer",
    "default_embedding_fake_quant",
    "default_embedding_fake_quant_4bit",
    "default_eval_fn",
    "default_fake_quant",
    "default_fixed_qparams_range_0to1_fake_quant",
    "default_fixed_qparams_range_0to1_observer",
    "default_fixed_qparams_range_neg1to1_fake_quant",
    "default_fixed_qparams_range_neg1to1_observer",
    "default_float_qparams_observer",
    "default_float_qparams_observer_4bit",
    "default_fused_act_fake_quant",
    "default_fused_per_channel_wt_fake_quant",
    "default_fused_wt_fake_quant",
    "default_histogram_fake_quant",
    "default_histogram_observer",
    "default_observer",
    "default_per_channel_weight_fake_quant",
    "default_per_channel_weight_observer",
    "default_placeholder_observer",
    "default_reuse_input_observer",
    "default_symmetric_fixed_qparams_fake_quant",
    "default_symmetric_fixed_qparams_observer",
    "default_weight_fake_quant",
    "default_weight_observer",
    "disable_fake_quant",
    "disable_observer",
    "enable_fake_quant",
    "enable_observer",
    "fuse_conv_bn",
    "fuse_conv_bn_jit",
    "fuse_conv_bn_relu",
    "fuse_convtranspose_bn",
    "fuse_linear_bn",
    "fuse_modules",
    "fuse_modules_qat",
    "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
    "fused_wt_fake_quant_range_neg_127_to_127",
    "get_combined_dict",
    "get_default_compare_output_module_list",
    "get_default_custom_config_dict",
    "get_default_dynamic_quant_module_mappings",
    "get_default_dynamic_sparse_quant_module_mappings",
    "get_default_float_to_quantized_operator_mappings",
    "get_default_qat_module_mappings",
    "get_default_qat_qconfig",
    "get_default_qat_qconfig_dict",
    "get_default_qat_qconfig_mapping",
    "get_default_qconfig",
    "get_default_qconfig_dict",
    "get_default_qconfig_mapping",
    "get_default_qconfig_propagation_list",
    "get_default_static_quant_module_mappings",
    "get_default_static_quant_reference_module_mappings",
    "get_default_static_sparse_quant_module_mappings",
    "get_dynamic_quant_module_class",
    "get_embedding_qat_module_mappings",
    "get_embedding_static_quant_module_mappings",
    "get_fuser_method",
    "get_fuser_method_new",
    "get_observer_state_dict",
    "get_quantized_operator",
    "get_static_quant_module_class",
    "load_observer_state_dict",
    "move_exported_model_to_eval",
    "move_exported_model_to_train",
    "allow_exported_model_train_eval",
    "no_observer_set",
    "per_channel_weight_observer_range_neg_127_to_127",
    "prepare",
    "prepare_dynamic_jit",
    "prepare_jit",
    "prepare_qat",
    "propagate_qconfig_",
    "qconfig_equals",
    "quantize",
    "quantize_dynamic",
    "quantize_dynamic_jit",
    "quantize_jit",
    "quantize_qat",
    "script_qconfig",
    "script_qconfig_dict",
    "swap_module",
    "weight_observer_range_neg_127_to_127",
    "generate_numeric_debug_handle",
    "CUSTOM_KEY",
    "NUMERIC_DEBUG_HANDLE_KEY",
    "prepare_for_propagation_comparison",
    "extract_results_from_loggers",
    "compare_results",
    # from torchao, should be merged with torchao
    # in the future
    "AffineQuantizedObserverBase",
    "Granularity",
    "MappingType",
    "PerAxis",
    "PerBlock",
    "PerGroup",
    "PerRow",
    "PerTensor",
    "PerToken",
    "TorchAODType",
    "ZeroPointDomain",
    "get_block_size",
]


def default_eval_fn(model, calib_data):
    r"""Define the default evaluation function.

    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data, _target in calib_data:
        model(data)


class _DerivedObserverOrFakeQuantize(ObserverBase):
    r"""This observer is used to describe an observer whose quantization parameters
    are derived from other observers
    """

    def __init__(
        self,
        dtype: torch.dtype,
        obs_or_fqs: list[ObserverOrFakeQuantize],
        derive_qparams_fn: Callable[
            [list[ObserverOrFakeQuantize]], tuple[Tensor, Tensor]
        ],
        quant_min: int | None = None,
        quant_max: int | None = None,
        qscheme: torch.qscheme | None = None,
        ch_axis: int | None = None,
    ):
        super().__init__(dtype)
        self.obs_or_fqs = obs_or_fqs
        self.derive_qparams_fn = derive_qparams_fn
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.qscheme = qscheme
        self.ch_axis = ch_axis

        from .utils import is_per_channel

        if is_per_channel(self.qscheme):
            if self.ch_axis is None:
                raise AssertionError(
                    "Must provide a valid ch_axis if qscheme is per channel"
                )

    def forward(self, x: Tensor) -> Tensor:
        return x

    def calculate_qparams(self):  # type:ignore[override]
        return self.derive_qparams_fn(self.obs_or_fqs)

```



## High-Level Overview


This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_DerivedObserverOrFakeQuantize`

**Functions defined**: `default_eval_fn`, `__init__`, `forward`, `calculate_qparams`

**Key imports**: sys, Callable, Optional, Union, torch, Tensor, fuse_modules, fuse_modules_qat  , TypeAliasType, is_per_channel


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `collections.abc`: Callable
- `typing`: Optional, Union
- `torch`
- `.fuse_modules`: fuse_modules, fuse_modules_qat  
- `.utils`: is_per_channel


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/ao/quantization`):

- [`quant_type.py_docs.md`](./quant_type.py_docs.md)
- [`fake_quantize.py_docs.md`](./fake_quantize.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`fuse_modules.py_docs.md`](./fuse_modules.py_docs.md)
- [`_equalize.py_docs.md`](./_equalize.py_docs.md)
- [`quantize.py_docs.md`](./quantize.py_docs.md)
- [`_learnable_fake_quantize.py_docs.md`](./_learnable_fake_quantize.py_docs.md)
- [`observer.py_docs.md`](./observer.py_docs.md)
- [`pattern.md_docs.md`](./pattern.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/quantization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/ao/quantization`):

- [`_correct_bias.py_kw.md_docs.md`](./_correct_bias.py_kw.md_docs.md)
- [`quant_type.py_kw.md_docs.md`](./quant_type.py_kw.md_docs.md)
- [`qconfig.py_docs.md_docs.md`](./qconfig.py_docs.md_docs.md)
- [`_learnable_fake_quantize.py_kw.md_docs.md`](./_learnable_fake_quantize.py_kw.md_docs.md)
- [`quantize_fx.py_kw.md_docs.md`](./quantize_fx.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`observer.py_kw.md_docs.md`](./observer.py_kw.md_docs.md)
- [`fuser_method_mappings.py_kw.md_docs.md`](./fuser_method_mappings.py_kw.md_docs.md)
- [`quantize.py_kw.md_docs.md`](./quantize.py_kw.md_docs.md)
- [`qconfig_mapping.py_kw.md_docs.md`](./qconfig_mapping.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
