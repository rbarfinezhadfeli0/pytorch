# Documentation: `docs/torch/ao/quantization/fx/pattern_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/fx/pattern_utils.py_docs.md`
- **Size**: 6,603 bytes (6.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/fx/pattern_utils.py`

## File Metadata

- **Path**: `torch/ao/quantization/fx/pattern_utils.py`
- **Size**: 3,668 bytes (3.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import copy
from collections import OrderedDict
from typing import Any

from torch.ao.quantization.fake_quantize import FixedQParamsFakeQuantize
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.utils import Pattern


__all__ = [
    "get_default_fusion_patterns",
    "get_default_quant_patterns",
    "get_default_output_activation_post_process_map",
]

# TODO(future PR): fix the typing on QuantizeHandler (currently a circular dependency)
QuantizeHandler = Any

# pattern for conv bn fusion
_DEFAULT_FUSION_PATTERNS: dict[Pattern, QuantizeHandler] = OrderedDict()


def _register_fusion_pattern(pattern):
    def insert(fn):
        _DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn

    return insert


def get_default_fusion_patterns() -> dict[Pattern, QuantizeHandler]:
    return copy.copy(_DEFAULT_FUSION_PATTERNS)


_DEFAULT_QUANTIZATION_PATTERNS: dict[Pattern, QuantizeHandler] = OrderedDict()

# Mapping from pattern to activation_post_process(observer/fake_quant) constructor for output activation
# e.g. pattern: torch.sigmoid,
#      output_activation_post_process: default_fixed_qparams_range_0to1_fake_quant
_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP: dict[Pattern, QuantizeHandler] = {}
_DEFAULT_OUTPUT_OBSERVER_MAP: dict[Pattern, QuantizeHandler] = {}


# Register pattern for both static quantization and qat
def _register_quant_pattern(pattern, fixed_qparams_observer=None):
    def insert(fn):
        _DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        if fixed_qparams_observer is not None:
            _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP[pattern] = (
                FixedQParamsFakeQuantize.with_args(observer=fixed_qparams_observer)
            )
            _DEFAULT_OUTPUT_OBSERVER_MAP[pattern] = fixed_qparams_observer
        return fn

    return insert


# Get patterns for both static quantization and qat
def get_default_quant_patterns() -> dict[Pattern, QuantizeHandler]:
    return copy.copy(_DEFAULT_QUANTIZATION_PATTERNS)


# a map from pattern to output activation post process constructor
# e.g. torch.sigmoid -> default_affine_fixed_qparam_fake_quant
def get_default_output_activation_post_process_map(
    is_training,
) -> dict[Pattern, ObserverBase]:
    if is_training:
        return copy.copy(_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP)
    else:
        return copy.copy(_DEFAULT_OUTPUT_OBSERVER_MAP)


# Example use of register pattern function:
# @_register_fusion_pattern(torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
# class ConvOrLinearBNReLUFusion():
#     def __init__(...):
#         ...
#


def _sorted_patterns_dict(
    patterns_dict: dict[Pattern, QuantizeHandler],
) -> dict[Pattern, QuantizeHandler]:
    """
    Return a sorted version of the patterns dictionary such that longer patterns are matched first,
    e.g. match (F.relu, F.linear) before F.relu.
    This works for current use cases, but we may need to have a more clever way to sort
    things to address more complex patterns
    """

    def get_len(pattern):
        """this will calculate the length of the pattern by counting all the entries
        in the pattern.
        this will make sure (nn.ReLU, (nn.BatchNorm, nn.Conv2d)) comes before
        (nn.BatchNorm, nn.Conv2d) so that we can match the former first
        """
        len = 0
        if isinstance(pattern, tuple):
            for item in pattern:
                len += get_len(item)
        else:
            len += 1
        return len

    return OrderedDict(
        sorted(
            patterns_dict.items(),
            key=lambda kv: -get_len(kv[0]) if isinstance(kv[0], tuple) else 1,
        )
    )

```



## High-Level Overview


This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConvOrLinearBNReLUFusion`

**Functions defined**: `_register_fusion_pattern`, `insert`, `get_default_fusion_patterns`, `_register_quant_pattern`, `insert`, `get_default_quant_patterns`, `get_default_output_activation_post_process_map`, `__init__`, `_sorted_patterns_dict`, `get_len`

**Key imports**: copy, OrderedDict, Any, FixedQParamsFakeQuantize, ObserverBase, Pattern


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `collections`: OrderedDict
- `typing`: Any
- `torch.ao.quantization.fake_quantize`: FixedQParamsFakeQuantize
- `torch.ao.quantization.observer`: ObserverBase
- `torch.ao.quantization.utils`: Pattern


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`torch/ao/quantization/fx`):

- [`lstm_utils.py_docs.md`](./lstm_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_lower_to_native_backend.py_docs.md`](./_lower_to_native_backend.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`convert.py_docs.md`](./convert.py_docs.md)
- [`lower_to_fbgemm.py_docs.md`](./lower_to_fbgemm.py_docs.md)
- [`_equalize.py_docs.md`](./_equalize.py_docs.md)
- [`_decomposed.py_docs.md`](./_decomposed.py_docs.md)
- [`graph_module.py_docs.md`](./graph_module.py_docs.md)
- [`fuse.py_docs.md`](./fuse.py_docs.md)


## Cross-References

- **File Documentation**: `pattern_utils.py_docs.md`
- **Keyword Index**: `pattern_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/quantization/fx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/ao/quantization/fx`):

- [`fuse_handler.py_docs.md_docs.md`](./fuse_handler.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`quantize_handler.py_kw.md_docs.md`](./quantize_handler.py_kw.md_docs.md)
- [`lstm_utils.py_kw.md_docs.md`](./lstm_utils.py_kw.md_docs.md)
- [`prepare.py_kw.md_docs.md`](./prepare.py_kw.md_docs.md)
- [`graph_module.py_docs.md_docs.md`](./graph_module.py_docs.md_docs.md)
- [`fuse_handler.py_kw.md_docs.md`](./fuse_handler.py_kw.md_docs.md)
- [`quantize_handler.py_docs.md_docs.md`](./quantize_handler.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`lower_to_qnnpack.py_kw.md_docs.md`](./lower_to_qnnpack.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `pattern_utils.py_docs.md_docs.md`
- **Keyword Index**: `pattern_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
