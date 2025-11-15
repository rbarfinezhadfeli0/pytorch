# Documentation: `docs/torch/jit/quantized.py_docs.md`

## File Metadata

- **Path**: `docs/torch/jit/quantized.py_docs.md`
- **Size**: 5,817 bytes (5.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/jit/quantized.py`

## File Metadata

- **Path**: `torch/jit/quantized.py`
- **Size**: 3,193 bytes (3.12 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch


class QuantizedLinear(torch.jit.ScriptModule):
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedLinear is no longer supported. Please use "
            "torch.ao.nn.quantized.dynamic.Linear instead."
        )


# FP16 weights
class QuantizedLinearFP16(torch.jit.ScriptModule):
    def __init__(self, other):
        super().__init__()
        raise RuntimeError(
            "torch.jit.QuantizedLinearFP16 is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.Linear instead."
        )


# Quantized RNN cell implementations
class QuantizedRNNCellBase(torch.jit.ScriptModule):
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedRNNCellBase is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.RNNCell instead."
        )


class QuantizedRNNCell(QuantizedRNNCellBase):
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedRNNCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.RNNCell instead."
        )


class QuantizedLSTMCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedLSTMCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.LSTMCell instead."
        )


class QuantizedGRUCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedGRUCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.GRUCell instead."
        )


class QuantizedRNNBase(torch.jit.ScriptModule):
    def __init__(self, other, dtype=torch.int8):
        raise RuntimeError(
            "torch.jit.QuantizedRNNBase is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic instead."
        )


class QuantizedLSTM(QuantizedRNNBase):
    def __init__(self, other, dtype):
        raise RuntimeError(
            "torch.jit.QuantizedLSTM is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.LSTM instead."
        )


class QuantizedGRU(QuantizedRNNBase):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "torch.jit.QuantizedGRU is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.GRU instead."
        )


def quantize_rnn_cell_modules(module):
    raise RuntimeError(
        "quantize_rnn_cell_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )


def quantize_linear_modules(module, dtype=torch.int8):
    raise RuntimeError(
        "quantize_linear_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )


def quantize_rnn_modules(module, dtype=torch.int8):
    raise RuntimeError(
        "quantize_rnn_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )

```



## High-Level Overview


This Python file contains 9 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `QuantizedLinear`, `QuantizedLinearFP16`, `QuantizedRNNCellBase`, `QuantizedRNNCell`, `QuantizedLSTMCell`, `QuantizedGRUCell`, `QuantizedRNNBase`, `QuantizedLSTM`, `QuantizedGRU`

**Functions defined**: `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `quantize_rnn_cell_modules`, `quantize_linear_modules`, `quantize_rnn_modules`

**Key imports**: torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`


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

Files in the same folder (`torch/jit`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_decompositions.py_docs.md`](./_decompositions.py_docs.md)
- [`_dataclass_impls.py_docs.md`](./_dataclass_impls.py_docs.md)
- [`frontend.py_docs.md`](./frontend.py_docs.md)
- [`_builtins.py_docs.md`](./_builtins.py_docs.md)
- [`_trace.py_docs.md`](./_trace.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)
- [`_state.py_docs.md`](./_state.py_docs.md)
- [`_await.py_docs.md`](./_await.py_docs.md)


## Cross-References

- **File Documentation**: `quantized.py_docs.md`
- **Keyword Index**: `quantized.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/jit`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/jit`):

- [`_check.py_kw.md_docs.md`](./_check.py_kw.md_docs.md)
- [`_shape_functions.py_docs.md_docs.md`](./_shape_functions.py_docs.md_docs.md)
- [`_trace.py_kw.md_docs.md`](./_trace.py_kw.md_docs.md)
- [`_logging.py_docs.md_docs.md`](./_logging.py_docs.md_docs.md)
- [`_async.py_kw.md_docs.md`](./_async.py_kw.md_docs.md)
- [`_state.py_docs.md_docs.md`](./_state.py_docs.md_docs.md)
- [`_decomposition_utils.py_kw.md_docs.md`](./_decomposition_utils.py_kw.md_docs.md)
- [`frontend.py_docs.md_docs.md`](./frontend.py_docs.md_docs.md)
- [`_check.py_docs.md_docs.md`](./_check.py_docs.md_docs.md)
- [`_script.pyi_docs.md_docs.md`](./_script.pyi_docs.md_docs.md)


## Cross-References

- **File Documentation**: `quantized.py_docs.md_docs.md`
- **Keyword Index**: `quantized.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
