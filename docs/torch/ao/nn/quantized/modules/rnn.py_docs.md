# Documentation: `torch/ao/nn/quantized/modules/rnn.py`

## File Metadata

- **Path**: `torch/ao/nn/quantized/modules/rnn.py`
- **Size**: 1,898 bytes (1.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any

import torch


__all__ = [
    "LSTM",
]


class LSTM(torch.ao.nn.quantizable.LSTM):
    r"""A quantized long short-term memory (LSTM).

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    Attributes:
        layers : instances of the `_LSTMLayer`

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples in :class:`~torch.ao.nn.quantizable.LSTM`

    Examples::
        >>> # xdoctest: +SKIP
        >>> custom_module_config = {
        ...     'float_to_observed_custom_module_class': {
        ...         nn.LSTM: nn.quantizable.LSTM,
        ...     },
        ...     'observed_to_quantized_custom_module_class': {
        ...         nn.quantizable.LSTM: nn.quantized.LSTM,
        ...     }
        ... }
        >>> tq.prepare(model, prepare_custom_module_class=custom_module_config)
        >>> tq.convert(model, convert_custom_module_class=custom_module_config)
    """

    _FLOAT_MODULE = torch.ao.nn.quantizable.LSTM  # type: ignore[assignment]

    def _get_name(self) -> str:
        return "QuantizedLSTM"

    @classmethod
    def from_float(cls, *args: Any, **kwargs: Any) -> None:
        # The whole flow is float -> observed -> quantized
        # This class does observed -> quantized only
        raise NotImplementedError(
            "It looks like you are trying to convert a "
            "non-observed LSTM module. Please, see "
            "the examples on quantizable LSTMs."
        )

    @classmethod
    def from_observed(cls: type["LSTM"], other: torch.ao.nn.quantizable.LSTM) -> "LSTM":
        assert isinstance(other, cls._FLOAT_MODULE)  # type: ignore[has-type]
        converted = torch.ao.quantization.convert(
            other, inplace=False, remove_qconfig=True
        )
        converted.__class__ = cls
        return converted

```



## High-Level Overview

r"""A quantized long short-term memory (LSTM).    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`    Attributes:        layers : instances of the `_LSTMLayer`    .. note::        To access the weights and biases, you need to access them per layer.        See examples in :class:`~torch.ao.nn.quantizable.LSTM`    Examples::        >>> # xdoctest: +SKIP        >>> custom_module_config = {        ...     'float_to_observed_custom_module_class': {        ...         nn.LSTM: nn.quantizable.LSTM,        ...     },        ...     'observed_to_quantized_custom_module_class': {        ...         nn.quantizable.LSTM: nn.quantized.LSTM,        ...     }        ... }        >>> tq.prepare(model, prepare_custom_module_class=custom_module_config)        >>> tq.convert(model, convert_custom_module_class=custom_module_config)

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LSTM`

**Functions defined**: `_get_name`, `from_float`, `from_observed`

**Key imports**: Any, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `torch`


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`torch/ao/nn/quantized/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`embedding_ops.py_docs.md`](./embedding_ops.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`functional_modules.py_docs.md`](./functional_modules.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)
- [`activation.py_docs.md`](./activation.py_docs.md)
- [`dropout.py_docs.md`](./dropout.py_docs.md)


## Cross-References

- **File Documentation**: `rnn.py_docs.md`
- **Keyword Index**: `rnn.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
