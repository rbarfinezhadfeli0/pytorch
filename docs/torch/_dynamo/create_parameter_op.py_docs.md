# Documentation: `torch/_dynamo/create_parameter_op.py`

## File Metadata

- **Path**: `torch/_dynamo/create_parameter_op.py`
- **Size**: 2,561 bytes (2.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch


# See [Note: Metadata mutation in proxy tracing] for why sacrificial parameter mutates
# metadata during proxy tracing and we should remove the sacrificial parameter logic.
doc = """
This is used when dynamo traces torch.nn.Parameter, which normally would not trace properly
with AOTAutograd.  We instead create a placeholder torch.nn.Parameter before the graph, which
becomes a graph arg and has no storage backing it.  At the point in the graph where the parameter
actually should be created we mutate this sacrificial placeholder into it.  This allows gradients
to flow into the parameter as if it were an input to the graph (which is the only thing we are
allowed to compute gradients on).
""".strip()


class TracableCreateParameter(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx: Any, tensor: Any, placeholder: Any) -> torch.nn.Parameter:
        assert not tensor.requires_grad
        return placeholder.set_(tensor)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, torch.Tensor]:
        grad = grad_outputs[0]
        return None, grad  # grad flows to placeholder


def tracable_create_parameter(
    tensor: torch.Tensor, placeholder: torch.nn.Parameter
) -> torch.nn.Parameter:
    with torch.set_grad_enabled(placeholder.requires_grad):
        out = TracableCreateParameter.apply(tensor, placeholder)
    return out


def new_parameter_placeholder(
    size: tuple[int, ...], dtype: torch.dtype, device: torch.device, requires_grad: bool
) -> torch.nn.Parameter:
    """Create a placeholder to be passed to the above functions"""
    result = torch.nn.Parameter(
        torch.empty(size, dtype=dtype, device=device), requires_grad=requires_grad
    )
    # TODO(jansel): alloc followed by free is inefficient, need a way to allocate an unbacked tensor.
    # Allocating a zero tensor would causes assert failures in autograd.
    result.untyped_storage().resize_(0)
    return result


_TLS = threading.local()


@contextmanager
def do_not_convert_to_tracable_parameter() -> Generator[bool, None, None]:
    old_flag = getattr(_TLS, "convert_tracable_parameter", True)
    _TLS.convert_tracable_parameter = False
    try:
        yield False
    finally:
        _TLS.convert_tracable_parameter = old_flag


def can_convert_to_tracable_parameter() -> bool:
    return getattr(_TLS, "convert_tracable_parameter", True)

```



## High-Level Overview

doc = """This is used when dynamo traces torch.nn.Parameter, which normally would not trace properlywith AOTAutograd.  We instead create a placeholder torch.nn.Parameter before the graph, whichbecomes a graph arg and has no storage backing it.  At the point in the graph where the parameteractually should be created we mutate this sacrificial placeholder into it.  This allows gradientsto flow into the parameter as if it were an input to the graph (which is the only thing we areallowed to compute gradients on).

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TracableCreateParameter`

**Functions defined**: `forward`, `backward`, `tracable_create_parameter`, `new_parameter_placeholder`, `do_not_convert_to_tracable_parameter`, `can_convert_to_tracable_parameter`

**Key imports**: threading, Generator, contextmanager, Any, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `threading`
- `collections.abc`: Generator
- `contextlib`: contextmanager
- `typing`: Any
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `create_parameter_op.py_docs.md`
- **Keyword Index**: `create_parameter_op.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
