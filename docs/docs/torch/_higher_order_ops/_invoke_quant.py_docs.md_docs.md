# Documentation: `docs/torch/_higher_order_ops/_invoke_quant.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_higher_order_ops/_invoke_quant.py_docs.md`
- **Size**: 4,991 bytes (4.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_higher_order_ops/_invoke_quant.py`

## File Metadata

- **Path**: `torch/_higher_order_ops/_invoke_quant.py`
- **Size**: 1,897 bytes (1.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# need to fix prim_hop_base type annotations first

import dataclasses
from typing import Optional

import torch
from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars


class InvokeQuantTracer(BaseHOP):
    def __init__(self) -> None:
        super().__init__("invoke_quant_packed")

    def __call__(self, subgraph, *operands, scheme=None, quant_options=None):
        subgraph = FunctionWithNoFreeVars(subgraph)
        return super().__call__(
            subgraph, *operands, scheme=scheme, quant_options=quant_options
        )


invoke_quant_packed = InvokeQuantTracer()


class InvokeQuantUnpacked(BaseHOP):
    def __init__(self) -> None:
        super().__init__("invoke_quant")

    def __call__(self, subgraph, *operands, scheme=None):
        return super().__call__(subgraph, *operands, scheme=scheme)


invoke_quant = InvokeQuantUnpacked()


@dataclasses.dataclass(frozen=True, repr=True)
class InvokeQuant:
    """
    Invoke a quantization function that will be preserved as a single operator. Preservation
    as a single operator aids in pattern matching and custom lowerings.

    The operation appears as:
        torch.ops.higher_order.invoke_quant(subgraph, *args, scheme=scheme)

    Args:
        codegen_low_precision: Use observed subgraph dtypes for codegen instead of
            upcasting to fp32. Can improve performance for prologue fusion but
            requires careful testing of numerics.
    """

    codegen_low_precision: bool = True

    def __call__(
        self,
        *args,
        scheme: Optional[str] = None,
        **kwargs,
    ):
        if not torch.compiler.is_compiling():
            return args[0](*args[1:], **kwargs)

        if scheme is not None:
            kwargs["scheme"] = scheme

        return invoke_quant_packed(*args, **kwargs, quant_options=self)  # type: ignore[call-arg]

```



## High-Level Overview

"""    Invoke a quantization function that will be preserved as a single operator. Preservation    as a single operator aids in pattern matching and custom lowerings.    The operation appears as:        torch.ops.higher_order.invoke_quant(subgraph, *args, scheme=scheme)    Args:        codegen_low_precision: Use observed subgraph dtypes for codegen instead of            upcasting to fp32. Can improve performance for prologue fusion but            requires careful testing of numerics.

This Python file contains 3 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `InvokeQuantTracer`, `InvokeQuantUnpacked`, `InvokeQuant`

**Functions defined**: `__init__`, `__call__`, `__init__`, `__call__`, `__call__`

**Key imports**: dataclasses, Optional, torch, BaseHOP, FunctionWithNoFreeVars


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_higher_order_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`
- `typing`: Optional
- `torch`
- `torch._higher_order_ops.base_hop`: BaseHOP, FunctionWithNoFreeVars


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

Files in the same folder (`torch/_higher_order_ops`):

- [`associative_scan.py_docs.md`](./associative_scan.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`effects.py_docs.md`](./effects.py_docs.md)
- [`foreach_map.py_docs.md`](./foreach_map.py_docs.md)
- [`strict_mode.py_docs.md`](./strict_mode.py_docs.md)
- [`torchbind.py_docs.md`](./torchbind.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`run_const_graph.py_docs.md`](./run_const_graph.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `_invoke_quant.py_docs.md`
- **Keyword Index**: `_invoke_quant.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_higher_order_ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_higher_order_ops`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_higher_order_ops`):

- [`schema.py_docs.md_docs.md`](./schema.py_docs.md_docs.md)
- [`run_const_graph.py_docs.md_docs.md`](./run_const_graph.py_docs.md_docs.md)
- [`effects.py_kw.md_docs.md`](./effects.py_kw.md_docs.md)
- [`partitioner.py_docs.md_docs.md`](./partitioner.py_docs.md_docs.md)
- [`strict_mode.py_docs.md_docs.md`](./strict_mode.py_docs.md_docs.md)
- [`out_dtype.py_kw.md_docs.md`](./out_dtype.py_kw.md_docs.md)
- [`wrap.py_docs.md_docs.md`](./wrap.py_docs.md_docs.md)
- [`while_loop.py_kw.md_docs.md`](./while_loop.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`invoke_subgraph.py_docs.md_docs.md`](./invoke_subgraph.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_invoke_quant.py_docs.md_docs.md`
- **Keyword Index**: `_invoke_quant.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
