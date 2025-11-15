# Documentation: `torch/_higher_order_ops/__init__.py`

## File Metadata

- **Path**: `torch/_higher_order_ops/__init__.py`
- **Size**: 2,519 bytes (2.46 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
from torch._higher_order_ops._invoke_quant import (
    invoke_quant,
    invoke_quant_packed,
    InvokeQuant,
)
from torch._higher_order_ops.aoti_call_delegate import aoti_call_delegate
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)
from torch._higher_order_ops.base_hop import BaseHOP
from torch._higher_order_ops.cond import cond
from torch._higher_order_ops.effects import with_effects
from torch._higher_order_ops.executorch_call_delegate import executorch_call_delegate
from torch._higher_order_ops.flat_apply import flat_apply
from torch._higher_order_ops.flex_attention import (
    flex_attention,
    flex_attention_backward,
)
from torch._higher_order_ops.foreach_map import _foreach_map, foreach_map
from torch._higher_order_ops.hints_wrap import hints_wrapper
from torch._higher_order_ops.invoke_subgraph import invoke_subgraph
from torch._higher_order_ops.local_map import local_map_hop
from torch._higher_order_ops.map import map
from torch._higher_order_ops.out_dtype import out_dtype
from torch._higher_order_ops.print import print
from torch._higher_order_ops.run_const_graph import run_const_graph
from torch._higher_order_ops.scan import scan
from torch._higher_order_ops.strict_mode import strict_mode
from torch._higher_order_ops.torchbind import call_torchbind
from torch._higher_order_ops.while_loop import (
    while_loop,
    while_loop_stack_output_op as while_loop_stack_output,
)
from torch._higher_order_ops.wrap import (
    dynamo_bypassing_wrapper,
    tag_activation_checkpoint,
    wrap_activation_checkpoint,
    wrap_with_autocast,
    wrap_with_set_grad_enabled,
)


__all__ = [
    "cond",
    "while_loop",
    "invoke_subgraph",
    "scan",
    "map",
    "flex_attention",
    "flex_attention_backward",
    "hints_wrapper",
    "BaseHOP",
    "flat_apply",
    "foreach_map",
    "_foreach_map",
    "with_effects",
    "tag_activation_checkpoint",
    "auto_functionalized",
    "auto_functionalized_v2",
    "associative_scan",
    "out_dtype",
    "executorch_call_delegate",
    "call_torchbind",
    "run_const_graph",
    "InvokeQuant",
    "invoke_quant",
    "invoke_quant_packed",
    "wrap_with_set_grad_enabled",
    "wrap_with_autocast",
    "wrap_activation_checkpoint",
    "dynamo_bypassing_wrapper",
    "strict_mode",
    "aoti_call_delegate",
    "map",
    "while_loop_stack_output",
    "local_map_hop",
    "print",
]

```



## High-Level Overview


This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: aoti_call_delegate, associative_scan, BaseHOP, cond, with_effects, executorch_call_delegate, flat_apply, _foreach_map, foreach_map, hints_wrapper, invoke_subgraph


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_higher_order_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch._higher_order_ops.aoti_call_delegate`: aoti_call_delegate
- `torch._higher_order_ops.associative_scan`: associative_scan
- `torch._higher_order_ops.base_hop`: BaseHOP
- `torch._higher_order_ops.cond`: cond
- `torch._higher_order_ops.effects`: with_effects
- `torch._higher_order_ops.executorch_call_delegate`: executorch_call_delegate
- `torch._higher_order_ops.flat_apply`: flat_apply
- `torch._higher_order_ops.foreach_map`: _foreach_map, foreach_map
- `torch._higher_order_ops.hints_wrap`: hints_wrapper
- `torch._higher_order_ops.invoke_subgraph`: invoke_subgraph
- `torch._higher_order_ops.local_map`: local_map_hop
- `torch._higher_order_ops.map`: map
- `torch._higher_order_ops.out_dtype`: out_dtype
- `torch._higher_order_ops.print`: print
- `torch._higher_order_ops.run_const_graph`: run_const_graph
- `torch._higher_order_ops.scan`: scan
- `torch._higher_order_ops.strict_mode`: strict_mode
- `torch._higher_order_ops.torchbind`: call_torchbind


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

Files in the same folder (`torch/_higher_order_ops`):

- [`associative_scan.py_docs.md`](./associative_scan.py_docs.md)
- [`effects.py_docs.md`](./effects.py_docs.md)
- [`foreach_map.py_docs.md`](./foreach_map.py_docs.md)
- [`strict_mode.py_docs.md`](./strict_mode.py_docs.md)
- [`torchbind.py_docs.md`](./torchbind.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`run_const_graph.py_docs.md`](./run_const_graph.py_docs.md)
- [`_invoke_quant.py_docs.md`](./_invoke_quant.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
