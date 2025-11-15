# Documentation: `torch/ao/quantization/pt2e/duplicate_dq_pass.py`

## File Metadata

- **Path**: `torch/ao/quantization/pt2e/duplicate_dq_pass.py`
- **Size**: 3,129 bytes (3.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import logging
import operator

import torch
from torch.ao.quantization.pt2e.utils import (
    _filter_sym_size_users,
    _is_valid_annotation,
)
from torch.fx.node import map_arg
from torch.fx.passes.infra.pass_base import PassBase, PassResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["DuplicateDQPass"]

_QUANTIZE_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]

_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]


def _maybe_duplicate_dq(
    gm: torch.fx.GraphModule, dq_node: torch.fx.Node, user: torch.fx.Node
) -> None:
    annotation = user.meta.get("quantization_annotation", None)
    if not _is_valid_annotation(annotation):  # type: ignore[arg-type]
        return
    with gm.graph.inserting_after(dq_node):
        new_node = gm.graph.node_copy(dq_node)

        def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
            if n == dq_node:
                return new_node
            else:
                return n

        new_args = map_arg(user.args, maybe_replace_node)
        new_kwargs = map_arg(user.kwargs, maybe_replace_node)
        user.args = new_args  # type: ignore[assignment]
        user.kwargs = new_kwargs  # type: ignore[assignment]


class DuplicateDQPass(PassBase):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target in _DEQUANTIZE_OPS:
                dq_users = _filter_sym_size_users(node)
                if len(dq_users) <= 1:
                    continue
                # Do not duplicate dq for dynamic quantization
                # Pattern: choose_qparam - getitem - q - dq
                q_node = node.args[0]
                if q_node.op == "call_function" and q_node.target in _QUANTIZE_OPS:
                    getitem_node = q_node.args[1]
                    if (
                        isinstance(getitem_node, torch.fx.node.Node)
                        and getitem_node.op == "call_function"
                        and getitem_node.target is operator.getitem
                    ):
                        choose_qparam_node = getitem_node.args[0]
                        if (
                            isinstance(choose_qparam_node, torch.fx.node.Node)
                            and choose_qparam_node.op == "call_function"
                            and choose_qparam_node.target
                            == torch.ops.quantized_decomposed.choose_qparams.tensor
                        ):
                            continue
                for user in dq_users:
                    _maybe_duplicate_dq(graph_module, node, user)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DuplicateDQPass`

**Functions defined**: `_maybe_duplicate_dq`, `maybe_replace_node`, `call`

**Key imports**: logging, operator, torch, map_arg, PassBase, PassResult


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/pt2e`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `operator`
- `torch`
- `torch.fx.node`: map_arg
- `torch.fx.passes.infra.pass_base`: PassBase, PassResult


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/ao/quantization/pt2e`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`port_metadata_pass.py_docs.md`](./port_metadata_pass.py_docs.md)
- [`_numeric_debugger.py_docs.md`](./_numeric_debugger.py_docs.md)
- [`lowering.py_docs.md`](./lowering.py_docs.md)
- [`_affine_quantization.py_docs.md`](./_affine_quantization.py_docs.md)
- [`qat_utils.py_docs.md`](./qat_utils.py_docs.md)
- [`prepare.py_docs.md`](./prepare.py_docs.md)
- [`export_utils.py_docs.md`](./export_utils.py_docs.md)


## Cross-References

- **File Documentation**: `duplicate_dq_pass.py_docs.md`
- **Keyword Index**: `duplicate_dq_pass.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
