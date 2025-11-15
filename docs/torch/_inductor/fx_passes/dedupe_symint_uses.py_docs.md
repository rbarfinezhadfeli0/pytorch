# Documentation: `torch/_inductor/fx_passes/dedupe_symint_uses.py`

## File Metadata

- **Path**: `torch/_inductor/fx_passes/dedupe_symint_uses.py`
- **Size**: 2,505 bytes (2.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from dataclasses import dataclass
from typing import Any

import torch
from torch import SymBool, SymFloat, SymInt
from torch.types import py_sym_types
from torch.utils._ordered_set import OrderedSet


@dataclass
class _SymExprHash:
    """
    Hash for a py_sym_types that will use the underlying sympy expression
    """

    sym_obj: SymInt | SymFloat | SymBool

    def __hash__(self) -> int:
        return hash((type(self.sym_obj), self.sym_obj.node.expr))

    def __eq__(self, value) -> bool:
        if not isinstance(value, _SymExprHash):
            return False
        return self.sym_obj.node.expr == value.sym_obj.node.expr


class _SymHashingDict:
    """
    Wrapper around a dictionary that will convert sym types to hash with _SymExprHash and reuse
    existing sym proxies.

    SymPy hash is not always reliable so optimistically hash sympy expression, and if those fail,
    fallback to symnodes.
    """

    def __init__(self):
        self.sym_hash_dict = {}

    def __setitem__(self, key, value):
        self.sym_hash_dict.__setitem__(self._wrap_to_sym_expr_hash(key), value)

    def __getitem__(self, key):
        return self.sym_hash_dict[self._wrap_to_sym_expr_hash(key)]

    def __contains__(self, key):
        return self._wrap_to_sym_expr_hash(key) in self.sym_hash_dict

    def get(self, key, default=None):
        return self.sym_hash_dict.get(self._wrap_to_sym_expr_hash(key), default)

    def _wrap_to_sym_expr_hash(self, key):
        return _SymExprHash(key) if isinstance(key, py_sym_types) else key


def dedupe_symints(graph: torch.fx.Graph):
    """
    Dedupes sym ints in the graph to nodes are resolvable to symint graph inputs.

    We only dedupe from graph inputs to avoid adding a potential dependency in the forward
    from the backward.

    """

    sym_dict = _SymHashingDict()
    resolvable_from_input_symints = OrderedSet[Any]()

    for node in graph.nodes:
        val = node.meta.get("val", None)
        if val is None or not isinstance(val, py_sym_types):
            continue

        if node.op == "placeholder":
            resolvable_from_input_symints.add(node)
            sym_dict[val] = node
        elif existing_node := sym_dict.get(val):
            node.replace_all_uses_with(existing_node)
            graph.erase_node(node)
        elif all(n in resolvable_from_input_symints for n in node.all_input_nodes):
            sym_dict[val] = node
            resolvable_from_input_symints.add(node)

```



## High-Level Overview

"""    Hash for a py_sym_types that will use the underlying sympy expression

This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_SymExprHash`, `_SymHashingDict`

**Functions defined**: `__hash__`, `__eq__`, `__init__`, `__setitem__`, `__getitem__`, `__contains__`, `get`, `_wrap_to_sym_expr_hash`, `dedupe_symints`

**Key imports**: dataclass, Any, torch, SymBool, SymFloat, SymInt, py_sym_types, OrderedSet


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`: dataclass
- `typing`: Any
- `torch`
- `torch.types`: py_sym_types
- `torch.utils._ordered_set`: OrderedSet


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/_inductor/fx_passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fuse_attention.py_docs.md`](./fuse_attention.py_docs.md)
- [`efficient_conv_bn_eval.py_docs.md`](./efficient_conv_bn_eval.py_docs.md)
- [`bucketing.py_docs.md`](./bucketing.py_docs.md)
- [`numeric_utils.py_docs.md`](./numeric_utils.py_docs.md)
- [`post_grad.py_docs.md`](./post_grad.py_docs.md)
- [`joint_graph.py_docs.md`](./joint_graph.py_docs.md)
- [`fsdp.py_docs.md`](./fsdp.py_docs.md)


## Cross-References

- **File Documentation**: `dedupe_symint_uses.py_docs.md`
- **Keyword Index**: `dedupe_symint_uses.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
