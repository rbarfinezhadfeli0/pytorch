# Documentation: `torch/_export/passes/replace_view_ops_with_view_copy_ops_pass.py`

## File Metadata

- **Path**: `torch/_export/passes/replace_view_ops_with_view_copy_ops_pass.py`
- **Size**: 2,411 bytes (2.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional

import torch
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._ops import HigherOrderOperator, OpOverload


__all__ = ["ReplaceViewOpsWithViewCopyOpsPass"]


_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS: dict[OpOverload, OpOverload] = {
    torch.ops.aten._unsafe_view.default: torch.ops.aten.view_copy.default,
}


def is_view_op(schema: torch._C.FunctionSchema) -> bool:
    if len(schema.arguments) == 0:
        return False
    alias_info = schema.arguments[0].alias_info
    return (alias_info is not None) and (not alias_info.is_write)


def get_view_copy_of_view_op(schema: torch._C.FunctionSchema) -> Optional[OpOverload]:
    if is_view_op(schema) and schema.name.startswith("aten::"):
        view_op_name = schema.name.split("::")[1]
        view_op_overload = (
            schema.overload_name if schema.overload_name != "" else "default"
        )
        view_copy_op_name = view_op_name + "_copy"
        if not hasattr(torch.ops.aten, view_copy_op_name):
            raise InternalError(f"{schema.name} is missing a view_copy variant")

        view_copy_op_overload_packet = getattr(torch.ops.aten, view_copy_op_name)

        if not hasattr(view_copy_op_overload_packet, view_op_overload):
            raise InternalError(f"{schema.name} is missing a view_copy variant")

        return getattr(view_copy_op_overload_packet, view_op_overload)

    return None


class ReplaceViewOpsWithViewCopyOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    """
    Our backend expects pure functional operators. For efficiency
    purposes, we keep view ops around while functionalizing the exported
    program. This pass replaces view ops with view copy ops for backends that
    need AOT memory planning.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op in _NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS:
            return super().call_operator(
                (_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS[op]), args, kwargs, meta
            )

        if isinstance(op, HigherOrderOperator):
            return super().call_operator(op, args, kwargs, meta)

        if view_copy_op := get_view_copy_of_view_op(op._schema):
            return super().call_operator(view_copy_op, args, kwargs, meta)

        return super().call_operator(op, args, kwargs, meta)

```



## High-Level Overview

"""    Our backend expects pure functional operators. For efficiency    purposes, we keep view ops around while functionalizing the exported    program. This pass replaces view ops with view copy ops for backends that    need AOT memory planning.

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ReplaceViewOpsWithViewCopyOpsPass`

**Functions defined**: `is_view_op`, `get_view_copy_of_view_op`, `call_operator`

**Key imports**: Optional, torch, InternalError, _ExportPassBaseDeprecatedDoNotUse, HigherOrderOperator, OpOverload


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`
- `torch._export.error`: InternalError
- `torch._export.pass_base`: _ExportPassBaseDeprecatedDoNotUse
- `torch._ops`: HigherOrderOperator, OpOverload


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

Files in the same folder (`torch/_export/passes`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_node_metadata_hook.py_docs.md`](./_node_metadata_hook.py_docs.md)
- [`replace_set_grad_with_hop_pass.py_docs.md`](./replace_set_grad_with_hop_pass.py_docs.md)
- [`functionalize_side_effectful_ops_pass.py_docs.md`](./functionalize_side_effectful_ops_pass.py_docs.md)
- [`insert_custom_op_guards.py_docs.md`](./insert_custom_op_guards.py_docs.md)
- [`constant_folding.py_docs.md`](./constant_folding.py_docs.md)
- [`replace_autocast_with_hop_pass.py_docs.md`](./replace_autocast_with_hop_pass.py_docs.md)
- [`add_runtime_assertions_for_constraints_pass.py_docs.md`](./add_runtime_assertions_for_constraints_pass.py_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- [`replace_with_hop_pass_util.py_docs.md`](./replace_with_hop_pass_util.py_docs.md)


## Cross-References

- **File Documentation**: `replace_view_ops_with_view_copy_ops_pass.py_docs.md`
- **Keyword Index**: `replace_view_ops_with_view_copy_ops_pass.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
