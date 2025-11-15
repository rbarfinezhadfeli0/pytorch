# Documentation: `docs/torch/_export/passes/functionalize_side_effectful_ops_pass.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_export/passes/functionalize_side_effectful_ops_pass.py_docs.md`
- **Size**: 6,827 bytes (6.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_export/passes/functionalize_side_effectful_ops_pass.py`

## File Metadata

- **Path**: `torch/_export/passes/functionalize_side_effectful_ops_pass.py`
- **Size**: 3,279 bytes (3.20 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import copy
from typing import Optional

import torch
from torch._export.pass_base import (
    _ExportPassBaseDeprecatedDoNotUse,
    Argument,
    PassResult,
)
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._ops import OpOverload


aten = torch.ops.aten

_NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS: dict[OpOverload, OpOverload] = {
    aten.sym_constrain_range.default: aten._functional_sym_constrain_range.default,
    aten._assert_async.msg: aten._functional_assert_async.msg,
}


class _FunctionalizeSideEffectfulOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    """
    Functionalize ops with side effect in graph module by replacing the op with
    functional version of it. A new dependency token (`dep_token`) will be
    created and propagated through functional ops to output.
    For example:
    ```
    def f(x):
        sym_constrain_range(x.shape[0], min=1, max=3)
        return x.add(3)
    ```
    Will be transformed to:
    ```
    def f(x):
        dep_token0 = _make_dep_token()
        dep_token1 = _functional_sym_constrain_range(
            x.shape[0], min=1, max=3, dep_token=dep_token0
        )

        return x.add(3), dep_token1
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self._dep_token: Optional[ProxyValue] = None
        self._next_dep_token_index: Optional[int] = None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Early return if no non-functional assertions.
        if not any(
            n.target in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS
            for n in graph_module.graph.nodes
        ):
            return PassResult(graph_module=graph_module, modified=False)

        gm = copy.deepcopy(graph_module)
        self._dep_token = None
        self._next_dep_token_index = None
        return super().call(gm)

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS:
            return super().call_operator(op, args, kwargs, meta)

        if self._dep_token is None:
            self._dep_token = super().call_operator(
                aten._make_dep_token,
                args=(),
                kwargs={},
                meta=self._create_dummy_node_metadata(),
            )
            self._dep_token.node.name = "dep_token0"
            self._next_dep_token_index = 1

        self._dep_token = super().call_operator(
            _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS[op],
            args=args,
            kwargs={**kwargs, "dep_token": self._dep_token},
            meta=meta,
        )
        assert self._next_dep_token_index is not None
        self._dep_token.node.name = f"dep_token{self._next_dep_token_index}"
        self._next_dep_token_index += 1

        return self._dep_token

    def output(self, results: list[Argument], meta: NodeMetadata) -> ProxyValue:
        assert self._dep_token is not None

        return super().output(results=(*results, self._dep_token), meta=meta)  # type: ignore[arg-type]

```



## High-Level Overview

"""    Functionalize ops with side effect in graph module by replacing the op with    functional version of it. A new dependency token (`dep_token`) will be    created and propagated through functional ops to output.    For example:    ```    def f(x):        sym_constrain_range(x.shape[0], min=1, max=3)        return x.add(3)    ```    Will be transformed to:    ```    def f(x):        dep_token0 = _make_dep_token()        dep_token1 = _functional_sym_constrain_range(            x.shape[0], min=1, max=3, dep_token=dep_token0        )        return x.add(3), dep_token1    ```

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_FunctionalizeSideEffectfulOpsPass`

**Functions defined**: `f`, `f`, `__init__`, `call`, `call_operator`, `output`

**Key imports**: copy, Optional, torch, NodeMetadata, ProxyValue, OpOverload


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `typing`: Optional
- `torch`
- `torch._export.pass_infra.node_metadata`: NodeMetadata
- `torch._export.pass_infra.proxy_value`: ProxyValue
- `torch._ops`: OpOverload


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

Files in the same folder (`torch/_export/passes`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_node_metadata_hook.py_docs.md`](./_node_metadata_hook.py_docs.md)
- [`replace_set_grad_with_hop_pass.py_docs.md`](./replace_set_grad_with_hop_pass.py_docs.md)
- [`insert_custom_op_guards.py_docs.md`](./insert_custom_op_guards.py_docs.md)
- [`constant_folding.py_docs.md`](./constant_folding.py_docs.md)
- [`replace_autocast_with_hop_pass.py_docs.md`](./replace_autocast_with_hop_pass.py_docs.md)
- [`add_runtime_assertions_for_constraints_pass.py_docs.md`](./add_runtime_assertions_for_constraints_pass.py_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- [`replace_with_hop_pass_util.py_docs.md`](./replace_with_hop_pass_util.py_docs.md)


## Cross-References

- **File Documentation**: `functionalize_side_effectful_ops_pass.py_docs.md`
- **Keyword Index**: `functionalize_side_effectful_ops_pass.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_export/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/_export/passes`):

- [`replace_set_grad_with_hop_pass.py_docs.md_docs.md`](./replace_set_grad_with_hop_pass.py_docs.md_docs.md)
- [`_node_metadata_hook.py_docs.md_docs.md`](./_node_metadata_hook.py_docs.md_docs.md)
- [`replace_view_ops_with_view_copy_ops_pass.py_kw.md_docs.md`](./replace_view_ops_with_view_copy_ops_pass.py_kw.md_docs.md)
- [`lift_constants_pass.py_kw.md_docs.md`](./lift_constants_pass.py_kw.md_docs.md)
- [`remove_runtime_assertions.py_kw.md_docs.md`](./remove_runtime_assertions.py_kw.md_docs.md)
- [`lift_constants_pass.py_docs.md_docs.md`](./lift_constants_pass.py_docs.md_docs.md)
- [`constant_folding.py_docs.md_docs.md`](./constant_folding.py_docs.md_docs.md)
- [`remove_runtime_assertions.py_docs.md_docs.md`](./remove_runtime_assertions.py_docs.md_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md)
- [`constant_folding.py_kw.md_docs.md`](./constant_folding.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `functionalize_side_effectful_ops_pass.py_docs.md_docs.md`
- **Keyword Index**: `functionalize_side_effectful_ops_pass.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
