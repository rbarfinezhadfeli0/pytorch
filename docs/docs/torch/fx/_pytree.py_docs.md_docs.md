# Documentation: `docs/torch/fx/_pytree.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/_pytree.py_docs.md`
- **Size**: 6,416 bytes (6.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/_pytree.py`

## File Metadata

- **Path**: `torch/fx/_pytree.py`
- **Size**: 3,610 bytes (3.53 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections import namedtuple
from collections.abc import Callable
from typing import Any, Optional, TypeVar
from typing_extensions import NamedTuple

import torch.return_types
from torch.utils._pytree import PyTree, tree_flatten, TreeSpec


FlattenFuncSpec = Callable[[PyTree, TreeSpec], list]
FlattenFuncExactMatchSpec = Callable[[PyTree, TreeSpec], bool]

SUPPORTED_NODES: dict[type[Any], FlattenFuncSpec] = {}
SUPPORTED_NODES_EXACT_MATCH: dict[type[Any], Optional[FlattenFuncExactMatchSpec]] = {}

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")


def register_pytree_flatten_spec(
    cls: type[Any],
    flatten_fn_spec: FlattenFuncSpec,
    flatten_fn_exact_match_spec: Optional[FlattenFuncExactMatchSpec] = None,
) -> None:
    SUPPORTED_NODES[cls] = flatten_fn_spec
    SUPPORTED_NODES_EXACT_MATCH[cls] = flatten_fn_exact_match_spec


def _deregister_pytree_flatten_spec(
    cls: type[Any],
) -> None:
    del SUPPORTED_NODES[cls]
    del SUPPORTED_NODES_EXACT_MATCH[cls]


def tree_flatten_spec(
    pytree: PyTree,
    spec: TreeSpec,
) -> list[Any]:
    if spec.is_leaf():
        return [pytree]
    # I guess these exist for BC, FC reasons.
    # In general, we should be able to directly
    # use pytree tree flattener to flatten them,
    # as export serializes the pytree separately.
    # Will remove it in follow up PR.
    if spec.type in SUPPORTED_NODES:
        flatten_fn_spec = SUPPORTED_NODES[spec.type]
        child_pytrees = flatten_fn_spec(pytree, spec)
        result = []
        for child, child_spec in zip(child_pytrees, spec.children()):
            flat = tree_flatten_spec(child, child_spec)
            result += flat
        return result
    flat_result, real_spec = tree_flatten(pytree)
    if spec != real_spec:
        raise RuntimeError(
            f"Real spec {real_spec} of object {pytree} is different from expected spec {spec}. "
            f"Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml"
        )
    return flat_result


def _dict_flatten_spec(d: dict[_K, _V], spec: TreeSpec) -> list[_V]:
    return [d[k] for k in spec.context]


def _list_flatten_spec(d: list[_T], spec: TreeSpec) -> list[_T]:
    return [d[i] for i in range(spec.num_children)]


def _tuple_flatten_spec(d: tuple[_T, ...], spec: TreeSpec) -> list[_T]:
    return [d[i] for i in range(spec.num_children)]


def _namedtuple_flatten_spec(d: NamedTuple, spec: TreeSpec) -> list[Any]:
    return [d[i] for i in range(spec.num_children)]


def _dict_flatten_spec_exact_match(d: dict[_K, _V], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _list_flatten_spec_exact_match(d: list[_T], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _tuple_flatten_spec_exact_match(d: tuple[_T, ...], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _namedtuple_flatten_spec_exact_match(d: NamedTuple, spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


register_pytree_flatten_spec(dict, _dict_flatten_spec, _dict_flatten_spec_exact_match)
register_pytree_flatten_spec(list, _list_flatten_spec, _list_flatten_spec_exact_match)
register_pytree_flatten_spec(
    tuple,
    _tuple_flatten_spec,
    _tuple_flatten_spec_exact_match,
)
for return_type in torch.return_types.all_return_types:
    register_pytree_flatten_spec(
        return_type,
        _tuple_flatten_spec,
        _tuple_flatten_spec_exact_match,
    )
register_pytree_flatten_spec(
    namedtuple,  # type: ignore[arg-type]
    _namedtuple_flatten_spec,
    _namedtuple_flatten_spec_exact_match,
)

```



## High-Level Overview


This Python file contains 0 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `register_pytree_flatten_spec`, `_deregister_pytree_flatten_spec`, `tree_flatten_spec`, `_dict_flatten_spec`, `_list_flatten_spec`, `_tuple_flatten_spec`, `_namedtuple_flatten_spec`, `_dict_flatten_spec_exact_match`, `_list_flatten_spec_exact_match`, `_tuple_flatten_spec_exact_match`, `_namedtuple_flatten_spec_exact_match`

**Key imports**: namedtuple, Callable, Any, Optional, TypeVar, NamedTuple, torch.return_types, PyTree, tree_flatten, TreeSpec


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`: namedtuple
- `collections.abc`: Callable
- `typing`: Any, Optional, TypeVar
- `typing_extensions`: NamedTuple
- `torch.return_types`
- `torch.utils._pytree`: PyTree, tree_flatten, TreeSpec


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

Files in the same folder (`torch/fx`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`tensor_type.py_docs.md`](./tensor_type.py_docs.md)
- [`traceback.py_docs.md`](./traceback.py_docs.md)
- [`_symbolic_trace.py_docs.md`](./_symbolic_trace.py_docs.md)
- [`graph.py_docs.md`](./graph.py_docs.md)
- [`node.py_docs.md`](./node.py_docs.md)
- [`annotate.py_docs.md`](./annotate.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`subgraph_rewriter.py_docs.md`](./subgraph_rewriter.py_docs.md)


## Cross-References

- **File Documentation**: `_pytree.py_docs.md`
- **Keyword Index**: `_pytree.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/fx`):

- [`annotate.py_kw.md_docs.md`](./annotate.py_kw.md_docs.md)
- [`_compatibility.py_docs.md_docs.md`](./_compatibility.py_docs.md_docs.md)
- [`tensor_type.py_kw.md_docs.md`](./tensor_type.py_kw.md_docs.md)
- [`_graph_pickler.py_kw.md_docs.md`](./_graph_pickler.py_kw.md_docs.md)
- [`_compatibility.py_kw.md_docs.md`](./_compatibility.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`interpreter.py_kw.md_docs.md`](./interpreter.py_kw.md_docs.md)
- [`subgraph_rewriter.py_docs.md_docs.md`](./subgraph_rewriter.py_docs.md_docs.md)
- [`node.py_docs.md_docs.md`](./node.py_docs.md_docs.md)
- [`graph_module.py_docs.md_docs.md`](./graph_module.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_pytree.py_docs.md_docs.md`
- **Keyword Index**: `_pytree.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
