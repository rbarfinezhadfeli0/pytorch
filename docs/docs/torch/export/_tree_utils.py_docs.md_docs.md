# Documentation: `docs/torch/export/_tree_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/export/_tree_utils.py_docs.md`
- **Size**: 5,006 bytes (4.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/export/_tree_utils.py`

## File Metadata

- **Path**: `torch/export/_tree_utils.py`
- **Size**: 2,258 bytes (2.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Callable
from typing import Any, Optional

from torch.utils._pytree import Context, TreeSpec


def reorder_kwargs(user_kwargs: dict[str, Any], spec: TreeSpec) -> dict[str, Any]:
    """Reorder user-provided kwargs to match the order in `spec`. `spec` is
    expected to be the in_spec of an exported program, i.e. the spec that
    results from flattening `(args, kwargs)`.

    We need this to provide consistent input ordering, such so that users can
    pass in foo(a=a, b=b) OR foo(b=b, a=a) and receive the same result.
    """
    # Make sure that the spec is actually shaped like (args, kwargs)
    assert spec.type is tuple
    assert spec.num_children == 2
    kwargs_spec = spec.child(1)
    assert kwargs_spec.type is dict

    if set(user_kwargs) != set(kwargs_spec.context):
        raise ValueError(
            f"Ran into a kwarg keyword mismatch: "
            f"Got the following keywords {list(user_kwargs)} but expected {kwargs_spec.context}"
        )

    reordered_kwargs = {}
    for kw in kwargs_spec.context:
        reordered_kwargs[kw] = user_kwargs[kw]

    return reordered_kwargs


def is_equivalent(
    spec1: TreeSpec,
    spec2: TreeSpec,
    equivalence_fn: Callable[[Optional[type], Context, Optional[type], Context], bool],
) -> bool:
    """Customizable equivalence check for two TreeSpecs.

    Arguments:
        spec1: The first TreeSpec to compare
        spec2: The second TreeSpec to compare
        equivalence_fn: A function to determine the equivalence of two
            TreeSpecs by examining their types and contexts. It will be called like:

                equivalence_fn(spec1.type, spec1.context, spec2.type, spec2.context)

            This function will be applied recursively to all children.

    Returns:
        True if the two TreeSpecs are equivalent, False otherwise.
    """
    if not equivalence_fn(spec1.type, spec1.context, spec2.type, spec2.context):
        return False

    # Recurse on children
    if spec1.num_children != spec2.num_children:
        return False

    for child_spec1, child_spec2 in zip(spec1.children(), spec2.children()):
        if not is_equivalent(child_spec1, child_spec2, equivalence_fn):
            return False

    return True

```



## High-Level Overview

"""Reorder user-provided kwargs to match the order in `spec`. `spec` is    expected to be the in_spec of an exported program, i.e. the spec that    results from flattening `(args, kwargs)`.    We need this to provide consistent input ordering, such so that users can    pass in foo(a=a, b=b) OR foo(b=b, a=a) and receive the same result.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `reorder_kwargs`, `is_equivalent`

**Key imports**: Callable, Any, Optional, Context, TreeSpec


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: Any, Optional
- `torch.utils._pytree`: Context, TreeSpec


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

Files in the same folder (`torch/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_remove_auto_functionalized_pass.py_docs.md`](./_remove_auto_functionalized_pass.py_docs.md)
- [`exported_program.py_docs.md`](./exported_program.py_docs.md)
- [`_wrapper_utils.py_docs.md`](./_wrapper_utils.py_docs.md)
- [`_unlift.py_docs.md`](./_unlift.py_docs.md)
- [`_trace.py_docs.md`](./_trace.py_docs.md)
- [`_swap.py_docs.md`](./_swap.py_docs.md)
- [`_safeguard.py_docs.md`](./_safeguard.py_docs.md)
- [`_remove_effect_tokens_pass.py_docs.md`](./_remove_effect_tokens_pass.py_docs.md)


## Cross-References

- **File Documentation**: `_tree_utils.py_docs.md`
- **Keyword Index**: `_tree_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/export`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/export`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/export`):

- [`custom_obj.py_kw.md_docs.md`](./custom_obj.py_kw.md_docs.md)
- [`_unlift.py_docs.md_docs.md`](./_unlift.py_docs.md_docs.md)
- [`_trace.py_kw.md_docs.md`](./_trace.py_kw.md_docs.md)
- [`_leakage_detection_utils.py_docs.md_docs.md`](./_leakage_detection_utils.py_docs.md_docs.md)
- [`_unlift.py_kw.md_docs.md`](./_unlift.py_kw.md_docs.md)
- [`_trace.py_docs.md_docs.md`](./_trace.py_docs.md_docs.md)
- [`_safeguard.py_kw.md_docs.md`](./_safeguard.py_kw.md_docs.md)
- [`custom_ops.py_docs.md_docs.md`](./custom_ops.py_docs.md_docs.md)
- [`graph_signature.py_kw.md_docs.md`](./graph_signature.py_kw.md_docs.md)
- [`_swap.py_docs.md_docs.md`](./_swap.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_tree_utils.py_docs.md_docs.md`
- **Keyword Index**: `_tree_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
