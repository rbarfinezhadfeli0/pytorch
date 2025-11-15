# Documentation: `docs/functorch/dim/_order.py_docs.md`

## File Metadata

- **Path**: `docs/functorch/dim/_order.py_docs.md`
- **Size**: 9,898 bytes (9.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `functorch/dim/_order.py`

## File Metadata

- **Path**: `functorch/dim/_order.py`
- **Size**: 7,008 bytes (6.84 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import Any, TYPE_CHECKING, Union


if TYPE_CHECKING:
    from collections.abc import Sequence

import torch  # noqa: TC002

from ._dim_entry import _match_levels, DimEntry, ndim_of_levels


def _wrap_dim(arg: Any, orig_ndim: int, allow_none: bool = True) -> DimEntry:
    """
    Convert various dimension representations to DimEntry.

    Args:
        arg: The argument to convert (Dim, int, or other)
        orig_ndim: Original number of dimensions
        allow_none: Whether to allow None values

    Returns:
        DimEntry representation of the dimension
    """
    from . import Dim

    if arg is None and allow_none:
        return DimEntry()  # None entry
    elif isinstance(arg, Dim):
        return DimEntry(arg)
    elif isinstance(arg, int):
        if arg < 0:
            pos = arg
        else:
            pos = arg - orig_ndim
        return DimEntry(pos)
    else:
        return DimEntry()


def order(
    tensor_or_dim: Union[torch.Tensor, Any], *dims: Union[Any, Sequence[Any]]
) -> torch.Tensor:
    """
    Reorder the dimensions of a tensor or create a tensor from a dimension.

    It allows reordering tensor dimensions using first-class dimensions and
    positional indices.

    Args:
        tensor_or_dim: Input tensor with first-class dimensions, or a Dim object
        *dims: Dimensions or sequences of dimensions specifying the new order

    Returns:
        Tensor with reordered dimensions

    Examples:
        >>> import torch
        >>> from functorch.dim import dims
        >>> batch, channel, height, width = dims(4)
        >>> x = torch.randn(2, 3, 4, 5)[batch, channel, height, width]
        >>> # Reorder to [height, width, batch, channel]
        >>> y = order(x, height, width, batch, channel)
    """
    from . import Dim, DimList, Tensor

    # Handle first argument - tensor or dimension
    if isinstance(tensor_or_dim, Tensor):
        # First-class tensor
        orig_levels = tensor_or_dim._levels[:]
        data = tensor_or_dim._tensor
        has_device = tensor_or_dim._has_device
    elif isinstance(tensor_or_dim, Dim):
        # Single dimension - create range tensor
        orig_levels = [DimEntry(tensor_or_dim)]
        data = tensor_or_dim._get_range()
        has_device = False
    else:
        raise ValueError("First argument must be a Tensor or Dim object")

    flat_positional_dims = []
    to_flatten = []  # List of (start_index, length) pairs for flattening
    levels = orig_levels[:]

    orig_ndim = ndim_of_levels(levels)

    def append_dim(d: DimEntry) -> None:
        """Add a dimension to the reordering, removing it from available levels."""
        try:
            idx = levels.index(d)
        except ValueError:
            idx = None
        if idx is None:
            if d.is_positional():
                raise ValueError(
                    f"tensor has {orig_ndim} positional dimensions, but {d.position() + orig_ndim} specified, "
                    f"or it was specified twice"
                )
            else:
                raise ValueError(
                    f"tensor does not contain dim {d.dim()} or it was specified twice"
                )

        levels[idx] = DimEntry()
        flat_positional_dims.append(d)

    n_new_positional = 0

    # Process each dimension argument
    for arg in dims:
        entry = _wrap_dim(arg, orig_ndim, False)
        if not entry.is_none():
            append_dim(entry)
            n_new_positional += 1
        elif isinstance(arg, DimList):
            # Handle DimList
            for dim in arg._dims:
                append_dim(DimEntry(dim))
                n_new_positional += 1
        else:
            # Handle sequences of dimensions for flattening
            n_new_positional += 1
            if not hasattr(arg, "__iter__"):
                raise ValueError("expected a Dim, List[Dim], or Sequence[Dim]")

            # Convert to list to get length
            seq = list(arg)
            to_flatten.append((len(flat_positional_dims), len(seq)))

            for item in seq:
                entry = _wrap_dim(item, orig_ndim, False)
                if entry.is_none():
                    raise ValueError("expected a Dim or int")
                append_dim(entry)

    # Build new level ordering
    insert_point = -1
    new_levels: list[DimEntry] = []

    # Add remaining (non-reordered) levels, finding insertion point for new dimensions
    for level in levels:
        if level.is_none():
            continue
        if level.is_positional():
            if insert_point == -1:
                insert_point = len(new_levels)
                new_levels.extend(flat_positional_dims)
        new_levels.append(level)

    # If no positional dimensions found, append new dims at the end
    if insert_point == -1:
        insert_point = len(new_levels)
        new_levels.extend(flat_positional_dims)

    # Match tensor to new level structure
    assert data is not None, "Cannot reorder None tensor"
    ndata = _match_levels(data, orig_levels, new_levels)

    # Handle dimension flattening if requested
    if to_flatten:
        # Now build the reshape target
        view_shape = []
        sizes = ndata.size()

        # Add dimensions before the reordered ones
        for i in range(insert_point):
            view_shape.append(sizes[i])

        # Process flattening groups
        i = 0
        for start_idx, length in to_flatten:
            # Add individual dims before this flattening group
            while i < start_idx:
                view_shape.append(sizes[insert_point + i])
                i += 1

            # Flatten the group
            new_size = 1
            for j in range(length):
                new_size *= sizes[insert_point + i + j]
            view_shape.append(new_size)
            i += length

        # Add remaining individual dims
        while i < len(flat_positional_dims):
            view_shape.append(sizes[insert_point + i])
            i += 1

        # Add dimensions after the reordered ones
        for i in range(insert_point + len(flat_positional_dims), len(levels)):
            view_shape.append(sizes[i])

        # Update levels by removing flattened dimensions
        n_to_remove = len(flat_positional_dims) - n_new_positional
        if n_to_remove > 0:
            # Remove flattened levels
            new_levels = (
                new_levels[:insert_point] + new_levels[insert_point + n_to_remove :]
            )

        ndata = ndata.reshape(view_shape)

    # Renumber positional dimensions (negative indexing from the right)
    seen = 0
    for i in range(len(new_levels) - 1, -1, -1):
        if new_levels[i].is_positional() or (
            i >= insert_point and i < insert_point + n_new_positional
        ):
            seen -= 1
            new_levels[i] = DimEntry(seen)

    result = Tensor.from_positional(ndata, new_levels, has_device)
    return result  # type: ignore[return-value]

```



## High-Level Overview

"""    Convert various dimension representations to DimEntry.    Args:        arg: The argument to convert (Dim, int, or other)        orig_ndim: Original number of dimensions        allow_none: Whether to allow None values    Returns:        DimEntry representation of the dimension

This Python file contains 3 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_wrap_dim`, `order`, `append_dim`

**Key imports**: annotations, Any, TYPE_CHECKING, Union, Sequence, torch  , _match_levels, DimEntry, ndim_of_levels, Dim, torch, dims, Dim, DimList, Tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `functorch/dim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: Any, TYPE_CHECKING, Union
- `collections.abc`: Sequence
- `torch  `
- `._dim_entry`: _match_levels, DimEntry, ndim_of_levels
- `.`: Dim
- `torch`
- `functorch.dim`: dims


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`functorch/dim`):

- [`magic_trace.py_docs.md`](./magic_trace.py_docs.md)
- [`_wrap.py_docs.md`](./_wrap.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_dim_entry.py_docs.md`](./_dim_entry.py_docs.md)
- [`_py_inst_decoder.py_docs.md`](./_py_inst_decoder.py_docs.md)
- [`wrap_type.py_docs.md`](./wrap_type.py_docs.md)
- [`_enable_all_layers.py_docs.md`](./_enable_all_layers.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`_tensor_info.py_docs.md`](./_tensor_info.py_docs.md)
- [`op_properties.py_docs.md`](./op_properties.py_docs.md)


## Cross-References

- **File Documentation**: `_order.py_docs.md`
- **Keyword Index**: `_order.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/functorch/dim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/functorch/dim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`docs/functorch/dim`):

- [`_py_inst_decoder.py_kw.md_docs.md`](./_py_inst_decoder.py_kw.md_docs.md)
- [`_py_inst_decoder.py_docs.md_docs.md`](./_py_inst_decoder.py_docs.md_docs.md)
- [`_getsetitem.py_docs.md_docs.md`](./_getsetitem.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`_getsetitem.py_kw.md_docs.md`](./_getsetitem.py_kw.md_docs.md)
- [`_order.py_kw.md_docs.md`](./_order.py_kw.md_docs.md)
- [`wrap_type.py_kw.md_docs.md`](./wrap_type.py_kw.md_docs.md)
- [`_tensor_info.py_kw.md_docs.md`](./_tensor_info.py_kw.md_docs.md)
- [`_dim_entry.py_kw.md_docs.md`](./_dim_entry.py_kw.md_docs.md)
- [`magic_trace.py_docs.md_docs.md`](./magic_trace.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_order.py_docs.md_docs.md`
- **Keyword Index**: `_order.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
