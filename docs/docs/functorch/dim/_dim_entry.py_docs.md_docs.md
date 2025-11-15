# Documentation: `docs/functorch/dim/_dim_entry.py_docs.md`

## File Metadata

- **Path**: `docs/functorch/dim/_dim_entry.py_docs.md`
- **Size**: 6,201 bytes (6.06 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `functorch/dim/_dim_entry.py`

## File Metadata

- **Path**: `functorch/dim/_dim_entry.py`
- **Size**: 3,626 bytes (3.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from collections.abc import Sequence

    from . import Dim

import torch  # noqa: TC002


# NB: The old code represented dimension was from as negative number, so we
# follow this convention even though it shouldn't be necessary now
class DimEntry:
    # The dimension this is from the rhs, or a FCD
    data: Union[Dim, int]

    def __init__(self, data: Union[Dim, int, None] = None) -> None:
        from . import Dim

        if type(data) is int:
            assert data < 0
        elif data is None:
            data = 0
        else:
            assert isinstance(data, Dim)
        self.data = data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DimEntry):
            return False
        # Use 'is' for Dim objects to avoid triggering __torch_function__
        # Use '==' only for positional (int) comparisons
        if self.is_positional() and other.is_positional():
            # Both are positional (ints)
            return self.data == other.data
        elif not self.is_positional() and not other.is_positional():
            # Both are Dim objects - use 'is' to avoid __eq__
            return self.data is other.data
        else:
            # One is positional, one is Dim - they can't be equal
            return False

    def is_positional(self) -> bool:
        return type(self.data) is int and self.data < 0

    def is_none(self) -> bool:
        # Use isinstance to check for Dim objects, avoid triggering __torch_function__
        from . import Dim

        if isinstance(self.data, Dim):
            # This is a Dim object, it can't be "none" (which is represented by 0)
            return False
        else:
            # This is an int or other type
            return self.data == 0

    def position(self) -> int:
        assert isinstance(self.data, int)
        return self.data

    def dim(self) -> Dim:
        assert not isinstance(self.data, int)
        return self.data

    def __repr__(self) -> str:
        return repr(self.data)


def ndim_of_levels(levels: Sequence[DimEntry]) -> int:
    r = 0
    for l in levels:
        if l.is_positional():
            r += 1
    return r


def _match_levels(
    tensor: torch.Tensor,
    from_levels: list[DimEntry],
    to_levels: list[DimEntry],
    drop_levels: bool = False,
) -> torch.Tensor:
    """
    Reshape a tensor to match target levels using as_strided.

    Args:
        tensor: Input tensor to reshape
        from_levels: Current levels of the tensor
        to_levels: Target levels to match
        drop_levels: If True, missing dimensions are assumed to have stride 0

    Returns:
        Reshaped tensor
    """
    if from_levels == to_levels:
        return tensor

    sizes = tensor.size()
    strides = tensor.stride()

    if not drop_levels:
        assert len(from_levels) <= len(to_levels), (
            "Cannot expand dimensions without drop_levels"
        )

    new_sizes = []
    new_strides = []

    for level in to_levels:
        # Find index of this level in from_levels
        try:
            idx = from_levels.index(level)
        except ValueError:
            # Level not found in from_levels
            if level.is_positional():
                new_sizes.append(1)
            else:
                new_sizes.append(level.dim().size)
            new_strides.append(0)
        else:
            new_sizes.append(sizes[idx])
            new_strides.append(strides[idx])

    return tensor.as_strided(new_sizes, new_strides, tensor.storage_offset())

```



## High-Level Overview


This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DimEntry`

**Functions defined**: `__init__`, `__eq__`, `is_positional`, `is_none`, `position`, `dim`, `__repr__`, `ndim_of_levels`, `_match_levels`

**Key imports**: annotations, TYPE_CHECKING, Union, Sequence, Dim, torch  , Dim, Dim


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `functorch/dim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: TYPE_CHECKING, Union
- `collections.abc`: Sequence
- `.`: Dim
- `torch  `


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
- [`_py_inst_decoder.py_docs.md`](./_py_inst_decoder.py_docs.md)
- [`wrap_type.py_docs.md`](./wrap_type.py_docs.md)
- [`_enable_all_layers.py_docs.md`](./_enable_all_layers.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`_tensor_info.py_docs.md`](./_tensor_info.py_docs.md)
- [`op_properties.py_docs.md`](./op_properties.py_docs.md)


## Cross-References

- **File Documentation**: `_dim_entry.py_docs.md`
- **Keyword Index**: `_dim_entry.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
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

- **File Documentation**: `_dim_entry.py_docs.md_docs.md`
- **Keyword Index**: `_dim_entry.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
