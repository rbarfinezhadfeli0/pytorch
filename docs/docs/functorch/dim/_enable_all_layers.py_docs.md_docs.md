# Documentation: `docs/functorch/dim/_enable_all_layers.py_docs.md`

## File Metadata

- **Path**: `docs/functorch/dim/_enable_all_layers.py_docs.md`
- **Size**: 8,358 bytes (8.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `functorch/dim/_enable_all_layers.py`

## File Metadata

- **Path**: `functorch/dim/_enable_all_layers.py`
- **Size**: 4,932 bytes (4.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from ._dim_entry import DimEntry


if TYPE_CHECKING:
    from . import Dim, Tensor


class EnableAllLayers:
    """
    RAII-style context manager for enabling functorch vmap layers.
    It manages the creation and cleanup of functorch dynamic layers.

    This is probably one of the more algorithmically important parts of first
    class dims. Intuitively, FCD can be thought of as another way of using
    vmap, where you don't actually have to vmap at the top level, instead the
    vmaps are implicitly determined by inspecting the bound dimensions on the
    FCD tensors involved in a compute (this is similar to our concept of
    non-lexical modes that we spent a long time talking about years ago). But
    under the hood you still need to actually enable the vmap mode. So once
    FCD has determined all of the dims we are batching over, it needs to
    enable all those layers so functorch can actually apply the batching
    rules. Therefore enable all layers!
    """

    levels_start: int
    levels_to_dim: list[Dim]

    def __init__(self, levels: list[DimEntry]):
        """
        Initialize and push dynamic layers for all first-class dimensions.

        Args:
            levels: List of dimension entries to create layers for
        """

        from . import Dim

        self.levels_start = 0
        self.levels_to_dim = []

        for l in levels:
            if not l.is_positional():
                d = l.dim()
                assert isinstance(d, Dim)
                self.levels_to_dim.append(d)

        # Sort by level for stable ordering
        self.levels_to_dim.sort(key=lambda d: d._level)

    def __enter__(self) -> EnableAllLayers:  # noqa: PYI034
        # Create functorch dynamic layers
        for i, dim in enumerate(self.levels_to_dim):
            batch_size = dim.size
            level = torch._C._functorch._vmap_increment_nesting(batch_size, "different")
            if i == 0:
                self.levels_start = level
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up dynamic layers in reverse order."""
        to_remove = self.levels_start + len(self.levels_to_dim) - 1
        for i in range(len(self.levels_to_dim)):
            popped = torch._C._functorch._vmap_decrement_nesting()
            assert popped == to_remove - i, (
                f"Expected layer {to_remove - i}, got {popped}"
            )

    def from_batched(self, batchedtensor: torch.Tensor, has_device: bool) -> Tensor:
        """
        Create a Tensor from a batched tensor by unwrapping functorch layers.

        Args:
            batchedtensor: Batched tensor from functorch operation
            has_device: Whether tensor has device info

        Returns:
            Tensor with appropriate levels
        """
        # Create positional levels for base dimensions
        levels: list[DimEntry] = []
        for i in range(-batchedtensor.dim(), 0):
            levels.append(DimEntry(i))

        tensor = batchedtensor

        while torch._C._functorch.is_batchedtensor(tensor):
            level = torch._C._functorch.maybe_get_level(tensor)
            assert level is not None
            assert level >= self.levels_start and level < self.levels_start + len(
                self.levels_to_dim
            )
            dim = DimEntry(self.levels_to_dim[level - self.levels_start])
            bdim = torch._C._functorch.maybe_get_bdim(tensor)
            assert bdim is not None
            levels.insert(bdim, dim)
            tensor = torch._C._functorch.get_unwrapped(tensor)

        from . import Tensor

        result = Tensor()
        result._tensor = tensor
        result._batchtensor = batchedtensor
        result._has_device = has_device
        result._levels = levels
        return result

    def inplace_update_layers(
        self, batchtensor: torch.Tensor, levels: list[DimEntry]
    ) -> None:
        """
        Update the levels of a batched tensor in place.

        This requires the _maybe_unsafe_set_level binding that we'll add to functorch.

        Args:
            batchtensor: Batched tensor to update
            levels: New levels to set
        """
        # Check if tensor is batched
        if not torch._C._functorch.is_batchedtensor(batchtensor):
            return

        impl = batchtensor

        for i in reversed(range(len(self.levels_to_dim))):
            if impl is None:
                break

            if any(l == DimEntry(self.levels_to_dim[i]) for l in levels):
                # This is very interesting!  The level on batch tensor is
                # meaningless!  We set it RIGHT before we go into vmap
                torch._C._functorch._maybe_unsafe_set_level(impl, self.levels_start + i)
                impl = torch._C._functorch.get_unwrapped(impl)

```



## High-Level Overview

"""    RAII-style context manager for enabling functorch vmap layers.    It manages the creation and cleanup of functorch dynamic layers.    This is probably one of the more algorithmically important parts of first    class dims. Intuitively, FCD can be thought of as another way of using    vmap, where you don't actually have to vmap at the top level, instead the    vmaps are implicitly determined by inspecting the bound dimensions on the    FCD tensors involved in a compute (this is similar to our concept of    non-lexical modes that we spent a long time talking about years ago). But    under the hood you still need to actually enable the vmap mode. So once    FCD has determined all of the dims we are batching over, it needs to    enable all those layers so functorch can actually apply the batching    rules. Therefore enable all layers!

This Python file contains 3 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `EnableAllLayers`

**Functions defined**: `__init__`, `__enter__`, `__exit__`, `from_batched`, `inplace_update_layers`

**Key imports**: annotations, Any, TYPE_CHECKING, torch, DimEntry, Dim, Tensor, Dim, Tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `functorch/dim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: Any, TYPE_CHECKING
- `torch`
- `._dim_entry`: DimEntry
- `.`: Dim, Tensor


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol


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
- [`README.md_docs.md`](./README.md_docs.md)
- [`_tensor_info.py_docs.md`](./_tensor_info.py_docs.md)
- [`op_properties.py_docs.md`](./op_properties.py_docs.md)


## Cross-References

- **File Documentation**: `_enable_all_layers.py_docs.md`
- **Keyword Index**: `_enable_all_layers.py_kw.md`
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
- **Context Manager**: Implements context manager protocol


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

- **File Documentation**: `_enable_all_layers.py_docs.md_docs.md`
- **Keyword Index**: `_enable_all_layers.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
