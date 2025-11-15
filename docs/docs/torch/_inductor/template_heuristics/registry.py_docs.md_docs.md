# Documentation: `docs/torch/_inductor/template_heuristics/registry.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/template_heuristics/registry.py_docs.md`
- **Size**: 9,078 bytes (8.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/template_heuristics/registry.py`

## File Metadata

- **Path**: `torch/_inductor/template_heuristics/registry.py`
- **Size**: 6,041 bytes (5.90 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Template heuristic registry system for PyTorch Inductor.

This module provides a centralized registration system for template heuristics,
allowing automatic registration based on device type and conditional registration
for CUDA vs ROCm based on torch.version.hip.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional, TYPE_CHECKING, Union

from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Iterator


# Module-wide registry for template heuristics
_TEMPLATE_HEURISTIC_REGISTRY: dict[
    tuple[Union[str, None], ...], type[TemplateConfigHeuristics]
] = {}

# Manual cache for successful lookups only (fallback instances are not cached)
_HEURISTIC_CACHE: dict[tuple[str, str, str], TemplateConfigHeuristics] = {}

log = logging.getLogger(__name__)


def register_template_heuristic(
    template_name: str,
    device_type: Union[str, None],
    register: bool = True,
    op_name: Optional[str] = None,
) -> Any:
    """
    Decorator to register template heuristic classes.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu")
            Set this to None to indicate that the heuristic is applicable to all device types.
        register: Whether to register this heuristic. Caller should pass the condition directly.
        op_name: Name of the operator (e.g., "mm", "bmm", "scaled_mm"). This is optional
            and is only used when a template uses different heuristics for different ops

    Returns:
        Decorator function that registers the class if conditions are met.

    Example:
        @register_template_heuristic("mm", "cuda", register=torch.version.hip is None)
        class CUDAMMTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
            pass
    """

    def decorator(
        cls: type[TemplateConfigHeuristics],
    ) -> type[TemplateConfigHeuristics]:
        if register:
            key: tuple[Union[str, None], ...] = (template_name, device_type, op_name)
            _TEMPLATE_HEURISTIC_REGISTRY[key] = cls
            log.info(
                f"Registered template heuristic: {cls.__name__} for '{template_name=}', '{device_type=}', '{op_name=}'"  # noqa: G004
            )
        return cls

    return decorator


def get_template_heuristic(
    template_name: str, device_type: str, op_name: str
) -> TemplateConfigHeuristics:
    """
    Retrieve a template heuristic instance for the given template and device type.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu")
        op_name: Name of the operator (e.g., "mm", "bmm", "scaled_mm")

    Returns:
        Template heuristic instance. If no specific heuristic is found,
        returns a fallback TemplateConfigHeuristics() instance (uncached).
    """
    # Check cache first
    cache_key = (template_name, device_type, op_name)
    if cache_key in _HEURISTIC_CACHE:
        return _HEURISTIC_CACHE[cache_key]

    keys = [
        # everything is specified
        (template_name, device_type, op_name),
        # heuristic is valid across all devices
        (template_name, None, op_name),
        # heuristic is valid across all ops for that device
        (template_name, device_type, None),
        # heuristic is always valid for that template
        (template_name, None, None),
    ]

    # Look up in registry
    heuristic_class = None
    for key in keys:
        if key in _TEMPLATE_HEURISTIC_REGISTRY:
            heuristic_class = _TEMPLATE_HEURISTIC_REGISTRY[key]
            break

    if heuristic_class is None:
        # Log error and return fallback instance (uncached)
        log.error(
            "No template heuristic found - template_name=%s, device_type=%s, op_name=%s. "
            "Available combinations: %s. Using fallback TemplateConfigHeuristics instance.",
            template_name,
            device_type,
            op_name,
            list(_TEMPLATE_HEURISTIC_REGISTRY.keys()),
        )
        return TemplateConfigHeuristics()

    # Cache successful lookup and return
    instance = heuristic_class()
    _HEURISTIC_CACHE[cache_key] = instance
    return instance


def clear_registry() -> None:
    """
    Clear all registered template heuristics.

    This is primarily useful for testing purposes to ensure a clean state.
    """
    _TEMPLATE_HEURISTIC_REGISTRY.clear()
    _HEURISTIC_CACHE.clear()


@contextlib.contextmanager
def override_template_heuristics(
    device_type: str,
    template_op_pairs: list[tuple[str, str]],
) -> Iterator[None]:
    """
    Context manager to temporarily override template heuristics with an empty heuristic.

    This is useful for testing purposes, where we want to ensure a specific template/op pair
    is not used

    Args:
        device_type: Device type ("cuda", "cpu", "xpu")
        template_op_pairs: List of (template_name, op_name) pairs to override.
    """
    # Save original entries to restore later
    original_entries = {}
    new_keys = []
    _HEURISTIC_CACHE.clear()
    try:
        for template_name, op_name in template_op_pairs:
            assert op_name is not None
            key = (device_type, template_name, op_name)
            if key in _TEMPLATE_HEURISTIC_REGISTRY:
                original_entries[key] = _TEMPLATE_HEURISTIC_REGISTRY[key]
                # TemplateConfigHeuristics base class returns no entries
                # so we use it for overriding
            _TEMPLATE_HEURISTIC_REGISTRY[key] = TemplateConfigHeuristics
            new_keys.append(key)
        yield
    finally:
        # Restore original entries or remove if they didn't exist before
        for key in new_keys:
            _TEMPLATE_HEURISTIC_REGISTRY.pop(key, None)
            if key in original_entries:
                _TEMPLATE_HEURISTIC_REGISTRY[key] = original_entries[key]
        _HEURISTIC_CACHE.clear()

```



## High-Level Overview

"""Template heuristic registry system for PyTorch Inductor.This module provides a centralized registration system for template heuristics,allowing automatic registration based on device type and conditional registrationfor CUDA vs ROCm based on torch.version.hip.

This Python file contains 4 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CUDAMMTemplateConfigHeuristic`

**Functions defined**: `register_template_heuristic`, `decorator`, `get_template_heuristic`, `clear_registry`, `override_template_heuristics`

**Key imports**: annotations, contextlib, logging, Any, Optional, TYPE_CHECKING, Union, TemplateConfigHeuristics, Iterator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/template_heuristics`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `contextlib`
- `logging`
- `typing`: Any, Optional, TYPE_CHECKING, Union
- `.base`: TemplateConfigHeuristics
- `collections.abc`: Iterator


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/_inductor/template_heuristics`):

- [`aten.py_docs.md`](./aten.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`params.py_docs.md`](./params.py_docs.md)
- [`cutedsl.py_docs.md`](./cutedsl.py_docs.md)
- [`decompose_k.py_docs.md`](./decompose_k.py_docs.md)
- [`base.py_docs.md`](./base.py_docs.md)
- [`contiguous_mm.py_docs.md`](./contiguous_mm.py_docs.md)
- [`triton.py_docs.md`](./triton.py_docs.md)
- [`triton_addmm.py_docs.md`](./triton_addmm.py_docs.md)


## Cross-References

- **File Documentation**: `registry.py_docs.md`
- **Keyword Index**: `registry.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/template_heuristics`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/template_heuristics`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/_inductor/template_heuristics`):

- [`decompose_k.py_docs.md_docs.md`](./decompose_k.py_docs.md_docs.md)
- [`registry.py_kw.md_docs.md`](./registry.py_kw.md_docs.md)
- [`params.py_docs.md_docs.md`](./params.py_docs.md_docs.md)
- [`aten.py_kw.md_docs.md`](./aten.py_kw.md_docs.md)
- [`decompose_k.py_kw.md_docs.md`](./decompose_k.py_kw.md_docs.md)
- [`base.py_kw.md_docs.md`](./base.py_kw.md_docs.md)
- [`triton.py_kw.md_docs.md`](./triton.py_kw.md_docs.md)
- [`cutedsl.py_docs.md_docs.md`](./cutedsl.py_docs.md_docs.md)
- [`gemm.py_kw.md_docs.md`](./gemm.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `registry.py_docs.md_docs.md`
- **Keyword Index**: `registry.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
