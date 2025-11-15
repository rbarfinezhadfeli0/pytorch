# Documentation: `torch/nn/attention/_registry.py`

## File Metadata

- **Path**: `torch/nn/attention/_registry.py`
- **Size**: 3,886 bytes (3.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
"""Registry for flash attention implementations.

This module contains the registration system for flash attention implementations.
It has no torch dependencies to avoid circular imports during initialization.
"""

from collections.abc import Callable
from typing import Literal, Protocol


class FlashAttentionHandle(Protocol):
    def remove(self) -> None: ...


_RegisterFn = Callable[..., FlashAttentionHandle | None]
_FlashAttentionImpl = Literal["FA4"]

_FLASH_ATTENTION_IMPLS: dict[str, _RegisterFn] = {}

_FLASH_ATTENTION_ACTIVE: str | None = None
_FLASH_ATTENTION_HANDLES: dict[str, FlashAttentionHandle] = {}


def register_flash_attention_impl(
    impl: str | _FlashAttentionImpl,
    *,
    register_fn: _RegisterFn,
) -> None:
    """
    Register the callable that activates a flash attention impl.

    .. note::
        This function is intended for SDPA backend providers to register their
        implementations. End users should use :func:`activate_flash_attention_impl`
        to activate a registered implementation.

    Args:
        impl: Implementation identifier (e.g., ``"FA4"``).
        register_fn: Callable that performs the actual dispatcher registration.
            This function will be invoked by :func:`activate_flash_attention_impl`
            and should register custom kernels with the PyTorch dispatcher.
            It may optionally return a handle implementing
            :class:`FlashAttentionHandle` to keep any necessary state alive.

    Example:
        >>> def my_impl_register(module_path: str = "my_flash_impl"):
        ...     # Register custom kernels with torch dispatcher
        ...     pass  # doctest: +SKIP
        >>> register_flash_attention_impl(
        ...     "MyImpl", register_fn=my_impl_register
        ... )  # doctest: +SKIP
    """
    _FLASH_ATTENTION_IMPLS[impl] = register_fn


def activate_flash_attention_impl(
    impl: str | _FlashAttentionImpl,
) -> None:
    """
    Activate into the dispatcher a previously registered flash attention impl.

    .. note::
        Backend providers should NOT automatically activate their implementation
        on import. Users should explicitly opt-in by calling this function or via
        environment variables to ensure multiple provider libraries can coexist.

    Args:
        impl: Implementation identifier to activate. See
            :func:`~torch.nn.attention.list_flash_attention_impls` for available
            implementations.
            If the backend's :func:`register_flash_attention_impl` callable
            returns a :class:`FlashAttentionHandle`, the registry keeps that
            handle alive for the lifetime of the process (until explicit
            uninstall support exists).

    Example:
        >>> activate_flash_attention_impl("FA4")  # doctest: +SKIP
    """
    global _FLASH_ATTENTION_ACTIVE
    register_fn = _FLASH_ATTENTION_IMPLS.get(impl)
    if register_fn is None:
        raise ValueError(
            f"Unknown flash attention impl '{impl}'. "
            f"Available implementations: {list_flash_attention_impls()}"
        )
    # TODO: The only way to actually register a new impl is to unregister the current impl
    # reinstall the default impl and then register the new impl
    if _FLASH_ATTENTION_ACTIVE == impl:
        return

    handle = register_fn()
    if handle is not None:
        _FLASH_ATTENTION_HANDLES[impl] = handle
    _FLASH_ATTENTION_ACTIVE = impl


def list_flash_attention_impls() -> list[str]:
    """Return the names of all available flash attention implementations."""
    return sorted(_FLASH_ATTENTION_IMPLS.keys())


def current_flash_attention_impl() -> str | None:
    """
    Return the currently activated flash attention impl name, if any.

    ``None`` indicates that no custom impl has been activated.
    """
    return _FLASH_ATTENTION_ACTIVE

```



## High-Level Overview

"""Registry for flash attention implementations.This module contains the registration system for flash attention implementations.It has no torch dependencies to avoid circular imports during initialization.

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FlashAttentionHandle`

**Functions defined**: `remove`, `register_flash_attention_impl`, `my_impl_register`, `activate_flash_attention_impl`, `list_flash_attention_impls`, `current_flash_attention_impl`

**Key imports**: Callable, Literal, Protocol


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/attention`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: Literal, Protocol


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`torch/nn/attention`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_fa4.py_docs.md`](./_fa4.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`bias.py_docs.md`](./bias.py_docs.md)
- [`flex_attention.py_docs.md`](./flex_attention.py_docs.md)
- [`varlen.py_docs.md`](./varlen.py_docs.md)


## Cross-References

- **File Documentation**: `_registry.py_docs.md`
- **Keyword Index**: `_registry.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
