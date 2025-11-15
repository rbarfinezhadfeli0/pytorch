# Documentation: `docs/torch/_library/simple_registry.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_library/simple_registry.py_docs.md`
- **Size**: 6,149 bytes (6.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_library/simple_registry.py`

## File Metadata

- **Path**: `torch/_library/simple_registry.py`
- **Size**: 2,949 bytes (2.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Callable
from typing import Any, Optional

from .effects import EffectHolder
from .fake_impl import FakeImplHolder
from .utils import RegistrationHandle


__all__ = ["SimpleLibraryRegistry", "SimpleOperatorEntry", "singleton"]


class SimpleLibraryRegistry:
    """Registry for the "simple" torch.library APIs

    The "simple" torch.library APIs are a higher-level API on top of the
    raw PyTorch DispatchKey registration APIs that includes:
    - fake impl

    Registrations for these APIs do not go into the PyTorch dispatcher's
    table because they may not directly involve a DispatchKey. For example,
    the fake impl is a Python function that gets invoked by FakeTensor.
    Instead, we manage them here.

    SimpleLibraryRegistry is a mapping from a fully qualified operator name
    (including the overload) to SimpleOperatorEntry.
    """

    def __init__(self) -> None:
        self._data: dict[str, SimpleOperatorEntry] = {}

    def find(self, qualname: str) -> "SimpleOperatorEntry":
        res = self._data.get(qualname, None)
        if res is None:
            self._data[qualname] = res = SimpleOperatorEntry(qualname)
        return res


singleton: SimpleLibraryRegistry = SimpleLibraryRegistry()


class SimpleOperatorEntry:
    """This is 1:1 to an operator overload.

    The fields of SimpleOperatorEntry are Holders where kernels can be
    registered to.
    """

    def __init__(self, qualname: str) -> None:
        self.qualname: str = qualname
        self.fake_impl: FakeImplHolder = FakeImplHolder(qualname)
        self.torch_dispatch_rules: GenericTorchDispatchRuleHolder = (
            GenericTorchDispatchRuleHolder(qualname)
        )

        self.effect: EffectHolder = EffectHolder(qualname)

    # For compatibility reasons. We can delete this soon.
    @property
    def abstract_impl(self) -> FakeImplHolder:
        return self.fake_impl


class GenericTorchDispatchRuleHolder:
    def __init__(self, qualname: str) -> None:
        self._data: dict[type, Callable[..., Any]] = {}
        self.qualname: str = qualname

    def register(
        self, torch_dispatch_class: type, func: Callable[..., Any]
    ) -> RegistrationHandle:
        if self.find(torch_dispatch_class):
            raise RuntimeError(
                f"{torch_dispatch_class} already has a `__torch_dispatch__` rule registered for {self.qualname}"
            )
        self._data[torch_dispatch_class] = func

        def deregister() -> None:
            del self._data[torch_dispatch_class]

        return RegistrationHandle(deregister)

    def find(self, torch_dispatch_class: type) -> Optional[Callable[..., Any]]:
        return self._data.get(torch_dispatch_class, None)


def find_torch_dispatch_rule(
    op: Any, torch_dispatch_class: type
) -> Optional[Callable[..., Any]]:
    return singleton.find(op.__qualname__).torch_dispatch_rules.find(
        torch_dispatch_class
    )

```



## High-Level Overview

"""Registry for the "simple" torch.library APIs    The "simple" torch.library APIs are a higher-level API on top of the    raw PyTorch DispatchKey registration APIs that includes:    - fake impl    Registrations for these APIs do not go into the PyTorch dispatcher's    table because they may not directly involve a DispatchKey. For example,    the fake impl is a Python function that gets invoked by FakeTensor.    Instead, we manage them here.    SimpleLibraryRegistry is a mapping from a fully qualified operator name    (including the overload) to SimpleOperatorEntry.

This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SimpleLibraryRegistry`, `SimpleOperatorEntry`, `GenericTorchDispatchRuleHolder`

**Functions defined**: `__init__`, `find`, `__init__`, `abstract_impl`, `__init__`, `register`, `deregister`, `find`, `find_torch_dispatch_rule`

**Key imports**: Callable, Any, Optional, EffectHolder, FakeImplHolder, RegistrationHandle


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_library`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: Any, Optional
- `.effects`: EffectHolder
- `.fake_impl`: FakeImplHolder
- `.utils`: RegistrationHandle


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

Files in the same folder (`torch/_library`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`effects.py_docs.md`](./effects.py_docs.md)
- [`autograd.py_docs.md`](./autograd.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`fake_impl.py_docs.md`](./fake_impl.py_docs.md)
- [`triton.py_docs.md`](./triton.py_docs.md)
- [`fake_profile.py_docs.md`](./fake_profile.py_docs.md)
- [`opaque_object.py_docs.md`](./opaque_object.py_docs.md)
- [`infer_schema.py_docs.md`](./infer_schema.py_docs.md)


## Cross-References

- **File Documentation**: `simple_registry.py_docs.md`
- **Keyword Index**: `simple_registry.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_library`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_library`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_library`):

- [`fake_impl.py_docs.md_docs.md`](./fake_impl.py_docs.md_docs.md)
- [`effects.py_kw.md_docs.md`](./effects.py_kw.md_docs.md)
- [`opaque_object.py_kw.md_docs.md`](./opaque_object.py_kw.md_docs.md)
- [`infer_schema.py_kw.md_docs.md`](./infer_schema.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`custom_ops.py_docs.md_docs.md`](./custom_ops.py_docs.md_docs.md)
- [`simple_registry.py_kw.md_docs.md`](./simple_registry.py_kw.md_docs.md)
- [`autograd.py_kw.md_docs.md`](./autograd.py_kw.md_docs.md)
- [`triton.py_kw.md_docs.md`](./triton.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `simple_registry.py_docs.md_docs.md`
- **Keyword Index**: `simple_registry.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
