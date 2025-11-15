# Documentation: `docs/torch/onnx/_internal/torchscript_exporter/registration.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/torchscript_exporter/registration.py_docs.md`
- **Size**: 14,656 bytes (14.31 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/torchscript_exporter/registration.py`

## File Metadata

- **Path**: `torch/onnx/_internal/torchscript_exporter/registration.py`
- **Size**: 11,182 bytes (10.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# mypy: allow-untyped-defs
"""Module for handling symbolic function registration."""

import warnings
from collections.abc import Callable, Collection, Sequence
from typing import Generic, Optional, TypeVar, Union
from typing_extensions import ParamSpec

from torch.onnx import _constants, errors


OpsetVersion = int


def _dispatch_opset_version(
    target: OpsetVersion, registered_opsets: Collection[OpsetVersion]
) -> Optional[OpsetVersion]:
    """Finds the registered opset given a target opset version and the available opsets.

    Args:
        target: The target opset version.
        registered_opsets: The available opsets.

    Returns:
        The registered opset version.
    """
    if not registered_opsets:
        return None

    descending_registered_versions = sorted(registered_opsets, reverse=True)
    # Linear search for the opset version, which is fine since the number of opset
    # versions is small.

    if target >= _constants.ONNX_BASE_OPSET:
        # Always look down toward opset 1 when the target is >= ONNX_BASE_OPSET (opset 9).
        # When a custom op is register at opset 1, we want to be able to discover it as a
        # fallback for all opsets >= ONNX_BASE_OPSET.
        for version in descending_registered_versions:
            if version <= target:
                return version
        return None

    # target < opset 9. This is the legacy behavior to support opset 7 and opset 8.
    # for caffe2 support. We search up toward opset 9.
    for version in reversed(descending_registered_versions):
        # Count back up until _constants.ONNX_BASE_OPSET
        if target <= version <= _constants.ONNX_BASE_OPSET:
            return version

    return None


_K = TypeVar("_K")
_V = TypeVar("_V")
_R = TypeVar("_R")
_P = ParamSpec("_P")


class OverrideDict(Collection[_K], Generic[_K, _V]):
    """A dictionary that merges built-in and custom symbolic functions.

    It supports overriding and un-overriding built-in symbolic functions with custom
    ones.
    """

    def __init__(self) -> None:
        self._base: dict[_K, _V] = {}
        self._overrides: dict[_K, _V] = {}
        self._merged: dict[_K, _V] = {}

    def set_base(self, key: _K, value: _V) -> None:
        self._base[key] = value
        if key not in self._overrides:
            self._merged[key] = value

    def in_base(self, key: _K) -> bool:
        """Checks if a key is in the base dictionary."""
        return key in self._base

    def override(self, key: _K, value: _V) -> None:
        """Overrides a base key-value with a new pair."""
        self._overrides[key] = value
        self._merged[key] = value

    def remove_override(self, key: _K) -> None:
        """Un-overrides a key-value pair."""
        self._overrides.pop(key, None)  # type: ignore[arg-type]
        self._merged.pop(key, None)  # type: ignore[arg-type]
        if key in self._base:
            self._merged[key] = self._base[key]

    def overridden(self, key: _K) -> bool:
        """Checks if a key-value pair is overridden."""
        return key in self._overrides

    def __getitem__(self, key: _K) -> _V:
        return self._merged[key]

    def get(self, key: _K, default: Optional[_V] = None):
        return self._merged.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._merged

    def __iter__(self):
        return iter(self._merged)

    def __len__(self) -> int:
        return len(self._merged)

    def __repr__(self) -> str:
        return f"OverrideDict(base={self._base}, overrides={self._overrides})"

    def __bool__(self) -> bool:
        return bool(self._merged)


class _SymbolicFunctionGroup:
    """Different versions of symbolic functions registered to the same name.

    O(number of registered versions of an op) search is performed to find the most
    recent version of the op.

    The registration is delayed until op is used to improve startup time.

    Function overloads with different arguments are not allowed.
    Custom op overrides are supported.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        # A dictionary of functions, keyed by the opset version.
        self._functions: OverrideDict[OpsetVersion, Callable] = OverrideDict()

    def __repr__(self) -> str:
        return f"_SymbolicFunctionGroup({self._name}, registered={self._functions})"

    def __getitem__(self, key: OpsetVersion) -> Callable:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    # TODO(justinchuby): Add @functools.lru_cache(maxsize=None) if lookup time becomes
    # a problem.
    def get(self, opset: OpsetVersion) -> Optional[Callable]:
        """Find the most recent version of the function."""
        version = _dispatch_opset_version(opset, self._functions)
        if version is None:
            return None

        return self._functions[version]

    def add(self, func: Callable, opset: OpsetVersion) -> None:
        """Adds a symbolic function.

        Args:
            func: The function to add.
            opset: The opset version of the function to add.
        """
        if self._functions.in_base(opset):
            warnings.warn(
                f"Symbolic function '{self._name}' already registered for opset {opset}. "
                f"Replacing the existing function with new function. This is unexpected. "
                f"Please report it on {_constants.PYTORCH_GITHUB_ISSUES_URL}.",
                errors.OnnxExporterWarning,
                stacklevel=2,
            )
        self._functions.set_base(opset, func)

    def add_custom(self, func: Callable, opset: OpsetVersion) -> None:
        """Adds a custom symbolic function.

        Args:
            func: The symbolic function to register.
            opset: The corresponding opset version.
        """
        self._functions.override(opset, func)

    def remove_custom(self, opset: OpsetVersion) -> None:
        """Removes a custom symbolic function.

        Args:
            opset: The opset version of the custom function to remove.
        """
        if not self._functions.overridden(opset):
            warnings.warn(
                f"No custom function registered for '{self._name}' opset {opset}",
                stacklevel=2,
            )
            return
        self._functions.remove_override(opset)

    def get_min_supported(self) -> OpsetVersion:
        """Returns the lowest built-in opset version supported by the function."""
        return min(self._functions)


class SymbolicRegistry:
    """Registry for symbolic functions.

    The registry maintains a mapping from qualified names to symbolic functions.
    It is used to register new symbolic functions and to dispatch calls to
    the appropriate function.
    """

    def __init__(self) -> None:
        self._registry: dict[str, _SymbolicFunctionGroup] = {}

    def register(
        self, name: str, opset: OpsetVersion, func: Callable, custom: bool = False
    ) -> None:
        """Registers a symbolic function.

        Args:
            name: The qualified name of the function to register. In the form of 'domain::op'.
                E.g. 'aten::add'.
            opset: The opset version of the function to register.
            func: The symbolic function to register.
            custom: Whether the function is a custom function that overrides existing ones.

        Raises:
            ValueError: If the separator '::' is not in the name.
        """
        if "::" not in name:
            raise ValueError(
                f"The name must be in the form of 'domain::op', not '{name}'"
            )
        symbolic_functions = self._registry.setdefault(
            name, _SymbolicFunctionGroup(name)
        )
        if custom:
            symbolic_functions.add_custom(func, opset)
        else:
            symbolic_functions.add(func, opset)

    def unregister(self, name: str, opset: OpsetVersion) -> None:
        """Unregisters a symbolic function.

        Args:
            name: The qualified name of the function to unregister.
            opset: The opset version of the function to unregister.
        """
        if name not in self._registry:
            return
        self._registry[name].remove_custom(opset)

    def get_function_group(self, name: str) -> Optional[_SymbolicFunctionGroup]:
        """Returns the function group for the given name."""
        return self._registry.get(name)

    def is_registered_op(self, name: str, version: int) -> bool:
        """Returns whether the given op is registered for the given opset version."""
        functions = self.get_function_group(name)
        if functions is None:
            return False
        return functions.get(version) is not None

    def all_functions(self) -> set[str]:
        """Returns the set of all registered function names."""
        return set(self._registry)


def onnx_symbolic(
    name: str,
    opset: Union[OpsetVersion, Sequence[OpsetVersion]],
    decorate: Optional[Sequence[Callable]] = None,
    custom: bool = False,
) -> Callable:
    """Registers a symbolic function.

    Usage::

    ```
    @onnx_symbolic(
        "aten::symbolic_b",
        opset=10,
        decorate=[quantized_aten_handler(scale=1 / 128, zero_point=0)],
    )
    @symbolic_helper.parse_args("v", "v", "b")
    def symbolic_b(g: _C.Graph, x: _C.Value, y: _C.Value, arg1: bool) -> _C.Value: ...
    ```

    Args:
        name: The qualified name of the function in the form of 'domain::op'.
            E.g. 'aten::add'.
        opset: The opset versions of the function to register at.
        decorate: A sequence of decorators to apply to the function.
        custom: Whether the function is a custom symbolic function.

    Raises:
        ValueError: If the separator '::' is not in the name.
    """

    def wrapper(func: Callable[_P, _R]) -> Callable[_P, _R]:
        decorated = func
        if decorate is not None:
            for decorate_func in decorate:
                decorated = decorate_func(decorated)

        global registry
        nonlocal opset
        if isinstance(opset, OpsetVersion):
            opset = (opset,)
        for opset_version in opset:
            registry.register(name, opset_version, decorated, custom=custom)

        # Return the original function because the decorators in "decorate" are only
        # specific to the instance being registered.
        return func

    return wrapper


def custom_onnx_symbolic(
    name: str,
    opset: Union[OpsetVersion, Sequence[OpsetVersion]],
    decorate: Optional[Sequence[Callable]] = None,
) -> Callable:
    """Registers a custom symbolic function.

    Args:
        name: the qualified name of the function.
        opset: the opset version of the function.
        decorate: a sequence of decorators to apply to the function.

    Returns:
        The decorator.

    Raises:
        ValueError: If the separator '::' is not in the name.
    """
    return onnx_symbolic(name, opset, decorate, custom=True)


# The registry for all symbolic functions.
registry = SymbolicRegistry()

```



## High-Level Overview

"""Module for handling symbolic function registration."""import warningsfrom collections.abc import Callable, Collection, Sequencefrom typing import Generic, Optional, TypeVar, Unionfrom typing_extensions import ParamSpecfrom torch.onnx import _constants, errorsOpsetVersion = intdef _dispatch_opset_version(    target: OpsetVersion, registered_opsets: Collection[OpsetVersion]) -> Optional[OpsetVersion]:

This Python file contains 3 class(es) and 32 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OverrideDict`, `_SymbolicFunctionGroup`, `SymbolicRegistry`

**Functions defined**: `_dispatch_opset_version`, `__init__`, `set_base`, `in_base`, `override`, `remove_override`, `overridden`, `__getitem__`, `get`, `__contains__`, `__iter__`, `__len__`, `__repr__`, `__bool__`, `__init__`, `__repr__`, `__getitem__`, `get`, `add`, `add_custom`

**Key imports**: warnings, Callable, Collection, Sequence, Generic, Optional, TypeVar, Union, ParamSpec, _constants, errors


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `warnings`
- `collections.abc`: Callable, Collection, Sequence
- `typing`: Generic, Optional, TypeVar, Union
- `typing_extensions`: ParamSpec
- `torch.onnx`: _constants, errors


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/onnx/_internal/torchscript_exporter`):

- [`symbolic_opset7.py_docs.md`](./symbolic_opset7.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`symbolic_opset14.py_docs.md`](./symbolic_opset14.py_docs.md)
- [`symbolic_opset11.py_docs.md`](./symbolic_opset11.py_docs.md)
- [`verification.py_docs.md`](./verification.py_docs.md)
- [`symbolic_opset12.py_docs.md`](./symbolic_opset12.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`_experimental.py_docs.md`](./_experimental.py_docs.md)
- [`symbolic_opset20.py_docs.md`](./symbolic_opset20.py_docs.md)
- [`symbolic_opset9.py_docs.md`](./symbolic_opset9.py_docs.md)


## Cross-References

- **File Documentation**: `registration.py_docs.md`
- **Keyword Index**: `registration.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/onnx/_internal/torchscript_exporter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/onnx/_internal/torchscript_exporter`):

- [`symbolic_opset14.py_docs.md_docs.md`](./symbolic_opset14.py_docs.md_docs.md)
- [`symbolic_opset18.py_kw.md_docs.md`](./symbolic_opset18.py_kw.md_docs.md)
- [`_experimental.py_kw.md_docs.md`](./_experimental.py_kw.md_docs.md)
- [`onnx_proto_utils.py_docs.md_docs.md`](./onnx_proto_utils.py_docs.md_docs.md)
- [`symbolic_opset13.py_kw.md_docs.md`](./symbolic_opset13.py_kw.md_docs.md)
- [`symbolic_opset12.py_docs.md_docs.md`](./symbolic_opset12.py_docs.md_docs.md)
- [`symbolic_opset16.py_docs.md_docs.md`](./symbolic_opset16.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`symbolic_helper.py_kw.md_docs.md`](./symbolic_helper.py_kw.md_docs.md)
- [`symbolic_opset8.py_docs.md_docs.md`](./symbolic_opset8.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `registration.py_docs.md_docs.md`
- **Keyword Index**: `registration.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
