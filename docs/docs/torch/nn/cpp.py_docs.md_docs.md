# Documentation: `docs/torch/nn/cpp.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/cpp.py_docs.md`
- **Size**: 5,567 bytes (5.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/cpp.py`

## File Metadata

- **Path**: `torch/nn/cpp.py`
- **Size**: 3,100 bytes (3.03 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
"""Functionality for Python <-> C++ frontend inter-op."""

from torch import nn


class OrderedDictWrapper:
    """A wrapper around a C++ OrderedDict.

    It dynamically evaluates the OrderedDict getter on a bound C++ module, such
    that new changes on the C++ side are picked up. Otherwise accessing e.g.
    ``cpp_module._parameters`` just once would get a frozen copy of the parameters
    at the time of access. ``torch.nn.Module`` accesses ``_parameters`` et al. via ``self.__dict__``
    so using properties does not work.
    """

    def __init__(self, cpp_module, attr) -> None:
        self.cpp_module = cpp_module
        self.attr = attr

    @property
    def cpp_dict(self):
        return getattr(self.cpp_module, self.attr)

    # Magic methods cannot be assigned dynamically and bypass ``getattr``, so we
    # must manually override them.

    def items(self):
        return self.cpp_dict.items()

    def keys(self):
        return self.cpp_dict.keys()

    def values(self):
        return self.cpp_dict.values()

    def __iter__(self):
        return self.cpp_dict.__iter__()

    def __len__(self) -> int:
        return self.cpp_dict.__len__()

    def __contains__(self, key) -> bool:
        return self.cpp_dict.__contains__(key)

    def __getitem__(self, key):
        return self.cpp_dict.__getitem__(key)


class ModuleWrapper(nn.Module):
    """A subclass of ``torch.nn.Module`` that wraps a C++ frontend module and delegates all access."""

    def __init__(self, cpp_module) -> None:
        # Assign before the super class constructor so ``self.training`` can be
        # assigned to in the super class constructor.
        self.cpp_module = cpp_module
        super().__init__()
        self._parameters = OrderedDictWrapper(cpp_module, "_parameters")  # type: ignore[assignment]
        self._buffers: OrderedDictWrapper = OrderedDictWrapper(cpp_module, "_buffers")  # type: ignore[assignment]
        self._modules: OrderedDictWrapper = OrderedDictWrapper(cpp_module, "_modules")  # type: ignore[assignment]
        for attr in dir(cpp_module):
            # Skip magic methods and the three attributes above.
            if not attr.startswith("_"):
                setattr(self, attr, getattr(self.cpp_module, attr))

    def _apply(self, fn, recurse=True):
        for param in self.parameters():
            # Tensors stored in modules are graph leaves, and we don't
            # want to create copy nodes, so we have to unpack the data.
            param.data = fn(param.data)
            if param._grad is not None:
                param._grad.data = fn(param._grad.data)

        for buf in self.buffers():
            buf.data = fn(buf.data)

        return self

    # nn.Module defines training as a boolean
    @property  # type: ignore[override]
    # pyrefly: ignore [bad-override]
    def training(self):
        return self.cpp_module.training

    @training.setter
    def training(self, mode) -> None:
        self.cpp_module.train(mode)

    def __repr__(self) -> str:
        return self.cpp_module.__repr__()

```



## High-Level Overview

"""Functionality for Python <-> C++ frontend inter-op."""from torch import nnclass OrderedDictWrapper:

This Python file contains 5 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OrderedDictWrapper`, `ModuleWrapper`

**Functions defined**: `__init__`, `cpp_dict`, `items`, `keys`, `values`, `__iter__`, `__len__`, `__contains__`, `__getitem__`, `__init__`, `_apply`, `training`, `training`, `__repr__`

**Key imports**: nn


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`: nn


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

Files in the same folder (`torch/nn`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`common_types.py_docs.md`](./common_types.py_docs.md)
- [`parameter.pyi_docs.md`](./parameter.pyi_docs.md)
- [`functional.py_docs.md`](./functional.py_docs.md)
- [`grad.py_docs.md`](./grad.py_docs.md)
- [`_reduction.py_docs.md`](./_reduction.py_docs.md)
- [`init.py_docs.md`](./init.py_docs.md)
- [`parameter.py_docs.md`](./parameter.py_docs.md)


## Cross-References

- **File Documentation**: `cpp.py_docs.md`
- **Keyword Index**: `cpp.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/nn`):

- [`grad.py_kw.md_docs.md`](./grad.py_kw.md_docs.md)
- [`init.py_docs.md_docs.md`](./init.py_docs.md_docs.md)
- [`parameter.pyi_kw.md_docs.md`](./parameter.pyi_kw.md_docs.md)
- [`common_types.py_docs.md_docs.md`](./common_types.py_docs.md_docs.md)
- [`common_types.py_kw.md_docs.md`](./common_types.py_kw.md_docs.md)
- [`functional.py_kw.md_docs.md`](./functional.py_kw.md_docs.md)
- [`_reduction.py_docs.md_docs.md`](./_reduction.py_docs.md_docs.md)
- [`init.py_kw.md_docs.md`](./init.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `cpp.py_docs.md_docs.md`
- **Keyword Index**: `cpp.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
