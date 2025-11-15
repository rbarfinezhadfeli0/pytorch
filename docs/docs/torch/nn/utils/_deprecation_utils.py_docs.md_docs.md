# Documentation: `docs/torch/nn/utils/_deprecation_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/utils/_deprecation_utils.py_docs.md`
- **Size**: 4,982 bytes (4.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/utils/_deprecation_utils.py`

## File Metadata

- **Path**: `torch/nn/utils/_deprecation_utils.py`
- **Size**: 1,698 bytes (1.66 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import importlib
import warnings
from collections.abc import Callable


_MESSAGE_TEMPLATE = (
    r"Usage of '{old_location}' is deprecated; please use '{new_location}' instead."
)


def lazy_deprecated_import(
    all: list[str],
    old_module: str,
    new_module: str,
) -> Callable:
    r"""Import utility to lazily import deprecated packages / modules / functional.

    The old_module and new_module are also used in the deprecation warning defined
    by the `_MESSAGE_TEMPLATE`.

    Args:
        all: The list of the functions that are imported. Generally, the module's
            __all__ list of the module.
        old_module: Old module location
        new_module: New module location / Migrated location

    Returns:
        Callable to assign to the `__getattr__`

    Usage:

        # In the `torch/nn/quantized/functional.py`
        from torch.nn.utils._deprecation_utils import lazy_deprecated_import
        _MIGRATED_TO = "torch.ao.nn.quantized.functional"
        __getattr__ = lazy_deprecated_import(
            all=__all__,
            old_module=__name__,
            new_module=_MIGRATED_TO)
    """
    warning_message = _MESSAGE_TEMPLATE.format(
        old_location=old_module, new_location=new_module
    )

    def getattr_dunder(name: str) -> None:
        if name in all:
            # We are using the "RuntimeWarning" to make sure it is not
            # ignored by default.
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
            package = importlib.import_module(new_module)
            return getattr(package, name)
        raise AttributeError(f"Module {new_module!r} has no attribute {name!r}.")

    return getattr_dunder

```



## High-Level Overview

r"""Import utility to lazily import deprecated packages / modules / functional.    The old_module and new_module are also used in the deprecation warning defined    by the `_MESSAGE_TEMPLATE`.    Args:        all: The list of the functions that are imported. Generally, the module's            __all__ list of the module.        old_module: Old module location        new_module: New module location / Migrated location    Returns:        Callable to assign to the `__getattr__`    Usage:        # In the `torch/nn/quantized/functional.py`        from torch.nn.utils._deprecation_utils import lazy_deprecated_import        _MIGRATED_TO = "torch.ao.nn.quantized.functional"        __getattr__ = lazy_deprecated_import(            all=__all__,            old_module=__name__,            new_module=_MIGRATED_TO)

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `lazy_deprecated_import`, `getattr_dunder`

**Key imports**: importlib, warnings, Callable, deprecated packages , lazy_deprecated_import


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `importlib`
- `warnings`
- `collections.abc`: Callable
- `deprecated packages `
- `torch.nn.utils._deprecation_utils`: lazy_deprecated_import


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

Files in the same folder (`torch/nn/utils`):

- [`parametrizations.py_docs.md`](./parametrizations.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`stateless.py_docs.md`](./stateless.py_docs.md)
- [`parametrize.py_docs.md`](./parametrize.py_docs.md)
- [`spectral_norm.py_docs.md`](./spectral_norm.py_docs.md)
- [`prune.py_docs.md`](./prune.py_docs.md)
- [`fusion.py_docs.md`](./fusion.py_docs.md)
- [`weight_norm.py_docs.md`](./weight_norm.py_docs.md)


## Cross-References

- **File Documentation**: `_deprecation_utils.py_docs.md`
- **Keyword Index**: `_deprecation_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`docs/torch/nn/utils`):

- [`init.py_docs.md_docs.md`](./init.py_docs.md_docs.md)
- [`memory_format.py_kw.md_docs.md`](./memory_format.py_kw.md_docs.md)
- [`_named_member_accessor.py_kw.md_docs.md`](./_named_member_accessor.py_kw.md_docs.md)
- [`_per_sample_grad.py_kw.md_docs.md`](./_per_sample_grad.py_kw.md_docs.md)
- [`_named_member_accessor.py_docs.md_docs.md`](./_named_member_accessor.py_docs.md_docs.md)
- [`parametrize.py_docs.md_docs.md`](./parametrize.py_docs.md_docs.md)
- [`memory_format.py_docs.md_docs.md`](./memory_format.py_docs.md_docs.md)
- [`weight_norm.py_kw.md_docs.md`](./weight_norm.py_kw.md_docs.md)
- [`convert_parameters.py_kw.md_docs.md`](./convert_parameters.py_kw.md_docs.md)
- [`parametrizations.py_docs.md_docs.md`](./parametrizations.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_deprecation_utils.py_docs.md_docs.md`
- **Keyword Index**: `_deprecation_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
