# Documentation: `docs/torch/__future__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/__future__.py_docs.md`
- **Size**: 6,175 bytes (6.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/__future__.py`

## File Metadata

- **Path**: `torch/__future__.py`
- **Size**: 3,185 bytes (3.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
_overwrite_module_params_on_conversion: bool = False
_swap_module_params_on_conversion: bool = False


def set_overwrite_module_params_on_conversion(value: bool) -> None:
    """
    Sets whether to assign new tensors to the parameters instead of changing the
    existing parameters in-place when converting an ``nn.Module``.

    When enabled, the following methods will assign new parameters to the module:

    #. ``module.{device}()`` (e.g. :meth:`nn.Module.cuda()`) for moving a module between devices
    #. ``module.{dtype}()`` (e.g. :meth:`nn.Module.float()`) for converting a module to a different dtype
    #. :meth:`nn.Module.to`
    #. :meth:`nn.Module.to_empty`

    Args:
        value (bool): Whether to assign new tensors or not.

    """
    global _overwrite_module_params_on_conversion
    _overwrite_module_params_on_conversion = value


def get_overwrite_module_params_on_conversion() -> bool:
    """
    Returns whether to assign new tensors to the parameters instead of changing the
    existing parameters in-place when converting an :class:`torch.nn.Module`. Defaults to ``False``.

    See :func:`~torch.__future__.set_overwrite_module_params_on_conversion` for more information.
    """
    return _overwrite_module_params_on_conversion


def set_swap_module_params_on_conversion(value: bool) -> None:
    """
    Sets whether to use :func:`~torch.utils.swap_tensors` instead of setting ``.data`` to
    change the existing parameters in-place when converting an ``nn.Module`` and instead
    of ``param.copy_(state_dict[key])`` when loading a state dict into an ``nn.Module``.

    .. note::
        This function takes precedence over :func:`~torch.__future__.get_overwrite_module_params_on_conversion`

    When enabled, the following methods will swap the existing parameters in-place:

    #. ``module.{device}()`` (e.g. :meth:`nn.Module.cuda()`) for moving a module between devices
    #. ``module.{dtype}()`` (e.g. :meth:`nn.Module.float()`) for converting a module to a different dtype
    #. :meth:`nn.Module.to`
    #. :meth:`nn.Module.to_empty`
    #. :meth:`nn.Module.load_state_dict`

    The semantics for :meth:`~nn.Module.load_state_dict` when this is set are as follows:

    #. For each parameter/buffer, its corresponding ``state_dict['key']`` is transformed via
       :meth:`~torch.Tensor.module_load` (i.e. ``res = param.module_load(state_dict['key'])``)
    #. If necessary, ``res`` will be wrapped in an :class:`~nn.Parameter`
    #. The parameter/buffer in the module will be swapped via :func:`~torch.utils.swap_tensors`
       with ``res``

    Args:
        value (bool): Whether to use :func:`~torch.utils.swap_tensors` or not.

    """
    global _swap_module_params_on_conversion
    _swap_module_params_on_conversion = value


def get_swap_module_params_on_conversion() -> bool:
    """
    Returns whether to use :func:`~torch.utils.swap_tensors` instead of setting .data to
    change the existing parameters in-place when converting an ``nn.Module``. Defaults to ``False``.

    See :func:`~torch.__future__.set_swap_module_params_on_conversion` for more information.
    """
    return _swap_module_params_on_conversion

```



## High-Level Overview

"""    Sets whether to assign new tensors to the parameters instead of changing the    existing parameters in-place when converting an ``nn.Module``.    When enabled, the following methods will assign new parameters to the module:    #. ``module.{device}()`` (e.g. :meth:`nn.Module.cuda()`) for moving a module between devices    #. ``module.{dtype}()`` (e.g. :meth:`nn.Module.float()`) for converting a module to a different dtype    #. :meth:`nn.Module.to`    #. :meth:`nn.Module.to_empty`    Args:        value (bool): Whether to assign new tensors or not.

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `set_overwrite_module_params_on_conversion`, `get_overwrite_module_params_on_conversion`, `set_swap_module_params_on_conversion`, `get_swap_module_params_on_conversion`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*No imports detected.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_tensor_docs.py_docs.md`](./_tensor_docs.py_docs.md)
- [`_classes.py_docs.md`](./_classes.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`_meta_registrations.py_docs.md`](./_meta_registrations.py_docs.md)
- [`_appdirs.py_docs.md`](./_appdirs.py_docs.md)
- [`_tensor.py_docs.md`](./_tensor.py_docs.md)
- [`_streambase.py_docs.md`](./_streambase.py_docs.md)
- [`_lowrank.py_docs.md`](./_lowrank.py_docs.md)
- [`_size_docs.py_docs.md`](./_size_docs.py_docs.md)


## Cross-References

- **File Documentation**: `__future__.py_docs.md`
- **Keyword Index**: `__future__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch`):

- [`types.py_kw.md_docs.md`](./types.py_kw.md_docs.md)
- [`storage.py_docs.md_docs.md`](./storage.py_docs.md_docs.md)
- [`serialization.py_kw.md_docs.md`](./serialization.py_kw.md_docs.md)
- [`serialization.py_docs.md_docs.md`](./serialization.py_docs.md_docs.md)
- [`library.py_kw.md_docs.md`](./library.py_kw.md_docs.md)
- [`overrides.py_docs.md_docs.md`](./overrides.py_docs.md_docs.md)
- [`script.h_kw.md_docs.md`](./script.h_kw.md_docs.md)
- [`_sources.py_kw.md_docs.md`](./_sources.py_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`_torch_docs.py_docs.md_docs.md`](./_torch_docs.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `__future__.py_docs.md_docs.md`
- **Keyword Index**: `__future__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
