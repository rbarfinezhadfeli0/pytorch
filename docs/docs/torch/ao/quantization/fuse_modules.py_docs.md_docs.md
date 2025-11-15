# Documentation: `docs/torch/ao/quantization/fuse_modules.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/fuse_modules.py_docs.md`
- **Size**: 9,525 bytes (9.30 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/fuse_modules.py`

## File Metadata

- **Path**: `torch/ao/quantization/fuse_modules.py`
- **Size**: 6,794 bytes (6.63 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import copy

import torch.nn as nn

# for backward compatibility
from torch.ao.quantization.fuser_method_mappings import (  # noqa: F401  # noqa: F401
    fuse_conv_bn,
    fuse_conv_bn_relu,
    get_fuser_method,
)
from torch.nn.utils.parametrize import type_before_parametrizations


__all__ = [
    "fuse_known_modules",
    "fuse_modules",
    "fuse_modules_qat",
]


# Generalization of getattr
def _get_module(model, submodule_key):
    tokens = submodule_key.split(".")
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


# Generalization of setattr
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split(".")
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)

    setattr(cur_mod, tokens[-1], module)


def fuse_known_modules(mod_list, is_qat, additional_fuser_method_mapping=None):
    r"""Return a list of known fuse modules.

    Returns a list of modules that fuses the operations specified
     in the input module list.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, bn
    linear, relu
    For these sequences, the first element in the output module list performs
    the fused operation. The rest of the elements are set to nn.Identity()
    """
    types = tuple(type_before_parametrizations(m) for m in mod_list)
    fuser_method = get_fuser_method(types, additional_fuser_method_mapping)
    if fuser_method is None:
        raise NotImplementedError(f"Cannot fuse modules: {types}")
    new_mod: list[nn.Module | None] = [None] * len(mod_list)
    fused = fuser_method(is_qat, *mod_list)
    # NOTE: forward hooks not processed in the two following for loops will be lost after the fusion
    # Move pre forward hooks of the base module to resulting fused module
    for pre_hook_fn in mod_list[0]._forward_pre_hooks.values():
        fused.register_forward_pre_hook(pre_hook_fn)
    mod_list[0]._forward_pre_hooks.clear()
    # Move post forward hooks of the last module to resulting fused module
    for hook_fn in mod_list[-1]._forward_hooks.values():
        fused.register_forward_hook(hook_fn)
    mod_list[-1]._forward_hooks.clear()
    new_mod[0] = fused

    for i in range(1, len(mod_list)):
        identity = nn.Identity()
        identity.training = mod_list[0].training
        new_mod[i] = identity

    return new_mod


def _fuse_modules_helper(
    model,
    modules_to_fuse,
    is_qat,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    if fuse_custom_config_dict is None:
        fuse_custom_config_dict = {}
    additional_fuser_method_mapping = fuse_custom_config_dict.get(
        "additional_fuser_method_mapping", {}
    )
    mod_list = [_get_module(model, item) for item in modules_to_fuse]

    # Fuse list of modules
    new_mod_list = fuser_func(mod_list, is_qat, additional_fuser_method_mapping)

    # Replace original module list with fused module list
    for i, item in enumerate(modules_to_fuse):
        _set_module(model, item, new_mod_list[i])


def _fuse_modules(
    model,
    modules_to_fuse,
    is_qat,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    if not inplace:
        model = copy.deepcopy(model)

    if all(isinstance(module_element, str) for module_element in modules_to_fuse):
        # Handle case of modules_to_fuse being a list
        _fuse_modules_helper(
            model, modules_to_fuse, is_qat, fuser_func, fuse_custom_config_dict
        )
    else:
        # Handle case of modules_to_fuse being a list of lists
        for module_list in modules_to_fuse:
            _fuse_modules_helper(
                model, module_list, is_qat, fuser_func, fuse_custom_config_dict
            )
    return model


def fuse_modules(
    model,
    modules_to_fuse,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    r"""Fuse a list of modules into a single module.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, relu
    bn, relu
    All other sequences are left unchanged.
    For these sequences, replaces the first item in the list
    with the fused module, replacing the rest of the modules
    with identity.

    Args:
        model: Model containing the modules to be fused
        modules_to_fuse: list of list of module names to fuse. Can also be a list
                         of strings if there is only a single list of modules to fuse.
        inplace: bool specifying if fusion happens in place on the model, by default
                 a new model is returned
        fuser_func: Function that takes in a list of modules and outputs a list of fused modules
                    of the same length. For example,
                    fuser_func([convModule, BNModule]) returns the list [ConvBNModule, nn.Identity()]
                    Defaults to torch.ao.quantization.fuse_known_modules
        `fuse_custom_config_dict`: custom configuration for fusion

    .. code-block:: python

       # Example of fuse_custom_config_dict
       fuse_custom_config_dict = {
           # Additional fuser_method mapping
           "additional_fuser_method_mapping": {
               (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn
           },
       }

    Returns:
        model with fused modules. A new copy is created if inplace=True.

    Examples::

            >>> # xdoctest: +SKIP
            >>> m = M().eval()
            >>> # m is a module containing the sub-modules below
            >>> modules_to_fuse = [ ['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

            >>> m = M().eval()
            >>> # Alternately provide a single list of modules to fuse
            >>> modules_to_fuse = ['conv1', 'bn1', 'relu1']
            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

    """
    return _fuse_modules(
        model,
        modules_to_fuse,
        is_qat=False,
        inplace=inplace,
        fuser_func=fuser_func,
        fuse_custom_config_dict=fuse_custom_config_dict,
    )


def fuse_modules_qat(
    model,
    modules_to_fuse,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    """QAT version for `fuse_modules`."""
    return _fuse_modules(
        model,
        modules_to_fuse,
        is_qat=True,
        inplace=inplace,
        fuser_func=fuser_func,
        fuse_custom_config_dict=fuse_custom_config_dict,
    )

```



## High-Level Overview

r"""Return a list of known fuse modules.    Returns a list of modules that fuses the operations specified     in the input module list.    Fuses only the following sequence of modules:    conv, bn    conv, bn, relu

This Python file contains 0 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_get_module`, `_set_module`, `fuse_known_modules`, `_fuse_modules_helper`, `_fuse_modules`, `fuse_modules`, `fuse_modules_qat`

**Key imports**: copy, torch.nn as nn, type_before_parametrizations


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `torch.nn as nn`
- `torch.nn.utils.parametrize`: type_before_parametrizations


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/ao/quantization`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`quant_type.py_docs.md`](./quant_type.py_docs.md)
- [`fake_quantize.py_docs.md`](./fake_quantize.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`_equalize.py_docs.md`](./_equalize.py_docs.md)
- [`quantize.py_docs.md`](./quantize.py_docs.md)
- [`_learnable_fake_quantize.py_docs.md`](./_learnable_fake_quantize.py_docs.md)
- [`observer.py_docs.md`](./observer.py_docs.md)
- [`pattern.md_docs.md`](./pattern.md_docs.md)


## Cross-References

- **File Documentation**: `fuse_modules.py_docs.md`
- **Keyword Index**: `fuse_modules.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/quantization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization`, which is part of the **core PyTorch library**.



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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/ao/quantization`):

- [`_correct_bias.py_kw.md_docs.md`](./_correct_bias.py_kw.md_docs.md)
- [`quant_type.py_kw.md_docs.md`](./quant_type.py_kw.md_docs.md)
- [`qconfig.py_docs.md_docs.md`](./qconfig.py_docs.md_docs.md)
- [`_learnable_fake_quantize.py_kw.md_docs.md`](./_learnable_fake_quantize.py_kw.md_docs.md)
- [`quantize_fx.py_kw.md_docs.md`](./quantize_fx.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`observer.py_kw.md_docs.md`](./observer.py_kw.md_docs.md)
- [`fuser_method_mappings.py_kw.md_docs.md`](./fuser_method_mappings.py_kw.md_docs.md)
- [`quantize.py_kw.md_docs.md`](./quantize.py_kw.md_docs.md)
- [`qconfig_mapping.py_kw.md_docs.md`](./qconfig_mapping.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `fuse_modules.py_docs.md_docs.md`
- **Keyword Index**: `fuse_modules.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
