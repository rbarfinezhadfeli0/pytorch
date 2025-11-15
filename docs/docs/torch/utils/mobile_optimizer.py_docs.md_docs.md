# Documentation: `docs/torch/utils/mobile_optimizer.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/mobile_optimizer.py_docs.md`
- **Size**: 9,599 bytes (9.37 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/mobile_optimizer.py`

## File Metadata

- **Path**: `torch/utils/mobile_optimizer.py`
- **Size**: 6,414 bytes (6.26 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
"""This module contains utility method for mobile model optimization and lint."""

import torch
from enum import Enum
from torch._C import _MobileOptimizerType as MobileOptimizerType
from typing import AnyStr

class LintCode(Enum):
    BUNDLED_INPUT = 1
    REQUIRES_GRAD = 2
    DROPOUT = 3
    BATCHNORM = 4

def optimize_for_mobile(
        script_module: torch.jit.ScriptModule,
        optimization_blocklist: set[MobileOptimizerType] | None = None,
        preserved_methods: list[AnyStr] | None = None,
        backend: str = 'CPU') -> torch.jit.RecursiveScriptModule:
    """
    Optimize a torch script module for mobile deployment.

    Args:
        script_module: An instance of torch script module with type of ScriptModule.
        optimization_blocklist: A set with type of MobileOptimizerType. When set is not passed,
            optimization method will run all the optimizer pass; otherwise, optimizer
            method will run the optimization pass that is not included inside optimization_blocklist.
        preserved_methods: A list of methods that needed to be preserved when freeze_module pass is invoked
        backend: Device type to use for running the result model ('CPU'(default), 'Vulkan' or 'Metal').
    Returns:
        A new optimized torch script module
    """
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(
            f'Got {type(script_module)}, but ScriptModule is expected.')

    if optimization_blocklist is None:
        optimization_blocklist = set()

    if preserved_methods is None:
        preserved_methods = []

    # Convert potential byte arrays into strings (if there is any) to pass type checking
    # Here we use a new name as assigning it back to preserved_methods will invoke
    # mypy errors (i.e. List[AnyStr] = List[str])
    preserved_methods_str: list[str] = [str(method) for method in preserved_methods]

    bundled_inputs_attributes = _get_bundled_inputs_preserved_attributes(script_module, preserved_methods_str)
    if all(hasattr(script_module, method) for method in bundled_inputs_attributes):
        preserved_methods_str = list(set(preserved_methods_str + bundled_inputs_attributes))

    non_exist_methods = [method for method in preserved_methods_str if not hasattr(script_module, method)]
    if non_exist_methods:
        raise AttributeError(
            f"The following methods to preserve do not exist in script_module: {', '.join(non_exist_methods)}")

    backend = backend.lower()
    if backend == 'cpu':
        optimized_cpp_module = torch._C._jit_pass_optimize_for_mobile(
            script_module._c,
            optimization_blocklist,
            preserved_methods_str)
    elif backend == 'vulkan':
        optimized_cpp_module = torch._C._jit_pass_vulkan_optimize_for_mobile(
            script_module._c,
            optimization_blocklist,
            preserved_methods_str)
    elif backend == 'metal':
        optimized_cpp_module = torch._C._jit_pass_metal_optimize_for_mobile(script_module._c, preserved_methods_str)
    else:
        raise TypeError("Unknown backend, must be one of 'CPU', 'Vulkan' or 'Metal'")

    return torch.jit._recursive.wrap_cpp_module(optimized_cpp_module)


def generate_mobile_module_lints(script_module: torch.jit.ScriptModule):
    """
    Generate a list of lints for a given torch script module.

    Args:
        script_module: An instance of torch script module with type of ScriptModule.

    Returns:
        lint_map: A list of dictionary that contains modules lints
    """
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(
            f'Got {type(script_module)}, but ScriptModule is expected.')

    lint_list = []

    if not hasattr(script_module, "_generate_bundled_inputs_for_forward"):
        lint_list.append({"name": LintCode.BUNDLED_INPUT.name, "message": "No bundled input for forward, please add bundled inputs "
                          "before saving the module using torch.utils.bundled_inputs.augment_model_with_bundled_inputs."})

    for name, param in script_module.named_parameters():
        if param.requires_grad:
            lint_list.append({"name": LintCode.REQUIRES_GRAD.name, "message": f"Param {name} requires grad, "
                             "please set torch.no_grad() to reduce memory usage and improve computation speed during "
                              "inference phase."})

    op_names = torch.jit.export_opnames(script_module)
    for op_name in op_names:
        if "dropout" in op_name:
            lint_list.append({"name": LintCode.DROPOUT.name,
                              "message": f"Operator {op_name} exists, remember to call eval() before "
                              "saving the module.and call torch.utils.mobile_optimizer.optimize_for_mobile to drop dropout "
                              "operator."})
        if "batch_norm" in op_name:
            lint_list.append({"name": LintCode.BATCHNORM.name,
                              "message": f"Operator {op_name} exists, remember to call eval() before "
                              "saving the module and call torch.utils.mobile_optimizer.optimize_for_mobile to drop batch_norm "
                              "operator."})

    return lint_list

def _get_bundled_inputs_preserved_attributes(script_module: torch.jit.ScriptModule, preserved_methods: list[str]) -> list[str]:

    bundled_inputs_attributes = []
    # Has bundled inputs for forward
    if hasattr(script_module, 'get_all_bundled_inputs'):
        bundled_inputs_attributes.append('get_all_bundled_inputs')
        bundled_inputs_attributes.append('get_num_bundled_inputs')

    # Bundled inputs in module after the change that introduced bundled inputs for multiple functions
    if hasattr(script_module, 'get_bundled_inputs_functions_and_info'):
        bundled_inputs_attributes.append('get_bundled_inputs_functions_and_info')
        all_info = script_module.get_bundled_inputs_functions_and_info()
        for function_name in all_info:
            if function_name not in preserved_methods:
                bundled_inputs_attributes.append(function_name)
            bundled_inputs_attributes.append("get_all_bundled_inputs_for_" + function_name)
            bundled_inputs_attributes.append("_bundled_inputs_deflated_" + function_name)

    return bundled_inputs_attributes

```



## High-Level Overview

"""This module contains utility method for mobile model optimization and lint."""import torchfrom enum import Enumfrom torch._C import _MobileOptimizerType as MobileOptimizerTypefrom typing import AnyStrclass LintCode(Enum):    BUNDLED_INPUT = 1    REQUIRES_GRAD = 2    DROPOUT = 3    BATCHNORM = 4def optimize_for_mobile(        script_module: torch.jit.ScriptModule,        optimization_blocklist: set[MobileOptimizerType] | None = None,        preserved_methods: list[AnyStr] | None = None,        backend: str = 'CPU') -> torch.jit.RecursiveScriptModule:

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintCode`

**Functions defined**: `optimize_for_mobile`, `generate_mobile_module_lints`, `_get_bundled_inputs_preserved_attributes`

**Key imports**: torch, Enum, _MobileOptimizerType as MobileOptimizerType, AnyStr


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `enum`: Enum
- `torch._C`: _MobileOptimizerType as MobileOptimizerType
- `typing`: AnyStr


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/utils`):

- [`_zip.py_docs.md`](./_zip.py_docs.md)
- [`weak.py_docs.md`](./weak.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_cpp_embed_headers.py_docs.md`](./_cpp_embed_headers.py_docs.md)
- [`_cpp_extension_versioner.py_docs.md`](./_cpp_extension_versioner.py_docs.md)
- [`module_tracker.py_docs.md`](./module_tracker.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`_content_store.py_docs.md`](./_content_store.py_docs.md)
- [`_triton.py_docs.md`](./_triton.py_docs.md)
- [`file_baton.py_docs.md`](./file_baton.py_docs.md)


## Cross-References

- **File Documentation**: `mobile_optimizer.py_docs.md`
- **Keyword Index**: `mobile_optimizer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/utils`):

- [`show_pickle.py_docs.md_docs.md`](./show_pickle.py_docs.md_docs.md)
- [`file_baton.py_docs.md_docs.md`](./file_baton.py_docs.md_docs.md)
- [`_filelock.py_kw.md_docs.md`](./_filelock.py_kw.md_docs.md)
- [`_config_module.py_docs.md_docs.md`](./_config_module.py_docs.md_docs.md)
- [`cpp_extension.py_docs.md_docs.md`](./cpp_extension.py_docs.md_docs.md)
- [`checkpoint.py_docs.md_docs.md`](./checkpoint.py_docs.md_docs.md)
- [`module_tracker.py_kw.md_docs.md`](./module_tracker.py_kw.md_docs.md)
- [`dlpack.py_docs.md_docs.md`](./dlpack.py_docs.md_docs.md)
- [`_import_utils.py_kw.md_docs.md`](./_import_utils.py_kw.md_docs.md)
- [`_traceback.py_kw.md_docs.md`](./_traceback.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `mobile_optimizer.py_docs.md_docs.md`
- **Keyword Index**: `mobile_optimizer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
