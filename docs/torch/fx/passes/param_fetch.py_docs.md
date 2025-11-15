# Documentation: `torch/fx/passes/param_fetch.py`

## File Metadata

- **Path**: `torch/fx/passes/param_fetch.py`
- **Size**: 3,741 bytes (3.65 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule


__all__ = [
    "default_matching",
    "extract_attrs_for_lowering",
    "lift_lowering_attrs_to_nodes",
]


# Matching method matches the attribute name of current version to the attribute name of `target_version`
@compatibility(is_backward_compatible=False)
def default_matching(name: str, target_version: int) -> str:
    """Default matching method"""
    return name


# This dict maps the nn.Module class name to the attribute name list that we want to fetch for lowering.
# The first integer in the tuple is the version number of the nn.Module class when we create the parameter list.
# If there's a version mismatch then it means the parameter names in the book might be mismatched with nn.Module.
module_fetch_book: dict[type, tuple[int, list[str], Callable[[str, int], str]]] = {
    torch.nn.modules.linear.Linear: (1, ["weight", "bias"], default_matching),
    torch.nn.modules.conv.Conv2d: (
        1,
        [
            "weight",
            "bias",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "padding_mode",
        ],
        default_matching,
    ),
    torch.nn.modules.batchnorm.BatchNorm2d: (
        2,
        ["weight", "bias", "running_mean", "running_var", "eps"],
        default_matching,
    ),
    torch.nn.modules.pooling.AdaptiveAvgPool2d: (1, [], default_matching),
    torch.nn.modules.pooling.MaxPool2d: (
        1,
        ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"],
        default_matching,
    ),
    torch.nn.modules.activation.ReLU: (1, ["inplace"], default_matching),
}


@compatibility(is_backward_compatible=False)
def extract_attrs_for_lowering(mod: nn.Module) -> dict[str, Any]:
    """If `mod` is in `module_fetch_book`, fetch the mod's attributes that in the `module_fetch_book`
    after checking module's version is compatible with the `module_fetch_book`.
    """
    attrs_for_lowering: dict[str, Any] = {}
    attrs_for_lowering["name"] = torch.typename(mod)

    if type(mod) in module_fetch_book:
        version, param_to_fetch, matching_method = module_fetch_book[type(mod)]
        if version < mod._version:
            raise RuntimeError(
                f"Fetcher version {version} try to fetch {torch.typename(mod)} version {mod._version}, "
                "please upgrade the module_fetch_book, open an issue and @842974287 "
                "or report a bug to AIACC team directly."
            )
        for attr in param_to_fetch:
            attrs_for_lowering[attr] = getattr(mod, matching_method(attr, mod._version))
    else:
        raise RuntimeError(
            f"{torch.typename(mod)} is not in the module_fetch_book yet, "
            "please add it to the module_fetch_book, open an issue and @842974287 "
            "or report a bug to AIACC team directly."
        )
    return attrs_for_lowering


@compatibility(is_backward_compatible=False)
def lift_lowering_attrs_to_nodes(fx_module: GraphModule) -> None:
    """Recursively traverse all `fx_module` nodes and fetch the module's attributes if the node is a leaf module."""
    submodules = dict(fx_module.named_modules())

    for node in fx_module.graph.nodes:
        if node.op == "call_module":
            if isinstance(submodules[node.target], GraphModule):
                lift_lowering_attrs_to_nodes(submodules[node.target])
            else:
                node.attrs_for_lowering = extract_attrs_for_lowering(
                    submodules[node.target]
                )

```



## High-Level Overview

"""Default matching method"""    return name# This dict maps the nn.Module class name to the attribute name list that we want to fetch for lowering.# The first integer in the tuple is the version number of the nn.Module class when we create the parameter list.# If there's a version mismatch then it means the parameter names in the book might be mismatched with nn.Module.module_fetch_book: dict[type, tuple[int, list[str], Callable[[str, int], str]]] = {    torch.nn.modules.linear.Linear: (1, ["weight", "bias"], default_matching),    torch.nn.modules.conv.Conv2d: (        1,        [            "weight",            "bias",            "kernel_size",            "stride",            "padding",            "dilation",            "groups",            "padding_mode",        ],        default_matching,    ),    torch.nn.modules.batchnorm.BatchNorm2d: (        2,        ["weight", "bias", "running_mean", "running_var", "eps"],        default_matching,    ),    torch.nn.modules.pooling.AdaptiveAvgPool2d: (1, [], default_matching),    torch.nn.modules.pooling.MaxPool2d: (        1,

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `default_matching`, `extract_attrs_for_lowering`, `lift_lowering_attrs_to_nodes`

**Key imports**: Callable, Any, torch, torch.nn as nn, compatibility, GraphModule


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: Any
- `torch`
- `torch.nn as nn`
- `torch.fx._compatibility`: compatibility
- `torch.fx.graph_module`: GraphModule


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

Files in the same folder (`torch/fx/passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`operator_support.py_docs.md`](./operator_support.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`graph_drawer.py_docs.md`](./graph_drawer.py_docs.md)
- [`shape_prop.py_docs.md`](./shape_prop.py_docs.md)
- [`split_utils.py_docs.md`](./split_utils.py_docs.md)
- [`runtime_assert.py_docs.md`](./runtime_assert.py_docs.md)
- [`splitter_base.py_docs.md`](./splitter_base.py_docs.md)
- [`graph_transform_observer.py_docs.md`](./graph_transform_observer.py_docs.md)
- [`fake_tensor_prop.py_docs.md`](./fake_tensor_prop.py_docs.md)


## Cross-References

- **File Documentation**: `param_fetch.py_docs.md`
- **Keyword Index**: `param_fetch.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
