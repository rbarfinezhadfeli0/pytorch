# Documentation: `torch/fx/experimental/_dynamism.py`

## File Metadata

- **Path**: `torch/fx/experimental/_dynamism.py`
- **Size**: 4,528 bytes (4.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import re
from collections.abc import Callable
from typing import Any, Union

import torch
from torch.utils._pytree import tree_flatten_with_path, tree_map


KeyPath = tuple[Any, ...]
NonTensorShapeFn = Callable[[Union[int, float]], tuple[Any, ...]]

__all__ = [
    "normalize_source_name",
    "module_to_nested_dict",
    "track_dynamism_across_examples",
    "clone_and_convert_to_meta",
]


def normalize_source_name(name: str) -> str:
    # Match attribute access like .x and replace with ['x']
    return re.sub(r"\.([a-zA-Z_][a-zA-Z0-9_]*)", r"['\1']", name)


def module_to_nested_dict(module: torch.nn.Module) -> dict[str, Any]:
    """Recursively converts an nn.Module into a nested dictionary with explicit 'parameters' and 'modules' keys."""
    self_dict: dict[str, Any] = {}

    self_dict["_parameters"] = {}
    self_dict["_modules"] = {}

    for attr_name in dir(module):
        try:
            if not attr_name.startswith("_") and not callable(
                getattr(module, attr_name)
            ):
                attr_value = getattr(module, attr_name)
                if (
                    not isinstance(attr_value, torch.nn.Module)
                    and isinstance(attr_value, (int, float, torch.Tensor))
                    and type(attr_value) is not bool
                ):
                    self_dict[attr_name] = attr_value
        except NotImplementedError:
            # Skip attributes that raise NotImplementedError since they won't
            # contain any dynamism anyways.
            continue

    for name, param in module.named_parameters(recurse=False):
        self_dict["_parameters"][name] = param
    for name, buffer in module.named_buffers(recurse=False):
        self_dict["_parameters"][name] = buffer

    for name, submodule in module.named_children():
        self_dict["_modules"][name] = module_to_nested_dict(submodule)

    return self_dict


def track_dynamism_across_examples(
    example_inputs: list[Any],
) -> dict[Any, Any]:
    """
    This function analyzes a list of example inputs to determine the dynamism of their shapes.
    It tracks whether the dimensions of tensors or non-tensor values change across
    different examples. The function returns a dictionary where each key represents
    a path to a value in the input examples, and the corresponding value is a tuple
    indicating which dimensions are dynamic (i.e., change across examples). This
    helps in understanding how the structure of data varies across different instances.
    """
    tracking: dict[KeyPath, tuple[list[set[Any]], bool]] = {}

    for ex in example_inputs:
        if "self" in ex and isinstance(ex["self"], torch.nn.Module):
            ex["self"] = module_to_nested_dict(ex["self"])
        leaves_with_paths, _ = tree_flatten_with_path(ex)
        for key_path, value in leaves_with_paths:
            if not isinstance(value, (int, float, torch.Tensor)):
                continue
            if isinstance(value, torch.Tensor):
                shape: tuple[int | float, ...] = tuple(value.shape)
                is_tensor = True
            else:
                shape = (value,)
                is_tensor = False
            if key_path not in tracking:
                tracking[key_path] = ([set() for _ in range(len(shape))], is_tensor)
            else:
                dim_sets, flag = tracking[key_path]
                if flag != is_tensor:
                    pass
                while len(dim_sets) < len(shape):
                    dim_sets.append(set())
            for i, dim in enumerate(shape):
                tracking[key_path][0][i].add(dim)

    output: dict[Any, Any] = {}
    for key_path, (dim_sets, _is_tensor) in tracking.items():
        final_dyn = tuple(len(s) > 1 for s in dim_sets)
        key_str = "L" + "".join(f"{str(k)}" for k in key_path)
        key = key_path[0].key  # type: ignore[attr-defined]
        if key not in output:
            output[key] = {}
        output[key][key_str] = final_dyn
    return output


def clone_and_convert_to_meta(example_input: Any) -> Any:
    """
    This function takes a list of example inputs and for each tensor, clones it and converts it to device=meta.
    For non-tensor values, it keeps the reference. It uses pytree to handle nested structures recursively.
    """

    def transform_fn(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone().to(device="meta")
        return value

    return tree_map(transform_fn, example_input)

```



## High-Level Overview

"""Recursively converts an nn.Module into a nested dictionary with explicit 'parameters' and 'modules' keys."""    self_dict: dict[str, Any] = {}    self_dict["_parameters"] = {}    self_dict["_modules"] = {}    for attr_name in dir(module):        try:            if not attr_name.startswith("_") and not callable(                getattr(module, attr_name)            ):                attr_value = getattr(module, attr_name)                if (                    not isinstance(attr_value, torch.nn.Module)                    and isinstance(attr_value, (int, float, torch.Tensor))                    and type(attr_value) is not bool                ):                    self_dict[attr_name] = attr_value        except NotImplementedError:            # Skip attributes that raise NotImplementedError since they won't            # contain any dynamism anyways.            continue    for name, param in module.named_parameters(recurse=False):        self_dict["_parameters"][name] = param

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `normalize_source_name`, `module_to_nested_dict`, `track_dynamism_across_examples`, `clone_and_convert_to_meta`, `transform_fn`

**Key imports**: re, Callable, Any, Union, torch, tree_flatten_with_path, tree_map


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `re`
- `collections.abc`: Callable
- `typing`: Any, Union
- `torch`
- `torch.utils._pytree`: tree_flatten_with_path, tree_map


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
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

Files in the same folder (`torch/fx/experimental`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`graph_gradual_typechecker.py_docs.md`](./graph_gradual_typechecker.py_docs.md)
- [`validator.py_docs.md`](./validator.py_docs.md)
- [`accelerator_partitioner.py_docs.md`](./accelerator_partitioner.py_docs.md)
- [`unify_refinements.py_docs.md`](./unify_refinements.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`const_fold.py_docs.md`](./const_fold.py_docs.md)
- [`merge_matmul.py_docs.md`](./merge_matmul.py_docs.md)
- [`rewriter.py_docs.md`](./rewriter.py_docs.md)
- [`partitioner_utils.py_docs.md`](./partitioner_utils.py_docs.md)


## Cross-References

- **File Documentation**: `_dynamism.py_docs.md`
- **Keyword Index**: `_dynamism.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
