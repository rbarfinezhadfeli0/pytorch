# Documentation: `docs/torch/_functorch/_aot_autograd/subclass_parametrization.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_functorch/_aot_autograd/subclass_parametrization.py_docs.md`
- **Size**: 6,930 bytes (6.77 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_functorch/_aot_autograd/subclass_parametrization.py`

## File Metadata

- **Path**: `torch/_functorch/_aot_autograd/subclass_parametrization.py`
- **Size**: 4,124 bytes (4.03 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import dataclasses
import itertools
from collections.abc import Iterable
from typing import Any, Union

import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


# This is technically very similar to SubclassCreatingMeta
# in aot_autograd, but we don't need all the stuff in there
# so just recreated a new dataclass.
@dataclasses.dataclass
class SubclassCreationMeta:
    start_idx: int
    num_tensors: int
    class_type: Any
    attrs: dict[str, "SubclassCreationMeta"]
    metadata: Any
    outer_size: Iterable[Union[None, int, torch.SymInt]]
    outer_stride: Iterable[Union[None, int, torch.SymInt]]


class UnwrapTensorSubclass(torch.nn.Module):
    def forward(self, *tensors) -> torch.Tensor:  # type: ignore[no-untyped-def]
        todo: list[torch.Tensor] = list(tensors)

        def _unwrap_tensor_subclasses(subclass_meta, tensors, offset):  # type: ignore[no-untyped-def]
            if subclass_meta is None:
                return tensors[offset], offset + 1
            inner_tensors = {}
            for attr, meta in subclass_meta.attrs.items():
                built_tensor, offset = _unwrap_tensor_subclasses(meta, tensors, offset)
                inner_tensors[attr] = built_tensor
            rebuilt = subclass_meta.class_type.__tensor_unflatten__(
                inner_tensors,
                subclass_meta.metadata,
                subclass_meta.outer_size,
                subclass_meta.outer_stride,
            )
            return rebuilt, offset

        return _unwrap_tensor_subclasses(self.subclass_meta, todo, 0)[0]

    def right_inverse(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        assert type(tensor) is not torch.Tensor
        plain_tensors: list[torch.Tensor] = []

        def _create_subclass_meta(tensor, idx, plain_tensor_container):  # type: ignore[no-untyped-def]
            if type(tensor) is torch.Tensor:
                plain_tensor_container.append(tensor)
                return None, idx + 1
            inner_tensors_attrnames, metadata = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
            new_idx = idx
            attr_to_meta = {}
            for attr in inner_tensors_attrnames:
                val = getattr(tensor, attr)
                subclass_meta, new_idx = _create_subclass_meta(
                    val, new_idx, plain_tensor_container
                )
                attr_to_meta[attr] = subclass_meta
            return (
                SubclassCreationMeta(
                    start_idx=idx,
                    num_tensors=new_idx - idx,
                    class_type=type(tensor),
                    attrs=attr_to_meta,
                    metadata=metadata,
                    outer_size=tensor.size(),
                    outer_stride=tensor.stride(),
                ),
                new_idx,
            )

        self.subclass_meta = _create_subclass_meta(tensor, 0, plain_tensors)[0]
        return plain_tensors


def unwrap_tensor_subclass_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Model transformation that replaces all the parameters that are subclasses to plain tensors.
    This reduces runtime overhead of flattening/unflattening the parameters.

    This transformation adds parametrization with `torch.nn.utils.parametrize`.
    The FQNs of the subclass parameters will be changed and state_dict will become incompatible with the original model.
    E.g.
    Original model state_dict: {"p1": torch.testing._internal.TwoTensor}
    becomes: {"parametrizations.p2.original0": torch.Tensor, "parametrizations.p2.original1": torch.Tensor}

    """
    for name, tensor in itertools.chain(
        list(module.named_parameters(recurse=False)),
        # pyrefly: ignore [no-matching-overload]
        list(module.named_buffers(recurse=False)),
    ):
        if is_traceable_wrapper_subclass(tensor):
            torch.nn.utils.parametrize.register_parametrization(
                module, name, UnwrapTensorSubclass()
            )

    for child in module.children():
        unwrap_tensor_subclass_parameters(child)

    return module

```



## High-Level Overview


This Python file contains 3 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SubclassCreationMeta`, `UnwrapTensorSubclass`

**Functions defined**: `forward`, `_unwrap_tensor_subclasses`, `right_inverse`, `_create_subclass_meta`, `unwrap_tensor_subclass_parameters`

**Key imports**: dataclasses, itertools, Iterable, Any, Union, torch, is_traceable_wrapper_subclass


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_functorch/_aot_autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`
- `itertools`
- `collections.abc`: Iterable
- `typing`: Any, Union
- `torch`
- `torch.utils._python_dispatch`: is_traceable_wrapper_subclass


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

Files in the same folder (`torch/_functorch/_aot_autograd`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_cache.py_docs.md`](./autograd_cache.py_docs.md)
- [`functional_utils.py_docs.md`](./functional_utils.py_docs.md)
- [`logging_utils.py_docs.md`](./logging_utils.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`descriptors.py_docs.md`](./descriptors.py_docs.md)
- [`collect_metadata_analysis.py_docs.md`](./collect_metadata_analysis.py_docs.md)
- [`frontend_utils.py_docs.md`](./frontend_utils.py_docs.md)
- [`runtime_wrappers.py_docs.md`](./runtime_wrappers.py_docs.md)


## Cross-References

- **File Documentation**: `subclass_parametrization.py_docs.md`
- **Keyword Index**: `subclass_parametrization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_functorch/_aot_autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_functorch/_aot_autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/_functorch/_aot_autograd`):

- [`graph_compile.py_kw.md_docs.md`](./graph_compile.py_kw.md_docs.md)
- [`frontend_utils.py_kw.md_docs.md`](./frontend_utils.py_kw.md_docs.md)
- [`autograd_cache.py_docs.md_docs.md`](./autograd_cache.py_docs.md_docs.md)
- [`input_output_analysis.py_kw.md_docs.md`](./input_output_analysis.py_kw.md_docs.md)
- [`schemas.py_docs.md_docs.md`](./schemas.py_docs.md_docs.md)
- [`collect_metadata_analysis.py_docs.md_docs.md`](./collect_metadata_analysis.py_docs.md_docs.md)
- [`functional_utils.py_docs.md_docs.md`](./functional_utils.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`logging_utils.py_docs.md_docs.md`](./logging_utils.py_docs.md_docs.md)
- [`graph_capture.py_kw.md_docs.md`](./graph_capture.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `subclass_parametrization.py_docs.md_docs.md`
- **Keyword Index**: `subclass_parametrization.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
