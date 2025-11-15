# Documentation: `docs/torch/nn/utils/convert_parameters.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/utils/convert_parameters.py_docs.md`
- **Size**: 5,908 bytes (5.77 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/utils/convert_parameters.py`

## File Metadata

- **Path**: `torch/nn/utils/convert_parameters.py`
- **Size**: 3,213 bytes (3.14 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Iterable

import torch


def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Flatten an iterable of parameters into a single vector.

    Args:
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Copy slices of a vector into an iterable of parameters.

    Args:
        vec (Tensor): a single vector representing the parameters of a model.
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def _check_param_device(param: torch.Tensor, old_param_device: int | None) -> int:
    r"""Check if the parameters are located on the same device.

    Currently, the conversion between model parameters and single vector form is not supported
    for multiple allocations, e.g. parameters in different GPUs/PrivateUse1s, or mixture of CPU/GPU/PrivateUse1.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """
    # Meet the first parameter
    support_device_types = ["cuda", torch._C._get_privateuse1_backend_name()]
    if old_param_device is None:
        old_param_device = (
            param.get_device() if param.device.type in support_device_types else -1
        )
    else:
        warn = False
        if (
            param.device.type in support_device_types
        ):  # Check if in same GPU/PrivateUse1
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device

```



## High-Level Overview

r"""Flatten an iterable of parameters into a single vector.    Args:        parameters (Iterable[Tensor]): an iterable of Tensors that are the            parameters of a model.    Returns:        The parameters represented by a single vector

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `parameters_to_vector`, `vector_to_parameters`, `_check_param_device`

**Key imports**: Iterable, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Iterable
- `torch`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/nn/utils`):

- [`_deprecation_utils.py_docs.md`](./_deprecation_utils.py_docs.md)
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

- **File Documentation**: `convert_parameters.py_docs.md`
- **Keyword Index**: `convert_parameters.py_kw.md`
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

*No specific patterns automatically detected.*


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

- **File Documentation**: `convert_parameters.py_docs.md_docs.md`
- **Keyword Index**: `convert_parameters.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
