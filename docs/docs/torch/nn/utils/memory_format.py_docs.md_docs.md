# Documentation: `docs/torch/nn/utils/memory_format.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/utils/memory_format.py_docs.md`
- **Size**: 13,030 bytes (12.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/utils/memory_format.py`

## File Metadata

- **Path**: `torch/nn/utils/memory_format.py`
- **Size**: 8,274 bytes (8.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import TypeVar

import torch


_M = TypeVar("_M", bound="torch.nn.Module")


def convert_conv2d_weight_memory_format(
    module: _M, memory_format: torch.memory_format
) -> _M:
    r"""Convert ``memory_format`` of ``nn.Conv2d.weight`` to ``memory_format``.

    The conversion recursively applies to nested ``nn.Module``, including ``module``.
    Note that it only changes the memory_format, but not the semantics of each dimensions.
    This function is used to facilitate the computation to adopt NHWC kernels, which
    provides considerable speed up for fp16 data on CUDA devices with compute capability >= 7.0

    .. note::
        Calling ``model.to(memory_format=torch.channels_last)`` is more aggressive
        than the utility function ``convert_conv2d_weight_memory_format``. Any
        layer with 4d weight will be affected by ``model.to``, which does not
        necessarily benefit from conversion to specified ``memory_format``.
        One place we are confident in is that NHWC(channels_last) conversion for
        convolution in cuDNN, as it is beneficial to run convolution in NHWC,
        even in cases where we have to apply permutation to input tensors.

        Hence our strategy here is to convert only the weight of convolution to
        channels_last. This ensures that;
        1. Fast convolution kernels will be used, the benefit of which could
        outweigh overhead of permutation (if input is not in the same format).
        2. No unnecessary permutations are applied on layers that do not benefit
        from memory_format conversion.

        The optimal case is that, layers between convolution layers are channels
        last compatible. Input tensor would be permuted to channels last when it
        encounters the first convolution layer and stay in that memory format.
        Hence following convolutions will not need to permute its input tensor.

        In case where a channels last incompatible layer is between convolution
        layers, we need to permute the input tensor back to contiguous format
        for that layer. The input tensor will go through the remaining layers in
        contiguous format and be permuted to channels last when it encounters
        another convolution layer. There's no point in propagating that
        permutation to an earlier layer, as most layers are quite agnostic to
        ``memory_format``.

        This claim might change when PyTorch supports fusion of permutation, as
        there might have been a better spot to fuse the permutation other than
        immediately before a convolution.

    Args:
        module (nn.Module): ``nn.Conv2d`` & ``nn.ConvTranspose2d`` or container
                            ``nn.Module``
        memory_format: user specified ``memory_format``,
            e.g. ``torch.channels_last`` or ``torch.contiguous_format``

    Returns:
        The original module with updated ``nn.Conv2d``

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:CUBLAS_WORKSPACE_CONFIG)
        >>> input = torch.randint(
        ...     1, 10, (2, 8, 4, 4), dtype=torch.float16, device="cuda"
        ... )
        >>> model = nn.Sequential(
        >>>     nn.Conv2d(8, 4, 3)).cuda().half()
        >>> # This is identical to:
        >>> # nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)
        >>> model = nn.utils.convert_conv2d_weight_memory_format(
        ...     model, torch.channels_last
        ... )
        >>> out = model(input)
    """
    # TODO: expand this to `_ConvNd` when channels_last support is extended
    # beyond only 4d tensors.
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        weight_data = module.weight.detach().clone(memory_format=memory_format)
        module.weight.data = weight_data.resize_(
            weight_data.size(), memory_format=memory_format
        )
    for child in module.children():
        convert_conv2d_weight_memory_format(child, memory_format)
    # pyrefly: ignore [bad-return]
    return module


def convert_conv3d_weight_memory_format(
    module: _M, memory_format: torch.memory_format
) -> _M:
    r"""Convert ``memory_format`` of ``nn.Conv3d.weight`` to ``memory_format``
    The conversion recursively applies to nested ``nn.Module``, including ``module``.
    Note that it only changes the memory_format, but not the semantics of each dimensions.
    This function is used to facilitate the computation to adopt NHWC kernels, which
    provides considerable speed up for fp16 data on CUDA devices with compute capability >= 7.0

    .. note::
        Calling ``model.to(memory_format=torch.channels_last_3d)`` is more aggressive
        than the utility function ``convert_conv3d_weight_memory_format``. Any
        layer with 4d weight will be affected by ``model.to``, which does not
        necessarily benefit from conversion to specified ``memory_format``.
        One place we are confident in is that NDHWC(channels_last_3d) conversion for
        convolution in cuDNN, as it is beneficial to run convolution in NDHWC,
        even in cases where we have to apply permutation to input tensors.

        Hence our strategy here is to convert only the weight of convolution to
        channels_last_3d. This ensures that;
        1. Fast convolution kernels will be used, the benefit of which could
        outweigh overhead of permutation (if input is not in the same format).
        2. No unnecessary permutations are applied on layers that do not benefit
        from memory_format conversion.

        The optimal case is that, layers between convolution layers are channels
        last compatible. Input tensor would be permuted to channels last when it
        encounters the first convolution layer and stay in that memory format.
        Hence following convolutions will not need to permute its input tensor.

        In case where a channels last incompatible layer is between convolution
        layers, we need to permute the input tensor back to contiguous format
        for that layer. The input tensor will go through the remaining layers in
        contiguous format and be permuted to channels last when it encounters
        another convolution layer. There's no point in propagating that
        permutation to an earlier layer, as most layers are quite agnostic to
        ``memory_format``.

        This claim might change when PyTorch supports fusion of permutation, as
        there might have been a better spot to fuse the permutation other than
        immediately before a convolution.

    Args:
        module (nn.Module): ``nn.Conv3d`` & ``nn.ConvTranspose3d`` or container
                            ``nn.Module``
        memory_format: user specified ``memory_format``,
            e.g. ``torch.channels_last`` or ``torch.contiguous_format``

    Returns:
        The original module with updated ``nn.Conv3d``

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:CUBLAS_WORKSPACE_CONFIG)
        >>> input = torch.randint(
        ...     1, 10, (2, 8, 4, 4, 4), dtype=torch.float16, device="cuda"
        ... )
        >>> model = nn.Sequential(
        >>>     nn.Conv3d(8, 4, 3)).cuda().half()
        >>> # This is identical to:
        >>> # nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
        >>> model = nn.utils.convert_conv3d_weight_memory_format(
        ...     model, torch.channels_last_3d
        ... )
        >>> out = model(input)
    """

    # TODO: expand this to `_ConvNd` when channels_last support is extended
    # beyond only 4d tensors.
    if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
        weight_data = module.weight.detach().clone(memory_format=memory_format)
        module.weight.data = weight_data.resize_(
            weight_data.size(), memory_format=memory_format
        )
    for child in module.children():
        convert_conv3d_weight_memory_format(child, memory_format)
    # pyrefly: ignore [bad-return]
    return module


__all__ = [
    "convert_conv2d_weight_memory_format",
    "convert_conv3d_weight_memory_format",
]

```



## High-Level Overview

r"""Convert ``memory_format`` of ``nn.Conv2d.weight`` to ``memory_format``.    The conversion recursively applies to nested ``nn.Module``, including ``module``.    Note that it only changes the memory_format, but not the semantics of each dimensions.    This function is used to facilitate the computation to adopt NHWC kernels, which    provides considerable speed up for fp16 data on CUDA devices with compute capability >= 7.0    .. note::        Calling ``model.to(memory_format=torch.channels_last)`` is more aggressive        than the utility function ``convert_conv2d_weight_memory_format``. Any        layer with 4d weight will be affected by ``model.to``, which does not        necessarily benefit from conversion to specified ``memory_format``.        One place we are confident in is that NHWC(channels_last) conversion for        convolution in cuDNN, as it is beneficial to run convolution in NHWC,        even in cases where we have to apply permutation to input tensors.        Hence our strategy here is to convert only the weight of convolution to        channels_last. This ensures that;        1. Fast convolution kernels will be used, the benefit of which could        outweigh overhead of permutation (if input is not in the same format).        2. No unnecessary permutations are applied on layers that do not benefit        from memory_format conversion.        The optimal case is that, layers between convolution layers are channels        last compatible. Input tensor would be permuted to channels last when it        encounters the first convolution layer and stay in that memory format.        Hence following convolutions will not need to permute its input tensor.        In case where a channels last incompatible layer is between convolution        layers, we need to permute the input tensor back to contiguous format        for that layer. The input tensor will go through the remaining layers in        contiguous format and be permuted to channels last when it encounters        another convolution layer. There's no point in propagating that        permutation to an earlier layer, as most layers are quite agnostic to        ``memory_format``.        This claim might change when PyTorch supports fusion of permutation, as

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `convert_conv2d_weight_memory_format`, `convert_conv3d_weight_memory_format`

**Key imports**: annotations, TypeVar, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: TypeVar
- `torch`


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

- **File Documentation**: `memory_format.py_docs.md`
- **Keyword Index**: `memory_format.py_kw.md`
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
- [`weight_norm.py_kw.md_docs.md`](./weight_norm.py_kw.md_docs.md)
- [`convert_parameters.py_kw.md_docs.md`](./convert_parameters.py_kw.md_docs.md)
- [`parametrizations.py_docs.md_docs.md`](./parametrizations.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `memory_format.py_docs.md_docs.md`
- **Keyword Index**: `memory_format.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
