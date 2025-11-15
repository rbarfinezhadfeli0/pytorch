# Documentation: `docs/torch/nn/utils/_expanded_weights/conv_expanded_weights.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/utils/_expanded_weights/conv_expanded_weights.py_docs.md`
- **Size**: 5,951 bytes (5.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/utils/_expanded_weights/conv_expanded_weights.py`

## File Metadata

- **Path**: `torch/nn/utils/_expanded_weights/conv_expanded_weights.py`
- **Size**: 2,925 bytes (2.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.nn.functional as F


_P = ParamSpec("_P")
_R = TypeVar("_R")

from .conv_utils import (
    conv_args_and_kwargs,
    conv_backward,
    conv_input_for_string_padding,
    conv_picker,
)
from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import forward_helper


@implements_per_sample_grads(F.conv1d)
@implements_per_sample_grads(F.conv2d)
@implements_per_sample_grads(F.conv3d)
class ConvPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any,
        kwarg_names: list[str],
        conv_fn: Callable[_P, _R],
        *expanded_args_and_kwargs: Any,
    ) -> torch.Tensor:
        expanded_args, expanded_kwargs = conv_args_and_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        orig_input = expanded_args[0]
        was_same_padding = expanded_kwargs["padding"] == "same"

        if isinstance(expanded_kwargs["padding"], str):
            # if padding is a string, we'll do the necessary padding (slowly) using F.pad
            kernel_size = expanded_args[1].shape[2:]
            padding, dilation = expanded_kwargs["padding"], expanded_kwargs["dilation"]
            input = conv_input_for_string_padding(
                conv_fn, padding, expanded_args[0], dilation, kernel_size
            )
            expanded_args = (input, expanded_args[1])
            # since we've already done the padding, don't need any more
            expanded_kwargs["padding"] = 0

        output = forward_helper(conv_fn, expanded_args, expanded_kwargs)
        input, weight = expanded_args
        batched_dim_size = conv_picker(conv_fn, 3, 4, 5)
        if input.dim() != batched_dim_size:
            raise RuntimeError(
                f"Expanded Weights only support convolution with batched input, got {conv_fn} with an"
                f"unbatched input of dim {input.dim()}, expected input of dim {batched_dim_size}"
            )

        # pyrefly: ignore [invalid-type-var]
        ctx.conv_fn = conv_fn

        ctx.batch_size = orig_input.shape[0]
        ctx.input_required_grad = orig_input.requires_grad
        ctx.orig_input_shape = orig_input.shape
        ctx.was_same_padding = was_same_padding
        ctx.stride, ctx.padding = expanded_kwargs["stride"], expanded_kwargs["padding"]
        ctx.dilation, ctx.groups = (
            expanded_kwargs["dilation"],
            expanded_kwargs["groups"],
        )

        if isinstance(weight, ExpandedWeight):
            ctx.input = input
        ctx.weight = weight
        ctx.bias = expanded_kwargs["bias"]

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return conv_backward(ctx.conv_fn, ctx, grad_outputs[0])

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConvPerSampleGrad`

**Functions defined**: `forward`, `backward`

**Key imports**: Callable, Any, TypeVar, ParamSpec, torch, torch.nn.functional as F, ExpandedWeight, implements_per_sample_grads, forward_helper


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/utils/_expanded_weights`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: Any, TypeVar
- `typing_extensions`: ParamSpec
- `torch`
- `torch.nn.functional as F`
- `.expanded_weights_impl`: ExpandedWeight, implements_per_sample_grads
- `.expanded_weights_utils`: forward_helper


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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

Files in the same folder (`torch/nn/utils/_expanded_weights`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`group_norm_expanded_weights.py_docs.md`](./group_norm_expanded_weights.py_docs.md)
- [`layer_norm_expanded_weights.py_docs.md`](./layer_norm_expanded_weights.py_docs.md)
- [`linear_expanded_weights.py_docs.md`](./linear_expanded_weights.py_docs.md)
- [`expanded_weights_utils.py_docs.md`](./expanded_weights_utils.py_docs.md)
- [`expanded_weights_impl.py_docs.md`](./expanded_weights_impl.py_docs.md)
- [`instance_norm_expanded_weights.py_docs.md`](./instance_norm_expanded_weights.py_docs.md)
- [`conv_utils.py_docs.md`](./conv_utils.py_docs.md)
- [`embedding_expanded_weights.py_docs.md`](./embedding_expanded_weights.py_docs.md)


## Cross-References

- **File Documentation**: `conv_expanded_weights.py_docs.md`
- **Keyword Index**: `conv_expanded_weights.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn/utils/_expanded_weights`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/utils/_expanded_weights`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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

Files in the same folder (`docs/torch/nn/utils/_expanded_weights`):

- [`instance_norm_expanded_weights.py_docs.md_docs.md`](./instance_norm_expanded_weights.py_docs.md_docs.md)
- [`expanded_weights_utils.py_docs.md_docs.md`](./expanded_weights_utils.py_docs.md_docs.md)
- [`embedding_expanded_weights.py_kw.md_docs.md`](./embedding_expanded_weights.py_kw.md_docs.md)
- [`conv_utils.py_kw.md_docs.md`](./conv_utils.py_kw.md_docs.md)
- [`conv_utils.py_docs.md_docs.md`](./conv_utils.py_docs.md_docs.md)
- [`embedding_expanded_weights.py_docs.md_docs.md`](./embedding_expanded_weights.py_docs.md_docs.md)
- [`expanded_weights_impl.py_kw.md_docs.md`](./expanded_weights_impl.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`linear_expanded_weights.py_kw.md_docs.md`](./linear_expanded_weights.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `conv_expanded_weights.py_docs.md_docs.md`
- **Keyword Index**: `conv_expanded_weights.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
