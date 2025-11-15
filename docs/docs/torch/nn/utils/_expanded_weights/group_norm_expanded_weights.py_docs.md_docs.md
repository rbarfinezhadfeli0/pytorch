# Documentation: `docs/torch/nn/utils/_expanded_weights/group_norm_expanded_weights.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/utils/_expanded_weights/group_norm_expanded_weights.py_docs.md`
- **Size**: 6,489 bytes (6.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/utils/_expanded_weights/group_norm_expanded_weights.py`

## File Metadata

- **Path**: `torch/nn/utils/_expanded_weights/group_norm_expanded_weights.py`
- **Size**: 3,576 bytes (3.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import operator
from functools import reduce

import torch
import torch.nn.functional as F

from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
    unpack_expanded_weight_or_tensor,
)


@implements_per_sample_grads(F.group_norm)
class GroupNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        input, num_groups = expanded_args
        N = input.shape[0]
        C = input.shape[1]
        HxW = reduce(operator.mul, input.shape[2:], 1)
        weight, bias, eps = (
            expanded_kwargs["weight"],
            expanded_kwargs["bias"],
            expanded_kwargs["eps"],
        )
        output, mean, rstd = forward_helper(
            torch.native_group_norm,
            (input, weight, bias, N, C, HxW, num_groups, eps),
            {},
        )
        ctx.input, ctx.num_groups = input, num_groups
        ctx.weight, ctx.eps = weight, eps
        ctx.mean, ctx.rstd = mean, rstd
        if isinstance(bias, ExpandedWeight):
            ctx.bias = bias
        if input.requires_grad and isinstance(weight, ExpandedWeight):
            ctx.weight = weight
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        input, num_groups = ctx.input, ctx.num_groups
        weight, bias, eps = ctx.weight, ctx.bias, ctx.eps
        mean, rstd = ctx.mean, ctx.rstd

        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference

        if input.requires_grad:
            weight_c = unpack_expanded_weight_or_tensor(
                weight, lambda t: t.contiguous()
            )
            input_c = input.contiguous()
            grad_output_c = (
                grad_output.contiguous() if grad_output is not None else None
            )
            N = input.shape[0]
            C = input.shape[1]
            HxW = 1
            for s in input.shape[2:]:
                HxW *= s
            bw_fn = torch.ops.aten.native_group_norm_backward
            results.append(
                bw_fn(
                    grad_output_c,
                    input_c,
                    mean,
                    rstd,
                    weight_c,
                    N,
                    C,
                    HxW,
                    num_groups,
                    (True, False, False),
                )[0]
            )
        else:
            results.append(None)

        # weight and bias don't compute batched gradients; no other arguments are differentiable
        results = results + [None] * 4

        # set grad_sample field for weight and bias with per sample gradients
        if hasattr(ctx, "weight"):
            set_grad_sample_if_exists(
                weight,
                lambda _: torch.einsum(
                    "ni...->ni",
                    # pyrefly: ignore [unsupported-operation]
                    F.group_norm(input, num_groups, eps=eps) * grad_output,
                ),
            )
        if hasattr(ctx, "bias"):
            set_grad_sample_if_exists(
                bias, lambda _: torch.einsum("ni...->ni", grad_output)
            )
        return tuple(results)

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GroupNormPerSampleGrad`

**Functions defined**: `forward`, `backward`

**Key imports**: operator, reduce, torch, torch.nn.functional as F, ExpandedWeight, implements_per_sample_grads


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/utils/_expanded_weights`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `functools`: reduce
- `torch`
- `torch.nn.functional as F`
- `.expanded_weights_impl`: ExpandedWeight, implements_per_sample_grads


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
- [`layer_norm_expanded_weights.py_docs.md`](./layer_norm_expanded_weights.py_docs.md)
- [`linear_expanded_weights.py_docs.md`](./linear_expanded_weights.py_docs.md)
- [`expanded_weights_utils.py_docs.md`](./expanded_weights_utils.py_docs.md)
- [`expanded_weights_impl.py_docs.md`](./expanded_weights_impl.py_docs.md)
- [`instance_norm_expanded_weights.py_docs.md`](./instance_norm_expanded_weights.py_docs.md)
- [`conv_utils.py_docs.md`](./conv_utils.py_docs.md)
- [`conv_expanded_weights.py_docs.md`](./conv_expanded_weights.py_docs.md)
- [`embedding_expanded_weights.py_docs.md`](./embedding_expanded_weights.py_docs.md)


## Cross-References

- **File Documentation**: `group_norm_expanded_weights.py_docs.md`
- **Keyword Index**: `group_norm_expanded_weights.py_kw.md`
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
- [`conv_expanded_weights.py_docs.md_docs.md`](./conv_expanded_weights.py_docs.md_docs.md)
- [`conv_utils.py_kw.md_docs.md`](./conv_utils.py_kw.md_docs.md)
- [`conv_utils.py_docs.md_docs.md`](./conv_utils.py_docs.md_docs.md)
- [`embedding_expanded_weights.py_docs.md_docs.md`](./embedding_expanded_weights.py_docs.md_docs.md)
- [`expanded_weights_impl.py_kw.md_docs.md`](./expanded_weights_impl.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`linear_expanded_weights.py_kw.md_docs.md`](./linear_expanded_weights.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `group_norm_expanded_weights.py_docs.md_docs.md`
- **Keyword Index**: `group_norm_expanded_weights.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
