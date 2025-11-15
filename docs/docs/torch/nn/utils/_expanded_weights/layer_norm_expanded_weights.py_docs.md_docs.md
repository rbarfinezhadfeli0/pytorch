# Documentation: `docs/torch/nn/utils/_expanded_weights/layer_norm_expanded_weights.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/utils/_expanded_weights/layer_norm_expanded_weights.py_docs.md`
- **Size**: 6,175 bytes (6.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/utils/_expanded_weights/layer_norm_expanded_weights.py`

## File Metadata

- **Path**: `torch/nn/utils/_expanded_weights/layer_norm_expanded_weights.py`
- **Size**: 3,289 bytes (3.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

import torch
import torch.nn.functional as F

from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
    sum_over_all_but_batch_and_last_n,
    unpack_expanded_weight_or_tensor,
)


@implements_per_sample_grads(F.layer_norm)
class LayerNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        input = expanded_args[0]
        normalized_shape = expanded_args[1]
        if len(input.shape) <= len(normalized_shape):
            raise RuntimeError(
                "Expanded Weights: Layer norm should not normalize over batch dimension for per sample gradient"
                f"computations but got that normalized shape, {normalized_shape}, matched input shape."
            )
        output, mean, rstd = forward_helper(
            torch.native_layer_norm, expanded_args, expanded_kwargs
        )
        ctx.args = expanded_args

        if input.requires_grad or isinstance(expanded_kwargs["weight"], ExpandedWeight):
            ctx.weight = expanded_kwargs["weight"]
        if input.requires_grad or isinstance(expanded_kwargs["bias"], ExpandedWeight):
            ctx.bias = expanded_kwargs["bias"]
        ctx.eps = expanded_kwargs["eps"]
        ctx.mean, ctx.rstd = mean, rstd
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        def weight_per_sample_grad(weight):
            return sum_over_all_but_batch_and_last_n(
                F.layer_norm(input, normalized_shape, eps=ctx.eps) * grad_output,
                weight.dim(),
            )

        input, normalized_shape = ctx.args
        mean, rstd = ctx.mean, ctx.rstd

        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference
        if input.requires_grad:
            weight_ = unpack_expanded_weight_or_tensor(ctx.weight)
            bias_ = unpack_expanded_weight_or_tensor(ctx.bias)
            results.append(
                torch.ops.aten.native_layer_norm_backward(
                    grad_output,
                    input,
                    normalized_shape,
                    mean,
                    rstd,
                    weight_,
                    bias_,
                    (True, False, False),
                )[0]
            )
        else:
            results.append(None)

        # weight and bias don't compute batched gradients; no other arguments are differentiable
        results = results + [None] * 4

        # set grad_sample field for weight and bias with per sample gradients
        if hasattr(ctx, "weight"):
            set_grad_sample_if_exists(ctx.weight, weight_per_sample_grad)
        if hasattr(ctx, "bias"):
            set_grad_sample_if_exists(
                ctx.bias,
                lambda bias: sum_over_all_but_batch_and_last_n(grad_output, bias.dim()),
            )
        return tuple(results)

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LayerNormPerSampleGrad`

**Functions defined**: `forward`, `backward`, `weight_per_sample_grad`

**Key imports**: torch, torch.nn.functional as F, ExpandedWeight, implements_per_sample_grads


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/utils/_expanded_weights`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

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
- [`group_norm_expanded_weights.py_docs.md`](./group_norm_expanded_weights.py_docs.md)
- [`linear_expanded_weights.py_docs.md`](./linear_expanded_weights.py_docs.md)
- [`expanded_weights_utils.py_docs.md`](./expanded_weights_utils.py_docs.md)
- [`expanded_weights_impl.py_docs.md`](./expanded_weights_impl.py_docs.md)
- [`instance_norm_expanded_weights.py_docs.md`](./instance_norm_expanded_weights.py_docs.md)
- [`conv_utils.py_docs.md`](./conv_utils.py_docs.md)
- [`conv_expanded_weights.py_docs.md`](./conv_expanded_weights.py_docs.md)
- [`embedding_expanded_weights.py_docs.md`](./embedding_expanded_weights.py_docs.md)


## Cross-References

- **File Documentation**: `layer_norm_expanded_weights.py_docs.md`
- **Keyword Index**: `layer_norm_expanded_weights.py_kw.md`
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

- **File Documentation**: `layer_norm_expanded_weights.py_docs.md_docs.md`
- **Keyword Index**: `layer_norm_expanded_weights.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
